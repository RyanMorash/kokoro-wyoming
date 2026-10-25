#!/usr/bin/env python3
"""Wyoming protocol server for Kokoro TTS.

Wraps kokoro-onnx in a Wyoming-compatible TCP server for use with
Home Assistant and other Wyoming clients.
"""
import argparse
import asyncio
import hashlib
import logging
import re
import signal
import sys
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Optional

import kokoro_onnx.config
import numpy as np
from kokoro_onnx import Kokoro
from kokoro_onnx.log import log

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import (
    Attribution,
    Describe,
    Info,
    SelectProgram,
    TtsProgram,
    TtsVoice,
)
from wyoming.server import AsyncEventHandler, AsyncServer, AsyncTcpServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeTextFormat,
    SynthesizeVoice,
)

try:
    from sentence_stream import SentenceBoundaryDetector
except ImportError:  # fallback: simple punctuation splitter
    SentenceBoundaryDetector = None

_LOGGER = log.getChild(__name__)

# Kokoro voice ids look like "af_heart": language prefix, gender, then name.
# ISO 639-1 codes are what Home Assistant Assist stores. kokoro-onnx forwards
# `lang` to phonemizer/espeak, which uses a different set (cmn, not zh).
_LANG_BY_PREFIX = {
    "a": ("en", "en-us", "American English"),
    "b": ("en", "en-gb", "British English"),
    "e": ("es", "es", "Spanish"),
    "f": ("fr", "fr-fr", "French"),
    "h": ("hi", "hi", "Hindi"),
    "i": ("it", "it", "Italian"),
    "p": ("pt", "pt-br", "Portuguese"),
    "j": ("ja", "ja", "Japanese"),
    "z": ("zh", "cmn", "Chinese"),
}
_DEFAULT_PROFILE = ("en", "en-us", "English")
_GENDER = {"f": "female", "m": "male"}

PROGRAM_NAME = "Kokoro"
KOKORO_ATTRIBUTION = Attribution(
    name="Kokoro",
    url="https://huggingface.co/hexgrad/Kokoro-82M",
)

# Maximum text length to prevent resource exhaustion on extremely long inputs.
_MAX_TEXT_LENGTH = 5000

# Maximum number of cached synthesis results to keep in memory.
_MAX_CACHE_ENTRIES = 128

_TAG_RE = re.compile(r"<[^>]*>")
_DANGLING_TAG_RE = re.compile(r"<[^>]*$")


def _read_version() -> str:
    path = Path(__file__).resolve().parent.parent / "VERSION"
    try:
        version = path.read_text(encoding="utf-8").strip()
    except OSError:
        _LOGGER.warning("VERSION file not found at %s", path)
        return "0.0.0"
    return version or "0.0.0"


VERSION = _read_version()


class _SimpleSBD:
    """Minimal sentence boundary detector (used only if sentence_stream is absent)."""

    def __init__(self):
        self._buf = ""

    def add_chunk(self, text):
        self._buf += text
        out = []
        while True:
            m = re.search(r"[.!?][\"')\]]*\s+", self._buf)
            if not m:
                break
            out.append(self._buf[: m.end()].strip())
            self._buf = self._buf[m.end():]
        return out

    def finish(self):
        rest, self._buf = self._buf.strip(), ""
        return rest


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    text = " ".join(text.strip().split())
    pattern = r"(?<=[.!?])\s+"
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def clean_text(text: str) -> str:
    """Strip markup artifacts that LLMs sometimes include in responses.

    Removes markdown bold/italic markers, heading markers, and other
    formatting that shouldn't be spoken aloud.
    """
    # Strip markdown bold/italic: **text** or __text__ or *text* or _text_
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    # Strip markdown headings: ## Heading
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Strip bullet points: - item or * item
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    # Collapse multiple spaces/newlines
    text = " ".join(text.split())
    return text.strip()


def strip_tags(text: str) -> str:
    """Remove SSML or XML tags, including a trailing unfinished tag."""
    return _DANGLING_TAG_RE.sub("", _TAG_RE.sub("", text))


def _is_ssml(text_format) -> bool:
    if text_format is None:
        return False
    if isinstance(text_format, SynthesizeTextFormat):
        return text_format == SynthesizeTextFormat.SSML
    return str(text_format).lower() == SynthesizeTextFormat.SSML.value


def voice_profile(voice_id: str) -> tuple[str, str, str, str]:
    """Return (iso_language, kokoro_lang, region, gender) for a voice id."""
    iso, kokoro_lang, region = _LANG_BY_PREFIX.get(voice_id[:1], _DEFAULT_PROFILE)
    gender = _GENDER.get(voice_id[1:2], "")
    return iso, kokoro_lang, region, gender


def voice_language(voice_id: str) -> str:
    """ISO 639-1 language advertised to Home Assistant."""
    return voice_profile(voice_id)[0]


def kokoro_lang(voice_id: str) -> str:
    """Language code kokoro-onnx passes to phonemizer."""
    return voice_profile(voice_id)[1]


def voice_description(voice_id: str) -> str:
    """Human-readable label shown in the Home Assistant voice picker."""
    _iso, _lang, region, gender = voice_profile(voice_id)
    raw = voice_id.split("_", 1)[1] if "_" in voice_id else voice_id
    name = " ".join(part.capitalize() for part in raw.split("_") if part)
    detail = f"{region}, {gender}" if gender else region
    return f"{name} ({detail})"


def zeroconf_host(host: Optional[str]) -> Optional[str]:
    """Return a host worth advertising, or None so zeroconf detects one.

    Binding to 0.0.0.0 is not an address Home Assistant can connect to.
    """
    if host in (None, "", "0.0.0.0", "::"):
        return None
    return host


def get_model_voices(model: Kokoro) -> list[TtsVoice]:
    voices = []
    for voice_id in model.voices.keys():
        if voice_id[:1] not in _LANG_BY_PREFIX:
            _LOGGER.warning(
                "Unknown voice prefix for %s; advertising as en", voice_id
            )
        voices.append(
            TtsVoice(
                name=voice_id,
                description=voice_description(voice_id),
                attribution=KOKORO_ATTRIBUTION,
                installed=True,
                version=None,
                languages=[voice_language(voice_id)],
            )
        )
    return voices


def build_wyoming_info(model: Kokoro) -> Info:
    """Describe the single Kokoro program Home Assistant will use."""
    return Info(
        tts=[
            TtsProgram(
                name=PROGRAM_NAME,
                description="Kokoro TTS via Wyoming protocol",
                attribution=KOKORO_ATTRIBUTION,
                installed=True,
                voices=sorted(get_model_voices(model), key=lambda v: v.name),
                version=VERSION,
                supports_synthesize_streaming=True,
            )
        ]
    )


class SynthesisCache:
    """LRU cache for synthesis results, keyed on (text, voice, speed)."""

    def __init__(self, max_entries: int = _MAX_CACHE_ENTRIES):
        self._cache: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        self._max = max_entries
        self._hits = 0
        self._misses = 0

    def _key(self, text: str, voice: str, speed: float) -> str:
        return hashlib.md5(
            f"{text}|{voice}|{speed}".encode()
        ).hexdigest()

    def get(self, text: str, voice: str, speed: float
            ) -> Optional[tuple[np.ndarray, int]]:
        k = self._key(text, voice, speed)
        if k in self._cache:
            self._cache.move_to_end(k)
            self._hits += 1
            return self._cache[k]
        self._misses += 1
        return None

    def put(self, text: str, voice: str, speed: float,
            audio: np.ndarray, sr: int) -> None:
        k = self._key(text, voice, speed)
        self._cache[k] = (audio, sr)
        self._cache.move_to_end(k)
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)

    @property
    def stats(self) -> str:
        total = self._hits + self._misses
        rate = (self._hits / total * 100) if total else 0
        return f"{self._hits}/{total} hits ({rate:.0f}%)"


class KokoroEventHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, kokoro_instance,
                 default_voice: str, default_speed: float,
                 synth_semaphore: asyncio.Semaphore,
                 cache: SynthesisCache,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kokoro = kokoro_instance
        self.default_voice = default_voice
        self.default_speed = default_speed
        self.wyoming_info_event = wyoming_info.event()
        self._semaphore = synth_semaphore
        self._cache = cache
        # streaming state (Wyoming synthesize-start / chunk / stop)
        self._streaming = False
        # Set when a stream fails. Home Assistant still writes the compatibility
        # synthesize and synthesize-stop on this connection, and those must not
        # start a second utterance.
        self._stream_aborted = False
        self._stream_voice = default_voice
        self._stream_ssml = False
        self._ssml_hold = ""
        self._sbd = None
        self._audio_started = False
        self._stream_t0 = 0.0
        self._first_audio: Optional[float] = None
        self._pcm_bytes = 0
        self._stream_sentences = 0
        self._stream_chars = 0
        self._stream_truncated = False

    def _new_sbd(self):
        return SentenceBoundaryDetector() if SentenceBoundaryDetector else _SimpleSBD()

    def _resolve_voice(self, voice: Optional[SynthesizeVoice]) -> str:
        """Use the requested voice name. A name overrides language."""
        if voice and voice.name:
            return voice.name
        return self.default_voice

    def _begin_utterance(self) -> None:
        self._audio_started = False
        self._first_audio = None
        self._pcm_bytes = 0

    def _prepare_stream_text(self, text: str, final: bool = False) -> str:
        """Strip SSML before the sentence splitter, holding a split tag."""
        if not self._stream_ssml:
            return text
        text = self._ssml_hold + text
        self._ssml_hold = ""
        if not final:
            start = text.rfind("<")
            if start != -1 and ">" not in text[start:]:
                self._ssml_hold = text[start:]
                text = text[:start]
        return strip_tags(text)

    def _limit_stream_text(self, text: str) -> str:
        """Apply the same character cap used for one-shot synthesis."""
        if not text:
            return ""
        remaining = _MAX_TEXT_LENGTH - self._stream_chars
        if remaining <= 0:
            if not self._stream_truncated:
                _LOGGER.warning("Stream truncated at %d chars", _MAX_TEXT_LENGTH)
                self._stream_truncated = True
            return ""
        if len(text) > remaining:
            _LOGGER.warning(
                "Stream truncated from %d to %d chars",
                self._stream_chars + len(text),
                _MAX_TEXT_LENGTH,
            )
            text = text[:remaining]
            self._stream_truncated = True
        self._stream_chars += len(text)
        return text

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming protocol events."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        # Wyoming 1.10. One program, so selecting it is a no-op.
        if SelectProgram.is_type(event.type):
            selected = SelectProgram.from_event(event)
            if selected.name != PROGRAM_NAME:
                await self.write_event(
                    Error(
                        text=f"Unknown program: {selected.name}",
                        code="unknown-program",
                    ).event()
                )
            return True

        try:
            if SynthesizeStart.is_type(event.type):
                start = SynthesizeStart.from_event(event)
                self._streaming = True
                self._stream_aborted = False
                self._stream_voice = self._resolve_voice(start.voice)
                self._stream_ssml = _is_ssml(start.text_format)
                self._ssml_hold = ""
                self._sbd = self._new_sbd()
                self._stream_t0 = time.monotonic()
                self._stream_sentences = 0
                self._stream_chars = 0
                self._stream_truncated = False
                self._begin_utterance()
                _LOGGER.debug("Text stream started: voice=%s", self._stream_voice)
                return True

            if SynthesizeChunk.is_type(event.type):
                if not self._streaming or self._stream_aborted:
                    return True
                chunk = SynthesizeChunk.from_event(event)
                text = self._limit_stream_text(self._prepare_stream_text(chunk.text))
                for sentence in self._sbd.add_chunk(text):
                    if await self._synthesize_sentence(sentence, self._stream_voice):
                        self._stream_sentences += 1
                return True

            if SynthesizeStop.is_type(event.type):
                if self._stream_aborted:
                    self._stream_aborted = False
                    self._streaming = False
                    return True
                if self._streaming:
                    tail = self._limit_stream_text(
                        self._prepare_stream_text("", final=True)
                    )
                    if tail:
                        for sentence in self._sbd.add_chunk(tail):
                            if await self._synthesize_sentence(
                                sentence, self._stream_voice
                            ):
                                self._stream_sentences += 1
                    rest = self._sbd.finish()
                    if rest:
                        if await self._synthesize_sentence(rest, self._stream_voice):
                            self._stream_sentences += 1
                    await self._end_audio()
                    await self.write_event(SynthesizeStopped().event())
                    first = (
                        (self._first_audio - self._stream_t0) * 1000
                        if self._first_audio else 0
                    )
                    _LOGGER.info(
                        "Stream done: voice=%s, lang=%s, sentences=%d, "
                        "first_audio=%.0fms, total=%.0fms, cache=%s",
                        self._stream_voice,
                        kokoro_lang(self._stream_voice),
                        self._stream_sentences,
                        first,
                        (time.monotonic() - self._stream_t0) * 1000,
                        self._cache.stats,
                    )
                    self._streaming = False
                return True

            if Synthesize.is_type(event.type):
                if self._streaming or self._stream_aborted:
                    # Home Assistant sends the full text again for clients
                    # that do not understand synthesize-chunk.
                    return True
                return await self._handle_synthesize(event)

            _LOGGER.warning("Unexpected event: %s", event)
            return True
        except Exception as err:
            await self._fail(err)
            return True

    async def _fail(self, err: Exception) -> None:
        """Tell the client synthesis failed, then unblock readers waiting on audio."""
        _LOGGER.exception("Error synthesizing: %s", err)
        await self.write_event(
            Error(text=str(err), code=err.__class__.__name__).event()
        )
        await self._end_audio()
        if self._streaming:
            await self.write_event(SynthesizeStopped().event())
            self._streaming = False
            self._stream_aborted = True

    async def _end_audio(self) -> None:
        """Finish the WAV stream. Home Assistant's one-shot reader stops on audio-stop."""
        if not self._audio_started:
            await self.write_event(
                AudioStart(
                    rate=kokoro_onnx.config.SAMPLE_RATE, width=2, channels=1
                ).event()
            )
            self._audio_started = True
        await self.write_event(AudioStop().event())

    async def _emit_pcm(self, audio: np.ndarray) -> None:
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        if self._first_audio is None:
            self._first_audio = time.monotonic()
        self._pcm_bytes += len(pcm)
        await self.write_event(
            AudioChunk(
                audio=pcm,
                rate=kokoro_onnx.config.SAMPLE_RATE,
                width=2,
                channels=1,
            ).event()
        )

    async def _synthesize_sentence(self, sentence: str, voice_name: str) -> bool:
        """Synthesize one sentence and push its audio. Shared by both paths."""
        text = clean_text(sentence)
        if not text:
            return False
        if len(text) > _MAX_TEXT_LENGTH:
            text = text[:_MAX_TEXT_LENGTH]

        sr = kokoro_onnx.config.SAMPLE_RATE
        if not self._audio_started:
            await self.write_event(
                AudioStart(rate=sr, width=2, channels=1).event()
            )
            self._audio_started = True

        cached = self._cache.get(text, voice_name, self.default_speed)
        if cached is not None:
            audio, _cached_sr = cached
            await self._emit_pcm(audio)
            return True

        parts = []
        async with self._semaphore:
            stream = self.kokoro.create_stream(
                text,
                voice=voice_name,
                speed=self.default_speed,
                lang=kokoro_lang(voice_name),
            )
            async for audio, _sample_rate in stream:
                parts.append(audio)
                await self._emit_pcm(audio)
        if parts:
            self._cache.put(
                text, voice_name, self.default_speed, np.concatenate(parts), sr
            )
        return True

    async def _handle_synthesize(self, event: Event) -> bool:
        synthesize = Synthesize.from_event(event)
        t_start = time.monotonic()
        voice_name = self._resolve_voice(synthesize.voice)
        self._begin_utterance()

        text = synthesize.text or ""
        if _is_ssml(synthesize.text_format):
            text = strip_tags(text)
        text = clean_text(text)

        if not text:
            _LOGGER.warning("Empty text after cleaning, skipping synthesis")
            await self._end_audio()
            return True

        if len(text) > _MAX_TEXT_LENGTH:
            _LOGGER.warning(
                "Text truncated from %d to %d chars", len(text), _MAX_TEXT_LENGTH
            )
            text = text[:_MAX_TEXT_LENGTH]

        for sentence in split_into_sentences(text):
            await self._synthesize_sentence(sentence, voice_name)

        await self._end_audio()

        t_done = time.monotonic()
        first_ms = (
            (self._first_audio - t_start) * 1000 if self._first_audio else 0
        )
        total_ms = (t_done - t_start) * 1000
        audio_dur = self._pcm_bytes / (kokoro_onnx.config.SAMPLE_RATE * 2)
        _LOGGER.info(
            "Synthesized: voice=%s, lang=%s, first_audio=%.0fms, total=%.0fms, "
            "audio=%.1fs, %d bytes, cache=%s, text=\"%s\"",
            voice_name,
            kokoro_lang(voice_name),
            first_ms,
            total_ms,
            audio_dur,
            self._pcm_bytes,
            self._cache.stats,
            text[:80],
        )
        return True


async def advertise_zeroconf(server: AsyncServer, name: str) -> None:
    """Publish `_wyoming._tcp` so Home Assistant can discover this server."""
    if not isinstance(server, AsyncTcpServer):
        raise ValueError("Zeroconf requires a tcp:// URI")

    from wyoming.zeroconf import HomeAssistantZeroconf

    hass_zeroconf = HomeAssistantZeroconf(
        name=name,
        port=server.port,
        host=zeroconf_host(server.host),
    )
    await hass_zeroconf.register_server()
    _LOGGER.info("Zeroconf discovery enabled as %s", name)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Wyoming protocol server for Kokoro TTS"
    )
    parser.add_argument(
        "--uri",
        default="tcp://0.0.0.0:10210",
        help="Server URI (default: tcp://0.0.0.0:10210)"
    )
    parser.add_argument(
        "--voice",
        default="af_heart",
        help="Default voice ID (default: af_heart). "
             "Use --list-voices to see available voices."
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--model",
        default="kokoro-v1.0.onnx",
        help="Path to ONNX model file (default: kokoro-v1.0.onnx)"
    )
    parser.add_argument(
        "--voices",
        default="voices-v1.0.bin",
        help="Path to voices file (default: voices-v1.0.bin)"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and exit"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup synthesis at startup"
    )
    parser.add_argument(
        "--max-cache",
        type=int,
        default=_MAX_CACHE_ENTRIES,
        help=f"Max cached synthesis results (default: {_MAX_CACHE_ENTRIES})"
    )
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="kokoro",
        default=None,
        help="Advertise over zeroconf for Home Assistant discovery. "
             "Optional service name (default: kokoro). Requires a tcp:// URI "
             "and host networking in Docker."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.debug:
        log.setLevel(level=logging.DEBUG)

    _LOGGER.info("Loading model: %s", args.model)
    kokoro_instance = Kokoro(args.model, args.voices)

    if args.list_voices:
        for voice_id in sorted(kokoro_instance.voices.keys()):
            print(voice_id)
        return

    if args.voice not in kokoro_instance.voices:
        _LOGGER.error(
            "Voice '%s' not found. Available: %s",
            args.voice,
            ", ".join(sorted(kokoro_instance.voices.keys()))
        )
        sys.exit(1)

    if not args.no_warmup:
        _LOGGER.info("Warming up with voice '%s'...", args.voice)
        t_warmup = time.monotonic()
        kokoro_instance.create(
            "warmup",
            voice=args.voice,
            speed=args.speed,
            lang=kokoro_lang(args.voice),
        )
        _LOGGER.info(
            "Warmup complete in %.0fms",
            (time.monotonic() - t_warmup) * 1000
        )

    # Shared state across all client handlers
    synth_semaphore = asyncio.Semaphore(1)
    cache = SynthesisCache(max_entries=args.max_cache)

    wyoming_info = build_wyoming_info(kokoro_instance)

    _LOGGER.info(
        "Starting on %s (voice=%s, speed=%.1f, cache=%d, version=%s)",
        args.uri, args.voice, args.speed, args.max_cache, VERSION
    )
    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        await advertise_zeroconf(server, args.zeroconf)

    try:
        loop = asyncio.get_event_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                s, lambda: asyncio.create_task(server.stop())
            )
    except (NotImplementedError, OSError):
        _LOGGER.debug("Signal handlers not available (container environment)")

    await server.run(
        partial(
            KokoroEventHandler,
            wyoming_info,
            kokoro_instance,
            args.voice,
            args.speed,
            synth_semaphore,
            cache,
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
