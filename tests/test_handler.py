"""Handler tests for the Home Assistant Wyoming TTS client.

Kokoro is replaced with a fake that records the phonemizer language and
returns a short float buffer. No ONNX model is loaded.
"""
import asyncio

import numpy as np
import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.info import Describe, Info, SelectProgram
from wyoming.server import AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeTextFormat,
    SynthesizeVoice,
)

from main import (
    PROGRAM_NAME,
    VERSION,
    _MAX_TEXT_LENGTH,
    KokoroEventHandler,
    SynthesisCache,
    advertise_zeroconf,
    build_wyoming_info,
    kokoro_lang,
    voice_description,
    voice_language,
    zeroconf_host,
)


class FakeKokoro:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
        self.voices = {
            "af_heart": object(),
            "bf_alice": object(),
            "bm_george": object(),
            "ef_dora": object(),
            "ff_siwis": object(),
            "hf_alpha": object(),
            "if_sara": object(),
            "jf_alpha": object(),
            "pf_dora": object(),
            "zf_xiaobei": object(),
            "xx_custom": object(),
        }

    def create_stream(self, text, voice, speed=1.0, lang="en-us", **kwargs):
        self.calls.append(
            {"text": text, "voice": voice, "speed": speed, "lang": lang}
        )

        async def _generate():
            if self.fail:
                raise RuntimeError("synth failed")
            yield np.array([2.0, -2.0, 0.25], dtype=np.float32), 24000

        return _generate()


def _handler(fake, voice="af_heart"):
    info = build_wyoming_info(fake)
    handler = KokoroEventHandler(
        info,
        fake,
        voice,
        1.0,
        asyncio.Semaphore(1),
        SynthesisCache(),
        None,
        None,
    )
    events = []

    async def capture(event):
        events.append(event)

    handler.write_event = capture
    return handler, events


def _types(events):
    return [event.type for event in events]


def test_voice_profile_matches_kokoro_and_home_assistant():
    assert voice_language("af_heart") == "en"
    assert kokoro_lang("af_heart") == "en-us"
    assert voice_language("bf_alice") == "en"
    assert kokoro_lang("bf_alice") == "en-gb"
    assert (voice_language("ef_dora"), kokoro_lang("ef_dora")) == ("es", "es")
    assert (voice_language("ff_siwis"), kokoro_lang("ff_siwis")) == ("fr", "fr-fr")
    assert (voice_language("hf_alpha"), kokoro_lang("hf_alpha")) == ("hi", "hi")
    assert (voice_language("if_sara"), kokoro_lang("if_sara")) == ("it", "it")
    assert (voice_language("pf_dora"), kokoro_lang("pf_dora")) == ("pt", "pt-br")
    assert (voice_language("jf_alpha"), kokoro_lang("jf_alpha")) == ("ja", "ja")
    assert (voice_language("zf_xiaobei"), kokoro_lang("zf_xiaobei")) == ("zh", "cmn")
    assert voice_language("xx_custom") == "en"
    assert kokoro_lang("xx_custom") == "en-us"

    assert voice_description("af_heart") == "Heart (American English, female)"
    assert voice_description("bm_george") == "George (British English, male)"
    assert voice_description("jf_alpha") == "Alpha (Japanese, female)"


def test_describe_languages_for_home_assistant():
    async def run():
        fake = FakeKokoro()
        handler, events = _handler(fake)
        assert await handler.handle_event(Describe().event()) is True
        info = Info.from_event(events[0])
        program = info.tts[0]
        assert program.name == PROGRAM_NAME
        assert program.version == VERSION
        assert program.supports_synthesize_streaming is True
        assert program.attribution.name == "Kokoro"
        assert program.attribution.url == "https://huggingface.co/hexgrad/Kokoro-82M"

        by_name = {voice.name: voice for voice in program.voices}
        languages = {lang for voice in program.voices for lang in voice.languages}
        assert languages == {"en", "es", "fr", "hi", "it", "ja", "pt", "zh"}
        assert "jp" not in languages
        assert "cn" not in languages
        assert by_name["af_heart"].languages == ["en"]
        assert by_name["bf_alice"].languages == ["en"]
        assert by_name["jf_alpha"].languages == ["ja"]
        assert by_name["zf_xiaobei"].languages == ["zh"]
        assert by_name["pf_dora"].languages == ["pt"]
        assert all(not voice.speakers for voice in program.voices)
        assert by_name["af_heart"].description == "Heart (American English, female)"

    asyncio.run(run())


def test_one_shot_ends_on_audio_stop_and_clips_samples():
    async def run():
        fake = FakeKokoro()
        handler, events = _handler(fake)
        result = await handler.handle_event(
            Synthesize(
                text="Hello there.",
                voice=SynthesizeVoice(name="ef_dora", language="en"),
            ).event()
        )
        assert result is True
        assert _types(events) == ["audio-start", "audio-chunk", "audio-stop"]
        assert SynthesizeStopped.is_type(events[-1].type) is False
        assert fake.calls == [
            {
                "text": "Hello there.",
                "voice": "ef_dora",
                "speed": 1.0,
                "lang": "es",
            }
        ]

        start = AudioStart.from_event(events[0])
        assert (start.rate, start.width, start.channels) == (24000, 2, 1)
        samples = np.frombuffer(
            AudioChunk.from_event(events[1]).audio, dtype=np.int16
        )
        assert samples.max() == np.int16(32767)
        assert samples.min() == np.int16(-32767)
        assert AudioStop.is_type(events[-1].type)

    asyncio.run(run())


def test_non_english_voices_use_phonemizer_language():
    async def run():
        fake = FakeKokoro()
        handler, _events = _handler(fake)
        for voice in ("jf_alpha", "zf_xiaobei", "pf_dora", "bf_alice"):
            await handler.handle_event(
                Synthesize(
                    text="Hello.",
                    voice=SynthesizeVoice(name=voice),
                ).event()
            )
        assert [call["lang"] for call in fake.calls] == ["ja", "cmn", "pt-br", "en-gb"]

    asyncio.run(run())


def test_streaming_matches_home_assistant_order():
    async def run():
        fake = FakeKokoro()
        handler, events = _handler(fake)
        voice = SynthesizeVoice(name="af_heart")
        assert await handler.handle_event(
            SynthesizeStart(voice=voice).event()
        ) is True
        await handler.handle_event(SynthesizeChunk(text="Hello. ").event())
        await handler.handle_event(SynthesizeChunk(text="World").event())
        # Compatibility copy of the full text. Must not be synthesized again.
        await handler.handle_event(
            Synthesize(
                text="THIS COMPAT TEXT SHOULD NOT BE SPOKEN.",
                voice=voice,
            ).event()
        )
        await handler.handle_event(SynthesizeStop().event())

        spoken = " ".join(call["text"] for call in fake.calls)
        assert "THIS COMPAT" not in spoken
        assert "Hello" in spoken
        assert "World" in spoken
        assert all(call["lang"] == "en-us" for call in fake.calls)
        assert _types(events)[0] == "audio-start"
        assert "audio-chunk" in _types(events)
        assert _types(events)[-2:] == ["audio-stop", "synthesize-stopped"]

    asyncio.run(run())


def test_stream_ssml_tags_split_across_chunks_are_not_spoken():
    async def run():
        fake = FakeKokoro()
        handler, events = _handler(fake)
        await handler.handle_event(
            SynthesizeStart(
                voice=SynthesizeVoice(name="af_heart"),
                text_format=SynthesizeTextFormat.SSML,
            ).event()
        )
        await handler.handle_event(SynthesizeChunk(text="<spe").event())
        await handler.handle_event(
            SynthesizeChunk(text='ak>Hello. World</speak>').event()
        )
        await handler.handle_event(SynthesizeStop().event())
        spoken = " ".join(call["text"] for call in fake.calls)
        assert "Hello" in spoken and "World" in spoken
        assert "<" not in spoken
        assert _types(events)[-1] == "synthesize-stopped"

    asyncio.run(run())


def test_ssml_is_spoken_as_text():
    async def run():
        fake = FakeKokoro()
        handler, _events = _handler(fake)
        await handler.handle_event(
            Synthesize(
                text='<speak>Hello <break time="200ms"/>there.</speak>',
                voice=SynthesizeVoice(name="af_heart"),
                text_format=SynthesizeTextFormat.SSML,
            ).event()
        )
        assert fake.calls[0]["text"] == "Hello there."
        assert "<" not in fake.calls[0]["text"]

    asyncio.run(run())


def test_stream_is_capped_at_max_length():
    async def run():
        fake = FakeKokoro()
        handler, _events = _handler(fake)
        await handler.handle_event(SynthesizeStart().event())
        await handler.handle_event(
            SynthesizeChunk(text="a" * (_MAX_TEXT_LENGTH + 250)).event()
        )
        await handler.handle_event(SynthesizeChunk(text="bbbb").event())
        await handler.handle_event(SynthesizeStop().event())
        spoken = "".join(call["text"] for call in fake.calls)
        assert spoken == "a" * _MAX_TEXT_LENGTH
        assert "b" not in spoken

    asyncio.run(run())


def test_synthesis_error_unblocks_one_shot_and_stream():
    async def one_shot():
        fake = FakeKokoro(fail=True)
        handler, events = _handler(fake)
        assert await handler.handle_event(
            Synthesize(text="Hello.", voice=SynthesizeVoice(name="af_heart")).event()
        ) is True
        assert Error.is_type(events[1].type)
        assert Error.from_event(events[1]).text == "synth failed"
        assert _types(events) == ["audio-start", "error", "audio-stop"]
        assert "synthesize-stopped" not in _types(events)

    async def streaming():
        fake = FakeKokoro(fail=True)
        handler, events = _handler(fake)
        await handler.handle_event(
            SynthesizeStart(voice=SynthesizeVoice(name="af_heart")).event()
        )
        # "Hello. World" crosses a boundary immediately; "Hello. " alone is held.
        assert await handler.handle_event(
            SynthesizeChunk(text="Hello. World").event()
        ) is True
        assert Error.is_type(events[1].type)
        assert _types(events)[-2:] == ["audio-stop", "synthesize-stopped"]
        # The compatibility synthesize and stop that follow an error must not
        # start another utterance or emit a second terminal pair.
        calls = len(fake.calls)
        await handler.handle_event(
            Synthesize(text="THIS COMPAT TEXT SHOULD NOT BE SPOKEN.").event()
        )
        await handler.handle_event(SynthesizeStop().event())
        assert len(fake.calls) == calls
        assert _types(events).count("synthesize-stopped") == 1
        assert _types(events).count("audio-stop") == 1

    asyncio.run(one_shot())
    asyncio.run(streaming())


def test_sentence_cache_is_shared_by_both_paths():
    async def run():
        fake = FakeKokoro()
        handler, _events = _handler(fake)
        await handler.handle_event(
            Synthesize(text="Hello.", voice=SynthesizeVoice(name="af_heart")).event()
        )
        await handler.handle_event(
            SynthesizeStart(voice=SynthesizeVoice(name="af_heart")).event()
        )
        await handler.handle_event(SynthesizeChunk(text="Hello.").event())
        await handler.handle_event(SynthesizeStop().event())
        assert len(fake.calls) == 1

    asyncio.run(run())


def test_select_program_accepts_kokoro_only():
    async def run():
        fake = FakeKokoro()
        handler, events = _handler(fake)
        assert await handler.handle_event(
            SelectProgram(name=PROGRAM_NAME).event()
        ) is True
        assert events == []
        assert await handler.handle_event(SelectProgram(name="piper").event()) is True
        assert _types(events) == ["error"]
        assert "piper" in Error.from_event(events[0]).text

    asyncio.run(run())


def test_zeroconf_host_skips_wildcard_binds():
    assert zeroconf_host("0.0.0.0") is None
    assert zeroconf_host("::") is None
    assert zeroconf_host("") is None
    assert zeroconf_host(None) is None
    assert zeroconf_host("192.168.1.20") == "192.168.1.20"


def test_zeroconf_requires_tcp_uri():
    async def run():
        server = AsyncServer.from_uri("stdio://")
        with pytest.raises(ValueError, match="tcp://"):
            await advertise_zeroconf(server, "kokoro")

    asyncio.run(run())
