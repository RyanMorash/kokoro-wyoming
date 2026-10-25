# Kokoro Wyoming

This is a [wyoming protocol](https://github.com/rhasspy/wyoming) server that implements [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) using the [Kokoro onnx](https://github.com/thewh1teagle/kokoro-onnx) python library.

Recent significant improvements from relvacode:

  - Updated to latest Kokoro model
  - Available voices are read directly from the model instead of hardcoding
  - Support for `SIGINT` and `SIGTERM` termination
  - Support for GPU accelerated inference (see [./docker-compose.intel.yml]())
  - Kokoro debug logging

Docker image available at:
https://hub.docker.com/r/nordwestt/kokoro-wyoming 

Latest improvements available at:

### CPU:
```
docker pull nordwestt/kokoro-wyoming:latest
```

### CUDA:
```
docker pull nordwestt/kokoro-wyoming:cuda
```

## Setup

Use one of the provided docker compose files in this repository (use Intel if you have a recent Intel iGPU). Those files publish port **10210**.

### Manual connection

In Home Assistant, go to **Settings > Devices & services**, select **Add integration**, and choose **Wyoming Protocol**. Enter the IP address of the machine running the container and port **10210**.

Then go to **Settings > Voice assistants**, edit or create an assistant, and select **Kokoro** as the text-to-speech provider.

### Discovery

Home Assistant discovers services advertised as `_wyoming._tcp.local.`. Pass `--zeroconf` to advertise this server (the optional name defaults to `kokoro`). Bridge networking would publish the container address, which Home Assistant cannot reach, so discovery needs host networking:

```yaml
services:
  app:
    network_mode: host
    command: ["--zeroconf", "kokoro"]
```

The service then shows up under **Settings > Devices & services** as a discovered Wyoming Protocol integration. With `network_mode: host` the process listens on port 10210 of the host, so the compose `ports:` mapping is not used.

### Languages

Voices are advertised with ISO 639-1 codes (`en`, `es`, `fr`, `hi`, `it`, `ja`, `pt`, `zh`). American and British voices are both `en`; the picker label includes the region. Japanese is `ja` and Chinese is `zh`. Portuguese voices are `pt`.

Voice ids such as `af_heart` are unchanged. After upgrading, point a Japanese, Chinese, or Portuguese assistant at the new language code and choose the voice again. Those voices were previously tagged `jp`, `cn`, or grouped under `en`.

Options for available voices can be found in the [model voices documentation](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
