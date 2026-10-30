# ccbsampler

An ML-based UTAU resampler. Rebuilt from hifisampler with better cross-platform performance and user experience.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.11+ (managed by uv)

## Installation

### 1. Clone the repo

```
git clone https://github.com/0x24a/ccbsampler.git
cd ccbsampler
```

### 2. Install dependencies

```
uv sync
```

### 3. Download models

```
uv run setup.py
```

This downloads the vocoder and the hnsep model. Model files are not part of the repo, so this step is necessary.

### 4. Run the server

```
uv run main.py
```

The server listens on `127.0.0.1:8572`. On the first launch, the vocoder checkpoint is converted to ONNX and it might take a moment. After converting, each launch can load the model in a couple of seconds.

> [!NOTE]
> On macOS the vocoder runs on CoreML by default. On Windows and Linux the default install runs on CPU. See [Hardware acceleration](docs/hardware-acceleration.md) to enable your GPU.

## Integrate with OpenUtau

- Download the client binary for your architecture from [Actions](https://github.com/0x24a/ccbsampler/actions/workflows/build-rust-client.yml) (`ccbsampler-client-<os>-<arch>.tar.gz`) or [Releases](https://github.com/0x24a/ccbsampler/releases)
- Unpack it, and you will see an executable file.
- Start OpenUtau and create a project. Select any UTAU singer, then change the renderer from the default "WORLDLINE-R" to "CLASSIC". (Leave it as-is if it's already CLASSIC)
- Click on the "Track Settings" icon on the right side to open a menu. In the "Resampler" section, click "Location", then your `Resamplers` folder will show up.
- Copy the executable file extracted earlier to the `Resamplers` folder.
- Close and re-open the "Track Settings" menu, then you will be able to find ccbsampler-client in the dropdown.
- Optionally set it as the default resampler by clicking on "Set As Default".
- Click on "OK" to apply the settings.

## Documentation

- [Hardware acceleration](docs/hardware-acceleration.md) · how to get the model running on your GPU
- [Configuration](docs/configuration.md) · explaining `config.yaml`
- [API reference](docs/api.md) · HTTP API, render metrics, and UTAU flags

## Credits

- [openhachimi](https://github.com/openhachimi)
- [openvpi](https://github.com/openvpi)

## License

[Apache-2.0](LICENSE)

Made with ❤️ by 0x24a
