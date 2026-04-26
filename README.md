# ccbsampler

A hifisampler rewrite with simpler setup and better cross-platform experience

## Installation

`uv` is required. [install](https://docs.astral.sh/uv/getting-started/installation/)

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
uv run setup.py models
```

### 4. Run the server

```
uv run main.py
```

## Intergrate with OpenUtau

There isn't a pre-built ccbsampler-client binary yet, so you will have to build your own.

```
cd client && cargo build --release
```

And copy the build artifact to your "Resamplers" folder and set it as the default resampler(optional). (OpenUtau -> Select Renderer -> CLASSIC -> Settings Icon -> Resampler)

# Credits
[openhachimi](https://github.com/openhachimi)
[openvpi](https://github.com/openvpi)