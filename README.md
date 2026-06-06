# ccbsampler

An ML-based UTAU resampler. Rebuilt from hifisampler with better cross-platform performance and user experience.

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

Download the build artifact for your architecture from [Actions](https://github.com/0x24a/ccbsampler/actions)

And copy the build artifact to your "Resamplers" folder and set it as the default resampler(optional). (OpenUtau -> Select Renderer -> CLASSIC -> Settings Icon -> Resampler)

# Credits
[openhachimi](https://github.com/openhachimi)
[openvpi](https://github.com/openvpi)