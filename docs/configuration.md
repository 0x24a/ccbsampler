# Configuration

ccbsampler reads `config.yaml` from the working directory at startup (you can pass another path as an argument: `uv run main.py my-config.yaml`). If the file is missing, the server exits with an error. The repo contains a working default. The defaults below describe what you'd get when a key is misisng from that file.

> [!TIP]
> Keys not present in `config.yaml` fall back to the built-in defaults listed here. The default `config.yaml` matches these defaults except where noted.

## `model`

| Key | Default | Description |
|---|---|---|
| `vocoder_path` | `./pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt` | Path to the NSF-HiFiGAN vocoder. A `.onnx` file next to it is preferred if present, otherwise the checkpoint is exported to ONNX once at first startup. An explicit `.onnx` path is used directly. |
| `model_type` | `ckpt` | Inferred from the file suffix (`.onnx` → `onnx`, `.ckpt` → `ckpt`). No need to set it manually. |
| `hnsep_model_path` | `null` (disabled) | Path to the hnsep vocal-separation model (`hnsep/vr/model.pt` after running setup). Set to enable `Hb`/`Hv`/`Ht` flags. The default `config.yaml` sets this, so hnsep is on by default when installing. |
| `provider` | `auto` | ONNX Runtime execution provider for the vocoder: `auto`, `directml`, `cuda`, `coreml`, or `cpu`. `auto` picks the first available GPU provider; a specific value forces it (startup fails if unavailable). See [Hardware acceleration](hardware-acceleration.md). Independent of the top-level `device`. This one controls the vocoder. |

## `audio`

Mel spectrogram / vocoder parameters. Matched to the distributed vocoder model. **DO NOT** change these unless you supply your own compatible model.

| Key | Default |
|---|---|
| `sample_rate` | `44100` |
| `win_size` | `2048` |
| `hop_size` | `512` |
| `origin_hop_size` | `128` |
| `n_fft` | `2048` |
| `n_mels` | `128` |
| `mel_fmin` | `40` |
| `mel_fmax` | `16000` |

## `processing`

| Key | Default | Description |
|---|---|---|
| `wave_norm` | `false` | Loudness normalization of the rendered output (needs `pyloudnorm`, included). When enabled, normalizes toward −16 LUFS; the `P` flag scales the strength (0–100, default 100). |
| `loop_mode` | `false` | Loop the rendered region to fill long notes (equivalent to always applying the `He` flag). |
| `peak_limit` | `1.0` | If the output peak exceeds this, it is scaled down to it. |
| `fill` | `6` | Margin (in frames) kept around the requested render span. |

## `performance`

| Key | Default | Description |
|---|---|---|
| `max_concurrent_renders` | `4` in the default `config.yaml` (`8` if the key is absent) | Size of the render queue. The vocoder always runs serially in a single worker thread; requests beyond this many queued jobs wait for a free slot (they are not rejected). Raise it if many notes are rendered in parallel and you see large `queue_ms` values in the logs or the server just crashes, lower it to bound memory use. |

## Top-level keys

| Key | Default | Description |
|---|---|---|
| `device` | auto-detected | Where PyTorch runs mel extraction and hnsep: `cuda`, `mps`, or `cpu` (first available wins). **Does not affect the vocoder**. See [Hardware acceleration](hardware-acceleration.md). |
| `port` | `8572` | Server port. The client always connects to `127.0.0.1:8572`, so change both together (and build the client yourself) or leave it as-is. |
