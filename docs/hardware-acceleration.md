# Hardware acceleration

How to get ccbsampler's vocoder running on your GPU and how to tell whether it worked.

> [!NOTE]
> Hardware acceleration is NOT always faster. GPU execution providers add overhead, and for a small, per-note model like this vocoder that overhead can outweigh the gains. Especially on macOS, where CoreML may actually be slower than the CPU Provider on long notes. Don't assume GPU is better. Instead, compare `infer_ms` in the per-render log with and without enabling the GPU provider and keep whichever suits your machine.

## Vocoder and Mel Extraction & hnsep Separation Runtimes

ccbsampler uses two runtimes, and the `device` setting in `config.yaml` only controls one of them:

| Stage | Runtime | Controlled by | Notes |
|---|---|---|---|
| Vocoder inference | ONNX Runtime | ONNX Runtime **execution providers** instead of `device` | Runs even if `device: cpu` |
| Mel extraction & hnsep separation | PyTorch | `device` in `config.yaml` | Auto-detected: cuda -> mps -> cpu |

The render time heavily depends on the vocoder, so "hardware acceleration" in practice means giving ONNX Runtime a GPU execution provider. At startup, the server picks the first available provider in this order (when `model.provider` is `auto`, the default):

```
1 DmlExecutionProvider
2 CUDAExecutionProvider
3 CoreMLExecutionProvider
4 CPUExecutionProvider
```

Setting `device: cuda` in `config.yaml` does **not** accelerate the vocoder, it only moves the feature-extraction stage to CUDA.

## Choosing a provider

Use `model.provider` in `config.yaml`:

```yaml
model:
  provider: auto   # auto | directml | cuda | coreml | cpu
```

| Value | Behavior |
|---|---|
| `auto` (default) | First available of Dml / CUDA / CoreML, with CPU as fallback. |
| `directml` / `cuda` / `coreml` | Force that GPU provider. Server fails to start if the provider isn't available in the installed onnxruntime build. |
| `cpu` | Force CPU-only vocode. Useful where the GPU provider is slower. |

Forcing a provider that isn't installed fails at startup with a clear error. e.g. `cuda` on a macOS default install tells you the build only contains CoreML. Install the matching package (see the platform sections below) or switch back to `auto`.

## What you get by default

The default dependency is the CPU `onnxruntime` package, which ships different providers per platform:

| Platform | Default providers | Default GPU? |
|---|---|---|
| macOS | CoreML + CPU | Yes |
| Windows | CPU only | No |
| Linux | CPU only | No |

On Windows and Linux you need to swap in a GPU build of ONNX Runtime. The rest of this page explains how.

## Windows (DirectML, any GPU vendor)

DirectML works on any DirectX 12 GPU (NVIDIA, AMD, Intel):

```
uv remove onnxruntime
uv add onnxruntime-directml
```

Then restart the server. No other setup is required.

## Linux / Windows (CUDA, NVIDIA only)

The CUDA provider needs an NVIDIA GPU and matching CUDA libraries. The easiest route is the PyPI extras, which pull CUDA and cuDNN in as regular Python wheels:

```
uv remove onnxruntime
uv add "onnxruntime-gpu[cuda,cudnn]"
```

If you prefer using a system-installed CUDA toolkit instead of the wheels, install CUDA and cuDNN yourself and just `uv add onnxruntime-gpu` and make sure their library directories are on `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux).

## macOS (CoreML)

Nothing to do. the default `onnxruntime` package already includes the CoreML provider.

If CoreML turns out slower than CPU on your machine (see the note at the top), set `provider: cpu` in `config.yaml` to force the CPU provider.

## Verifying that it worked

Check the first lines the server prints after loading the model:

```
INFO (ccbsampler) Loaded onnx vocoder with providers ['CoreMLExecutionProvider', 'CPUExecutionProvider']
```

("CPUExecutionProvider" is normal here since it's always used as a fallback.)

What each result means:

| Provider shown | Meaning |
|---|---|
| `DmlExecutionProvider` | DirectML active (Windows, any DX12 GPU) |
| `CUDAExecutionProvider` | CUDA active (NVIDIA) |
| `CoreMLExecutionProvider` | macOS GPU active (CoreML) |
| only `CPUExecutionProvider` | Running on CPU. See troubleshooting below |

The per-render log line also shows the effect:

```
do.wav → do.wav.out   feature=  12ms  queue=   0ms  infer= 210ms  total= 240ms
```

Use it to compare `infer_ms` between providers on your own machine as noted above, the GPU provider is not guaranteed to be faster.

## Troubleshooting

**Providers show only CPU after installing a GPU package.**
Make sure the swap actually took effect. `uv remove onnxruntime` must succeed so `onnxruntime` itself is gone from the environment, having both installed can shadow the GPU build. Then restart the server.

**`device: cuda` / `device: mps` set, but no speedup.**
Completely expected. `device` only affects mel extraction and hnsep, not the vocoder. See [Vocoder and Mel Extraction & hnsep Separation Runtimes](#vocoder-and-mel-extraction--hnsep-separation-runtimes).

**hnsep separation (`Hb`/`Hv`/`Ht` flags) is slow on Apple Silicon.**
hnsep runs through PyTorch and does use MPS when available; it's simply a heavier model. If you don't use these flags, you can disable hnsep entirely by removing `hnsep_model_path` from `config.yaml`, the server then loads one less model and rejects those flags.

**Windows: DirectML device selection.**
DirectML always uses the system default adapter. If you have multiple GPUs and want a different one, change the preferred GPU in Windows' graphics settings, or update your GPU driver if renders misbehave.

**First launch is slow.**
If no `.onnx` file sits next to the vocoder checkpoint, the server exports one at startup (one-time). Future launches skip this.
