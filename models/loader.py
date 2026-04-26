from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import onnxruntime
import torch
import yaml

from config import Settings
from util.wav2mel import PitchAdjustableMelSpectrogram

logger = logging.getLogger(__name__)


class Vocoder(Protocol):
    def infer(self, mel: torch.Tensor, f0: torch.Tensor) -> np.ndarray:
        """mel: [1, n_mels, T]  f0: [1, T]  → waveform ndarray [N]"""
        ...


class CkptVocoder:
    def __init__(self, model_path: Path, device: torch.device) -> None:
        # import here so onnx-only setups don't need torch model code
        from util.nsf_hifigan import NsfHifiGAN

        self._vocoder = NsfHifiGAN(model_path=model_path)
        # MPS: ConvTranspose1d output channels > 65536 not supported — fall back to CPU
        vocoder_device = torch.device("cpu") if device.type == "mps" else device
        self._vocoder.to_device(vocoder_device)
        self._device = vocoder_device
        logger.info("Loaded ckpt vocoder on %s", vocoder_device)

    def infer(self, mel: torch.Tensor, f0: torch.Tensor) -> np.ndarray:
        mel = mel.to(self._device)
        f0 = f0.to(self._device)
        with torch.no_grad():
            wav = self._vocoder.spec2wav_torch(mel, f0=f0)
        return wav.cpu().numpy()


class OnnxVocoder:
    def __init__(self, model_path: Path) -> None:
        available = onnxruntime.get_available_providers()
        providers: list[str] = []
        for ep in (
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
        ):
            if ep in available:
                providers.append(ep)
                break
        providers.append("CPUExecutionProvider")

        self._session = onnxruntime.InferenceSession(
            str(model_path.resolve()), providers=providers
        )
        logger.info(
            "Loaded onnx vocoder with providers %s", self._session.get_providers()
        )

    def infer(self, mel: torch.Tensor, f0: torch.Tensor) -> np.ndarray:
        mel_np = mel.cpu().numpy().astype(np.float32)  # [1, n_mels, T]
        f0_np = f0.cpu().numpy().astype(np.float32)  # [1, T]
        output: Any = self._session.run(["waveform"], {"mel": mel_np, "f0": f0_np})[0]
        return output.reshape(-1)  # flatten [1,1,N] or [1,N] or [N] → [N]


@dataclass
class HnsepBundle:
    model: torch.nn.Module
    args: object  # DotDict


def load_hnsep(model_path: Path, device: torch.device) -> HnsepBundle:
    from hnsep.nets import CascadedNet

    config_file = model_path.parent / "config.yaml"
    with open(config_file) as f:
        args_raw = yaml.safe_load(f)

    class DotDict(dict):
        def __getattr__(self, *a):
            val = dict.get(self, *a)
            return DotDict(val) if type(val) is dict else val

        __setattr__ = dict.__setitem__  # type: ignore[assignment]
        __delattr__ = dict.__delitem__  # type: ignore[assignment]

    args = DotDict(args_raw)
    a: Any = args  # untyped...
    model = CascadedNet(
        a.n_fft,
        a.hop_length,
        int(a.n_out),
        int(a.n_out_lstm),
        True,
        is_mono=bool(a.is_mono),
        fixed_length=True if a.fixed_length is None else bool(a.fixed_length),
    )
    model.to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Failed to load hnsep model: {e}\nPlease run `uv run setup.py models` to download the necessary models."
        )
    model.eval()
    logger.info("Loaded hnsep model on %s", device)
    return HnsepBundle(model=model, args=args)


@dataclass
class ModelBundle:
    vocoder: Vocoder
    hnsep: HnsepBundle | None
    mel_analysis: PitchAdjustableMelSpectrogram


def _export_onnx(ckpt_path: Path) -> Path:
    from util.nsf_hifigan import load_model as load_hifigan

    onnx_path = ckpt_path.with_suffix(".onnx")
    logger.info("Exporting vocoder to ONNX: %s", onnx_path)
    generator, h = load_hifigan(ckpt_path)
    generator.eval()
    T = 100
    mel_dummy = torch.randn(1, h.num_mels, T)
    f0_dummy = torch.full((1, T), 440.0)
    with torch.no_grad():
        torch.onnx.export(
            generator,
            (mel_dummy, f0_dummy),
            str(onnx_path),
            input_names=["mel", "f0"],
            output_names=["waveform"],
            dynamic_axes={
                "mel": {2: "frames"},
                "f0": {1: "frames"},
                "waveform": {1: "samples"},
            },
            opset_version=18,
            do_constant_folding=True,
        )

    # Inline external data so the .onnx is self-contained (required for CoreML EP)
    import onnx

    model_proto = onnx.load(str(onnx_path))
    onnx.save_model(
        model_proto,
        str(onnx_path),
        save_as_external_data=False,
    )

    logger.info(
        "ONNX export done: %s (%.1f MB)",
        onnx_path,
        onnx_path.stat().st_size / 1_048_576,
    )
    return onnx_path


def load_models(settings: Settings) -> ModelBundle:
    device = torch.device(settings.device)
    vocoder_path = Path(settings.model.vocoder_path)

    if settings.model.model_type == "ckpt":
        # Auto-export to ONNX if not present alongside the ckpt
        onnx_path = vocoder_path.with_suffix(".onnx")
        if onnx_path.exists():
            logger.info("Found existing ONNX alongside ckpt, using it")
            vocoder: Vocoder = OnnxVocoder(onnx_path)
        else:
            logger.info(
                "No ONNX found, exporting from ckpt (one-time, may take a moment)"
            )
            onnx_path = _export_onnx(vocoder_path)
            vocoder = OnnxVocoder(onnx_path)
    else:
        vocoder = OnnxVocoder(vocoder_path)

    hnsep: HnsepBundle | None = None
    if settings.model.hnsep_model_path is not None:
        hnsep = load_hnsep(Path(settings.model.hnsep_model_path), device)

    a = settings.audio
    mel_analysis = PitchAdjustableMelSpectrogram(
        sample_rate=a.sample_rate,
        n_fft=a.n_fft,
        win_length=a.win_size,
        hop_length=a.origin_hop_size,
        f_min=int(a.mel_fmin),
        f_max=int(a.mel_fmax),
        n_mels=a.n_mels,
    )
    logger.info("Mel analysis initialized (hop=%d)", a.origin_hop_size)

    return ModelBundle(vocoder=vocoder, hnsep=hnsep, mel_analysis=mel_analysis)
