from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import yaml
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelConfig(BaseSettings):
    vocoder_path: str = "./pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt"
    model_type: str = "ckpt"  # inferred from suffix
    hnsep_model_path: str | None = None  # None = disabled


class AudioConfig(BaseSettings):
    sample_rate: int = 44100
    win_size: int = 2048
    hop_size: int = 512
    origin_hop_size: int = 128
    n_fft: int = 2048
    n_mels: int = 128
    mel_fmin: float = 40.0
    mel_fmax: float = 16000.0


class ProcessingConfig(BaseSettings):
    wave_norm: bool = False
    loop_mode: bool = False
    peak_limit: float = 1.0
    fill: int = 6


class PerformanceConfig(BaseSettings):
    # Requests beyond this block (await) until a slot is free
    max_concurrent_renders: int = 8


class Settings(BaseSettings):
    model: ModelConfig = ModelConfig()
    audio: AudioConfig = AudioConfig()
    processing: ProcessingConfig = ProcessingConfig()
    performance: PerformanceConfig = PerformanceConfig()
    device: str = _detect_device()
    port: int = 8572

    @field_validator("device", mode="before")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cuda", "mps", "cpu"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def infer_model_type(self) -> Settings:
        suffix = Path(self.model.vocoder_path).suffix
        if suffix == ".onnx":
            self.model.model_type = "onnx"
        elif suffix == ".ckpt":
            self.model.model_type = "ckpt"
        return self


def load_settings(path: str | Path = "config.yaml") -> Settings:
    path = Path(path)
    if not path.exists():
        logging.fatal(
            "Config file not found! Please run `uv run setup.py` to create one."
        )
        sys.exit(1)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return Settings(
        model=ModelConfig(**data.get("model", {})),
        audio=AudioConfig(**data.get("audio", {})),
        processing=ProcessingConfig(**data.get("processing", {})),
        performance=PerformanceConfig(**data.get("performance", {})),
        device=data.get("device", _detect_device()),
        port=data.get("port", 8572),
    )
