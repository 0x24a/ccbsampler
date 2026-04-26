from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, field_validator

# All supported UTAU flags
FlagKey = Literal[
    "fe",
    "fl",
    "fo",
    "fv",
    "fp",
    "ve",
    "vo",
    "g",
    "t",
    "A",
    "B",
    "P",
    "S",
    "p",
    "R",
    "D",
    "C",
    "Z",
    "Hv",
    "Hb",
    "Ht",
    # Switch-only flags (no value) — pass True
    "G",
    "He",
]

# bool must come before int, otherwise Pydantic v2 coerces True -> 1
FlagValue = Union[bool, int, float]


class ResampleRequest(BaseModel):
    in_file: str
    out_file: str
    pitch: str  # e.g. "A4", "C#3"
    velocity: float
    flags: dict[FlagKey, FlagValue] = {}
    offset: float = 0.0
    length: int = 1000
    consonant: float = 0.0
    cutoff: float = 0.0
    volume: float = 100.0
    modulation: float = 0.0
    tempo: float = 100.0
    pitchbend: list[float] = []  # cents array, decoded by Rust client

    @field_validator("flags", mode="before")
    @classmethod
    def validate_flags(cls, v: object) -> object:
        if not isinstance(v, dict):
            raise ValueError("flags must be a dict")
        for key, val in v.items():
            if not isinstance(val, (bool, int, float)):
                raise ValueError(
                    f"flag '{key}' has invalid value type {type(val).__name__}; "
                    "expected bool, int, or float"
                )
        return v


class RenderMetrics(BaseModel):
    feature_ms: float  # mel extraction (cache miss) or cache load
    queue_ms: float  # time spent waiting in GPU queue
    infer_ms: float  # vocoder inference + postprocessing
    total_ms: float


class ResampleResponse(BaseModel):
    status: Literal["ok", "error"]
    out_file: str | None = None
    error: str | None = None
    metrics: RenderMetrics | None = None
