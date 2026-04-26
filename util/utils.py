from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicSpline


def get_mel_fn(
    sr: float,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    htk: bool,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Returns mel filterbank weights: Tensor [n_mels, n_fft // 2 + 1]
    htk: use HTK formula (True) or Slaney formula (False)
    """
    fmin_t = torch.tensor(fmin, device=device)
    fmax_t = torch.tensor(fmax, device=device)

    if htk:
        min_mel = 2595.0 * torch.log10(1.0 + fmin_t / 700.0)
        max_mel = 2595.0 * torch.log10(1.0 + fmax_t / 700.0)
        mels = torch.linspace(min_mel, max_mel, n_mels + 2, device=device)
        mel_f = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    else:
        f_sp = 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = torch.log(torch.tensor(6.4, device=device)) / 27.0

        min_mel = (
            min_log_mel + torch.log(fmin_t / min_log_hz) / logstep
            if fmin >= min_log_hz
            else fmin_t / f_sp
        )
        max_mel = (
            min_log_mel + torch.log(fmax_t / min_log_hz) / logstep
            if fmax >= min_log_hz
            else fmax_t / f_sp
        )

        mels = torch.linspace(min_mel, max_mel, n_mels + 2, device=device)
        mel_f = torch.zeros_like(mels)
        log_t = mels >= min_log_mel
        mel_f[~log_t] = f_sp * mels[~log_t]
        mel_f[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

    N = 1 + n_fft // 2
    fftfreqs = (sr / n_fft) * torch.arange(0, N, device=device)
    fdiff = torch.diff(mel_f)
    ramps = mel_f.unsqueeze(1) - fftfreqs.unsqueeze(0)
    lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
    upper = ramps[2:] / fdiff[1:].unsqueeze(1)
    weights = torch.max(torch.tensor(0.0), torch.min(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm.unsqueeze(1)
    return weights


def expand_uv(uv: np.ndarray) -> np.ndarray:
    uv = uv.astype("float")
    uv = np.min(np.array([uv[:-2], uv[1:-1], uv[2:]]), axis=0)
    return np.pad(uv, (1, 1), constant_values=(uv[0], uv[-1]))


def norm_f0(f0: np.ndarray, uv: np.ndarray | None = None) -> np.ndarray:
    if uv is None:
        uv = f0 == 0
    f0 = np.log2(f0 + uv)  # avoid log(0)
    f0[uv] = -np.inf
    return f0


def denorm_f0(
    f0: np.ndarray,
    uv: np.ndarray | None,
    pitch_padding: np.ndarray | None = None,
) -> np.ndarray:
    f0 = 2.0 ** f0
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def interp_f0_spline(
    f0: np.ndarray, uv: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    uv_mask: np.ndarray = (f0 == 0) if uv is None else uv
    f0max = np.max(f0)
    f0 = norm_f0(f0, uv_mask)
    if uv_mask.any() and not uv_mask.all():
        spline = CubicSpline(np.where(~uv_mask)[0], f0[~uv_mask])
        f0[uv_mask] = spline(np.where(uv_mask)[0])
    return np.clip(denorm_f0(f0, uv=None), 0, f0max), uv_mask


def interp_f0(
    f0: np.ndarray, uv: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    uv_mask: np.ndarray = (f0 == 0) if uv is None else uv
    f0 = norm_f0(f0, uv_mask)
    if uv_mask.any() and not uv_mask.all():
        f0[uv_mask] = np.interp(np.where(uv_mask)[0], np.where(~uv_mask)[0], f0[~uv_mask])
    return denorm_f0(f0, uv=None), uv_mask


class AttrDict(dict):
    """dict with attribute-style access."""

    def __getstate__(self) -> list[tuple]:  # type: ignore[override]
        return list(self.__dict__.items())

    def __setstate__(self, items: list[tuple]) -> None:
        for key, val in items:
            self.__dict__[key] = val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict.__repr__(self)})"

    __getattr__ = dict.__getitem__  # type: ignore[assignment]
    __setattr__ = dict.__setitem__  # type: ignore[assignment]
    __delattr__ = dict.__delitem__  # type: ignore[assignment]

    def copy(self) -> AttrDict:
        return AttrDict(self)


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)  # type: ignore[union-attr]


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2
