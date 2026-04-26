from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn


class PitchAdjustableMelSpectrogram:
    mel_basis: dict[str, torch.Tensor]
    hann_window: dict[str, torch.Tensor]

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        f_min: int = 40,
        f_max: int = 16000,
        n_mels: int = 128,
        center: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.center = center
        self.mel_basis = {}
        self.hann_window = {}

    def __call__(
        self,
        y: torch.Tensor,
        key_shift: float = 0,
        speed: float = 1.0,
    ) -> torch.Tensor:
        factor = 2 ** (key_shift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_size_new = int(np.round(self.win_size * factor))
        hop_length = int(np.round(self.hop_length * speed))

        mel_basis_key = f"{self.f_max}_{y.device}"
        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max,
            )
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)

        hann_window_key = f"{key_shift}_{y.device}"
        if hann_window_key not in self.hann_window:
            self.hann_window[hann_window_key] = torch.hann_window(
                win_size_new, device=y.device
            )

        y = F.pad(
            y.unsqueeze(1),
            (
                (win_size_new - hop_length) // 2,
                (win_size_new - hop_length + 1) // 2,
            ),
            mode="reflect",
        ).squeeze(1)

        spec = torch.stft(
            y,
            n_fft_new,
            hop_length=hop_length,
            win_length=win_size_new,
            window=self.hann_window[hann_window_key],
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()

        if key_shift != 0:
            size = self.n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * self.win_size / win_size_new

        return torch.matmul(self.mel_basis[mel_basis_key], spec)

    def dynamic_range_compression_torch(
        self, x: torch.Tensor, C: float = 1, clip_val: float = 1e-5
    ) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=clip_val) * C)
