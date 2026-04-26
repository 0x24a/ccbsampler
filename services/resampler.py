from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import resampy
import scipy.interpolate as interp
import soundfile as sf
import torch
import torch.nn.functional as F

from cache.manager import CacheManager
from config import Settings
from models.loader import ModelBundle
from schemas import RenderMetrics, ResampleRequest, ResampleResponse

logger = logging.getLogger(__name__)

_NOTES = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}
_NOTE_RE = re.compile(r"([A-G]#?)(-?\d+)")
_HNSEP_FLAGS = frozenset({"Hb", "Hv", "Ht"})


@dataclass
class InferencePayload:
    req: ResampleRequest
    mel_origin: np.ndarray  # [n_mels, T_origin]
    scale: float
    feature_ms: float  # time spent on feature extraction / cache
    queue_ms: float = 0.0  # filled in by GPUQueue worker before infer


def _note_to_midi(x: str) -> int:
    m = _NOTE_RE.match(x)
    assert m is not None, f"Invalid pitch string: {x!r}"
    note, octave = m.group(1, 2)
    return (int(octave) + 1) * 12 + _NOTES[note]


def _midi_to_hz(x: np.ndarray) -> np.ndarray:
    return 440.0 * np.exp2((x - 69.0) / 12.0)


def _drc(x: torch.Tensor, clip_val: float = 1e-9) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val))


def _read_wav(path: Path, sample_rate: int) -> np.ndarray:
    if not path.exists():
        for ext in sf.available_formats():
            alt = path.with_suffix("." + ext.lower())
            if alt.exists():
                path = alt
                break
        else:
            raise FileNotFoundError(f"No audio file found at {path}")
    x, fs = sf.read(str(path))
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    if fs != sample_rate:
        x = resampy.resample(x, fs, sample_rate)
    return x.astype(np.float32)


def _pre_emphasis_tension(wave: torch.Tensor, b: float, cfg: Settings) -> torch.Tensor:
    a = cfg.audio
    orig_len = wave.size(-1)
    pad = (a.hop_size - orig_len % a.hop_size) % a.hop_size
    w = wave.squeeze(1)
    w = F.pad(w, (0, pad))

    win = torch.hann_window(a.win_size, device=w.device)
    spec = torch.stft(
        w,
        a.n_fft,
        hop_length=a.hop_size,
        win_length=a.win_size,
        window=win,
        return_complex=True,
    )
    amp = torch.abs(spec)
    phase = torch.atan2(spec.imag, spec.real)
    amp_db = torch.log(torch.clamp(amp, min=1e-9))

    fft_bin = a.n_fft // 2 + 1
    x0 = fft_bin / ((a.sample_rate / 2) / 1500)
    freq_filter = (-b / x0) * torch.arange(fft_bin, device=w.device) + b
    amp_db = amp_db + torch.clamp(freq_filter, -2.0, 2.0).unsqueeze(0).unsqueeze(2)

    filtered = torch.istft(
        torch.complex(
            torch.exp(amp_db) * torch.cos(phase), torch.exp(amp_db) * torch.sin(phase)
        ),
        n_fft=a.n_fft,
        hop_length=a.hop_size,
        win_length=a.win_size,
        window=win,
    )
    orig_max = torch.max(torch.abs(w))
    filt_max = torch.max(torch.abs(filtered))
    filtered = filtered * (orig_max / filt_max) * (float(np.clip(b / -15, 0, 0.33)) + 1)
    return filtered.unsqueeze(1)[:, :, :orig_len]


class Renderer:
    def __init__(
        self, settings: Settings, models: ModelBundle, cache: CacheManager
    ) -> None:
        self.cfg = settings
        self.models = models
        self.cache = cache

    async def prepare(self, req: ResampleRequest) -> InferencePayload:
        """
        Async: validate, extract/cache features.
        Returns an InferencePayload ready for the GPU worker.
        """
        self._check_hnsep_flags(req)
        t0 = time.monotonic()
        mel_origin, scale = await self._get_features(req)
        feature_ms = (time.monotonic() - t0) * 1000
        return InferencePayload(
            req=req, mel_origin=mel_origin, scale=scale, feature_ms=feature_ms
        )

    def infer(self, payload: InferencePayload) -> ResampleResponse:
        """Sync: vocoder inference + write output file."""
        t0 = time.monotonic()
        try:
            infer_ms = self._resample(payload)
            total_ms = (time.monotonic() - t0) * 1000
            metrics = RenderMetrics(
                feature_ms=round(payload.feature_ms, 1),
                queue_ms=round(payload.queue_ms, 1),
                infer_ms=round(infer_ms, 1),
                total_ms=round(total_ms, 1),
            )
            logger.info(
                "%-30s  feature=%5.0fms  queue=%5.0fms  infer=%5.0fms  total=%5.0fms",
                f"{Path(payload.req.in_file).stem} → {Path(payload.req.out_file).name}",
                metrics.feature_ms,
                metrics.queue_ms,
                metrics.infer_ms,
                metrics.total_ms,
            )
            return ResampleResponse(
                status="ok", out_file=payload.req.out_file, metrics=metrics
            )
        except Exception as exc:
            logger.exception("Inference failed for %s", payload.req.in_file)
            return ResampleResponse(status="error", error=str(exc))

    def _check_hnsep_flags(self, req: ResampleRequest) -> None:
        if self.models.hnsep is None:
            used = _HNSEP_FLAGS & req.flags.keys()
            if used:
                raise ValueError(
                    f"hnsep is disabled but flags {sorted(used)} were used. "
                    "Set hnsep_model_path in config.yaml to enable."
                )

    async def _get_features(self, req: ResampleRequest) -> tuple[np.ndarray, float]:
        flags = req.flags
        hb = int(flags.get("Hb", 100))
        hv = int(flags.get("Hv", 100))
        ht = int(flags.get("Ht", 0))
        g = int(flags.get("g", 0))
        flag_suffix = f"Hb{hb}_Hv{hv}_Ht{ht}_g{g}"

        wav_path = Path(req.in_file)
        force = req.flags.get("G") is True

        features = await self.cache.get_or_generate(
            wav_path=wav_path,
            flag_suffix=flag_suffix,
            force=force,
            generate_fn=lambda: self._generate_features(wav_path, hb, hv, ht, g),
        )
        return features["mel_origin"], float(features["scale"])

    def _generate_features(
        self, wav_path: Path, hb: int, hv: int, ht: int, g: int
    ) -> dict:
        a = self.cfg.audio
        device = torch.device(self.cfg.device)

        wave = _read_wav(wav_path, a.sample_rate)
        wave_t = torch.from_numpy(wave).to(device).unsqueeze(0).unsqueeze(0)  # [1,1,T]

        if hb != 100 or hv != 100 or ht != 0:
            logger.info("hnsep separation (Hb=%d Hv=%d Ht=%d)", hb, hv, ht)
            assert self.models.hnsep is not None
            with torch.no_grad():
                harmonic = self.models.hnsep.model.predict_fromaudio(wave_t)  # type: ignore[operator]
            hb_c = float(np.clip(hb, 0, 500))
            hv_c = float(np.clip(hv, 0, 150))
            if ht != 0:
                ht_c = float(np.clip(ht, -100, 100))
                wave_t = (hb_c / 100) * (wave_t - harmonic) + _pre_emphasis_tension(
                    (hv_c / 100) * harmonic, -ht_c / 50, self.cfg
                )
            else:
                wave_t = (hb_c / 100) * (wave_t - harmonic) + (hv_c / 100) * harmonic

        wave_t = wave_t.squeeze().unsqueeze(0)  # [1, T]
        wave_max = torch.max(torch.abs(wave_t))
        if wave_max >= 0.5:
            scale = (0.5 / wave_max).item()
            wave_t = wave_t * scale
        else:
            scale = 1.0

        g_c = float(np.clip(g, -600, 600))
        mel = self.models.mel_analysis(wave_t, g_c / 100, 1).squeeze()  # [n_mels, T]
        mel = _drc(mel).cpu().numpy()

        return {"mel_origin": mel, "scale": np.float32(scale)}

    def _resample(self, payload: InferencePayload) -> float:
        t0 = time.monotonic()
        req = payload.req
        if req.out_file == "nul":
            return (time.monotonic() - t0) * 1000

        a = self.cfg.audio
        p = self.cfg.processing
        flags = req.flags

        mel_origin = payload.mel_origin.copy()
        scale = payload.scale

        thop_o = a.origin_hop_size / a.sample_rate  # hop time at origin resolution
        thop = a.hop_size / a.sample_rate  # hop time at render resolution

        t_area = np.arange(mel_origin.shape[1]) * thop_o + thop_o / 2
        total_time = t_area[-1] + thop_o / 2

        vel = np.exp2(1.0 - req.velocity / 100.0)
        start = req.offset / 1000.0
        end = (
            (start - req.cutoff / 1000.0)
            if req.cutoff < 0
            else (total_time - req.cutoff / 1000.0)
        )
        con = start + req.consonant / 1000.0
        length_req = req.length / 1000.0

        if p.loop_mode or flags.get("He") is True:
            con_f = int((con + thop_o / 2) // thop_o)
            end_f = int((end + thop_o / 2) // thop_o)
            mel_loop = mel_origin[:, con_f:end_f]
            pad_n = int(length_req / thop_o) + 1
            padded = np.pad(mel_loop, ((0, 0), (0, pad_n)), mode="reflect")
            mel_origin = np.concatenate((mel_origin[:, :con_f], padded), axis=1)
            stretch_length = pad_n * thop_o
            t_area = np.arange(mel_origin.shape[1]) * thop_o + thop_o / 2
            total_time = t_area[-1] + thop_o / 2
        else:
            stretch_length = end - con

        mel_interp = interp.interp1d(t_area, mel_origin, axis=1)
        scaling_ratio = (
            max(length_req / stretch_length, 1.0)
            if stretch_length < length_req
            else 1.0
        )

        def stretch(t: np.ndarray) -> np.ndarray:
            return np.where(
                t < vel * con, t / vel, con + (t - vel * con) / scaling_ratio
            )

        n_frames = int((con * vel + (total_time - con) * scaling_ratio) / thop) + 1
        t_stretched = np.arange(n_frames) * thop + thop / 2

        cut_l = max(int((start * vel + thop / 2) / thop) - p.fill, 0)
        cut_r = max(
            n_frames - int((length_req + con * vel + thop / 2) / thop) - p.fill, 0
        )
        t_stretched = t_stretched[cut_l : n_frames - cut_r]

        new_start = start * vel - cut_l * thop
        new_end = (length_req + con * vel) - cut_l * thop

        mel_render = mel_interp(np.clip(stretch(t_stretched), 0, t_area[-1]))

        # Pitch
        pitch_midi = np.array(req.pitchbend) / 100.0 + _note_to_midi(req.pitch)
        if flags.get("t"):
            pitch_midi = pitch_midi + float(flags["t"]) / 100.0

        t_audio = np.arange(mel_render.shape[1]) * thop
        t_pitch = 60.0 * np.arange(len(pitch_midi)) / (req.tempo * 96.0) + new_start
        pitch_render = interp.Akima1DInterpolator(t_pitch, pitch_midi)(
            np.clip(t_audio, new_start, t_pitch[-1])
        )
        f0_render = _midi_to_hz(pitch_render)

        mel_t = torch.from_numpy(mel_render).unsqueeze(0).to(dtype=torch.float32)
        f0_t = torch.from_numpy(f0_render).unsqueeze(0).to(dtype=torch.float32)
        wav_full = self.models.vocoder.infer(mel_t, f0_t)
        logger.debug(
            "vocoder: mel_frames=%d wav_samples=%d expected=%d",
            mel_render.shape[1],
            len(wav_full),
            mel_render.shape[1] * self.cfg.audio.hop_size,
        )

        sample_l = int(new_start * a.sample_rate)
        sample_r = int(new_end * a.sample_rate)
        sample_r = min(sample_r, len(wav_full))
        render = wav_full[sample_l:sample_r]

        if len(render) == 0:
            logger.warning(
                "Empty render slice (sample_l=%d sample_r=%d wav_len=%d), writing silence",
                sample_l,
                sample_r,
                len(wav_full),
            )
            render = np.zeros(max(int(length_req * a.sample_rate), 1), dtype=np.float32)

        a_flag = flags.get("A", 0)
        if a_flag and len(pitch_render) > 1:
            a_c = float(np.clip(a_flag, -100, 100))
            deriv = np.gradient(pitch_render, t_audio)
            gain = 5.0 ** (1e-4 * a_c * deriv)
            audio_t = np.linspace(new_start, new_end, len(render), endpoint=False)
            render = render * np.interp(
                audio_t, t_audio, gain, left=gain[0], right=gain[-1]
            )

        render = render / scale

        if p.wave_norm:
            try:
                import pyloudnorm as pyln

                p_val = flags.get("P")
                strength = int(p_val) if isinstance(p_val, (int, float)) else 100
                orig_len = len(render)
                min_len = int(a.sample_rate * 0.4)
                padded = np.pad(render, (0, max(min_len - orig_len, 0)), mode="reflect")
                meter = pyln.Meter(a.sample_rate, block_size=0.4)
                loudness = meter.integrated_loudness(padded)
                target = loudness + (-16.0 - loudness) * strength / 100.0
                render = pyln.normalize.loudness(padded, loudness, target)[:orig_len]
            except ImportError:
                logger.warning("pyloudnorm not installed; skipping wave_norm")

        peak = np.max(np.abs(render))
        if peak > p.peak_limit:
            render = render / peak

        sf.write(req.out_file, render, a.sample_rate, "PCM_16")
        return (time.monotonic() - t0) * 1000
