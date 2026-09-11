"""Native audio preprocessing used by F5-TTS and its Vocos decoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from voicehub.processing.audio import htk_mel_filter_bank as _htk_mel_filter_bank


def htk_mel_filter_bank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Return torchaudio-compatible, unnormalised HTK mel filters.

    The returned orientation is ``[frequency, mel]``, matching the
    buffer in ``charactr/vocos-mel-24khz``.
    """
    return _htk_mel_filter_bank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        minimum_frequency=f_min,
        maximum_frequency=f_max,
    )


class F5MelSpectrogram(nn.Module):
    """Magnitude log-mel frontend matching F5-TTS' Vocos configuration."""

    def __init__(
        self,
        *,
        sample_rate: int = 24_000,
        n_fft: int = 1_024,
        hop_length: int = 256,
        win_length: int = 1_024,
        n_mels: int = 100,
        clamp_min: float = 1e-5,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.clamp_min = clamp_min
        self.register_buffer(
            "window",
            torch.hann_window(win_length),
            persistent=False,
        )
        self.register_buffer(
            "filter_bank",
            htk_mel_filter_bank(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
            ),
            persistent=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0]
        if waveform.ndim != 2:
            raise ValueError("F5-TTS mel extraction expects `[batch, samples]` audio.")
        # cuFFT does not implement FP16/BF16 transforms, and running the
        # logarithmic acoustic frontend in FP32 is also materially more
        # stable near ``clamp_min``. Preserve the model-facing dtype at the
        # boundary so mixed-precision DiT inference remains end-to-end
        # compatible.
        output_dtype = waveform.dtype
        fft_dtype = (waveform.dtype if waveform.dtype in {torch.float32, torch.float64} else torch.float32)
        fft_waveform = waveform.to(dtype=fft_dtype)
        window = self.window.to(device=waveform.device, dtype=fft_dtype)
        spectrum = torch.stft(
            fft_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()
        filter_bank = self.filter_bank.to(
            device=waveform.device,
            dtype=spectrum.dtype,
        )
        mel = torch.matmul(spectrum.transpose(-1, -2), filter_bank)
        return (mel.transpose(-1, -2).clamp_min(self.clamp_min).log().to(dtype=output_dtype))


def normalize_reference_rms(
    waveform: torch.Tensor,
    *,
    target_rms: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """Raise a quiet reference to the source recipe's target RMS."""
    rms = float(torch.sqrt(torch.mean(waveform.float().square())).item())
    if not math.isfinite(rms):
        raise ValueError("Reference audio contains non-finite samples.")
    if rms <= 0:
        raise ValueError("Reference audio is silent.")
    if rms < target_rms:
        return waveform * (target_rms / rms), rms
    return waveform, rms


def trim_silence(
    waveform: torch.Tensor,
    *,
    threshold: float = 1e-3,
    padding: int = 0,
) -> torch.Tensor:
    """Trim leading/trailing low-energy samples without a DSP dependency."""
    if waveform.ndim != 1:
        raise ValueError("Silence trimming expects one mono waveform.")
    active = torch.where(waveform.abs() >= threshold)[0]
    if active.numel() == 0:
        return waveform[:0]
    start = max(0, int(active[0].item()) - padding)
    end = min(waveform.numel(), int(active[-1].item()) + padding + 1)
    return waveform[start:end]


def cross_fade(
    first: torch.Tensor,
    second: torch.Tensor,
    overlap_samples: int,
) -> torch.Tensor:
    """Concatenate mono waveforms using a linear cross-fade."""
    overlap = min(overlap_samples, first.numel(), second.numel())
    if overlap <= 0:
        return torch.cat((first, second))
    fade_out = torch.linspace(
        1.0,
        0.0,
        overlap,
        device=first.device,
        dtype=first.dtype,
    )
    fade_in = 1.0 - fade_out
    mixed = first[-overlap:] * fade_out + second[:overlap] * fade_in
    return torch.cat((first[:-overlap], mixed, second[overlap:]))


def pad_mel(
    mel: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    if target_length < mel.shape[1]:
        raise ValueError("Target mel length cannot be shorter than the input.")
    return F.pad(mel, (0, 0, 0, target_length - mel.shape[1]))


__all__ = [
    "F5MelSpectrogram",
    "cross_fade",
    "htk_mel_filter_bank",
    "normalize_reference_rms",
    "pad_mel",
    "trim_silence",
]
