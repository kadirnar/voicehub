"""PyTorch-only waveform and mel operations used by XTTS v2."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from voicehub.processing.waveform import load_pcm_wave, resample_waveform


def load_reference_audio(
    path,
    *,
    sample_rate: int,
    device: torch.device | str | None = None,
) -> Tensor:
    waveform, source_rate = load_pcm_wave(path)
    if source_rate != sample_rate:
        waveform = resample_waveform(waveform, source_rate, sample_rate)
    return waveform.clamp(-1, 1).unsqueeze(0).to(device=device)


def _hz_to_mel(value: Tensor) -> Tensor:
    return 2_595.0 * torch.log10(1.0 + value / 700.0)


def _mel_to_hz(value: Tensor) -> Tensor:
    return 700.0 * (torch.pow(10.0, value / 2_595.0) - 1.0)


def mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    slaney_norm: bool = True,
    device: torch.device | str | None = None,
) -> Tensor:
    frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, device=device)
    mel_edges = torch.linspace(
        _hz_to_mel(torch.tensor(float(f_min), device=device)),
        _hz_to_mel(torch.tensor(float(f_max), device=device)),
        n_mels + 2,
        device=device,
    )
    hz_edges = _mel_to_hz(mel_edges)
    lower = hz_edges[:-2, None]
    center = hz_edges[1:-1, None]
    upper = hz_edges[2:, None]
    left = (frequencies - lower) / (center - lower).clamp_min(1e-12)
    right = (upper - frequencies) / (upper - center).clamp_min(1e-12)
    filters = torch.minimum(left, right).clamp_min(0)
    if slaney_norm:
        enorm = 2.0 / (upper[:, 0] - lower[:, 0]).clamp_min(1e-12)
        filters = filters * enorm[:, None]
    return filters


class MelSpectrogram(nn.Module):

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float = 0,
        f_max: float | None = None,
        hamming: bool = False,
        power: float = 2.0,
        slaney_norm: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = sample_rate / 2 if f_max is None else f_max
        self.power = power
        window = (
            torch.hamming_window(win_length, periodic=True) if hamming else torch.hann_window(
                win_length, periodic=True))
        filters = mel_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max,
            slaney_norm=slaney_norm,
        )
        # Keep torchaudio's historical buffer namespace so published XTTS
        # speaker-encoder tensors validate without key rewriting.
        self.spectrogram = _SpectrogramBuffers(window)
        self.mel_scale = _MelScaleBuffers(filters.transpose(0, 1))

    def forward(self, waveform: Tensor) -> Tensor:
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.spectrogram.window.to(device=waveform.device, dtype=waveform.dtype),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).abs().pow(self.power)
        return torch.matmul(
            self.mel_scale.fb.to(device=waveform.device, dtype=waveform.dtype).transpose(0, 1),
            spectrum,
        )


class _SpectrogramBuffers(nn.Module):

    def __init__(self, window: Tensor) -> None:
        super().__init__()
        self.register_buffer("window", window)


class _MelScaleBuffers(nn.Module):

    def __init__(self, filters: Tensor) -> None:
        super().__init__()
        self.register_buffer("fb", filters)


def cloning_mel(
    waveform: Tensor,
    mel_norms: Tensor,
    *,
    sample_rate: int = 22_050,
    n_fft: int = 2_048,
    hop_length: int = 256,
    win_length: int = 1_024,
) -> Tensor:
    transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=80,
        f_min=0,
        f_max=8_000,
    ).to(device=waveform.device)
    mel = transform(waveform)
    mel = torch.log(mel.clamp_min(1e-5))
    return mel / mel_norms.to(device=mel.device, dtype=mel.dtype)[None, :, None]


__all__ = [
    "MelSpectrogram",
    "cloning_mel",
    "load_reference_audio",
    "mel_filterbank",
]
