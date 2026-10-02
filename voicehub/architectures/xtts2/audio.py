"""PyTorch-only waveform and mel operations used by XTTS v2."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from voicehub.processing.waveform import load_pcm_wave, resample_waveform_hann


def load_reference_audio(
    path,
    *,
    sample_rate: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Load like the source ``load_audio``: float32 CPU resampling with
    ``torchaudio.functional.resample`` semantics, then clipping."""
    waveform, source_rate = load_pcm_wave(path)
    waveform = waveform.to(dtype=torch.float32)
    if source_rate != sample_rate:
        waveform = resample_waveform_hann(
            waveform,
            source_rate,
            sample_rate,
            match_functional=True,
        )
    return waveform.clamp(-1, 1).unsqueeze(0).to(device=device)


def _hz_to_mel(value: float) -> float:
    return 2_595.0 * math.log10(1.0 + value / 700.0)


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
    """HTK triangular filters, ``[n_mels, n_fft // 2 + 1]``.

    Evaluated with the operation order of
    ``torchaudio.functional.melscale_fbanks`` so cloning mels match the
    source frontend bit for bit.
    """
    frequencies = torch.linspace(0, sample_rate // 2, n_fft // 2 + 1, device=device)
    mel_edges = torch.linspace(
        _hz_to_mel(float(f_min)),
        _hz_to_mel(float(f_max)),
        n_mels + 2,
        device=device,
    )
    hz_edges = _mel_to_hz(mel_edges)
    edge_differences = hz_edges[1:] - hz_edges[:-1]
    slopes = hz_edges.unsqueeze(0) - frequencies.unsqueeze(1)
    down = (-1.0 * slopes[:, :-2]) / edge_differences[:-1]
    up = slopes[:, 2:] / edge_differences[1:]
    filters = torch.max(torch.zeros(1, device=device), torch.min(down, up))
    if slaney_norm:
        enorm = 2.0 / (hz_edges[2:n_mels + 2] - hz_edges[:n_mels])
        filters = filters * enorm.unsqueeze(0)
    return filters.transpose(0, 1)


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
            spectrum.transpose(-1, -2),
            self.mel_scale.fb.to(device=waveform.device, dtype=waveform.dtype),
        ).transpose(-1, -2)


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
    """Source ``wav_to_mel_cloning``; like the source it always runs on the
    CPU in float32 and returns the mel on the waveform's device/dtype."""
    transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=80,
        f_min=0,
        f_max=8_000,
    )
    mel = transform(waveform.detach().to(device="cpu", dtype=torch.float32))
    mel = torch.log(torch.clamp(mel, min=1e-5))
    mel = mel / mel_norms.detach().to(device="cpu", dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    return mel.to(device=waveform.device, dtype=waveform.dtype)


__all__ = [
    "MelSpectrogram",
    "cloning_mel",
    "load_reference_audio",
    "mel_filterbank",
]
