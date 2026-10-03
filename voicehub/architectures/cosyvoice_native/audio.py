"""Source-exact CosyVoice 3 prompt-audio frontend (PyTorch only).

The source frontend loads prompt audio with ``torchaudio.load``, resamples
with ``torchaudio.transforms.Resample`` (reproduced by
:func:`voicehub.processing.resample_waveform_hann` with
``match="transform"``), extracts speech tokens at 16 kHz and the flow's
prompt mel at 24 kHz with Matcha-TTS ``mel_spectrogram`` (n_fft
1920, hop 480, 80 Slaney mels, ``center=False`` with reflect padding,
natural-log compression). Both steps run on CPU in float32 like the
source; only the results move to the model device.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor
from torch.nn import functional

PROMPT_MEL_SAMPLE_RATE = 24_000
PROMPT_MEL_N_FFT = 1_920
PROMPT_MEL_HOP_LENGTH = 480
PROMPT_MEL_WIN_LENGTH = 1_920
PROMPT_MEL_BINS = 80


def _slaney_hz_to_mel(frequencies: Tensor) -> Tensor:
    spacing = 200.0 / 3
    minimum_log_mel = 1_000.0 / spacing
    log_step = math.log(6.4) / 27.0
    mels = frequencies / spacing
    logarithmic = minimum_log_mel + torch.log(frequencies / 1_000.0) / log_step
    return torch.where(frequencies >= 1_000.0, logarithmic, mels)


def _slaney_mel_to_hz(mels: Tensor) -> Tensor:
    spacing = 200.0 / 3
    minimum_log_mel = 1_000.0 / spacing
    log_step = math.log(6.4) / 27.0
    frequencies = spacing * mels
    logarithmic = 1_000.0 * torch.exp(log_step * (mels - minimum_log_mel))
    return torch.where(mels >= minimum_log_mel, logarithmic, frequencies)


@lru_cache(maxsize=2)
def _prompt_mel_filters() -> Tensor:
    """``librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80)`` step for step.

    librosa evaluates in float64 but stores the triangles in a float32 array
    *before* applying the Slaney area normalization, so the bank is rounded
    twice; NumPy's ``linspace``/``rfftfreq`` arithmetic is reproduced too.
    """
    sample_rate, n_fft, n_mels = PROMPT_MEL_SAMPLE_RATE, PROMPT_MEL_N_FFT, PROMPT_MEL_BINS
    fft_frequencies = torch.arange(
        n_fft // 2 + 1, dtype=torch.float64) * (1.0 / (n_fft * (1.0 / sample_rate)))
    minimum = _slaney_hz_to_mel(torch.tensor(0.0, dtype=torch.float64))
    maximum = _slaney_hz_to_mel(torch.tensor(sample_rate / 2.0, dtype=torch.float64))
    count = n_mels + 2
    mel_points = torch.arange(count, dtype=torch.float64) * ((maximum - minimum) / (count - 1)) + minimum
    mel_points[-1] = maximum
    edges = _slaney_mel_to_hz(mel_points)
    widths = edges[1:] - edges[:-1]
    ramps = edges[:, None] - fft_frequencies[None, :]
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    triangles = torch.clamp(torch.minimum(lower, upper), min=0.0).to(torch.float32)
    normalization = 2.0 / (edges[2:n_mels + 2] - edges[:n_mels])
    return (triangles.to(torch.float64) * normalization[:, None]).to(torch.float32)


def prompt_mel_features(waveform: Tensor) -> Tensor:
    """Return ``[frames, 80]`` log-mel prompt features from 24 kHz audio."""
    if not isinstance(waveform, Tensor) or waveform.ndim != 1 or not waveform.is_floating_point():
        raise ValueError("`waveform` must be a rank-one floating-point tensor.")
    padding = (PROMPT_MEL_N_FFT - PROMPT_MEL_HOP_LENGTH) // 2
    values = functional.pad(
        waveform.float()[None, None],
        (padding, padding),
        mode="reflect",
    )[:, 0]
    window = torch.hann_window(PROMPT_MEL_WIN_LENGTH, device=values.device)
    spectrum = torch.view_as_real(
        torch.stft(
            values,
            PROMPT_MEL_N_FFT,
            hop_length=PROMPT_MEL_HOP_LENGTH,
            win_length=PROMPT_MEL_WIN_LENGTH,
            window=window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ))
    magnitude = torch.sqrt(spectrum.pow(2).sum(-1) + 1e-9)
    mel = torch.matmul(_prompt_mel_filters().to(magnitude.device), magnitude)
    return torch.log(torch.clamp(mel, min=1e-5))[0].transpose(0, 1).contiguous()


__all__ = [
    "PROMPT_MEL_SAMPLE_RATE",
    "prompt_mel_features",
]
