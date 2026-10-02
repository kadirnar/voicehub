"""BS.1770 loudness normalization used by the released Irodori codec frontend.

The original Irodori-TTS runtime normalizes reference and training audio
with ``audiotools.AudioSignal.normalize(-16)`` followed by
``ensure_max_of_audio()``. This module reproduces that computation with
PyTorch only: the pyloudnorm K-weighting biquads (coefficients rounded to
float32 as audiotools does), 400 ms gating blocks with 75 % overlap whose
final block is zero-padded like ``julius.core.unfold``, zero padding of
signals shorter than 0.5 s, the -70 LUFS absolute and -10 LU relative
gates, and the exponential gain formula.

The only intentional difference is the IIR evaluation: audiotools runs a
sequential float32 ``torchaudio.functional.lfilter``; here the same filters
are applied as a float64 FFT convolution with their impulse response,
truncated far below float32 resolution. Measured loudness agrees to a few
float32 ulps.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

_GAIN_FACTOR = math.log(10.0) / 20.0
_MIN_LOUDNESS = -70.0
_BLOCK_SECONDS = 0.4
_CHANNEL_GAINS = (1.0, 1.0, 1.0, 1.41, 1.41)


def _k_weighting_biquads(sample_rate: int) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    """Return pyloudnorm's K-weighting stages as float32-rounded (b, a)."""
    stages = []
    for kind, gain_db, quality, frequency in (
        ("high_shelf", 4.0, 1.0 / math.sqrt(2.0), 1500.0),
        ("high_pass", 0.0, 0.5, 38.0),
    ):
        amplitude = 10.0**(gain_db / 40.0)
        omega = 2.0 * math.pi * (frequency / sample_rate)
        alpha = math.sin(omega) / (2.0 * quality)
        cosine = math.cos(omega)
        if kind == "high_shelf":
            root = 2.0 * math.sqrt(amplitude) * alpha
            numerator = (
                amplitude * ((amplitude + 1) + (amplitude - 1) * cosine + root),
                -2.0 * amplitude * ((amplitude - 1) + (amplitude + 1) * cosine),
                amplitude * ((amplitude + 1) + (amplitude - 1) * cosine - root),
            )
            denominator = (
                (amplitude + 1) - (amplitude - 1) * cosine + root,
                2.0 * ((amplitude - 1) - (amplitude + 1) * cosine),
                (amplitude + 1) - (amplitude - 1) * cosine - root,
            )
        else:
            numerator = ((1 + cosine) / 2.0, -(1 + cosine), (1 + cosine) / 2.0)
            denominator = (1 + alpha, -2.0 * cosine, 1 - alpha)
        scale = denominator[0]
        b = torch.tensor([value / scale for value in numerator], dtype=torch.float64).float()
        a = torch.tensor([value / scale for value in denominator], dtype=torch.float64).float()
        stages.append((tuple(b.double().tolist()), tuple(a.double().tolist())))
    return tuple(stages)


@lru_cache(maxsize=8)
def _k_weighting_impulse_response(sample_rate: int) -> torch.Tensor:
    """Impulse response of both K-weighting stages, in float64."""
    response = [1.0]
    for b, a in _k_weighting_biquads(sample_rate):
        output: list[float] = []
        previous_1 = previous_2 = 0.0
        index = 0
        # Run until the input is exhausted and the stable tail is negligible.
        while index < len(response) or abs(previous_1) + abs(previous_2) > 1e-30 or index < 64:
            value = b[0] * (response[index] if index < len(response) else 0.0)
            if 1 <= index <= len(response):
                value += b[1] * response[index - 1]
            if 2 <= index <= len(response) + 1:
                value += b[2] * response[index - 2]
            value -= a[1] * previous_1 + a[2] * previous_2
            output.append(value)
            previous_2, previous_1 = previous_1, value
            index += 1
            if index > 1 << 22:  # pragma: no cover - stable filters decay long before this.
                raise RuntimeError("K-weighting impulse response did not decay.")
        response = output
    return torch.tensor(response, dtype=torch.float64)


def _k_weight(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Apply K-weighting along the last axis of float32 audio."""
    response = _k_weighting_impulse_response(int(sample_rate))
    length = audio.shape[-1]
    size = 1 << math.ceil(math.log2(length + response.numel() - 1))
    spectrum = torch.fft.rfft(audio.detach().to(device="cpu", dtype=torch.float64), size)
    filtered = torch.fft.irfft(spectrum * torch.fft.rfft(response, size), size)[..., :length]
    return filtered.to(dtype=torch.float32)


def integrated_loudness(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Gated BS.1770 loudness of ``(batch, channels, time)`` float audio.

    Mirrors ``audiotools.AudioSignal.loudness()`` (including its 0.5 s
    zero padding and -70 LUFS floor) and returns one float32 value per item.
    """
    if audio.ndim != 3:
        raise ValueError("Loudness input must have shape (batch, channels, time).")
    if audio.shape[1] > len(_CHANNEL_GAINS):
        raise ValueError("BS.1770 loudness supports at most five channels.")
    audio = audio.detach().to(device="cpu", dtype=torch.float32)
    minimum_samples = 0.5 * sample_rate
    if audio.shape[-1] < minimum_samples:
        padding = int((0.5 - audio.shape[-1] / sample_rate) * sample_rate)
        audio = torch.nn.functional.pad(audio, (0, padding))
    filtered = _k_weight(audio, sample_rate)

    block = int(_BLOCK_SECONDS * sample_rate)
    stride = int(_BLOCK_SECONDS * sample_rate * 0.25)
    length = filtered.shape[-1]
    frames = math.ceil((max(length, block) - block) / stride) + 1
    filtered = torch.nn.functional.pad(filtered, (0, (frames - 1) * stride + block - length)).contiguous()
    # Same strided view and reduction axis as ``julius.core.unfold(...).transpose(-1, -2)``.
    blocks = filtered.unfold(-1, block, stride).transpose(-1, -2)
    energy = (1.0 / (_BLOCK_SECONDS * sample_rate)) * blocks.square().sum(2)

    gains = torch.tensor(_CHANNEL_GAINS[:energy.shape[1]], dtype=torch.float64)[None, :, None]
    block_loudness = (-0.691 + 10.0 * torch.log10((gains * energy).sum(1, keepdim=True))).expand_as(energy)
    absolute = block_loudness > _MIN_LOUDNESS
    gated = torch.where(absolute, energy, torch.zeros_like(energy))
    relative = -0.691 + 10.0 * torch.log10((gated.sum(2) / absolute.sum(2) * gains[..., 0]).sum(-1)) - 10.0
    selected = absolute & (block_loudness > relative[:, None, None])
    gated = torch.where(selected, energy, torch.zeros_like(energy)).sum(2) / selected.sum(2)
    gated = torch.nan_to_num(
        gated,
        nan=0.0,
        posinf=torch.finfo(torch.float32).max,
        neginf=torch.finfo(torch.float32).min,
    )
    loudness = (-0.691 + 10.0 * torch.log10((gains[..., 0] * gated).sum(1))).float()
    return torch.maximum(loudness, torch.full_like(loudness, _MIN_LOUDNESS))


def normalize_loudness(waveform: torch.Tensor, sample_rate: int, target_db: float) -> torch.Tensor:
    """Normalize one mono float32 waveform like audiotools ``normalize``.

    Equivalent to ``AudioSignal(w).normalize(target_db).ensure_max_of_audio()``.
    """
    if waveform.ndim != 1:
        raise ValueError("Loudness normalization expects a mono waveform.")
    waveform = waveform.to(dtype=torch.float32)
    measured = integrated_loudness(waveform[None, None], sample_rate).to(waveform.device)
    gain = torch.exp((torch.as_tensor(float(target_db), device=waveform.device) - measured) * _GAIN_FACTOR)
    normalized = waveform * gain[0]
    peak = normalized.abs().max()
    if peak > 1.0:
        normalized = normalized * (1.0 / peak)
    return normalized


__all__ = ["integrated_loudness", "normalize_loudness"]
