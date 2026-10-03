"""BS.1770 integrated loudness compatible with ``descript-audiotools``.

The released DAC, Semantic-DACVAE and Irodori-TTS frontends measure and
normalize loudness with ``audiotools.AudioSignal.loudness()`` and
``normalize()``. This module reproduces that computation with PyTorch only:
the pyloudnorm K-weighting biquads (coefficients rounded to float32 as
audiotools does), 400 ms gating blocks with 75 % overlap whose final block is
zero-padded like ``julius.core.unfold``, zero padding of signals shorter than
0.5 s, the audiotools channel gains, the -70 LUFS absolute and -10 LU relative
gates, and the exponential gain formula.

The only intentional difference is the IIR evaluation: audiotools runs a
sequential float32 ``torchaudio.functional.lfilter`` on CPU; here the same
filters are applied on CPU as a float64 FFT convolution with their impulse
response, truncated far below float32 resolution. Measured loudness agrees to
a few float32 ulps. audiotools' optional FIR approximation for CUDA tensors is
not reproduced; the exact IIR response is used on every device.
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
    spectrum = torch.fft.rfft(audio.to(device="cpu", dtype=torch.float64), size)
    filtered = torch.fft.irfft(spectrum * torch.fft.rfft(response, size), size)[..., :length]
    return filtered.to(dtype=torch.float32)


def _as_batched_channels(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 1:
        return audio[None, None]
    if audio.ndim == 2:
        return audio[None]
    if audio.ndim != 3:
        raise ValueError("Audio must have shape (time,), (channels, time), or "
                         f"(batch, channels, time); received {tuple(audio.shape)}.")
    return audio


def integrated_loudness(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Gated BS.1770 loudness like ``audiotools.AudioSignal.loudness()``.

    ``audio`` has shape ``(time,)``, ``(channels, time)`` or ``(batch,
    channels, time)`` with at most five channels. Signals shorter than 0.5 s
    are zero-padded and the result is floored at -70 LUFS, as in audiotools.
    Returns one float32 value per batch item on the CPU. The measurement is
    differentiable with respect to ``audio``.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    audio = _as_batched_channels(torch.as_tensor(audio))
    if audio.shape[1] > len(_CHANNEL_GAINS):
        raise ValueError("BS.1770 loudness supports at most five channels.")
    audio = audio.to(device="cpu", dtype=torch.float32)
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


def normalize_loudness(
    audio: torch.Tensor,
    sample_rate: int,
    target_db: float | torch.Tensor,
    *,
    peak_limit: float = 1.0,
) -> torch.Tensor:
    """Normalize loudness like audiotools ``normalize(target_db)``.

    With the default ``peak_limit`` this is
    ``AudioSignal(audio).normalize(target_db).ensure_max_of_audio()``; pass
    ``float("inf")`` to skip the peak limit. The input shape and floating
    dtype are preserved.
    """
    audio = torch.as_tensor(audio)
    original_shape = audio.shape
    output_dtype = audio.dtype
    normalized = _as_batched_channels(audio).to(dtype=torch.float32)
    measured = integrated_loudness(normalized, sample_rate).to(normalized.device)
    target = torch.as_tensor(target_db, dtype=normalized.dtype, device=normalized.device)
    gain = torch.exp((target - measured) * _GAIN_FACTOR)
    normalized = normalized * gain[:, None, None]
    if math.isfinite(peak_limit):
        peak = normalized.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(
            peak > peak_limit,
            peak_limit / peak.clamp_min(torch.finfo(normalized.dtype).tiny),
            torch.ones_like(peak),
        )
        normalized = normalized * scale
    if output_dtype.is_floating_point:
        normalized = normalized.to(dtype=output_dtype)
    return normalized.reshape(original_shape)


__all__ = ["integrated_loudness", "normalize_loudness"]
