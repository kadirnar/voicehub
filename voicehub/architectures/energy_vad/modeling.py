"""Native short-term-energy activity detection.

The energy definition and automatic threshold estimators follow Auditok
at revision ``833ae725aef73a489366cc5940b831e16223059f``. VoiceHub owns
this implementation and does not import Auditok or NumPy at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

ThresholdMethod = Literal["fixed", "otsu", "percentile"] | str

_ENERGY_EPSILON = 1e-10
_SILENCE_SENTINEL_DB = 20.0 * math.log10(_ENERGY_EPSILON)
_PERCENTILE_MARGIN_DB = 6.0
_OTSU_BINS = 128


@dataclass(frozen=True, slots=True)
class EnergyRegion:
    """One detected region in source-sample coordinates."""

    start_sample: int
    end_sample: int

    def __post_init__(self) -> None:
        if (isinstance(self.start_sample, bool) or not isinstance(self.start_sample, int) or
                self.start_sample < 0):
            raise ValueError("`start_sample` must be a non-negative integer.")
        if (isinstance(self.end_sample, bool) or not isinstance(self.end_sample, int) or
                self.end_sample <= self.start_sample):
            raise ValueError("`end_sample` must be greater than `start_sample`.")


@dataclass(frozen=True, slots=True)
class EnergyDetection:
    """Frame analysis and regions produced by the detector."""

    regions: tuple[EnergyRegion, ...]
    frame_energies_db: Tensor
    threshold_db: float
    window_samples: int

    def __post_init__(self) -> None:
        if (not isinstance(self.frame_energies_db, Tensor) or self.frame_energies_db.ndim != 1):
            raise ValueError("`frame_energies_db` must be a rank-one tensor.")
        if (isinstance(self.window_samples, bool) or not isinstance(self.window_samples, int) or
                self.window_samples <= 0):
            raise ValueError("`window_samples` must be a positive integer.")


def _window_energies(waveform: Tensor, window_samples: int) -> Tensor:
    if not isinstance(waveform, Tensor) or waveform.ndim != 1:
        raise ValueError("`waveform` must be a rank-one PyTorch tensor.")
    if not waveform.is_floating_point():
        raise TypeError("`waveform` must use a floating-point dtype.")
    if waveform.numel() == 0:
        raise ValueError("`waveform` cannot be empty.")
    if (isinstance(window_samples, bool) or not isinstance(window_samples, int) or window_samples <= 0):
        raise ValueError("`window_samples` must be a positive integer.")

    # Auditok computes energies in float64 on the int16 amplitude scale.
    waveform = waveform.double() * 32_768.0
    complete_frames = waveform.numel() // window_samples
    energies: list[Tensor] = []
    if complete_frames:
        frames = waveform[:complete_frames * window_samples].reshape(complete_frames, window_samples)
        energies.append(frames.square().mean(dim=-1).sqrt())
    remainder = waveform[complete_frames * window_samples:]
    if remainder.numel():
        energies.append(remainder.square().mean().sqrt().reshape(1))
    root_mean_square = torch.cat(energies)
    return 20.0 * torch.log10(root_mean_square.clamp_min(_ENERGY_EPSILON))


def _non_silent_energies(energies: Tensor) -> Tensor:
    return energies[energies > _SILENCE_SENTINEL_DB]


def _otsu_threshold(energies: Tensor) -> float:
    minimum = float(energies.min().item())
    maximum = float(energies.max().item())
    if minimum == maximum:
        return minimum
    histogram, edges = torch.histogram(
        energies.double().cpu(),
        bins=_OTSU_BINS,
        range=(minimum, maximum),
    )
    histogram = histogram.double()
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_0 = histogram.cumsum(dim=0)[:-1]
    weight_1 = histogram.sum() - weight_0
    cumulative_mass = (histogram * centers).cumsum(dim=0)[:-1]
    total_mass = (histogram * centers).sum()
    valid = (weight_0 > 0) & (weight_1 > 0)
    between_variance = torch.full_like(weight_0, -1.0)
    mean_0 = cumulative_mass[valid] / weight_0[valid]
    mean_1 = (total_mass - cumulative_mass[valid]) / weight_1[valid]
    between_variance[valid] = (weight_0[valid] * weight_1[valid] * (mean_0 - mean_1).square())
    candidates = torch.nonzero(
        between_variance == between_variance.max(),
        as_tuple=False,
    ).flatten()
    split = int(candidates[(candidates.numel() - 1) // 2].item())
    return float(edges[split + 1].item())


def estimate_energy_threshold(
    energies: Tensor,
    *,
    method: str,
) -> float:
    """Estimate an activity threshold from frame log energies."""
    if not isinstance(energies, Tensor) or energies.ndim != 1:
        raise ValueError("`energies` must be a rank-one PyTorch tensor.")
    if energies.numel() == 0:
        raise ValueError("Cannot estimate a threshold from no energy frames.")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("`method` must be a non-empty string.")
    normalized = method.strip().lower()
    percentile = 10.0
    if normalized.startswith("p") and normalized[1:].isdigit():
        percentile = float(normalized[1:])
    elif normalized not in {"otsu", "percentile"}:
        raise ValueError("Energy threshold method must be 'otsu', 'percentile', or "
                         "'p1' through 'p99'.")
    if not 1.0 <= percentile <= 99.0:
        raise ValueError("Energy percentile must be between 1 and 99.")
    materialized = _non_silent_energies(energies)
    if materialized.numel() == 0:
        return math.inf
    if bool(materialized.min() == materialized.max()):
        # Auditok returns a constant energy itself, before any method
        # (so no percentile margin is added).
        return float(materialized[0].item())
    if normalized == "otsu":
        return _otsu_threshold(materialized)
    quantile = torch.quantile(
        materialized.double(),
        percentile / 100.0,
    )
    return float(quantile.item() + _PERCENTILE_MARGIN_DB)


_DURATION_EPSILON = 1e-10


def _duration_to_windows(duration_s: float, window_s: float, round_fn) -> int:
    # Auditok's `_duration_to_nb_windows`: the same float expression, so
    # e.g. ceil(0.45 / 0.05) == 10 windows exactly as upstream computes it.
    if duration_s == 0:
        return 0
    epsilon = 0.0 if round_fn is math.ceil else _DURATION_EPSILON
    return int(round_fn(duration_s / window_s + epsilon))


def _tokenize(
    decisions: list[bool],
    *,
    min_length: int,
    max_length: int | float,
    max_continuous_silence: int,
    strict_min_length: bool,
    drop_trailing_silence: bool,
) -> list[tuple[int, int]]:
    """Return ``[start, end)`` frame tokens like Auditok's StreamTokenizer.

    This follows ``auditok.core.StreamTokenizer`` for the configuration
    ``split`` uses (no leading silence, ``init_min=0``, and trailing
    silence either kept up to ``max_continuous_silence`` or dropped):
    tokens are truncated at ``max_length`` frames, a remainder shorter
    than ``min_length`` is kept only when it directly continues a
    truncated token (unless ``strict_min_length``), and the minimum length
    counts frames, a partial final window included.
    """
    tokens: list[tuple[int, int]] = []
    silence, noise, possible_silence = 0, 1, 2
    state = silence
    start = 0
    length = 0
    silence_length = 0
    contiguous = False

    def end_detection(current: int, truncated: bool = False) -> None:
        nonlocal start, length, contiguous
        if not truncated and drop_trailing_silence and silence_length > 0:
            length -= silence_length
        if length >= min_length or (length > 0 and not strict_min_length and contiguous):
            tokens.append((start, start + length))
            if truncated:
                start = current + 1
            contiguous = truncated
        else:
            contiguous = False
        length = 0

    for current, valid in enumerate(decisions):
        if state == silence:
            if valid:
                silence_length = 0
                start = current
                length = 1
                state = noise
                if length >= max_length:
                    end_detection(current, truncated=True)
        elif state == noise:
            if valid:
                length += 1
                if length >= max_length:
                    end_detection(current, truncated=True)
            elif max_continuous_silence <= 0:
                state = silence
                end_detection(current)
            else:
                silence_length = 1
                length += 1
                state = possible_silence
                if length == max_length:
                    end_detection(current, truncated=True)
        elif valid:
            length += 1
            silence_length = 0
            state = noise
            if length >= max_length:
                end_detection(current, truncated=True)
        elif silence_length >= max_continuous_silence:
            state = silence
            if silence_length < length:
                end_detection(current)
            else:
                length = 0
                silence_length = 0
        else:
            length += 1
            silence_length += 1
            if length >= max_length:
                end_detection(current, truncated=True)
    if state != silence and length > 0 and length > silence_length:
        end_detection(len(decisions))
    return tokens


def _pad_regions(
    regions: list[tuple[int, int]],
    *,
    padding: int,
    duration_samples: int,
) -> list[EnergyRegion]:
    # VoiceHub's symmetric speech padding; overlapping pads meet halfway,
    # like the shared frame-probability segmenter, so pieces of a token
    # truncated at the maximum duration stay separate.
    padded: list[tuple[int, int]] = []
    for start, end in regions:
        start = max(0, start - padding)
        end = min(duration_samples, end + padding)
        if padded and start < padded[-1][1]:
            midpoint = (padded[-1][1] + start) // 2
            padded[-1] = (padded[-1][0], midpoint)
            start = midpoint
        padded.append((start, end))
    return [EnergyRegion(start, end) for start, end in padded if end > start]


class EnergyVoiceActivityDetector:
    """Detect energetic regions with deterministic VoiceHub tensor code."""

    def detect(
        self,
        waveform: Tensor,
        *,
        sampling_rate: int,
        energy_threshold_db: float,
        threshold_method: str,
        analysis_window_s: float,
        minimum_energy_threshold_db: float | None,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
        max_speech_duration_s: float | None,
        strict_min_duration: bool,
        window_size_samples: int | None = None,
        drop_trailing_silence: bool = True,
    ) -> EnergyDetection:
        if not isinstance(drop_trailing_silence, bool):
            raise TypeError("`drop_trailing_silence` must be a boolean.")
        if (isinstance(sampling_rate, bool) or not isinstance(sampling_rate, int) or sampling_rate <= 0):
            raise ValueError("`sampling_rate` must be a positive integer.")
        if window_size_samples is None:
            # Auditok reads blocks of int(analysis_window * sampling_rate).
            window_samples = int(analysis_window_s * sampling_rate)
            if window_samples <= 0:
                raise ValueError("`analysis_window_s` must cover at least one audio sample.")
        else:
            window_samples = window_size_samples
        energies = _window_energies(waveform, window_samples)
        if threshold_method == "fixed":
            threshold = float(energy_threshold_db)
        else:
            threshold_energies = energies
            if waveform.numel() % window_samples:
                threshold_energies = energies[:-1]
            if threshold_energies.numel() == 0:
                raise ValueError(
                    "Automatic energy calibration requires at least one "
                    "complete analysis window.")
            # Auditok estimates from the whole offline signal without a
            # floor; its `min_energy_threshold` only guards live calibration.
            threshold = estimate_energy_threshold(
                threshold_energies,
                method=threshold_method,
            )
            if minimum_energy_threshold_db is not None:
                threshold = max(float(minimum_energy_threshold_db), threshold)
        window_s = window_samples / sampling_rate
        min_length = max(
            1,
            _duration_to_windows(min_speech_duration_ms / 1_000, window_s, math.ceil),
        )
        max_continuous_silence = _duration_to_windows(
            min_silence_duration_ms / 1_000,
            window_s,
            math.floor,
        )
        max_length = (
            math.inf
            if max_speech_duration_s is None else _duration_to_windows(max_speech_duration_s, window_s, math.floor))
        if min_length > max_length or max_continuous_silence >= max_length:
            raise ValueError(
                "`max_speech_duration_s` must span more analysis windows than "
                "the minimum speech and silence durations.")
        tokens = _tokenize(
            [bool(value) for value in (energies >= threshold).tolist()],
            min_length=min_length,
            max_length=max_length,
            max_continuous_silence=max_continuous_silence,
            strict_min_length=strict_min_duration,
            drop_trailing_silence=drop_trailing_silence,
        )
        duration_samples = waveform.numel()
        regions = _pad_regions(
            [(start * window_samples, min(end * window_samples, duration_samples)) for start, end in tokens],
            padding=round(speech_pad_ms * sampling_rate / 1_000),
            duration_samples=duration_samples,
        )
        return EnergyDetection(
            regions=tuple(regions),
            frame_energies_db=energies,
            threshold_db=threshold,
            window_samples=window_samples,
        )


__all__ = [
    "EnergyDetection",
    "EnergyRegion",
    "EnergyVoiceActivityDetector",
    "estimate_energy_threshold",
]
