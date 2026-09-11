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

    waveform = waveform.float()
    complete_frames = waveform.numel() // window_samples
    energies: list[Tensor] = []
    if complete_frames:
        frames = waveform[:complete_frames * window_samples].reshape(complete_frames, window_samples)
        energies.append(frames.square().mean(dim=-1).sqrt())
    remainder = waveform[complete_frames * window_samples:]
    if remainder.numel():
        energies.append(remainder.square().mean().sqrt().reshape(1))
    root_mean_square = torch.cat(energies)
    pcm_amplitude = root_mean_square * 32_768.0
    return 20.0 * torch.log10(pcm_amplitude.clamp_min(_ENERGY_EPSILON))


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
    materialized = _non_silent_energies(energies)
    if materialized.numel() == 0:
        return math.inf
    if normalized == "otsu":
        return _otsu_threshold(materialized)
    percentile = 10.0
    if normalized.startswith("p") and normalized[1:].isdigit():
        percentile = float(normalized[1:])
    elif normalized != "percentile":
        raise ValueError("Energy threshold method must be 'otsu', 'percentile', or "
                         "'p1' through 'p99'.")
    if not 1.0 <= percentile <= 99.0:
        raise ValueError("Energy percentile must be between 1 and 99.")
    quantile = torch.quantile(
        materialized.double(),
        percentile / 100.0,
    )
    return float(quantile.item() + _PERCENTILE_MARGIN_DB)


def _speech_runs(decisions: Tensor) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(decisions.tolist()):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, decisions.numel()))
    return runs


def _join_runs(
    runs: list[tuple[int, int]],
    *,
    maximum_gap_frames: int,
) -> list[tuple[int, int]]:
    joined: list[tuple[int, int]] = []
    for start, end in runs:
        if joined and start - joined[-1][1] <= maximum_gap_frames:
            joined[-1] = (joined[-1][0], end)
        else:
            joined.append((start, end))
    return joined


def _split_region(
    start: int,
    end: int,
    *,
    maximum_samples: int | None,
    minimum_samples: int,
    strict_minimum: bool,
) -> list[EnergyRegion]:
    if maximum_samples is None:
        return [EnergyRegion(start, end)]
    regions: list[EnergyRegion] = []
    cursor = start
    while end - cursor > maximum_samples:
        regions.append(EnergyRegion(cursor, cursor + maximum_samples))
        cursor += maximum_samples
    remainder = end - cursor
    if remainder and (not strict_minimum or remainder >= minimum_samples):
        regions.append(EnergyRegion(cursor, end))
    return regions


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
        minimum_energy_threshold_db: float,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
        max_speech_duration_s: float | None,
        strict_min_duration: bool,
        window_size_samples: int | None = None,
    ) -> EnergyDetection:
        if (isinstance(sampling_rate, bool) or not isinstance(sampling_rate, int) or sampling_rate <= 0):
            raise ValueError("`sampling_rate` must be a positive integer.")
        if window_size_samples is None:
            window_samples = max(1, round(analysis_window_s * sampling_rate))
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
            threshold = max(
                float(minimum_energy_threshold_db),
                estimate_energy_threshold(
                    threshold_energies,
                    method=threshold_method,
                ),
            )
        decisions = energies >= threshold
        maximum_gap_frames = max(
            0,
            math.floor(min_silence_duration_ms * sampling_rate / 1_000 / window_samples),
        )
        runs = _join_runs(
            _speech_runs(decisions),
            maximum_gap_frames=maximum_gap_frames,
        )
        duration_samples = waveform.numel()
        minimum_samples = max(
            window_samples,
            round(min_speech_duration_ms * sampling_rate / 1_000),
        )
        padding = round(speech_pad_ms * sampling_rate / 1_000)
        maximum_samples = (
            None if max_speech_duration_s is None else round(max_speech_duration_s * sampling_rate))
        if maximum_samples is not None and maximum_samples < minimum_samples:
            raise ValueError(
                "`max_speech_duration_s` cannot be shorter than the "
                "effective minimum speech duration.")
        regions: list[EnergyRegion] = []
        for start_frame, end_frame in runs:
            raw_start = start_frame * window_samples
            raw_end = min(end_frame * window_samples, duration_samples)
            if raw_end - raw_start < minimum_samples:
                continue
            start = max(0, raw_start - padding)
            end = min(duration_samples, raw_end + padding)
            regions.extend(
                _split_region(
                    start,
                    end,
                    maximum_samples=maximum_samples,
                    minimum_samples=minimum_samples,
                    strict_minimum=strict_min_duration,
                ))
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
