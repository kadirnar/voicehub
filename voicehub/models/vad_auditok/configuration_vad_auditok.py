"""Configuration for Auditok's adaptive energy detector."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from re import fullmatch

from voicehub.configuration_utils import VoiceHubConfig, reject_serialized_secrets


def _finite_number(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise ValueError(f"`{name}` must be a finite number.")
    return float(value)


def _threshold_method(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("`threshold_method` must be a non-empty string.")
    normalized = value.strip().lower()
    if normalized in {"fixed", "otsu", "percentile"}:
        return normalized
    match = fullmatch(r"p([1-9]|[1-9][0-9])", normalized)
    if match is None:
        raise ValueError(
            "`threshold_method` must be 'fixed', 'otsu', 'percentile', "
            "or a percentile selector from 'p1' through 'p99'.")
    return normalized


class AuditokVADConfig(VoiceHubConfig):
    """Configure fixed or automatically calibrated Auditok detection.

    ``energy_threshold_db`` controls fixed-threshold detection. Set
    ``threshold_method`` to ``"otsu"``, ``"percentile"``, or ``"pXX"``
    to calibrate the threshold from the input energy distribution.
    """

    model_type = "vad_auditok"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        energy_threshold_db: float = 50.0,
        threshold_method: str = "fixed",
        analysis_window_s: float = 0.05,
        calibration_duration_s: float = 3.0,
        minimum_energy_threshold_db: float = 40.0,
        strict_min_duration: bool = False,
        drop_trailing_silence: bool = False,
        inference_config=None,
        **kwargs,
    ):
        reject_serialized_secrets(
            {
                "inference_config": inference_config,
                "kwargs": kwargs,
            },
            owner="AuditokVADConfig",
        )
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("`sample_rate` must be a positive integer.")
        energy_threshold_db = _finite_number(
            energy_threshold_db,
            name="energy_threshold_db",
        )
        minimum_energy_threshold_db = _finite_number(
            minimum_energy_threshold_db,
            name="minimum_energy_threshold_db",
        )
        analysis_window_s = _finite_number(
            analysis_window_s,
            name="analysis_window_s",
        )
        if not 0.01 <= analysis_window_s <= 0.1:
            raise ValueError("`analysis_window_s` must be between 0.01 and 0.1 seconds.")
        calibration_duration_s = _finite_number(
            calibration_duration_s,
            name="calibration_duration_s",
        )
        if calibration_duration_s <= 0:
            raise ValueError("`calibration_duration_s` must be greater than zero.")
        if not isinstance(strict_min_duration, bool):
            raise TypeError("`strict_min_duration` must be a boolean.")
        if not isinstance(drop_trailing_silence, bool):
            raise TypeError("`drop_trailing_silence` must be a boolean.")

        super().__init__(
            sample_rate=sample_rate,
            energy_threshold_db=energy_threshold_db,
            threshold_method=_threshold_method(threshold_method),
            analysis_window_s=analysis_window_s,
            calibration_duration_s=calibration_duration_s,
            minimum_energy_threshold_db=minimum_energy_threshold_db,
            strict_min_duration=strict_min_duration,
            drop_trailing_silence=drop_trailing_silence,
            inference_config=inference_config or {},
            **kwargs,
        )
