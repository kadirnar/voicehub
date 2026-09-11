"""Public configuration for VoiceHub-native SpeechBrain CRDNN VAD."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig

_SECRET_OPTIONS = frozenset({
    "access_token",
    "api_key",
    "auth_token",
    "fetch_config",
    "hf_token",
    "token",
    "use_auth_token",
})


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for name, nested in value.items():
            keys.add(str(name).strip().lower())
            keys.update(_nested_keys(nested))
    elif isinstance(value, (tuple, list)):
        for nested in value:
            keys.update(_nested_keys(nested))
    return keys


class SpeechBrainVADConfig(VoiceHubConfig):
    """Configure safe CRDNN artifacts and source-compatible segmentation."""

    model_type = "vad_speechbrain"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        large_chunk_size: float = 30.0,
        small_chunk_size: float = 10.0,
        overlap_small_chunk: bool = False,
        apply_energy_vad: bool = False,
        double_check: bool = True,
        deactivation_threshold: float = 0.25,
        training_max_duration_s: float = 60.0,
        training_positive_weight: float | None = None,
        # Accepted only as inert compatibility fields.  Native runtime never
        # executes HyperPyYAML or forwards provider loader arguments.
        hparams_file: str = "hyperparams.yaml",
        savedir: str | Path | None = None,
        overrides: Mapping | None = None,
        loader_kwargs: Mapping | None = None,
        inference_config=None,
        **kwargs: Any,
    ) -> None:
        secret_fields = (
            _nested_keys(kwargs)
            | _nested_keys(inference_config)
            | _nested_keys(loader_kwargs)
            | _nested_keys(overrides)) & _SECRET_OPTIONS
        if secret_fields:
            raise ValueError(
                "Authentication tokens are runtime-only values. Pass `token` "
                "to SpeechBrainVADForVoiceActivityDetection.")
        if sample_rate != 16_000:
            raise ValueError("SpeechBrain CRDNN VAD requires 16 kHz audio.")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("`revision` must be a non-empty string or None.")
        if cache_dir is not None and not isinstance(cache_dir, (str, Path)):
            raise TypeError("`cache_dir` must be a string, Path, or None.")
        if not isinstance(local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        for name, value in (
            ("large_chunk_size", large_chunk_size),
            ("small_chunk_size", small_chunk_size),
            ("training_max_duration_s", training_max_duration_s),
        ):
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or float(value) <= 0.0):
                raise ValueError(f"`{name}` must be finite and positive.")
        ratio = float(large_chunk_size) / float(small_chunk_size)
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("`large_chunk_size / small_chunk_size` must be an integer.")
        for name, value in (
            ("overlap_small_chunk", overlap_small_chunk),
            ("apply_energy_vad", apply_energy_vad),
            ("double_check", double_check),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if (isinstance(deactivation_threshold, bool) or not isinstance(deactivation_threshold,
                                                                       (int, float)) or
                not 0.0 <= float(deactivation_threshold) <= 1.0):
            raise ValueError("`deactivation_threshold` must be in [0, 1].")
        if training_positive_weight is not None and (isinstance(training_positive_weight, bool) or
                                                     not isinstance(training_positive_weight, (int, float)) or
                                                     not math.isfinite(float(training_positive_weight)) or
                                                     float(training_positive_weight) <= 0.0):
            raise ValueError("`training_positive_weight` must be positive or None.")
        if hparams_file != "hyperparams.yaml":
            raise ValueError(
                "Native SpeechBrain VAD does not execute arbitrary HyperPyYAML; "
                "`hparams_file` must remain 'hyperparams.yaml'.")
        if savedir is not None and not isinstance(savedir, (str, Path)):
            raise TypeError("`savedir` must be a string, Path, or None.")
        if overrides:
            raise ValueError("Native SpeechBrain VAD does not execute HyperPyYAML overrides.")
        if loader_kwargs:
            raise ValueError("Native SpeechBrain VAD does not forward SpeechBrain loader options.")
        super().__init__(
            sample_rate=sample_rate,
            revision=None if revision is None else revision.strip(),
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            large_chunk_size=float(large_chunk_size),
            small_chunk_size=float(small_chunk_size),
            overlap_small_chunk=overlap_small_chunk,
            apply_energy_vad=apply_energy_vad,
            double_check=double_check,
            deactivation_threshold=float(deactivation_threshold),
            training_max_duration_s=float(training_max_duration_s),
            training_positive_weight=(
                None if training_positive_weight is None else float(training_positive_weight)),
            hparams_file=hparams_file,
            savedir=savedir,
            overrides={},
            loader_kwargs={},
            inference_config=inference_config or {},
            **kwargs,
        )


__all__ = ["SpeechBrainVADConfig"]
