"""Public configuration for the VoiceHub-native FunASR FSMN VAD."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from voicehub.configuration import VoiceHubConfig

_SECRET_OPTIONS = frozenset({
    "api_key",
    "auth_token",
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


class FunASRVADConfig(VoiceHubConfig):
    """Configure a native FSMN artifact without importing FunASR."""

    model_type = "vad_funasr"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        hub: str = "hf",
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        training_max_duration_s: float = 60.0,
        inference_config=None,
        **kwargs,
    ) -> None:
        secret_fields = (_nested_keys(kwargs) | _nested_keys(inference_config)) & _SECRET_OPTIONS
        if secret_fields:
            raise ValueError(
                "Authentication credentials are runtime state and cannot be "
                "stored in FunASRVADConfig.")
        if sample_rate != 16_000:
            raise ValueError("FunASR FSMN VAD requires 16 kHz audio.")
        if hub not in {"hf", "ms"}:
            raise ValueError("`hub` must be 'hf' or 'ms'.")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("`revision` must be a non-empty string or None.")
        if cache_dir is not None and (not isinstance(cache_dir, str) or not cache_dir.strip()):
            raise ValueError("`cache_dir` must be a non-empty string or None.")
        if not isinstance(local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if (isinstance(training_max_duration_s, bool) or not isinstance(training_max_duration_s,
                                                                        (int, float)) or
                not math.isfinite(float(training_max_duration_s)) or float(training_max_duration_s) <= 0):
            raise ValueError("`training_max_duration_s` must be positive.")
        super().__init__(
            sample_rate=sample_rate,
            hub=hub,
            revision=None if revision is None else revision.strip(),
            cache_dir=(None if cache_dir is None else cache_dir.strip()),
            local_files_only=local_files_only,
            training_max_duration_s=float(training_max_duration_s),
            inference_config=inference_config or {},
            **kwargs,
        )


__all__ = ["FunASRVADConfig"]
