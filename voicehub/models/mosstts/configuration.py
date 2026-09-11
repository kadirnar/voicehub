"""Serializable configuration for the mosstts model family."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig

_VARIANT_ALIASES = {"local_v15": "local_v1_5"}


class MossTTSConfig(VoiceHubConfig):
    """Serializable controls for native MOSS-TTS loading and training."""

    model_type = "mosstts"

    def __init__(
        self,
        *,
        variant: str = "auto",
        codec_name_or_path: str | None = None,
        compute_dtype: str = "bfloat16",
        torch_dtype: str | None = None,
        attention_implementation: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        training_channelwise_loss_weights: (tuple[float, ...] | list[float] | str) = (1.0, 32.0),
        training_adam_beta1: float = 0.9,
        training_adam_beta2: float = 0.95,
        training_adam_epsilon: float = 1e-4,
        sample_rate: int = 24_000,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        generation_defaults = {
            "max_new_tokens": 4_096,
        }
        generation_defaults.update(dict(generation_config or {}))
        super().__init__(
            sample_rate=sample_rate,
            generation_config=generation_defaults,
            **kwargs,
        )
        self.variant = variant
        self.codec_name_or_path = codec_name_or_path
        self.compute_dtype = (compute_dtype if torch_dtype is None else torch_dtype)
        # Retain the compatibility spelling in serialized wrapper configs.
        self.torch_dtype = self.compute_dtype
        self.attention_implementation = attention_implementation
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.training_channelwise_loss_weights = (training_channelwise_loss_weights)
        self.training_adam_beta1 = training_adam_beta1
        self.training_adam_beta2 = training_adam_beta2
        self.training_adam_epsilon = training_adam_epsilon
        self.validate()

    @staticmethod
    def normalize_variant(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("MOSS-TTS `variant` must be a non-empty string.")
        normalized = (value.strip().lower().replace("-", "_").replace(".", "_"))
        return _VARIANT_ALIASES.get(normalized, normalized)

    def validate(self) -> None:
        # Keep unsupported release names serializable so tooling can inspect
        # a configuration without importing or allocating a runtime.  The
        # wrapper rejects them at its dependency-free validation boundary.
        self.normalize_variant(self.variant)
        if (self.codec_name_or_path is not None and
            (not isinstance(self.codec_name_or_path,
                            (str, Path)) or not str(self.codec_name_or_path).strip())):
            raise ValueError("`codec_name_or_path` must be a non-empty identifier or "
                             "path when supplied.")
        if (not isinstance(self.compute_dtype, str) or not self.compute_dtype.strip()):
            raise ValueError("`compute_dtype` must be a non-empty string.")
        if self.attention_implementation is not None:
            raise ValueError(
                "MOSS-TTS no longer accepts a provider-specific "
                "`attention_implementation`. Select a reversible VoiceHub "
                "InferenceStrategy instead.")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("`revision` must be non-empty or None.")
        if self.cache_dir is not None and (not isinstance(self.cache_dir,
                                                          (str, Path)) or not str(self.cache_dir).strip()):
            raise ValueError("`cache_dir` must be a non-empty path or None.")
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int) or
                self.sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")
        for name in (
                "training_adam_beta1",
                "training_adam_beta2",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not 0.0 <= float(value) < 1.0):
                raise ValueError(f"`{name}` must be in [0, 1).")
        epsilon = self.training_adam_epsilon
        if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or
                not math.isfinite(float(epsilon)) or float(epsilon) <= 0):
            raise ValueError("`training_adam_epsilon` must be finite and positive.")


__all__ = ['MossTTSConfig']
