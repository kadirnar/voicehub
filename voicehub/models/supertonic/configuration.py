"""Public configuration for VoiceHub-native Supertonic 3."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.models.supertonic.native.metadata import SUPERTONIC_LANGUAGES

SUPERTONIC_SAMPLE_RATE = 44_100
_DTYPE_ALIASES = {
    "auto": "auto",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "float16": "float16",
    "float32": "float32",
    "fp16": "float16",
    "fp32": "float32",
    "half": "float16",
}


def _non_negative_float(value: Any, *, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or
            float(value) < 0):
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return float(value)


class SupertonicConfig(VoiceHubConfig):
    """Configuration for exact inference and reconstructed fine-tuning.

    Supertonic's released graph is fixed at 44.1 kHz. Fine-tuning is
    deliberately opt-in because the authors publish only inference
    graphs, not their raw-audio encoders or original training recipe.
    """

    model_type = "supertonic"

    def __init__(
        self,
        *,
        voice: str = "M1",
        language: str = "en",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        verify_integrity: bool = True,
        torch_dtype: str = "auto",
        enable_preprocessed_training: bool = False,
        training_duration_loss_weight: float = 1.0,
        training_flow_loss_weight: float = 1.0,
        training_vocoder_loss_weight: float = 1.0,
        sample_rate: int = SUPERTONIC_SAMPLE_RATE,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(kwargs, owner=self.__class__.__name__)
        super().__init__(sample_rate=SUPERTONIC_SAMPLE_RATE, **kwargs)
        self.voice = voice
        self.language = language
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.verify_integrity = verify_integrity
        self.torch_dtype = torch_dtype
        self.enable_preprocessed_training = enable_preprocessed_training
        self.training_duration_loss_weight = training_duration_loss_weight
        self.training_flow_loss_weight = training_flow_loss_weight
        self.training_vocoder_loss_weight = training_vocoder_loss_weight
        self.validate()

    def validate(self) -> None:
        """Reject unsafe or unsupported values before model allocation."""
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if not isinstance(self.voice, str) or not self.voice.strip():
            raise ValueError("`voice` must be a non-empty voice ID or style JSON path.")
        self.voice = self.voice.strip()
        if not isinstance(self.language, str):
            raise TypeError("`language` must be a string.")
        self.language = self.language.strip().lower()
        if self.language not in SUPERTONIC_LANGUAGES:
            supported = ", ".join(sorted(SUPERTONIC_LANGUAGES))
            raise ValueError(
                f"Unsupported Supertonic language {self.language!r}. "
                f"Supported: {supported}.")
        if self.revision is not None:
            if not isinstance(self.revision, str) or not self.revision.strip():
                raise ValueError("`revision` must be non-empty or None.")
            self.revision = self.revision.strip()
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be path-like or None.")
            self.cache_dir = str(Path(self.cache_dir).expanduser())
        for name in (
                "local_files_only",
                "verify_integrity",
                "enable_preprocessed_training",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        try:
            self.torch_dtype = _DTYPE_ALIASES[self.torch_dtype.strip().lower()]
        except KeyError as error:
            choices = ", ".join(sorted(set(_DTYPE_ALIASES.values())))
            raise ValueError(f"`torch_dtype` must be one of: {choices}.") from error
        for name in (
                "training_duration_loss_weight",
                "training_flow_loss_weight",
                "training_vocoder_loss_weight",
        ):
            setattr(
                self,
                name,
                _non_negative_float(getattr(self, name), name=name),
            )
        self.sample_rate = SUPERTONIC_SAMPLE_RATE

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()

    @classmethod
    def from_dict(
        cls,
        config_dict: Mapping[str, Any],
        **kwargs: Any,
    ) -> SupertonicConfig:
        values = dict(config_dict)
        values.pop("model_type", None)
        values.update(kwargs)
        return cls(**values)


__all__ = [
    "SUPERTONIC_SAMPLE_RATE",
    "SupertonicConfig",
]
