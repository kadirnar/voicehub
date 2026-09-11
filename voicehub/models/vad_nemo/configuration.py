"""Configuration for VoiceHub's native multilingual MarbleNet VAD."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.inference_configuration import VADInferenceConfig
from voicehub.models.vad_nemo.native.metadata import MARBLENET_VAD_REVISION


class NeMoVADConfig(VoiceHubConfig):
    """Configure the released Frame-VAD graph without importing NeMo."""

    model_type = "vad_nemo"
    architecture_family = "frame"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        architecture_family: str = "frame",
        speech_class_id: int = 1,
        revision: str | None = MARBLENET_VAD_REVISION,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        model_kwargs: Mapping[str, Any] | None = None,
        training_max_duration_s: float = 30.0,
        training_label_threshold: float = 0.5,
        training_dither: float = 1e-5,
        training_white_noise_probability: float = 0.9,
        training_white_noise_min_db: float = -90.0,
        training_white_noise_max_db: float = -46.0,
        training_gain_probability: float = 0.8,
        training_gain_min_db: float = -20.0,
        training_gain_max_db: float = 10.0,
        training_noise_probability: float = 0.6,
        training_noise_min_snr_db: float = 0.0,
        training_noise_max_snr_db: float = 20.0,
        inference_config: VADInferenceConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(
            {
                "model_kwargs": model_kwargs,
                "inference_config": inference_config,
                **kwargs,
            },
            owner=self.__class__.__name__,
        )
        if isinstance(inference_config, VADInferenceConfig):
            inference_values = inference_config.to_dict()
        elif inference_config is None:
            inference_values = {}
        elif isinstance(inference_config, Mapping):
            inference_values = dict(inference_config)
        else:
            raise TypeError("`inference_config` must be a VADInferenceConfig, mapping, or None.")
        for name in VADInferenceConfig._COMMON_FIELDS:
            if name in kwargs:
                inference_values[name] = kwargs.pop(name)
        if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
            raise TypeError("`model_kwargs` must be a mapping or None.")
        if model_kwargs:
            raise ValueError("`model_kwargs` is not supported by VoiceHub's native "
                             "MarbleNet runtime.")

        super().__init__(
            sample_rate=sample_rate,
            architecture_family=architecture_family,
            speech_class_id=speech_class_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            model_kwargs={},
            training_max_duration_s=training_max_duration_s,
            training_label_threshold=training_label_threshold,
            training_dither=training_dither,
            training_white_noise_probability=training_white_noise_probability,
            training_white_noise_min_db=training_white_noise_min_db,
            training_white_noise_max_db=training_white_noise_max_db,
            training_gain_probability=training_gain_probability,
            training_gain_min_db=training_gain_min_db,
            training_gain_max_db=training_gain_max_db,
            training_noise_probability=training_noise_probability,
            training_noise_min_snr_db=training_noise_min_snr_db,
            training_noise_max_snr_db=training_noise_max_snr_db,
            inference_config=VADInferenceConfig.from_dict(inference_values).to_dict(),
            **kwargs,
        )
        self.validate()

    def validate(self) -> None:
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if self.sample_rate != 16_000:
            raise ValueError("The released multilingual MarbleNet VAD requires 16 kHz.")
        normalized_family = str(self.architecture_family).strip().lower()
        if normalized_family == "auto":
            normalized_family = "frame"
        if normalized_family != "frame":
            raise ValueError(
                "VoiceHub's verified native NeMo provider supports the "
                "multilingual Frame-VAD checkpoint only; window classifiers "
                "use a different graph.")
        self.architecture_family = normalized_family
        if self.speech_class_id != 1:
            raise ValueError("The released checkpoint defines class 1 as speech.")
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if self.revision is not None:
            if not isinstance(self.revision, str) or not self.revision.strip():
                raise ValueError("`revision` must be a non-empty string or None.")
            self.revision = self.revision.strip()
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be path-like or None.")
            self.cache_dir = str(Path(self.cache_dir).expanduser())
        for name in (
                "training_max_duration_s",
                "training_label_threshold",
                "training_dither",
                "training_white_noise_probability",
                "training_white_noise_min_db",
                "training_white_noise_max_db",
                "training_gain_probability",
                "training_gain_min_db",
                "training_gain_max_db",
                "training_noise_probability",
                "training_noise_min_snr_db",
                "training_noise_max_snr_db",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)):
                raise ValueError(f"`{name}` must be a finite real number.")
            setattr(self, name, float(value))
        if self.training_max_duration_s <= 0:
            raise ValueError("`training_max_duration_s` must be positive.")
        if not 0 <= self.training_label_threshold <= 1:
            raise ValueError("`training_label_threshold` must be in [0, 1].")
        if self.training_dither < 0:
            raise ValueError("`training_dither` cannot be negative.")
        for name in (
                "training_white_noise_probability",
                "training_gain_probability",
                "training_noise_probability",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"`{name}` must be in [0, 1].")
        for minimum_name, maximum_name in (
            (
                "training_white_noise_min_db",
                "training_white_noise_max_db",
            ),
            ("training_gain_min_db", "training_gain_max_db"),
            ("training_noise_min_snr_db", "training_noise_max_snr_db"),
        ):
            if getattr(self, minimum_name) > getattr(self, maximum_name):
                raise ValueError(f"`{minimum_name}` cannot exceed `{maximum_name}`.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


__all__ = ["NeMoVADConfig"]
