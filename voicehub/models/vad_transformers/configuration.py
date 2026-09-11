"""Configuration for generic Transformers VAD checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.inference_configuration import VADInferenceConfig


class TransformersVADConfig(VoiceHubConfig):
    """Configure native Wav2Vec2 clip- or frame-classification dispatch."""

    model_type = "vad_transformers"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        architecture_family: str = "auto",
        config_name_or_path: str | Path | None = None,
        processor_name_or_path: str | Path | None = None,
        speech_labels: tuple[str, ...] = (
            "speech",
            "voice",
            "talking",
            "active",
            "1",
        ),
        speech_class_id: int | None = None,
        window_duration_s: float = 1.0,
        hop_duration_s: float = 0.5,
        trust_remote_code: bool = False,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        use_safetensors: bool | None = None,
        checkpoint_filename: str | None = None,
        torch_dtype: str = "auto",
        model_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        inference_config=None,
        **kwargs,
    ):
        reject_serialized_secrets(
            {
                "model_kwargs": model_kwargs,
                "processor_kwargs": processor_kwargs,
                "inference_config": inference_config,
                **kwargs,
            },
            owner=self.__class__.__name__,
        )
        super().__init__(
            sample_rate=sample_rate,
            inference_config=({} if inference_config is None else inference_config),
            architecture_family=architecture_family,
            config_name_or_path=config_name_or_path,
            processor_name_or_path=processor_name_or_path,
            speech_labels=speech_labels,
            speech_class_id=speech_class_id,
            window_duration_s=window_duration_s,
            hop_duration_s=hop_duration_s,
            trust_remote_code=trust_remote_code,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_safetensors=use_safetensors,
            checkpoint_filename=checkpoint_filename,
            torch_dtype=torch_dtype,
            model_kwargs=({} if model_kwargs is None else model_kwargs),
            processor_kwargs=({} if processor_kwargs is None else processor_kwargs),
            **kwargs,
        )
        self.validate()

    @staticmethod
    def _copy_mapping(
        value: Mapping[str, Any] | None,
        *,
        name: str,
    ) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"`{name}` must be a mapping or None.")
        return dict(value)

    def validate(self) -> None:
        """Validate construction values and any later config overrides."""
        reject_serialized_secrets(
            self.__dict__,
            owner=self.__class__.__name__,
        )
        inference_values = getattr(self, "inference_config", {})
        if isinstance(inference_values, VADInferenceConfig):
            inference_values = inference_values.to_dict()
        elif isinstance(inference_values, Mapping):
            inference_values = dict(inference_values)
        else:
            raise TypeError("`inference_config` must be a mapping.")
        for name in VADInferenceConfig._COMMON_FIELDS:
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            inference_values[name] = value
            delattr(self, name)
        self.inference_config = VADInferenceConfig.from_dict(inference_values).to_dict()
        families = ("auto", "audio-classification", "frame-classification")
        if not isinstance(self.architecture_family, str):
            raise TypeError("`architecture_family` must be a string.")
        self.architecture_family = self.architecture_family.strip().lower()
        if self.architecture_family not in families:
            raise ValueError("`architecture_family` must be one of: " + ", ".join(families))
        if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, Integral) or
                self.sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")
        self.sample_rate = int(self.sample_rate)
        for name in ("config_name_or_path", "processor_name_or_path"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"`{name}` must be a non-empty path or Hub ID.")
            setattr(self, name, str(value))

        labels = self.speech_labels
        if (isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence) or not labels or
                any(not isinstance(label, str) or not label.strip() for label in labels)):
            raise ValueError("`speech_labels` must contain non-empty strings.")
        self.speech_labels = tuple(label.strip().lower() for label in labels)
        if self.speech_class_id is not None and (isinstance(self.speech_class_id, bool) or
                                                 not isinstance(self.speech_class_id, Integral) or
                                                 self.speech_class_id < 0):
            raise ValueError("`speech_class_id` must be a non-negative integer or None.")
        if self.speech_class_id is not None:
            self.speech_class_id = int(self.speech_class_id)

        for name in ("window_duration_s", "hop_duration_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or
                    value <= 0):
                raise ValueError(f"`{name}` must be finite and greater than zero.")
            setattr(self, name, float(value))
        if self.hop_duration_s > self.window_duration_s:
            raise ValueError("VAD hop duration cannot exceed the window duration.")
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("`trust_remote_code` must be a boolean.")
        if self.trust_remote_code:
            raise ValueError("Native VoiceHub VAD does not execute remote architecture code.")
        if self.revision is not None:
            if not isinstance(self.revision, str) or not self.revision.strip():
                raise ValueError("`revision` must be a non-empty string or None.")
            self.revision = self.revision.strip()
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be a path-like value or None.")
            self.cache_dir = str(self.cache_dir)
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if self.use_safetensors is not None and not isinstance(
                self.use_safetensors,
                bool,
        ):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        if self.use_safetensors is False:
            raise ValueError("Native VoiceHub VAD requires a Safetensors checkpoint.")
        if self.checkpoint_filename is not None:
            if (not isinstance(self.checkpoint_filename, str) or not self.checkpoint_filename.strip()):
                raise ValueError("`checkpoint_filename` must be a non-empty string or None.")
            self.checkpoint_filename = self.checkpoint_filename.strip()
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        self.torch_dtype = self.torch_dtype.strip().lower()
        if self.torch_dtype not in {
                "auto",
                "bfloat16",
                "float16",
                "float32",
        }:
            raise ValueError("`torch_dtype` must be auto, float32, float16, or bfloat16.")

        self.model_kwargs = self._copy_mapping(
            self.model_kwargs,
            name="model_kwargs",
        )
        self.processor_kwargs = self._copy_mapping(
            self.processor_kwargs,
            name="processor_kwargs",
        )
        reserved_model_options = {
            "config",
            "state_dict",
            "trust_remote_code",
        }
        conflicts = reserved_model_options.intersection(self.model_kwargs)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"`model_kwargs` cannot override provider-owned option(s): {names}.")
        if "trust_remote_code" in self.processor_kwargs:
            raise ValueError("`processor_kwargs` cannot override `trust_remote_code`.")
        if self.config_name_or_path is not None:
            raise ValueError(
                "Native VoiceHub VAD resolves one coherent artifact set; "
                "`config_name_or_path` overrides are unsupported.")
        if self.processor_name_or_path is not None:
            raise ValueError(
                "Native VoiceHub VAD resolves one coherent artifact set; "
                "`processor_name_or_path` overrides are unsupported.")
        if self.model_kwargs:
            names = ", ".join(sorted(self.model_kwargs))
            raise ValueError(
                "Native VoiceHub VAD does not accept external model loader "
                f"options: {names}.")
        if self.processor_kwargs:
            names = ", ".join(sorted(self.processor_kwargs))
            raise ValueError("Native VoiceHub VAD does not accept external processor "
                             f"options: {names}.")

    def to_dict(self) -> dict[str, Any]:
        """Validate mutable overrides before serializing the provider."""
        self.validate()
        return super().to_dict()
