"""Configuration for VoiceHub's native VITS and MMS-TTS provider."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets

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


def _real(
    value: object,
    *,
    name: str,
    allow_zero: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    normalized = float(value)
    valid = normalized >= 0.0 if allow_zero else normalized > 0.0
    if not math.isfinite(normalized) or not valid:
        qualifier = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"`{name}` must be finite and {qualifier}.")
    return normalized


def _artifact_name(value: object, *, name: str, optional: bool) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        suffix = " or None" if optional else ""
        raise ValueError(f"`{name}` must be a non-empty filename{suffix}.")
    normalized = value.strip()
    path = PurePosixPath(normalized)
    if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
        raise ValueError(f"`{name}` must be one safe checkpoint-root filename.")
    return normalized


class VitsConfig(VoiceHubConfig):
    """Configure native VITS loading, synthesis, and fine-tuning.

    The legacy Transformers loader fields remain accepted so migrations
    fail with precise messages. VoiceHub never delegates model
    construction, checkpoint loading, or tokenization to an external
    architecture runtime.
    """

    model_type = "vits"

    def __init__(
        self,
        *,
        config_name_or_path: str | Path | None = None,
        processor_name_or_path: str | Path | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        use_safetensors: bool | None = None,
        torch_dtype: str = "auto",
        model_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        checkpoint_filename: str | None = None,
        vocabulary_filename: str = "vocab.json",
        tokenizer_config_filename: str = "tokenizer_config.json",
        speaking_rate: float = 1.0,
        noise_scale: float | None = None,
        noise_scale_duration: float | None = None,
        max_output_frames: int = 100_000,
        enable_native_generator_training: bool = False,
        enable_native_adversarial_training: bool = False,
        enable_experimental_reconstruction_training: bool | None = None,
        training_acoustic_config: Mapping[str, Any] | None = None,
        training_waveform_loss_weight: float = 1.0,
        training_spectral_loss_weight: float = 0.1,
        training_duration_loss_weight: float = 1.0,
        training_kl_loss_weight: float = 1.0,
        training_mel_loss_weight: float = 45.0,
        sample_rate: int = 16_000,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(
            {
                "model_kwargs": model_kwargs,
                "processor_kwargs": processor_kwargs,
                **kwargs,
            },
            owner=self.__class__.__name__,
        )
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.config_name_or_path = config_name_or_path
        self.processor_name_or_path = processor_name_or_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.use_safetensors = use_safetensors
        self.torch_dtype = torch_dtype
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.processor_kwargs = ({} if processor_kwargs is None else dict(processor_kwargs))
        self.checkpoint_filename = checkpoint_filename
        self.vocabulary_filename = vocabulary_filename
        self.tokenizer_config_filename = tokenizer_config_filename
        self.speaking_rate = speaking_rate
        self.noise_scale = noise_scale
        self.noise_scale_duration = noise_scale_duration
        self.max_output_frames = max_output_frames
        self.enable_native_generator_training = enable_native_generator_training
        self.enable_native_adversarial_training = (enable_native_adversarial_training)
        legacy_training = (
            False if enable_experimental_reconstruction_training is None else
            enable_experimental_reconstruction_training)
        self.enable_experimental_reconstruction_training = legacy_training
        if legacy_training is True:
            self.enable_native_generator_training = True
        if (training_acoustic_config is not None and not isinstance(training_acoustic_config, Mapping)):
            raise TypeError("`training_acoustic_config` must be a mapping or None.")
        self.training_acoustic_config = (
            None if training_acoustic_config is None else copy.deepcopy(dict(training_acoustic_config)))
        self.training_waveform_loss_weight = training_waveform_loss_weight
        self.training_spectral_loss_weight = training_spectral_loss_weight
        self.training_duration_loss_weight = training_duration_loss_weight
        self.training_kl_loss_weight = training_kl_loss_weight
        self.training_mel_loss_weight = training_mel_loss_weight
        self.validate()

    def validate(self) -> None:
        """Validate without importing PyTorch or model code."""
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if self.config_name_or_path is not None:
            raise ValueError(
                "Native VITS resolves `config.json` from the checkpoint's "
                "coherent artifact root; `config_name_or_path` is unsupported.")
        if self.processor_name_or_path is not None:
            raise ValueError(
                "Native VITS resolves tokenizer assets from the checkpoint "
                "root; `processor_name_or_path` is unsupported.")
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("`trust_remote_code` must be a boolean.")
        if self.trust_remote_code:
            raise ValueError(
                "Native VITS never executes repository code; "
                "`trust_remote_code=True` is unsupported.")
        if self.revision is not None:
            if not isinstance(self.revision, str) or not self.revision.strip():
                raise ValueError("`revision` must be a non-empty string or None.")
            self.revision = self.revision.strip()
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be path-like or None.")
            self.cache_dir = str(Path(self.cache_dir).expanduser())
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if self.use_safetensors is False:
            raise ValueError(
                "Native VITS accepts Safetensors only; "
                "`use_safetensors=False` is unsupported.")
        if self.use_safetensors not in (None, True):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        for name in ("model_kwargs", "processor_kwargs"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"`{name}` must be a mapping or None.")
            value = dict(value)
            if value:
                options = ", ".join(sorted(str(key) for key in value))
                raise ValueError(
                    f"Native VITS does not delegate `{name}`; unsupported "
                    f"option(s): {options}.")
            setattr(self, name, value)
        self.checkpoint_filename = _artifact_name(
            self.checkpoint_filename,
            name="checkpoint_filename",
            optional=True,
        )
        self.vocabulary_filename = _artifact_name(
            self.vocabulary_filename,
            name="vocabulary_filename",
            optional=False,
        )
        self.tokenizer_config_filename = _artifact_name(
            self.tokenizer_config_filename,
            name="tokenizer_config_filename",
            optional=False,
        )
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        try:
            self.torch_dtype = _DTYPE_ALIASES[self.torch_dtype.strip().lower()]
        except KeyError as error:
            choices = ", ".join(sorted(set(_DTYPE_ALIASES.values())))
            raise ValueError(f"`torch_dtype` must be one of: {choices}.") from error
        if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, Integral) or
                self.sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")
        self.sample_rate = int(self.sample_rate)
        self.speaking_rate = _real(
            self.speaking_rate,
            name="speaking_rate",
            allow_zero=False,
        )
        for name in ("noise_scale", "noise_scale_duration"):
            value = getattr(self, name)
            if value is not None:
                setattr(
                    self,
                    name,
                    _real(value, name=name, allow_zero=True),
                )
        if (isinstance(self.max_output_frames, bool) or not isinstance(self.max_output_frames, Integral) or
                self.max_output_frames <= 0):
            raise ValueError("`max_output_frames` must be a positive integer.")
        self.max_output_frames = int(self.max_output_frames)
        for name in (
                "enable_native_generator_training",
                "enable_native_adversarial_training",
                "enable_experimental_reconstruction_training",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.enable_experimental_reconstruction_training:
            self.enable_native_generator_training = True
        if self.enable_native_adversarial_training:
            self.enable_native_generator_training = True
        if (self.training_acoustic_config is not None and
                not isinstance(self.training_acoustic_config, Mapping)):
            raise TypeError("`training_acoustic_config` must be a mapping or None.")
        if self.training_acoustic_config is not None:
            self.training_acoustic_config = copy.deepcopy(dict(self.training_acoustic_config))
        for name in (
                "training_waveform_loss_weight",
                "training_spectral_loss_weight",
                "training_duration_loss_weight",
                "training_kl_loss_weight",
                "training_mel_loss_weight",
        ):
            setattr(
                self,
                name,
                _real(getattr(self, name), name=name, allow_zero=True),
            )

    def to_dict(self) -> dict[str, Any]:
        """Validate mutable overrides before serialization."""
        self.validate()
        return super().to_dict()


__all__ = ["VitsConfig"]
