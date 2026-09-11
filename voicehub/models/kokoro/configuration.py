"""Serializable Kokoro configuration, independent of the model runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets

KOKORO_SAMPLE_RATE = 24_000
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


def _finite_non_negative(value: Any, *, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or
            float(value) < 0):
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return float(value)


class KokoroConfig(VoiceHubConfig):
    """Configuration for native Kokoro artifacts and reconstructed training."""

    model_type = "kokoro"

    def __init__(
        self,
        *,
        language_code: str = "a",
        lang_code: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        checkpoint_filename: str | None = None,
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        use_safetensors: bool | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        allow_legacy_checkpoint_conversion: bool = False,
        enable_preprocessed_training: bool = False,
        training_duration_loss_weight: float = 1.0,
        training_f0_loss_weight: float = 1.0,
        training_energy_loss_weight: float = 1.0,
        training_waveform_loss_weight: float = 1.0,
        training_spectral_loss_weight: float = 0.1,
        sample_rate: int = KOKORO_SAMPLE_RATE,
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
        # The released decoder and voice packs are fixed at 24 kHz.
        super().__init__(sample_rate=KOKORO_SAMPLE_RATE, **kwargs)
        self.language_code = (language_code if lang_code is None else lang_code)
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.checkpoint_filename = checkpoint_filename
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.use_safetensors = use_safetensors
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.processor_kwargs = ({} if processor_kwargs is None else dict(processor_kwargs))
        self.allow_legacy_checkpoint_conversion = (allow_legacy_checkpoint_conversion)
        self.enable_preprocessed_training = enable_preprocessed_training
        self.training_duration_loss_weight = training_duration_loss_weight
        self.training_f0_loss_weight = training_f0_loss_weight
        self.training_energy_loss_weight = training_energy_loss_weight
        self.training_waveform_loss_weight = training_waveform_loss_weight
        self.training_spectral_loss_weight = training_spectral_loss_weight
        self.validate()

    def validate(self) -> None:
        """Validate without importing model or provider frameworks."""
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if (not isinstance(self.language_code, str) or not self.language_code.strip()):
            raise ValueError("`language_code` must be a non-empty string.")
        self.language_code = self.language_code.strip().lower()
        if self.revision is not None:
            if (not isinstance(self.revision, str) or not self.revision.strip()):
                raise ValueError("`revision` must be non-empty or None.")
            self.revision = self.revision.strip()
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be path-like or None.")
            self.cache_dir = str(Path(self.cache_dir).expanduser())
        for name in (
                "local_files_only",
                "trust_remote_code",
                "allow_legacy_checkpoint_conversion",
                "enable_preprocessed_training",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.trust_remote_code:
            raise ValueError(
                "Native Kokoro never executes repository code; "
                "`trust_remote_code=True` is unsupported.")
        if self.use_safetensors is False:
            raise ValueError(
                "Native Kokoro uses Safetensors for steady-state runtime. "
                "The released .pth is accepted only by the restricted "
                "one-time converter.")
        if self.use_safetensors not in (None, True):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        for name in ("model_kwargs", "processor_kwargs"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"`{name}` must be a mapping.")
            if value:
                options = ", ".join(sorted(str(key) for key in value))
                raise ValueError(
                    f"Native Kokoro does not delegate `{name}`; unsupported "
                    f"option(s): {options}.")
        if self.checkpoint_filename is not None:
            if (not isinstance(self.checkpoint_filename, str) or
                    Path(self.checkpoint_filename).name != self.checkpoint_filename):
                raise ValueError("`checkpoint_filename` must be one root filename.")
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        try:
            self.torch_dtype = _DTYPE_ALIASES[self.torch_dtype.strip().lower()]
        except KeyError as error:
            choices = ", ".join(sorted(set(_DTYPE_ALIASES.values())))
            raise ValueError(f"`torch_dtype` must be one of: {choices}.") from error
        for name in (
                "training_duration_loss_weight",
                "training_f0_loss_weight",
                "training_energy_loss_weight",
                "training_waveform_loss_weight",
                "training_spectral_loss_weight",
        ):
            setattr(
                self,
                name,
                _finite_non_negative(getattr(self, name), name=name),
            )
        self.sample_rate = KOKORO_SAMPLE_RATE

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


__all__ = ["KokoroConfig"]
