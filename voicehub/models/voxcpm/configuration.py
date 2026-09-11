"""Configuration for VoiceHub's dependency-free VoxCPM2 provider."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "float16": "float16",
    "float32": "float32",
    "fp16": "float16",
    "fp32": "float32",
    "half": "float16",
}
_LORA_FIELDS = frozenset({
    "alpha",
    "dropout",
    "enable_dit",
    "enable_lm",
    "enable_proj",
    "r",
    "rank",
    "target_modules_dit",
    "target_modules_lm",
    "target_proj_modules",
})


def _optional_text(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a non-empty string or None.")
    return value.strip()


def _optional_path(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"`{name}` must be a non-empty path or None.")
    return str(Path(value).expanduser())


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return int(value)


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return normalized


class VoxCPMConfig(VoiceHubConfig):
    """Configure native VoxCPM2 loading, generation, and fine-tuning.

    Credentials are constructor-only runtime values on the model
    wrapper. They are intentionally absent from this serializable
    configuration.
    """

    model_type = "voxcpm"

    def __init__(
        self,
        *,
        revision: str | None = None,
        codec_path: str | Path | None = None,
        lora_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        torch_dtype: str = "bfloat16",
        use_safetensors: bool | None = None,
        trust_remote_code: bool = False,
        trust_legacy_codec: bool = False,
        verify_integrity: bool = True,
        verify_checkpoint_integrity: bool = False,
        training_lora_config: Mapping[str, Any] | None = None,
        training_diffusion_loss_weight: float = 1.0,
        training_stop_loss_weight: float = 1.0,
        generation_config: Mapping[str, Any] | None = None,
        sample_rate: int = 48_000,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(kwargs, owner=self.__class__.__name__)
        generation_defaults = {
            "cfg_value": 2.0,
            "inference_timesteps": 10,
            "max_len": 2_000,
            "min_len": 2,
        }
        if generation_config is not None:
            if not isinstance(generation_config, Mapping):
                raise TypeError("`generation_config` must be a mapping or None.")
            generation_defaults.update(dict(generation_config))
        super().__init__(
            sample_rate=sample_rate,
            generation_config=generation_defaults,
            **kwargs,
        )
        self.revision = revision
        self.codec_path = codec_path
        self.lora_path = lora_path
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.torch_dtype = torch_dtype
        self.use_safetensors = use_safetensors
        self.trust_remote_code = trust_remote_code
        self.trust_legacy_codec = trust_legacy_codec
        self.verify_integrity = verify_integrity
        self.verify_checkpoint_integrity = verify_checkpoint_integrity
        self.training_lora_config = (None if training_lora_config is None else dict(training_lora_config))
        self.training_diffusion_loss_weight = training_diffusion_loss_weight
        self.training_stop_loss_weight = training_stop_loss_weight
        self.validate()

    def validate(self) -> None:
        """Validate provider options without importing PyTorch."""
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if self.sample_rate != 48_000:
            raise ValueError("VoxCPM2 AudioVAE produces audio at 48,000 Hz.")
        self.revision = _optional_text(self.revision, name="revision")
        self.codec_path = _optional_path(self.codec_path, name="codec_path")
        self.lora_path = _optional_path(self.lora_path, name="lora_path")
        self.cache_dir = _optional_path(self.cache_dir, name="cache_dir")
        for name in (
                "local_files_only",
                "trust_remote_code",
                "trust_legacy_codec",
                "verify_integrity",
                "verify_checkpoint_integrity",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.trust_remote_code:
            raise ValueError(
                "Native VoxCPM never executes repository code; "
                "`trust_remote_code=True` is unsupported.")
        if self.use_safetensors is False:
            raise ValueError(
                "Native VoxCPM model and steady-state codec checkpoints "
                "must use Safetensors.")
        if self.use_safetensors not in (None, True):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        normalized_dtype = _DTYPE_ALIASES.get(self.torch_dtype.strip().lower().removeprefix("torch."), )
        if normalized_dtype is None:
            supported = ", ".join(sorted(set(_DTYPE_ALIASES.values())))
            raise ValueError(
                f"Unsupported VoxCPM dtype {self.torch_dtype!r}; "
                f"expected one of: {supported}.")
        self.torch_dtype = normalized_dtype
        if (self.codec_path is not None and
                Path(self.codec_path).suffix.lower() in {".pt", ".pth", ".ckpt"} and
                not self.trust_legacy_codec):
            raise PermissionError(
                "A legacy VoxCPM AudioVAE path requires "
                "`trust_legacy_codec=True` for its one-time reviewed conversion.")
        if self.training_lora_config is not None:
            if not isinstance(self.training_lora_config, Mapping):
                raise TypeError("`training_lora_config` must be a mapping or None.")
            unknown = sorted(set(self.training_lora_config) - _LORA_FIELDS)
            if unknown:
                raise ValueError("Unsupported VoxCPM LoRA option(s): " + ", ".join(unknown))
            self.training_lora_config = dict(self.training_lora_config)
        self.training_diffusion_loss_weight = _finite_nonnegative(
            self.training_diffusion_loss_weight,
            name="training_diffusion_loss_weight",
        )
        self.training_stop_loss_weight = _finite_nonnegative(
            self.training_stop_loss_weight,
            name="training_stop_loss_weight",
        )
        if (self.training_diffusion_loss_weight + self.training_stop_loss_weight <= 0):
            raise ValueError("At least one VoxCPM training loss must be active.")
        generation = self.generation_config
        if not isinstance(generation, Mapping):
            raise TypeError("`generation_config` must be a mapping.")
        for name in ("min_len", "max_len", "inference_timesteps"):
            if name in generation:
                generation[name] = _positive_integer(
                    generation[name],
                    name=f"generation_config.{name}",
                )
        if ("min_len" in generation and "max_len" in generation and
                generation["min_len"] >= generation["max_len"]):
            raise ValueError("VoxCPM generation `min_len` must be below `max_len`.")
        if "cfg_value" in generation:
            generation["cfg_value"] = _finite_nonnegative(
                generation["cfg_value"],
                name="generation_config.cfg_value",
            )
        self.generation_config = dict(generation)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


__all__ = ["VoxCPMConfig"]
