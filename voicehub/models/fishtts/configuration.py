"""Configuration for VoiceHub-native Fish Speech S2."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets

_DTYPES = {
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


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return result


class FishTTSConfig(VoiceHubConfig):
    """Serializable native semantic/codec and generation controls."""

    model_type = "fishtts"

    def __init__(
        self,
        *,
        revision: str | None = None,
        codec_name_or_path: str | Path | None = None,
        codec_revision: str | None = None,
        codec_conversion_directory: str | Path | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        torch_dtype: str = "auto",
        use_safetensors: bool | None = None,
        trust_remote_code: bool = False,
        verify_official_integrity: bool = True,
        verify_full_shard_hashes: bool = False,
        compile: bool = False,
        seed: int | None = None,
        max_new_tokens: int = 1_024,
        chunk_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 30,
        training_max_length: int | None = None,
        training_base_loss_weight: float = 1.0,
        training_semantic_loss_weight: float = 1.0,
        training_adam_beta1: float = 0.9,
        training_adam_beta2: float = 0.95,
        training_adam_epsilon: float = 1e-5,
        training_warmup_steps: int | None = None,
        training_lora_config: Mapping[str, Any] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        generation_config: Mapping[str, Any] | None = None,
        sample_rate: int = 44_100,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(
            {
                "generation_config": generation_config,
                "model_kwargs": model_kwargs,
                "tokenizer_kwargs": tokenizer_kwargs,
                **kwargs,
            },
            owner=self.__class__.__name__,
        )
        defaults = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
        if generation_config is not None:
            if not isinstance(generation_config, Mapping):
                raise TypeError("`generation_config` must be a mapping.")
            defaults.update(dict(generation_config))
        super().__init__(
            sample_rate=sample_rate,
            generation_config=defaults,
            **kwargs,
        )
        self.revision = revision
        self.codec_name_or_path = codec_name_or_path
        self.codec_revision = codec_revision
        self.codec_conversion_directory = codec_conversion_directory
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.torch_dtype = torch_dtype
        self.use_safetensors = use_safetensors
        self.trust_remote_code = trust_remote_code
        self.verify_official_integrity = verify_official_integrity
        self.verify_full_shard_hashes = verify_full_shard_hashes
        self.compile = compile
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.chunk_length = chunk_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.training_max_length = training_max_length
        self.training_base_loss_weight = training_base_loss_weight
        self.training_semantic_loss_weight = training_semantic_loss_weight
        self.training_adam_beta1 = training_adam_beta1
        self.training_adam_beta2 = training_adam_beta2
        self.training_adam_epsilon = training_adam_epsilon
        self.training_warmup_steps = training_warmup_steps
        self.training_lora_config = (None if training_lora_config is None else dict(training_lora_config))
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.tokenizer_kwargs = ({} if tokenizer_kwargs is None else dict(tokenizer_kwargs))
        self.validate()

    def validate(self) -> None:
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if self.sample_rate != 44_100:
            raise ValueError("Fish S2 ModifiedDAC requires 44,100 Hz output.")
        for name in ("revision", "codec_revision"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"`{name}` must be non-empty or None.")
                setattr(self, name, value.strip())
        for name in (
                "codec_name_or_path",
                "codec_conversion_directory",
                "cache_dir",
        ):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (str, Path)) or not str(value).strip():
                    raise ValueError(f"`{name}` must be path-like or None.")
                setattr(self, name, str(Path(value).expanduser()))
        for name in (
                "local_files_only",
                "trust_remote_code",
                "verify_official_integrity",
                "verify_full_shard_hashes",
                "compile",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.trust_remote_code:
            raise ValueError(
                "Native Fish S2 never executes repository code; "
                "`trust_remote_code=True` is unsupported.")
        if self.use_safetensors is False:
            raise ValueError(
                "Native Fish S2 semantic and steady-state codec loading "
                "require Safetensors.")
        if self.use_safetensors not in (None, True):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        if self.compile:
            raise ValueError(
                "Fish-specific `compile=True` is retired. Select a "
                "reversible VoiceHub inference strategy instead.")
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        try:
            self.torch_dtype = _DTYPES[self.torch_dtype.strip().lower()]
        except KeyError as error:
            raise ValueError("`torch_dtype` must be auto, float32, float16, or bfloat16.") from error
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise TypeError("`seed` must be an integer or None.")
        self.max_new_tokens = _positive_integer(
            self.max_new_tokens,
            name="max_new_tokens",
        )
        self.chunk_length = _positive_integer(
            self.chunk_length,
            name="chunk_length",
        )
        if (isinstance(self.temperature, bool) or not isinstance(self.temperature, Real) or
                not math.isfinite(float(self.temperature)) or not 0.0 < float(self.temperature) < 2.0):
            raise ValueError("`temperature` must be finite and in (0, 2).")
        if (isinstance(self.top_p, bool) or not isinstance(self.top_p, Real) or
                not math.isfinite(float(self.top_p)) or not 0.0 < float(self.top_p) <= 1.0):
            raise ValueError("`top_p` must be finite and in (0, 1].")
        self.top_k = _positive_integer(self.top_k, name="top_k")
        if self.training_max_length is not None:
            self.training_max_length = _positive_integer(
                self.training_max_length,
                name="training_max_length",
            )
        for name in (
                "training_base_loss_weight",
                "training_semantic_loss_weight",
                "training_adam_epsilon",
        ):
            setattr(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if (self.training_base_loss_weight + self.training_semantic_loss_weight == 0):
            raise ValueError("At least one Fish training loss must be active.")
        for name in ("training_adam_beta1", "training_adam_beta2"):
            value = _finite_nonnegative(getattr(self, name), name=name)
            if value >= 1.0:
                raise ValueError(f"`{name}` must be smaller than one.")
            setattr(self, name, value)
        if self.training_warmup_steps is not None:
            if (isinstance(self.training_warmup_steps, bool) or
                    not isinstance(self.training_warmup_steps, int) or self.training_warmup_steps < 0):
                raise ValueError("`training_warmup_steps` must be non-negative or None.")
        if self.training_lora_config is not None:
            raise ValueError(
                "Fish provider-owned LoRA modules are not loaded by the "
                "native graph. Use full fine-tuning now; a shared VoiceHub "
                "parameter-efficient strategy can be attached separately.")
        for name in ("model_kwargs", "tokenizer_kwargs"):
            values = getattr(self, name)
            if not isinstance(values, Mapping):
                raise TypeError(f"`{name}` must be a mapping.")
            if values:
                raise ValueError(f"Native Fish S2 does not delegate `{name}`.")
            setattr(self, name, {})

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


__all__ = ["FishTTSConfig"]
