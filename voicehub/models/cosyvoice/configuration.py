"""Public configuration for the dependency-free CosyVoice provider."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.models.cosyvoice.native.metadata import COSYVOICE3_MODEL_REVISION

_DTYPES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "float16": "float16",
    "float32": "float32",
    "fp16": "float16",
    "fp32": "float32",
    "half": "float16",
}
_COMPONENTS = {
    "llm",
    "language_model",
    "flow",
    "hifigan_generator",
    "hifigan_discriminator",
}


class CosyVoiceConfig(VoiceHubConfig):
    """Safe loading, generation, and source-component training controls."""

    model_type = "cosyvoice"

    def __init__(
        self,
        *,
        revision: str | None = COSYVOICE3_MODEL_REVISION,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        torch_dtype: str = "float32",
        use_safetensors: bool | None = None,
        trust_remote_code: bool = False,
        training_component: str = "llm",
        generation_config: Mapping[str, Any] | None = None,
        sample_rate: int = 24_000,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(kwargs, owner=self.__class__.__name__)
        defaults = {
            "flow_steps": 10,
            "max_new_tokens": 1_024,
            "min_new_tokens": 0,
            "temperature": 1.0,
            "top_k": 25,
            "top_p": 0.8,
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
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.torch_dtype = torch_dtype
        self.use_safetensors = use_safetensors
        self.trust_remote_code = trust_remote_code
        self.training_component = training_component
        self.validate()

    def validate(self) -> None:
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if self.sample_rate != 24_000:
            raise ValueError("CosyVoice3 HiFT produces 24 kHz audio.")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("`revision` must be a non-empty string or None.")
        if self.revision is not None:
            self.revision = self.revision.strip()
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)) or not str(self.cache_dir).strip():
                raise ValueError("`cache_dir` must be a non-empty path or None.")
            self.cache_dir = str(Path(self.cache_dir).expanduser())
        for name in ("local_files_only", "trust_remote_code"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.trust_remote_code:
            raise ValueError(
                "Native CosyVoice never executes repository code; "
                "`trust_remote_code=True` is unsupported.")
        if self.use_safetensors is False:
            raise ValueError(
                "Native CosyVoice requires Safetensors. Published .pt files "
                "must pass the explicit audited one-time converter.")
        if self.use_safetensors not in (None, True):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        dtype = _DTYPES.get(self.torch_dtype.strip().lower().removeprefix("torch."))
        if dtype is None:
            raise ValueError(f"Unsupported CosyVoice dtype {self.torch_dtype!r}.")
        self.torch_dtype = dtype
        if not isinstance(self.training_component, str):
            raise TypeError("`training_component` must be a string.")
        component = self.training_component.strip().lower().replace("-", "_")
        if component == "language_model":
            component = "llm"
        if component not in _COMPONENTS:
            raise ValueError(
                "`training_component` must be llm, flow, "
                "hifigan_generator, or hifigan_discriminator.")
        self.training_component = component
        generation = self.generation_config
        if not isinstance(generation, Mapping):
            raise TypeError("`generation_config` must be a mapping.")
        unknown = set(generation) - {
            "flow_steps",
            "max_new_tokens",
            "min_new_tokens",
            "temperature",
            "top_k",
            "top_p",
        }
        if unknown:
            raise ValueError("Unsupported CosyVoice generation options: " + ", ".join(sorted(unknown)))
        for name in (
                "flow_steps",
                "max_new_tokens",
                "min_new_tokens",
                "top_k",
        ):
            value = generation[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "min_new_tokens"
                                                                                 else 1):
                raise ValueError(f"`generation_config.{name}` is invalid.")
        if generation["min_new_tokens"] > generation["max_new_tokens"]:
            raise ValueError("Minimum generation length exceeds maximum.")
        for name in ("temperature", "top_p"):
            value = generation[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or value <= 0):
                raise ValueError(f"`generation_config.{name}` must be positive.")
        if generation["top_p"] > 1:
            raise ValueError("`generation_config.top_p` cannot exceed one.")
        self.generation_config = dict(generation)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


__all__ = ["CosyVoiceConfig"]
