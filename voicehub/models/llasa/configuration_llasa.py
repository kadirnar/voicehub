"""Configuration for VoiceHub-native LLaSA inference and fine-tuning."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig, reject_serialized_secrets
from voicehub.models.llasa.artifacts import XCODEC2_HF_REPOSITORY, XCODEC2_HF_REVISION

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


def _safe_filename(
    value: object,
    *,
    name: str,
    optional: bool,
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        suffix = " or None" if optional else ""
        raise ValueError(f"`{name}` must be a non-empty filename{suffix}.")
    normalized = value.strip()
    path = PurePosixPath(normalized)
    if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
        raise ValueError(f"`{name}` must be one safe artifact-root filename.")
    return normalized


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return value


def _sampling_value(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"`{name}` must be finite.")
    return result


class LlasaConfig(VoiceHubConfig):
    """Configure native Llama generation and the frozen XCodec2 codec."""

    model_type = "llasa"

    def __init__(
        self,
        *,
        codec_name_or_path: str | Path = XCODEC2_HF_REPOSITORY,
        revision: str | None = None,
        codec_revision: str | None = XCODEC2_HF_REVISION,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        checkpoint_filename: str | None = None,
        codec_checkpoint_filename: str = "model.safetensors",
        torch_dtype: str = "auto",
        max_new_tokens: int = 2_048,
        max_total_tokens: int = 2_048,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 50,
        date_string: str | None = None,
        sample_rate: int = 16_000,
        trust_remote_code: bool = False,
        use_safetensors: bool | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(
            {
                "model_kwargs": model_kwargs,
                "tokenizer_kwargs": tokenizer_kwargs,
                **kwargs,
            },
            owner=self.__class__.__name__,
        )
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.codec_name_or_path = codec_name_or_path
        self.revision = revision
        self.codec_revision = codec_revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.checkpoint_filename = checkpoint_filename
        self.codec_checkpoint_filename = codec_checkpoint_filename
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.max_total_tokens = max_total_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.date_string = date_string
        self.trust_remote_code = trust_remote_code
        self.use_safetensors = use_safetensors
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.tokenizer_kwargs = ({} if tokenizer_kwargs is None else dict(tokenizer_kwargs))
        self.validate()

    def validate(self) -> None:
        """Reject options that would delegate execution outside VoiceHub."""
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if self.sample_rate != 16_000:
            raise ValueError("LLaSA XCodec2 checkpoints require `sample_rate=16000`.")
        if not isinstance(self.codec_name_or_path, (str, Path)):
            raise TypeError("`codec_name_or_path` must be path-like.")
        if not str(self.codec_name_or_path).strip():
            raise ValueError("`codec_name_or_path` must be non-empty.")
        self.codec_name_or_path = str(self.codec_name_or_path)
        for name in ("revision", "codec_revision"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"`{name}` must be a non-empty string or None.")
                setattr(self, name, value.strip())
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be path-like or None.")
            self.cache_dir = str(Path(self.cache_dir).expanduser())
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        self.checkpoint_filename = _safe_filename(
            self.checkpoint_filename,
            name="checkpoint_filename",
            optional=True,
        )
        self.codec_checkpoint_filename = _safe_filename(
            self.codec_checkpoint_filename,
            name="codec_checkpoint_filename",
            optional=False,
        )
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        try:
            self.torch_dtype = _DTYPE_ALIASES[self.torch_dtype.strip().lower()]
        except KeyError as error:
            choices = ", ".join(sorted(set(_DTYPE_ALIASES.values())))
            raise ValueError(f"`torch_dtype` must be one of: {choices}.") from error
        self.max_new_tokens = _positive_integer(
            self.max_new_tokens,
            name="max_new_tokens",
        )
        self.max_total_tokens = _positive_integer(
            self.max_total_tokens,
            name="max_total_tokens",
        )
        self.temperature = _sampling_value(
            self.temperature,
            name="temperature",
        )
        if self.temperature <= 0.0:
            raise ValueError("`temperature` must be greater than zero.")
        self.top_p = _sampling_value(self.top_p, name="top_p")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("`top_p` must be in the interval (0, 1].")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("`top_k` must be an integer.")
        if self.top_k < 0:
            raise ValueError("`top_k` cannot be negative; use zero to disable filtering.")
        if self.date_string is not None and (not isinstance(self.date_string, str) or
                                             not self.date_string.strip()):
            raise ValueError("`date_string` must be a non-empty string or None.")
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("`trust_remote_code` must be a boolean.")
        if self.trust_remote_code:
            raise ValueError(
                "Native LLaSA never executes repository code; "
                "`trust_remote_code=True` is unsupported.")
        if self.use_safetensors is False:
            raise ValueError(
                "Native LLaSA accepts Safetensors only; "
                "`use_safetensors=False` is unsupported.")
        if self.use_safetensors not in (None, True):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        for name in ("model_kwargs", "tokenizer_kwargs"):
            values = getattr(self, name)
            if not isinstance(values, Mapping):
                raise TypeError(f"`{name}` must be a mapping.")
            values = dict(values)
            if values:
                options = ", ".join(sorted(str(key) for key in values))
                raise ValueError(
                    f"Native LLaSA does not delegate `{name}`; unsupported "
                    f"option(s): {options}.")
            setattr(self, name, values)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


__all__ = ["LlasaConfig"]
