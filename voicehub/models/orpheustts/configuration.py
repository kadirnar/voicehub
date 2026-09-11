"""Configuration for VoiceHub-native Orpheus TTS."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.models.orpheustts.artifacts import SNAC_SAFE_REVISION

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


def _filename(value: object, *, name: str, optional: bool) -> str | None:
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


class OrpheusTTSConfig(VoiceHubConfig):
    """Configure native Llama inference/training and the frozen SNAC codec."""

    model_type = "orpheustts"

    def __init__(
        self,
        *,
        codec_name_or_path: str | Path = "hubertsiuzdak/snac_24khz",
        revision: str | None = None,
        codec_revision: str | None = SNAC_SAFE_REVISION,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        checkpoint_filename: str | None = None,
        tokenizer_filename: str = "tokenizer.json",
        codec_checkpoint_filename: str = "model.safetensors",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        use_safetensors: bool | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        sample_rate: int = 24_000,
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
        self.tokenizer_filename = tokenizer_filename
        self.codec_checkpoint_filename = codec_checkpoint_filename
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.use_safetensors = use_safetensors
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.tokenizer_kwargs = ({} if tokenizer_kwargs is None else dict(tokenizer_kwargs))
        self.validate()

    def validate(self) -> None:
        """Reject options that would delegate the architecture externally."""
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if self.sample_rate != 24_000:
            raise ValueError("Orpheus SNAC checkpoints require `sample_rate=24000`.")
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
        self.checkpoint_filename = _filename(
            self.checkpoint_filename,
            name="checkpoint_filename",
            optional=True,
        )
        self.tokenizer_filename = _filename(
            self.tokenizer_filename,
            name="tokenizer_filename",
            optional=False,
        )
        self.codec_checkpoint_filename = _filename(
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
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("`trust_remote_code` must be a boolean.")
        if self.trust_remote_code:
            raise ValueError(
                "Native Orpheus never executes repository code; "
                "`trust_remote_code=True` is unsupported.")
        if self.use_safetensors is False:
            raise ValueError(
                "Native Orpheus accepts Safetensors only; "
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
                    f"Native Orpheus does not delegate `{name}`; unsupported "
                    f"option(s): {options}.")
            setattr(self, name, values)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


__all__ = ["OrpheusTTSConfig"]
