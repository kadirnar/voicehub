"""Transformers-style generation configuration for speech synthesis."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from voicehub.configuration import reject_serialized_secrets
from voicehub.hub import read_json_file, resolve_pretrained_file, write_json_file
from voicehub.serialization_utils import serialize_paths

GENERATION_CONFIG_NAME = "generation_config.json"
TORCH_SEED_MIN = -(2**63)
TORCH_SEED_MAX = 2**64 - 1


class TTSGenerationConfig:
    """Serializable, extensible generation options for every TTS model."""

    _COMMON_FIELDS = frozenset({
        "output_file",
        "seed",
        "speed",
        "temperature",
        "top_p",
        "max_new_tokens",
    })

    def __init__(
        self,
        *,
        output_file: str | Path | None = None,
        seed: int | None = None,
        speed: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ):
        values = {
            "output_file": output_file,
            "seed": seed,
            "speed": speed,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            **kwargs,
        }
        reject_serialized_secrets(
            values,
            owner=self.__class__.__name__,
        )
        for key, value in values.items():
            if value is not None:
                setattr(self, key, value)
        self.validate()

    def validate(self) -> None:
        """Validate common values without rejecting backend extensions."""
        output_file = getattr(self, "output_file", None)
        if output_file is not None and (not isinstance(output_file,
                                                       (str, Path)) or not str(output_file).strip()):
            raise ValueError("`output_file` must be a non-empty path.")
        if (output_file is not None and Path(output_file).expanduser().is_dir()):
            raise IsADirectoryError(f"`output_file` is a directory: {Path(output_file).expanduser()}.")

        seed = getattr(self, "seed", None)
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise TypeError("`seed` must be an integer.")
        if seed is not None and not TORCH_SEED_MIN <= seed <= TORCH_SEED_MAX:
            raise ValueError(
                "`seed` must be in Torch's supported range "
                f"[{TORCH_SEED_MIN}, {TORCH_SEED_MAX}].")

        speed = getattr(self, "speed", None)
        if speed is not None:
            if isinstance(speed, bool) or not isinstance(speed, Real):
                raise TypeError("`speed` must be a real number.")
            if not isfinite(speed) or speed <= 0:
                raise ValueError("`speed` must be finite and greater than zero.")

        temperature = getattr(self, "temperature", None)
        if temperature is not None:
            if (isinstance(temperature, bool) or not isinstance(temperature, Real)):
                raise TypeError("`temperature` must be a real number.")
            if not isfinite(temperature) or temperature < 0:
                raise ValueError("`temperature` must be finite and non-negative.")

        top_p = getattr(self, "top_p", None)
        if top_p is not None:
            if isinstance(top_p, bool) or not isinstance(top_p, Real):
                raise TypeError("`top_p` must be a real number.")
            if not isfinite(top_p) or not 0 <= top_p <= 1:
                raise ValueError("`top_p` must be finite and in the interval [0, 1].")

        max_new_tokens = getattr(self, "max_new_tokens", None)
        if max_new_tokens is not None:
            if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral)):
                raise TypeError("`max_new_tokens` must be an integer.")
            if max_new_tokens <= 0:
                raise ValueError("`max_new_tokens` must be greater than zero.")

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy for generation or JSON serialization."""
        values = deepcopy(self.__dict__)
        reject_serialized_secrets(
            values,
            owner=self.__class__.__name__,
        )
        return serialize_paths(values)

    @classmethod
    def from_dict(cls, values: dict[str, Any], **kwargs):
        """Construct a generation configuration with explicit overrides."""
        merged = dict(values)
        merged.update(kwargs)
        return cls(**merged)

    @classmethod
    def from_model_config(cls, config):
        """Read optional generation defaults from a model configuration."""
        values = getattr(config, "generation_config", {})
        if isinstance(values, cls):
            return cls.from_dict(values.to_dict())
        if not isinstance(values, dict):
            raise TypeError("`generation_config` must be a mapping.")
        return cls.from_dict(values)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "",
        **kwargs,
    ):
        """Load ``generation_config.json`` from a path or Hub repository."""
        source = Path(pretrained_model_name_or_path).expanduser()
        if source.is_file() and source.name == GENERATION_CONFIG_NAME:
            config_path = source
        else:
            config_path = resolve_pretrained_file(
                pretrained_model_name_or_path,
                GENERATION_CONFIG_NAME,
                subfolder=subfolder,
                cache_dir=kwargs.pop("cache_dir", None),
                revision=kwargs.pop("revision", None),
                token=kwargs.pop("token", None),
                local_files_only=kwargs.pop("local_files_only", False),
            )
        return cls.from_dict(read_json_file(config_path), **kwargs)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Save generation defaults as ``generation_config.json``."""
        output_path = Path(save_directory).expanduser() / GENERATION_CONFIG_NAME
        values = self.to_dict()
        reject_serialized_secrets(
            values,
            owner=self.__class__.__name__,
        )
        write_json_file(output_path, values)
        return output_path

    def update(self, **kwargs) -> dict[str, Any]:
        """Apply known options and return unknown options."""
        unused = {}
        for key, value in kwargs.items():
            if key not in self._COMMON_FIELDS and not hasattr(self, key):
                unused[key] = value
                continue
            if value is None:
                self.__dict__.pop(key, None)
            else:
                setattr(self, key, value)
        self.validate()
        return unused

    def __repr__(self) -> str:
        values = self.to_dict()
        reject_serialized_secrets(
            values,
            owner=self.__class__.__name__,
        )
        fields = ", ".join(f"{key}={value!r}" for key, value in sorted(values.items()))
        return f"{self.__class__.__name__}({fields})"
