"""Transformers-style configuration primitives for VoiceHub models."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, resolve_pretrained_file, write_json_file
from voicehub.path_utils import normalize_model_source
from voicehub.serialization_utils import reject_serialized_secrets, serialize_paths

CONFIG_NAME = "config.json"


class VoiceHubConfig:
    """Serializable configuration shared by all VoiceHub speech
    architectures."""

    model_type = "voicehub"
    is_composition = False
    _RUNTIME_ONLY_FIELDS = frozenset({"token"})

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        architectures: list[str] | None = None,
        name_or_path: str | Path = "",
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        generation_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        reject_serialized_secrets(
            {
                "generation_config": generation_config,
                **kwargs,
            },
            owner=self.__class__.__name__,
        )
        self.sample_rate = sample_rate
        self.architectures = architectures or []
        self.name_or_path = name_or_path
        self.return_dict = return_dict
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.generation_config = generation_config or {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep-copied JSON-serializable representation."""
        values = deepcopy(self.__dict__)
        for field_name in self._RUNTIME_ONLY_FIELDS:
            values.pop(field_name, None)
        reject_serialized_secrets(
            values,
            owner=self.__class__.__name__,
        )
        output = serialize_paths(values)
        output["model_type"] = self.model_type
        return output

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs):
        """Create a config and apply explicit keyword overrides."""
        values = dict(config_dict)
        values.pop("model_type", None)
        values.update(kwargs)
        return cls(**values)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "",
        cache_dir: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        """Load ``config.json`` from a directory, file, or Hub repository."""
        pretrained_model_name_or_path = normalize_model_source(pretrained_model_name_or_path)
        source = Path(pretrained_model_name_or_path).expanduser()
        if source.is_file():
            if source.suffix.lower() != ".json":
                if cls is VoiceHubConfig:
                    raise ValueError(
                        "A raw checkpoint file does not identify its model "
                        "type. Use AutoConfig.from_pretrained(..., "
                        "model_type=...) or a concrete config class.")
                return cls(
                    name_or_path=str(source),
                    **kwargs,
                )
            config_path = source
        else:
            config_path = resolve_pretrained_file(
                pretrained_model_name_or_path,
                CONFIG_NAME,
                subfolder=subfolder,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
                local_files_only=local_files_only,
            )
        config = cls.from_dict(read_json_file(config_path), **kwargs)
        config.name_or_path = pretrained_model_name_or_path
        return config

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Save this configuration as ``config.json``."""
        output_path = Path(save_directory).expanduser() / CONFIG_NAME
        values = self.to_dict()
        reject_serialized_secrets(
            values,
            owner=self.__class__.__name__,
        )
        write_json_file(output_path, values)
        return output_path

    def to_json_string(self, *, use_diff: bool = False) -> str:
        """Serialize this configuration as stable, readable JSON."""
        values = self.to_diff_dict() if use_diff else self.to_dict()
        reject_serialized_secrets(
            values,
            owner=self.__class__.__name__,
        )
        return json.dumps(
            values,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    def to_json_file(
        self,
        json_file_path: str | Path,
        *,
        use_diff: bool = False,
    ) -> Path:
        """Write this configuration to an explicit JSON file."""
        output_path = Path(json_file_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self.to_json_string(use_diff=use_diff),
            encoding="utf-8",
        )
        return output_path

    def to_diff_dict(self) -> dict[str, Any]:
        """Return values that differ from the common base configuration."""
        defaults = VoiceHubConfig().to_dict()
        values = self.to_dict()
        reject_serialized_secrets(
            values,
            owner=self.__class__.__name__,
        )
        return {
            key: value
            for key, value in values.items() if key == "model_type" or defaults.get(key) != value
        }

    def update(self, values: dict[str, Any]) -> None:
        """Apply configuration values in place."""
        for key, value in values.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        values = self.to_dict()
        reject_serialized_secrets(
            values,
            owner=self.__class__.__name__,
        )
        fields = ", ".join(f"{key}={value!r}" for key, value in values.items())
        return f"{self.__class__.__name__}({fields})"
