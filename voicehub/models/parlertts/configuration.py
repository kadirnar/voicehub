"""Dependency-light configuration for VoiceHub-native Parler-TTS."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from voicehub.configuration import VoiceHubConfig


class ParlerTTSConfig(VoiceHubConfig):
    """Loading configuration for the dependency-free Parler-TTS runtime."""

    model_type = "parlertts"

    def __init__(
        self,
        *,
        architecture: Mapping[str, Any] | None = None,
        attention_implementation: str | None = "sdpa",
        compile_model: bool = False,
        freeze_text_encoder: bool = False,
        torch_dtype: str | None = None,
        sample_rate: int = 44_100,
        cache_dir: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        verify_artifacts: bool = False,
        verify_checkpoint: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.architecture = dict(architecture or {})
        self.attention_implementation = (
            "eager" if attention_implementation is None else attention_implementation)
        self.compile_model = compile_model
        self.freeze_text_encoder = freeze_text_encoder
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.revision = revision
        self.token = token
        self.local_files_only = local_files_only
        self.verify_artifacts = verify_artifacts
        self.verify_checkpoint = verify_checkpoint
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.architecture, Mapping):
            raise TypeError("`architecture` must be a mapping.")
        if self.attention_implementation not in {"eager", "sdpa"}:
            raise ValueError("`attention_implementation` must be 'eager' or 'sdpa'.")
        for name in (
                "compile_model",
                "freeze_text_encoder",
                "local_files_only",
                "verify_artifacts",
                "verify_checkpoint",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")

    def to_dict(self) -> dict[str, Any]:
        values = super().to_dict()
        values.pop("token", None)
        return values


__all__ = ["ParlerTTSConfig"]
