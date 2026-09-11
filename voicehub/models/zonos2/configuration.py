"""Dependency-light configuration for VoiceHub-native ZONOS2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from voicehub.configuration import VoiceHubConfig


class Zonos2Config(VoiceHubConfig):
    """Loading and execution settings for native ZONOS2."""

    model_type = "zonos2"

    def __init__(
        self,
        *,
        architecture: Mapping[str, Any] | None = None,
        torch_dtype: str | None = "bfloat16",
        decode_audio: bool = True,
        sample_rate: int = 44_100,
        cache_dir: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        verify_artifacts: bool = False,
        compile_model: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.architecture = dict(architecture or {})
        self.torch_dtype = torch_dtype
        self.decode_audio = decode_audio
        self.cache_dir = cache_dir
        self.revision = revision
        self.token = token
        self.local_files_only = local_files_only
        self.verify_artifacts = verify_artifacts
        self.compile_model = compile_model
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.architecture, Mapping):
            raise TypeError("`architecture` must be a mapping.")
        for name in (
                "decode_audio",
                "local_files_only",
                "verify_artifacts",
                "compile_model",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int) or
                self.sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")

    def to_dict(self) -> dict[str, Any]:
        values = super().to_dict()
        values.pop("token", None)
        return values


__all__ = ["Zonos2Config"]
