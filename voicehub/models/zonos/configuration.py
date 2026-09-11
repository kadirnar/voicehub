"""Dependency-light configuration for VoiceHub-native Zonos v0.1."""

from __future__ import annotations

from typing import Any

from voicehub.configuration import VoiceHubConfig


class ZonosConfig(VoiceHubConfig):
    """Loading and execution settings for native Zonos v0.1."""

    model_type = "zonos"

    def __init__(
        self,
        *,
        torch_dtype: str | None = "auto",
        sample_rate: int = 44_100,
        cache_dir: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        verify_artifacts: bool = False,
        decode_audio: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.revision = revision
        self.token = token
        self.local_files_only = local_files_only
        self.verify_artifacts = verify_artifacts
        self.decode_audio = decode_audio
        self.validate()

    def validate(self) -> None:
        for name in (
                "local_files_only",
                "verify_artifacts",
                "decode_audio",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int) or
                self.sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")
        if self.sample_rate != 44_100:
            raise ValueError("The published Zonos v0.1 codec operates at 44,100 Hz.")

    def to_dict(self) -> dict[str, Any]:
        values = super().to_dict()
        values.pop("token", None)
        return values


__all__ = ["ZonosConfig"]
