"""Serializable configuration for the vui model family."""

from __future__ import annotations

from typing import Any

from voicehub.configuration import VoiceHubConfig
from voicehub.models.vui.artifacts import VUI_CODEC_FILENAME, VUI_REVISION


class VuiConfig(VoiceHubConfig):
    """Configuration for Vui checkpoint loading."""

    model_type = "vui"

    def __init__(
        self,
        *,
        sample_rate: int = 22_050,
        checkpoint_filename: str | None = None,
        codec_filename: str = VUI_CODEC_FILENAME,
        native_artifact_format: str | None = None,
        native_artifact_format_version: int | None = None,
        native_model_config: dict[str, Any] | None = None,
        native_codec_config: dict[str, Any] | None = None,
        revision: str = VUI_REVISION,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        verify_official_integrity: bool = True,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            checkpoint_filename=checkpoint_filename,
            codec_filename=codec_filename,
            native_artifact_format=native_artifact_format,
            native_artifact_format_version=native_artifact_format_version,
            native_model_config=native_model_config,
            native_codec_config=native_codec_config,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            verify_official_integrity=verify_official_integrity,
            **kwargs,
        )


__all__ = ['VuiConfig']
