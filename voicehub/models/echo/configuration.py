"""Serializable configuration for the echo model family."""

from __future__ import annotations

from pathlib import Path

from voicehub.configuration import VoiceHubConfig


class EchoTTSConfig(VoiceHubConfig):
    """Configuration for Echo-TTS flow matching inference."""

    model_type = "echo"

    def __init__(
        self,
        *,
        compile_model: bool = False,
        compile: bool | None = None,
        codec_name_or_path: str | Path = "jordand/fish-s1-dac-min",
        pca_name_or_path: str | Path | None = None,
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.compile_model = compile_model if compile is None else compile
        if not isinstance(self.compile_model, bool):
            raise TypeError("`compile_model` must be a boolean.")
        if (not isinstance(codec_name_or_path, (str, Path)) or not str(codec_name_or_path).strip()):
            raise ValueError("`codec_name_or_path` must be a non-empty path or Hub ID.")
        self.codec_name_or_path = str(codec_name_or_path)
        if pca_name_or_path is not None and (not isinstance(pca_name_or_path, (str, Path)) or
                                             not str(pca_name_or_path).strip()):
            raise ValueError("`pca_name_or_path` must be a non-empty path, Hub ID, or None.")
        self.pca_name_or_path = (None if pca_name_or_path is None else str(pca_name_or_path))


__all__ = ['EchoTTSConfig']
