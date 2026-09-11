"""Configuration for VoiceHub's native StyleTTS 2 runtime."""

from __future__ import annotations

from typing import Any

from voicehub.configuration import VoiceHubConfig


class StyleTTS2Config(VoiceHubConfig):
    """Configure local StyleTTS 2 artifacts and training-only components."""

    model_type = "styletts2"

    def __init__(
        self,
        *,
        config_path: str | None = None,
        assets_directory: str | None = None,
        language: str = "en-us",
        sample_rate: int = 24_000,
        trust_pickle_checkpoint: bool = False,
        dtype: str = "float32",
        enable_native_finetuning: bool = False,
        training_enable_discriminators: bool = True,
        **kwargs: Any,
    ) -> None:
        if config_path is not None and (not isinstance(config_path, str) or not config_path.strip()):
            raise ValueError("`config_path` must be non-empty or None.")
        if assets_directory is not None and (not isinstance(assets_directory, str) or
                                             not assets_directory.strip()):
            raise ValueError("`assets_directory` must be non-empty or None.")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("`language` must be a non-empty string.")
        if not isinstance(trust_pickle_checkpoint, bool):
            raise TypeError("`trust_pickle_checkpoint` must be a boolean.")
        if not isinstance(dtype, str) or not dtype.strip():
            raise ValueError("`dtype` must be a non-empty string.")
        if not isinstance(enable_native_finetuning, bool):
            raise TypeError("`enable_native_finetuning` must be a boolean.")
        if not isinstance(training_enable_discriminators, bool):
            raise TypeError("`training_enable_discriminators` must be a boolean.")
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.config_path = (None if config_path is None else config_path.strip())
        self.assets_directory = (None if assets_directory is None else assets_directory.strip())
        self.language = language.strip().lower()
        self.trust_pickle_checkpoint = trust_pickle_checkpoint
        self.dtype = dtype.strip()
        self.enable_native_finetuning = enable_native_finetuning
        self.training_enable_discriminators = (training_enable_discriminators)


__all__ = ["StyleTTS2Config"]
