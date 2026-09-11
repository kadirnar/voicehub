"""Public configuration for VoiceHub's native MeloTTS runtime."""

from __future__ import annotations

from typing import Any

from voicehub.configuration import VoiceHubConfig


class MeloTTSConfig(VoiceHubConfig):
    """Configure pinned releases, local artifacts, and native fine-tuning."""

    model_type = "melotts"

    def __init__(
        self,
        *,
        language: str = "EN",
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        checkpoint_filename: str | None = None,
        revision: str | None = None,
        use_huggingface: bool = True,
        sample_rate: int = 44_100,
        trust_pickle_checkpoint: bool = False,
        dtype: str = "float32",
        enable_native_finetuning: bool = False,
        training_enable_discriminators: bool = True,
        **kwargs: Any,
    ) -> None:
        if not isinstance(language, str) or not language.strip():
            raise ValueError("`language` must be a non-empty checkpoint code.")
        for name, value in (
            ("config_path", config_path),
            ("checkpoint_path", checkpoint_path),
            ("checkpoint_filename", checkpoint_filename),
            ("revision", revision),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"`{name}` must be non-empty or None.")
        if checkpoint_filename is not None and ("/" in checkpoint_filename or "\\" in checkpoint_filename):
            raise ValueError("`checkpoint_filename` must be one plain file name.")
        if not isinstance(use_huggingface, bool):
            raise TypeError("`use_huggingface` must be a boolean.")
        if use_huggingface is not True:
            raise ValueError(
                "VoiceHub's native MeloTTS aliases use pinned Hugging Face "
                "artifacts. Pass a local artifact path for offline loading.")
        if not isinstance(trust_pickle_checkpoint, bool):
            raise TypeError("`trust_pickle_checkpoint` must be a boolean.")
        if not isinstance(dtype, str) or not dtype.strip():
            raise ValueError("`dtype` must be a non-empty string.")
        if not isinstance(enable_native_finetuning, bool):
            raise TypeError("`enable_native_finetuning` must be a boolean.")
        if not isinstance(training_enable_discriminators, bool):
            raise TypeError("`training_enable_discriminators` must be a boolean.")
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.language = language.strip().upper()
        self.config_path = (None if config_path is None else config_path.strip())
        self.checkpoint_path = (None if checkpoint_path is None else checkpoint_path.strip())
        self.checkpoint_filename = (None if checkpoint_filename is None else checkpoint_filename.strip())
        self.revision = None if revision is None else revision.strip()
        self.use_huggingface = use_huggingface
        self.trust_pickle_checkpoint = trust_pickle_checkpoint
        self.dtype = dtype.strip()
        self.enable_native_finetuning = enable_native_finetuning
        self.training_enable_discriminators = training_enable_discriminators


__all__ = ["MeloTTSConfig"]
