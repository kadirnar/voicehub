"""Configuration for VoiceHub's native Tiron speech recognizer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.inference_configuration import ASRInferenceConfig
from voicehub.models.asr_tiron.metadata import TIRON_CHECKPOINT_REVISION
from voicehub.models.asr_whisper_native.configuration import WhisperASRConfig


class TironASRConfig(WhisperASRConfig):
    """Configure native Tiron checkpoint loading, decoding, and fine-tuning.

    Tiron uses the Whisper large-v3 graph with an extended speaker-token
    vocabulary. The public checkpoint is pinned by default so model
    weights, tokenizer grammar, and generation metadata cannot drift
    independently.
    """

    model_type = "asr_tiron"
    architecture_family = "speech-seq2seq"

    def __init__(
        self,
        *,
        default_language: str = "en",
        constrained_decoding: bool = True,
        max_speakers: int | None = None,
        revision: str | None = TIRON_CHECKPOINT_REVISION,
        architecture_family: str | None = None,
        inference_config: ASRInferenceConfig | Mapping[str, Any] | None = None,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        requested_family = (
            self.architecture_family
            if architecture_family is None else str(architecture_family).strip().lower().replace("_", "-"))
        if requested_family != self.architecture_family:
            raise ValueError(
                "TironASRConfig requires "
                "`architecture_family='speech-seq2seq'`; received "
                f"{architecture_family!r}.")
        pipeline_kwargs = kwargs.pop("pipeline_kwargs", None)
        if pipeline_kwargs:
            raise ValueError(
                "`pipeline_kwargs` are unavailable in the native Tiron "
                "runtime; pass supported inference options directly.")
        self.default_language = default_language
        self.constrained_decoding = constrained_decoding
        self.max_speakers = max_speakers
        super().__init__(
            revision=revision,
            cache_dir=cache_dir,
            inference_config=inference_config,
            architecture_family=self.architecture_family,
            **kwargs,
        )

    def validate(self) -> None:
        """Validate Tiron-specific grammar controls and Whisper settings."""
        super().validate()
        if (not isinstance(self.default_language, str) or not self.default_language.strip()):
            raise ValueError("`default_language` must be a non-empty Whisper language.")
        self.default_language = self.default_language.strip().lower()
        if not isinstance(self.constrained_decoding, bool):
            raise TypeError("`constrained_decoding` must be a boolean.")
        if self.max_speakers is not None:
            if (isinstance(self.max_speakers, bool) or not isinstance(self.max_speakers, int)):
                raise TypeError("`max_speakers` must be an integer or None.")
            if not 1 <= self.max_speakers <= 8:
                raise ValueError("`max_speakers` must be between 1 and 8.")


__all__ = ["TironASRConfig"]
