"""OpenAI Whisper compatibility provider backed by VoiceHub's native graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.models.asr_native.configuration import OpenAIWhisperConfig
from voicehub.models.asr_native.whisper_compat import normalize_whisper_source
from voicehub.models.asr_whisper_native.modeling import WhisperForSpeechRecognition


class OpenAIWhisperForSpeechRecognition(WhisperForSpeechRecognition):
    """Run legacy OpenAI model names with VoiceHub-owned Whisper code.

    Official model aliases resolve to the corresponding Hugging Face
    Safetensors repository. No ``openai-whisper`` runtime is imported.
    Inference, teacher-forced fine-tuning, and native export all use the
    same graph as :class:`WhisperForSpeechRecognition`.
    """

    config_class = OpenAIWhisperConfig
    default_model_name_or_path = "openai/whisper-small"

    def __init__(
        self,
        config: OpenAIWhisperConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, OpenAIWhisperConfig):
            values = config.to_dict()
            values["name_or_path"] = normalize_whisper_source(values.get("name_or_path", ""))
            config = OpenAIWhisperConfig.from_dict(values)
        elif isinstance(config, (str, Path)):
            config = normalize_whisper_source(config)
        if model_path is not None:
            model_path = normalize_whisper_source(model_path)
        super().__init__(
            config,
            model_path=model_path,
            device=device,
            lazy_load=lazy_load,
            token=token,
            **kwargs,
        )


__all__ = ["OpenAIWhisperForSpeechRecognition"]
