"""Faster-whisper compatibility provider using VoiceHub's native graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.models.asr_native.configuration import FasterWhisperConfig
from voicehub.models.asr_native.whisper_compat import normalize_whisper_source
from voicehub.models.asr_whisper_native.modeling import WhisperForSpeechRecognition


class FasterWhisperForSpeechRecognition(WhisperForSpeechRecognition):
    """Preserve legacy model names without depending on CTranslate2.

    This compatibility provider intentionally uses the canonical native
    Whisper graph. Runtime quantization belongs to VoiceHub's
    optimization layer rather than a second model implementation.
    """

    config_class = FasterWhisperConfig
    default_model_name_or_path = "openai/whisper-small"

    def __init__(
        self,
        config: FasterWhisperConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, FasterWhisperConfig):
            values = config.to_dict()
            values["name_or_path"] = normalize_whisper_source(values.get("name_or_path", ""))
            config = FasterWhisperConfig.from_dict(values)
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


__all__ = ["FasterWhisperForSpeechRecognition"]
