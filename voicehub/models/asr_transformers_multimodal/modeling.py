"""Native compatibility dispatch for historical multimodal ASR models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.models.asr_transformers_multimodal.configuration import _load_config_class, _resolve_provider


def _load_model_class(provider: str):
    if provider == "qwen3":
        from voicehub.models.asr_qwen3.modeling import Qwen3ASRForSpeechRecognition

        return Qwen3ASRForSpeechRecognition
    if provider == "vibevoice":
        from voicehub.models.asr_vibevoice.modeling import VibeVoiceForSpeechRecognition

        return VibeVoiceForSpeechRecognition
    raise ValueError(f"Unsupported native multimodal ASR provider: {provider!r}.")


def _config_model_type(config: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get("model_type")
    return getattr(config, "model_type", None)


class MultimodalTransformersASRForSpeechRecognition:
    """Compatibility factory returning a dedicated VoiceHub-native model.

    Callers should prefer the concrete Qwen3-ASR or VibeVoice-ASR class.
    This factory is retained for source compatibility and requires
    enough information to select one architecture before any model code
    is imported.
    """

    def __new__(
        cls,
        config: Any = None,
        *,
        provider: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ):
        source = config if isinstance(config, (str, Path)) else kwargs.get("model_path")
        resolved = _resolve_provider(
            provider=provider,
            model_type=model_type or _config_model_type(config),
            source=source,
        )
        if isinstance(config, Mapping):
            config = _load_config_class(resolved).from_dict(dict(config))
        return _load_model_class(resolved)(config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config: Any = None,
        provider: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ):
        resolved = _resolve_provider(
            provider=provider,
            model_type=model_type or _config_model_type(config),
            source=pretrained_model_name_or_path,
        )
        if isinstance(config, Mapping):
            config = _load_config_class(resolved).from_dict(dict(config))
        return _load_model_class(resolved).from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            **kwargs,
        )


_LAZY_EXPORTS = (
    "Qwen3ASRForSpeechRecognition",
    "VibeVoiceASRForSpeechRecognition",
)
__all__ = [
    "MultimodalTransformersASRForSpeechRecognition",
    *_LAZY_EXPORTS,
]


def __getattr__(name: str) -> Any:
    if name == "Qwen3ASRForSpeechRecognition":
        value = _load_model_class("qwen3")
    elif name == "VibeVoiceASRForSpeechRecognition":
        value = _load_model_class("vibevoice")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
