"""Compatibility namespace for VoiceHub-native multimodal ASR providers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "MultimodalTransformersASRConfig": (
        "voicehub.models.asr_transformers_multimodal.configuration",
        "MultimodalTransformersASRConfig",
    ),
    "MultimodalTransformersASRForSpeechRecognition": (
        "voicehub.models.asr_transformers_multimodal.modeling",
        "MultimodalTransformersASRForSpeechRecognition",
    ),
    "Qwen3ASRConfig": (
        "voicehub.models.asr_qwen3.configuration",
        "Qwen3ASRConfig",
    ),
    "Qwen3ASRForSpeechRecognition": (
        "voicehub.models.asr_qwen3.modeling",
        "Qwen3ASRForSpeechRecognition",
    ),
    "VibeVoiceASRConfig": (
        "voicehub.models.asr_vibevoice.configuration",
        "VibeVoiceASRConfig",
    ),
    "VibeVoiceASRForSpeechRecognition": (
        "voicehub.models.asr_vibevoice.modeling",
        "VibeVoiceForSpeechRecognition",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
