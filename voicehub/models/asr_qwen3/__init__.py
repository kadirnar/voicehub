"""VoiceHub-native Qwen3-ASR provider with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.asr_qwen3."
_EXPORTS = {
    "NativeQwen3ASRTrainingAdapter": _PACKAGE + "training_asr_qwen3",
    "Qwen3ASRConfig": _PACKAGE + "configuration",
    "Qwen3ASRForSpeechRecognition": _PACKAGE + "modeling",
}

__all__ = [
    "NativeQwen3ASRTrainingAdapter",
    "Qwen3ASRConfig",
    "Qwen3ASRForSpeechRecognition",
]


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
