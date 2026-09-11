"""Public VoiceHub-native VibeVoice ASR provider."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.asr_vibevoice."
_EXPORTS = {
    "NativeVibeVoiceASRTrainingAdapter": _PACKAGE + "training_asr_vibevoice",
    "VibeVoiceASRConfig": _PACKAGE + "configuration",
    "VibeVoiceForSpeechRecognition": _PACKAGE + "modeling",
}

__all__ = [
    "NativeVibeVoiceASRTrainingAdapter",
    "VibeVoiceASRConfig",
    "VibeVoiceForSpeechRecognition",
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
