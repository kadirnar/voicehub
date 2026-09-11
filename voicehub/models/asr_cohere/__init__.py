"""VoiceHub-native Cohere Transcribe provider with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.asr_cohere."
_EXPORTS = {
    "CohereASRConfig": _PACKAGE + "configuration",
    "CohereForSpeechRecognition": _PACKAGE + "modeling",
    "NativeCohereASRTrainingAdapter": _PACKAGE + "training_asr_cohere",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
