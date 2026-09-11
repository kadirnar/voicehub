"""VoiceHub-native Parakeet TDT provider with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.asr_parakeet_tdt."
_EXPORTS = {
    "NativeParakeetTDTTrainingAdapter": (_PACKAGE + "training_asr_parakeet_tdt"),
    "ParakeetTDTASRConfig": (_PACKAGE + "configuration"),
    "ParakeetTDTForSpeechRecognition": (_PACKAGE + "modeling"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve graph/training components only when explicitly requested."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
