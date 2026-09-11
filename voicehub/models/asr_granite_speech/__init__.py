"""VoiceHub-native Granite Speech provider with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "GraniteSpeechASRConfig": ("voicehub.models.asr_granite_speech."
                               "configuration"),
    "GraniteSpeechForSpeechRecognition": ("voicehub.models.asr_granite_speech."
                                          "modeling"),
    "NativeGraniteSpeechTrainingAdapter":
    ("voicehub.models.asr_granite_speech."
     "training_asr_granite_speech"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(
        import_module(module_name),
        name,
    )
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
