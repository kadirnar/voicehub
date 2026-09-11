"""Lazy public exports for native SeamlessM4T-v2 speech recognition."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.asr_seamless_m4t_v2."
_EXPORTS = {
    "NativeSeamlessM4Tv2TrainingAdapter": (_PACKAGE + "training_asr_seamless_m4t_v2"),
    "SeamlessM4Tv2ASRConfig": (_PACKAGE + "configuration"),
    "SeamlessM4Tv2ForSpeechRecognition": (_PACKAGE + "modeling"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
