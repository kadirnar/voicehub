"""VoiceHub-native WavLM CTC provider with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.asr_wavlm."
_EXPORTS = {
    "NativeWavLMTrainingAdapter": _PACKAGE + "training_asr_wavlm",
    "WavLMASRConfig": _PACKAGE + "configuration",
    "WavLMForSpeechRecognition": _PACKAGE + "modeling",
    "WavLMProcessor": _PACKAGE + "processing",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve provider components only when explicitly requested."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
