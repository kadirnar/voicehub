"""Lazy public exports for the VoiceHub-native NeuTTS family."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "NeuTTSConfig": "voicehub.models.neutts.configuration",
    "NeuTTSForTextToSpeech": "voicehub.models.neutts.modeling",
    "NeuTTSModel": "voicehub.models.neutts.modeling",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = ["NeuTTSConfig", "NeuTTSForTextToSpeech", "NeuTTSModel"]
