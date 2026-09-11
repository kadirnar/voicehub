"""Lazy public exports for VoiceHub's native MeloTTS family."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "MeloTTSConfig": ("voicehub.models.melotts.configuration"),
    "MeloTTSForTextToSpeech": "voicehub.models.melotts.modeling",
    "MeloTTSTrainingAdapter": "voicehub.models.melotts.training",
    "MeloTTSTrainingCollator": "voicehub.models.melotts.training",
}


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


__all__ = sorted(_EXPORTS)
