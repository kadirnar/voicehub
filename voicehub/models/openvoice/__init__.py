"""Lazy public exports for VoiceHub's native OpenVoice V2 runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "OpenVoiceConfig": (
        "voicehub.models.openvoice.configuration",
        "OpenVoiceConfig",
    ),
    "OpenVoiceForTextToSpeech": (
        "voicehub.models.openvoice.modeling",
        "OpenVoiceForTextToSpeech",
    ),
    "OpenVoiceTrainingAdapter": (
        "voicehub.models.openvoice.training",
        "OpenVoiceTrainingAdapter",
    ),
    "OpenVoiceTrainingCollator": (
        "voicehub.models.openvoice.training",
        "OpenVoiceTrainingCollator",
    ),
    "OpenVoiceTTS": (
        "voicehub.models.openvoice.modeling",
        "OpenVoiceTTS",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
