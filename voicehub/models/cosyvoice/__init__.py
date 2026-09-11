"""Lazy public exports for the VoiceHub-native CosyVoice 3 runtime."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "CosyVoiceConfig": (
        "voicehub.models.cosyvoice.configuration",
        "CosyVoiceConfig",
    ),
    "CosyVoiceForTextToSpeech": (
        "voicehub.models.cosyvoice.modeling",
        "CosyVoiceForTextToSpeech",
    ),
    "CosyVoiceTTS": (
        "voicehub.models.cosyvoice.modeling",
        "CosyVoiceTTS",
    ),
    "CosyVoiceTrainingAdapter": (
        "voicehub.models.cosyvoice.training",
        "CosyVoiceTrainingAdapter",
    ),
    "CosyVoiceTrainingCollator": (
        "voicehub.models.cosyvoice.training",
        "CosyVoiceTrainingCollator",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
