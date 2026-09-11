"""Lazy public exports for VoiceHub-native XTTS v2."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "XTTS": ("voicehub.models.xtts.modeling", "XTTS"),
    "XTTSConfig": (
        "voicehub.models.xtts.configuration",
        "XTTSConfig",
    ),
    "XTTSForTextToSpeech": (
        "voicehub.models.xtts.modeling",
        "XTTSForTextToSpeech",
    ),
    "XTTSTrainingAdapter": (
        "voicehub.models.xtts.training",
        "XTTSTrainingAdapter",
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
