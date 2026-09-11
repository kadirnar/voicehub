"""Public, dependency-light imports for VoiceHub-native OmniVoice."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.omnivoice."
_EXPORTS = {
    "OmniVoiceConfig": _PACKAGE + "configuration",
    "OmniVoiceForTextToSpeech": _PACKAGE + "modeling",
    "OmniVoiceTTS": _PACKAGE + "modeling",
    "OmniVoiceTrainingAdapter": _PACKAGE + "training",
    "OmniVoiceTrainingCollator": _PACKAGE + "training",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
