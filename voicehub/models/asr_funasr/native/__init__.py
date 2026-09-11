"""VoiceHub-native SenseVoiceSmall architecture with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.asr_funasr.native."
_EXPORTS = {
    "SenseVoiceSmallConfig": _PACKAGE + "configuration",
    "SenseVoiceSmallForCTC": _PACKAGE + "modeling",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve the native graph only when a caller explicitly requests it."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}", ) from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
