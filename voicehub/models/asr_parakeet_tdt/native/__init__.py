"""VoiceHub-native Parakeet TDT architecture with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.asr_parakeet_tdt.native."
_EXPORTS = {
    "ParakeetEncoderConfig": _PACKAGE + "configuration",
    "ParakeetForTDT": _PACKAGE + "modeling",
    "ParakeetTDTConfig": _PACKAGE + "configuration",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
