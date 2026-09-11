"""VoiceHub-native SpeechT5 architecture registration with lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.speecht5.native."
_EXPORTS = {
    "DEFAULT_SPEECHT5_ALIASES": _PACKAGE + "registration",
    "NATIVE_SPEECHT5_FORMAT": _PACKAGE + "metadata",
    "SPEECHT5_HIFIGAN_CONFIG_SHA256": _PACKAGE + "metadata",
    "SPEECHT5_HIFIGAN_REFERENCE_INVENTORY": _PACKAGE + "metadata",
    "SPEECHT5_PROCESSOR_INTEGRITY": _PACKAGE + "metadata",
    "SPEECHT5_REFERENCE_INVENTORY": _PACKAGE + "metadata",
    "SPEECHT5_SOURCE_FILES": _PACKAGE + "metadata",
    "SPEECHT5_SOURCE_LICENSE": _PACKAGE + "metadata",
    "SPEECHT5_SOURCE_REPOSITORY": _PACKAGE + "metadata",
    "SPEECHT5_SOURCE_REVISION": _PACKAGE + "metadata",
    "SPEECHT5_SOURCE_TAG": _PACKAGE + "metadata",
    "create_speecht5_architecture_spec": _PACKAGE + "registration",
    "register_speecht5_architecture": _PACKAGE + "registration",
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
