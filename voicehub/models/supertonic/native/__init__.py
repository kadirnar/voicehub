"""VoiceHub-native Supertonic architecture with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.supertonic.native."
_EXPORTS = {
    "DEFAULT_SUPERTONIC_ALIASES": _PACKAGE + "registration",
    "NativeSupertonicRuntime": _PACKAGE + "runtime",
    "SupertonicArchitectureConfig": _PACKAGE + "configuration",
    "SupertonicArtifacts": _PACKAGE + "artifacts",
    "SupertonicFineTuningOutput": _PACKAGE + "runtime",
    "SupertonicStyle": _PACKAGE + "frontend",
    "SupertonicUnicodeProcessor": _PACKAGE + "frontend",
    "chunk_text": _PACKAGE + "runtime",
    "create_supertonic_architecture_spec": _PACKAGE + "registration",
    "load_native_supertonic_runtime": _PACKAGE + "runtime",
    "load_supertonic_native_weights": _PACKAGE + "checkpoint",
    "register_supertonic_architecture": _PACKAGE + "registration",
    "resolve_supertonic_artifacts": _PACKAGE + "artifacts",
    "resolve_supertonic_style": _PACKAGE + "artifacts",
    "save_supertonic_native_weights": _PACKAGE + "checkpoint",
}

__all__ = sorted(_EXPORTS)


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
