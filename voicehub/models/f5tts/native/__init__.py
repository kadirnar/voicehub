"""VoiceHub-owned F5-TTS architecture with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.f5tts.native."
_EXPORTS = {
    "F5ConditionalFlowMatcher": _PACKAGE + "modeling",
    "F5DiT": _PACKAGE + "modeling",
    "F5TTSArchitectureConfig": _PACKAGE + "configuration",
    "build_f5tts_model": _PACKAGE + "modeling",
    "f5tts_architecture_config": _PACKAGE + "configuration",
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
