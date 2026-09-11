"""VoiceHub-native OpenVoice V2 architecture with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.openvoice.native."
_EXPORTS = {
    "OpenVoiceAudioProcessor": _PACKAGE + "processing",
    "OpenVoiceArtifacts": _PACKAGE + "artifacts",
    "OpenVoiceConverterConfig": _PACKAGE + "configuration",
    "OpenVoiceConverterOutput": _PACKAGE + "modeling",
    "OpenVoiceRuntime": _PACKAGE + "runtime",
    "OpenVoiceSpectrogramBatch": _PACKAGE + "processing",
    "OpenVoiceToneColorConverter": _PACKAGE + "modeling",
    "OpenVoiceWaveformBatch": _PACKAGE + "processing",
    "load_openvoice_runtime": _PACKAGE + "runtime",
    "resolve_openvoice_artifacts": _PACKAGE + "artifacts",
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
