"""VoiceHub-owned Inflect Micro/Nano v2 architecture."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.inflecttts.native."
_EXPORTS = {
    "INFLECT_MICRO_V2_CONFIG": _PACKAGE + "configuration",
    "INFLECT_NANO_V2_CONFIG": _PACKAGE + "configuration",
    "InflectLossWeights": _PACKAGE + "training",
    "InflectTTSArchitectureConfig": _PACKAGE + "configuration",
    "InflectV2Config": _PACKAGE + "configuration",
    "InflectV2Runtime": _PACKAGE + "runtime",
    "InflectV2TrainingModel": _PACKAGE + "training",
    "SynthesizerTrn": _PACKAGE + "modeling",
    "build_inflect_model": _PACKAGE + "modeling",
    "convert_inflect_legacy_checkpoint": _PACKAGE + "checkpoint",
    "export_inflect_checkpoint": _PACKAGE + "checkpoint",
    "load_inflect_checkpoint": _PACKAGE + "checkpoint",
    "phonemes_to_ids": _PACKAGE + "frontend",
    "resolve_inflect_artifacts": _PACKAGE + "checkpoint",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name == "InflectTTSArchitectureConfig":
        name = "InflectV2Config"
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
