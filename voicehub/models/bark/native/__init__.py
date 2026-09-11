"""VoiceHub-native Bark with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.bark.native."
_EXPORTS = {
    "BarkArchitectureConfig": _PACKAGE + "configuration",
    "BarkCoarseConfig": _PACKAGE + "configuration",
    "BarkCoarseGenerationConfig": _PACKAGE + "configuration",
    "BarkFineConfig": _PACKAGE + "configuration",
    "BarkFineGenerationConfig": _PACKAGE + "configuration",
    "BarkGenerationConfig": _PACKAGE + "configuration",
    "BarkSemanticConfig": _PACKAGE + "configuration",
    "BarkSemanticGenerationConfig": _PACKAGE + "configuration",
    "BarkModel": _PACKAGE + "modeling",
    "BarkProcessor": _PACKAGE + "processing",
    "BarkTrainingAdapter": _PACKAGE + "training",
    "BarkTrainingModel": _PACKAGE + "training",
    "BarkWordPieceTokenizer": _PACKAGE + "processing",
    "DEFAULT_BARK_ALIASES": _PACKAGE + "registration",
    "convert_official_bark_checkpoint": _PACKAGE + "checkpoint",
    "create_bark_architecture_spec": _PACKAGE + "registration",
    "load_bark_model_from_safetensors": _PACKAGE + "checkpoint",
    "load_bark_safetensors": _PACKAGE + "checkpoint",
    "register_bark_architecture": _PACKAGE + "registration",
    "resolve_bark_artifacts": _PACKAGE + "artifacts",
    "save_bark_safetensors": _PACKAGE + "checkpoint",
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
