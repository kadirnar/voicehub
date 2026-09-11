"""VoiceHub-native MeloTTS architecture with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.melotts.native."
_EXPORTS = {
    "MeloTTSArchitectureConfig": _PACKAGE + "configuration",
    "MeloTTSArtifacts": _PACKAGE + "artifacts",
    "MeloTTSCheckpointReport": _PACKAGE + "checkpoint",
    "MeloTTSDataConfig": _PACKAGE + "configuration",
    "MeloTTSFeatureBatch": _PACKAGE + "frontend",
    "MeloTTSLossWeights": _PACKAGE + "training",
    "MeloTTSModelConfig": _PACKAGE + "configuration",
    "MeloTTSRuntime": _PACKAGE + "runtime",
    "MeloTTSTrainingCollator": _PACKAGE + "training",
    "MeloTTSTrainingModel": _PACKAGE + "training",
    "NativeMeloTTSFrontend": _PACKAGE + "frontend",
    "build_melotts_model": _PACKAGE + "modeling",
    "convert_legacy_melotts_checkpoint": _PACKAGE + "checkpoint",
    "export_melotts_checkpoint": _PACKAGE + "checkpoint",
    "load_melotts_checkpoint": _PACKAGE + "checkpoint",
    "load_melotts_config": _PACKAGE + "configuration",
    "resolve_melotts_artifacts": _PACKAGE + "artifacts",
    "save_melotts_pretrained": _PACKAGE + "checkpoint",
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
