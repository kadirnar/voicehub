"""VoiceHub-native StyleTTS 2 architecture with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.styletts2.native."
_EXPORTS = {
    "NativeStyleTTS2Frontend": _PACKAGE + "frontend",
    "StyleTTS2ArchitectureConfig": _PACKAGE + "configuration",
    "StyleTTS2CheckpointReport": _PACKAGE + "checkpoint",
    "StyleTTS2LossWeights": _PACKAGE + "training",
    "StyleTTS2MelSpectrogram": _PACKAGE + "frontend",
    "StyleTTS2Runtime": _PACKAGE + "runtime",
    "StyleTTS2TrainingModel": _PACKAGE + "training",
    "build_styletts2_model": _PACKAGE + "modeling",
    "convert_legacy_styletts2_checkpoint": _PACKAGE + "checkpoint",
    "export_styletts2_checkpoint": _PACKAGE + "checkpoint",
    "load_styletts2_checkpoint": _PACKAGE + "checkpoint",
    "save_styletts2_pretrained": _PACKAGE + "checkpoint",
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
