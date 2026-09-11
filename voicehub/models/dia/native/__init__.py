"""VoiceHub-native Dia architecture with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.dia.native."
_EXPORTS = {
    "DEFAULT_DIA_ALIASES": _PACKAGE + "registration",
    "DiaArchitectureConfig": _PACKAGE + "configuration",
    "DiaArtifacts": _PACKAGE + "artifacts",
    "DiaBatch": _PACKAGE + "processing",
    "DiaByteTokenizer": _PACKAGE + "processing",
    "DiaConditionalGenerationOutput": _PACKAGE + "modeling",
    "DiaConfig": _PACKAGE + "configuration",
    "DiaDecoderConfig": _PACKAGE + "configuration",
    "DiaEncoderConfig": _PACKAGE + "configuration",
    "DiaForConditionalGeneration": _PACKAGE + "modeling",
    "DiaModel": _PACKAGE + "modeling",
    "DiaProcessor": _PACKAGE + "processing",
    "DiaRuntime": _PACKAGE + "runtime",
    "HFDiaCheckpointAdapter": _PACKAGE + "checkpoint",
    "HuggingFaceDiaCheckpointAdapter": _PACKAGE + "checkpoint",
    "NARI_DIA_CHECKPOINT_REVISION": _PACKAGE + "metadata",
    "NARI_DIA_HEADER_FINGERPRINT": _PACKAGE + "metadata",
    "NARI_DIA_SOURCE_REVISION": _PACKAGE + "metadata",
    "TRANSFORMERS_DIA_SOURCE_REVISION": _PACKAGE + "metadata",
    "create_dia_architecture_spec": _PACKAGE + "registration",
    "dia_header_fingerprint": _PACKAGE + "checkpoint",
    "load_dia_runtime": _PACKAGE + "runtime",
    "native_dia_tensor_names": _PACKAGE + "checkpoint",
    "native_dia_tensor_shapes": _PACKAGE + "checkpoint",
    "register_dia_architecture": _PACKAGE + "registration",
    "resolve_dia_artifacts": _PACKAGE + "artifacts",
    "resolve_dia_dtype": _PACKAGE + "runtime",
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
