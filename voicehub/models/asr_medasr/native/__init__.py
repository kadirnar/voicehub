"""VoiceHub-native LASR/MedASR architecture with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.asr_medasr.native."
_EXPORTS = {
    "MEDASR_ASSET_GIT_OIDS": _PACKAGE + "metadata",
    "MEDASR_CHECKPOINT": _PACKAGE + "metadata",
    "MEDASR_MODEL_ID": _PACKAGE + "metadata",
    "MEDASR_MODEL_REVISION": _PACKAGE + "metadata",
    "MEDASR_RECIPE_REVISION": _PACKAGE + "metadata",
    "MEDASR_SOURCE_REVISION": _PACKAGE + "metadata",
    "MedASRArtifacts": _PACKAGE + "artifacts",
    "MedASRCheckpointAdapter": _PACKAGE + "checkpoint",
    "MedASRConfig": _PACKAGE + "configuration",
    "MedASRDecodedText": _PACKAGE + "tokenization",
    "MedASRFeatureExtractor": _PACKAGE + "frontend",
    "MedASRForCTC": _PACKAGE + "modeling",
    "MedASRProcessor": _PACKAGE + "processing",
    "MedASRTokenizer": _PACKAGE + "tokenization",
    "medasr_header_fingerprint": _PACKAGE + "checkpoint",
    "medasr_mel_filter_bank": _PACKAGE + "frontend",
    "native_medasr_tensor_dtypes": _PACKAGE + "checkpoint",
    "native_medasr_tensor_shapes": _PACKAGE + "checkpoint",
    "resolve_medasr_artifacts": _PACKAGE + "artifacts",
    "validate_published_medasr_inventory": _PACKAGE + "checkpoint",
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
