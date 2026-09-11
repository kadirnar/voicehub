"""VoiceHub-native FunASR FSMN VAD architecture."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.vad_funasr.native."
_EXPORTS = {
    "DEFAULT_FSMN_VAD_ALIASES": _PACKAGE + "registration",
    "FSMNEncoder": _PACKAGE + "modeling",
    "FSMNVADBoundary": _PACKAGE + "inference",
    "FSMNVADConfig": _PACKAGE + "configuration",
    "FSMNVADDecoder": _PACKAGE + "inference",
    "FSMNVADFrontend": _PACKAGE + "frontend",
    "FSMNVADModel": _PACKAGE + "modeling",
    "FSMNVADOutput": _PACKAGE + "modeling",
    "FSMNVADSafeTensorsCheckpointAdapter": _PACKAGE + "checkpoint",
    "FUNASR_CMVN_SHA256": _PACKAGE + "metadata",
    "FUNASR_HF_REPOSITORY": _PACKAGE + "metadata",
    "FUNASR_HF_REVISION": _PACKAGE + "metadata",
    "FUNASR_MODEL_SHA256": _PACKAGE + "metadata",
    "FUNASR_SOURCE_REVISION": _PACKAGE + "metadata",
    "convert_funasr_fsmn_checkpoint": _PACKAGE + "checkpoint",
    "create_fsmn_vad_architecture_spec": _PACKAGE + "registration",
    "frame_decibels": _PACKAGE + "inference",
    "fsmn_vad_loss": _PACKAGE + "objective",
    "register_fsmn_vad_architecture": _PACKAGE + "registration",
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
