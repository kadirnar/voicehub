"""VoiceHub-owned SpeechBrain CRDNN VAD with lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.vad_speechbrain.native."
_EXPORTS = {
    "DEFAULT_SPEECHBRAIN_VAD_ALIASES": _PACKAGE + "registration",
    "NATIVE_SPEECHBRAIN_VAD_FILENAME": _PACKAGE + "checkpoint",
    "NATIVE_SPEECHBRAIN_VAD_FORMAT": _PACKAGE + "checkpoint",
    "SpeechBrainCRDNNVADConfig": _PACKAGE + "configuration",
    "SpeechBrainCRDNNVADModel": _PACKAGE + "modeling",
    "SpeechBrainCRDNNVADOutput": _PACKAGE + "modeling",
    "SpeechBrainVADBoundary": _PACKAGE + "inference",
    "SpeechBrainVADFrontend": _PACKAGE + "frontend",
    "SpeechBrainVADInference": _PACKAGE + "inference",
    "SpeechBrainVADSafeTensorsCheckpointAdapter": _PACKAGE + "checkpoint",
    "convert_speechbrain_vad_checkpoint": _PACKAGE + "checkpoint",
    "create_speechbrain_vad_architecture_spec": _PACKAGE + "registration",
    "native_speechbrain_vad_tensor_shapes": _PACKAGE + "checkpoint",
    "register_speechbrain_vad_architecture": _PACKAGE + "registration",
    "speechbrain_vad_binary_cross_entropy": _PACKAGE + "objective",
    "speechbrain_mel_filterbank": _PACKAGE + "frontend",
    "speechbrain_source_tensor_mapping": _PACKAGE + "checkpoint",
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
