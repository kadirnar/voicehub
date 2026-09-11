"""VoiceHub-owned Silero VAD architecture with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.vad_silero.native."
_EXPORTS = {
    "DEFAULT_SILERO_VAD_ALIASES": _PACKAGE + "registration",
    "OFFICIAL_SILERO_VAD_16K_HEADER_FINGERPRINT": _PACKAGE + "checkpoint",
    "OFFICIAL_SILERO_VAD_REVISION": _PACKAGE + "registration",
    "OFFICIAL_SILERO_VAD_VERSION": _PACKAGE + "registration",
    "OfficialSileroVADCheckpointAdapter": _PACKAGE + "checkpoint",
    "OfficialSileroVADSafeTensorsCheckpointAdapter": _PACKAGE + "checkpoint",
    "OfficialSileroVADTorchScriptCheckpointAdapter": _PACKAGE + "checkpoint",
    "SileroVADAudioOutput": _PACKAGE + "modeling",
    "SileroVADBinaryCrossEntropyLoss": _PACKAGE + "objective",
    "SileroVADConfig": _PACKAGE + "configuration",
    "SileroVADFrameOutput": _PACKAGE + "modeling",
    "SileroVADModel": _PACKAGE + "modeling",
    "SileroVADSegmentationConfig": _PACKAGE + "segmentation",
    "SileroVADSegmenter": _PACKAGE + "segmentation",
    "SileroVADState": _PACKAGE + "modeling",
    "SileroVADStream": _PACKAGE + "modeling",
    "SpeechSegment": _PACKAGE + "segmentation",
    "create_silero_vad_architecture_spec": _PACKAGE + "registration",
    "native_silero_vad_tensor_names": _PACKAGE + "checkpoint",
    "native_silero_vad_tensor_shapes": _PACKAGE + "checkpoint",
    "official_safetensors_tensor_mapping": _PACKAGE + "checkpoint",
    "official_torchscript_tensor_mapping": _PACKAGE + "checkpoint",
    "register_silero_vad_architecture": _PACKAGE + "registration",
    "segment_speech_probabilities": _PACKAGE + "segmentation",
    "silero_vad_binary_cross_entropy": _PACKAGE + "objective",
    "tensor_inventory_fingerprint": _PACKAGE + "checkpoint",
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
