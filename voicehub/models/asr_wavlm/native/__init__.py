"""VoiceHub-owned WavLM CTC architecture with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.asr_wavlm.native."
_EXPORTS = {
    "DEFAULT_WAVLM_ALIASES": _PACKAGE + "registration",
    "HFWavLMCheckpointAdapter": _PACKAGE + "checkpoint",
    "HuggingFaceWavLMCheckpointAdapter": _PACKAGE + "checkpoint",
    "MICROSOFT_WAVLM_SOURCE_REVISION": _PACKAGE + "checkpoint",
    "TRANSFORMERS_WAVLM_REVISION": _PACKAGE + "checkpoint",
    "WAVLM_BASE_PLUS_CTC_HEADER_FINGERPRINT": _PACKAGE + "checkpoint",
    "WAVLM_BASE_PLUS_CTC_REVISION": _PACKAGE + "checkpoint",
    "WAVLM_BASE_PLUS_CTC_SAFETENSORS_REVISION": _PACKAGE + "checkpoint",
    "WAVLM_BASE_PLUS_CTC_SAFETENSORS_SHA256": _PACKAGE + "checkpoint",
    "WavLMArtifacts": _PACKAGE + "artifacts",
    "WavLMAttention": _PACKAGE + "modeling",
    "WavLMCTCOutput": _PACKAGE + "modeling",
    "WavLMConfig": _PACKAGE + "configuration",
    "WavLMEncoder": _PACKAGE + "modeling",
    "WavLMEncoderLayer": _PACKAGE + "modeling",
    "WavLMEncoderLayerStableLayerNorm": _PACKAGE + "modeling",
    "WavLMEncoderOutput": _PACKAGE + "modeling",
    "WavLMForCTC": _PACKAGE + "modeling",
    "WavLMModel": _PACKAGE + "modeling",
    "WavLMModelOutput": _PACKAGE + "modeling",
    "create_wavlm_architecture_spec": _PACKAGE + "registration",
    "huggingface_wavlm_tensor_mapping": _PACKAGE + "checkpoint",
    "huggingface_wavlm_tensor_shapes": _PACKAGE + "checkpoint",
    "native_wavlm_tensor_names": _PACKAGE + "checkpoint",
    "native_wavlm_tensor_shapes": _PACKAGE + "checkpoint",
    "register_wavlm_architecture": _PACKAGE + "registration",
    "resolve_wavlm_artifacts": _PACKAGE + "artifacts",
    "safetensors_header_fingerprint": _PACKAGE + "checkpoint",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public components only when explicitly requested."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
