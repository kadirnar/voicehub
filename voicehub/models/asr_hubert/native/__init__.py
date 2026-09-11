"""VoiceHub-owned HuBERT CTC architecture with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.asr_hubert.native."
_EXPORTS = {
    "DEFAULT_HUBERT_ALIASES": _PACKAGE + "registration",
    "FACEBOOK_HUBERT_LARGE_LS960_FT_HEADER_FINGERPRINT": _PACKAGE + "checkpoint",
    "FACEBOOK_HUBERT_LARGE_LS960_FT_REVISION": _PACKAGE + "checkpoint",
    "FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_REVISION": _PACKAGE + "checkpoint",
    "FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_SHA256": _PACKAGE + "checkpoint",
    "HFHubertCheckpointAdapter": _PACKAGE + "checkpoint",
    "HubertArtifacts": _PACKAGE + "artifacts",
    "HubertCTCOutput": _PACKAGE + "modeling",
    "HubertConfig": _PACKAGE + "configuration",
    "HubertFeatureProjection": _PACKAGE + "modeling",
    "HubertForCTC": _PACKAGE + "modeling",
    "HubertModel": _PACKAGE + "modeling",
    "HubertModelOutput": _PACKAGE + "modeling",
    "HuggingFaceHubertCheckpointAdapter": _PACKAGE + "checkpoint",
    "TRANSFORMERS_HUBERT_REVISION": _PACKAGE + "checkpoint",
    "create_hubert_architecture_spec": _PACKAGE + "registration",
    "huggingface_hubert_tensor_mapping": _PACKAGE + "checkpoint",
    "huggingface_hubert_tensor_shapes": _PACKAGE + "checkpoint",
    "native_hubert_tensor_names": _PACKAGE + "checkpoint",
    "native_hubert_tensor_shapes": _PACKAGE + "checkpoint",
    "register_hubert_architecture": _PACKAGE + "registration",
    "resolve_hubert_artifacts": _PACKAGE + "artifacts",
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
