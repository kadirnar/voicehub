"""VoiceHub-owned Moonshine ASR architecture with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.asr_moonshine.native."
_EXPORTS = {
    "DEFAULT_MOONSHINE_ALIASES": _PACKAGE + "registration",
    "HFMoonshineCheckpointAdapter": _PACKAGE + "checkpoint",
    "HuggingFaceMoonshineCheckpointAdapter": _PACKAGE + "checkpoint",
    "MOONSHINE_MAIN_LIBRARY_REVISION": _PACKAGE + "configuration",
    "MoonshineArtifacts": _PACKAGE + "artifacts",
    "MoonshineAttention": _PACKAGE + "modeling",
    "MoonshineConfig": _PACKAGE + "configuration",
    "MoonshineDecoder": _PACKAGE + "modeling",
    "MoonshineDecoderLayer": _PACKAGE + "modeling",
    "MoonshineEncoder": _PACKAGE + "modeling",
    "MoonshineEncoderLayer": _PACKAGE + "modeling",
    "MoonshineEncoderOutput": _PACKAGE + "modeling",
    "MoonshineForConditionalGeneration": _PACKAGE + "modeling",
    "MoonshineModel": _PACKAGE + "modeling",
    "MoonshineModelOutput": _PACKAGE + "modeling",
    "MoonshineProcessor": _PACKAGE + "processing",
    "MoonshineSeq2SeqLMOutput": _PACKAGE + "modeling",
    "TRANSFORMERS_MOONSHINE_REVISION": _PACKAGE + "configuration",
    "USEFULSENSORS_MOONSHINE_BASE_FILE_BYTES": _PACKAGE + "checkpoint",
    "USEFULSENSORS_MOONSHINE_BASE_HEADER_FINGERPRINT": (_PACKAGE + "checkpoint"),
    "USEFULSENSORS_MOONSHINE_BASE_REVISION": _PACKAGE + "checkpoint",
    "USEFULSENSORS_MOONSHINE_TINY_FILE_BYTES": _PACKAGE + "checkpoint",
    "USEFULSENSORS_MOONSHINE_TINY_HEADER_FINGERPRINT": (_PACKAGE + "checkpoint"),
    "USEFULSENSORS_MOONSHINE_TINY_REVISION": _PACKAGE + "checkpoint",
    "create_moonshine_architecture_spec": _PACKAGE + "registration",
    "huggingface_moonshine_tensor_mapping": _PACKAGE + "checkpoint",
    "huggingface_moonshine_tensor_shapes": _PACKAGE + "checkpoint",
    "native_moonshine_tensor_names": _PACKAGE + "checkpoint",
    "native_moonshine_tensor_shapes": _PACKAGE + "checkpoint",
    "register_moonshine_architecture": _PACKAGE + "registration",
    "resolve_moonshine_artifacts": _PACKAGE + "artifacts",
    "safetensors_header_fingerprint": _PACKAGE + "checkpoint",
    "shift_tokens_right": _PACKAGE + "modeling",
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
