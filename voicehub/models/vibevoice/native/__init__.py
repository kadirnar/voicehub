"""Lazy public namespace for native Microsoft VibeVoice architectures."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.vibevoice.native."
_EXPORTS = {
    "VIBEVOICE_TOKEN_IDS": _PACKAGE + "tokenization",
    "VibeVoiceASRConfig": _PACKAGE + "configuration",
    "VibeVoiceASRForConditionalGeneration": _PACKAGE + "modeling",
    "VibeVoiceASRProcessor": _PACKAGE + "processing",
    "VibeVoiceASRTokenizerConfig": _PACKAGE + "configuration",
    "VibeVoiceArtifacts": _PACKAGE + "artifacts",
    "VibeVoiceAudioProcessor": _PACKAGE + "processing",
    "VibeVoiceCheckpointAdapter": _PACKAGE + "checkpoint",
    "VibeVoiceCheckpointInventory": _PACKAGE + "checkpoint",
    "VibeVoiceDiffusionConfig": _PACKAGE + "configuration",
    "VibeVoiceForConditionalGeneration": _PACKAGE + "modeling",
    "VibeVoiceLegacyTokenizerConfig": _PACKAGE + "configuration",
    "VibeVoiceRealtimeForConditionalGeneration": _PACKAGE + "modeling",
    "VibeVoiceRuntime": _PACKAGE + "runtime",
    "VibeVoiceTTSConfig": _PACKAGE + "configuration",
    "VibeVoiceTTSProcessor": _PACKAGE + "processing",
    "VibeVoiceTokenizer": _PACKAGE + "tokenization",
    "build_vibevoice_model": _PACKAGE + "checkpoint",
    "export_vibevoice_checkpoint": _PACKAGE + "checkpoint",
    "inspect_vibevoice_checkpoint": _PACKAGE + "checkpoint",
    "load_vibevoice_runtime": _PACKAGE + "runtime",
    "native_vibevoice_tensor_names": _PACKAGE + "checkpoint",
    "native_vibevoice_tensor_shapes": _PACKAGE + "checkpoint",
    "parse_vibevoice_config": _PACKAGE + "configuration",
    "parse_vibevoice_script": _PACKAGE + "processing",
    "render_vibevoice_asr_prompt": _PACKAGE + "processing",
    "resolve_vibevoice_artifacts": _PACKAGE + "artifacts",
    "resolve_vibevoice_dtype": _PACKAGE + "runtime",
    "save_vibevoice_runtime": _PACKAGE + "runtime",
    "validate_published_vibevoice_inventory": _PACKAGE + "checkpoint",
    "validate_vibevoice_training_record": _PACKAGE + "processing",
}

__all__ = [
    "VIBEVOICE_TOKEN_IDS",
    "VibeVoiceASRConfig",
    "VibeVoiceASRForConditionalGeneration",
    "VibeVoiceASRProcessor",
    "VibeVoiceASRTokenizerConfig",
    "VibeVoiceArtifacts",
    "VibeVoiceAudioProcessor",
    "VibeVoiceCheckpointAdapter",
    "VibeVoiceCheckpointInventory",
    "VibeVoiceDiffusionConfig",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceLegacyTokenizerConfig",
    "VibeVoiceRealtimeForConditionalGeneration",
    "VibeVoiceRuntime",
    "VibeVoiceTTSConfig",
    "VibeVoiceTTSProcessor",
    "VibeVoiceTokenizer",
    "build_vibevoice_model",
    "export_vibevoice_checkpoint",
    "inspect_vibevoice_checkpoint",
    "load_vibevoice_runtime",
    "native_vibevoice_tensor_names",
    "native_vibevoice_tensor_shapes",
    "parse_vibevoice_config",
    "parse_vibevoice_script",
    "render_vibevoice_asr_prompt",
    "resolve_vibevoice_artifacts",
    "resolve_vibevoice_dtype",
    "save_vibevoice_runtime",
    "validate_published_vibevoice_inventory",
    "validate_vibevoice_training_record",
]


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
