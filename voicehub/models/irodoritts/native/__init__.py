"""VoiceHub-owned Irodori-TTS architecture with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.irodoritts.native."
_EXPORTS = {
    "IRODORI_CHECKPOINTS": _PACKAGE + "metadata",
    "IRODORI_CODEC_CHECKPOINT": _PACKAGE + "metadata",
    "IRODORI_CODEC_ID": _PACKAGE + "metadata",
    "IRODORI_CODEC_REVISION": _PACKAGE + "metadata",
    "IRODORI_SOURCE_REVISION": _PACKAGE + "metadata",
    "IRODORI_TOKENIZER_ID": _PACKAGE + "metadata",
    "IRODORI_TOKENIZER_REVISION": _PACKAGE + "metadata",
    "InferenceRuntime": _PACKAGE + "runtime",
    "IrodoriBatchProcessor": _PACKAGE + "training",
    "IrodoriCheckpointAdapter": _PACKAGE + "checkpoint",
    "IrodoriDACVAECodec": _PACKAGE + "codec",
    "IrodoriModelConfig": _PACKAGE + "configuration",
    "IrodoriTokenizer": _PACKAGE + "tokenization",
    "RuntimeKey": _PACKAGE + "runtime",
    "SamplingRequest": _PACKAGE + "runtime",
    "SamplingResult": _PACKAGE + "runtime",
    "TextToLatentRFDiT": _PACKAGE + "modeling",
    "irodori_header_fingerprint": _PACKAGE + "checkpoint",
    "irodori_training_step": _PACKAGE + "training",
    "load_irodori_safetensors": _PACKAGE + "checkpoint",
    "native_irodori_tensor_shapes": _PACKAGE + "checkpoint",
    "save_irodori_safetensors": _PACKAGE + "checkpoint",
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
