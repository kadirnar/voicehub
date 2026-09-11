"""VoiceHub-native OmniVoice architecture with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.omnivoice.native."
_EXPORTS = {
    "HiggsAcousticConfig": _PACKAGE + "configuration",
    "HiggsAudioDecoderOutput": _PACKAGE + "codec",
    "HiggsAudioEncoderOutput": _PACKAGE + "codec",
    "HiggsAudioOutput": _PACKAGE + "codec",
    "HiggsAudioV2Config": _PACKAGE + "configuration",
    "HiggsAudioV2Tokenizer": _PACKAGE + "codec",
    "OmniVoiceArchitectureConfig": _PACKAGE + "configuration",
    "OmniVoiceBackboneOutput": _PACKAGE + "modeling",
    "OmniVoiceGenerationConfig": _PACKAGE + "generation",
    "OmniVoiceGenerator": _PACKAGE + "generation",
    "OmniVoiceMaskingConfig": _PACKAGE + "processing",
    "OmniVoiceModel": _PACKAGE + "modeling",
    "OmniVoiceModelOutput": _PACKAGE + "modeling",
    "OmniVoicePackingCollator": _PACKAGE + "processing",
    "OmniVoicePaddingCollator": _PACKAGE + "processing",
    "OmniVoicePrompt": _PACKAGE + "generation",
    "OmniVoiceQwen3Backbone": _PACKAGE + "modeling",
    "OmniVoiceSampleProcessor": _PACKAGE + "processing",
    "OmniVoiceTokenizer": _PACKAGE + "processing",
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
