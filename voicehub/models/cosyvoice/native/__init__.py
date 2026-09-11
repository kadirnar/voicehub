"""VoiceHub-owned CosyVoice 3 architecture with lazy exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.cosyvoice.native."
_EXPORTS = {
    "CosyVoiceArchitectureConfig": _PACKAGE + "configuration",
    "CosyVoiceFlowConfig": _PACKAGE + "configuration",
    "CosyVoiceHiFTConfig": _PACKAGE + "configuration",
    "CosyVoiceLanguageConfig": _PACKAGE + "configuration",
    "CosyVoiceFlowMatchingModel": _PACKAGE + "flow",
    "CosyVoiceLanguageModel": _PACKAGE + "language_model",
    "CosyVoiceNativeModel": _PACKAGE + "modeling",
    "CosyVoiceNativeRuntime": _PACKAGE + "runtime",
    "CosyVoiceSpeechTokenizer": _PACKAGE + "speech_tokenizer",
    "CosyVoiceSpeechTokenizerConfig": _PACKAGE + "speech_tokenizer",
    "CosyVoiceSpeechTokenizerEncoder": _PACKAGE + "speech_tokenizer",
    "CosyVoiceHiFTGenerator": _PACKAGE + "vocoder",
    "CosyVoiceHiFTTrainingModel": _PACKAGE + "vocoder",
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
