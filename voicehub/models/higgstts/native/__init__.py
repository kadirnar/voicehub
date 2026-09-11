"""VoiceHub-owned Higgs Audio v2 architecture with lazy exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.higgstts.native."
_EXPORTS = {
    "HiggsAcousticCodecConfig": _PACKAGE + "tokenizer_configuration",
    "HiggsAudioV2Config": _PACKAGE + "configuration",
    "HiggsAudioV2ForConditionalGeneration": _PACKAGE + "modeling",
    "HiggsAudioV2GenerationOutput": _PACKAGE + "generation",
    "HiggsAudioV2Generator": _PACKAGE + "generation",
    "HiggsAudioV2Model": _PACKAGE + "modeling",
    "HiggsAudioV2Output": _PACKAGE + "modeling",
    "HiggsAudioV2Runtime": _PACKAGE + "runtime",
    "HiggsAudioV2Batch": _PACKAGE + "processing",
    "HiggsAudioV2Processor": _PACKAGE + "processing",
    "HiggsAudioV2TextTokenizer": _PACKAGE + "processing",
    "HiggsAudioV2TokenizerConfig": _PACKAGE + "tokenizer_configuration",
    "HiggsAudioV2TokenizerModel": _PACKAGE + "tokenizer",
    "HiggsAudioV2TokenizerOutput": _PACKAGE + "tokenizer",
    "load_higgs_audio_v2_runtime": _PACKAGE + "runtime",
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
