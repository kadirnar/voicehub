"""Lazy exports for the native SeamlessM4T-v2 S2T architecture."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.asr_seamless_m4t_v2.native."
_EXPORTS = {
    "SeamlessM4Tv2FeatureExtractor": _PACKAGE + "frontend",
    "SeamlessM4Tv2ForSpeechToText": _PACKAGE + "modeling",
    "SeamlessM4Tv2Processor": _PACKAGE + "processing",
    "SeamlessM4Tv2S2TCheckpointAdapter": _PACKAGE + "checkpoint",
    "SeamlessM4Tv2S2TConfig": _PACKAGE + "configuration",
    "SeamlessM4Tv2S2TOutput": _PACKAGE + "modeling",
    "SeamlessM4Tv2S2TRuntime": _PACKAGE + "runtime",
    "SeamlessM4Tv2Tokenizer": _PACKAGE + "tokenization",
    "create_seamless_m4t_v2_architecture_spec": _PACKAGE + "registration",
    "load_seamless_m4t_v2_runtime": _PACKAGE + "runtime",
    "register_seamless_m4t_v2_architecture": _PACKAGE + "registration",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
