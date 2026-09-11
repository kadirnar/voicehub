"""VoiceHub-native Qwen3-TTS architecture with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.qwen3tts.native."
_EXPORTS = {
    "NativeQwen3TTSRuntime": _PACKAGE + "runtime",
    "Qwen3TTSArchitectureConfig": _PACKAGE + "configuration",
    "Qwen3TTSForConditionalGeneration": _PACKAGE + "modeling",
    "Qwen3TTSSpeechDecoder": _PACKAGE + "codec",
    "Qwen3TTSTextTokenizer": _PACKAGE + "tokenization",
    "Qwen3TTSTokenizerConfig": _PACKAGE + "configuration",
    "load_qwen3_tts_runtime": _PACKAGE + "runtime",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
