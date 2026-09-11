"""VoiceHub-owned Whisper architecture with lazy public components."""

from __future__ import annotations

import importlib
from typing import Any

_MODULE = "voicehub.models.asr_whisper_native.native."
_EXPORTS = {
    "DEFAULT_WHISPER_ALIASES": _MODULE + "registration",
    "Float32LayerNorm": _MODULE + "modeling",
    "HFWhisperCheckpointAdapter": _MODULE + "checkpoint",
    "HuggingFaceWhisperCheckpointAdapter": _MODULE + "checkpoint",
    "LANGUAGES": _MODULE + "tokenization",
    "Language": _MODULE + "decoding",
    "OPENAI_WHISPER_REVISION": _MODULE + "registration",
    "OpenAIWhisperCheckpointAdapter": _MODULE + "checkpoint",
    "Prompt": _MODULE + "decoding",
    "TRANSFORMERS_WHISPER_REVISION": _MODULE + "registration",
    "WhisperAttention": _MODULE + "modeling",
    "WhisperAttentionCache": _MODULE + "modeling",
    "WhisperArtifacts": _MODULE + "artifacts",
    "WhisperConfig": _MODULE + "configuration",
    "WhisperDecoder": _MODULE + "modeling",
    "WhisperDecoderCache": _MODULE + "modeling",
    "WhisperDecoderOutput": _MODULE + "modeling",
    "WhisperDecodingConfig": _MODULE + "decoding",
    "WhisperEncoder": _MODULE + "modeling",
    "WhisperForConditionalGeneration": _MODULE + "modeling",
    "WhisperGenerationAdapter": _MODULE + "decoding",
    "WhisperGenerationOutput": _MODULE + "decoding",
    "WhisperLayerCache": _MODULE + "modeling",
    "WhisperModel": _MODULE + "modeling",
    "WhisperSpecialTokens": _MODULE + "tokenization",
    "WhisperTokenizer": _MODULE + "tokenization",
    "WhisperTokenizerFormatError": _MODULE + "tokenization",
    "NativeWhisperCheckpointAdapter": _MODULE + "checkpoint",
    "WhisperOutput": _MODULE + "modeling",
    "WhisperTokenSet": _MODULE + "decoding",
    "WhisperTokenizerProtocol": _MODULE + "decoding",
    "apply_whisper_suppression": _MODULE + "decoding",
    "apply_whisper_timestamp_rules": _MODULE + "decoding",
    "build_openai_whisper_special_tokens": _MODULE + "tokenization",
    "create_whisper_architecture_spec": _MODULE + "registration",
    "huggingface_whisper_tensor_mapping": _MODULE + "checkpoint",
    "native_whisper_tensor_names": _MODULE + "checkpoint",
    "openai_whisper_tensor_mapping": _MODULE + "checkpoint",
    "register_whisper_architecture": _MODULE + "registration",
    "resolve_whisper_artifacts": _MODULE + "artifacts",
    "whisper_sinusoids": _MODULE + "modeling",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public components only when a caller requests one."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable results for interactive discovery."""
    return sorted((*globals(), *_EXPORTS))
