"""VoiceHub-native ConversationTTS architecture with lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.conversationtts.native."
_EXPORTS = {
    "ConversationDecoder": _PACKAGE + "decoder",
    "ConversationDecoderLayer": _PACKAGE + "decoder",
    "ConversationKVCache": _PACKAGE + "decoder",
    "ConversationMultiHeadAttention": _PACKAGE + "decoder",
    "ConversationRMSNorm": _PACKAGE + "decoder",
    "ConversationTTSArchitectureConfig": _PACKAGE + "modeling",
    "ConversationTTSModel": _PACKAGE + "modeling",
    "ConversationTTSProtocol": _PACKAGE + "processing",
    "build_conversationtts_sequence": _PACKAGE + "processing",
    "build_llama32_decoder": _PACKAGE + "decoder",
    "collate_conversationtts_sequences": _PACKAGE + "processing",
    "create_conversationtts_architecture_spec": _PACKAGE + "registration",
    "export_conversationtts_checkpoint": _PACKAGE + "checkpoint",
    "load_conversationtts_checkpoint": _PACKAGE + "checkpoint",
    "register_conversationtts_architecture": _PACKAGE + "registration",
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
