"""ConversationTTS configuration and model exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ConversationTTS": ("voicehub.models.conversationtts.modeling"),
    "ConversationTTSConfig": ("voicehub.models.conversationtts.configuration"),
    "ConversationTTSForTextToSpeech": ("voicehub.models.conversationtts.modeling"),
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
