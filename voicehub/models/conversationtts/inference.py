"""Backward-compatible ConversationTTS imports."""

from voicehub.models.conversationtts.configuration import ConversationTTSConfig
from voicehub.models.conversationtts.modeling import ConversationTTS, ConversationTTSForTextToSpeech

__all__ = [
    "ConversationTTS",
    "ConversationTTSConfig",
    "ConversationTTSForTextToSpeech",
]
