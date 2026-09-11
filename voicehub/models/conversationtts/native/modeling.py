"""Stable native model entry points for ConversationTTS."""

from voicehub.models.conversationtts.source.conversationtts.models import model_new as _model_new

ConversationTTSArchitectureConfig = _model_new.ModelArgs
ConversationTTSModel = _model_new.Model

__all__ = [
    "ConversationTTSArchitectureConfig",
    "ConversationTTSModel",
]
