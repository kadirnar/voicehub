"""VoiceHub-native Granite Speech architecture.

Only lightweight source metadata is imported from this package root.
Graph, processor, and checkpoint modules remain lazy through the
architecture catalogue.
"""

from voicehub.models.asr_granite_speech.native.metadata import (
    GRANITE_SPEECH_CHECKPOINTS,
    GRANITE_SPEECH_RELEASE_SOURCE_REVISION,
    GRANITE_SPEECH_SOURCE_REVISION,
)

__all__ = [
    "GRANITE_SPEECH_CHECKPOINTS",
    "GRANITE_SPEECH_RELEASE_SOURCE_REVISION",
    "GRANITE_SPEECH_SOURCE_REVISION",
]
