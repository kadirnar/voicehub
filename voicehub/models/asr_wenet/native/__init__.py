"""VoiceHub-native WeNet GigaSpeech U2++ architecture."""

from voicehub.models.asr_wenet.native.configuration import WeNetU2PPConfig
from voicehub.models.asr_wenet.native.tokenization import WeNetGigaSpeechTokenizer

__all__ = [
    "WeNetGigaSpeechTokenizer",
    "WeNetU2PPConfig",
]
