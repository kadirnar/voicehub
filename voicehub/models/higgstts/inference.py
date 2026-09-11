"""Backward-compatible imports for VoiceHub-native Higgs Audio v2."""

from voicehub.models.higgstts.configuration import HiggsTTSConfig
from voicehub.models.higgstts.modeling import HiggsTTS, HiggsTTSForTextToSpeech

__all__ = [
    "HiggsTTS",
    "HiggsTTSConfig",
    "HiggsTTSForTextToSpeech",
]
