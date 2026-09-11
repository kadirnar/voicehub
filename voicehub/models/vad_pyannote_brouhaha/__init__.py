"""Pyannote Brouhaha multi-task voice activity detection."""

from .configuration import PyannoteBrouhahaVADConfig
from .modeling import PyannoteBrouhahaVADForVoiceActivityDetection

__all__ = [
    "PyannoteBrouhahaVADConfig",
    "PyannoteBrouhahaVADForVoiceActivityDetection",
]
