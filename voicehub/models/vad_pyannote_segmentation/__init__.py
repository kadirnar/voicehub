"""Pyannote segmentation-3.0 voice activity detection."""

from .configuration import PyannoteSegmentationVADConfig
from .modeling import PyannoteSegmentationVADForVoiceActivityDetection

__all__ = [
    "PyannoteSegmentationVADConfig",
    "PyannoteSegmentationVADForVoiceActivityDetection",
]
