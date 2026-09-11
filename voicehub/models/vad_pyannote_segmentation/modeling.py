"""VoiceHub-native pyannote segmentation-3.0 VAD provider."""

from __future__ import annotations

from voicehub.models.vad_pyannote.modeling import PyannoteVADForVoiceActivityDetection

from .configuration import PyannoteSegmentationVADConfig


class PyannoteSegmentationVADForVoiceActivityDetection(PyannoteVADForVoiceActivityDetection):
    """Execute and fine-tune the pinned seven-class powerset graph."""

    config_class = PyannoteSegmentationVADConfig
    default_model_name_or_path = "pyannote/segmentation-3.0"
    native_variant = "powerset-segmentation"


__all__ = ["PyannoteSegmentationVADForVoiceActivityDetection"]
