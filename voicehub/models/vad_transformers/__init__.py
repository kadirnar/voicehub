"""Native Wav2Vec2 audio/frame classification VAD integration."""

from voicehub.models.vad_transformers.configuration import TransformersVADConfig
from voicehub.models.vad_transformers.modeling import TransformersVADForVoiceActivityDetection

__all__ = [
    "TransformersVADConfig",
    "TransformersVADForVoiceActivityDetection",
]
