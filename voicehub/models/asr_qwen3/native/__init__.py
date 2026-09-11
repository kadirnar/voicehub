"""VoiceHub-native Qwen3-ASR architecture.

Concrete graph modules remain lazy through the architecture catalogue.
This package root intentionally exports only lightweight source
metadata.
"""

from voicehub.models.asr_qwen3.native.metadata import QWEN3_ASR_CHECKPOINTS, QWEN3_ASR_SOURCE_REVISION

__all__ = [
    "QWEN3_ASR_CHECKPOINTS",
    "QWEN3_ASR_SOURCE_REVISION",
]
