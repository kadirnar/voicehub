"""Lazy declaration for VoiceHub's native generic ASR dispatcher."""

from voicehub.models.asr_transformers.native.registration import (
    DEFAULT_ASR_DISPATCH_ALIASES,
    create_asr_dispatch_architecture_spec,
    register_asr_dispatch_architecture,
)

__all__ = [
    "DEFAULT_ASR_DISPATCH_ALIASES",
    "create_asr_dispatch_architecture_spec",
    "register_asr_dispatch_architecture",
]
