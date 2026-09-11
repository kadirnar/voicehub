"""VoiceHub-native Meta Encodec architecture declaration."""

from voicehub.models.encodec.native.registration import (
    DEFAULT_ENCODEC_ALIASES,
    create_encodec_architecture_spec,
    register_encodec_architecture,
)

__all__ = [
    "DEFAULT_ENCODEC_ALIASES",
    "create_encodec_architecture_spec",
    "register_encodec_architecture",
]
