"""Lazy declaration for VoiceHub's native Echo rectified-flow graph."""

from voicehub.models.echo.native.registration import (
    DEFAULT_ECHO_ALIASES,
    ECHO_SOURCE_REVISION,
    create_echo_architecture_spec,
    register_echo_architecture,
)

__all__ = [
    "DEFAULT_ECHO_ALIASES",
    "ECHO_SOURCE_REVISION",
    "create_echo_architecture_spec",
    "register_echo_architecture",
]
