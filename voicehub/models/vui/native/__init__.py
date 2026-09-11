"""VoiceHub-native Vui architecture declaration."""

from voicehub.models.vui.native.registration import (
    DEFAULT_VUI_ALIASES,
    create_vui_architecture_spec,
    register_vui_architecture,
)

__all__ = [
    "DEFAULT_VUI_ALIASES",
    "create_vui_architecture_spec",
    "register_vui_architecture",
]
