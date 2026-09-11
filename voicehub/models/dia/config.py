"""Compatibility exports for Dia's native architecture configuration.

New code should import these classes from
``voicehub.models.dia.native.configuration``. The provider package keeps the
historical module path so existing applications can migrate without importing
Pydantic or executing upstream configuration code.
"""

from voicehub.models.dia.native.configuration import DiaArchitectureConfig, DiaDecoderConfig, DiaEncoderConfig

DiaConfig = DiaArchitectureConfig
ModelConfig = DiaArchitectureConfig
EncoderConfig = DiaEncoderConfig
DecoderConfig = DiaDecoderConfig

__all__ = [
    "DecoderConfig",
    "DiaArchitectureConfig",
    "DiaConfig",
    "DiaDecoderConfig",
    "DiaEncoderConfig",
    "EncoderConfig",
    "ModelConfig",
]
