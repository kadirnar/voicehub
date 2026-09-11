"""Compatibility exports for VoiceHub's native Dia layers."""

from voicehub.models.dia.native.modeling import (
    DiaCrossAttention,
    DiaDecoder,
    DiaDecoderLayer,
    DiaEncoder,
    DiaEncoderLayer,
    DiaMLP,
    DiaModel,
    DiaMultiChannelEmbedding,
    DiaRMSNorm,
    DiaRotaryEmbedding,
    DiaSelfAttention,
)

CrossAttention = DiaCrossAttention
Decoder = DiaDecoder
DecoderLayer = DiaDecoderLayer
Encoder = DiaEncoder
EncoderLayer = DiaEncoderLayer
MlpBlock = DiaMLP
RotaryEmbedding = DiaRotaryEmbedding
SelfAttention = DiaSelfAttention

__all__ = [
    "CrossAttention",
    "Decoder",
    "DecoderLayer",
    "DiaCrossAttention",
    "DiaDecoder",
    "DiaDecoderLayer",
    "DiaEncoder",
    "DiaEncoderLayer",
    "DiaMLP",
    "DiaModel",
    "DiaMultiChannelEmbedding",
    "DiaRMSNorm",
    "DiaRotaryEmbedding",
    "DiaSelfAttention",
    "Encoder",
    "EncoderLayer",
    "MlpBlock",
    "RotaryEmbedding",
    "SelfAttention",
]
