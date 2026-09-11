"""Validated configuration for VoiceHub's native Sesame CSM graph."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return value


def _positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"`{name}` must be finite and greater than zero.")
    return result


@dataclass(frozen=True, slots=True)
class CSMTransformerConfig:
    """One Llama-3.2-style decoder used by CSM."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_sequence_length: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500_000.0
    rope_scale_factor: float = 32.0
    rope_low_frequency_factor: float = 1.0
    rope_high_frequency_factor: float = 4.0
    rope_original_context_length: int = 8_192
    attention_dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "max_sequence_length",
                "rope_original_context_length",
        ):
            _positive_integer(name, getattr(self, name))
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("`num_key_value_heads` must divide `num_attention_heads`.")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("CSM attention head dimensions must be even.")
        for name in (
                "rms_norm_eps",
                "rope_theta",
                "rope_scale_factor",
                "rope_low_frequency_factor",
                "rope_high_frequency_factor",
        ):
            object.__setattr__(
                self,
                name,
                _positive_float(name, getattr(self, name)),
            )
        dropout = float(self.attention_dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("`attention_dropout` must be in [0, 1).")
        object.__setattr__(self, "attention_dropout", dropout)
        if (self.rope_high_frequency_factor <= self.rope_low_frequency_factor):
            raise ValueError("`rope_high_frequency_factor` must exceed "
                             "`rope_low_frequency_factor`.")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> CSMTransformerConfig:
        if not isinstance(values, Mapping):
            raise TypeError("CSM transformer configuration must be a mapping.")
        known = {item.name for item in fields(cls)}
        return cls(**{name: value for name, value in values.items() if name in known})


def _official_backbone() -> CSMTransformerConfig:
    return CSMTransformerConfig(
        hidden_size=2_048,
        intermediate_size=8_192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_sequence_length=2_048,
    )


def _official_depth_decoder() -> CSMTransformerConfig:
    return CSMTransformerConfig(
        hidden_size=1_024,
        intermediate_size=8_192,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_sequence_length=32,
    )


@dataclass(frozen=True, slots=True)
class CSMArchitectureConfig:
    """Complete source-checkpoint configuration for Sesame CSM.

    The official source artifact contains the conversational language
    model only. Mimi is a separately versioned codec and is deliberately
    not part of this state dictionary.
    """

    text_vocabulary_size: int = 128_256
    audio_vocabulary_size: int = 2_051
    num_audio_codebooks: int = 32
    sample_rate: int = 24_000
    frame_rate: float = 12.5
    backbone: CSMTransformerConfig = field(default_factory=_official_backbone)
    depth_decoder: CSMTransformerConfig = field(default_factory=_official_depth_decoder, )

    def __post_init__(self) -> None:
        for name in (
                "text_vocabulary_size",
                "audio_vocabulary_size",
                "num_audio_codebooks",
                "sample_rate",
        ):
            _positive_integer(name, getattr(self, name))
        object.__setattr__(
            self,
            "frame_rate",
            _positive_float("frame_rate", self.frame_rate),
        )
        if not isinstance(self.backbone, CSMTransformerConfig):
            object.__setattr__(
                self,
                "backbone",
                CSMTransformerConfig.from_dict(self.backbone),
            )
        if not isinstance(self.depth_decoder, CSMTransformerConfig):
            object.__setattr__(
                self,
                "depth_decoder",
                CSMTransformerConfig.from_dict(self.depth_decoder),
            )
        if self.depth_decoder.max_sequence_length < self.num_audio_codebooks:
            raise ValueError("The depth decoder sequence limit must cover every audio "
                             "codebook.")

    @classmethod
    def tiny(
        cls,
        *,
        text_vocabulary_size: int = 97,
        audio_vocabulary_size: int = 19,
        num_audio_codebooks: int = 4,
    ) -> CSMArchitectureConfig:
        """Return a small executable graph for tests and integrations."""
        return cls(
            text_vocabulary_size=text_vocabulary_size,
            audio_vocabulary_size=audio_vocabulary_size,
            num_audio_codebooks=num_audio_codebooks,
            backbone=CSMTransformerConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_sequence_length=64,
            ),
            depth_decoder=CSMTransformerConfig(
                hidden_size=24,
                intermediate_size=48,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_sequence_length=num_audio_codebooks,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": "csm",
            "format_version": 1,
            "text_vocabulary_size": self.text_vocabulary_size,
            "audio_vocabulary_size": self.audio_vocabulary_size,
            "num_audio_codebooks": self.num_audio_codebooks,
            "sample_rate": self.sample_rate,
            "frame_rate": self.frame_rate,
            "backbone": self.backbone.to_dict(),
            "depth_decoder": self.depth_decoder.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> CSMArchitectureConfig:
        if not isinstance(values, Mapping):
            raise TypeError("CSM configuration must be a mapping.")
        architecture = values.get("architecture")
        if architecture not in (None, "csm"):
            raise ValueError(f"Expected a CSM configuration, received {architecture!r}.")
        known = {item.name for item in fields(cls)}
        normalized = {name: value for name, value in values.items() if name in known}
        for name in ("backbone", "depth_decoder"):
            if name in normalized:
                normalized[name] = CSMTransformerConfig.from_dict(normalized[name], )
        return cls(**normalized)


__all__ = ["CSMArchitectureConfig", "CSMTransformerConfig"]
