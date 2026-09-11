"""Validated configuration for VoiceHub's native Granite Speech graph.

The field semantics mirror IBM Granite Speech as implemented by Transformers
at the pinned revision recorded in :mod:`voicehub.models.asr_granite_speech.native.metadata`.
This module is intentionally dependency-free and rejects checkpoint features
whose mathematics are not represented by the native graph.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

from voicehub.models.causal_lm.native.configuration import GraniteConfig


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}; found {value}.")
    return value


def _positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"`{name}` must be finite and greater than zero.")
    return result


def _probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"`{name}` must be finite and in [0, 1).")
    return result


def _extras(
    values: Mapping[str, Any],
    config_type: type[Any],
) -> dict[str, Any]:
    canonical = {item.name for item in fields(config_type) if item.name != "extra_config"}
    return {name: copy.deepcopy(value) for name, value in values.items() if name not in canonical}


def _freeze(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(copy.deepcopy(dict(values)))


@dataclass(frozen=True, slots=True)
class GraniteSpeechEncoderConfig:
    """Conformer encoder configuration used by Granite Speech."""

    model_type: str = "granite_speech_encoder"
    input_dim: int = 160
    num_layers: int = 10
    hidden_dim: int = 1_024
    feedforward_mult: int = 4
    num_heads: int = 8
    dim_head: int | None = None
    output_dim: int = 42
    context_size: int = 200
    max_pos_emb: int = 512
    dropout: float = 0.1
    conv_kernel_size: int = 15
    conv_expansion_factor: int = 2
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.model_type != "granite_speech_encoder":
            raise ValueError("GraniteSpeechEncoderConfig requires "
                             "`model_type='granite_speech_encoder'`.")
        for name in (
                "input_dim",
                "num_layers",
                "hidden_dim",
                "feedforward_mult",
                "num_heads",
                "output_dim",
                "context_size",
                "max_pos_emb",
                "conv_kernel_size",
                "conv_expansion_factor",
        ):
            _integer(name, getattr(self, name), minimum=1)
        dim_head = self.dim_head
        if dim_head is None:
            if self.hidden_dim % self.num_heads:
                raise ValueError(
                    "`hidden_dim` must divide evenly by `num_heads` when "
                    "`dim_head` is omitted.")
            dim_head = self.hidden_dim // self.num_heads
            object.__setattr__(self, "dim_head", dim_head)
        _integer("dim_head", dim_head, minimum=1)
        if self.context_size > self.max_pos_emb:
            raise ValueError("`context_size` cannot exceed `max_pos_emb`.")
        object.__setattr__(
            self,
            "dropout",
            _probability("dropout", self.dropout),
        )
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(self, "extra_config", _freeze(self.extra_config))

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> GraniteSpeechEncoderConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Granite Speech encoder configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        extras = _extras(source, cls)
        supplied = source.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**resolved, extra_config=extras)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name != "extra_config":
                result[item.name] = copy.deepcopy(getattr(self, item.name))
        return result


@dataclass(frozen=True, slots=True)
class GraniteSpeechProjectorConfig:
    """BLIP-2 Q-Former subset used as Granite Speech's audio projector."""

    model_type: str = "blip_2_qformer"
    hidden_size: int = 768
    encoder_hidden_size: int = 1_024
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3_072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12
    cross_attention_frequency: int = 2
    use_qformer_text_input: bool = False
    initializer_range: float = 0.02
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.model_type != "blip_2_qformer":
            raise ValueError(
                "Granite Speech requires `projector_config.model_type` to be "
                "'blip_2_qformer'.")
        for name in (
                "hidden_size",
                "encoder_hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "intermediate_size",
                "cross_attention_frequency",
        ):
            _integer(name, getattr(self, name), minimum=1)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                "`projector_config.hidden_size` must divide evenly by "
                "`num_attention_heads`.")
        if self.hidden_act != "gelu":
            raise ValueError(
                "Native Granite Speech checkpoint parity requires "
                "`projector_config.hidden_act='gelu'`.")
        object.__setattr__(
            self,
            "hidden_dropout_prob",
            _probability(
                "hidden_dropout_prob",
                self.hidden_dropout_prob,
            ),
        )
        object.__setattr__(
            self,
            "attention_probs_dropout_prob",
            _probability(
                "attention_probs_dropout_prob",
                self.attention_probs_dropout_prob,
            ),
        )
        object.__setattr__(
            self,
            "layer_norm_eps",
            _positive_float("layer_norm_eps", self.layer_norm_eps),
        )
        object.__setattr__(
            self,
            "initializer_range",
            _positive_float("initializer_range", self.initializer_range),
        )
        if not isinstance(self.use_qformer_text_input, bool):
            raise TypeError("`use_qformer_text_input` must be a boolean.")
        if self.use_qformer_text_input:
            raise ValueError(
                "Granite Speech's native audio projector does not own text "
                "embeddings; `use_qformer_text_input` must be false.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(self, "extra_config", _freeze(self.extra_config))

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> GraniteSpeechProjectorConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Granite Speech projector configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        extras = _extras(source, cls)
        supplied = source.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**resolved, extra_config=extras)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name != "extra_config":
                result[item.name] = copy.deepcopy(getattr(self, item.name))
        return result


@dataclass(frozen=True, slots=True)
class GraniteSpeechArchitectureConfig:
    """Complete executable Granite Speech architecture configuration."""

    model_type: str = "granite_speech"
    text_config: GraniteConfig | Mapping[str, Any] = field(default_factory=GraniteConfig, )
    encoder_config: GraniteSpeechEncoderConfig | Mapping[str, Any] = field(
        default_factory=GraniteSpeechEncoderConfig, )
    projector_config: GraniteSpeechProjectorConfig | Mapping[str, Any] = field(
        default_factory=GraniteSpeechProjectorConfig, )
    audio_token_index: int = 49_155
    initializer_range: float = 0.02
    has_lora_adapter: bool = False
    downsample_rate: int = 5
    window_size: int = 15
    tie_word_embeddings: bool = False
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.model_type != "granite_speech":
            raise ValueError(
                "Only the `granite_speech` architecture is supported; "
                "`granite_speech_plus` has a different graph.")
        text_config = (
            self.text_config if isinstance(self.text_config, GraniteConfig) else GraniteConfig.from_dict(
                self.text_config))
        encoder_config = (
            self.encoder_config if isinstance(self.encoder_config, GraniteSpeechEncoderConfig) else
            GraniteSpeechEncoderConfig.from_dict(self.encoder_config))
        projector_config = (
            self.projector_config if isinstance(self.projector_config, GraniteSpeechProjectorConfig) else
            GraniteSpeechProjectorConfig.from_dict(self.projector_config))
        object.__setattr__(self, "text_config", text_config)
        object.__setattr__(self, "encoder_config", encoder_config)
        object.__setattr__(self, "projector_config", projector_config)

        _integer("audio_token_index", self.audio_token_index, minimum=0)
        if self.audio_token_index >= text_config.vocab_size:
            raise ValueError("`audio_token_index` must be inside the text vocabulary.")
        object.__setattr__(
            self,
            "initializer_range",
            _positive_float("initializer_range", self.initializer_range),
        )
        for name in ("has_lora_adapter", "tie_word_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.has_lora_adapter:
            raise ValueError(
                "Checkpoints declaring an embedded PEFT adapter cannot be "
                "loaded as a dense native graph. Use the non-adapter 4.1 "
                "checkpoint and VoiceHub's native LoRA support.")
        _integer("downsample_rate", self.downsample_rate, minimum=1)
        _integer("window_size", self.window_size, minimum=1)
        if self.window_size % self.downsample_rate:
            raise ValueError("`window_size` must be divisible by `downsample_rate`.")
        if projector_config.encoder_hidden_size != encoder_config.hidden_dim:
            raise ValueError(
                "The Q-Former `encoder_hidden_size` must match the audio "
                "encoder `hidden_dim`.")
        if self.tie_word_embeddings != text_config.tie_word_embeddings:
            raise ValueError("Top-level and text `tie_word_embeddings` values must agree.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(self, "extra_config", _freeze(self.extra_config))

    @property
    def audio_token_id(self) -> int:
        return self.audio_token_index

    @property
    def projector_tokens_per_window(self) -> int:
        return self.window_size // self.downsample_rate

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> GraniteSpeechArchitectureConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Granite Speech configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        model_type = str(source.get("model_type", "granite_speech"))
        if model_type != "granite_speech":
            raise ValueError(f"Unsupported Granite Speech `model_type` {model_type!r}.")
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        extras = _extras(source, cls)
        supplied = source.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**resolved, extra_config=extras)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        result.update({
            'architectures': ["GraniteSpeechForConditionalGeneration"],
            "model_type": self.model_type,
            "text_config": self.text_config.to_dict(),
            "encoder_config": self.encoder_config.to_dict(),
            "projector_config": self.projector_config.to_dict(),
            "audio_token_index": self.audio_token_index,
            "initializer_range": self.initializer_range,
            "has_lora_adapter": self.has_lora_adapter,
            "downsample_rate": self.downsample_rate,
            "window_size": self.window_size,
            "tie_word_embeddings": self.tie_word_embeddings,
        })
        return result


__all__ = [
    "GraniteSpeechArchitectureConfig",
    "GraniteSpeechEncoderConfig",
    "GraniteSpeechProjectorConfig",
]
