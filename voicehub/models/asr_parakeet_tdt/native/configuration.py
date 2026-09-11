"""Validated configuration for VoiceHub's native Parakeet TDT graph.

The schema follows the official Transformers Parakeet implementation at
revision ``af71155683b4d34dd92d8f037392fa6bf334035e``.  It is
implemented locally and has no dependency on Transformers, NeMo, or
Hugging Face Hub.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any


def _integer(name: str, value: int, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}; found {value}.")


def _probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"`{name}` must be finite and in [0, 1).")


@dataclass(frozen=True, slots=True)
class ParakeetEncoderConfig:
    """FastConformer encoder dimensions and regularization."""

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int | None = None
    intermediate_size: int = 4096
    hidden_act: str = "silu"
    attention_bias: bool = False
    convolution_bias: bool = False
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    num_mel_bins: int = 128
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2
    dropout: float = 0.1
    dropout_positions: float = 0.0
    layerdrop: float = 0.1
    activation_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 5000
    scale_input: bool = False
    initializer_range: float = 0.02
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "intermediate_size",
                "conv_kernel_size",
                "subsampling_factor",
                "subsampling_conv_channels",
                "num_mel_bins",
                "subsampling_conv_kernel_size",
                "subsampling_conv_stride",
                "max_position_embeddings",
        ):
            _integer(name, getattr(self, name), minimum=1)
        key_value_heads = (
            self.num_attention_heads if self.num_key_value_heads is None else self.num_key_value_heads)
        _integer("num_key_value_heads", key_value_heads, minimum=1)
        object.__setattr__(self, "num_key_value_heads", key_value_heads)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        if self.num_attention_heads % key_value_heads:
            raise ValueError("`num_attention_heads` must be divisible by `num_key_value_heads`.")
        if self.conv_kernel_size % 2 != 1:
            raise ValueError("`conv_kernel_size` must be odd for same padding.")
        if self.subsampling_conv_kernel_size % 2 != 1:
            raise ValueError("`subsampling_conv_kernel_size` must be odd.")
        if self.subsampling_factor & (self.subsampling_factor - 1):
            raise ValueError("`subsampling_factor` must be a power of two.")
        expected_factor = self.subsampling_conv_stride**int(math.log2(self.subsampling_factor))
        if expected_factor != self.subsampling_factor:
            raise ValueError("The subsampling factor must equal stride ** number_of_layers.")
        if self.num_mel_bins % self.subsampling_factor:
            raise ValueError("`num_mel_bins` must be divisible by `subsampling_factor`.")
        if self.hidden_act not in {"relu", "silu"}:
            raise ValueError("Parakeet encoder `hidden_act` must be 'relu' or 'silu'.")
        for name in (
                "dropout",
                "dropout_positions",
                "layerdrop",
                "activation_dropout",
                "attention_dropout",
        ):
            _probability(name, getattr(self, name))
        if (isinstance(self.initializer_range, bool) or not isinstance(self.initializer_range,
                                                                       (int, float)) or
                not math.isfinite(self.initializer_range) or self.initializer_range <= 0):
            raise ValueError("`initializer_range` must be finite and positive.")
        for name in ("attention_bias", "convolution_bias", "scale_input"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("Encoder `extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ParakeetEncoderConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Parakeet encoder configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        names = {item.name for item in fields(cls) if item.name != "extra_config"}
        known = {name: source[name] for name in names if name in source}
        extras = {name: value for name, value in source.items() if name not in names}
        supplied = source.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("Encoder `extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**known, extra_config=extras)

    @classmethod
    def coerce(
        cls,
        value: ParakeetEncoderConfig | Mapping[str, Any],
    ) -> ParakeetEncoderConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name != "extra_config":
                result[item.name] = getattr(self, item.name)
        result.setdefault("model_type", "parakeet_encoder")
        return result


@dataclass(frozen=True, slots=True)
class ParakeetTDTConfig:
    """Complete Token-and-Duration Transducer configuration."""

    vocab_size: int = 8193
    decoder_hidden_size: int = 640
    num_decoder_layers: int = 2
    hidden_act: str = "relu"
    max_symbols_per_step: int = 10
    encoder_config: ParakeetEncoderConfig | Mapping[str, Any] = field(default_factory=ParakeetEncoderConfig)
    pad_token_id: int = 2
    eos_token_id: int = 3
    blank_token_id: int = 8192
    durations: tuple[int, ...] | Sequence[int] = (0, 1, 2, 3, 4)
    is_encoder_decoder: bool = True
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "vocab_size",
                "decoder_hidden_size",
                "num_decoder_layers",
                "max_symbols_per_step",
        ):
            _integer(name, getattr(self, name), minimum=1)
        for name in ("pad_token_id", "eos_token_id", "blank_token_id"):
            value = getattr(self, name)
            _integer(name, value)
            if value >= self.vocab_size:
                raise ValueError(f"`{name}` must be smaller than `vocab_size`.")
        if len({self.pad_token_id, self.blank_token_id}) != 2:
            raise ValueError("Parakeet pad and blank token IDs must differ.")
        if self.hidden_act not in {"relu", "silu"}:
            raise ValueError("TDT joint `hidden_act` must be 'relu' or 'silu'.")
        if not isinstance(self.is_encoder_decoder, bool) or not self.is_encoder_decoder:
            raise ValueError("Parakeet TDT must be configured as an encoder-decoder.")
        encoder = ParakeetEncoderConfig.coerce(self.encoder_config)
        object.__setattr__(self, "encoder_config", encoder)
        if isinstance(self.durations, (str, bytes)) or not isinstance(self.durations, Sequence):
            raise TypeError("`durations` must be a sequence of integers.")
        durations = tuple(self.durations)
        for value in durations:
            _integer("duration", value)
        if not durations or durations[0] != 0:
            raise ValueError("TDT `durations` must begin with zero.")
        if tuple(sorted(set(durations))) != durations:
            raise ValueError("TDT `durations` must be unique and strictly increasing.")
        if not any(value > 0 for value in durations):
            raise ValueError("TDT requires at least one positive duration.")
        object.__setattr__(self, "durations", durations)
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("TDT `extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ParakeetTDTConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Parakeet TDT configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        model_type = str(source.get("model_type", "parakeet_tdt")).lower()
        if model_type not in {"parakeet_tdt", "asr_parakeet_tdt"}:
            raise ValueError(
                "Native Parakeet TDT rejects CTC/RNNT and unrelated checkpoints; "
                f"received model_type={model_type!r}.")
        architectures = source.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        if architectures and (not isinstance(architectures, Sequence) or
                              not any(str(item) in {"ParakeetForTDT", "ParakeetTDTForSpeechRecognition"}
                                      for item in architectures)):
            raise ValueError("Native Parakeet TDT requires a ParakeetForTDT checkpoint.")
        names = {item.name for item in fields(cls) if item.name != "extra_config"}
        known = {name: source[name] for name in names if name in source}
        extras = {
            name: value
            for name, value in source.items() if name not in names and name != "extra_config"
        }
        supplied = source.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("TDT `extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**known, extra_config=extras)

    @classmethod
    def coerce(
        cls,
        value: ParakeetTDTConfig | Mapping[str, Any],
    ) -> ParakeetTDTConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            result[item.name] = (value.to_dict() if isinstance(value, ParakeetEncoderConfig) else value)
        result["durations"] = list(self.durations)
        result.setdefault('architectures', ["ParakeetForTDT"])
        result["model_type"] = "parakeet_tdt"
        return result

    @property
    def frame_seconds(self) -> float:
        """Duration represented by one encoder frame for the official
        frontend."""
        return 0.01 * self.encoder_config.subsampling_factor


__all__ = ["ParakeetEncoderConfig", "ParakeetTDTConfig"]
