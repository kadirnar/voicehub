"""Validated configuration for the native Nemotron 3.5 ASR graph.

The public checkpoint is a prompt-conditioned, cache-aware FastConformer
encoder followed by an RNN-T prediction and joint network.  This module
keeps the checkpoint schema local to VoiceHub and deliberately rejects
adjacent Nemotron/Parakeet architectures instead of guessing a
compatible graph.
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


def _immutable_mapping(value: Mapping[str, Any], *, name: str):
    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a mapping.")
    return MappingProxyType(copy.deepcopy(dict(value)))


@dataclass(frozen=True, slots=True)
class NemotronEncoderConfig:
    """Cache-aware FastConformer encoder dimensions and regularization."""

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
    sliding_window: int = 57
    default_num_lookahead_tokens: int = 3
    supported_num_lookahead_tokens: tuple[int, ...] | Sequence[int] = (
        3,
        0,
        6,
        13,
    )
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
                "sliding_window",
        ):
            _integer(name, getattr(self, name), minimum=1)
        key_value_heads = (
            self.num_attention_heads if self.num_key_value_heads is None else self.num_key_value_heads)
        _integer("num_key_value_heads", key_value_heads, minimum=1)
        object.__setattr__(self, "num_key_value_heads", key_value_heads)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        if self.num_attention_heads % key_value_heads:
            raise ValueError("`num_attention_heads` must be divisible by "
                             "`num_key_value_heads`.")
        if self.conv_kernel_size % 2 != 1:
            raise ValueError("`conv_kernel_size` must be odd.")
        if self.subsampling_conv_kernel_size % 2 != 1:
            raise ValueError("`subsampling_conv_kernel_size` must be odd.")
        if self.subsampling_factor & (self.subsampling_factor - 1):
            raise ValueError("`subsampling_factor` must be a power of two.")
        layer_count = int(math.log2(self.subsampling_factor))
        if self.subsampling_conv_stride**layer_count != self.subsampling_factor:
            raise ValueError("The subsampling factor must equal stride ** layer_count.")
        if self.num_mel_bins % self.subsampling_factor:
            raise ValueError("`num_mel_bins` must be divisible by `subsampling_factor`.")
        if self.hidden_act not in {"relu", "silu"}:
            raise ValueError("Nemotron encoder activation must be relu or silu.")
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
        lookaheads = tuple(self.supported_num_lookahead_tokens)
        if not lookaheads:
            raise ValueError("At least one streaming lookahead is required.")
        for value in lookaheads:
            _integer("supported lookahead", value)
        if len(set(lookaheads)) != len(lookaheads):
            raise ValueError("Streaming lookaheads cannot contain duplicates.")
        if self.default_num_lookahead_tokens not in lookaheads:
            raise ValueError("The default lookahead must be present in the supported set.")
        object.__setattr__(
            self,
            "supported_num_lookahead_tokens",
            lookaheads,
        )
        object.__setattr__(
            self,
            "extra_config",
            _immutable_mapping(self.extra_config, name="extra_config"),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def subsampling_out_hidden_size(self) -> int:
        """Flattened channel/frequency size after causal Conv2d subsampling."""
        total_pad = (self.subsampling_conv_kernel_size - 1 + self.subsampling_conv_stride - 1)
        length = self.num_mel_bins
        for _ in range(int(math.log2(self.subsampling_factor))):
            length = (
                length + total_pad - self.subsampling_conv_kernel_size) // self.subsampling_conv_stride + 1
        return self.subsampling_conv_channels * length

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> NemotronEncoderConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Nemotron encoder configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        names = {item.name for item in fields(cls) if item.name != "extra_config"}
        known = {name: source[name] for name in names if name in source}
        extras = {
            name: value
            for name, value in source.items() if name not in names and name != "extra_config"
        }
        supplied = source.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("Encoder `extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**known, extra_config=extras)

    @classmethod
    def coerce(
        cls,
        value: NemotronEncoderConfig | Mapping[str, Any],
    ) -> NemotronEncoderConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name != "extra_config":
                value = getattr(self, item.name)
                result[item.name] = (list(value) if item.name == "supported_num_lookahead_tokens" else value)
        result.setdefault("model_type", "nemotron_asr_streaming_encoder")
        return result


@dataclass(frozen=True, slots=True)
class NemotronASRArchitectureConfig:
    """Complete prompt-conditioned Nemotron 3.5 RNN-T configuration."""

    vocab_size: int = 13088
    decoder_hidden_size: int = 640
    num_decoder_layers: int = 2
    hidden_act: str = "relu"
    max_symbols_per_step: int = 10
    encoder_config: NemotronEncoderConfig | Mapping[str, Any] = field(default_factory=NemotronEncoderConfig, )
    pad_token_id: int = 0
    blank_token_id: int = 13087
    num_prompts: int = 128
    prompt_intermediate_size: int = 2048
    default_prompt_id: int = 101
    durations: tuple[int, ...] | Sequence[int] = ()
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
                "num_prompts",
                "prompt_intermediate_size",
        ):
            _integer(name, getattr(self, name), minimum=1)
        for name in ("pad_token_id", "blank_token_id", "default_prompt_id"):
            _integer(name, getattr(self, name))
        if self.pad_token_id >= self.vocab_size:
            raise ValueError("`pad_token_id` must be smaller than `vocab_size`.")
        if self.blank_token_id >= self.vocab_size:
            raise ValueError("`blank_token_id` must be smaller than `vocab_size`.")
        if self.default_prompt_id >= self.num_prompts:
            raise ValueError("`default_prompt_id` must be smaller than `num_prompts`.")
        if self.hidden_act not in {"relu", "silu"}:
            raise ValueError("Nemotron joint activation must be relu or silu.")
        if not isinstance(self.is_encoder_decoder, bool):
            raise TypeError("`is_encoder_decoder` must be a boolean.")
        if not self.is_encoder_decoder:
            raise ValueError("Nemotron ASR must be encoder-decoder.")
        encoder = NemotronEncoderConfig.coerce(self.encoder_config)
        object.__setattr__(self, "encoder_config", encoder)
        durations = tuple(self.durations)
        if durations:
            raise ValueError(
                "Nemotron 3.5 uses RNN-T; duration-token/TDT checkpoints "
                "are not compatible.")
        object.__setattr__(self, "durations", durations)
        object.__setattr__(
            self,
            "extra_config",
            _immutable_mapping(self.extra_config, name="extra_config"),
        )

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> NemotronASRArchitectureConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Nemotron configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        model_type = str(source.get("model_type", "nemotron3_5_asr")).lower()
        if model_type not in {"nemotron3_5_asr", "asr_nemotron"}:
            raise ValueError(
                "Native Nemotron ASR requires model_type "
                f"'nemotron3_5_asr'; found {model_type!r}.")
        architectures = source.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        if architectures:
            if (not isinstance(architectures, Sequence) or
                    "Nemotron3_5AsrForRNNT" not in {str(item) for item in architectures}):
                raise ValueError("Native Nemotron ASR requires a "
                                 "Nemotron3_5AsrForRNNT checkpoint.")
        names = {item.name for item in fields(cls) if item.name != "extra_config"}
        known = {name: source[name] for name in names if name in source}
        extras = {
            name: value
            for name, value in source.items()
            if name not in names and name not in {"extra_config", 'architectures', "model_type"}
        }
        supplied = source.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**known, extra_config=extras)

    @classmethod
    def coerce(
        cls,
        value: NemotronASRArchitectureConfig | Mapping[str, Any],
    ) -> NemotronASRArchitectureConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            if isinstance(value, NemotronEncoderConfig):
                value = value.to_dict()
            elif item.name == "durations":
                value = list(value)
            result[item.name] = value
        result['architectures'] = ["Nemotron3_5AsrForRNNT"]
        result["model_type"] = "nemotron3_5_asr"
        return result

    @property
    def frame_seconds(self) -> float:
        return 0.01 * self.encoder_config.subsampling_factor


@dataclass(frozen=True, slots=True)
class NemotronFrontendConfig:
    """Published Nemotron 3.5 waveform frontend."""

    feature_size: int = 128
    sampling_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 512
    win_length: int = 400
    preemphasis: float = 0.97
    padding_value: float = 0.0

    def __post_init__(self) -> None:
        for name in (
                "feature_size",
                "sampling_rate",
                "hop_length",
                "n_fft",
                "win_length",
        ):
            _integer(name, getattr(self, name), minimum=1)
        if self.win_length > self.n_fft:
            raise ValueError("`win_length` cannot exceed `n_fft`.")
        if (isinstance(self.preemphasis, bool) or not isinstance(self.preemphasis, (int, float)) or
                not math.isfinite(self.preemphasis) or not 0.0 <= self.preemphasis < 1.0):
            raise ValueError("`preemphasis` must be finite and in [0, 1).")
        if (isinstance(self.padding_value, bool) or not isinstance(self.padding_value, (int, float)) or
                not math.isfinite(self.padding_value)):
            raise ValueError("`padding_value` must be finite.")

    @classmethod
    def from_processor_dict(
        cls,
        values: Mapping[str, Any],
    ) -> NemotronFrontendConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Nemotron processor configuration must be a mapping.")
        nested = values.get("feature_extractor", values)
        if not isinstance(nested, Mapping):
            raise TypeError("Nemotron `feature_extractor` configuration must be a mapping.")
        names = {item.name for item in fields(cls)}
        return cls(**{name: nested[name] for name in names if name in nested})


__all__ = [
    "NemotronASRArchitectureConfig",
    "NemotronEncoderConfig",
    "NemotronFrontendConfig",
]
