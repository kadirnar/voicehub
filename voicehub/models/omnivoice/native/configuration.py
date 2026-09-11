"""Validated configuration for VoiceHub's native OmniVoice architecture.

The public checkpoint is a masked-token speech model built around a
dense Qwen3-0.6B backbone and the Higgs Audio V2 tokenizer.  This module
keeps the two checkpoint contracts explicit and rejects options which
would change the published mathematics.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from functools import reduce
from operator import mul
from types import MappingProxyType
from typing import Any

from voicehub.models.asr_hubert.native import HubertConfig
from voicehub.models.causal_lm.native import Qwen3Config


def _published_llm_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=151_676,
        hidden_size=1_024,
        intermediate_size=3_072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40_960,
        rope_theta=1_000_000.0,
        pad_token_id=None,
        bos_token_id=151_643,
        eos_token_id=151_645,
        tie_word_embeddings=True,
        max_window_layers=28,
        layer_types=("full_attention", ) * 28,
    )


def _published_hubert_config() -> HubertConfig:
    # The Higgs tokenizer disables HuBERT time masking and therefore has no
    # `masked_spec_embed` tensor in its 527-tensor checkpoint.
    return HubertConfig(mask_time_prob=0.0)


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


def _positive_integer_tuple(
    name: str,
    value: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence of integers.")
    result = tuple(value)
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    for item in result:
        _positive_integer(name, item)
    return result


@dataclass(frozen=True, slots=True)
class HiggsAcousticConfig:
    """DAC subgraph serialized inside the Higgs Audio V2 tokenizer."""

    encoder_hidden_size: int = 64
    downsampling_ratios: tuple[int, ...] = (8, 5, 4, 2, 3)
    decoder_hidden_size: int = 1_024
    upsampling_ratios: tuple[int, ...] = (8, 5, 4, 2, 3)
    hidden_size: int = 256
    n_codebooks: int = 9
    codebook_size: int = 1_024
    codebook_dim: int = 8
    quantizer_dropout: float = 0.0
    commitment_loss_weight: float = 0.25
    codebook_loss_weight: float = 1.0
    sampling_rate: int = 16_000
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "encoder_hidden_size",
                "decoder_hidden_size",
                "hidden_size",
                "n_codebooks",
                "codebook_size",
                "codebook_dim",
                "sampling_rate",
        ):
            _positive_integer(name, getattr(self, name))
        object.__setattr__(
            self,
            "downsampling_ratios",
            _positive_integer_tuple(
                "downsampling_ratios",
                self.downsampling_ratios,
            ),
        )
        object.__setattr__(
            self,
            "upsampling_ratios",
            _positive_integer_tuple(
                "upsampling_ratios",
                self.upsampling_ratios,
            ),
        )
        if len(self.downsampling_ratios) != len(self.upsampling_ratios):
            raise ValueError("Higgs DAC encoder and decoder must have equal depth.")
        if reduce(mul, self.downsampling_ratios, 1) != reduce(
                mul,
                self.upsampling_ratios,
                1,
        ):
            raise ValueError("Higgs DAC encoder and decoder rates must have equal products.")
        for name in (
                "quantizer_dropout",
                "commitment_loss_weight",
                "codebook_loss_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"`{name}` must be finite and non-negative.")
            object.__setattr__(self, name, value)
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @property
    def hop_length(self) -> int:
        return reduce(mul, self.downsampling_ratios, 1)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> HiggsAcousticConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Higgs acoustic configuration must be a mapping.")
        model_type = values.get("model_type", "dac")
        if str(model_type).lower() != "dac":
            raise ValueError("Higgs acoustic configuration requires `model_type='dac'`.")
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {
            name:
            tuple(values[name]) if name in {"downsampling_ratios", "upsampling_ratios"} else values[name]
            for name in canonical if name in values
        }
        extras = {
            name: copy.deepcopy(value)
            for name, value in values.items() if name not in canonical | {"model_type", "extra_config"}
        }
        supplied = values.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**resolved, extra_config=extras)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            result[item.name] = list(value) if isinstance(value, tuple) else value
        result["model_type"] = "dac"
        result["hop_length"] = self.hop_length
        return result


@dataclass(frozen=True, slots=True)
class HiggsAudioV2Config:
    """Complete, checkpoint-bound Higgs Audio V2 tokenizer config."""

    target_bandwidths: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    sample_rate: int = 24_000
    kernel_size: int = 3
    channel_ratios: tuple[int, ...] = (1, 1)
    strides: tuple[int, ...] = (1, 1)
    block_dilations: tuple[int, ...] = (1, 1)
    unit_kernel_size: int = 3
    codebook_size: int = 1_024
    codebook_dim: int = 64
    initializer_range: float = 0.02
    semantic_sample_rate: int = 16_000
    downsample_factor: int = 320
    acoustic_model_config: HiggsAcousticConfig = field(default_factory=HiggsAcousticConfig)
    semantic_model_config: HubertConfig = field(default_factory=_published_hubert_config)
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "sample_rate",
                "kernel_size",
                "unit_kernel_size",
                "codebook_size",
                "codebook_dim",
                "semantic_sample_rate",
                "downsample_factor",
        ):
            _positive_integer(name, getattr(self, name))
        for name in ("channel_ratios", "strides", "block_dilations"):
            object.__setattr__(
                self,
                name,
                _positive_integer_tuple(name, getattr(self, name)),
            )
        if len(self.channel_ratios) != len(self.strides):
            raise ValueError("Higgs semantic channel ratios and strides must align.")
        bandwidths = tuple(_positive_float("target_bandwidths", value) for value in self.target_bandwidths)
        if tuple(sorted(set(bandwidths))) != bandwidths:
            raise ValueError("`target_bandwidths` must be unique and increasing.")
        object.__setattr__(self, "target_bandwidths", bandwidths)
        object.__setattr__(
            self,
            "initializer_range",
            _positive_float("initializer_range", self.initializer_range),
        )
        if not isinstance(self.acoustic_model_config, HiggsAcousticConfig):
            object.__setattr__(
                self,
                "acoustic_model_config",
                HiggsAcousticConfig.from_mapping(self.acoustic_model_config),
            )
        if not isinstance(self.semantic_model_config, HubertConfig):
            object.__setattr__(
                self,
                "semantic_model_config",
                HubertConfig.from_dict(self.semantic_model_config),
            )
        if self.acoustic_model_config.hop_length % self.downsample_factor:
            raise ValueError("Higgs semantic downsampling must divide the DAC hop.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @property
    def hop_length(self) -> int:
        return self.acoustic_model_config.hop_length

    @property
    def frame_rate(self) -> int:
        return math.ceil(self.sample_rate / self.hop_length)

    @property
    def semantic_hidden_size(self) -> int:
        return self.semantic_model_config.hidden_size

    @property
    def hidden_size(self) -> int:
        return (self.acoustic_model_config.hidden_size + self.semantic_model_config.hidden_size)

    @property
    def codebook_nbits(self) -> int:
        return math.ceil(math.log2(self.codebook_size))

    @property
    def num_quantizers(self) -> int:
        return int(1_000 * self.target_bandwidths[-1] // (self.frame_rate * self.codebook_nbits))

    @property
    def semantic_downsample_factor(self) -> int:
        return int(self.hop_length / (self.sample_rate / self.semantic_sample_rate) / self.downsample_factor)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> HiggsAudioV2Config:
        if not isinstance(values, Mapping):
            raise TypeError("Higgs tokenizer configuration must be a mapping.")
        if str(values.get("model_type", "")).lower() != "higgs_audio_v2_tokenizer":
            raise ValueError("Higgs tokenizer requires `model_type='higgs_audio_v2_tokenizer'`.")
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved: dict[str, Any] = {}
        for name in canonical:
            if name not in values:
                continue
            value = values[name]
            if name in {
                    "target_bandwidths",
                    "channel_ratios",
                    "strides",
                    "block_dilations",
            }:
                value = tuple(value)
            elif name == "acoustic_model_config":
                value = HiggsAcousticConfig.from_mapping(value)
            elif name == "semantic_model_config":
                value = HubertConfig.from_dict(value)
            resolved[name] = value
        extras = {
            name: copy.deepcopy(value)
            for name, value in values.items() if name not in canonical
            | {
                'architectures',
                "dtype",
                "model_type",
                "transformers_version",
                "extra_config",
            }
        }
        supplied = values.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**resolved, extra_config=extras)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            if isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, (HiggsAcousticConfig, HubertConfig)):
                value = value.to_dict()
            result[item.name] = value
        result.update({
            'architectures': ["HiggsAudioV2TokenizerModel"],
            "model_type": "higgs_audio_v2_tokenizer",
        })
        return result


@dataclass(frozen=True, slots=True)
class OmniVoiceArchitectureConfig:
    """Published OmniVoice model graph and masked-token loss contract."""

    audio_vocab_size: int = 1_025
    audio_mask_id: int = 1_024
    num_audio_codebook: int = 8
    audio_codebook_weights: tuple[float, ...] = (
        8.0,
        8.0,
        6.0,
        6.0,
        4.0,
        4.0,
        2.0,
        2.0,
    )
    llm_config: Qwen3Config = field(default_factory=_published_llm_config)
    pad_token_id: int = 151_643
    bos_token_id: int | None = None
    eos_token_id: int = 151_645
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "audio_vocab_size",
                "num_audio_codebook",
                "pad_token_id",
                "eos_token_id",
        ):
            _positive_integer(name, getattr(self, name))
        if (isinstance(self.audio_mask_id, bool) or not isinstance(self.audio_mask_id, int) or
                not 0 <= self.audio_mask_id < self.audio_vocab_size):
            raise ValueError("`audio_mask_id` must be inside the audio vocabulary.")
        weights = tuple(
            _positive_float("audio_codebook_weights", value) for value in self.audio_codebook_weights)
        if len(weights) != self.num_audio_codebook:
            raise ValueError("One loss weight is required for every OmniVoice codebook.")
        object.__setattr__(self, "audio_codebook_weights", weights)
        if not isinstance(self.llm_config, Qwen3Config):
            object.__setattr__(
                self,
                "llm_config",
                Qwen3Config.from_dict(self.llm_config),
            )
        if self.llm_config.model_type != "qwen3":
            raise ValueError("The published OmniVoice backbone is dense Qwen3.")
        if self.llm_config.vocab_size != 151_676:
            raise ValueError("Published OmniVoice token embeddings require 151,676 rows.")
        if self.pad_token_id >= self.llm_config.vocab_size:
            raise ValueError("`pad_token_id` must be inside the text vocabulary.")
        if self.eos_token_id >= self.llm_config.vocab_size:
            raise ValueError("`eos_token_id` must be inside the text vocabulary.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> OmniVoiceArchitectureConfig:
        if not isinstance(values, Mapping):
            raise TypeError("OmniVoice configuration must be a mapping.")
        if str(values.get("model_type", "")).lower() != "omnivoice":
            raise ValueError("OmniVoice requires `model_type='omnivoice'`.")
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved: dict[str, Any] = {}
        for name in canonical:
            if name not in values:
                continue
            value = values[name]
            if name == "audio_codebook_weights":
                value = tuple(value)
            elif name == "llm_config":
                value = Qwen3Config.from_dict(value)
            resolved[name] = value
        extras = {
            name: copy.deepcopy(value)
            for name, value in values.items() if name not in canonical
            | {
                'architectures',
                "dtype",
                "model_type",
                "transformers_version",
                "extra_config",
            }
        }
        supplied = values.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**resolved, extra_config=extras)

    @classmethod
    def tiny(
        cls,
        *,
        vocab_size: int = 64,
        hidden_size: int = 16,
    ) -> OmniVoiceArchitectureConfig:
        """Create a small executable graph for isolated contract tests."""
        llm = Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=hidden_size // 4,
            max_position_embeddings=128,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=True,
        )
        # Tiny graphs deliberately use a reduced text vocabulary.  Construct
        # without routing through the published-size guard.
        instance = object.__new__(cls)
        object.__setattr__(instance, "audio_vocab_size", 17)
        object.__setattr__(instance, "audio_mask_id", 16)
        object.__setattr__(instance, "num_audio_codebook", 2)
        object.__setattr__(instance, "audio_codebook_weights", (2.0, 1.0))
        object.__setattr__(instance, "llm_config", llm)
        object.__setattr__(instance, "pad_token_id", 0)
        object.__setattr__(instance, "bos_token_id", 1)
        object.__setattr__(instance, "eos_token_id", 2)
        object.__setattr__(instance, "extra_config", MappingProxyType({}))
        return instance

    @property
    def normalized_audio_codebook_weights(self) -> tuple[float, ...]:
        total = sum(self.audio_codebook_weights)
        return tuple(value / total for value in self.audio_codebook_weights)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        result.update({
            'architectures': ["OmniVoice"],
            "audio_codebook_weights": list(self.audio_codebook_weights),
            "audio_mask_id": self.audio_mask_id,
            "audio_vocab_size": self.audio_vocab_size,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "llm_config": self.llm_config.to_dict(),
            "model_type": "omnivoice",
            "num_audio_codebook": self.num_audio_codebook,
            "pad_token_id": self.pad_token_id,
        })
        return result


__all__ = [
    "HiggsAcousticConfig",
    "HiggsAudioV2Config",
    "OmniVoiceArchitectureConfig",
]
