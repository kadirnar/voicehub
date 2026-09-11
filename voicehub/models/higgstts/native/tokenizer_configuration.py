"""Validated configuration for the native Higgs Audio v2 tokenizer.

The graph follows the Apache-2.0 Transformers implementation at revision
``af71155683b4d34dd92d8f037392fa6bf334035e`` and the immutable Boson
tokenizer checkpoint at revision
``403fbacf2f60caaa102f893fdfabb694619b2417``.  Configuration parsing is
local and never imports an external model runtime.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any

from voicehub.models.asr_hubert.native import HubertConfig


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be positive.")
    return value


def _integer_tuple(name: str, value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence of integers.")
    result = tuple(value)
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    for item in result:
        _positive_integer(name, item)
    return result


def _positive_real_tuple(
    name: str,
    value: Sequence[int | float],
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence of real numbers.")
    result = tuple(float(item) for item in value)
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    if any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError(f"Every `{name}` value must be finite and positive.")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError(f"`{name}` must be strictly increasing.")
    return result


@dataclass(frozen=True, slots=True)
class HiggsAcousticCodecConfig:
    """DAC encoder/decoder dimensions embedded in the Higgs tokenizer."""

    encoder_hidden_size: int = 64
    downsampling_ratios: tuple[int, ...] = (8, 5, 4, 2, 3)
    decoder_hidden_size: int = 1_024
    upsampling_ratios: tuple[int, ...] = (8, 5, 4, 2, 3)
    hidden_size: int = 256
    codebook_size: int = 1_024
    codebook_dim: int = 8
    n_codebooks: int = 9
    quantizer_dropout: float = 0.0
    sampling_rate: int = 16_000
    commitment_loss_weight: float = 0.25
    codebook_loss_weight: float = 1.0
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
                "codebook_size",
                "codebook_dim",
                "n_codebooks",
                "sampling_rate",
        ):
            _positive_integer(name, getattr(self, name))
        for name in ("downsampling_ratios", "upsampling_ratios"):
            object.__setattr__(
                self,
                name,
                _integer_tuple(name, getattr(self, name)),
            )
        if self.downsampling_ratios != self.upsampling_ratios:
            raise ValueError("The audited Higgs codec uses symmetric DAC sampling ratios.")
        if (isinstance(self.quantizer_dropout, bool) or not isinstance(self.quantizer_dropout,
                                                                       (int, float)) or
                not 0.0 <= float(self.quantizer_dropout) <= 1.0):
            raise ValueError("`quantizer_dropout` must be in [0, 1].")
        for name in ("commitment_loss_weight", "codebook_loss_weight"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"`{name}` must be finite and non-negative.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @property
    def hop_length(self) -> int:
        return math.prod(self.downsampling_ratios)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> HiggsAcousticCodecConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Higgs acoustic configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        if source.get("model_type", "dac") != "dac":
            raise ValueError("Higgs acoustic configuration must be DAC.")
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        for name in ("downsampling_ratios", "upsampling_ratios"):
            if name in resolved:
                resolved[name] = tuple(resolved[name])
        declared_hop = source.get("hop_length")
        consumed = canonical | {
            'architectures',
            "dtype",
            "hop_length",
            "model_type",
            "transformers_version",
        }
        extras = {name: value for name, value in source.items() if name not in consumed}
        result = cls(**resolved, extra_config=extras)
        if declared_hop is not None and declared_hop != result.hop_length:
            raise ValueError("Serialized DAC `hop_length` does not match its sampling "
                             "ratios.")
        return result

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            result[item.name] = list(value) if isinstance(value, tuple) else value
        result.update({
            "hop_length": self.hop_length,
            "model_type": "dac",
        })
        return result


def _official_semantic_config() -> HubertConfig:
    return HubertConfig(mask_time_prob=0.0)


@dataclass(frozen=True, slots=True)
class HiggsAudioV2TokenizerConfig:
    """Complete semantic/acoustic residual tokenizer configuration."""

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
    acoustic_model_config: HiggsAcousticCodecConfig = field(default_factory=HiggsAcousticCodecConfig)
    semantic_model_config: HubertConfig = field(default_factory=_official_semantic_config)
    semantic_sample_rate: int = 16_000
    downsample_factor: int = 320
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
        object.__setattr__(
            self,
            "target_bandwidths",
            _positive_real_tuple(
                "target_bandwidths",
                self.target_bandwidths,
            ),
        )
        for name in (
                "channel_ratios",
                "strides",
                "block_dilations",
        ):
            object.__setattr__(
                self,
                name,
                _integer_tuple(name, getattr(self, name)),
            )
        if len(self.channel_ratios) != len(self.strides):
            raise ValueError("`channel_ratios` and `strides` must have equal lengths.")
        if self.kernel_size % 2 == 0 or self.unit_kernel_size % 2 == 0:
            raise ValueError("Higgs semantic convolution kernels must be odd.")
        if (isinstance(self.initializer_range, bool) or not isinstance(self.initializer_range,
                                                                       (int, float)) or
                not math.isfinite(float(self.initializer_range)) or float(self.initializer_range) <= 0.0):
            raise ValueError("`initializer_range` must be finite and positive.")
        if not isinstance(
                self.acoustic_model_config,
                HiggsAcousticCodecConfig,
        ):
            raise TypeError("`acoustic_model_config` must be HiggsAcousticCodecConfig.")
        if not isinstance(self.semantic_model_config, HubertConfig):
            raise TypeError("`semantic_model_config` must be HubertConfig.")
        if self.semantic_model_config.sampling_rate != self.semantic_sample_rate:
            raise ValueError("HuBERT and tokenizer semantic sampling rates must agree.")
        if self.codebook_size != self.acoustic_model_config.codebook_size:
            raise ValueError("Higgs residual and acoustic codebook sizes must agree.")
        if self.hop_length % self.downsample_factor:
            raise ValueError(
                "The acoustic hop length must be divisible by the semantic "
                "downsample factor.")
        semantic_scale = self.sample_rate / self.semantic_sample_rate
        semantic_hop = self.hop_length / semantic_scale
        if not semantic_hop.is_integer():
            raise ValueError("The sample-rate conversion must preserve an integral "
                             "semantic hop.")
        if int(semantic_hop) % self.downsample_factor:
            raise ValueError("Semantic HuBERT frames do not align with acoustic frames.")
        if self.num_quantizers > self.acoustic_model_config.n_codebooks:
            raise ValueError("Requested Higgs residual quantizers exceed the DAC "
                             "configuration.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @property
    def frame_rate(self) -> int:
        return math.ceil(self.sample_rate / self.hop_length)

    @property
    def hop_length(self) -> int:
        return self.acoustic_model_config.hop_length

    @property
    def codebook_nbits(self) -> int:
        return math.ceil(math.log2(self.codebook_size))

    @property
    def hidden_size(self) -> int:
        return (self.acoustic_model_config.hidden_size + self.semantic_model_config.hidden_size)

    @property
    def semantic_hidden_size(self) -> int:
        return self.semantic_model_config.hidden_size

    @property
    def num_quantizers(self) -> int:
        return int(1_000 * self.target_bandwidths[-1] // (self.frame_rate * self.codebook_nbits))

    @property
    def semantic_downsample_factor(self) -> int:
        factor = (self.hop_length / (self.sample_rate / self.semantic_sample_rate) / self.downsample_factor)
        return int(factor)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> HiggsAudioV2TokenizerConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Higgs tokenizer configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        if source.get("model_type", "higgs_audio_v2_tokenizer") != ("higgs_audio_v2_tokenizer"):
            raise ValueError("Native Higgs tokenizer requires "
                             "`model_type='higgs_audio_v2_tokenizer'`.")
        architectures = source.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        if architectures and set(architectures) != {"HiggsAudioV2TokenizerModel"}:
            raise ValueError("Checkpoint does not declare the Higgs Audio v2 tokenizer.")
        acoustic = HiggsAcousticCodecConfig.from_dict(source.get("acoustic_model_config", {}))
        semantic = HubertConfig.from_dict(source.get("semantic_model_config", {}))
        canonical = {
            item.name
            for item in fields(cls) if item.name not in {
                "acoustic_model_config",
                "semantic_model_config",
                "extra_config",
            }
        }
        resolved = {name: source[name] for name in canonical if name in source}
        for name in (
                "target_bandwidths",
                "channel_ratios",
                "strides",
                "block_dilations",
        ):
            if name in resolved:
                resolved[name] = tuple(resolved[name])
        consumed = canonical | {
            "_name_or_path",
            "acoustic_model_config",
            'architectures',
            "dtype",
            "extra_config",
            "model_type",
            "semantic_model_config",
            "torch_dtype",
            "transformers_version",
        }
        extras = {name: value for name, value in source.items() if name not in consumed}
        supplied_extras = source.get("extra_config")
        if supplied_extras is not None:
            if not isinstance(supplied_extras, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied_extras)))
        return cls(
            **resolved,
            acoustic_model_config=acoustic,
            semantic_model_config=semantic,
            extra_config=extras,
        )

    @classmethod
    def coerce(
        cls,
        value: HiggsAudioV2TokenizerConfig | Mapping[str, Any],
    ) -> HiggsAudioV2TokenizerConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    @classmethod
    def tiny(cls) -> HiggsAudioV2TokenizerConfig:
        """Return a lightweight graph preserving all executable contracts."""
        semantic = HubertConfig(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout=0.0,
            activation_dropout=0.0,
            attention_dropout=0.0,
            final_dropout=0.0,
            layerdrop=0.0,
            conv_dim=(4, 4),
            conv_stride=(2, 2),
            conv_kernel=(3, 3),
            num_conv_pos_embeddings=4,
            num_conv_pos_embedding_groups=2,
            apply_spec_augment=False,
            mask_time_prob=0.0,
            sampling_rate=16_000,
        )
        acoustic = HiggsAcousticCodecConfig(
            encoder_hidden_size=4,
            downsampling_ratios=(2, 2),
            decoder_hidden_size=16,
            upsampling_ratios=(2, 2),
            hidden_size=8,
            codebook_size=16,
            codebook_dim=4,
            n_codebooks=2,
            sampling_rate=16_000,
        )
        return replace(
            cls(),
            target_bandwidths=(32.0, ),
            sample_rate=16_000,
            codebook_size=16,
            codebook_dim=4,
            acoustic_model_config=acoustic,
            semantic_model_config=semantic,
            semantic_sample_rate=16_000,
            downsample_factor=4,
        )

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name in {
                    "acoustic_model_config",
                    "semantic_model_config",
                    "extra_config",
            }:
                continue
            value = getattr(self, item.name)
            result[item.name] = list(value) if isinstance(value, tuple) else value
        result.update({
            "acoustic_model_config": self.acoustic_model_config.to_dict(),
            'architectures': ["HiggsAudioV2TokenizerModel"],
            "model_type": "higgs_audio_v2_tokenizer",
            "semantic_model_config": self.semantic_model_config.to_dict(),
        })
        return result


__all__ = [
    "HiggsAcousticCodecConfig",
    "HiggsAudioV2TokenizerConfig",
]
