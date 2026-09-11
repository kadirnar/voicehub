"""Validated architecture configuration for Fish Speech S2.

The published ``fish_qwen3_omni`` JSON nests the slow text transformer
and the fast residual-codebook decoder.  VoiceHub keeps that file format
at the artifact boundary and exposes one immutable, executable
configuration internally.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return value


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"`{name}` must be finite and greater than zero.")
    return result


@dataclass(frozen=True, slots=True)
class FishTransformerConfig:
    """One fused-QKV Qwen3-style transformer stack."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    attention_qkv_bias: bool = False
    attention_o_bias: bool = False
    attention_qk_norm: bool = False
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        for name in (
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
        ):
            _positive_integer(getattr(self, name), name=name)
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("`num_attention_heads` must be divisible by "
                             "`num_key_value_heads`.")
        if self.head_dim % 2:
            raise ValueError("Fish rotary attention requires an even `head_dim`.")
        if self.num_attention_heads * self.head_dim < self.hidden_size:
            raise ValueError("Fish attention projection width cannot be smaller than "
                             "`hidden_size`.")
        object.__setattr__(
            self,
            "rope_theta",
            _finite_positive(self.rope_theta, name="rope_theta"),
        )
        object.__setattr__(
            self,
            "rms_norm_eps",
            _finite_positive(self.rms_norm_eps, name="rms_norm_eps"),
        )
        object.__setattr__(
            self,
            "initializer_range",
            _finite_positive(
                self.initializer_range,
                name="initializer_range",
            ),
        )
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or
                not math.isfinite(float(self.dropout)) or not 0.0 <= float(self.dropout) < 1.0):
            raise ValueError("`dropout` must be finite and in [0, 1).")
        for name in (
                "attention_qkv_bias",
                "attention_o_bias",
                "attention_qk_norm",
                "tie_word_embeddings",
                "gradient_checkpointing",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")

    @classmethod
    def from_upstream(
        cls,
        values: Mapping[str, Any],
        *,
        expected_model_type: str,
    ) -> FishTransformerConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Fish transformer configuration must be a mapping.")
        required = {
            "vocab_size",
            "dim",
            "intermediate_size",
            "n_layer",
            "n_head",
            "head_dim",
            "max_seq_len",
        }
        missing = sorted(required - set(values))
        if missing:
            raise ValueError("Fish transformer config is missing: " + ", ".join(missing) + ".")
        if values.get("model_type") != expected_model_type:
            raise ValueError("Fish transformer config requires "
                             f"`model_type={expected_model_type!r}`.")
        if values.get("use_moe", False):
            raise ValueError(
                "This Fish checkpoint enables an MoE graph that the dense "
                "S2 architecture cannot execute.")
        return cls(
            vocab_size=values["vocab_size"],
            hidden_size=values["dim"],
            intermediate_size=values["intermediate_size"],
            num_hidden_layers=values["n_layer"],
            num_attention_heads=values["n_head"],
            num_key_value_heads=values.get(
                "n_local_heads",
                values["n_head"],
            ),
            head_dim=values["head_dim"],
            max_position_embeddings=values["max_seq_len"],
            rope_theta=values.get("rope_base", 1_000_000.0),
            rms_norm_eps=values.get("norm_eps", 1e-6),
            dropout=values.get("dropout", 0.0),
            attention_qkv_bias=values.get("attention_qkv_bias", False),
            attention_o_bias=values.get("attention_o_bias", False),
            attention_qk_norm=values.get("attention_qk_norm", False),
            tie_word_embeddings=values.get("tie_word_embeddings", False),
            initializer_range=values.get("initializer_range", 0.02),
            gradient_checkpointing=values.get(
                "use_gradient_checkpointing",
                True,
            ),
        )

    def to_upstream_dict(self, *, model_type: str) -> dict[str, Any]:
        return {
            "attention_o_bias": self.attention_o_bias,
            "attention_qk_norm": self.attention_qk_norm,
            "attention_qkv_bias": self.attention_qkv_bias,
            "dim": self.hidden_size,
            "dropout": float(self.dropout),
            "head_dim": self.head_dim,
            "initializer_range": self.initializer_range,
            "intermediate_size": self.intermediate_size,
            "max_seq_len": self.max_position_embeddings,
            "model_type": model_type,
            "n_head": self.num_attention_heads,
            "n_layer": self.num_hidden_layers,
            "n_local_heads": self.num_key_value_heads,
            "norm_eps": self.rms_norm_eps,
            "rope_base": self.rope_theta,
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_gradient_checkpointing": self.gradient_checkpointing,
            "vocab_size": self.vocab_size,
        }


@dataclass(frozen=True, slots=True)
class FishS2Config:
    """Complete slow/fast semantic architecture and protocol values."""

    text: FishTransformerConfig
    audio_decoder: FishTransformerConfig
    num_codebooks: int = 10
    codebook_size: int = 4096
    semantic_begin_id: int = 151_678
    semantic_end_id: int = 155_773
    end_of_text_id: int = 151_643
    im_start_id: int = 151_644
    im_end_id: int = 151_645
    pad_token_id: int = 151_669
    audio_pad_token_id: int = 151_677
    sample_rate: int = 44_100
    scale_codebook_embeddings: bool = True
    norm_fastlayer_input: bool = True
    source_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.text, FishTransformerConfig):
            raise TypeError("`text` must be a FishTransformerConfig.")
        if not isinstance(self.audio_decoder, FishTransformerConfig):
            raise TypeError("`audio_decoder` must be a FishTransformerConfig.")
        for name in (
                "num_codebooks",
                "codebook_size",
                "semantic_begin_id",
                "semantic_end_id",
                "end_of_text_id",
                "im_start_id",
                "im_end_id",
                "pad_token_id",
                "audio_pad_token_id",
                "sample_rate",
        ):
            value = getattr(self, name)
            if name.endswith("_id"):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"`{name}` must be a non-negative integer.")
            else:
                _positive_integer(value, name=name)
        if self.semantic_end_id - self.semantic_begin_id + 1 != self.codebook_size:
            raise ValueError("Fish semantic token range must contain exactly "
                             "`codebook_size` IDs.")
        if self.semantic_end_id >= self.text.vocab_size:
            raise ValueError("Fish semantic token range exceeds the text vocabulary.")
        protocol_ids = {
            "end_of_text_id": self.end_of_text_id,
            "im_start_id": self.im_start_id,
            "im_end_id": self.im_end_id,
            "pad_token_id": self.pad_token_id,
            "audio_pad_token_id": self.audio_pad_token_id,
        }
        outside_vocabulary = {
            name: value
            for name, value in protocol_ids.items() if value >= self.text.vocab_size
        }
        if outside_vocabulary:
            raise ValueError(
                "Fish protocol token IDs exceed the text vocabulary: "
                f"{outside_vocabulary!r}.")
        if self.audio_decoder.vocab_size != self.codebook_size:
            raise ValueError("Fish fast-decoder vocabulary must equal `codebook_size`.")
        if self.audio_decoder.max_position_embeddings < self.num_codebooks:
            raise ValueError("Fish fast-decoder context is shorter than the codebook count.")
        if self.text.hidden_size != self.audio_decoder.hidden_size:
            raise ValueError("The released Fish S2 graph requires equal slow and fast "
                             "hidden sizes.")
        if self.text.tie_word_embeddings is not True:
            raise ValueError("The published Fish S2 text head is tied to its embedding.")
        if self.audio_decoder.tie_word_embeddings:
            raise ValueError("The published Fish S2 fast output head is not tied.")
        for name in ("scale_codebook_embeddings", "norm_fastlayer_input"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if not isinstance(self.source_config, Mapping):
            raise TypeError("`source_config` must be a mapping.")
        object.__setattr__(
            self,
            "source_config",
            copy.deepcopy(dict(self.source_config)),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> FishS2Config:
        if not isinstance(values, Mapping):
            raise TypeError("Fish S2 configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        model_type = str(source.get("model_type", "")).lower()
        if model_type != "fish_qwen3_omni":
            raise ValueError("Native Fish S2 requires "
                             "`model_type='fish_qwen3_omni'`.")
        text_values = source.get("text_config")
        audio_values = source.get("audio_decoder_config")
        if not isinstance(text_values, Mapping) or not isinstance(
                audio_values,
                Mapping,
        ):
            raise ValueError("Fish S2 config requires `text_config` and "
                             "`audio_decoder_config` mappings.")
        text = FishTransformerConfig.from_upstream(
            text_values,
            expected_model_type="fish_qwen3",
        )
        audio = FishTransformerConfig.from_upstream(
            audio_values,
            expected_model_type="fish_qwen3_audio_decoder",
        )
        text_dimension = audio_values.get("text_dim", text.hidden_size)
        if (isinstance(text_dimension, bool) or not isinstance(text_dimension, int) or
                text_dimension != text.hidden_size):
            raise ValueError(
                "Fish audio-decoder `text_dim` must equal the slow "
                "transformer's hidden size.")
        num_codebooks = audio_values.get("num_codebooks")
        if num_codebooks is None:
            raise ValueError("Fish audio-decoder config must declare `num_codebooks`.")
        im_end_id = source.get(
            "im_end_token_id",
            source.get("eos_token_id", 151_645),
        )
        if ("eos_token_id" in source and source["eos_token_id"] != im_end_id):
            raise ValueError("Fish `eos_token_id` must identify the IM_END token.")
        return cls(
            text=text,
            audio_decoder=audio,
            num_codebooks=num_codebooks,
            codebook_size=audio.vocab_size,
            semantic_begin_id=source.get(
                "semantic_start_token_id",
                151_678,
            ),
            semantic_end_id=source.get(
                "semantic_end_token_id",
                155_773,
            ),
            end_of_text_id=source.get(
                "end_of_text_token_id",
                151_643,
            ),
            im_start_id=source.get("im_start_token_id", 151_644),
            im_end_id=im_end_id,
            pad_token_id=source.get("pad_token_id", 151_669),
            audio_pad_token_id=source.get(
                "audio_pad_token_id",
                151_677,
            ),
            sample_rate=source.get("sample_rate", 44_100),
            source_config=source,
        )

    @classmethod
    def tiny(
        cls,
        *,
        vocab_size: int = 64,
        codebook_size: int = 8,
        num_codebooks: int = 3,
        hidden_size: int = 16,
        num_hidden_layers: int = 2,
        num_fast_layers: int = 1,
    ) -> FishS2Config:
        """Construct a fully executable small graph for contract tests."""
        text = FishTransformerConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=hidden_size // 2,
            max_position_embeddings=64,
            attention_qk_norm=True,
            tie_word_embeddings=True,
            gradient_checkpointing=False,
        )
        fast = FishTransformerConfig(
            vocab_size=codebook_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_fast_layers,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=hidden_size // 2,
            max_position_embeddings=num_codebooks + 1,
            tie_word_embeddings=False,
            gradient_checkpointing=False,
        )
        semantic_begin = vocab_size - codebook_size
        return cls(
            text=text,
            audio_decoder=fast,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            semantic_begin_id=semantic_begin,
            semantic_end_id=vocab_size - 1,
            end_of_text_id=1,
            im_start_id=2,
            im_end_id=3,
            pad_token_id=0,
            audio_pad_token_id=0,
        )

    def to_dict(self) -> dict[str, Any]:
        source = copy.deepcopy(dict(self.source_config))
        source.update({
            "audio_decoder_config":
            self.audio_decoder.to_upstream_dict(model_type="fish_qwen3_audio_decoder", ),
            "audio_pad_token_id":
            self.audio_pad_token_id,
            "dtype":
            source.get("dtype", "bfloat16"),
            "end_of_text_token_id":
            self.end_of_text_id,
            "eos_token_id":
            self.im_end_id,
            "im_end_token_id":
            self.im_end_id,
            "im_start_token_id":
            self.im_start_id,
            "model_type":
            "fish_qwen3_omni",
            "pad_token_id":
            self.pad_token_id,
            "sample_rate":
            self.sample_rate,
            "semantic_end_token_id":
            self.semantic_end_id,
            "semantic_start_token_id":
            self.semantic_begin_id,
            "text_config":
            self.text.to_upstream_dict(model_type="fish_qwen3", ),
        })
        source["audio_decoder_config"]["num_codebooks"] = self.num_codebooks
        source["audio_decoder_config"]["text_dim"] = self.text.hidden_size
        return source


@dataclass(frozen=True, slots=True)
class FishCodecConfig:
    """Executable ModifiedDAC configuration published with S2-Pro."""

    sample_rate: int = 44_100
    encoder_dim: int = 64
    encoder_rates: tuple[int, ...] = (2, 4, 8, 8)
    decoder_dim: int = 1_536
    decoder_rates: tuple[int, ...] = (8, 8, 4, 2)
    encoder_transformer_layers: tuple[int, ...] = (0, 0, 0, 4)
    decoder_transformer_layers: tuple[int, ...] = (4, 0, 0, 0)
    semantic_codebook_size: int = 4_096
    residual_codebook_size: int = 1_024
    residual_codebooks: int = 9
    codebook_dim: int = 8
    quantizer_dropout: float = 0.5
    downsample_factors: tuple[int, ...] = (2, 2)
    transformer_layers: int = 8
    transformer_heads: int = 16
    transformer_hidden_size: int = 1_024
    transformer_intermediate_size: int = 3_072
    transformer_window_size: int = 128

    def __post_init__(self) -> None:
        for name in (
                "sample_rate",
                "encoder_dim",
                "decoder_dim",
                "semantic_codebook_size",
                "residual_codebook_size",
                "residual_codebooks",
                "codebook_dim",
                "transformer_layers",
                "transformer_heads",
                "transformer_hidden_size",
                "transformer_intermediate_size",
                "transformer_window_size",
        ):
            _positive_integer(getattr(self, name), name=name)
        for name in (
                "encoder_rates",
                "decoder_rates",
                "encoder_transformer_layers",
                "decoder_transformer_layers",
                "downsample_factors",
        ):
            values = tuple(getattr(self, name))
            if not values:
                raise ValueError(f"`{name}` cannot be empty.")
            if any(isinstance(item, bool) or not isinstance(item, int) or item < (
                    0 if "transformer_layers" in name else 1) for item in values):
                raise ValueError(f"`{name}` contains an invalid value.")
            object.__setattr__(self, name, values)
        if len(self.encoder_rates) != len(self.encoder_transformer_layers):
            raise ValueError("Encoder rates and transformer-layer declarations differ.")
        if len(self.decoder_rates) != len(self.decoder_transformer_layers):
            raise ValueError("Decoder rates and transformer-layer declarations differ.")
        if math.prod(self.encoder_rates) != math.prod(self.decoder_rates):
            raise ValueError("Fish codec encoder and decoder hops must match.")
        if (isinstance(self.quantizer_dropout, bool) or not isinstance(self.quantizer_dropout,
                                                                       (int, float)) or
                not 0.0 <= float(self.quantizer_dropout) <= 1.0):
            raise ValueError("`quantizer_dropout` must be in [0, 1].")

    @property
    def latent_dim(self) -> int:
        return self.encoder_dim * 2**len(self.encoder_rates)

    @property
    def hop_length(self) -> int:
        return math.prod(self.encoder_rates) * math.prod(self.downsample_factors)

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def num_codebooks(self) -> int:
        return self.residual_codebooks + 1

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["model_type"] = "fish_modified_dac"
        for name, value in tuple(output.items()):
            if isinstance(value, tuple):
                output[name] = list(value)
        return output

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> FishCodecConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Fish codec configuration must be a mapping.")
        source = dict(values)
        model_type = source.pop("model_type", "fish_modified_dac")
        if model_type != "fish_modified_dac":
            raise ValueError("Fish codec config requires "
                             "`model_type='fish_modified_dac'`.")
        known = set(cls.__dataclass_fields__)
        unexpected = sorted(set(source) - known)
        if unexpected:
            raise ValueError("Unsupported Fish codec configuration fields: " + ", ".join(unexpected) + ".")
        for name in (
                "encoder_rates",
                "decoder_rates",
                "encoder_transformer_layers",
                "decoder_transformer_layers",
                "downsample_factors",
        ):
            if name in source:
                source[name] = tuple(source[name])
        return cls(**source)


__all__ = [
    "FishCodecConfig",
    "FishS2Config",
    "FishTransformerConfig",
]
