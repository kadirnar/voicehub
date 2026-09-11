"""Validated configuration for the native Higgs Audio v2 graph.

The executable fields follow the Apache-2.0 Transformers implementation
at immutable revision
``af71155683b4d34dd92d8f037392fa6bf334035e`` and the official Boson
checkpoint at revision
``d80c511612b3040ff2877ce3d408747df1739f11``. VoiceHub does not import
Transformers or execute repository code.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any

from voicehub.models.causal_lm.native.configuration import LlamaConfig

_DEFAULT_ROPE = {
    "factor": 32.0,
    "high_freq_factor": 0.5,
    "low_freq_factor": 0.125,
    "original_max_position_embeddings": 1_024,
    "rope_theta": 500_000.0,
    "rope_type": "llama3",
}


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _token_id(name: str, value: Any, *, vocabulary_size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if not 0 <= value < vocabulary_size:
        raise ValueError(f"`{name}` must be in [0, {vocabulary_size}); found {value}.")
    return value


@dataclass(frozen=True, slots=True)
class HiggsAudioV2Config:
    """Complete dual-FFN text/audio decoder configuration."""

    vocab_size: int = 128_256
    hidden_size: int = 3_072
    intermediate_size: int = 8_192
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 2_048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    rope_parameters: Mapping[str, Any] = field(default_factory=lambda: dict(_DEFAULT_ROPE))
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    use_cache: bool = True
    pad_token_id: int = 128_001
    bos_token_id: int = 1
    eos_token_id: int = 128_009
    tie_word_embeddings: bool = False
    pretraining_tp: int = 1
    num_codebooks: int = 8
    codebook_size: int = 1_026
    audio_token_id: int = 128_016
    audio_bos_token_id: int = 128_013
    audio_delay_token_id: int = 128_014
    audio_stream_bos_id: int = 1_024
    audio_stream_eos_id: int = 1_025
    hidden_act: str = "silu"
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

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
                "num_codebooks",
                "codebook_size",
        ):
            _positive_integer(name, getattr(self, name))
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("`num_attention_heads` must be divisible by "
                             "`num_key_value_heads`.")
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("Higgs attention heads and `head_dim` must span "
                             "`hidden_size`.")
        if self.hidden_act != "silu":
            raise ValueError("Native Higgs Audio v2 supports only SiLU MLPs.")
        for name in (
                "attention_bias",
                "mlp_bias",
                "tie_word_embeddings",
                "use_cache",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.attention_bias or self.mlp_bias:
            raise ValueError(
                "The audited Higgs Audio v2 checkpoint uses bias-free "
                "attention and MLP projections.")
        if self.tie_word_embeddings:
            raise ValueError("Higgs Audio v2 publishes distinct text input/output weights.")
        if self.pretraining_tp != 1:
            raise ValueError("Native Higgs Audio v2 does not emulate serialized tensor "
                             "parallel slicing.")
        if self.initializer_range <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("`initializer_range` and `rms_norm_eps` must be positive.")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("`attention_dropout` must be in [0, 1).")
        for name in (
                "pad_token_id",
                "bos_token_id",
                "eos_token_id",
                "audio_token_id",
                "audio_bos_token_id",
                "audio_delay_token_id",
        ):
            _token_id(
                name,
                getattr(self, name),
                vocabulary_size=self.vocab_size,
            )
        for name in ("audio_stream_bos_id", "audio_stream_eos_id"):
            _token_id(
                name,
                getattr(self, name),
                vocabulary_size=self.codebook_size,
            )
        if self.audio_stream_bos_id == self.audio_stream_eos_id:
            raise ValueError("Higgs audio stream BOS and EOS IDs must differ.")
        if not isinstance(self.rope_parameters, Mapping):
            raise TypeError("`rope_parameters` must be a mapping.")
        rope = copy.deepcopy(dict(self.rope_parameters))
        if rope.get("rope_type") != "llama3":
            raise ValueError("The audited Higgs graph requires Llama-3 scaled RoPE.")
        object.__setattr__(
            self,
            "rope_parameters",
            MappingProxyType(rope),
        )
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )
        # Reuse the shared strict RoPE and decoder validation.
        self.as_causal_lm_config()

    @property
    def audio_vocabulary_size(self) -> int:
        return self.num_codebooks * self.codebook_size

    def as_causal_lm_config(self) -> LlamaConfig:
        """Return the structurally shared Llama attention/MLP contract."""
        rope = dict(self.rope_parameters)
        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=float(rope["rope_theta"]),
            rope_scaling=rope,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            mlp_bias=self.mlp_bias,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            tie_word_embeddings=False,
            pretraining_tp=self.pretraining_tp,
        )

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> HiggsAudioV2Config:
        if not isinstance(values, Mapping):
            raise TypeError("Higgs configuration values must be a mapping.")
        source = copy.deepcopy(dict(values))
        model_type = source.get("model_type", "higgs_audio_v2")
        if model_type != "higgs_audio_v2":
            raise ValueError("Native Higgs requires `model_type='higgs_audio_v2'`.")
        architectures = source.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        supported = {
            "HiggsAudioV2ForConditionalGeneration",
            "HiggsAudioV2Model",
        }
        if architectures and not set(architectures) <= supported:
            raise ValueError("The checkpoint does not declare a supported Higgs Audio v2 "
                             "architecture.")
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        consumed = canonical | {
            "_name_or_path",
            'architectures',
            "dtype",
            "model_type",
            "torch_dtype",
            "transformers_version",
        }
        extras = {name: value for name, value in source.items() if name not in consumed}
        supplied_extras = source.get("extra_config")
        if supplied_extras is not None:
            if not isinstance(supplied_extras, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied_extras)))
        return cls(**resolved, extra_config=extras)

    @classmethod
    def coerce(
        cls,
        value: HiggsAudioV2Config | Mapping[str, Any],
    ) -> HiggsAudioV2Config:
        return value if isinstance(value, cls) else cls.from_dict(value)

    @classmethod
    def tiny(cls) -> HiggsAudioV2Config:
        """Return a small graph with the same executable contracts."""
        return replace(
            cls(),
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            num_codebooks=2,
            codebook_size=6,
            audio_token_id=3,
            audio_bos_token_id=4,
            audio_delay_token_id=5,
            audio_stream_bos_id=4,
            audio_stream_eos_id=5,
            rope_parameters={
                "factor": 2.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 64,
                "rope_theta": 10_000.0,
                "rope_type": "llama3",
            },
        )

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            result[item.name] = (copy.deepcopy(dict(value)) if isinstance(value, Mapping) else value)
        result.update({
            'architectures': ["HiggsAudioV2ForConditionalGeneration"],
            "model_type": "higgs_audio_v2",
        })
        return result


__all__ = ["HiggsAudioV2Config"]
