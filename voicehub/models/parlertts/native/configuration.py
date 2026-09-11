"""Validated configuration for the VoiceHub-native Parler-TTS graph."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

from voicehub.models.dac.native.configuration import DacConfig


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _probability(name: str, value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or
            not 0.0 <= value <= 1.0):
        raise ValueError(f"`{name}` must be finite and in the interval [0, 1].")
    return float(value)


def _optional_weights(
    value: Sequence[float] | None,
    *,
    expected: int,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("`codebook_weights` must be a numeric sequence or None.")
    weights = tuple(float(item) for item in value)
    if len(weights) != expected:
        raise ValueError(f"`codebook_weights` contains {len(weights)} values; expected "
                         f"{expected}.")
    if any(not math.isfinite(item) or item < 0.0 for item in weights):
        raise ValueError("`codebook_weights` must be finite and non-negative.")
    if not any(weights):
        raise ValueError("At least one codebook weight must be positive.")
    return weights


@dataclass(frozen=True, slots=True)
class T5EncoderConfig:
    """Subset of T5 required by the released Parler-TTS text encoder."""

    vocab_size: int = 32_128
    d_model: int = 1_024
    d_kv: int = 64
    d_ff: int = 2_816
    num_layers: int = 24
    num_heads: int = 16
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    dense_act_fn: str = "gelu_new"
    pad_token_id: int = 0
    eos_token_id: int = 1
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "vocab_size",
                "d_model",
                "d_kv",
                "d_ff",
                "num_layers",
                "num_heads",
                "relative_attention_num_buckets",
                "relative_attention_max_distance",
        ):
            _positive_integer(name, getattr(self, name))
        if self.d_model != self.d_kv * self.num_heads:
            raise ValueError("Parler-TTS T5 requires `d_model == d_kv * num_heads`.")
        _probability("dropout_rate", self.dropout_rate)
        if (isinstance(self.layer_norm_epsilon, bool) or not isinstance(self.layer_norm_epsilon,
                                                                        (int, float)) or
                not math.isfinite(self.layer_norm_epsilon) or self.layer_norm_epsilon <= 0.0):
            raise ValueError("`layer_norm_epsilon` must be finite and positive.")
        if self.dense_act_fn not in {"gelu", "gelu_new"}:
            raise ValueError("Native Parler-TTS supports T5 `gelu` and `gelu_new`.")
        for name in ("pad_token_id", "eos_token_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"`{name}` must be a non-negative integer.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> T5EncoderConfig:
        if not isinstance(values, Mapping):
            raise TypeError("T5 configuration values must be a mapping.")
        source = copy.deepcopy(dict(values))
        if str(source.get("model_type", "t5")).lower() != "t5":
            raise ValueError("Parler-TTS requires a T5 text encoder.")
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        extras = {
            name: value
            for name, value in source.items() if name not in canonical | {"extra_config", "model_type"}
        }
        if "extra_config" in source:
            supplied = source["extra_config"]
            if not isinstance(supplied, Mapping):
                raise TypeError("T5 `extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**resolved, extra_config=extras)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name != "extra_config":
                result[item.name] = getattr(self, item.name)
        result["model_type"] = "t5"
        return result


@dataclass(frozen=True, slots=True)
class ParlerDecoderConfig:
    """Checkpoint-compatible autoregressive acoustic decoder configuration."""

    vocab_size: int = 1_088
    max_position_embeddings: int = 4_096
    num_hidden_layers: int = 24
    ffn_dim: int = 4_096
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    num_cross_attention_key_value_heads: int = 16
    hidden_size: int = 1_024
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    layerdrop: float = 0.0
    scale_embedding: bool = False
    num_codebooks: int = 9
    pad_token_id: int = 1_024
    bos_token_id: int = 1_025
    eos_token_id: int = 1_024
    rope_embeddings: bool = False
    rope_theta: float = 10_000.0
    codebook_weights: tuple[float, ...] | None = None
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "vocab_size",
                "max_position_embeddings",
                "num_hidden_layers",
                "ffn_dim",
                "num_attention_heads",
                "num_key_value_heads",
                "num_cross_attention_key_value_heads",
                "hidden_size",
                "num_codebooks",
        ):
            _positive_integer(name, getattr(self, name))
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be divisible by attention heads.")
        for name in (
                "num_key_value_heads",
                "num_cross_attention_key_value_heads",
        ):
            if self.num_attention_heads % getattr(self, name):
                raise ValueError(f"`num_attention_heads` must be divisible by `{name}`.")
        for name in (
                "dropout",
                "attention_dropout",
                "activation_dropout",
                "layerdrop",
        ):
            _probability(name, getattr(self, name))
        if self.activation_function not in {"gelu", "gelu_new", "relu", "silu"}:
            raise ValueError(f"Unsupported activation {self.activation_function!r}.")
        for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"`{name}` must be a non-negative integer.")
        if self.pad_token_id >= self.vocab_size + 1:
            raise ValueError("`pad_token_id` exceeds the decoder embedding table.")
        if (isinstance(self.rope_theta, bool) or not isinstance(self.rope_theta, (int, float)) or
                not math.isfinite(self.rope_theta) or self.rope_theta <= 0.0):
            raise ValueError("`rope_theta` must be finite and positive.")
        object.__setattr__(
            self,
            "codebook_weights",
            _optional_weights(
                self.codebook_weights,
                expected=self.num_codebooks,
            ),
        )
        for name in ("scale_embedding", "rope_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ParlerDecoderConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Parler decoder configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        if str(source.get("model_type", "parler_tts_decoder")).lower() != ("parler_tts_decoder"):
            raise ValueError("Configuration is not a Parler-TTS decoder.")
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        if "codebook_weights" in resolved and resolved["codebook_weights"] is not None:
            resolved["codebook_weights"] = tuple(resolved["codebook_weights"])
        extras = {
            name: value
            for name, value in source.items() if name not in canonical | {"extra_config", "model_type"}
        }
        if "extra_config" in source:
            supplied = source["extra_config"]
            if not isinstance(supplied, Mapping):
                raise TypeError("Decoder `extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**resolved, extra_config=extras)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            result[item.name] = (list(value) if isinstance(value, tuple) else value)
        result["model_type"] = "parler_tts_decoder"
        return result


@dataclass(frozen=True, slots=True)
class ParlerTTSArchitectureConfig:
    """Composite text encoder, acoustic decoder, and DAC configuration."""

    text_encoder: T5EncoderConfig = field(default_factory=T5EncoderConfig)
    audio_encoder: DacConfig = field(default_factory=DacConfig)
    decoder: ParlerDecoderConfig = field(default_factory=ParlerDecoderConfig)
    vocab_size: int = 32_128
    prompt_cross_attention: bool = False
    decoder_start_token_id: int = 1_025
    pad_token_id: int = 1_024
    sampling_rate: int = 44_100
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.text_encoder, T5EncoderConfig):
            object.__setattr__(
                self,
                "text_encoder",
                T5EncoderConfig.from_dict(self.text_encoder),
            )
        if not isinstance(self.audio_encoder, DacConfig):
            object.__setattr__(
                self,
                "audio_encoder",
                DacConfig.from_dict(self.audio_encoder),
            )
        if not isinstance(self.decoder, ParlerDecoderConfig):
            object.__setattr__(
                self,
                "decoder",
                ParlerDecoderConfig.from_dict(self.decoder),
            )
        for name in ("vocab_size", "sampling_rate"):
            _positive_integer(name, getattr(self, name))
        if not isinstance(self.prompt_cross_attention, bool):
            raise TypeError("`prompt_cross_attention` must be a boolean.")
        for name in ("decoder_start_token_id", "pad_token_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"`{name}` must be a non-negative integer.")
        if self.decoder.num_codebooks != self.audio_encoder.n_codebooks:
            raise ValueError("Decoder and DAC must declare the same number of codebooks.")
        if self.decoder.vocab_size < self.audio_encoder.codebook_size:
            raise ValueError("Decoder vocabulary cannot be smaller than the DAC codebook.")
        if self.sampling_rate != self.audio_encoder.sampling_rate:
            raise ValueError("Composite and DAC sampling rates must be identical.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> ParlerTTSArchitectureConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Parler-TTS configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        if str(source.get("model_type", "parler_tts")).lower() != "parler_tts":
            raise ValueError("Configuration is not a Parler-TTS checkpoint.")
        try:
            text_values = source.pop("text_encoder")
            audio_values = source.pop("audio_encoder")
            decoder_values = source.pop("decoder")
        except KeyError as error:
            raise ValueError("Parler-TTS config requires text, audio, and decoder sections.") from error
        canonical = {
            item.name
            for item in fields(cls) if item.name not in {
                "text_encoder",
                "audio_encoder",
                "decoder",
                "extra_config",
            }
        }
        resolved = {name: source.pop(name) for name in tuple(source) if name in canonical}
        source.pop("model_type", None)
        supplied_extras = source.pop("extra_config", None)
        if supplied_extras is not None:
            if not isinstance(supplied_extras, Mapping):
                raise TypeError("Composite `extra_config` must be a mapping.")
            source.update(copy.deepcopy(dict(supplied_extras)))
        if not isinstance(audio_values, Mapping):
            raise TypeError("Parler-TTS audio encoder config must be a mapping.")
        normalized_audio = copy.deepcopy(dict(audio_values))
        # The pinned wrapper predates Transformers' integrated DacModel and
        # serializes ``DACModel`` plus the wrapper-facing field name.
        architectures = normalized_audio.get('architectures')
        if architectures == ["DACModel"]:
            normalized_audio['architectures'] = ["DacModel"]
        if ("num_codebooks" in normalized_audio and "n_codebooks" not in normalized_audio):
            normalized_audio["n_codebooks"] = normalized_audio["num_codebooks"]
        audio_config = DacConfig.from_dict(normalized_audio)
        resolved.setdefault("sampling_rate", audio_config.sampling_rate)
        return cls(
            text_encoder=T5EncoderConfig.from_dict(text_values),
            audio_encoder=audio_config,
            decoder=ParlerDecoderConfig.from_dict(decoder_values),
            extra_config=source,
            **resolved,
        )

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for name in (
                "vocab_size",
                "prompt_cross_attention",
                "decoder_start_token_id",
                "pad_token_id",
                "sampling_rate",
        ):
            result[name] = getattr(self, name)
        result.update({
            "model_type": "parler_tts",
            "text_encoder": self.text_encoder.to_dict(),
            "audio_encoder": self.audio_encoder.to_dict(),
            "decoder": self.decoder.to_dict(),
        })
        return result


__all__ = [
    "ParlerDecoderConfig",
    "ParlerTTSArchitectureConfig",
    "T5EncoderConfig",
]
