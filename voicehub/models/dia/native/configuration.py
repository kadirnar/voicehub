"""Validated configuration for VoiceHub's native Dia graph.

The field names intentionally match the published
``nari-labs/Dia-1.6B-0626`` configuration.  Keeping the public schema intact
lets VoiceHub load the released Safetensors without executing provider code or
depending on a model framework.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _finite_positive(name: str, value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or
            value <= 0):
        raise ValueError(f"`{name}` must be finite and positive.")
    return float(value)


def _integer_tuple(name: str, value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence of integers.")
    result = tuple(value)
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise TypeError(f"`{name}` must contain only integers.")
    return result


def _extra_values(
    source: Mapping[str, Any],
    owner: type,
    *,
    consumed: set[str] | None = None,
) -> dict[str, Any]:
    known = {item.name for item in fields(owner)}
    known.discard("extra_config")
    known.update(consumed or ())
    extras = {name: copy.deepcopy(value) for name, value in source.items() if name not in known}
    supplied = source.get("extra_config")
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        extras.update(copy.deepcopy(dict(supplied)))
    return extras


@dataclass(frozen=True, slots=True)
class DiaEncoderConfig:
    """Byte-text encoder parameters."""

    hidden_size: int = 1_024
    intermediate_size: int = 4_096
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    max_position_embeddings: int = 1_024
    vocab_size: int = 256
    hidden_act: str = "silu"
    norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    initializer_range: float = 0.02
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
                "vocab_size",
        ):
            _positive_integer(name, getattr(self, name))
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("`num_attention_heads` must be divisible by "
                             "`num_key_value_heads`.")
        if self.hidden_act != "silu":
            raise ValueError("Native Dia currently implements `hidden_act='silu'`.")
        _finite_positive("norm_eps", self.norm_eps)
        _finite_positive("rope_theta", self.rope_theta)
        _finite_positive("initializer_range", self.initializer_range)
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DiaEncoderConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Dia encoder configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        aliases = {
            "n_embd": "hidden_size",
            "n_hidden": "intermediate_size",
            "n_layer": "num_hidden_layers",
            "n_head": "num_attention_heads",
        }
        for old, new in aliases.items():
            if old in source and new not in source:
                source[new] = source[old]
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        return cls(
            **resolved,
            extra_config=_extra_values(
                source,
                cls,
                consumed=set(aliases) | {"model_type", "rope_scaling"},
            ),
        )

    @classmethod
    def coerce(
        cls,
        value: DiaEncoderConfig | Mapping[str, Any],
    ) -> DiaEncoderConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name != "extra_config":
                result[item.name] = getattr(self, item.name)
        result["model_type"] = "dia_encoder"
        result.setdefault("rope_scaling", None)
        return result


@dataclass(frozen=True, slots=True)
class DiaDecoderConfig:
    """Nine-codebook autoregressive decoder parameters."""

    hidden_size: int = 2_048
    intermediate_size: int = 8_192
    num_hidden_layers: int = 18
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    cross_hidden_size: int = 1_024
    cross_num_attention_heads: int = 16
    cross_num_key_value_heads: int = 16
    cross_head_dim: int = 128
    max_position_embeddings: int = 3_072
    vocab_size: int = 1_028
    num_channels: int = 9
    hidden_act: str = "silu"
    norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    initializer_range: float = 0.02
    bos_token_id: int = 1_026
    eos_token_id: int = 1_024
    pad_token_id: int = 1_025
    use_cache: bool = True
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "cross_hidden_size",
                "cross_num_attention_heads",
                "cross_num_key_value_heads",
                "cross_head_dim",
                "max_position_embeddings",
                "vocab_size",
                "num_channels",
        ):
            _positive_integer(name, getattr(self, name))
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("`num_attention_heads` must be divisible by "
                             "`num_key_value_heads`.")
        if self.cross_num_attention_heads % self.cross_num_key_value_heads:
            raise ValueError(
                "`cross_num_attention_heads` must be divisible by "
                "`cross_num_key_value_heads`.")
        if self.hidden_act != "silu":
            raise ValueError("Native Dia currently implements `hidden_act='silu'`.")
        _finite_positive("norm_eps", self.norm_eps)
        _finite_positive("rope_theta", self.rope_theta)
        _finite_positive("initializer_range", self.initializer_range)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"`{name}` must be an integer.")
            if not 0 <= value < self.vocab_size:
                raise ValueError(f"`{name}` must be within decoder vocabulary range.")
        if len({self.bos_token_id, self.eos_token_id, self.pad_token_id}) != 3:
            raise ValueError("Dia BOS, EOS, and PAD token IDs must be distinct.")
        if not isinstance(self.use_cache, bool):
            raise TypeError("`use_cache` must be a boolean.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DiaDecoderConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Dia decoder configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        aliases = {
            "n_embd": "hidden_size",
            "n_hidden": "intermediate_size",
            "n_layer": "num_hidden_layers",
            "gqa_query_heads": "num_attention_heads",
            "kv_heads": "num_key_value_heads",
            "gqa_head_dim": "head_dim",
            "cross_query_heads": "cross_num_attention_heads",
        }
        for old, new in aliases.items():
            if old in source and new not in source:
                source[new] = source[old]
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        return cls(
            **resolved,
            extra_config=_extra_values(
                source,
                cls,
                consumed=set(aliases) | {"model_type", "rope_scaling"},
            ),
        )

    @classmethod
    def coerce(
        cls,
        value: DiaDecoderConfig | Mapping[str, Any],
    ) -> DiaDecoderConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name != "extra_config":
                result[item.name] = getattr(self, item.name)
        result["model_type"] = "dia_decoder"
        result.setdefault("rope_scaling", None)
        return result


@dataclass(frozen=True, slots=True)
class DiaArchitectureConfig:
    """Complete native Dia architecture and token protocol."""

    encoder_config: DiaEncoderConfig = field(default_factory=DiaEncoderConfig)
    decoder_config: DiaDecoderConfig = field(default_factory=DiaDecoderConfig)
    delay_pattern: tuple[int, ...] = (0, 8, 9, 10, 11, 12, 13, 14, 15)
    initializer_range: float = 0.02
    use_cache: bool = True
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "encoder_config",
            DiaEncoderConfig.coerce(self.encoder_config),
        )
        object.__setattr__(
            self,
            "decoder_config",
            DiaDecoderConfig.coerce(self.decoder_config),
        )
        delays = _integer_tuple("delay_pattern", self.delay_pattern)
        if any(delay < 0 for delay in delays):
            raise ValueError("`delay_pattern` values must be non-negative.")
        if len(delays) != self.decoder_config.num_channels:
            raise ValueError("`delay_pattern` length must equal decoder `num_channels`.")
        if len(set(delays)) != len(delays):
            raise ValueError("`delay_pattern` values must be unique.")
        if delays[0] != 0:
            raise ValueError("Dia channel zero must have zero delay.")
        object.__setattr__(self, "delay_pattern", delays)
        _finite_positive("initializer_range", self.initializer_range)
        if not isinstance(self.use_cache, bool):
            raise TypeError("`use_cache` must be a boolean.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DiaArchitectureConfig:
        """Parse either the converted or original Dia configuration schema."""
        if not isinstance(values, Mapping):
            raise TypeError("Dia configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        model_type = source.get("model_type", "dia")
        converted = "encoder_config" in source or "decoder_config" in source
        if converted and str(model_type).lower() != "dia":
            raise ValueError(f"Native Dia requires `model_type='dia'`; found {model_type!r}.")

        if converted:
            encoder_values = source.get("encoder_config", {})
            decoder_values = source.get("decoder_config", {})
            delay_pattern = source.get(
                "delay_pattern",
                cls().delay_pattern,
            )
        else:
            model = source.get("model")
            data = source.get("data")
            if not isinstance(model, Mapping) or not isinstance(data, Mapping):
                raise ValueError(
                    "Dia configuration must use the converted `encoder_config`/"
                    "`decoder_config` schema or the original `model`/`data` "
                    "schema.")
            encoder_values = model.get("encoder", {})
            decoder_values = dict(model.get("decoder", {}))
            decoder_values.update({
                "vocab_size": model.get("tgt_vocab_size", 1_028),
                "num_channels": data.get("channels", 9),
                "bos_token_id": data.get("audio_bos_value", 1_026),
                "eos_token_id": data.get("audio_eos_value", 1_024),
                "pad_token_id": data.get("audio_pad_value", 1_025),
                "cross_hidden_size": dict(encoder_values).get(
                    "n_embd",
                    1_024,
                ),
                "cross_num_key_value_heads": dict(encoder_values).get(
                    "n_head",
                    16,
                ),
            })
            encoder_values = dict(encoder_values)
            encoder_values["vocab_size"] = model.get("src_vocab_size", 256)
            for key in (
                    "normalization_layer_epsilon",
                    "rope_max_timescale",
            ):
                if key in model:
                    target = {
                        "normalization_layer_epsilon": "norm_eps",
                        "rope_max_timescale": "rope_theta",
                    }[key]
                    encoder_values.setdefault(target, model[key])
                    decoder_values.setdefault(target, model[key])
            delay_pattern = data.get(
                "delay_pattern",
                cls().delay_pattern,
            )

        top_level_ids = {
            "bos_token_id": "bos_token_id",
            "eos_token_id": "eos_token_id",
            "pad_token_id": "pad_token_id",
        }
        decoder_values = dict(decoder_values)
        for source_name, target_name in top_level_ids.items():
            if source_name in source:
                decoder_values[target_name] = source[source_name]

        return cls(
            encoder_config=DiaEncoderConfig.from_dict(encoder_values),
            decoder_config=DiaDecoderConfig.from_dict(decoder_values),
            delay_pattern=tuple(delay_pattern),
            initializer_range=source.get("initializer_range", 0.02),
            use_cache=source.get(
                "use_cache",
                decoder_values.get("use_cache", True),
            ),
            extra_config=_extra_values(
                source,
                cls,
                consumed={
                    'architectures',
                    "bos_token_id",
                    "data",
                    "decoder_config",
                    "encoder_config",
                    "eos_token_id",
                    "is_encoder_decoder",
                    "model",
                    "model_type",
                    "norm_eps",
                    "pad_token_id",
                    "torch_dtype",
                    "training",
                    "transformers_version",
                    "version",
                },
            ),
        )

    @classmethod
    def coerce(
        cls,
        value: DiaArchitectureConfig | Mapping[str, Any],
    ) -> DiaArchitectureConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        result.update({
            'architectures': ["DiaForConditionalGeneration"],
            "bos_token_id": self.decoder_config.bos_token_id,
            "decoder_config": self.decoder_config.to_dict(),
            "delay_pattern": list(self.delay_pattern),
            "encoder_config": self.encoder_config.to_dict(),
            "eos_token_id": self.decoder_config.eos_token_id,
            "initializer_range": self.initializer_range,
            "is_encoder_decoder": True,
            "model_type": "dia",
            "norm_eps": self.decoder_config.norm_eps,
            "pad_token_id": self.decoder_config.pad_token_id,
            "use_cache": self.use_cache,
        })
        return result


# Public architecture-level name.
DiaConfig = DiaArchitectureConfig

__all__ = [
    "DiaArchitectureConfig",
    "DiaConfig",
    "DiaDecoderConfig",
    "DiaEncoderConfig",
]
