"""Validated configuration for the VoiceHub-native Zonos v0.1 graph.

The public Zonos checkpoints describe both a dense Transformer and a
hybrid Mamba-2 model with the same JSON envelope.  VoiceHub currently
implements the dense Transformer exactly.  Hybrid configurations are
rejected before model allocation instead of being routed through a graph
with incompatible tensor names or numerics.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _finite_positive(name: str, value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or
            value <= 0):
        raise ValueError(f"`{name}` must be finite and greater than zero.")
    return float(value)


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a mapping.")
    return MappingProxyType(copy.deepcopy(dict(value)))


@dataclass(frozen=True, slots=True)
class ZonosBackboneConfig:
    """Checkpoint-visible dense Transformer parameters."""

    d_model: int = 2_048
    d_intermediate: int = 0
    attn_mlp_d_intermediate: int = 8_192
    n_layer: int = 26
    ssm_cfg: Mapping[str, Any] = field(default_factory=dict)
    attn_layer_idx: tuple[int, ...] = tuple(range(26))
    attn_cfg: Mapping[str, Any] = field(
        default_factory=lambda: {
            "causal": True,
            "num_heads": 16,
            "num_heads_kv": 4,
            "rotary_emb_dim": 128,
            "rotary_emb_interleaved": True,
            "qkv_proj_bias": False,
            "out_proj_bias": False,
        })
    rms_norm: bool = False
    residual_in_fp32: bool = False
    norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        for name in (
                "d_model",
                "attn_mlp_d_intermediate",
                "n_layer",
        ):
            _positive_integer(name, getattr(self, name))
        if (isinstance(self.d_intermediate, bool) or not isinstance(self.d_intermediate, int) or
                self.d_intermediate < 0):
            raise ValueError("`d_intermediate` must be a non-negative integer.")
        object.__setattr__(self, "ssm_cfg", _mapping("ssm_cfg", self.ssm_cfg))
        object.__setattr__(self, "attn_cfg", _mapping("attn_cfg", self.attn_cfg))
        if isinstance(self.attn_layer_idx, (str, bytes)) or not isinstance(
                self.attn_layer_idx,
                Sequence,
        ):
            raise TypeError("`attn_layer_idx` must be an integer sequence.")
        layer_indices = tuple(self.attn_layer_idx)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in layer_indices):
            raise TypeError("`attn_layer_idx` must contain integers.")
        object.__setattr__(self, "attn_layer_idx", layer_indices)
        object.__setattr__(
            self,
            "norm_epsilon",
            _finite_positive("norm_epsilon", self.norm_epsilon),
        )
        for name in ("rms_norm", "residual_in_fp32"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        self._validate_native_transformer()

    def _validate_native_transformer(self) -> None:
        if self.ssm_cfg:
            raise NotImplementedError(
                "VoiceHub native Zonos supports the published v0.1 dense "
                "Transformer checkpoint. The hybrid Mamba-2 checkpoint has a "
                "different state-space graph and is intentionally rejected.")
        expected_layers = tuple(range(self.n_layer))
        if self.attn_layer_idx != expected_layers:
            raise ValueError(
                "Dense Zonos requires every layer to be an attention layer; "
                f"expected {expected_layers!r}, received "
                f"{self.attn_layer_idx!r}.")
        if self.d_intermediate != 0:
            raise ValueError("The native dense Zonos graph requires `d_intermediate=0`.")
        if self.rms_norm:
            raise ValueError("The published dense Zonos checkpoint uses LayerNorm, not "
                             "RMSNorm.")
        if self.residual_in_fp32:
            raise ValueError(
                "The published dense Zonos checkpoint does not use fp32 "
                "residual accumulation.")
        required = {
            "causal",
            "num_heads",
            "num_heads_kv",
            "rotary_emb_dim",
            "qkv_proj_bias",
            "out_proj_bias",
        }
        missing = required - set(self.attn_cfg)
        if missing:
            raise ValueError("Zonos attention configuration is missing "
                             f"{sorted(missing)!r}.")
        if self.attn_cfg["causal"] is not True:
            raise ValueError("The Zonos acoustic language model must be causal.")
        num_heads = _positive_integer(
            "attn_cfg.num_heads",
            self.attn_cfg["num_heads"],
        )
        num_heads_kv = _positive_integer(
            "attn_cfg.num_heads_kv",
            self.attn_cfg["num_heads_kv"],
        )
        if self.d_model % num_heads:
            raise ValueError("`d_model` must be divisible by `num_heads`.")
        if num_heads % num_heads_kv:
            raise ValueError("`num_heads` must be divisible by `num_heads_kv`.")
        head_dim = self.d_model // num_heads
        if self.attn_cfg["rotary_emb_dim"] != head_dim:
            raise ValueError(
                "Native Zonos requires rotary embeddings over the complete "
                f"head dimension ({head_dim}).")
        if self.attn_cfg.get("rotary_emb_interleaved", True) is not True:
            raise ValueError("Native Zonos implements the published interleaved rotary "
                             "layout only.")
        if self.attn_cfg["qkv_proj_bias"] is not False:
            raise ValueError("The published Zonos QKV projection has no bias.")
        if self.attn_cfg["out_proj_bias"] is not False:
            raise ValueError("The published Zonos output projection has no bias.")

    @property
    def num_heads(self) -> int:
        return int(self.attn_cfg["num_heads"])

    @property
    def num_heads_kv(self) -> int:
        return int(self.attn_cfg["num_heads_kv"])

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "d_intermediate": self.d_intermediate,
            "attn_mlp_d_intermediate": self.attn_mlp_d_intermediate,
            "n_layer": self.n_layer,
            "ssm_cfg": copy.deepcopy(dict(self.ssm_cfg)),
            "attn_layer_idx": list(self.attn_layer_idx),
            "attn_cfg": copy.deepcopy(dict(self.attn_cfg)),
            "rms_norm": self.rms_norm,
            "residual_in_fp32": self.residual_in_fp32,
            "norm_epsilon": self.norm_epsilon,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ZonosBackboneConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Zonos `backbone` must be a mapping.")
        resolved = copy.deepcopy(dict(values))
        if "attn_layer_idx" in resolved:
            resolved["attn_layer_idx"] = tuple(resolved["attn_layer_idx"])
        return cls(**resolved)


@dataclass(frozen=True, slots=True)
class ZonosPrefixConditionerConfig:
    """Ordered prefix-conditioner declaration from ``config.json``."""

    conditioners: tuple[Mapping[str, Any], ...] = (
        MappingProxyType({
            "type": "EspeakPhonemeConditioner",
            "name": "espeak",
        }),
        MappingProxyType({
            "cond_dim": 128,
            "uncond_type": "learned",
            "projection": "linear",
            "type": "PassthroughConditioner",
            "name": "speaker",
        }),
        MappingProxyType({
            "input_dim": 8,
            "uncond_type": "learned",
            "type": "FourierConditioner",
            "name": "emotion",
        }),
        MappingProxyType({
            "min_val": 0,
            "max_val": 24_000,
            "uncond_type": "learned",
            "type": "FourierConditioner",
            "name": "fmax",
        }),
        MappingProxyType({
            "min_val": 0,
            "max_val": 400,
            "uncond_type": "learned",
            "type": "FourierConditioner",
            "name": "pitch_std",
        }),
        MappingProxyType({
            "min_val": 0,
            "max_val": 40,
            "uncond_type": "learned",
            "type": "FourierConditioner",
            "name": "speaking_rate",
        }),
        MappingProxyType({
            "min_val": -1,
            "max_val": 126,
            "uncond_type": "learned",
            "type": "IntegerConditioner",
            "name": "language_id",
        }),
    )
    projection: str = "linear"

    def __post_init__(self) -> None:
        if isinstance(self.conditioners, (str, bytes)) or not isinstance(
                self.conditioners,
                Sequence,
        ):
            raise TypeError("`conditioners` must be a sequence of mappings.")
        conditioners = tuple(_mapping("conditioner", value) for value in self.conditioners)
        if not conditioners:
            raise ValueError("Zonos requires at least one prefix conditioner.")
        names: list[str] = []
        supported = {
            "EspeakPhonemeConditioner",
            "FourierConditioner",
            "IntegerConditioner",
            "PassthroughConditioner",
        }
        for index, conditioner in enumerate(conditioners):
            name = conditioner.get("name")
            kind = conditioner.get("type")
            if not isinstance(name, str) or not name:
                raise ValueError(f"Zonos conditioner {index} requires a non-empty `name`.")
            if kind not in supported:
                raise ValueError(f"Unsupported Zonos conditioner type {kind!r}.")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("Zonos conditioner names must be unique.")
        if self.projection not in {"none", "linear", "mlp"}:
            raise ValueError("`prefix_conditioner.projection` must be none, linear, or mlp.")
        object.__setattr__(self, "conditioners", conditioners)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditioners": [copy.deepcopy(dict(conditioner)) for conditioner in self.conditioners],
            "projection": self.projection,
        }

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> ZonosPrefixConditionerConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Zonos `prefix_conditioner` must be a mapping.")
        resolved = copy.deepcopy(dict(values))
        if "conditioners" in resolved:
            resolved["conditioners"] = tuple(resolved["conditioners"])
        return cls(**resolved)


@dataclass(frozen=True, slots=True)
class ZonosArchitectureConfig:
    """Complete, strict configuration for ``Zonos-v0.1-transformer``."""

    backbone: ZonosBackboneConfig = field(default_factory=ZonosBackboneConfig)
    prefix_conditioner: ZonosPrefixConditionerConfig = field(default_factory=ZonosPrefixConditionerConfig, )
    eos_token_id: int = 1_024
    masked_token_id: int = 1_025
    pad_vocab_to_multiple_of: int = 8
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.backbone, ZonosBackboneConfig):
            raise TypeError("`backbone` must be a ZonosBackboneConfig.")
        if not isinstance(
                self.prefix_conditioner,
                ZonosPrefixConditionerConfig,
        ):
            raise TypeError("`prefix_conditioner` must be a "
                            "ZonosPrefixConditionerConfig.")
        for name in (
                "eos_token_id",
                "masked_token_id",
                "pad_vocab_to_multiple_of",
        ):
            _positive_integer(name, getattr(self, name))
        if self.eos_token_id != self.codebook_size:
            raise ValueError("The published Zonos graph requires EOS token 1024.")
        if self.masked_token_id != self.codebook_size + 1:
            raise ValueError("The published Zonos graph requires mask token 1025.")
        object.__setattr__(
            self,
            "extra_config",
            _mapping("extra_config", self.extra_config),
        )

    @property
    def num_codebooks(self) -> int:
        return 9

    @property
    def codebook_size(self) -> int:
        return 1_024

    @property
    def input_vocab_size(self) -> int:
        return self.masked_token_id + 1

    @property
    def output_vocab_size(self) -> int:
        return self.eos_token_id + 1

    @property
    def sample_rate(self) -> int:
        return 44_100

    @property
    def hop_length(self) -> int:
        return 512

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> ZonosArchitectureConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Zonos architecture configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        backbone = ZonosBackboneConfig.from_dict(source.pop("backbone"))
        prefix = ZonosPrefixConditionerConfig.from_dict(source.pop("prefix_conditioner"), )
        known = {
            "eos_token_id",
            "masked_token_id",
            "pad_vocab_to_multiple_of",
        }
        resolved = {name: source.pop(name) for name in tuple(source) if name in known}
        supplied_extras = source.pop("extra_config", None)
        if supplied_extras is not None:
            if not isinstance(supplied_extras, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            source.update(copy.deepcopy(dict(supplied_extras)))
        return cls(
            backbone=backbone,
            prefix_conditioner=prefix,
            extra_config=source,
            **resolved,
        )

    def to_dict(self) -> dict[str, Any]:
        values = copy.deepcopy(dict(self.extra_config))
        values.update({
            "backbone": self.backbone.to_dict(),
            "prefix_conditioner": self.prefix_conditioner.to_dict(),
            "eos_token_id": self.eos_token_id,
            "masked_token_id": self.masked_token_id,
            "pad_vocab_to_multiple_of": self.pad_vocab_to_multiple_of,
        })
        return values


__all__ = [
    "ZonosArchitectureConfig",
    "ZonosBackboneConfig",
    "ZonosPrefixConditionerConfig",
]
