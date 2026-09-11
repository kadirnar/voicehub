"""Validated configuration for the native Kokoro decoder graph."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _finite_probability(name: str, value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or
            not 0.0 <= float(value) < 1.0):
        raise ValueError(f"`{name}` must be finite and in [0, 1).")
    return float(value)


def _integer_tuple(name: str, value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence of integers.")
    result = tuple(value)
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    for item in result:
        _positive_integer(name, item)
    return result


def _nested_integer_tuples(
    name: str,
    value: Any,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence of integer sequences.")
    result = tuple(_integer_tuple(f"{name}[{index}]", item) for index, item in enumerate(value))
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    return result


@dataclass(frozen=True, slots=True)
class KokoroAlbertConfig:
    """ALBERT parameters used by the released PL-BERT checkpoint.

    Defaults intentionally match ``transformers.AlbertConfig`` 4.48.3.
    Kokoro's public ``plbert.dropout`` key was not consumed by ALBERT
    and is retained as source metadata rather than silently changing
    inference.
    """

    vocab_size: int = 178
    embedding_size: int = 128
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_hidden_groups: int = 1
    num_attention_heads: int = 12
    intermediate_size: int = 2_048
    inner_group_num: int = 1
    hidden_act: str = "gelu_new"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    position_embedding_type: str = "absolute"
    pad_token_id: int = 0
    source_dropout: float | None = None
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "vocab_size",
                "embedding_size",
                "hidden_size",
                "num_hidden_layers",
                "num_hidden_groups",
                "num_attention_heads",
                "intermediate_size",
                "inner_group_num",
                "max_position_embeddings",
                "type_vocab_size",
        ):
            _positive_integer(name, getattr(self, name))
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        if self.num_hidden_layers % self.num_hidden_groups:
            raise ValueError("`num_hidden_layers` must be divisible by `num_hidden_groups`.")
        if self.hidden_act != "gelu_new":
            raise ValueError("Native Kokoro implements the released `gelu_new` PL-BERT.")
        _finite_probability("hidden_dropout_prob", self.hidden_dropout_prob)
        _finite_probability(
            "attention_probs_dropout_prob",
            self.attention_probs_dropout_prob,
        )
        if (isinstance(self.initializer_range, bool) or not isinstance(self.initializer_range,
                                                                       (int, float)) or
                not math.isfinite(float(self.initializer_range)) or self.initializer_range <= 0):
            raise ValueError("`initializer_range` must be finite and positive.")
        if (isinstance(self.layer_norm_eps, bool) or not isinstance(self.layer_norm_eps, (int, float)) or
                not math.isfinite(float(self.layer_norm_eps)) or self.layer_norm_eps <= 0):
            raise ValueError("`layer_norm_eps` must be finite and positive.")
        if self.position_embedding_type != "absolute":
            raise ValueError("The released Kokoro PL-BERT uses absolute position embeddings.")
        if (isinstance(self.pad_token_id, bool) or not isinstance(self.pad_token_id, int) or
                not 0 <= self.pad_token_id < self.vocab_size):
            raise ValueError("`pad_token_id` must be within the vocabulary.")
        if self.source_dropout is not None:
            _finite_probability("source_dropout", self.source_dropout)
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
        *,
        vocab_size: int,
    ) -> KokoroAlbertConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Kokoro `plbert` configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        source_dropout = source.pop("dropout", None)
        known = {
            "embedding_size",
            "hidden_size",
            "num_hidden_layers",
            "num_hidden_groups",
            "num_attention_heads",
            "intermediate_size",
            "inner_group_num",
            "hidden_act",
            "hidden_dropout_prob",
            "attention_probs_dropout_prob",
            "max_position_embeddings",
            "type_vocab_size",
            "initializer_range",
            "layer_norm_eps",
            "position_embedding_type",
            "pad_token_id",
        }
        resolved = {name: source.pop(name) for name in tuple(source) if name in known}
        return cls(
            vocab_size=vocab_size,
            source_dropout=source_dropout,
            extra_config=source,
            **resolved,
        )

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        result.update({
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_hidden_groups": self.num_hidden_groups,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "inner_group_num": self.inner_group_num,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": (self.attention_probs_dropout_prob),
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "position_embedding_type": self.position_embedding_type,
            "pad_token_id": self.pad_token_id,
        })
        if self.source_dropout is not None:
            result["dropout"] = self.source_dropout
        return result


@dataclass(frozen=True, slots=True)
class KokoroIstftNetConfig:
    """Released iSTFTNet generator dimensions."""

    upsample_kernel_sizes: tuple[int, ...] = (20, 12)
    upsample_rates: tuple[int, ...] = (10, 6)
    gen_istft_hop_size: int = 5
    gen_istft_n_fft: int = 20
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    upsample_initial_channel: int = 512

    def __post_init__(self) -> None:
        for name in (
                "gen_istft_hop_size",
                "gen_istft_n_fft",
                "upsample_initial_channel",
        ):
            _positive_integer(name, getattr(self, name))
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("`upsample_rates` and `upsample_kernel_sizes` must match.")
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilation_sizes):
            raise ValueError("`resblock_kernel_sizes` and `resblock_dilation_sizes` "
                             "must match.")
        if self.gen_istft_n_fft % 2:
            raise ValueError("`gen_istft_n_fft` must be even.")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> KokoroIstftNetConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Kokoro `istftnet` configuration must be a mapping.")
        required = {
            "upsample_kernel_sizes",
            "upsample_rates",
            "gen_istft_hop_size",
            "gen_istft_n_fft",
            "resblock_dilation_sizes",
            "resblock_kernel_sizes",
            "upsample_initial_channel",
        }
        missing = sorted(required - set(values))
        unknown = sorted(set(values) - required)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing {missing!r}")
            if unknown:
                details.append(f"unknown {unknown!r}")
            raise ValueError("Invalid Kokoro `istftnet` configuration: " + "; ".join(details))
        return cls(
            upsample_kernel_sizes=_integer_tuple(
                "upsample_kernel_sizes",
                values["upsample_kernel_sizes"],
            ),
            upsample_rates=_integer_tuple(
                "upsample_rates",
                values["upsample_rates"],
            ),
            gen_istft_hop_size=values["gen_istft_hop_size"],
            gen_istft_n_fft=values["gen_istft_n_fft"],
            resblock_dilation_sizes=_nested_integer_tuples(
                "resblock_dilation_sizes",
                values["resblock_dilation_sizes"],
            ),
            resblock_kernel_sizes=_integer_tuple(
                "resblock_kernel_sizes",
                values["resblock_kernel_sizes"],
            ),
            upsample_initial_channel=values["upsample_initial_channel"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "upsample_kernel_sizes": list(self.upsample_kernel_sizes),
            "upsample_rates": list(self.upsample_rates),
            "gen_istft_hop_size": self.gen_istft_hop_size,
            "gen_istft_n_fft": self.gen_istft_n_fft,
            "resblock_dilation_sizes": [list(item) for item in self.resblock_dilation_sizes],
            "resblock_kernel_sizes": list(self.resblock_kernel_sizes),
            "upsample_initial_channel": self.upsample_initial_channel,
        }


@dataclass(frozen=True, slots=True)
class KokoroArchitectureConfig:
    """Complete, checkpoint-shaped Kokoro decoder configuration."""

    vocab: Mapping[str, int]
    n_token: int = 178
    hidden_dim: int = 512
    n_layer: int = 3
    max_dur: int = 50
    dropout: float = 0.2
    style_dim: int = 128
    text_encoder_kernel_size: int = 5
    n_mels: int = 80
    dim_in: int = 64
    max_conv_dim: int = 512
    multispeaker: bool = True
    plbert: KokoroAlbertConfig | Mapping[str, Any] = field(default_factory=KokoroAlbertConfig)
    istftnet: KokoroIstftNetConfig | Mapping[str, Any] = field(default_factory=KokoroIstftNetConfig)
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "n_token",
                "hidden_dim",
                "n_layer",
                "max_dur",
                "style_dim",
                "text_encoder_kernel_size",
                "n_mels",
                "dim_in",
                "max_conv_dim",
        ):
            _positive_integer(name, getattr(self, name))
        _finite_probability("dropout", self.dropout)
        if not isinstance(self.multispeaker, bool):
            raise TypeError("`multispeaker` must be a boolean.")
        if not isinstance(self.vocab, Mapping) or not self.vocab:
            raise ValueError("Kokoro `vocab` must be a non-empty mapping.")
        vocabulary: dict[str, int] = {}
        for symbol, token_id in self.vocab.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("Kokoro vocabulary symbols must be non-empty.")
            if (isinstance(token_id, bool) or not isinstance(token_id, int) or
                    not 0 <= token_id < self.n_token):
                raise ValueError(f"Kokoro token ID for {symbol!r} is outside [0, "
                                 f"{self.n_token}).")
            vocabulary[symbol] = token_id
        if len(set(vocabulary.values())) != len(vocabulary):
            raise ValueError("Kokoro vocabulary token IDs must be unique.")
        object.__setattr__(
            self,
            "vocab",
            MappingProxyType(vocabulary),
        )
        plbert = (
            self.plbert if isinstance(self.plbert, KokoroAlbertConfig) else KokoroAlbertConfig.from_dict(
                self.plbert,
                vocab_size=self.n_token,
            ))
        if plbert.vocab_size != self.n_token:
            raise ValueError("PL-BERT vocabulary size must equal `n_token`.")
        object.__setattr__(self, "plbert", plbert)
        object.__setattr__(
            self,
            "istftnet",
            (
                self.istftnet if isinstance(self.istftnet, KokoroIstftNetConfig) else
                KokoroIstftNetConfig.from_dict(self.istftnet)),
        )
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
    ) -> KokoroArchitectureConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Kokoro architecture configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        required = {"vocab", "plbert", "istftnet"}
        missing = sorted(required - set(source))
        if missing:
            raise ValueError(f"Kokoro configuration is missing {missing!r}.")
        known = {
            "vocab",
            "n_token",
            "hidden_dim",
            "n_layer",
            "max_dur",
            "dropout",
            "style_dim",
            "text_encoder_kernel_size",
            "n_mels",
            "dim_in",
            "max_conv_dim",
            "multispeaker",
            "plbert",
            "istftnet",
        }
        resolved = {name: source.pop(name) for name in tuple(source) if name in known}
        return cls(
            **resolved,
            extra_config=source,
        )

    @classmethod
    def coerce(
        cls,
        value: KokoroArchitectureConfig | Mapping[str, Any],
    ) -> KokoroArchitectureConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        result.update({
            "vocab": dict(self.vocab),
            "n_token": self.n_token,
            "hidden_dim": self.hidden_dim,
            "n_layer": self.n_layer,
            "max_dur": self.max_dur,
            "dropout": self.dropout,
            "style_dim": self.style_dim,
            "text_encoder_kernel_size": self.text_encoder_kernel_size,
            "n_mels": self.n_mels,
            "dim_in": self.dim_in,
            "max_conv_dim": self.max_conv_dim,
            "multispeaker": self.multispeaker,
            "plbert": self.plbert.to_dict(),
            "istftnet": self.istftnet.to_dict(),
        })
        return result


__all__ = [
    "KokoroAlbertConfig",
    "KokoroArchitectureConfig",
    "KokoroIstftNetConfig",
]
