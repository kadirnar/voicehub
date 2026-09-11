"""Validated configuration for VoiceHub's native Moonshine ASR family.

The executable fields follow Hugging Face Transformers' Moonshine
implementation at immutable revision
``d0e91c7ff1d3ab41b49de369a396f65191e34d2b``.  The default dimensions are
the official ``UsefulSensors/moonshine-tiny`` configuration at revision
``390624ed33d594443aa4aa221f5b9f283b545b5a``.  Neither upstream package is
imported or executed.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

TRANSFORMERS_MOONSHINE_REVISION = "d0e91c7ff1d3ab41b49de369a396f65191e34d2b"
MOONSHINE_MAIN_LIBRARY_REVISION = "cc1695646a560f2eec7f7c058f3c4d580f039e4b"

_ACTIVATIONS = frozenset({"gelu", "silu"})


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}; found {value}.")
    return value


def _positive_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"`{name}` must be finite and greater than zero.")
    return result


def _probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"`{name}` must be finite and in [0, 1).")
    return result


def _unit_interval(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"`{name}` must be finite and in [0, 1].")
    return result


@dataclass(frozen=True, slots=True)
class MoonshineConfig:
    """Executable Moonshine convolutional encoder-decoder configuration.

    Moonshine checkpoints use a learned raw-waveform convolutional
    frontend, bidirectional rotary encoder blocks, and causal rotary
    decoder blocks with cross-attention.  Unknown declarative metadata
    is retained in :attr:`extra_config`; graph-changing features not
    implemented here are rejected instead of being silently ignored.
    """

    vocab_size: int = 32_768
    hidden_size: int = 288
    intermediate_size: int = 1_152
    encoder_num_hidden_layers: int = 6
    decoder_num_hidden_layers: int = 6
    encoder_num_attention_heads: int = 8
    decoder_num_attention_heads: int = 8
    encoder_num_key_value_heads: int | None = None
    decoder_num_key_value_heads: int | None = None
    encoder_hidden_act: str = "gelu"
    decoder_hidden_act: str = "silu"
    max_position_embeddings: int = 194
    initializer_range: float = 0.02
    attention_bias: bool = False
    attention_dropout: float = 0.0
    partial_rotary_factor: float = 0.9
    rope_theta: float = 10_000.0
    rope_scaling: Mapping[str, Any] | None = None
    pad_head_dim_to_multiple_of: int | None = 8
    decoder_start_token_id: int = 1
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 2
    use_cache: bool = True
    tie_word_embeddings: bool = True
    sampling_rate: int = 16_000
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
                "encoder_num_hidden_layers",
                "decoder_num_hidden_layers",
                "encoder_num_attention_heads",
                "decoder_num_attention_heads",
                "max_position_embeddings",
                "sampling_rate",
        ):
            _integer(name, getattr(self, name), minimum=1)
        for name in (
                "decoder_start_token_id",
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
        ):
            token_id = _integer(name, getattr(self, name))
            if token_id >= self.vocab_size:
                raise ValueError(f"`{name}` must be smaller than `vocab_size`; found "
                                 f"{token_id}.")

        encoder_key_value_heads = self.encoder_num_key_value_heads
        if encoder_key_value_heads is None:
            encoder_key_value_heads = self.encoder_num_attention_heads
            object.__setattr__(
                self,
                "encoder_num_key_value_heads",
                encoder_key_value_heads,
            )
        decoder_key_value_heads = self.decoder_num_key_value_heads
        if decoder_key_value_heads is None:
            decoder_key_value_heads = self.decoder_num_attention_heads
            object.__setattr__(
                self,
                "decoder_num_key_value_heads",
                decoder_key_value_heads,
            )
        for name, heads, key_value_heads in (
            (
                "encoder",
                self.encoder_num_attention_heads,
                encoder_key_value_heads,
            ),
            (
                "decoder",
                self.decoder_num_attention_heads,
                decoder_key_value_heads,
            ),
        ):
            _integer(f"{name}_num_key_value_heads", key_value_heads, minimum=1)
            if self.hidden_size % heads:
                raise ValueError(f"`hidden_size` must be divisible by "
                                 f"`{name}_num_attention_heads`.")
            if heads % key_value_heads:
                raise ValueError(
                    f"`{name}_num_key_value_heads` must divide "
                    f"`{name}_num_attention_heads`.")
            if heads != key_value_heads:
                raise ValueError(
                    "Native Moonshine currently accepts the multi-head "
                    "attention layout used by every published official "
                    "checkpoint; grouped-query variants are not supported.")

        for name in ("encoder_hidden_act", "decoder_hidden_act"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"`{name}` must be a string.")
            if value not in _ACTIVATIONS:
                choices = ", ".join(sorted(_ACTIVATIONS))
                raise ValueError(f"`{name}` must be one of {choices}; found {value!r}.")
        _positive_real("initializer_range", self.initializer_range)
        _positive_real("rope_theta", self.rope_theta)
        _probability("attention_dropout", self.attention_dropout)
        partial = _unit_interval(
            "partial_rotary_factor",
            self.partial_rotary_factor,
        )
        if partial == 0.0:
            raise ValueError("`partial_rotary_factor` must be greater than zero.")

        for name in ("attention_bias", "use_cache", "tie_word_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if not self.tie_word_embeddings:
            raise ValueError(
                "Published Moonshine Safetensors tie the output projection to "
                "the decoder embedding; `tie_word_embeddings=False` is "
                "unsupported.")
        if self.rope_scaling is not None:
            if not isinstance(self.rope_scaling, Mapping):
                raise TypeError("`rope_scaling` must be a mapping or None.")
            rope_type = self.rope_scaling.get(
                "rope_type",
                self.rope_scaling.get("type", "default"),
            )
            if rope_type not in (None, "default"):
                raise ValueError("Native Moonshine currently supports default RoPE only.")
            unsupported = {
                name
                for name, value in self.rope_scaling.items()
                if name not in {"rope_type", "type"} and value is not None
            }
            if unsupported:
                raise ValueError(
                    "Native Moonshine received unsupported RoPE parameters: "
                    f"{', '.join(sorted(unsupported))}.")

        multiple = self.pad_head_dim_to_multiple_of
        if multiple is not None:
            _integer("pad_head_dim_to_multiple_of", multiple, minimum=1)
        for heads in (
                self.encoder_num_attention_heads,
                self.decoder_num_attention_heads,
        ):
            head_dim = self.hidden_size // heads
            rotary_dim = int(head_dim * self.partial_rotary_factor)
            if rotary_dim <= 0 or rotary_dim % 2:
                raise ValueError(
                    "`partial_rotary_factor` must produce a positive even "
                    "rotary dimension for every attention head.")

        if self.sampling_rate != 16_000:
            raise ValueError("Published Moonshine checkpoints require `sampling_rate=16000`.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> MoonshineConfig:
        """Parse one Hugging Face-compatible Moonshine configuration."""
        if not isinstance(values, Mapping):
            raise TypeError("Moonshine configuration values must be a mapping.")
        source = copy.deepcopy(dict(values))
        canonical_names = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical_names if name in source}

        # Older official exports used one shared attention-head count.
        shared_heads = source.get("n_head")
        if shared_heads is not None:
            for name in (
                    "encoder_num_attention_heads",
                    "decoder_num_attention_heads",
            ):
                if name in resolved and resolved[name] != shared_heads:
                    raise ValueError(f"`{name}` conflicts with legacy `n_head`.")
                resolved.setdefault(name, shared_heads)
        legacy_aliases = {
            "dim": "hidden_size",
            "enc_depth": "encoder_num_hidden_layers",
            "dec_depth": "decoder_num_hidden_layers",
            "dec_voc_size": "vocab_size",
        }
        for legacy_name, canonical_name in legacy_aliases.items():
            if legacy_name not in source:
                continue
            if (canonical_name in resolved and resolved[canonical_name] != source[legacy_name]):
                raise ValueError(f"`{canonical_name}` conflicts with legacy "
                                 f"`{legacy_name}`.")
            resolved.setdefault(canonical_name, source[legacy_name])

        if source.get("is_encoder_decoder", True) is not True:
            raise ValueError("Moonshine requires `is_encoder_decoder=True`.")
        model_type = str(source.get("model_type", "moonshine")).strip().lower()
        if model_type not in {"asr_moonshine", "moonshine"}:
            raise ValueError(
                "Native Moonshine requires model type 'moonshine'; received "
                f"{model_type or '<missing>'!r}.")
        if source.get("dec_ff_swiglu") is False:
            raise ValueError(
                "Native published Moonshine checkpoints require the decoder "
                "SwiGLU feed-forward graph.")

        consumed = canonical_names | {
            "dec_depth",
            "dec_ff_swiglu",
            "dec_voc_size",
            "dim",
            "enc_depth",
            "extra_config",
            "is_encoder_decoder",
            "n_head",
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
        value: MoonshineConfig | Mapping[str, Any],
    ) -> MoonshineConfig:
        """Return ``value`` as a validated configuration."""
        if isinstance(value, cls):
            return value
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached Hugging Face-compatible configuration."""
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name != "extra_config":
                result[item.name] = getattr(self, item.name)
        result.update({
            'architectures': ["MoonshineForConditionalGeneration"],
            "is_encoder_decoder": True,
            "model_type": "moonshine",
        })
        return result

    @property
    def encoder_head_dim(self) -> int:
        return self.hidden_size // self.encoder_num_attention_heads

    @property
    def decoder_head_dim(self) -> int:
        return self.hidden_size // self.decoder_num_attention_heads

    @property
    def minimum_input_samples(self) -> int:
        """Shortest waveform accepted by all three frontend convolutions."""
        required = 1
        for kernel, stride in reversed(((127, 64), (7, 3), (3, 2))):
            required = (required - 1) * stride + kernel
        return required

    @property
    def input_to_feature_ratio(self) -> int:
        return 64 * 3 * 2

    def feature_output_length(self, input_samples: int) -> int:
        """Return the exact frontend length for a waveform sample count."""
        length = _integer("input_samples", input_samples)
        for kernel, stride in ((127, 64), (7, 3), (3, 2)):
            length = (length - kernel) // stride + 1
            if length <= 0:
                return 0
        return length
