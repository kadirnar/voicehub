"""Validated configurations for VoiceHub's native decoder-only LMs.

The field semantics were reviewed against Hugging Face Transformers'
Granite, Llama, Qwen2, and Qwen3 implementations at immutable revision
``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``.  This module is an
independent VoiceHub implementation and does not import Transformers.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

TRANSFORMERS_CAUSAL_LM_REVISION = ("ebea912f0bb6f9e28ad2df04acd9b4df035933a9")
SUPPORTED_CAUSAL_LM_FAMILIES = frozenset({
    "granite",
    "llama",
    "qwen2",
    "qwen3",
})


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}; found {value}.")
    return value


def _positive_float(name: str, value: Any) -> float:
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
        raise ValueError(f"`{name}` must be finite and in the interval [0, 1).")
    return result


def _token_id(
    name: str,
    value: Any,
    *,
    vocabulary_size: int,
) -> int | None:
    if value is None:
        return None
    result = _integer(name, value)
    if result >= vocabulary_size:
        raise ValueError(f"`{name}` must be smaller than `vocab_size`; found {result}.")
    return result


def _eos_token_ids(
    value: int | Sequence[int] | None,
    *,
    vocabulary_size: int,
) -> int | tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        values = (value, )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if not values:
            raise ValueError("`eos_token_id` cannot be an empty sequence.")
    else:
        raise TypeError("`eos_token_id` must be an integer, a sequence of integers, or None.")
    normalized = tuple(_token_id(
        "eos_token_id",
        item,
        vocabulary_size=vocabulary_size,
    ) for item in values)
    if any(item is None for item in normalized):
        raise TypeError("`eos_token_id` sequences cannot contain None.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("`eos_token_id` cannot contain duplicates.")
    return normalized[0] if len(normalized) == 1 else normalized


def _validate_rope(
    rope_scaling: Mapping[str, Any] | None,
    *,
    rope_theta: float,
    max_position_embeddings: int,
) -> None:
    if rope_scaling is None:
        return
    if not isinstance(rope_scaling, Mapping):
        raise TypeError("`rope_scaling` must be a mapping or None.")
    rope_type = rope_scaling.get(
        "rope_type",
        rope_scaling.get("type", "default"),
    )
    if rope_type not in (None, "default", "llama3"):
        raise ValueError(
            "This native decoder supports default RoPE and Llama-3 RoPE; "
            f"received rope type {rope_type!r}.")
    declared_theta = rope_scaling.get("rope_theta")
    if declared_theta is not None and float(declared_theta) != rope_theta:
        raise ValueError("`rope_scaling.rope_theta` conflicts with top-level `rope_theta`.")
    for name in ("partial_rotary_factor", "attention_factor"):
        value = rope_scaling.get(name)
        if value is not None and float(value) != 1.0:
            raise ValueError(f"Native RoPE requires `{name}=1.0`; found {value!r}.")
    common = {
        "attention_factor",
        "partial_rotary_factor",
        "rope_theta",
        "rope_type",
        "type",
    }
    if rope_type == "llama3":
        required = {
            "factor",
            "high_freq_factor",
            "low_freq_factor",
            "original_max_position_embeddings",
        }
        missing = {name for name in required if rope_scaling.get(name) is None}
        if missing:
            raise ValueError(
                "Llama-3 RoPE is missing required parameters: "
                f"{', '.join(sorted(missing))}.")
        factor = _positive_float("rope_scaling.factor", rope_scaling["factor"])
        low = _positive_float(
            "rope_scaling.low_freq_factor",
            rope_scaling["low_freq_factor"],
        )
        high = _positive_float(
            "rope_scaling.high_freq_factor",
            rope_scaling["high_freq_factor"],
        )
        if factor < 1.0:
            raise ValueError("`rope_scaling.factor` must be at least 1.0.")
        if high <= low:
            raise ValueError("`rope_scaling.high_freq_factor` must be greater than "
                             "`low_freq_factor`.")
        original = _integer(
            "rope_scaling.original_max_position_embeddings",
            rope_scaling["original_max_position_embeddings"],
            minimum=1,
        )
        if original > max_position_embeddings:
            raise ValueError(
                "`rope_scaling.original_max_position_embeddings` cannot "
                "exceed `max_position_embeddings`.")
        common |= required
    unsupported = {
        name: value
        for name, value in rope_scaling.items() if name not in common and value is not None
    }
    if unsupported:
        raise ValueError(
            "Native RoPE received unsupported parameters: "
            f"{', '.join(sorted(unsupported))}.")


@dataclass(frozen=True, slots=True)
class CausalLMConfig:
    """Common executable configuration for Llama/Qwen decoder backbones.

    Unknown checkpoint metadata is retained in :attr:`extra_config`.
    Features which alter model mathematics but are not implemented are
    rejected during construction, preventing a checkpoint from loading
    into a subtly incompatible graph.
    """

    model_type: str = "llama"
    vocab_size: int = 32_000
    hidden_size: int = 4_096
    intermediate_size: int = 11_008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2_048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    rope_scaling: Mapping[str, Any] | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    embedding_multiplier: float = 1.0
    logits_scaling: float = 1.0
    residual_multiplier: float = 1.0
    attention_multiplier: float | None = None
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | Sequence[int] | None = 2
    tie_word_embeddings: bool = False
    pretraining_tp: int = 1
    use_sliding_window: bool = False
    sliding_window: int | None = None
    max_window_layers: int | None = None
    layer_types: tuple[str, ...] | None = None
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model_type, str):
            raise TypeError("`model_type` must be a string.")
        model_type = self.model_type.strip().lower().replace("-", "_")
        aliases = {
            "qwen_2": "qwen2",
            "qwen_3": "qwen3",
        }
        model_type = aliases.get(model_type, model_type)
        if model_type not in SUPPORTED_CAUSAL_LM_FAMILIES:
            choices = ", ".join(sorted(SUPPORTED_CAUSAL_LM_FAMILIES))
            raise ValueError(
                f"Unsupported causal-LM `model_type` {self.model_type!r}; "
                f"expected one of {choices}.")
        object.__setattr__(self, "model_type", model_type)

        for name in (
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "max_position_embeddings",
        ):
            _integer(name, getattr(self, name), minimum=1)

        key_value_heads = self.num_key_value_heads
        if key_value_heads is None:
            key_value_heads = self.num_attention_heads
            object.__setattr__(self, "num_key_value_heads", key_value_heads)
        _integer("num_key_value_heads", key_value_heads, minimum=1)
        if self.num_attention_heads % key_value_heads:
            raise ValueError("`num_key_value_heads` must divide `num_attention_heads`.")

        head_dim = self.head_dim
        if head_dim is None:
            if self.hidden_size % self.num_attention_heads:
                raise ValueError(
                    "`hidden_size` must be divisible by `num_attention_heads` "
                    "when `head_dim` is not explicit.")
            head_dim = self.hidden_size // self.num_attention_heads
            object.__setattr__(self, "head_dim", head_dim)
        _integer("head_dim", head_dim, minimum=1)
        if head_dim % 2:
            raise ValueError("`head_dim` must be even for rotary embeddings.")

        if not isinstance(self.hidden_act, str):
            raise TypeError("`hidden_act` must be a string.")
        if self.hidden_act != "silu":
            raise ValueError("Llama/Qwen checkpoint parity currently requires "
                             "`hidden_act='silu'`.")
        object.__setattr__(
            self,
            "initializer_range",
            _positive_float("initializer_range", self.initializer_range),
        )
        object.__setattr__(
            self,
            "rms_norm_eps",
            _positive_float("rms_norm_eps", self.rms_norm_eps),
        )
        rope_theta = _positive_float("rope_theta", self.rope_theta)
        if rope_theta <= 1.0:
            raise ValueError("`rope_theta` must be greater than one.")
        object.__setattr__(self, "rope_theta", rope_theta)
        object.__setattr__(
            self,
            "attention_dropout",
            _probability("attention_dropout", self.attention_dropout),
        )
        for name in (
                "embedding_multiplier",
                "logits_scaling",
                "residual_multiplier",
        ):
            object.__setattr__(
                self,
                name,
                _positive_float(name, getattr(self, name)),
            )
        if self.attention_multiplier is None:
            object.__setattr__(
                self,
                "attention_multiplier",
                self.head_dim**-0.5,
            )
        else:
            object.__setattr__(
                self,
                "attention_multiplier",
                _positive_float(
                    "attention_multiplier",
                    self.attention_multiplier,
                ),
            )

        for name in (
                "attention_bias",
                "mlp_bias",
                "use_cache",
                "tie_word_embeddings",
                "use_sliding_window",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.model_type == "qwen2" and self.attention_bias:
            raise ValueError(
                "Qwen2 has fixed Q/K/V-only projection biases; "
                "`attention_bias` must remain False.")
        if self.model_type in {"qwen2", "qwen3"} and self.mlp_bias:
            raise ValueError(
                f"{self.model_type} uses bias-free SwiGLU projections; "
                "`mlp_bias` must remain False.")
        _integer("pretraining_tp", self.pretraining_tp, minimum=1)
        if self.pretraining_tp != 1:
            raise ValueError(
                "Tensor-parallel checkpoint slicing is not part of the native "
                "graph; `pretraining_tp` must be 1.")

        object.__setattr__(
            self,
            "pad_token_id",
            _token_id(
                "pad_token_id",
                self.pad_token_id,
                vocabulary_size=self.vocab_size,
            ),
        )
        object.__setattr__(
            self,
            "bos_token_id",
            _token_id(
                "bos_token_id",
                self.bos_token_id,
                vocabulary_size=self.vocab_size,
            ),
        )
        object.__setattr__(
            self,
            "eos_token_id",
            _eos_token_ids(
                self.eos_token_id,
                vocabulary_size=self.vocab_size,
            ),
        )

        if self.use_sliding_window:
            raise ValueError(
                "Sliding-window attention is not implemented by this native "
                "decoder; use full attention or a compatible future backend.")
        if self.sliding_window is not None:
            _integer("sliding_window", self.sliding_window, minimum=1)
            object.__setattr__(self, "sliding_window", None)
        if self.max_window_layers is not None:
            _integer("max_window_layers", self.max_window_layers, minimum=0)
        if self.layer_types is not None:
            if isinstance(self.layer_types, (str, bytes)):
                raise TypeError("`layer_types` must be a sequence of strings or None.")
            layer_types = tuple(self.layer_types)
            if len(layer_types) != self.num_hidden_layers:
                raise ValueError("`layer_types` must contain one value per decoder layer.")
            if any(value != "full_attention" for value in layer_types):
                raise ValueError("Only `full_attention` layer types are currently supported.")
            object.__setattr__(self, "layer_types", layer_types)

        _validate_rope(
            self.rope_scaling,
            rope_theta=self.rope_theta,
            max_position_embeddings=self.max_position_embeddings,
        )
        if self.rope_scaling is not None:
            object.__setattr__(
                self,
                "rope_scaling",
                MappingProxyType(copy.deepcopy(dict(self.rope_scaling))),
            )
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @property
    def qkv_bias(self) -> bool:
        """Whether official-family Q/K/V projections contain biases."""
        return True if self.model_type == "qwen2" else self.attention_bias

    @property
    def attention_output_bias(self) -> bool:
        """Whether the attention output projection contains a bias."""
        return False if self.model_type == "qwen2" else self.attention_bias

    @property
    def uses_qk_norm(self) -> bool:
        """Whether projected Q/K heads are RMS-normalized before RoPE."""
        return self.model_type == "qwen3"

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        if self.eos_token_id is None:
            return ()
        if isinstance(self.eos_token_id, int):
            return (self.eos_token_id, )
        return tuple(self.eos_token_id)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> CausalLMConfig:
        """Parse a Hugging Face-compatible configuration mapping."""
        if not isinstance(values, Mapping):
            raise TypeError("Causal-LM configuration values must be a mapping.")
        source = copy.deepcopy(dict(values))
        model_type = str(source.get("model_type", "llama")).lower()
        if model_type in {"qwen2_moe", "qwen3_moe"}:
            raise ValueError(
                f"{model_type!r} is a mixture-of-experts architecture, not "
                "the dense causal-LM family implemented here.")

        rope_parameters = source.pop("rope_parameters", None)
        if rope_parameters is not None:
            legacy_rope = source.get("rope_scaling")
            if legacy_rope is not None and legacy_rope != rope_parameters:
                raise ValueError("`rope_parameters` conflicts with legacy `rope_scaling`.")
            source["rope_scaling"] = rope_parameters
            if (isinstance(rope_parameters, Mapping) and "rope_theta" in rope_parameters and
                    "rope_theta" not in source):
                source["rope_theta"] = rope_parameters["rope_theta"]

        canonical = {item.name for item in fields(CausalLMConfig) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        consumed = canonical | {"extra_config"}
        extras = {name: value for name, value in source.items() if name not in consumed}
        supplied_extras = source.get("extra_config")
        if supplied_extras is not None:
            if not isinstance(supplied_extras, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied_extras)))

        target: type[CausalLMConfig]
        if cls is CausalLMConfig:
            targets = {
                "granite": GraniteConfig,
                "llama": LlamaConfig,
                "qwen2": Qwen2Config,
                "qwen3": Qwen3Config,
            }
            try:
                target = targets[model_type]
            except KeyError:
                target = CausalLMConfig
        else:
            target = cls
            expected_model_type = {
                GraniteConfig: "granite",
                LlamaConfig: "llama",
                Qwen2Config: "qwen2",
                Qwen3Config: "qwen3",
            }.get(cls)
            if ("model_type" in source and expected_model_type is not None and
                    model_type != expected_model_type):
                raise ValueError(f"{cls.__name__} cannot parse model_type {model_type!r}.")
        if target is not CausalLMConfig:
            resolved.pop("model_type", None)
        return target(**resolved, extra_config=extras)

    @classmethod
    def coerce(
        cls,
        value: CausalLMConfig | Mapping[str, Any],
    ) -> CausalLMConfig:
        if isinstance(value, CausalLMConfig):
            if cls is not CausalLMConfig and not isinstance(value, cls):
                raise ValueError(f"Expected a {cls.__name__}, found {type(value).__name__}.")
            return value
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached Hugging Face-compatible mapping."""
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(CausalLMConfig):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            if isinstance(value, Mapping):
                value = copy.deepcopy(dict(value))
            elif isinstance(value, tuple):
                value = list(value)
            result[item.name] = value
        result.setdefault('architectures', [self.huggingface_architecture_name])
        return result

    @property
    def huggingface_architecture_name(self) -> str:
        return {
            "granite": "GraniteForCausalLM",
            "llama": "LlamaForCausalLM",
            "qwen2": "Qwen2ForCausalLM",
            "qwen3": "Qwen3ForCausalLM",
        }[self.model_type]


class GraniteConfig(CausalLMConfig):
    """IBM Granite configuration with its architecture multipliers."""

    __slots__ = ()

    def __init__(self, **values: Any) -> None:
        supplied = values.pop("model_type", "granite")
        if supplied != "granite":
            raise ValueError("GraniteConfig requires `model_type='granite'`.")
        super().__init__(model_type="granite", **values)


class LlamaConfig(CausalLMConfig):
    """Llama-family configuration with official defaults."""

    __slots__ = ()

    def __init__(self, **values: Any) -> None:
        supplied = values.pop("model_type", "llama")
        if supplied != "llama":
            raise ValueError("LlamaConfig requires `model_type='llama'`.")
        super().__init__(model_type="llama", **values)


class Qwen2Config(CausalLMConfig):
    """Dense Qwen2 configuration with official positional defaults."""

    __slots__ = ()

    def __init__(self, **values: Any) -> None:
        supplied = values.pop("model_type", "qwen2")
        if supplied != "qwen2":
            raise ValueError("Qwen2Config requires `model_type='qwen2'`.")
        values.setdefault("vocab_size", 151_936)
        values.setdefault("hidden_size", 4_096)
        values.setdefault("intermediate_size", 22_016)
        values.setdefault("num_hidden_layers", 32)
        values.setdefault("num_attention_heads", 32)
        values.setdefault("num_key_value_heads", 32)
        values.setdefault("max_position_embeddings", 32_768)
        values.setdefault("bos_token_id", None)
        values.setdefault("eos_token_id", None)
        values.setdefault("max_window_layers", 28)
        super().__init__(model_type="qwen2", **values)


class Qwen3Config(CausalLMConfig):
    """Dense Qwen3 configuration with per-head Q/K normalization."""

    __slots__ = ()

    def __init__(self, **values: Any) -> None:
        supplied = values.pop("model_type", "qwen3")
        if supplied != "qwen3":
            raise ValueError("Qwen3Config requires `model_type='qwen3'`.")
        values.setdefault("vocab_size", 151_936)
        values.setdefault("hidden_size", 4_096)
        values.setdefault("intermediate_size", 22_016)
        values.setdefault("num_hidden_layers", 32)
        values.setdefault("num_attention_heads", 32)
        values.setdefault("num_key_value_heads", 32)
        values.setdefault("head_dim", 128)
        values.setdefault("max_position_embeddings", 32_768)
        values.setdefault("bos_token_id", None)
        values.setdefault("eos_token_id", None)
        values.setdefault("max_window_layers", 28)
        super().__init__(model_type="qwen3", **values)


__all__ = [
    "SUPPORTED_CAUSAL_LM_FAMILIES",
    "TRANSFORMERS_CAUSAL_LM_REVISION",
    "CausalLMConfig",
    "GraniteConfig",
    "LlamaConfig",
    "Qwen2Config",
    "Qwen3Config",
]
