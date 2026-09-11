"""Validated configuration for VoiceHub's native Wav2Vec2 CTC family.

Field semantics were reviewed against Hugging Face Transformers revision
``ebea912f0bb6f9e28ad2df04acd9b4df035933a9`` and the official
``facebook/wav2vec2-base-960h`` configuration at revision
``22aad52d435eb6dbaf354bdad9b0da84ce7d6156``.  This module does not import or
execute either upstream runtime.
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

_ACTIVATIONS = frozenset({"gelu", "gelu_new", "relu", "selu", "silu"})
_FEATURE_NORMALIZATIONS = frozenset({"group", "layer"})
_CTC_REDUCTIONS = frozenset({"mean", "none", "sum"})
_PROBLEM_TYPES = frozenset({
    "multi_label_classification",
    "regression",
    "single_label_classification",
})


def _require_integer(name: str, value: int, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}; found {value}.")


def _require_positive_real(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"`{name}` must be finite and greater than zero.")


def _require_probability(
    name: str,
    value: float,
    *,
    inclusive_one: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    upper_bound = value <= 1.0 if inclusive_one else value < 1.0
    if not math.isfinite(value) or value < 0.0 or not upper_bound:
        interval = "[0, 1]" if inclusive_one else "[0, 1)"
        raise ValueError(f"`{name}` must be finite and in the interval {interval}.")


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
        _require_integer(name, item, minimum=1)
    return result


@dataclass(frozen=True, slots=True)
class Wav2Vec2Config:
    """Executable configuration for Wav2Vec2-family CTC encoders.

    The convolutional frontend and Transformer normalization mode are
    explicit so compatible HuBERT and WavLM adapters can reuse the same
    validated dimension contract in later migrations.  Unknown JSON
    fields are retained in :attr:`extra_config`.
    """

    vocab_size: int = 32
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3_072
    hidden_act: str = "gelu"
    hidden_dropout: float = 0.1
    activation_dropout: float = 0.1
    attention_dropout: float = 0.1
    feat_proj_dropout: float = 0.0
    final_dropout: float = 0.1
    layerdrop: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    feat_extract_norm: str = "group"
    feat_extract_activation: str = "gelu"
    conv_dim: tuple[int, ...] = (
        512,
        512,
        512,
        512,
        512,
        512,
        512,
    )
    conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_bias: bool = False
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    do_stable_layer_norm: bool = False
    apply_spec_augment: bool = True
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    ctc_loss_reduction: str = "sum"
    ctc_zero_infinity: bool = False
    num_labels: int = 2
    classifier_proj_size: int = 256
    use_weighted_layer_sum: bool = False
    problem_type: str | None = None
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
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
                "num_hidden_layers",
                "num_attention_heads",
                "intermediate_size",
                "num_conv_pos_embeddings",
                "num_conv_pos_embedding_groups",
                "mask_time_length",
                "mask_feature_length",
                "sampling_rate",
                "num_labels",
                "classifier_proj_size",
        ):
            _require_integer(name, getattr(self, name), minimum=1)
        for name in ("mask_time_min_masks", "mask_feature_min_masks"):
            _require_integer(name, getattr(self, name), minimum=0)
        for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
            value = getattr(self, name)
            _require_integer(name, value, minimum=0)
            if value >= self.vocab_size:
                raise ValueError(f"`{name}` must be smaller than `vocab_size`; found {value}.")

        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        if self.hidden_size % self.num_conv_pos_embedding_groups:
            raise ValueError("`hidden_size` must be divisible by "
                             "`num_conv_pos_embedding_groups`.")

        for name in (
                "hidden_dropout",
                "activation_dropout",
                "attention_dropout",
                "feat_proj_dropout",
                "final_dropout",
        ):
            _require_probability(name, getattr(self, name))
        for name in ("layerdrop", "mask_time_prob", "mask_feature_prob"):
            _require_probability(
                name,
                getattr(self, name),
                inclusive_one=True,
            )
        for name in ("initializer_range", "layer_norm_eps"):
            _require_positive_real(name, getattr(self, name))

        for name in ("hidden_act", "feat_extract_activation"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"`{name}` must be a string.")
            if value not in _ACTIVATIONS:
                choices = ", ".join(sorted(_ACTIVATIONS))
                raise ValueError(f"`{name}` must be one of {choices}; found {value!r}.")
        if not isinstance(self.feat_extract_norm, str):
            raise TypeError("`feat_extract_norm` must be a string.")
        if self.feat_extract_norm not in _FEATURE_NORMALIZATIONS:
            choices = ", ".join(sorted(_FEATURE_NORMALIZATIONS))
            raise ValueError(
                "`feat_extract_norm` must be one of "
                f"{choices}; found {self.feat_extract_norm!r}.")

        for name in ("conv_dim", "conv_stride", "conv_kernel"):
            object.__setattr__(
                self,
                name,
                _positive_integer_tuple(name, getattr(self, name)),
            )
        layer_count = len(self.conv_dim)
        if len(self.conv_stride) != layer_count or len(self.conv_kernel) != layer_count:
            raise ValueError("`conv_dim`, `conv_stride`, and `conv_kernel` must have equal "
                             "lengths.")

        for name in (
                "conv_bias",
                "do_stable_layer_norm",
                "apply_spec_augment",
                "ctc_zero_infinity",
                "use_weighted_layer_sum",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if not isinstance(self.ctc_loss_reduction, str):
            raise TypeError("`ctc_loss_reduction` must be a string.")
        if self.ctc_loss_reduction not in _CTC_REDUCTIONS:
            choices = ", ".join(sorted(_CTC_REDUCTIONS))
            raise ValueError(
                "`ctc_loss_reduction` must be one of "
                f"{choices}; found {self.ctc_loss_reduction!r}.")
        if self.problem_type is not None:
            if not isinstance(self.problem_type, str):
                raise TypeError("`problem_type` must be a string or None.")
            if self.problem_type not in _PROBLEM_TYPES:
                choices = ", ".join(sorted(_PROBLEM_TYPES))
                raise ValueError(
                    f"`problem_type` must be one of {choices}; found "
                    f"{self.problem_type!r}.")

        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Wav2Vec2Config:
        """Parse a Hugging Face-compatible configuration mapping."""
        if not isinstance(values, Mapping):
            raise TypeError("Wav2Vec2 configuration values must be a mapping.")
        source = copy.deepcopy(dict(values))
        canonical_names = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical_names if name in source}

        hidden_dropout_alias = source.get("hidden_dropout_prob")
        if hidden_dropout_alias is not None:
            if ("hidden_dropout" in resolved and resolved["hidden_dropout"] != hidden_dropout_alias):
                raise ValueError("`hidden_dropout` conflicts with `hidden_dropout_prob`.")
            resolved.setdefault("hidden_dropout", hidden_dropout_alias)

        for name in ("conv_dim", "conv_stride", "conv_kernel"):
            if name in resolved:
                resolved[name] = tuple(resolved[name])

        declared_layers = source.get("num_feat_extract_layers")
        if declared_layers is not None:
            _require_integer(
                "num_feat_extract_layers",
                declared_layers,
                minimum=1,
            )
            conv_dimensions = resolved.get("conv_dim", cls().conv_dim)
            if declared_layers != len(conv_dimensions):
                raise ValueError("`num_feat_extract_layers` does not match `conv_dim`.")

        if source.get("add_adapter", False):
            raise ValueError(
                "Native Wav2Vec2 CTC does not yet support the optional "
                "language-adapter graph declared by `add_adapter=True`.")
        if source.get("adapter_attn_dim") is not None:
            raise ValueError("Native Wav2Vec2 CTC does not yet support attention adapters.")

        consumed = canonical_names | {
            "extra_config",
            "hidden_dropout_prob",
            "num_feat_extract_layers",
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
        value: Wav2Vec2Config | Mapping[str, Any],
    ) -> Wav2Vec2Config:
        """Return ``value`` as a validated configuration."""
        if isinstance(value, cls):
            return value
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached Hugging Face-compatible mapping."""
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            result[item.name] = list(value) if isinstance(value, tuple) else value
        result.setdefault("model_type", "wav2vec2")
        result.setdefault('architectures', ["Wav2Vec2ForCTC"])
        result["num_feat_extract_layers"] = self.num_feat_extract_layers
        return result

    @property
    def num_feat_extract_layers(self) -> int:
        """Number of convolutional frontend layers."""
        return len(self.conv_dim)

    @property
    def inputs_to_logits_ratio(self) -> int:
        """Nominal waveform samples represented by one output frame."""
        return reduce(mul, self.conv_stride, 1)

    @property
    def minimum_input_samples(self) -> int:
        """Shortest waveform that produces one feature frame."""
        required = 1
        for kernel, stride in reversed(tuple(zip(self.conv_kernel, self.conv_stride))):
            required = (required - 1) * stride + kernel
        return required

    def feature_output_length(self, input_samples: int) -> int:
        """Return exact valid feature frames for one waveform length."""
        _require_integer("input_samples", input_samples, minimum=0)
        length = input_samples
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            length = (length - kernel) // stride + 1
            if length <= 0:
                return 0
        return length


__all__ = ["Wav2Vec2Config"]
