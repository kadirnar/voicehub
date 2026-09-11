"""Validated configuration for VoiceHub's native LASR/MedASR graph."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

_ACTIVATIONS = frozenset({"silu"})
_CTC_REDUCTIONS = frozenset({"mean", "sum"})
_RELEASE_VARIANTS = frozenset({"medasr", "custom"})


def _integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}.")
    return value


def _positive_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"`{name}` must be finite and positive.")
    return result


def _probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"`{name}` must be finite and in [0, 1).")
    return result


def _weights(name: str, value: Any) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must contain two real numbers.")
    resolved = tuple(value)
    if len(resolved) != 2:
        raise ValueError(f"`{name}` must contain exactly two values.")
    output = []
    for item in resolved:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"`{name}` must contain real numbers.")
        item = float(item)
        if not math.isfinite(item):
            raise ValueError(f"`{name}` values must be finite.")
        output.append(item)
    return output[0], output[1]


@dataclass(frozen=True, slots=True)
class MedASRConfig:
    """Executable LASR CTC configuration.

    ``variant="medasr"`` locks graph-defining fields to the audited
    ``google/medasr`` release. ``variant="custom"`` is for VoiceHub-
    owned reduced graphs, tests, and derived artifacts; it does not
    imply compatibility with unrelated LASR checkpoints.
    """

    variant: str = "medasr"
    vocab_size: int = 512
    hidden_size: int = 512
    num_hidden_layers: int = 17
    num_attention_heads: int = 8
    intermediate_size: int = 2_048
    hidden_act: str = "silu"
    attention_bias: bool = False
    convolution_bias: bool = False
    conv_kernel_size: int = 32
    subsampling_conv_channels: int = 256
    subsampling_conv_kernel_size: int = 5
    subsampling_conv_stride: int = 2
    num_mel_bins: int = 128
    dropout: float = 0.1
    dropout_positions: float = 0.0
    layerdrop: float = 0.1
    activation_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 10_000
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    feed_forward_residual_weights: tuple[float, float] = (1.5, 0.5)
    conv_residual_weights: tuple[float, float] = (2.0, 1.0)
    batch_norm_momentum: float = 0.01
    rope_theta: float = 10_000.0
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True
    pad_token_id: int = 0
    sampling_rate: int = 16_000
    feature_hop_length: int = 160
    feature_fft_size: int = 512
    feature_window_length: int = 400
    feature_lower_hertz: float = 125.0
    feature_upper_hertz: float = 7_500.0
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ValueError("`variant` must be a non-empty string.")
        variant = self.variant.strip().lower().replace("_", "-")
        if variant not in _RELEASE_VARIANTS:
            choices = ", ".join(sorted(_RELEASE_VARIANTS))
            raise ValueError(f"`variant` must be one of: {choices}.")
        object.__setattr__(self, "variant", variant)

        for name in (
                "vocab_size",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "intermediate_size",
                "conv_kernel_size",
                "subsampling_conv_channels",
                "subsampling_conv_kernel_size",
                "subsampling_conv_stride",
                "num_mel_bins",
                "max_position_embeddings",
                "sampling_rate",
                "feature_hop_length",
                "feature_fft_size",
                "feature_window_length",
        ):
            object.__setattr__(
                self,
                name,
                _integer(name, getattr(self, name), minimum=1),
            )
        object.__setattr__(
            self,
            "pad_token_id",
            _integer("pad_token_id", self.pad_token_id, minimum=0),
        )
        if self.pad_token_id >= self.vocab_size:
            raise ValueError("`pad_token_id` must be smaller than `vocab_size`.")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        if self.feature_fft_size < self.feature_window_length:
            raise ValueError("`feature_fft_size` must be at least `feature_window_length`.")
        if self.feature_upper_hertz > self.sampling_rate / 2:
            raise ValueError("`feature_upper_hertz` cannot exceed the Nyquist frequency.")
        if self.feature_lower_hertz >= self.feature_upper_hertz:
            raise ValueError("`feature_lower_hertz` must be below `feature_upper_hertz`.")
        for name in (
                "initializer_range",
                "layer_norm_eps",
                "batch_norm_momentum",
                "rope_theta",
                "feature_lower_hertz",
                "feature_upper_hertz",
        ):
            object.__setattr__(
                self,
                name,
                _positive_real(name, getattr(self, name)),
            )
        for name in (
                "dropout",
                "dropout_positions",
                "layerdrop",
                "activation_dropout",
                "attention_dropout",
        ):
            object.__setattr__(
                self,
                name,
                _probability(name, getattr(self, name)),
            )
        for name in (
                "attention_bias",
                "convolution_bias",
                "ctc_zero_infinity",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.hidden_act not in _ACTIVATIONS:
            raise ValueError("Native LASR currently implements the released SiLU graph.")
        if self.ctc_loss_reduction not in _CTC_REDUCTIONS:
            choices = ", ".join(sorted(_CTC_REDUCTIONS))
            raise ValueError(f"`ctc_loss_reduction` must be one of: {choices}.")
        for name in (
                "feed_forward_residual_weights",
                "conv_residual_weights",
        ):
            object.__setattr__(
                self,
                name,
                _weights(name, getattr(self, name)),
            )
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )
        if variant == "medasr":
            self._validate_release_graph()

    def _validate_release_graph(self) -> None:
        reference = type(self)(variant="custom")
        ignored = {"variant", "extra_config"}
        changed = [
            item.name for item in fields(self)
            if item.name not in ignored and getattr(self, item.name) != getattr(reference, item.name)
        ]
        if changed:
            raise ValueError(
                "The released MedASR graph and frontend are immutable; "
                f"changed field(s): {', '.join(changed)}. Use "
                "`variant='custom'` only for VoiceHub-owned artifacts.")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> MedASRConfig:
        if not isinstance(values, Mapping):
            raise TypeError("MedASR configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        encoder = source.get("encoder_config", {})
        if encoder is None:
            encoder = {}
        if not isinstance(encoder, Mapping):
            raise TypeError("MedASR `encoder_config` must be a mapping.")
        encoder = dict(encoder)
        known = {item.name for item in fields(cls)}
        resolved: dict[str, Any] = {}
        for name in known - {"extra_config"}:
            if name in source:
                resolved[name] = source[name]
            elif name in encoder:
                resolved[name] = encoder[name]
        rope = encoder.get("rope_parameters")
        if rope is not None:
            if not isinstance(rope, Mapping):
                raise TypeError("MedASR encoder `rope_parameters` must be a mapping.")
            if rope.get("rope_type", "default") != "default":
                raise ValueError("Native MedASR supports the released default RoPE only.")
            if "rope_theta" in rope:
                resolved["rope_theta"] = rope["rope_theta"]
        model_type = str(source.get("model_type", "")).strip().lower()
        if "variant" not in resolved:
            resolved["variant"] = ("medasr" if model_type == "lasr_ctc" else "custom")
        consumed = known | {
            'architectures',
            "encoder_config",
            "initializer_range",
            "model_type",
            "name_or_path",
            "source_artifact_revision",
            "source_code_revision",
            "transformers_version",
            "voicehub_checkpoint_format",
            "voicehub_provider",
        }
        extras = {name: value for name, value in source.items() if name not in consumed}
        supplied = source.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        resolved["extra_config"] = extras
        return cls(**resolved)

    @classmethod
    def coerce(
        cls,
        value: MedASRConfig | Mapping[str, Any] | None,
    ) -> MedASRConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        encoder_names = {
            "activation_dropout",
            "attention_bias",
            "attention_dropout",
            "batch_norm_momentum",
            "conv_kernel_size",
            "conv_residual_weights",
            "convolution_bias",
            "dropout",
            "dropout_positions",
            "feed_forward_residual_weights",
            "hidden_act",
            "hidden_size",
            "initializer_range",
            "intermediate_size",
            "layer_norm_eps",
            "layerdrop",
            "max_position_embeddings",
            "num_attention_heads",
            "num_hidden_layers",
            "num_mel_bins",
            "subsampling_conv_channels",
            "subsampling_conv_kernel_size",
            "subsampling_conv_stride",
        }
        encoder = {
            name: (
                list(getattr(self, name)) if isinstance(getattr(self, name), tuple) else copy.deepcopy(
                    getattr(self, name)))
            for name in encoder_names
        }
        encoder.update({
            "model_type": "lasr_encoder",
            "num_key_value_heads": self.num_attention_heads,
            "rope_parameters": {
                "rope_theta": self.rope_theta,
                "rope_type": "default",
            },
        })
        result = copy.deepcopy(dict(self.extra_config))
        result.update({
            "ctc_loss_reduction": self.ctc_loss_reduction,
            "ctc_zero_infinity": self.ctc_zero_infinity,
            "encoder_config": encoder,
            "feature_fft_size": self.feature_fft_size,
            "feature_hop_length": self.feature_hop_length,
            "feature_lower_hertz": self.feature_lower_hertz,
            "feature_upper_hertz": self.feature_upper_hertz,
            "feature_window_length": self.feature_window_length,
            "initializer_range": self.initializer_range,
            "model_type": "lasr_ctc",
            "pad_token_id": self.pad_token_id,
            "sampling_rate": self.sampling_rate,
            "variant": self.variant,
            "vocab_size": self.vocab_size,
        })
        return result

    @property
    def minimum_feature_frames(self) -> int:
        required = 1
        for _ in range(2):
            required = ((required - 1) * self.subsampling_conv_stride + self.subsampling_conv_kernel_size)
        return required

    @property
    def minimum_input_samples(self) -> int:
        return (self.feature_window_length + (self.minimum_feature_frames - 1) * self.feature_hop_length)

    def encoder_output_length(self, feature_frames: int) -> int:
        length = _integer(
            "feature_frames",
            feature_frames,
            minimum=0,
        )
        for _ in range(2):
            length = ((length - self.subsampling_conv_kernel_size) // self.subsampling_conv_stride + 1)
            if length <= 0:
                return 0
        return length


__all__ = ["MedASRConfig"]
