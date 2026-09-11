"""Validated configuration for VoiceHub's native VITS/MMS-TTS graph.

The field layout follows Hugging Face Transformers VITS revision
``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``. Architectural behavior was
also checked against the original MIT-licensed VITS implementation at revision
``2e561ba58618d021b5b8323d3765880f7e0ecfdb``. Neither runtime is imported.
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

_ACTIVATIONS = frozenset({"gelu", "relu", "silu"})


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}; found {value}.")
    return value


def _positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"`{name}` must be finite and greater than zero.")
    return normalized


def _nonnegative_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return normalized


def _probability(name: str, value: object, *, inclusive_one: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    normalized = float(value)
    upper_valid = normalized <= 1.0 if inclusive_one else normalized < 1.0
    if not math.isfinite(normalized) or normalized < 0.0 or not upper_valid:
        interval = "[0, 1]" if inclusive_one else "[0, 1)"
        raise ValueError(f"`{name}` must be finite and in {interval}.")
    return normalized


def _integer_tuple(name: str, value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence of integers.")
    normalized = tuple(value)
    if not normalized:
        raise ValueError(f"`{name}` cannot be empty.")
    for item in normalized:
        _integer(name, item, minimum=1)
    return normalized


def _nested_integer_tuple(
    name: str,
    value: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a nested sequence of integers.")
    normalized = tuple(_integer_tuple(f"{name}[{index}]", item) for index, item in enumerate(value))
    if not normalized:
        raise ValueError(f"`{name}` cannot be empty.")
    return normalized


@dataclass(frozen=True, slots=True)
class VitsConfig:
    """Complete executable configuration for VITS and MMS-TTS checkpoints."""

    vocab_size: int = 38
    hidden_size: int = 192
    num_hidden_layers: int = 6
    num_attention_heads: int = 2
    window_size: int | None = 4
    use_bias: bool = True
    ffn_dim: int = 768
    layerdrop: float = 0.1
    ffn_kernel_size: int = 3
    flow_size: int = 192
    spectrogram_bins: int = 513
    hidden_act: str = "relu"
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_stochastic_duration_prediction: bool = True
    num_speakers: int = 1
    speaker_embedding_size: int = 0
    upsample_initial_channel: int = 512
    upsample_rates: tuple[int, ...] = (8, 8, 2, 2)
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 4, 4)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    leaky_relu_slope: float = 0.1
    depth_separable_channels: int = 2
    depth_separable_num_layers: int = 3
    duration_predictor_flow_bins: int = 10
    duration_predictor_tail_bound: float = 5.0
    duration_predictor_kernel_size: int = 3
    duration_predictor_dropout: float = 0.5
    duration_predictor_num_flows: int = 4
    duration_predictor_filter_channels: int = 256
    prior_encoder_num_flows: int = 4
    prior_encoder_num_wavenet_layers: int = 4
    posterior_encoder_num_wavenet_layers: int = 16
    wavenet_kernel_size: int = 5
    wavenet_dilation_rate: int = 1
    wavenet_dropout: float = 0.0
    speaking_rate: float = 1.0
    noise_scale: float = 0.667
    noise_scale_duration: float = 0.8
    sampling_rate: int = 16_000
    pad_token_id: int | None = None
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
                "ffn_dim",
                "ffn_kernel_size",
                "flow_size",
                "spectrogram_bins",
                "num_speakers",
                "upsample_initial_channel",
                "depth_separable_channels",
                "depth_separable_num_layers",
                "duration_predictor_flow_bins",
                "duration_predictor_kernel_size",
                "duration_predictor_num_flows",
                "duration_predictor_filter_channels",
                "prior_encoder_num_flows",
                "prior_encoder_num_wavenet_layers",
                "posterior_encoder_num_wavenet_layers",
                "wavenet_kernel_size",
                "wavenet_dilation_rate",
                "sampling_rate",
        ):
            _integer(name, getattr(self, name), minimum=1)
        if self.window_size is not None:
            _integer("window_size", self.window_size, minimum=1)
        _integer(
            "speaker_embedding_size",
            self.speaker_embedding_size,
            minimum=0,
        )
        if self.pad_token_id is not None:
            _integer("pad_token_id", self.pad_token_id, minimum=0)
            if self.pad_token_id >= self.vocab_size:
                raise ValueError("`pad_token_id` must be smaller than `vocab_size`.")

        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        if self.flow_size % 2:
            raise ValueError("`flow_size` must be even for coupling flows.")
        if self.spectrogram_bins < 2:
            raise ValueError("`spectrogram_bins` must be at least two.")
        if self.depth_separable_channels % 2:
            raise ValueError("`depth_separable_channels` must be even for duration flows.")
        if self.wavenet_kernel_size % 2 == 0:
            raise ValueError("`wavenet_kernel_size` must be odd.")
        if self.duration_predictor_kernel_size % 2 == 0:
            raise ValueError("`duration_predictor_kernel_size` must be odd.")
        if self.duration_predictor_flow_bins >= 1_000:
            raise ValueError(
                "`duration_predictor_flow_bins` must be smaller than 1000 "
                "for the source spline's minimum bin width.")

        for name in (
                "hidden_dropout",
                "attention_dropout",
                "activation_dropout",
                "layerdrop",
                "duration_predictor_dropout",
                "wavenet_dropout",
        ):
            object.__setattr__(
                self,
                name,
                _probability(name, getattr(self, name), inclusive_one=True),
            )
        for name in (
                "initializer_range",
                "layer_norm_eps",
                "duration_predictor_tail_bound",
                "speaking_rate",
        ):
            object.__setattr__(
                self,
                name,
                _positive_real(name, getattr(self, name)),
            )
        for name in ("noise_scale", "noise_scale_duration"):
            object.__setattr__(
                self,
                name,
                _nonnegative_real(name, getattr(self, name)),
            )
        object.__setattr__(
            self,
            "leaky_relu_slope",
            _probability(
                "leaky_relu_slope",
                self.leaky_relu_slope,
                inclusive_one=True,
            ),
        )

        for name in (
                "use_bias",
                "use_stochastic_duration_prediction",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if not isinstance(self.hidden_act, str) or self.hidden_act not in _ACTIVATIONS:
            choices = ", ".join(sorted(_ACTIVATIONS))
            raise ValueError(f"`hidden_act` must be one of {choices}; "
                             f"found {self.hidden_act!r}.")

        for name in (
                "upsample_rates",
                "upsample_kernel_sizes",
                "resblock_kernel_sizes",
        ):
            object.__setattr__(
                self,
                name,
                _integer_tuple(name, getattr(self, name)),
            )
        object.__setattr__(
            self,
            "resblock_dilation_sizes",
            _nested_integer_tuple(
                "resblock_dilation_sizes",
                self.resblock_dilation_sizes,
            ),
        )
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("`upsample_rates` and `upsample_kernel_sizes` must have "
                             "equal lengths.")
        if self.upsample_initial_channel < 2**len(self.upsample_rates):
            raise ValueError(
                "`upsample_initial_channel` is too small for the configured "
                "number of halving upsampling stages.")
        for index, (rate, kernel) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            if kernel < rate or (kernel - rate) % 2:
                raise ValueError(
                    f"Upsampler {index} requires kernel >= rate and an even "
                    "kernel-rate difference for exact length scaling.")
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilation_sizes):
            raise ValueError(
                "`resblock_kernel_sizes` and `resblock_dilation_sizes` must "
                "have equal lengths.")
        if any(kernel % 2 == 0 for kernel in self.resblock_kernel_sizes):
            raise ValueError("Every residual-block kernel must be odd.")
        if self.num_speakers > 1 and self.speaker_embedding_size < 1:
            raise ValueError("Multi-speaker VITS requires a positive "
                             "`speaker_embedding_size`.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> VitsConfig:
        """Parse a Hugging Face-compatible VITS configuration mapping."""
        if not isinstance(values, Mapping):
            raise TypeError("VITS configuration values must be a mapping.")
        source = copy.deepcopy(dict(values))
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        for name in (
                "upsample_rates",
                "upsample_kernel_sizes",
                "resblock_kernel_sizes",
        ):
            if name in resolved:
                resolved[name] = tuple(resolved[name])
        if "resblock_dilation_sizes" in resolved:
            resolved["resblock_dilation_sizes"] = tuple(
                tuple(item) for item in resolved["resblock_dilation_sizes"])

        consumed = canonical | {"extra_config"}
        extras = {name: value for name, value in source.items() if name not in consumed}
        supplied = source.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**resolved, extra_config=extras)

    @classmethod
    def coerce(cls, value: VitsConfig | Mapping[str, Any]) -> VitsConfig:
        """Return ``value`` as a validated configuration."""
        if isinstance(value, cls):
            return value
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached Hugging Face-compatible configuration."""
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            if isinstance(value, tuple):
                value = [list(member) if isinstance(member, tuple) else member for member in value]
            result[item.name] = value
        result.setdefault("model_type", "vits")
        result.setdefault('architectures', ["VitsModel"])
        return result

    @property
    def upsample_factor(self) -> int:
        """Waveform samples produced for every latent frame."""
        return reduce(mul, self.upsample_rates, 1)

    @property
    def fft_size(self) -> int:
        """FFT size implied by the linear spectrogram channel count."""
        return (self.spectrogram_bins - 1) * 2

    @property
    def is_multispeaker(self) -> bool:
        return self.num_speakers > 1


__all__ = ["VitsConfig"]
