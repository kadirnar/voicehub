"""Validated configuration for VoiceHub's native Descript DAC graph."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from functools import reduce
from operator import mul
from types import MappingProxyType
from typing import Any


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _positive_integer_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"`{name}` must be a sequence of integers.")
    result = tuple(values)
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    for value in result:
        _positive_integer(name, value)
    return result


@dataclass(frozen=True, slots=True)
class DacConfig:
    """Checkpoint-compatible configuration for the native DAC codec."""

    encoder_hidden_size: int = 64
    downsampling_ratios: tuple[int, ...] = (2, 4, 8, 8)
    decoder_hidden_size: int = 1_536
    n_codebooks: int = 9
    codebook_size: int = 1_024
    codebook_dim: int = 8
    quantizer_dropout: float = 0.0
    commitment_loss_weight: float = 0.25
    codebook_loss_weight: float = 1.0
    sampling_rate: int = 44_100
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "encoder_hidden_size",
                "decoder_hidden_size",
                "n_codebooks",
                "codebook_size",
                "codebook_dim",
                "sampling_rate",
        ):
            _positive_integer(name, getattr(self, name))
        object.__setattr__(
            self,
            "downsampling_ratios",
            _positive_integer_tuple(
                "downsampling_ratios",
                self.downsampling_ratios,
            ),
        )
        for name in (
                "quantizer_dropout",
                "commitment_loss_weight",
                "codebook_loss_weight",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or
                    value < 0.0):
                raise ValueError(f"`{name}` must be finite and non-negative.")
        if self.quantizer_dropout > 1.0:
            raise ValueError("`quantizer_dropout` cannot exceed one.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DacConfig:
        """Parse and validate a published DAC configuration mapping."""
        if not isinstance(values, Mapping):
            raise TypeError("DAC configuration values must be a mapping.")
        source = copy.deepcopy(dict(values))
        model_type = source.get("model_type", "dac")
        if str(model_type).strip().lower() != "dac":
            raise ValueError(f"Native DAC requires `model_type='dac'`; found {model_type!r}.")
        architectures = source.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        if architectures and "DacModel" not in architectures:
            raise ValueError("Native DAC requires a `DacModel` checkpoint architecture.")
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        if "downsampling_ratios" in resolved:
            resolved["downsampling_ratios"] = tuple(resolved["downsampling_ratios"])
        provisional = cls(**resolved)
        declared_values = {
            "hidden_size": provisional.hidden_size,
            "hop_length": provisional.hop_length,
            "upsampling_ratios": list(provisional.upsampling_ratios),
        }
        for name, expected in declared_values.items():
            if name in source and source[name] != expected:
                raise ValueError(
                    f"DAC configuration `{name}` is inconsistent: expected "
                    f"{expected!r}, found {source[name]!r}.")
        consumed = canonical | {
            'architectures',
            "extra_config",
            "hidden_size",
            "hop_length",
            "model_type",
            "upsampling_ratios",
        }
        extras = {name: value for name, value in source.items() if name not in consumed}
        supplied_extras = source.get("extra_config")
        if supplied_extras is not None:
            if not isinstance(supplied_extras, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied_extras)))
        return cls(**resolved, extra_config=extras)

    @classmethod
    def coerce(cls, value: DacConfig | Mapping[str, Any]) -> DacConfig:
        if isinstance(value, cls):
            return value
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            result[item.name] = (list(value) if isinstance(value, tuple) else value)
        result.update({
            'architectures': ["DacModel"],
            "hidden_size": self.hidden_size,
            "hop_length": self.hop_length,
            "model_type": "dac",
            "upsampling_ratios": list(self.upsampling_ratios),
        })
        return result

    @property
    def hidden_size(self) -> int:
        return self.encoder_hidden_size * 2**len(self.downsampling_ratios)

    @property
    def upsampling_ratios(self) -> tuple[int, ...]:
        return tuple(reversed(self.downsampling_ratios))

    @property
    def hop_length(self) -> int:
        return reduce(mul, self.downsampling_ratios, 1)

    @property
    def frame_rate(self) -> int:
        return math.ceil(self.sampling_rate / self.hop_length)


__all__ = ["DacConfig"]
