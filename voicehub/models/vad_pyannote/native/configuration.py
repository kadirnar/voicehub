"""Validated configuration for VoiceHub's native PyanNet family.

The graph dimensions mirror ``pyannote.audio`` PyanNet at immutable source
revision ``795b92ab265888c58d160f90ae4d91b7bcc6aa2c`` (the commit behind the
3.0.0 release).  Brouhaha uses the same frontend with the dimensions published
at ``marianne-m/brouhaha-vad@9132cbe62ac78f90abdbc21bcf6ec6cfe9bb4891``.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, ClassVar


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < 1:
        raise ValueError(f"`{name}` must be positive.")
    return value


def _non_negative_probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"`{name}` must be finite and in [0, 1).")
    return result


@dataclass(frozen=True, slots=True)
class PyanNetConfig:
    """One complete, serializable PyanNet graph description."""

    VARIANTS: ClassVar[tuple[str, ...]] = (
        "segmentation",
        "powerset-segmentation",
        "brouhaha",
    )

    variant: str = "segmentation"
    sampling_rate: int = 16_000
    num_channels: int = 1
    sinc_stride: int = 10
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 4
    lstm_bidirectional: bool = True
    lstm_dropout: float = 0.5
    linear_hidden_size: int = 128
    linear_num_layers: int = 2
    num_classes: int = 4
    max_active_classes: int | None = None
    chunk_duration_s: float = 5.0
    chunk_step_s: float = 0.5
    repeat_final_chunk: bool = False
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.variant, str):
            raise TypeError("`variant` must be a string.")
        variant = self.variant.strip().lower().replace("_", "-")
        if variant not in self.VARIANTS:
            raise ValueError(f"`variant` must be one of {', '.join(self.VARIANTS)}.")
        object.__setattr__(self, "variant", variant)
        for name in (
                "sampling_rate",
                "num_channels",
                "sinc_stride",
                "lstm_hidden_size",
                "lstm_num_layers",
                "linear_hidden_size",
                "linear_num_layers",
                "num_classes",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(name, getattr(self, name)),
            )
        if self.sampling_rate != 16_000:
            raise ValueError("Published PyanNet checkpoints require 16 kHz audio.")
        if self.num_channels != 1:
            raise ValueError("Published PyanNet checkpoints require mono audio.")
        for name in ("lstm_bidirectional", "repeat_final_chunk"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        object.__setattr__(
            self,
            "lstm_dropout",
            _non_negative_probability("lstm_dropout", self.lstm_dropout),
        )
        for name in ("chunk_duration_s", "chunk_step_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a real number.")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"`{name}` must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.chunk_step_s > self.chunk_duration_s:
            raise ValueError("`chunk_step_s` cannot exceed `chunk_duration_s`.")
        maximum = self.max_active_classes
        if maximum is not None:
            maximum = _positive_integer("max_active_classes", maximum)
            if maximum > self.num_classes:
                raise ValueError("`max_active_classes` cannot exceed `num_classes`.")
            object.__setattr__(self, "max_active_classes", maximum)
        if self.is_powerset != (maximum is not None):
            raise ValueError("Only the powerset variant may define `max_active_classes`.")
        expected_output = {
            "segmentation": 4,
            "powerset-segmentation": 7,
            "brouhaha": 3,
        }[variant]
        if self.output_size != expected_output:
            raise ValueError(
                f"Published {variant} checkpoints require {expected_output} "
                f"outputs; this configuration declares {self.output_size}.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @property
    def is_powerset(self) -> bool:
        return self.variant == "powerset-segmentation"

    @property
    def is_brouhaha(self) -> bool:
        return self.variant == "brouhaha"

    @property
    def feature_size(self) -> int:
        if self.linear_num_layers:
            return self.linear_hidden_size
        return self.lstm_hidden_size * (2 if self.lstm_bidirectional else 1)

    @property
    def output_size(self) -> int:
        if self.is_brouhaha:
            return 3
        if not self.is_powerset:
            return self.num_classes
        from math import comb

        return sum(comb(self.num_classes, size) for size in range(self.max_active_classes + 1))

    @classmethod
    def segmentation(cls) -> PyanNetConfig:
        return cls()

    @classmethod
    def segmentation_3(cls) -> PyanNetConfig:
        return cls(
            variant="powerset-segmentation",
            lstm_dropout=0.0,
            num_classes=3,
            max_active_classes=2,
            chunk_duration_s=10.0,
            chunk_step_s=1.0,
        )

    @classmethod
    def brouhaha(cls) -> PyanNetConfig:
        return cls(
            variant="brouhaha",
            lstm_hidden_size=256,
            lstm_num_layers=3,
            num_classes=3,
            chunk_duration_s=6.0,
            chunk_step_s=0.6,
            repeat_final_chunk=True,
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> PyanNetConfig:
        if not isinstance(values, Mapping):
            raise TypeError("PyanNet configuration values must be a mapping.")
        source = copy.deepcopy(dict(values))
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        extras = {name: value for name, value in source.items() if name not in canonical | {"extra_config"}}
        supplied_extras = source.get("extra_config")
        if supplied_extras is not None:
            if not isinstance(supplied_extras, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied_extras)))
        return cls(**resolved, extra_config=extras)

    @classmethod
    def coerce(
        cls,
        value: PyanNetConfig | Mapping[str, Any],
    ) -> PyanNetConfig:
        if isinstance(value, cls):
            return value
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name != "extra_config":
                result[item.name] = getattr(self, item.name)
        result.setdefault("model_type", "pyannet")
        result.setdefault('architectures', ["PyanNet"])
        return result


__all__ = ["PyanNetConfig"]
