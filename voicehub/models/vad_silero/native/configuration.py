"""Validated configuration for VoiceHub's native Silero VAD graph.

The fixed 8 kHz and 16 kHz dimensions are taken from the official Silero
VAD v6.2.1 TorchScript modules at revision
``7e30209a3e901f9842f81b225f3e93d8199902b1``.  The released standalone
Safetensors checkpoint contains the 16 kHz branch only.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, ClassVar


def _probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"`{name}` must be finite and in the interval [0, 1).")
    return result


@dataclass(frozen=True, slots=True)
class SileroVADConfig:
    """The two immutable model layouts released by the Silero team.

    Silero v5 and v6 use fixed windows: 512 samples at 16 kHz and 256
    samples at 8 kHz.  Making these dimensions derived properties prevents a
    configuration from describing a graph for which no official checkpoint
    exists.
    """

    SUPPORTED_SAMPLE_RATES: ClassVar[tuple[int, ...]] = (8_000, 16_000)

    sampling_rate: int = 16_000
    decoder_dropout: float = 0.1
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if isinstance(self.sampling_rate, bool) or not isinstance(self.sampling_rate, int):
            raise TypeError("`sampling_rate` must be an integer.")
        if self.sampling_rate not in self.SUPPORTED_SAMPLE_RATES:
            supported = ", ".join(str(value) for value in self.SUPPORTED_SAMPLE_RATES)
            raise ValueError(f"`sampling_rate` must be one of {supported}; "
                             f"found {self.sampling_rate}.")
        object.__setattr__(
            self,
            "decoder_dropout",
            _probability("decoder_dropout", self.decoder_dropout),
        )
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> SileroVADConfig:
        """Parse a detached model configuration mapping."""
        if not isinstance(values, Mapping):
            raise TypeError("Silero VAD configuration values must be a mapping.")
        source = copy.deepcopy(dict(values))
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        consumed = canonical | {"extra_config"}
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
        value: SileroVADConfig | Mapping[str, Any],
    ) -> SileroVADConfig:
        """Return *value* as a validated configuration."""
        if isinstance(value, cls):
            return value
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, serialization-safe configuration."""
        result = copy.deepcopy(dict(self.extra_config))
        result.update({
            "sampling_rate": self.sampling_rate,
            "decoder_dropout": self.decoder_dropout,
        })
        result.setdefault("model_type", "silero-vad")
        result.setdefault('architectures', ["SileroVADModel"])
        return result

    @property
    def frame_size(self) -> int:
        """Samples consumed by one official streaming call."""
        return 512 if self.sampling_rate == 16_000 else 256

    @property
    def context_size(self) -> int:
        """Samples carried from the preceding frame."""
        return 64 if self.sampling_rate == 16_000 else 32

    @property
    def filter_length(self) -> int:
        """Fourier analysis convolution width."""
        return 256 if self.sampling_rate == 16_000 else 128

    @property
    def hop_length(self) -> int:
        """Fourier analysis convolution stride."""
        return self.filter_length // 2

    @property
    def reflection_padding(self) -> int:
        """Right reflection padding used before Fourier analysis."""
        return self.filter_length // 4

    @property
    def spectrum_bins(self) -> int:
        """Non-redundant real-valued magnitude bins."""
        return self.filter_length // 2 + 1

    @property
    def recurrent_size(self) -> int:
        """Width of the official LSTM cell."""
        return 128


__all__ = ["SileroVADConfig"]
