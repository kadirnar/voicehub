"""Validated configuration for NVIDIA's multilingual Frame-VAD MarbleNet."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be positive.")
    return value


@dataclass(frozen=True, slots=True)
class MarbleNetVADConfig:
    """Complete graph and frontend description for the v2 multilingual VAD.

    The defaults reproduce
    ``nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0`` at immutable Hub
    revision ``1f6df5f07c68baacbb91b155ddd54503b4ef2d80``.
    """

    sampling_rate: int = 16_000
    window_length: int = 400
    hop_length: int = 160
    n_fft: int = 512
    num_mel_bins: int = 80
    preemphasis: float = 0.97
    log_guard: float = 2**-24
    dither: float = 1e-5
    pad_to: int = 2
    dropout: float = 0.1
    spec_augment_frequency_masks: int = 5
    spec_augment_time_masks: int = 5
    spec_augment_frequency_width: int = 10
    spec_augment_time_width: float = 0.05
    speech_class_id: int = 1
    frontend_gradients: bool = False
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "sampling_rate",
                "window_length",
                "hop_length",
                "n_fft",
                "num_mel_bins",
                "pad_to",
                "spec_augment_frequency_width",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(name, getattr(self, name)),
            )
        if self.sampling_rate != 16_000:
            raise ValueError("The published multilingual MarbleNet VAD requires 16 kHz audio.")
        if self.window_length != 400 or self.hop_length != 160:
            raise ValueError(
                "The published multilingual MarbleNet VAD requires a "
                "400-sample window and 160-sample hop.")
        if self.n_fft != 512 or self.num_mel_bins != 80:
            raise ValueError(
                "The published multilingual MarbleNet VAD requires a "
                "512-point FFT and 80 mel bins.")
        if self.pad_to != 2:
            raise ValueError("The released graph pads mel frames to a multiple of two.")
        for name in (
                "spec_augment_frequency_masks",
                "spec_augment_time_masks",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"`{name}` must be an integer.")
            if value < 0:
                raise ValueError(f"`{name}` cannot be negative.")
        for name in ("preemphasis", "log_guard", "dither", "dropout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a real number.")
            object.__setattr__(self, name, float(value))
        if not 0.0 <= self.preemphasis < 1.0:
            raise ValueError("`preemphasis` must be in [0, 1).")
        if self.log_guard <= 0.0:
            raise ValueError("`log_guard` must be positive.")
        if self.dither < 0.0:
            raise ValueError("`dither` cannot be negative.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("`dropout` must be in [0, 1).")
        if (isinstance(self.spec_augment_time_width, bool) or not isinstance(self.spec_augment_time_width,
                                                                             (int, float)) or
                not 0.0 <= self.spec_augment_time_width <= 1.0):
            raise ValueError("`spec_augment_time_width` must be in [0, 1].")
        object.__setattr__(
            self,
            "spec_augment_time_width",
            float(self.spec_augment_time_width),
        )
        if self.speech_class_id not in (0, 1):
            raise ValueError("`speech_class_id` must be 0 or 1.")
        if not isinstance(self.frontend_gradients, bool):
            raise TypeError("`frontend_gradients` must be a boolean.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @property
    def output_frame_hop_samples(self) -> int:
        """The first encoder block subsamples the 10 ms frontend by two."""
        return self.hop_length * 2

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> MarbleNetVADConfig:
        if not isinstance(values, Mapping):
            raise TypeError("MarbleNet VAD configuration values must be a mapping.")
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
        value: MarbleNetVADConfig | Mapping[str, Any],
    ) -> MarbleNetVADConfig:
        if isinstance(value, cls):
            return value
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name != "extra_config":
                result[item.name] = getattr(self, item.name)
        result.setdefault("model_type", "marblenet-vad")
        result.setdefault('architectures', ["MarbleNetVADModel"])
        return result


__all__ = ["MarbleNetVADConfig"]
