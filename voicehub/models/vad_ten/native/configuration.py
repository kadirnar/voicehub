"""Validated graph configuration for VoiceHub-native TEN VAD."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < 1:
        raise ValueError(f"`{name}` must be positive.")
    return value


@dataclass(frozen=True, slots=True)
class TENVADConfig:
    """Complete released TEN neural graph and Sherpa frontend contract."""

    sampling_rate: int = 16_000
    window_size: int = 256
    analysis_window_size: int = 768
    fft_size: int = 1_024
    mel_bins: int = 40
    feature_size: int = 41
    context_frames: int = 3
    convolution_channels: int = 16
    recurrent_size: int = 64
    dense_size: int = 32
    preemphasis: float = 0.97
    input_scale: float = 32_768.0
    log_floor: float = 1e-10
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "sampling_rate",
                "window_size",
                "analysis_window_size",
                "fft_size",
                "mel_bins",
                "feature_size",
                "context_frames",
                "convolution_channels",
                "recurrent_size",
                "dense_size",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(name, getattr(self, name)),
            )
        if self.sampling_rate != 16_000:
            raise ValueError("The released TEN VAD checkpoint requires 16 kHz audio.")
        if self.window_size > self.analysis_window_size:
            raise ValueError("`window_size` cannot exceed the 768-sample analysis window.")
        if self.analysis_window_size != 768 or self.fft_size != 1_024:
            raise ValueError("The released TEN frontend requires a 768-sample window and 1024 FFT.")
        if self.mel_bins != 40 or self.feature_size != 41:
            raise ValueError("The released TEN graph requires 40 log-mel bins plus pitch.")
        if self.context_frames != 3:
            raise ValueError("The released TEN graph requires exactly three feature frames.")
        if (self.convolution_channels != 16 or self.recurrent_size != 64 or self.dense_size != 32):
            raise ValueError(
                "The released TEN graph requires 16 convolution channels, "
                "64 recurrent units, and a 32-unit dense layer.")
        for name in ("preemphasis", "input_scale", "log_floor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a real number.")
            object.__setattr__(self, name, float(value))
        if not 0.0 <= self.preemphasis < 1.0:
            raise ValueError("`preemphasis` must be in [0, 1).")
        if self.input_scale <= 0.0 or self.log_floor <= 0.0:
            raise ValueError("Input scale and log floor must be positive.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TENVADConfig:
        if not isinstance(values, Mapping):
            raise TypeError("TEN VAD configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        extras = {name: value for name, value in source.items() if name not in canonical | {"extra_config"}}
        supplied = source.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**resolved, extra_config=extras)

    @classmethod
    def coerce(cls, value: TENVADConfig | Mapping[str, Any]) -> TENVADConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name != "extra_config":
                result[item.name] = getattr(self, item.name)
        result.setdefault("model_type", "vad_sherpa_onnx")
        result.setdefault("architecture", "ten-vad")
        result.setdefault('architectures', ["TENVADModel"])
        return result


__all__ = ["TENVADConfig"]
