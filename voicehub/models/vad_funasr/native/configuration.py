"""Validated configuration for the VoiceHub-native FunASR FSMN VAD."""

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


def _non_negative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < 0:
        raise ValueError(f"`{name}` cannot be negative.")
    return value


@dataclass(frozen=True, slots=True)
class FSMNVADConfig:
    """Complete graph and frontend description for ``fsmn-vad``.

    Defaults reproduce ``funasr/fsmn-vad`` at immutable artifact
    revision ``df20e6b30c653645fa4ff125cacfcabd1020a669``.
    """

    sampling_rate: int = 16_000
    num_mel_bins: int = 80
    frame_length_ms: int = 25
    frame_shift_ms: int = 10
    lfr_m: int = 5
    lfr_n: int = 1
    input_dim: int = 400
    input_affine_dim: int = 140
    fsmn_layers: int = 4
    linear_dim: int = 250
    projection_dim: int = 128
    left_order: int = 20
    right_order: int = 0
    left_stride: int = 1
    right_stride: int = 0
    output_affine_dim: int = 140
    output_dim: int = 248
    silence_pdf_ids: tuple[int, ...] = (0, )
    window_size_ms: int = 200
    silence_to_speech_ms: int = 150
    speech_to_silence_ms: int = 150
    lookback_start_ms: int = 200
    lookahead_end_ms: int = 100
    max_end_silence_ms: int = 800
    max_start_silence_ms: int = 3_000
    max_single_segment_ms: int = 60_000
    speech_noise_threshold: float = 0.6
    speech_to_noise_ratio: float = 1.0
    noise_history_frames: int = 100
    decibel_threshold: float = -100.0
    snr_threshold: float = -100.0
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "sampling_rate",
                "num_mel_bins",
                "frame_length_ms",
                "frame_shift_ms",
                "lfr_m",
                "lfr_n",
                "input_dim",
                "input_affine_dim",
                "fsmn_layers",
                "linear_dim",
                "projection_dim",
                "left_order",
                "left_stride",
                "output_affine_dim",
                "output_dim",
                "window_size_ms",
                "silence_to_speech_ms",
                "speech_to_silence_ms",
                "lookback_start_ms",
                "lookahead_end_ms",
                "max_end_silence_ms",
                "max_start_silence_ms",
                "max_single_segment_ms",
                "noise_history_frames",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(name, getattr(self, name)),
            )
        for name in ("right_order", "right_stride"):
            object.__setattr__(
                self,
                name,
                _non_negative_integer(name, getattr(self, name)),
            )
        if self.sampling_rate != 16_000:
            raise ValueError("The published FSMN VAD checkpoint requires 16 kHz audio.")
        if self.input_dim != self.num_mel_bins * self.lfr_m:
            raise ValueError("`input_dim` must equal `num_mel_bins * lfr_m`.")
        if self.right_order == 0 and self.right_stride != 0:
            raise ValueError("`right_stride` must be zero when `right_order` is zero.")
        if not isinstance(self.silence_pdf_ids, tuple) or not self.silence_pdf_ids:
            raise TypeError("`silence_pdf_ids` must be a non-empty tuple.")
        silence_ids = tuple(
            _non_negative_integer("silence_pdf_ids item", item) for item in self.silence_pdf_ids)
        if len(set(silence_ids)) != len(silence_ids):
            raise ValueError("`silence_pdf_ids` cannot contain duplicates.")
        if max(silence_ids) >= self.output_dim:
            raise ValueError("Every silence PDF id must be below `output_dim`.")
        object.__setattr__(self, "silence_pdf_ids", silence_ids)
        for name in (
                "speech_noise_threshold",
                "speech_to_noise_ratio",
                "decibel_threshold",
                "snr_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a real number.")
            object.__setattr__(self, name, float(value))
        if not 0.0 <= self.speech_noise_threshold <= 1.0:
            raise ValueError("`speech_noise_threshold` must be in [0, 1].")
        if self.speech_to_noise_ratio <= 0.0:
            raise ValueError("`speech_to_noise_ratio` must be positive.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @property
    def frame_length_samples(self) -> int:
        return self.sampling_rate * self.frame_length_ms // 1_000

    @property
    def frame_shift_samples(self) -> int:
        return self.sampling_rate * self.frame_shift_ms // 1_000

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> FSMNVADConfig:
        if not isinstance(values, Mapping):
            raise TypeError("FSMN VAD configuration values must be a mapping.")
        source = copy.deepcopy(dict(values))
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        if "silence_pdf_ids" in resolved:
            resolved["silence_pdf_ids"] = tuple(resolved["silence_pdf_ids"])
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
        value: FSMNVADConfig | Mapping[str, Any],
    ) -> FSMNVADConfig:
        if isinstance(value, cls):
            return value
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            result[item.name] = list(value) if isinstance(value, tuple) else value
        result.setdefault("model_type", "fsmn-vad")
        result.setdefault('architectures', ["FSMNVADModel"])
        return result


__all__ = ["FSMNVADConfig"]
