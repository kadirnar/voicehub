"""Native hysteresis and endpointing policy for Silero VAD probabilities.

The state machine follows ``get_speech_timestamps`` from Silero VAD
v6.2.1 at revision ``7e30209a3e901f9842f81b225f3e93d8199902b1``.  Neural
inference and segmentation are intentionally separate: thresholds can be
retuned without rerunning or mutating the model.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import Tensor

from voicehub.models.vad_silero.native.configuration import SileroVADConfig


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"`{name}` cannot be negative.")
    return result


def _probability(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    lower_valid = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not lower_valid or result > 1.0:
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"`{name}` must be finite and in {interval}.")
    return result


@dataclass(frozen=True, slots=True)
class SileroVADSegmentationConfig:
    """Threshold and duration policy independent from the neural graph."""

    threshold: float = 0.5
    negative_threshold: float | None = None
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    max_speech_duration_s: float = math.inf
    min_silence_at_max_speech_ms: int = 98
    use_max_possible_silence: bool = True

    def __post_init__(self) -> None:
        threshold = _probability("threshold", self.threshold, positive=True)
        object.__setattr__(self, "threshold", threshold)
        negative = self.negative_threshold
        if negative is None:
            negative = max(threshold - 0.15, 0.01)
        negative = _probability("negative_threshold", negative)
        if negative >= threshold:
            raise ValueError("`negative_threshold` must be below `threshold`.")
        object.__setattr__(self, "negative_threshold", negative)

        for name in (
                "min_speech_duration_ms",
                "min_silence_duration_ms",
                "speech_pad_ms",
                "min_silence_at_max_speech_ms",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_integer(name, getattr(self, name)),
            )
        maximum = self.max_speech_duration_s
        if isinstance(maximum, bool) or not isinstance(maximum, Real):
            raise TypeError("`max_speech_duration_s` must be a real number.")
        maximum = float(maximum)
        if math.isnan(maximum) or maximum <= 0.0:
            raise ValueError("`max_speech_duration_s` must be positive or infinity.")
        object.__setattr__(self, "max_speech_duration_s", maximum)
        if not isinstance(self.use_max_possible_silence, bool):
            raise TypeError("`use_max_possible_silence` must be a boolean.")

    @property
    def exit_threshold(self) -> float:
        """Hysteresis exit threshold, resolved during validation."""
        if self.negative_threshold is None:  # pragma: no cover - invariant
            raise RuntimeError("Negative threshold was not resolved.")
        return self.negative_threshold


@dataclass(frozen=True, order=True, slots=True)
class SpeechSegment:
    """Half-open speech interval measured in waveform samples."""

    start: int
    end: int

    def __post_init__(self) -> None:
        start = _nonnegative_integer("start", self.start)
        end = _nonnegative_integer("end", self.end)
        if end <= start:
            raise ValueError("A speech segment must end after it starts.")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def duration(self) -> int:
        return self.end - self.start

    def seconds(self, sampling_rate: int) -> tuple[float, float]:
        if isinstance(sampling_rate, bool) or not isinstance(sampling_rate, int):
            raise TypeError("`sampling_rate` must be an integer.")
        if sampling_rate < 1:
            raise ValueError("`sampling_rate` must be positive.")
        return self.start / sampling_rate, self.end / sampling_rate


def _probability_values(probabilities: Tensor | Iterable[float], ) -> tuple[float, ...]:
    if isinstance(probabilities, Tensor):
        if probabilities.ndim != 1:
            raise ValueError("`probabilities` must have shape [frames].")
        if not probabilities.is_floating_point():
            raise TypeError("`probabilities` must use a floating-point dtype.")
        values = tuple(probabilities.detach().to(device="cpu", dtype=torch.float32).tolist())
    else:
        if isinstance(probabilities, (str, bytes)):
            raise TypeError("`probabilities` must be an iterable of numbers.")
        try:
            values = tuple(probabilities)
        except TypeError as error:
            raise TypeError("`probabilities` must be an iterable of numbers.") from error
    return tuple(_probability(f"probabilities[{index}]", value) for index, value in enumerate(values))


def _pad_segments(
    segments: list[tuple[int, int]],
    *,
    audio_length_samples: int,
    padding_samples: int,
) -> tuple[SpeechSegment, ...]:
    if not segments:
        return ()
    padded = [list(item) for item in segments]
    for index, segment in enumerate(padded):
        if index == 0:
            segment[0] = max(0, segment[0] - padding_samples)
        if index == len(padded) - 1:
            segment[1] = min(audio_length_samples, segment[1] + padding_samples)
            continue
        next_segment = padded[index + 1]
        silence = next_segment[0] - segment[1]
        if silence < 2 * padding_samples:
            left_padding = silence // 2
            segment[1] += left_padding
            next_segment[0] -= silence - left_padding
        else:
            segment[1] = min(audio_length_samples, segment[1] + padding_samples)
            next_segment[0] = max(0, next_segment[0] - padding_samples)
    return tuple(SpeechSegment(start, end) for start, end in padded)


def segment_speech_probabilities(
    probabilities: Tensor | Iterable[float],
    *,
    audio_length_samples: int,
    model_config: SileroVADConfig | Mapping[str, object] | None = None,
    config: SileroVADSegmentationConfig | None = None,
) -> tuple[SpeechSegment, ...]:
    """Convert frame probabilities into padded, non-overlapping speech."""
    model_config = SileroVADConfig.coerce(model_config or {})
    config = config or SileroVADSegmentationConfig()
    if not isinstance(config, SileroVADSegmentationConfig):
        raise TypeError("`config` must be a SileroVADSegmentationConfig.")
    audio_length_samples = _nonnegative_integer(
        "audio_length_samples",
        audio_length_samples,
    )
    if audio_length_samples < 1:
        raise ValueError("`audio_length_samples` must be positive.")

    values = _probability_values(probabilities)
    expected_frames = math.ceil(audio_length_samples / model_config.frame_size)
    if len(values) != expected_frames:
        raise ValueError(
            f"Expected {expected_frames} frame probabilities for "
            f"{audio_length_samples} samples; found {len(values)}.")

    sampling_rate = model_config.sampling_rate
    frame_size = model_config.frame_size
    min_speech_samples = (sampling_rate * config.min_speech_duration_ms / 1_000)
    min_silence_samples = (sampling_rate * config.min_silence_duration_ms / 1_000)
    padding_samples = int(sampling_rate * config.speech_pad_ms / 1_000)
    min_silence_at_max = (sampling_rate * config.min_silence_at_max_speech_ms / 1_000)
    if math.isinf(config.max_speech_duration_s):
        max_speech_samples = math.inf
    else:
        max_speech_samples = (sampling_rate * config.max_speech_duration_s - frame_size - 2 * padding_samples)
        if max_speech_samples <= 0:
            raise ValueError(
                "`max_speech_duration_s` is too short for one frame and "
                "the configured speech padding.")

    triggered = False
    speech_start = 0
    pending_silence: int | None = None
    possible_ends: list[tuple[int, int]] = []
    raw_segments: list[tuple[int, int]] = []

    def append_if_long_enough(start: int, end: int) -> None:
        if end - start >= min_speech_samples:
            raw_segments.append((start, min(end, audio_length_samples)))

    for index, probability in enumerate(values):
        current_sample = frame_size * index

        if probability >= config.threshold and pending_silence is not None:
            silence_duration = current_sample - pending_silence
            if silence_duration >= min_silence_at_max:
                possible_ends.append((pending_silence, silence_duration))
            pending_silence = None

        if probability >= config.threshold and not triggered:
            triggered = True
            speech_start = current_sample

        if (triggered and current_sample - speech_start >= max_speech_samples):
            if config.use_max_possible_silence and possible_ends:
                segment_end, silence_duration = max(
                    possible_ends,
                    key=lambda item: item[1],
                )
                append_if_long_enough(speech_start, segment_end)
                speech_start = segment_end + silence_duration
                triggered = True
            else:
                append_if_long_enough(speech_start, current_sample)
                triggered = probability >= config.threshold
                speech_start = current_sample
            pending_silence = None
            possible_ends = []
            continue

        if probability < config.exit_threshold and triggered:
            if pending_silence is None:
                pending_silence = current_sample
            if current_sample - pending_silence >= min_silence_samples:
                append_if_long_enough(speech_start, pending_silence)
                triggered = False
                pending_silence = None
                possible_ends = []

    if triggered:
        append_if_long_enough(speech_start, audio_length_samples)

    return _pad_segments(
        raw_segments,
        audio_length_samples=audio_length_samples,
        padding_samples=padding_samples,
    )


class SileroVADSegmenter:
    """Reusable, immutable endpointing policy."""

    def __init__(
        self,
        model_config: SileroVADConfig | Mapping[str, object] | None = None,
        config: SileroVADSegmentationConfig | None = None,
    ) -> None:
        self.model_config = SileroVADConfig.coerce(model_config or {})
        self.config = config or SileroVADSegmentationConfig()
        if not isinstance(self.config, SileroVADSegmentationConfig):
            raise TypeError("`config` must be a SileroVADSegmentationConfig.")

    def segment(
        self,
        probabilities: Tensor | Iterable[float],
        *,
        audio_length_samples: int,
    ) -> tuple[SpeechSegment, ...]:
        return segment_speech_probabilities(
            probabilities,
            audio_length_samples=audio_length_samples,
            model_config=self.model_config,
            config=self.config,
        )

    __call__ = segment


__all__ = [
    "SileroVADSegmentationConfig",
    "SileroVADSegmenter",
    "SpeechSegment",
    "segment_speech_probabilities",
]
