"""Backend-independent VAD segmentation and timestamp normalization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from numbers import Integral, Real
from typing import Any

from voicehub.inference_configuration import VADInferenceConfig
from voicehub.outputs import SpeechSegment


def _score(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("VAD frame scores must be real numbers.")
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("VAD frame scores must be probabilities between 0 and 1.")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    if (isinstance(value, bool) or not isinstance(value, Integral) or value <= 0):
        raise ValueError(f"`{name}` must be a positive integer.")
    return int(value)


def merge_speech_segments(
    segments: Iterable[SpeechSegment],
    *,
    max_gap: float = 0.0,
) -> tuple[SpeechSegment, ...]:
    """Sort and merge same-label regions separated by at most *max_gap*."""
    if (isinstance(max_gap, bool) or not isinstance(max_gap, Real) or not isfinite(float(max_gap)) or
            max_gap < 0):
        raise ValueError("`max_gap` must be a finite non-negative number.")
    max_gap = float(max_gap)
    ordered = sorted(tuple(segments), key=lambda segment: (segment.start, segment.end))
    if not ordered:
        return ()
    merged = [ordered[0]]
    for segment in ordered[1:]:
        previous = merged[-1]
        if (segment.label == previous.label and segment.channel == previous.channel and
                segment.start - previous.end <= max_gap):
            scores = [score for score in (previous.score, segment.score) if score is not None]
            merged[-1] = SpeechSegment(
                start=previous.start,
                end=max(previous.end, segment.end),
                score=sum(scores) / len(scores) if scores else None,
                label=previous.label,
                channel=previous.channel,
                metadata={
                    **previous.metadata,
                    **segment.metadata
                },
            )
        else:
            if segment.start < previous.end:
                raise ValueError("Speech segments with different labels cannot overlap.")
            merged.append(segment)
    return tuple(merged)


def frame_probabilities_to_segments(
    probabilities: Sequence[Any],
    *,
    sampling_rate: int,
    frame_hop_samples: int,
    frame_length_samples: int | None = None,
    config: VADInferenceConfig | None = None,
    duration_samples: int | None = None,
) -> tuple[SpeechSegment, ...]:
    """Convert frame speech probabilities to deterministic speech regions."""
    sampling_rate = _positive_integer(
        sampling_rate,
        name="sampling_rate",
    )
    frame_hop_samples = _positive_integer(
        frame_hop_samples,
        name="frame_hop_samples",
    )
    if frame_length_samples is None:
        frame_length_samples = frame_hop_samples
    frame_length_samples = _positive_integer(
        frame_length_samples,
        name="frame_length_samples",
    )
    if duration_samples is not None:
        if (isinstance(duration_samples, bool) or not isinstance(duration_samples, Integral) or
                duration_samples < 0):
            raise ValueError("`duration_samples` must be a non-negative integer.")
        duration_samples = int(duration_samples)
    config = config or VADInferenceConfig()
    config.validate()
    scores = tuple(_score(value) for value in probabilities)
    if not scores:
        return ()

    onset = getattr(config, "onset", None)
    offset = getattr(config, "offset", None)
    onset = config.threshold if onset is None else onset
    offset = config.threshold if offset is None else offset

    raw: list[tuple[int, int, float, int]] = []
    active_start = None
    active_end = None
    active_scores: list[float] = []
    for index, score in enumerate(scores):
        frame_start = index * frame_hop_samples
        frame_end = frame_start + frame_length_samples
        if active_start is None:
            if score >= onset:
                active_start = frame_start
                active_end = frame_end
                active_scores = [score]
            continue
        if score < offset:
            raw.append((
                active_start,
                active_end,
                sum(active_scores),
                len(active_scores),
            ))
            active_start = None
            active_end = None
            active_scores = []
            continue
        active_end = frame_end
        active_scores.append(score)
    if active_start is not None:
        raw.append((
            active_start,
            active_end,
            sum(active_scores),
            len(active_scores),
        ))

    minimum_speech = round(config.min_speech_duration_ms * sampling_rate / 1000)
    minimum_silence = round(config.min_silence_duration_ms * sampling_rate / 1000)
    padding = round(config.speech_pad_ms * sampling_rate / 1000)
    maximum = (
        None if getattr(config, "max_speech_duration_s", None) is None else round(
            config.max_speech_duration_s * sampling_rate))
    if maximum is not None and maximum <= 0:
        raise ValueError("`max_speech_duration_s` must resolve to at least one audio sample.")
    audio_end = (
        duration_samples if duration_samples is not None else
        (len(scores) - 1) * frame_hop_samples + frame_length_samples)

    clamped = [(start, min(end, audio_end), score_sum, score_count)
               for start, end, score_sum, score_count in raw if min(end, audio_end) > start]
    joined: list[tuple[int, int, float, int]] = []
    for start, end, score_sum, score_count in clamped:
        if joined and start - joined[-1][1] <= minimum_silence:
            joined[-1] = (
                joined[-1][0],
                max(joined[-1][1], end),
                joined[-1][2] + score_sum,
                joined[-1][3] + score_count,
            )
        else:
            joined.append((start, end, score_sum, score_count))
    joined = [region for region in joined if region[1] - region[0] >= minimum_speech]

    padded = []
    for start, end, score_sum, score_count in joined:
        start = max(0, start - padding)
        end = min(audio_end, end + padding)
        if padded and start < padded[-1][1]:
            midpoint = (padded[-1][1] + start) // 2
            previous = padded[-1]
            padded[-1] = (
                previous[0],
                midpoint,
                previous[2],
                previous[3],
            )
            start = midpoint
        padded.append((start, end, score_sum, score_count))

    split = []
    for start, end, score_sum, score_count in padded:
        if maximum is None:
            split.append((start, end, score_sum, score_count))
            continue
        cursor = start
        while end - cursor > maximum:
            split.append((
                cursor,
                cursor + maximum,
                score_sum,
                score_count,
            ))
            cursor += maximum
        if end > cursor:
            split.append((cursor, end, score_sum, score_count))

    return tuple(
        SpeechSegment(
            start=start / sampling_rate,
            end=end / sampling_rate,
            score=score_sum / score_count,
        ) for start, end, score_sum, score_count in split if end > start)


def normalize_backend_segments(
    values: Iterable[Mapping[str, Any] | Any],
    *,
    sampling_rate: int,
    timestamps_are_samples: bool = False,
    default_label: str = "speech",
) -> tuple[SpeechSegment, ...]:
    """Normalize mapping/object segments from a model-specific runtime."""
    if timestamps_are_samples:
        sampling_rate = _positive_integer(
            sampling_rate,
            name="sampling_rate",
        )
    normalized = []
    for value in values:
        if isinstance(value, Mapping):
            get = value.get
        else:
            get = lambda name, default=None: getattr(value, name, default)
        start = get("start")
        end = get("end", get("stop"))
        if start is None or end is None:
            raise ValueError("Backend VAD segments must provide start and end.")
        if timestamps_are_samples:
            start = float(start) / sampling_rate
            end = float(end) / sampling_rate
        normalized.append(
            SpeechSegment(
                start=float(start),
                end=float(end),
                score=get("score", get("confidence")),
                label=get("label", default_label),
                channel=get("channel"),
            ))
    return merge_speech_segments(normalized)
