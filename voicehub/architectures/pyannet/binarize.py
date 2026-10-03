"""Torch-free hysteresis binarization matching pyannote.audio ``Binarize``.

``pyannote.audio.utils.signal.Binarize`` (pinned revision
``795b92ab265888c58d160f90ae4d91b7bcc6aa2c``) places each frame at its
*middle* timestamp on the exact (non-integer) frame grid, opens a region
when a score is strictly above ``onset``, closes it at the first frame
strictly below ``offset``, and post-processes with
``pyannote.core.Annotation.support(collar=min_duration_off)`` before
removing regions shorter than ``min_duration_on``.  This module repeats
those float operations in the same order so VoiceHub reports the same
boundaries as the official ``VoiceActivityDetection`` pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

# ``pyannote.core.segment.SEGMENT_PRECISION``: segments (and gaps) no longer
# than one microsecond are treated as empty.
SEGMENT_PRECISION = 1e-6


def _is_empty(start: float, end: float) -> bool:
    return not (end - start) > SEGMENT_PRECISION


def _support(
    regions: list[tuple[float, float]],
    collar: float,
) -> list[tuple[float, float]]:
    """Mirror ``pyannote.core.Timeline.support_iter(collar)``."""
    merged: list[tuple[float, float]] = []
    for segment in sorted(regions):
        if not merged:
            merged.append(segment)
            continue
        current = merged[-1]
        # Gap ``segment ^ current`` as defined by pyannote.core.Segment.
        gap_start = min(segment[1], current[1])
        gap_end = max(segment[0], current[0])
        empty_gap = _is_empty(gap_start, gap_end)
        if empty_gap or gap_end - gap_start < collar:
            merged[-1] = (min(current[0], segment[0]), max(current[1], segment[1]))
        else:
            merged.append(segment)
    return merged


def frame_middles(num_frames: int, *, frame_step_s: float, frame_duration_s: float) -> list[float]:
    """Return ``SlidingWindow(start=0, duration, step)[i].middle`` values."""
    middles = []
    for index in range(num_frames):
        start = 0.0 + index * frame_step_s
        middles.append(0.5 * (start + (start + frame_duration_s)))
    return middles


def binarize_frame_scores(
    scores: Sequence[float],
    *,
    frame_step_s: float,
    frame_duration_s: float,
    onset: float = 0.5,
    offset: float | None = None,
    min_duration_on: float = 0.0,
    min_duration_off: float = 0.0,
    pad_onset: float = 0.0,
    pad_offset: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    """Return speech regions in seconds exactly as pyannote ``Binarize``.

    Boundaries are not clipped to the recording; callers that require
    in-range segments clip them afterwards.
    """
    offset = onset if offset is None else offset
    values = [float(value) for value in scores]
    if not values:
        return ()
    timestamps = frame_middles(
        len(values),
        frame_step_s=frame_step_s,
        frame_duration_s=frame_duration_s,
    )
    regions: list[tuple[float, float]] = []

    def add(start: float, end: float) -> None:
        # ``Annotation.__setitem__`` silently ignores empty segments.
        if not _is_empty(start, end):
            regions.append((start, end))

    start = timestamps[0]
    is_active = values[0] > onset
    last = timestamps[0]
    for timestamp, value in zip(timestamps[1:], values[1:]):
        last = timestamp
        if is_active:
            if value < offset:
                add(start - pad_onset, timestamp + pad_offset)
                start = timestamp
                is_active = False
        elif value > onset:
            start = timestamp
            is_active = True
    if is_active:
        add(start - pad_onset, last + pad_offset)

    if pad_offset > 0.0 or pad_onset > 0.0 or min_duration_off > 0.0:
        # ``Annotation.support(collar)`` runs ``Timeline.support(collar)``
        # followed by a collar-free ``support()``.
        regions = _support(_support(regions, min_duration_off), 0.0)
    if min_duration_on > 0:
        regions = [(start, end) for start, end in regions
                   if (0.0 if _is_empty(start, end) else end - start) >= min_duration_on]
    return tuple(sorted(regions))


__all__ = ["SEGMENT_PRECISION", "binarize_frame_scores", "frame_middles"]
