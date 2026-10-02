"""Native port of OmniVoice's pydub-based silence trimming.

Upstream OmniVoice converts audio to 16-bit PCM, splits it on long
silences with ``pydub.silence.split_on_silence`` and trims the edges with
``detect_leading_silence``. Those steps run on the voice-cloning prompt
(``preprocess_prompt``) and on the generated waveform
(``postprocess_output``) by default, so they change both the prompt codes
and the returned audio length. This module reproduces pydub 0.25.1
exactly (millisecond slicing, zero padding of the final partial
millisecond, and integer ``audioop.rms``) with only PyTorch and the
standard library.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_SILENCE_THRESHOLD_DBFS = -50.0
_MAXIMUM_AMPLITUDE = 32_768.0
_SEEK_STEP_MS = 10


class _PCMSegment:
    """The subset of ``pydub.AudioSegment`` used by OmniVoice (mono)."""

    def __init__(self, samples: Tensor, sample_rate: int) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        squares = samples.to(torch.int64).square()
        self._cumulative = torch.cat((squares.new_zeros(1), squares.cumsum(0))).tolist()

    def __len__(self) -> int:
        return round(1000 * (self.samples.numel() / self.sample_rate))

    def _frame(self, milliseconds: int) -> int:
        return int(milliseconds * (self.sample_rate / 1000.0))

    def _bounds(self, start: int, end: int) -> tuple[int, int]:
        length = len(self)
        return self._frame(min(start, length)), self._frame(min(end, length))

    def rms(self, start: int, end: int) -> int:
        """``audioop.rms`` of ``segment[start:end]`` including zero padding."""
        first, last = self._bounds(start, end)
        count = last - first
        if count <= 0:
            return 0
        available = min(last, self.samples.numel())
        total = self._cumulative[available] - self._cumulative[min(first, available)]
        return int(math.sqrt(total / count))

    def dbfs(self, start: int, end: int) -> float:
        rms = self.rms(start, end)
        if not rms:
            return -math.inf
        return 20 * math.log(rms / _MAXIMUM_AMPLITUDE, 10)

    def slice(self, start: int, end: int) -> Tensor:
        first, last = self._bounds(start, end)
        values = self.samples[first:last]
        missing = max(0, last - first) - values.numel()
        if missing > 0:
            values = torch.cat((values, values.new_zeros(missing)))
        return values


def _detect_silence(segment: _PCMSegment, minimum: int, threshold: float) -> list[list[int]]:
    length = len(segment)
    if length < minimum:
        return []
    threshold = 10**(threshold / 20) * _MAXIMUM_AMPLITUDE
    last_start = length - minimum
    starts = list(range(0, last_start + 1, _SEEK_STEP_MS))
    if last_start % _SEEK_STEP_MS:
        starts.append(last_start)
    silent = [start for start in starts if segment.rms(start, start + minimum) <= threshold]
    if not silent:
        return []
    ranges = []
    previous = silent.pop(0)
    range_start = previous
    for start in silent:
        continuous = start == previous + _SEEK_STEP_MS
        has_gap = start > previous + minimum
        if not continuous and has_gap:
            ranges.append([range_start, previous + minimum])
            range_start = start
        previous = start
    ranges.append([range_start, previous + minimum])
    return ranges


def _detect_nonsilent(segment: _PCMSegment, minimum: int, threshold: float) -> list[list[int]]:
    silent = _detect_silence(segment, minimum, threshold)
    length = len(segment)
    if not silent:
        return [[0, length]]
    if silent[0][0] == 0 and silent[0][1] == length:
        return []
    previous_end = 0
    ranges = []
    for start, end in silent:
        ranges.append([previous_end, start])
        previous_end = end
    if end != length:
        ranges.append([previous_end, length])
    if ranges[0] == [0, 0]:
        ranges.pop(0)
    return ranges


def _split_on_silence(segment: _PCMSegment, minimum: int) -> Tensor:
    ranges = [[start - minimum, end + minimum]
              for start, end in _detect_nonsilent(segment, minimum, _SILENCE_THRESHOLD_DBFS)]
    for current, following in zip(ranges, ranges[1:]):
        if following[0] < current[1]:
            current[1] = (current[1] + following[0]) // 2
            following[0] = current[1]
    length = len(segment)
    pieces = [segment.slice(max(start, 0), min(end, length)) for start, end in ranges]
    return torch.cat(pieces) if pieces else segment.samples[:0]


def _leading_silence(segment: _PCMSegment) -> int:
    trim = 0
    length = len(segment)
    while (segment.dbfs(trim, trim + _SEEK_STEP_MS) < _SILENCE_THRESHOLD_DBFS and trim < length):
        trim += _SEEK_STEP_MS
    return min(trim, length)


def _trim_edges(samples: Tensor, sample_rate: int, *, leading: int, trailing: int) -> Tensor:
    segment = _PCMSegment(samples, sample_rate)
    start = max(0, _leading_silence(segment) - leading)
    samples = segment.slice(start, len(segment)).flip(0)
    segment = _PCMSegment(samples, sample_rate)
    start = max(0, _leading_silence(segment) - trailing)
    return segment.slice(start, len(segment)).flip(0)


def quantize_pcm16(waveform: Tensor) -> Tensor:
    """Integer samples of ``(x * 32768).clip(-32768, 32767).astype(int16)``."""
    return (waveform.float() * _MAXIMUM_AMPLITUDE).clamp(-32_768.0, 32_767.0).trunc().to(torch.int16)


def remove_silence(
    waveform: Tensor,
    sample_rate: int,
    *,
    middle_ms: int,
    leading_ms: int,
    trailing_ms: int,
) -> Tensor:
    """Remove long middle silences and trim edges, as upstream OmniVoice does.

    The returned mono float32 waveform is quantized to 16-bit PCM, exactly
    like the upstream pydub round trip, even when no silence is removed.
    """
    if not isinstance(waveform, Tensor) or waveform.ndim != 1:
        raise ValueError("Silence removal expects a mono [sample] waveform.")
    samples = quantize_pcm16(waveform.detach().cpu())
    if middle_ms > 0:
        samples = _split_on_silence(_PCMSegment(samples, sample_rate), middle_ms)
    samples = _trim_edges(samples, sample_rate, leading=leading_ms, trailing=trailing_ms)
    return (samples.float() / _MAXIMUM_AMPLITUDE).to(waveform.device)


__all__ = ["quantize_pcm16", "remove_silence"]
