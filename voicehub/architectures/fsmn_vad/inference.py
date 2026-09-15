"""FunASR-compatible endpoint decoding for native FSMN VAD scores."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from voicehub.architectures.fsmn_vad.configuration import FSMNVADConfig


@dataclass(frozen=True, slots=True)
class FSMNVADBoundary:
    """One native millisecond speech boundary."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if (isinstance(self.start_ms, bool) or isinstance(self.end_ms, bool) or
                not isinstance(self.start_ms, int) or not isinstance(self.end_ms, int)):
            raise TypeError("FSMN VAD boundaries must use integer milliseconds.")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("FSMN VAD boundaries require 0 <= start < end.")


class _WindowDetector:

    def __init__(self, config: FSMNVADConfig) -> None:
        self.size = config.window_size_ms // config.frame_shift_ms
        self.speech_threshold = (config.silence_to_speech_ms // config.frame_shift_ms)
        self.silence_threshold = (config.speech_to_silence_ms // config.frame_shift_ms)
        self.reset()

    def reset(self) -> None:
        self.values = [0] * self.size
        self.position = 0
        self.total = 0
        self.speech = False

    def update(self, speech: bool) -> str:
        value = int(speech)
        self.total -= self.values[self.position]
        self.total += value
        self.values[self.position] = value
        self.position = (self.position + 1) % self.size
        if not self.speech and self.total >= self.speech_threshold:
            self.speech = True
            return "silence-to-speech"
        if self.speech and self.total <= self.silence_threshold:
            self.speech = False
            return "speech-to-silence"
        return "speech" if self.speech else "silence"


class FSMNVADDecoder:
    """Stateful endpoint detector with request-local, resettable state."""

    def __init__(
        self,
        config: FSMNVADConfig,
        *,
        speech_noise_threshold: float | None = None,
        max_end_silence_ms: int | None = None,
        max_single_segment_ms: int | None = None,
    ) -> None:
        self.config = FSMNVADConfig.coerce(config)
        self.speech_noise_threshold = (
            self.config.speech_noise_threshold
            if speech_noise_threshold is None else float(speech_noise_threshold))
        if not 0.0 <= self.speech_noise_threshold <= 1.0:
            raise ValueError("`speech_noise_threshold` must be in [0, 1].")
        self.max_end_silence_ms = (
            self.config.max_end_silence_ms if max_end_silence_ms is None else int(max_end_silence_ms))
        self.max_single_segment_ms = (
            self.config.max_single_segment_ms
            if max_single_segment_ms is None else int(max_single_segment_ms))
        if self.max_end_silence_ms < 0:
            raise ValueError("`max_end_silence_ms` cannot be negative.")
        if self.max_single_segment_ms < 1:
            raise ValueError("`max_single_segment_ms` must be positive.")
        self.window = _WindowDetector(self.config)
        self.reset()

    @property
    def latency_frames(self) -> int:
        return (self.window.size + self.config.lookback_start_ms // self.config.frame_shift_ms)

    def reset(self) -> None:
        self.window.reset()
        self.frame_index = 0
        self.data_start_frame = 0
        self.segment_start_frame: int | None = None
        self.latest_speech_frame = 0
        self.continuous_silence_frames = 0
        self.noise_average_decibel = -100.0
        self.boundaries: list[FSMNVADBoundary] = []

    def _is_speech(
        self,
        speech_probability: float,
        silence_probability: float,
        decibel: float,
    ) -> bool:
        signal_to_noise = decibel - self.noise_average_decibel
        required_noise = silence_probability**self.config.speech_to_noise_ratio
        speech = (
            speech_probability >= required_noise + self.speech_noise_threshold and
            signal_to_noise >= self.config.snr_threshold and decibel >= self.config.decibel_threshold)
        if not speech:
            if self.noise_average_decibel < -99.9:
                self.noise_average_decibel = decibel
            else:
                history = self.config.noise_history_frames
                self.noise_average_decibel = (decibel + self.noise_average_decibel * (history - 1)) / history
        return speech

    def _finish(self, end_frame: int) -> FSMNVADBoundary | None:
        start = self.segment_start_frame
        if start is None:
            return None
        end_frame = max(start, end_frame)
        boundary = FSMNVADBoundary(
            start_ms=start * self.config.frame_shift_ms,
            end_ms=(end_frame + 1) * self.config.frame_shift_ms,
        )
        self.boundaries.append(boundary)
        self.data_start_frame = end_frame + 1
        self.segment_start_frame = None
        self.latest_speech_frame = 0
        self.continuous_silence_frames = 0
        self.window.reset()
        return boundary

    def process(
        self,
        speech_probabilities: Tensor,
        *,
        silence_probabilities: Tensor | None = None,
        decibels: Tensor | None = None,
        final: bool = False,
    ) -> tuple[FSMNVADBoundary, ...]:
        """Consume consecutive frames and return newly completed boundaries."""
        if not isinstance(speech_probabilities, Tensor):
            speech_probabilities = torch.as_tensor(speech_probabilities)
        speech_values = speech_probabilities.detach().float().cpu().reshape(-1)
        if silence_probabilities is None:
            silence_values = 1.0 - speech_values
        else:
            if not isinstance(silence_probabilities, Tensor):
                silence_probabilities = torch.as_tensor(silence_probabilities)
            silence_values = silence_probabilities.detach().float().cpu().reshape(-1)
        if decibels is None:
            decibel_values = torch.zeros_like(speech_values)
        else:
            if not isinstance(decibels, Tensor):
                decibels = torch.as_tensor(decibels)
            decibel_values = decibels.detach().float().cpu().reshape(-1)
        if not (speech_values.numel() == silence_values.numel() == decibel_values.numel()):
            raise ValueError("Speech, silence, and decibel frame counts must match.")
        if (not torch.isfinite(speech_values).all() or not torch.isfinite(silence_values).all() or
                not torch.isfinite(decibel_values).all()):
            raise ValueError("FSMN decoder inputs must be finite.")
        invalid_probability = (((speech_values < 0) | (speech_values > 1)).any() or
                               ((silence_values < 0) | (silence_values > 1)).any())
        if invalid_probability:
            raise ValueError("FSMN decoder probabilities must be in [0, 1].")

        emitted: list[FSMNVADBoundary] = []
        # Convert once: indexing three scalar tensors per 10 ms frame adds
        # substantial Python/PyTorch dispatch overhead to endpoint decoding.
        for probability, silence, decibel in zip(speech_values.tolist(), silence_values.tolist(),
                                                 decibel_values.tolist()):
            frame = self.frame_index
            is_speech = self._is_speech(probability, silence, decibel)
            change = self.window.update(is_speech)
            if change == "silence-to-speech":
                self.continuous_silence_frames = 0
                if self.segment_start_frame is None:
                    self.segment_start_frame = max(
                        self.data_start_frame,
                        frame - self.latency_frames,
                    )
                self.latest_speech_frame = frame
            elif change in {"speech-to-silence", "speech"}:
                self.continuous_silence_frames = 0
                if self.segment_start_frame is not None:
                    self.latest_speech_frame = frame
            else:
                self.continuous_silence_frames += 1
                if self.segment_start_frame is None:
                    if frame >= self.latency_frames:
                        self.data_start_frame = frame - self.latency_frames
                else:
                    end_silence_threshold = max(
                        0,
                        (self.max_end_silence_ms - self.config.speech_to_silence_ms),
                    )
                    if (self.continuous_silence_frames * self.config.frame_shift_ms >= end_silence_threshold):
                        lookback = (end_silence_threshold // self.config.frame_shift_ms)
                        lookback -= (self.config.lookahead_end_ms // self.config.frame_shift_ms)
                        lookback = max(0, lookback - 1)
                        boundary = self._finish(frame - lookback)
                        if boundary is not None:
                            emitted.append(boundary)
            if (self.segment_start_frame is not None and
                (frame - self.segment_start_frame + 1) * self.config.frame_shift_ms
                    > self.max_single_segment_ms):
                boundary = self._finish(frame)
                if boundary is not None:
                    emitted.append(boundary)
            self.frame_index += 1

        if final and self.segment_start_frame is not None:
            final_frame = max(
                self.segment_start_frame,
                self.frame_index - 1,
            )
            boundary = self._finish(final_frame)
            if boundary is not None:
                emitted.append(boundary)
        return tuple(emitted)


def frame_decibels(
    waveform: Tensor,
    *,
    config: FSMNVADConfig,
    frame_count: int | None = None,
) -> Tensor:
    """Compute the upstream 25 ms energy measure at 10 ms steps."""
    resolved = FSMNVADConfig.coerce(config)
    if not isinstance(waveform, Tensor):
        waveform = torch.as_tensor(waveform)
    values = waveform.reshape(-1)
    available = (
        0 if values.numel() < resolved.frame_length_samples else 1 +
        (values.numel() - resolved.frame_length_samples) // resolved.frame_shift_samples)
    count = available if frame_count is None else int(frame_count)
    if count < 0 or count > available:
        raise ValueError(f"Requested {count} decibel frames, but only {available} are available.")
    if count == 0:
        return values.new_empty(0, dtype=torch.float32)
    frames = values.unfold(
        0,
        resolved.frame_length_samples,
        resolved.frame_shift_samples,
    )[:count]
    return 10.0 * (frames.float().square().sum(dim=-1) + 0.000001).log10()


__all__ = [
    "FSMNVADBoundary",
    "FSMNVADDecoder",
    "frame_decibels",
]
