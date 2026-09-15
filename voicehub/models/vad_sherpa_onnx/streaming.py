"""Native scorers and the pinned Sherpa VAD streaming state machine."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True, slots=True)
class NativeSpeechSegment:
    """Sherpa-compatible sample-index segment."""

    start: int
    samples: Any


class _SampleBuffer:
    """Linear-index sample buffer matching Sherpa's circular-buffer API."""

    def __init__(self) -> None:
        import torch

        self._values = torch.empty(0, dtype=torch.float32)
        self._head = 0

    @property
    def head(self) -> int:
        return self._head

    @property
    def tail(self) -> int:
        return self._head + self._values.numel()

    @property
    def size(self) -> int:
        return self._values.numel()

    def push(self, values: Any) -> None:
        import torch

        values = values.detach().cpu().to(dtype=torch.float32)
        self._values = torch.cat((self._values, values))

    def get(self, start: int, length: int):
        if start < self.head or start >= self.tail:
            raise IndexError("Sample-buffer start is outside retained audio.")
        if length < 0 or start + length > self.tail:
            raise IndexError("Sample-buffer range is outside retained audio.")
        offset = start - self.head
        return self._values[offset:offset + length].clone()

    def pop(self, length: int) -> None:
        if length < 0 or length > self.size:
            raise IndexError("Cannot pop outside the retained sample buffer.")
        self._values = self._values[length:].clone()
        self._head += length

    def reset(self) -> None:
        self._values = self._values[:0]
        self._head = 0


class NativeTENScorer:
    """Request-local TEN frontend and four recurrent tensors."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.state = model.initial_state(
            1,
            device=next(model.parameters()).device,
        )

    @property
    def window_size(self) -> int:
        return self.model.config.window_size

    @property
    def window_shift(self) -> int:
        return self.model.config.window_size

    def compute(self, samples: Any) -> float:
        import torch

        values = samples.to(
            device=next(self.model.parameters()).device,
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.inference_mode():
            output, state = self.model.score_audio_frame(values, self.state)
        self.state = state.detached()
        return float(output.speech_probabilities.item())

    def reset(self) -> None:
        self.state = self.model.initial_state(
            1,
            device=next(self.model.parameters()).device,
        )


class NativeSileroScorer:
    """Sherpa v5 window contract over VoiceHub's verified Silero graph."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.state = model.initial_state(
            1,
            device=next(model.parameters()).device,
        )

    @property
    def window_size(self) -> int:
        return self.model.config.frame_size + self.model.config.context_size

    @property
    def window_shift(self) -> int:
        return self.model.config.frame_size

    def compute(self, samples: Any) -> float:
        import torch

        values = samples.to(
            device=next(self.model.parameters()).device,
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.inference_mode():
            self.model._validate_model_input(values)
            probabilities, _, (hidden, cell) = self.model.forward_with_context(
                values, (self.state.hidden, self.state.cell))
        # Sherpa supplies overlapping context+frame windows (576 samples at
        # 16 kHz), rather than the zero-prefixed first frame of the standalone
        # Silero API. Do not prepend another context or shift its timestamps.
        self.state = type(self.state)(
            hidden=hidden.detach(),
            cell=cell.detach(),
            context=values[:, -self.model.config.context_size:].detach())
        return float(probabilities.item())

    def reset(self) -> None:
        self.state = self.model.initial_state(
            1,
            device=next(self.model.parameters()).device,
        )


class _SpeechDecision:
    """Exact TEN/Silero decision hysteresis from the pinned Sherpa source."""

    def __init__(
        self,
        *,
        family: str,
        sample_rate: int,
        window_shift: int,
        threshold: float,
        negative_threshold: float | None,
        min_speech_duration: float,
        min_silence_duration: float,
    ) -> None:
        self.family = family
        self.sample_rate = sample_rate
        self.window_shift = window_shift
        self.original_threshold = threshold
        self.threshold = threshold
        self.negative_threshold = negative_threshold
        self.original_min_silence = min_silence_duration
        self.min_silence_samples = int(sample_rate * min_silence_duration)
        self.min_speech_samples = int(sample_rate * min_speech_duration)
        self.reset()

    def reset(self) -> None:
        self.triggered = False
        self.current_sample = 0
        self.temp_start = 0
        self.temp_end = 0
        self.threshold = self.original_threshold
        self.min_silence_samples = int(self.sample_rate * self.original_min_silence)

    def configure_long_utterance(self, enabled: bool) -> None:
        if enabled:
            self.min_silence_samples = int(self.sample_rate * 0.1)
            self.threshold = 0.9
        else:
            self.min_silence_samples = int(self.sample_rate * self.original_min_silence)
            self.threshold = self.original_threshold

    def update(self, probability: float) -> bool:
        threshold = self.threshold
        self.current_sample += self.window_shift
        if probability > threshold and self.temp_end != 0:
            self.temp_end = 0
        if probability > threshold and self.temp_start == 0:
            self.temp_start = self.current_sample
            return False
        if probability > threshold and self.temp_start != 0 and not self.triggered:
            if self.current_sample - self.temp_start < self.min_speech_samples:
                return False
            self.triggered = True
            return True
        if probability < threshold and not self.triggered:
            self.temp_start = 0
            self.temp_end = 0
            return False
        if self.family == "silero":
            negative = (
                max(threshold -
                    0.15, 0.01) if self.negative_threshold is None else max(self.negative_threshold, 0.01))
        else:
            negative = threshold - 0.15
        if probability > negative and self.triggered:
            return True
        if probability > threshold and not self.triggered:
            self.triggered = True
            return True
        if probability < threshold and self.triggered:
            if self.temp_end == 0:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return True
            self.temp_start = 0
            self.temp_end = 0
            self.triggered = False
            return False
        return False


class NativeSherpaVoiceActivityDetector:
    """Sherpa-compatible buffering, decisions, and segment queue."""

    def __init__(
        self,
        scorer: NativeTENScorer | NativeSileroScorer,
        *,
        family: str,
        sample_rate: int,
        threshold: float,
        negative_threshold: float | None,
        min_speech_duration: float,
        min_silence_duration: float,
        max_speech_duration: float,
    ) -> None:
        import torch

        self.scorer = scorer
        self.family = family
        self.sample_rate = sample_rate
        self.max_utterance_length = int(sample_rate * max_speech_duration)
        self.decision = _SpeechDecision(
            family=family,
            sample_rate=sample_rate,
            window_shift=scorer.window_shift,
            threshold=threshold,
            negative_threshold=negative_threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
        )
        self.buffer = _SampleBuffer()
        self.last = torch.empty(0, dtype=torch.float32)
        self.start = -1
        self.current_segment = NativeSpeechSegment(
            start=-1,
            samples=torch.empty(0, dtype=torch.float32),
        )
        self._segments: list[NativeSpeechSegment] = []
        self._lock = RLock()

    @property
    def empty(self) -> bool:
        return not self._segments

    @property
    def front(self) -> NativeSpeechSegment:
        if not self._segments:
            raise IndexError("No completed speech segment is available.")
        return self._segments[0]

    def pop(self) -> None:
        if not self._segments:
            raise IndexError("No completed speech segment is available.")
        self._segments.pop(0)

    def accept_waveform(self, samples: Any) -> tuple[float, ...]:
        import torch

        with self._lock:
            values = torch.as_tensor(samples, dtype=torch.float32).reshape(-1)
            if values.numel() == 0:
                return ()
            if not torch.isfinite(values).all():
                raise ValueError("VAD audio cannot contain NaN or infinite values.")
            self.decision.configure_long_utterance(self.buffer.size > self.max_utterance_length)
            combined = torch.cat((self.last, values.detach().cpu()))
            window_size = self.scorer.window_size
            shift = self.scorer.window_shift
            if combined.numel() < window_size:
                self.last = combined
                return ()
            count = (combined.numel() - window_size) // shift + 1
            probabilities = []
            any_speech = False
            cursor = 0
            for _ in range(count):
                window = combined[cursor:cursor + window_size]
                self.buffer.push(window[:shift])
                probability = self.scorer.compute(window)
                probabilities.append(probability)
                any_speech = self.decision.update(probability) or any_speech
                cursor += shift
            self.last = combined[cursor:].clone()

            if any_speech:
                if self.start == -1:
                    self.start = max(
                        self.buffer.tail - 2 * window_size - self.decision.min_speech_samples,
                        self.buffer.head,
                    )
                length = self.buffer.tail - self.start - 1
                current = (self.buffer.get(self.start, length) if length > 0 else self.last[:0])
                self.current_segment = NativeSpeechSegment(
                    start=self.start,
                    samples=current,
                )
            else:
                self.current_segment = NativeSpeechSegment(
                    start=-1,
                    samples=self.last[:0],
                )
                if self.start != -1 and self.buffer.size:
                    end = self.buffer.tail - self.decision.min_silence_samples
                    if end > self.start:
                        self._segments.append(
                            NativeSpeechSegment(
                                start=self.start,
                                samples=self.buffer.get(
                                    self.start,
                                    end - self.start,
                                ),
                            ))
                        self.buffer.pop(end - self.buffer.head)
                if self.start == -1:
                    end = (self.buffer.tail - 2 * window_size - self.decision.min_speech_samples)
                    length = max(0, end - self.buffer.head)
                    if length:
                        self.buffer.pop(length)
                self.start = -1
            return tuple(probabilities)

    def flush(self) -> None:
        with self._lock:
            if self.start == -1 or self.buffer.size == 0:
                return
            end = self.buffer.tail
            if end <= self.start:
                return
            self._segments.append(
                NativeSpeechSegment(
                    start=self.start,
                    samples=self.buffer.get(self.start, end - self.start),
                ))
            self.buffer.pop(end - self.buffer.head)
            self.start = -1
            self.current_segment = NativeSpeechSegment(
                start=-1,
                samples=self.last[:0],
            )

    def reset(self) -> None:
        with self._lock:
            self._segments.clear()
            self.scorer.reset()
            self.decision.reset()
            self.buffer.reset()
            self.last = self.last[:0]
            self.start = -1
            self.current_segment = NativeSpeechSegment(
                start=-1,
                samples=self.last[:0],
            )


__all__ = [
    "NativeSherpaVoiceActivityDetector",
    "NativeSileroScorer",
    "NativeSpeechSegment",
    "NativeTENScorer",
]
