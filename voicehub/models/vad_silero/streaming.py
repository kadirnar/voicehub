"""Request-local incremental sessions for native Silero VAD."""

from __future__ import annotations

from threading import RLock
from typing import Any


class SileroVADStreamingSession:
    """Buffer only a partial frame while owning one explicit Silero state."""

    def __init__(
        self,
        wrapper: Any,
        *,
        sampling_rate: int,
        segmentation_config: Any,
        return_frames: bool,
    ) -> None:
        from voicehub.models.vad_silero.native.modeling import SileroVADStream

        if wrapper.model is None or wrapper.native_config is None:
            raise RuntimeError("Silero VAD must be loaded before streaming.")
        if sampling_rate != wrapper.sample_rate:
            raise ValueError("Streaming sample rate must match the loaded Silero graph.")
        if not isinstance(return_frames, bool):
            raise TypeError("`return_frames` must be a boolean.")
        self.wrapper = wrapper
        self.sampling_rate = sampling_rate
        self.segmentation_config = segmentation_config
        self.return_frames = return_frames
        self._stream = SileroVADStream(wrapper.model)
        self._buffer: Any | None = None
        self._probabilities: list[Any] = []
        self._valid_samples = 0
        self._result: Any | None = None
        self._closed = False
        self._lock = RLock()

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def _process_complete_frames(self) -> tuple[float, ...]:
        import torch

        if self._buffer is None:
            return ()
        frame_size = self.wrapper.native_config.frame_size
        complete_samples = (self._buffer.numel() // frame_size) * frame_size
        if complete_samples == 0:
            return ()
        complete = self._buffer[:complete_samples]
        self._buffer = self._buffer[complete_samples:]
        emitted: list[float] = []
        with torch.inference_mode():
            for offset in range(0, complete_samples, frame_size):
                probability = self._stream.process(complete[offset:offset + frame_size].unsqueeze(0))
                detached = probability.detach().float().cpu().reshape(1)
                self._probabilities.append(detached)
                emitted.append(float(detached.item()))
        return tuple(emitted)

    def push(self, audio_chunk: Any) -> tuple[float, ...]:
        """Process every complete frame and retain at most one partial
        frame."""
        from voicehub.processing.waveform import load_native_audio

        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot push audio to a closed stream.")
            if self._result is not None:
                raise RuntimeError("Reset the stream before pushing after flush().")
            materialized = load_native_audio(
                audio_chunk,
                sampling_rate=self.sampling_rate,
                target_sampling_rate=self.sampling_rate,
            )
            chunk = materialized.waveform.to(device=self.wrapper.device)
            self._valid_samples += chunk.numel()
            if self._buffer is None or self._buffer.numel() == 0:
                self._buffer = chunk
            else:
                import torch

                self._buffer = torch.cat((self._buffer, chunk))
            return self._process_complete_frames()

    def flush(self) -> Any:
        """Pad the final frame once, segment all scores, and cache the
        result."""
        import torch

        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot flush a closed stream.")
            if self._result is not None:
                return self._result
            if self._valid_samples == 0:
                raise ValueError("Cannot flush a stream that has no audio.")
            self._process_complete_frames()
            if self._buffer is not None and self._buffer.numel():
                frame_size = self.wrapper.native_config.frame_size
                padded = torch.nn.functional.pad(
                    self._buffer,
                    (0, frame_size - self._buffer.numel()),
                )
                with torch.inference_mode():
                    probability = self._stream.process(padded.unsqueeze(0))
                self._probabilities.append(probability.detach().float().cpu().reshape(1))
                self._buffer = self._buffer[:0]
            probabilities = torch.cat(self._probabilities)
            self._result = self.wrapper._probabilities_to_output(
                probabilities,
                valid_samples=self._valid_samples,
                segmentation_config=self.segmentation_config,
                return_frames=self.return_frames,
                streaming=True,
            )
            return self._result

    def reset(self) -> None:
        """Discard only this session's audio and recurrent state."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot reset a closed stream.")
            self._stream.reset()
            self._buffer = None
            self._probabilities.clear()
            self._valid_samples = 0
            self._result = None

    def close(self) -> None:
        with self._lock:
            self._stream.reset()
            self._buffer = None
            self._probabilities.clear()
            self._closed = True

    def __enter__(self) -> SileroVADStreamingSession:
        if self.is_closed:
            raise RuntimeError("Cannot re-enter a closed stream.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["SileroVADStreamingSession"]
