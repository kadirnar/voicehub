"""Torch-only sliding inference and overlap-add for PyanNet."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from voicehub.architectures.pyannet.modeling import PyanNet


@dataclass(frozen=True, slots=True)
class PyanNetFrameOutput:
    """Aggregated full-recording frame output."""

    scores: Tensor
    frame_hop_samples: int
    frame_length_samples: int
    frame_start_samples: int
    valid_samples: int
    frame_step_s: float | None = None
    frame_duration_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scores, Tensor) or self.scores.ndim != 2:
            raise ValueError("`scores` must have shape [frames, outputs].")
        if self.scores.shape[0] < 1:
            raise ValueError("`scores` must contain at least one frame.")


def _repeat_pad(values: Tensor, target_samples: int) -> Tensor:
    if values.shape[-1] == 0:
        return functional.pad(values, (0, target_samples))
    repeats = math.ceil(target_samples / values.shape[-1])
    return values.repeat(repeats)[:target_samples]


def _closest_frame(time_s: float, *, frame_step_s: float) -> int:
    """Mirror ``pyannote.core.SlidingWindow.closest_frame`` (start 0).

    pyannote evaluates the rule in seconds with ``np.rint`` (round half
    to even, like Python's ``round``).  Evaluating it in samples instead
    rounds ties such as the 30 s chunk start on the segmentation-3.0
    grid differently and shifts that chunk by one frame.
    """
    return max(0, round((time_s - 0.0 - 0.5 * frame_step_s) / frame_step_s))


def _numpy_hamming(size: int) -> Tensor:
    """Return ``np.hamming(size)`` as float64 using NumPy's formula."""
    if size == 1:
        return torch.ones(1, dtype=torch.float64)
    return torch.tensor(
        [0.54 + 0.46 * math.cos(math.pi * index / (size - 1)) for index in range(1 - size, size, 2)],
        dtype=torch.float64,
    )


class PyanNetFrameInference:
    """Pinned pyannote-style chunking and Hamming overlap-add."""

    def __init__(
        self,
        model: PyanNet,
        *,
        batch_size: int = 32,
        duration_s: float | None = None,
        step_s: float | None = None,
    ) -> None:
        if not isinstance(model, PyanNet):
            raise TypeError("`model` must be a PyanNet instance.")
        if (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("`batch_size` must be a positive integer.")
        self.model = model
        self.batch_size = batch_size
        self.duration_s = (model.config.chunk_duration_s if duration_s is None else float(duration_s))
        self.step_s = (model.config.chunk_step_s if step_s is None else float(step_s))
        for name, value in (
            ("duration_s", self.duration_s),
            ("step_s", self.step_s),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"`{name}` must be finite and positive.")
        if self.step_s > self.duration_s:
            raise ValueError("`step_s` cannot exceed `duration_s`.")

    def _chunks(
        self,
        waveform: Tensor,
    ) -> tuple[Tensor, tuple[int, ...], bool]:
        config = self.model.config
        chunk_samples = round(self.duration_s * config.sampling_rate)
        step_samples = round(self.step_s * config.sampling_rate)
        sample_count = waveform.shape[-1]
        chunks = []
        starts = []
        if sample_count >= chunk_samples:
            complete = (sample_count - chunk_samples) // step_samples + 1
            for index in range(complete):
                start = index * step_samples
                starts.append(start)
                chunks.append(waveform[start:start + chunk_samples])
            next_start = complete * step_samples
            has_last = (sample_count - chunk_samples) % step_samples > 0
        else:
            next_start = 0
            has_last = True
        if has_last:
            final = waveform[next_start:]
            if config.repeat_final_chunk:
                final = _repeat_pad(final, chunk_samples)
            else:
                final = functional.pad(
                    final,
                    (0, chunk_samples - final.shape[-1]),
                )
            starts.append(next_start)
            chunks.append(final)
        return torch.stack(chunks), tuple(starts), has_last

    def __call__(self, waveform: Tensor) -> PyanNetFrameOutput:
        if not isinstance(waveform, Tensor):
            raise TypeError("`waveform` must be a PyTorch tensor.")
        if waveform.ndim != 1 or waveform.numel() < 1:
            raise ValueError("`waveform` must be a non-empty rank-one tensor.")
        if not waveform.is_floating_point():
            raise TypeError("`waveform` must use a floating-point dtype.")
        if not torch.isfinite(waveform).all():
            raise ValueError("`waveform` cannot contain NaN or infinite values.")

        parameter = next(self.model.parameters())
        chunks, starts, has_last = self._chunks(waveform.to(device=parameter.device, dtype=parameter.dtype))
        chunk_outputs = []
        for offset in range(0, chunks.shape[0], self.batch_size):
            probabilities = self.model(chunks[offset:offset + self.batch_size])
            if self.model.config.is_brouhaha:
                chunk_outputs.append(probabilities)
            else:
                chunk_outputs.append(self.model.speech_probabilities(probabilities).unsqueeze(-1))
        scores = torch.cat(chunk_outputs, dim=0)
        _, frames_per_chunk, output_size = scores.shape
        config = self.model.config
        chunk_samples = round(self.duration_s * config.sampling_rate)
        frame_hop = (
            self.model.config.sinc_stride * 27 if config.is_brouhaha else chunk_samples / frames_per_chunk)
        # pyannote.audio builds its frame grid in seconds:
        # SlidingWindow(start=0, duration=step=chunk duration / frames).
        frame_step_s = (
            frame_hop / config.sampling_rate if config.is_brouhaha else self.duration_s / frames_per_chunk)
        starts_in_frames = tuple(
            _closest_frame(0.0 + index * self.step_s, frame_step_s=frame_step_s)
            for index in range(len(starts)))
        required_frames = _closest_frame(
            0.0 + self.duration_s + (len(starts) - 1) * self.step_s,
            frame_step_s=frame_step_s,
        ) + 1
        final_stop = starts_in_frames[-1] + frames_per_chunk
        if final_stop > required_frames:
            raise RuntimeError("PyanNet frame geometry produced an invalid aggregation "
                               "extent.")
        # pyannote.audio.core.inference.Inference.aggregate multiplies the
        # float32 scores by a float64 ``np.hamming`` window and accumulates
        # into float32 buffers; repeat that rounding order.
        summed = torch.zeros(
            required_frames,
            output_size,
            dtype=torch.float32,
            device=scores.device,
        )
        weights = torch.zeros_like(summed)
        window = _numpy_hamming(frames_per_chunk).to(device=scores.device).unsqueeze(-1)
        for index, start in enumerate(starts_in_frames):
            stop = start + frames_per_chunk
            summed[start:stop] = (summed[start:stop].double() + scores[index].double() * window).float()
            weights[start:stop] = (weights[start:stop].double() + window).float()
        aggregated = (summed / weights.clamp_min(1e-12)).to(dtype=scores.dtype)
        if has_last:
            # Segment(0, duration) "loose" crop of the frame grid.
            valid_frames = max(
                1,
                min(
                    aggregated.shape[0],
                    math.floor((waveform.numel() / config.sampling_rate - 0.0) / frame_step_s) + 1,
                ),
            )
            aggregated = aggregated[:valid_frames]
        rounded_hop = max(1, round(frame_hop))
        return PyanNetFrameOutput(
            scores=aggregated,
            frame_hop_samples=rounded_hop,
            frame_length_samples=rounded_hop,
            frame_start_samples=0,
            valid_samples=waveform.numel(),
            frame_step_s=frame_step_s,
            frame_duration_s=frame_step_s,
        )


__all__ = ["PyanNetFrameInference", "PyanNetFrameOutput"]
