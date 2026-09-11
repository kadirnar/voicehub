"""Torch-only sliding inference and overlap-add for PyanNet."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from voicehub.models.vad_pyannote.native.modeling import PyanNet


@dataclass(frozen=True, slots=True)
class PyanNetFrameOutput:
    """Aggregated full-recording frame output."""

    scores: Tensor
    frame_hop_samples: int
    frame_length_samples: int
    frame_start_samples: int
    valid_samples: int

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


def _closest_frame(sample: int, *, frame_hop: float) -> int:
    """Mirror pyannote.core's nearest frame-center rule."""
    return max(0, round((sample - 0.5 * frame_hop) / frame_hop))


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
        starts_in_frames = tuple(_closest_frame(start, frame_hop=frame_hop) for start in starts)
        required_frames = _closest_frame(
            starts[-1] + chunk_samples,
            frame_hop=frame_hop,
        ) + 1
        final_stop = starts_in_frames[-1] + frames_per_chunk
        if final_stop > required_frames:
            raise RuntimeError("PyanNet frame geometry produced an invalid aggregation "
                               "extent.")
        summed = torch.zeros(
            required_frames,
            output_size,
            dtype=scores.dtype,
            device=scores.device,
        )
        weights = torch.zeros_like(summed)
        window = torch.hamming_window(
            frames_per_chunk,
            periodic=False,
            dtype=scores.dtype,
            device=scores.device,
        ).unsqueeze(-1)
        for index, start in enumerate(starts_in_frames):
            stop = start + frames_per_chunk
            summed[start:stop] += scores[index] * window
            weights[start:stop] += window
        aggregated = summed / weights.clamp_min(torch.finfo(weights.dtype).eps)
        if has_last:
            valid_frames = max(
                1,
                min(
                    aggregated.shape[0],
                    math.floor(waveform.numel() / frame_hop) + 1,
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
        )


__all__ = ["PyanNetFrameInference", "PyanNetFrameOutput"]
