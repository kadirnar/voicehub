"""Source-compatible frame inference and segmentation for CRDNN VAD."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from voicehub.models.vad_speechbrain.native.configuration import SpeechBrainCRDNNVADConfig


@dataclass(frozen=True, slots=True)
class SpeechBrainVADBoundary:
    """One unpadded source-compatible speech interval."""

    start: float
    end: float
    score: float | None = None


def _positive_chunk_size(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"`{name}` must be positive.")
    return value


class SpeechBrainVADInference:
    """Run the published two-level chunk pipeline without SpeechBrain."""

    def __init__(
        self,
        model,
        *,
        large_chunk_size: float = 30.0,
        small_chunk_size: float = 10.0,
        overlap_small_chunk: bool = False,
    ) -> None:
        self.model = model
        self.config = SpeechBrainCRDNNVADConfig.coerce(model.config)
        self.large_chunk_size = _positive_chunk_size(
            "large_chunk_size",
            large_chunk_size,
        )
        self.small_chunk_size = _positive_chunk_size(
            "small_chunk_size",
            small_chunk_size,
        )
        ratio = self.large_chunk_size / self.small_chunk_size
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("`large_chunk_size / small_chunk_size` must be an integer.")
        if not isinstance(overlap_small_chunk, bool):
            raise TypeError("`overlap_small_chunk` must be a boolean.")
        self.overlap_small_chunk = overlap_small_chunk

    def _chunk_probabilities(self, chunks: Tensor) -> Tensor:
        output = self.model(chunks)
        probabilities = output.speech_probabilities
        if probabilities.shape[1] < 2:
            raise RuntimeError("CRDNN VAD produced too few frames for chunk inference.")
        return probabilities[:, :-1]

    def __call__(self, waveform: Tensor) -> Tensor:
        if not isinstance(waveform, Tensor) or waveform.ndim != 1:
            raise ValueError("SpeechBrain VAD inference expects one [samples] waveform.")
        if waveform.numel() < self.config.hop_length:
            raise ValueError("SpeechBrain VAD requires at least 10 ms of audio.")
        sample_count = waveform.numel()
        large_samples = int(self.config.sampling_rate * self.large_chunk_size)
        small_samples = int(self.config.sampling_rate * self.small_chunk_size)
        step_seconds = (self.small_chunk_size / 2.0 if self.overlap_small_chunk else self.small_chunk_size)
        step_samples = int(self.config.sampling_rate * step_seconds)
        small_frames = int(self.small_chunk_size / self.config.time_resolution)
        step_frames = int(step_seconds / self.config.time_resolution)
        chunks_output = []
        begin = 0
        while True:
            final = begin + large_samples >= sample_count
            large = waveform[begin:begin + large_samples]
            if final or large.numel() < small_samples:
                large = torch.cat((large, large.new_zeros(small_samples)))
            chunks = large.unfold(0, small_samples, step_samples)
            probabilities = self._chunk_probabilities(chunks)
            if probabilities.shape[1] != small_frames:
                raise RuntimeError(
                    "CRDNN VAD frame geometry does not match the configured "
                    f"10 ms resolution ({probabilities.shape[1]} != {small_frames}).")
            if self.overlap_small_chunk:
                window = torch.hamming_window(
                    small_frames,
                    device=probabilities.device,
                    dtype=probabilities.dtype,
                )
                midpoint = small_frames // 2
                probabilities = probabilities.clone()
                probabilities[0, midpoint:] *= window[midpoint:]
                probabilities[-1, :midpoint] *= window[:midpoint]
                if probabilities.shape[0] > 2:
                    probabilities[1:-1] *= window.unsqueeze(0)
            output_frames = int(large.numel() / self.config.hop_length)
            folded = probabilities.new_zeros(output_frames)
            for index, values in enumerate(probabilities):
                frame_start = index * step_frames
                folded[frame_start:frame_start + small_frames] += values
            chunks_output.append(folded)
            if final:
                break
            begin += large_samples
        result = torch.cat(chunks_output)
        return result[:int(sample_count / self.config.hop_length)]

    @staticmethod
    def threshold(
        probabilities: Tensor,
        *,
        activation_threshold: float,
        deactivation_threshold: float,
    ) -> Tensor:
        if probabilities.ndim != 1:
            raise ValueError("Frame probabilities must be one-dimensional.")
        for name, value in (
            ("activation_threshold", activation_threshold),
            ("deactivation_threshold", deactivation_threshold),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a real number.")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"`{name}` must be in [0, 1].")
        if deactivation_threshold > activation_threshold:
            raise ValueError("The deactivation threshold cannot exceed activation.")
        active = probabilities >= float(activation_threshold)
        does_not_deactivate = probabilities >= float(deactivation_threshold)
        active = active.clone()
        for index in range(1, active.numel()):
            active[index] = (active[index] | active[index - 1]) & does_not_deactivate[index]
        return active

    def boundaries(
        self,
        decisions: Tensor,
        *,
        probabilities: Tensor | None = None,
    ) -> tuple[SpeechBrainVADBoundary, ...]:
        if decisions.ndim != 1:
            raise ValueError("VAD decisions must be one-dimensional.")
        if decisions.numel() == 0:
            return ()
        current = decisions.to(dtype=torch.int64)
        shifted = torch.roll(current, shifts=1)
        shifted[0] = 0
        changes = current + shifted
        changes[0] = int(changes[0] >= 1)
        changes[-1] = int(changes[-1] >= 1)
        indexes = torch.nonzero(changes == 1, as_tuple=False).flatten()
        if indexes.numel() % 2:
            indexes = torch.cat((indexes, indexes.new_tensor([decisions.numel()])), )
        intervals = indexes.reshape(-1, 2)
        intervals[:, 1] -= 1
        output = []
        for start_index, end_index in intervals.tolist():
            if end_index <= start_index:
                continue
            score = None
            if probabilities is not None:
                score = float(probabilities[start_index:end_index + 1].mean())
            output.append(
                SpeechBrainVADBoundary(
                    start=start_index * self.config.time_resolution,
                    end=end_index * self.config.time_resolution,
                    score=score,
                ))
        return tuple(output)

    @staticmethod
    def merge_close(
        boundaries: tuple[SpeechBrainVADBoundary, ...],
        *,
        maximum_gap: float,
    ) -> tuple[SpeechBrainVADBoundary, ...]:
        if not boundaries:
            return ()
        merged = []
        current = boundaries[0]
        for boundary in boundaries[1:]:
            if boundary.start - current.end <= maximum_gap:
                scores = [score for score in (current.score, boundary.score) if score is not None]
                current = SpeechBrainVADBoundary(
                    current.start,
                    boundary.end,
                    None if not scores else sum(scores) / len(scores),
                )
            else:
                merged.append(current)
                current = boundary
        merged.append(current)
        return tuple(merged)

    @staticmethod
    def remove_short(
        boundaries: tuple[SpeechBrainVADBoundary, ...],
        *,
        minimum_duration: float,
    ) -> tuple[SpeechBrainVADBoundary, ...]:
        # The author implementation uses strict `>` rather than `>=`.
        return tuple(boundary for boundary in boundaries if boundary.end - boundary.start > minimum_duration)

    def double_check(
        self,
        waveform: Tensor,
        boundaries: tuple[SpeechBrainVADBoundary, ...],
        *,
        threshold: float,
    ) -> tuple[SpeechBrainVADBoundary, ...]:
        retained = []
        for boundary in boundaries:
            start = int(boundary.start * self.config.sampling_rate)
            end = int(boundary.end * self.config.sampling_rate)
            segment = waveform[start:end]
            if segment.numel() < self.config.hop_length:
                continue
            probability = self.model(segment.unsqueeze(0), ).speech_probabilities.mean()
            if probability > threshold:
                retained.append(SpeechBrainVADBoundary(
                    boundary.start,
                    boundary.end,
                    float(probability),
                ))
        return tuple(retained)

    def energy_refine(
        self,
        waveform: Tensor,
        boundaries: tuple[SpeechBrainVADBoundary, ...],
        *,
        activation_threshold: float = 0.5,
        deactivation_threshold: float = 0.0,
    ) -> tuple[SpeechBrainVADBoundary, ...]:
        refined = []
        chunk_size = self.config.hop_length
        for boundary in boundaries:
            start = int(boundary.start * self.config.sampling_rate)
            end = int(boundary.end * self.config.sampling_rate)
            segment = waveform[start:end]
            frame_count = segment.numel() // chunk_size
            if frame_count < 2:
                continue
            frames = segment[:frame_count * chunk_size].reshape(frame_count, chunk_size)
            energy = (frames.abs().sum(dim=-1) + 1e-6).log()
            std = energy.std(correction=1)
            if not torch.isfinite(std) or std <= 0:
                continue
            normalized = (energy - energy.mean()) / (2.0 * std) + 0.5
            decisions = self.threshold(
                normalized,
                activation_threshold=activation_threshold,
                deactivation_threshold=deactivation_threshold,
            )
            for local in self.boundaries(decisions, probabilities=normalized):
                refined.append(
                    SpeechBrainVADBoundary(
                        boundary.start + local.start,
                        boundary.start + local.end,
                        local.score,
                    ))
        return tuple(refined)

    def segment(
        self,
        waveform: Tensor,
        *,
        activation_threshold: float = 0.5,
        deactivation_threshold: float = 0.25,
        minimum_speech_duration: float = 0.25,
        maximum_silence_duration: float = 0.25,
        apply_energy_vad: bool = False,
        double_check: bool = True,
        speech_threshold: float = 0.5,
    ) -> tuple[Tensor, tuple[SpeechBrainVADBoundary, ...]]:
        probabilities = self(waveform)
        decisions = self.threshold(
            probabilities,
            activation_threshold=activation_threshold,
            deactivation_threshold=deactivation_threshold,
        )
        boundaries = self.boundaries(
            decisions,
            probabilities=probabilities,
        )
        if apply_energy_vad:
            boundaries = self.energy_refine(waveform, boundaries)
        boundaries = self.merge_close(
            boundaries,
            maximum_gap=maximum_silence_duration,
        )
        boundaries = self.remove_short(
            boundaries,
            minimum_duration=minimum_speech_duration,
        )
        if double_check:
            boundaries = self.double_check(
                waveform,
                boundaries,
                threshold=speech_threshold,
            )
        return probabilities, boundaries


__all__ = [
    "SpeechBrainVADBoundary",
    "SpeechBrainVADInference",
]
