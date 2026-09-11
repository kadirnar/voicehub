"""VoiceHub-native speech trimming utilities for the original Vui runtime.

The upstream Vui release delegated trimming to Pyannote and WhisperX.
This module preserves its hysteresis and chunk-merging semantics with
small native data structures, and accepts any VoiceHub VAD provider as
an injected detector. The default is a deterministic short-term-energy
detector so Vui does not download or import a second framework during
synthesis.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

VAD_SEGMENTATION_URL = (
    "https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/"
    "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/"
    "pytorch_model.bin")
VAD_SEGMENTATION_SHA256 = ("0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea")


@dataclass(frozen=True, order=True, slots=True)
class Segment:
    """Half-open time region measured in seconds."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if (not math.isfinite(self.start) or not math.isfinite(self.end) or self.start < 0.0 or
                self.end <= self.start):
            raise ValueError("A VAD segment requires finite 0 <= start < end.")

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def middle(self) -> float:
        return 0.5 * (self.start + self.end)


@dataclass(frozen=True, slots=True)
class SlidingWindow:
    """Frame geometry compatible with the subset used by Vui."""

    start: float = 0.0
    duration: float = 0.02
    step: float = 0.02

    def __post_init__(self) -> None:
        if (not math.isfinite(self.start) or not math.isfinite(self.duration) or
                not math.isfinite(self.step) or self.duration <= 0.0 or self.step <= 0.0):
            raise ValueError("Sliding-window geometry must be finite and positive.")

    def __getitem__(self, index: int) -> Segment:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise IndexError("Sliding-window indices must be non-negative integers.")
        start = self.start + index * self.step
        return Segment(start, start + self.duration)


@dataclass(frozen=True, slots=True)
class SlidingWindowFeature:
    """Frame scores and timing metadata used by :class:`Binarize`."""

    data: Tensor
    sliding_window: SlidingWindow
    labels: tuple[Any, ...] | None = None

    def __init__(
        self,
        data: Any,
        sliding_window: SlidingWindow,
        labels: Sequence[Any] | None = None,
    ) -> None:
        values = torch.as_tensor(data).detach().float().cpu()
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError("VAD frame scores must have shape [frames, classes].")
        if not torch.isfinite(values).all():
            raise ValueError("VAD frame scores cannot contain NaN or infinity.")
        if not isinstance(sliding_window, SlidingWindow):
            raise TypeError("`sliding_window` must be a SlidingWindow.")
        normalized_labels = None if labels is None else tuple(labels)
        if normalized_labels is not None and len(normalized_labels) != values.shape[1]:
            raise ValueError("VAD labels must match the score class dimension.")
        object.__setattr__(self, "data", values)
        object.__setattr__(self, "sliding_window", sliding_window)
        object.__setattr__(self, "labels", normalized_labels)


class Timeline(tuple):
    """Immutable collection of segments with Pyannote-style ``support``."""

    def __new__(cls, segments: Iterable[Segment] = ()) -> Timeline:
        return super().__new__(cls, sorted(segments))

    def support(self, collar: float = 0.0) -> Timeline:
        if not math.isfinite(collar) or collar < 0.0:
            raise ValueError("`collar` must be finite and non-negative.")
        merged: list[Segment] = []
        for segment in self:
            if merged and segment.start - merged[-1].end <= collar:
                merged[-1] = Segment(
                    merged[-1].start,
                    max(merged[-1].end, segment.end),
                )
            else:
                merged.append(segment)
        return Timeline(merged)


class Annotation:
    """Minimal tracked-segment container used by the Vui helper surface."""

    def __init__(self) -> None:
        self._entries: dict[tuple[Segment, Any], Any] = {}

    def __setitem__(self, key: tuple[Segment, Any], label: Any) -> None:
        segment, track = key
        if not isinstance(segment, Segment):
            raise TypeError("Annotation keys must start with a Segment.")
        self._entries[(segment, track)] = label

    def __delitem__(self, key: tuple[Segment, Any]) -> None:
        del self._entries[key]

    def itertracks(self):
        yield from sorted(self._entries)

    def support(self, collar: float = 0.0) -> Annotation:
        output = Annotation()
        for index, segment in enumerate(self.get_timeline().support(collar)):
            output[segment, index] = 1
        return output

    def get_timeline(self) -> Timeline:
        return Timeline(segment for segment, _ in self._entries)

    def for_json(self) -> dict[str, Any]:
        return {
            "content": [{
                "segment": {
                    "start": segment.start,
                    "end": segment.end,
                },
                "track": track,
                "label": label,
            } for (segment, track), label in sorted(self._entries.items())],
        }


def _segments_from_result(value: Any) -> list[tuple[float, float]]:
    if hasattr(value, "segments"):
        value = value.segments
    if hasattr(value, "get_timeline"):
        value = value.get_timeline().support()
    segments: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, Mapping):
            start, end = item.get("start"), item.get("end")
        elif hasattr(item, "start") and hasattr(item, "end"):
            start, end = item.start, item.end
        else:
            try:
                start, end = item[:2]
            except (TypeError, ValueError):
                raise TypeError("VAD results must contain start/end regions.") from None
        segment = Segment(float(start), float(end))
        segments.append((segment.start, segment.end))
    return segments


class _EnergyPipeline:
    """Callable native fallback with the provider shape expected by Vui."""

    def __init__(
        self,
        *,
        onset: float = 0.8,
        offset: float = 0.5,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
    ) -> None:
        self.instantiate({
            "onset": onset,
            "offset": offset,
            "min_duration_on": min_duration_on,
            "min_duration_off": min_duration_off,
        })

    def instantiate(self, parameters: Mapping[str, float]) -> _EnergyPipeline:
        self.onset = float(parameters.get("onset", 0.8))
        self.offset = float(parameters.get("offset", self.onset))
        self.min_duration_on = float(parameters.get("min_duration_on", 0.0))
        self.min_duration_off = float(parameters.get("min_duration_off", 0.0))
        return self

    def to(self, _device: Any) -> _EnergyPipeline:
        return self

    def __call__(self, value: Any) -> Annotation:
        from voicehub.models.vad_auditok.native.modeling import EnergyVoiceActivityDetector

        if isinstance(value, Mapping):
            waveform = value["waveform"]
            sample_rate = int(value.get("sample_rate", 16_000))
        else:
            waveform = value
            sample_rate = 16_000
        waveform = torch.as_tensor(waveform).detach().float().reshape(-1).cpu()
        detection = EnergyVoiceActivityDetector().detect(
            waveform,
            sampling_rate=sample_rate,
            energy_threshold_db=-42.0,
            threshold_method="fixed",
            analysis_window_s=0.032,
            minimum_energy_threshold_db=-60.0,
            min_speech_duration_ms=max(0, round(self.min_duration_on * 1_000)),
            min_silence_duration_ms=max(0, round(self.min_duration_off * 1_000)),
            speech_pad_ms=0,
            max_speech_duration_s=None,
            strict_min_duration=False,
        )
        annotation = Annotation()
        for index, region in enumerate(detection.regions):
            annotation[
                Segment(
                    region.start_sample / sample_rate,
                    region.end_sample / sample_rate,
                ),
                index,
            ] = 1
        return annotation


pipeline: Any | None = None
pipeline_name = "voicehub/native-energy-vad"


@torch.autocast("cuda", enabled=False)
def detect_voice_activity(waveform: Any, pipe: Any | None = None):
    """Detect speech regions in a 16 kHz waveform.

    ``pipe`` may be a VoiceHub VAD provider, a Pyannote-compatible
    callable, or any callable returning start/end regions.
    """
    global pipeline
    values = torch.as_tensor(waveform).detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("VAD waveform cannot be empty.")
    if pipe is not None:
        pipeline = pipe
    elif pipeline is None:
        pipeline = _EnergyPipeline()

    if hasattr(pipeline, "detect"):
        result = pipeline.detect(values, sampling_rate=16_000)
    elif callable(pipeline):
        try:
            result = pipeline({
                "waveform": values.unsqueeze(0),
                "sample_rate": 16_000,
            })
        except (TypeError, KeyError):
            result = pipeline(values)
    else:
        raise TypeError("The configured VAD pipeline is not callable.")
    return _segments_from_result(result)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified_checkpoint(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".incomplete",
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(
            VAD_SEGMENTATION_URL,
            headers={"User-Agent": "voicehub"},
        )
        with urllib.request.urlopen(request, timeout=30.0) as source, os.fdopen(
                descriptor,
                "wb",
        ) as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != VAD_SEGMENTATION_SHA256:
            raise RuntimeError("Downloaded Vui VAD checkpoint failed SHA-256 verification.")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


class _VoiceHubVADPipeline:

    def __init__(
        self,
        model: Any,
        *,
        onset: float,
        offset: float,
    ) -> None:
        self.model = model
        self.onset = onset
        self.offset = offset

    def __call__(self, value: Any) -> Annotation:
        if isinstance(value, Mapping):
            waveform = value["waveform"]
            sample_rate = int(value.get("sample_rate", 16_000))
        else:
            waveform = value
            sample_rate = 16_000
        output = self.model.detect(
            waveform,
            sampling_rate=sample_rate,
            onset=self.onset,
            offset=self.offset,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
        )
        annotation = Annotation()
        for index, segment in enumerate(output.segments):
            annotation[Segment(segment.start, segment.end), index] = 1
        return annotation


def load_vad_model(
    device: str,
    vad_onset: float = 0.500,
    vad_offset: float = 0.363,
    use_auth_token: str | None = None,
    model_fp: str | os.PathLike[str] | None = None,
    batch_size: int = 32,
):
    """Convert and load WhisperX's pinned PyanNet artifact natively.

    The published file is a Lightning pickle. Its immutable URL and
    SHA-256 are verified before VoiceHub performs its restricted one-
    time conversion to Safetensors; runtime inference never executes
    pickle or Pyannote code.
    """
    del use_auth_token
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("`batch_size` must be a positive integer.")
    if model_fp is None:
        model_fp = Path(torch.hub.get_dir()) / "whisperx-vad-segmentation.bin"
    checkpoint = Path(model_fp).expanduser()
    if checkpoint.exists() and not checkpoint.is_file():
        raise RuntimeError(f"{checkpoint} exists and is not a regular file.")
    if not checkpoint.is_file():
        _download_verified_checkpoint(checkpoint)
    elif _sha256_file(checkpoint) != VAD_SEGMENTATION_SHA256:
        raise RuntimeError("Cached Vui VAD checkpoint failed SHA-256 verification.")

    from voicehub.models.vad_pyannote import PyannoteVADForVoiceActivityDetection
    from voicehub.models.vad_pyannote.native.checkpoint import convert_pyannote_lightning_checkpoint

    destination = checkpoint.parent / ".voicehub-native" / "vui-pyannet"
    if not (destination / "model.safetensors").is_file():
        convert_pyannote_lightning_checkpoint(
            checkpoint,
            destination,
            variant="segmentation",
            trust_pickle_checkpoint=True,
            expected_sha256=VAD_SEGMENTATION_SHA256,
        )
    model = PyannoteVADForVoiceActivityDetection(
        model_path=destination,
        device=device,
        lazy_load=False,
        batch_size=batch_size,
    )
    return _VoiceHubVADPipeline(
        model,
        onset=float(vad_onset),
        offset=float(vad_offset),
    )


class Binarize:
    """Convert frame scores into segments using source hysteresis semantics."""

    def __init__(
        self,
        onset: float = 0.5,
        offset: float | None = None,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        pad_onset: float = 0.0,
        pad_offset: float = 0.0,
        max_duration: float = math.inf,
    ) -> None:
        self.onset = float(onset)
        self.offset = self.onset if offset is None else float(offset)
        self.pad_onset = float(pad_onset)
        self.pad_offset = float(pad_offset)
        self.min_duration_on = float(min_duration_on)
        self.min_duration_off = float(min_duration_off)
        self.max_duration = float(max_duration)

    def __call__(self, scores: SlidingWindowFeature | Any) -> Annotation:
        values = torch.as_tensor(scores.data).detach().float().cpu()
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        frames = scores.sliding_window
        timestamps = [frames[index].middle for index in range(values.shape[0])]
        labels = getattr(scores, "labels", None)
        active = Annotation()
        for class_index, class_scores in enumerate(values.transpose(0, 1)):
            label = class_index if labels is None else labels[class_index]
            start = timestamps[0]
            is_active = bool(class_scores[0] > self.onset)
            region_scores = [float(class_scores[0])]
            region_times = [start]
            timestamp = start
            for timestamp, score in zip(
                    timestamps[1:],
                    class_scores[1:].tolist(),
                    strict=True,
            ):
                if is_active:
                    if timestamp - start > self.max_duration:
                        search_after = len(region_scores) // 2
                        tail = torch.tensor(region_scores[search_after:])
                        split = search_after + int(tail.argmin().item())
                        split_time = region_times[split]
                        active[
                            Segment(
                                max(0.0, start - self.pad_onset),
                                split_time + self.pad_offset,
                            ),
                            class_index,
                        ] = label
                        start = split_time
                        region_scores = region_scores[split + 1:]
                        region_times = region_times[split + 1:]
                    elif score < self.offset:
                        active[
                            Segment(
                                max(0.0, start - self.pad_onset),
                                timestamp + self.pad_offset,
                            ),
                            class_index,
                        ] = label
                        start = timestamp
                        is_active = False
                        region_scores = []
                        region_times = []
                    region_scores.append(score)
                    region_times.append(timestamp)
                elif score > self.onset:
                    start = timestamp
                    is_active = True
                    region_scores = [score]
                    region_times = [timestamp]
            if is_active and timestamp + self.pad_offset > start - self.pad_onset:
                active[
                    Segment(
                        max(0.0, start - self.pad_onset),
                        timestamp + self.pad_offset,
                    ),
                    class_index,
                ] = label

        if self.pad_offset or self.pad_onset or self.min_duration_off:
            active = active.support(collar=self.min_duration_off)
        if self.min_duration_on > 0:
            for segment, track in list(active.itertracks()):
                if segment.duration < self.min_duration_on:
                    del active[segment, track]
        return active


class VoiceActivitySegmentation:
    """Small compatibility wrapper around a native segmentation callable."""

    CACHED_SEGMENTATION = "cache/segmentation"

    def __init__(self, segmentation: Any, **_kwargs: Any) -> None:
        if not callable(segmentation):
            raise TypeError("`segmentation` must be callable.")
        self._segmentation = segmentation
        self.training = False
        self.parameters: dict[str, float] = {}

    def instantiate(self, parameters: Mapping[str, float]) -> None:
        self.parameters = dict(parameters)

    def to(self, _device: Any) -> VoiceActivitySegmentation:
        return self

    def apply(self, file: Any, hook: Callable | None = None) -> Any:
        del hook
        return self._segmentation(file)

    __call__ = apply


def merge_vad(
    vad_arr: Iterable[Sequence[float]],
    pad_onset: float = 0.0,
    pad_offset: float = 0.0,
    min_duration_off: float = 0.0,
    min_duration_on: float = 0.0,
) -> tuple[dict[str, float], ...]:
    """Pad, merge, and filter raw VAD regions without Pandas."""
    annotation = Annotation()
    for index, values in enumerate(vad_arr):
        start, end = float(values[0]), float(values[1])
        annotation[
            Segment(max(0.0, start - pad_onset), end + pad_offset),
            index,
        ] = 1
    annotation = annotation.support(collar=min_duration_off)
    return tuple({
        "start": segment.start,
        "end": segment.end,
    } for segment in annotation.get_timeline() if segment.duration >= min_duration_on)


def merge_chunks(
    segments: SlidingWindowFeature | Any,
    chunk_size: float,
    onset: float = 0.5,
    offset: float | None = None,
) -> list[dict[str, float]]:
    """Merge hysteresis regions into chunks no longer than ``chunk_size``."""
    if not math.isfinite(chunk_size) or chunk_size <= 0.0:
        raise ValueError("`chunk_size` must be finite and positive.")
    regions = list(Binarize(
        max_duration=chunk_size,
        onset=onset,
        offset=offset,
    )(segments).get_timeline())
    if not regions:
        return []
    merged: list[dict[str, float]] = []
    current_start = regions[0].start
    current_end = regions[0].end
    for region in regions[1:]:
        if region.end - current_start > chunk_size and current_end > current_start:
            merged.append({
                "start": current_start,
                "end": current_end,
            })
            current_start = region.start
        current_end = region.end
    merged.append({
        "start": current_start,
        "end": current_end,
    })
    return merged


__all__ = [
    "Annotation",
    "Binarize",
    "Segment",
    "SlidingWindow",
    "SlidingWindowFeature",
    "VAD_SEGMENTATION_SHA256",
    "VAD_SEGMENTATION_URL",
    "VoiceActivitySegmentation",
    "detect_voice_activity",
    "load_vad_model",
    "merge_chunks",
    "merge_vad",
]
