"""Raw-audio fine-tuning for VoiceHub-native SpeechBrain CRDNN VAD."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.training.adapters import FrameClassificationTrainingAdapter
from voicehub.training.datasets import SpeechDataset


class SpeechBrainVADTrainingDataset(SpeechDataset):
    """Validated records with raw audio and frame or interval targets."""

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        transform=None,
    ) -> None:
        if isinstance(records, (str, bytes, Mapping)):
            raise TypeError("`records` must be an iterable of mappings.")
        normalized = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"SpeechBrain VAD record {index} must be a mapping.")
            value = dict(record)
            if not any(name in value for name in ("audio", "audio_path", "input_values")):
                raise ValueError(f"SpeechBrain VAD record {index} requires raw audio.")
            if not any(name in value for name in ("labels", "frame_labels", "segments", "speech")):
                raise ValueError(f"SpeechBrain VAD record {index} requires frame or interval targets.")
            if "audio" not in value and "audio_path" in value:
                value["audio"] = value["audio_path"]
            if "labels" not in value and "frame_labels" in value:
                value["labels"] = value["frame_labels"]
            if "segments" not in value and "speech" in value:
                value["segments"] = value["speech"]
            normalized.append(value)
        super().__init__(normalized, transform=transform)


def _numeric_sequence(value: Any) -> bool:
    return (
        isinstance(value, (tuple, list)) and bool(value) and
        all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value))


def _audio_batch(value: Any) -> list[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            return [value]
        if value.ndim == 2:
            return [row for row in value]
        raise ValueError("SpeechBrain VAD audio must have shape [samples] or [batch, samples].")
    if _numeric_sequence(value):
        return [value]
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("SpeechBrain VAD audio batches cannot be empty.")
        return list(value)
    return [value]


def _batch_item(value: Any, *, index: int, batch_size: int) -> Any:
    import torch

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim > 1 and value.shape[0] == batch_size:
            return value[index]
        return value
    if isinstance(value, (tuple, list)) and batch_size > 1 and len(value) == batch_size:
        return value[index]
    return value


def _annotation_bounds(value: Any, *, sample_rate: int) -> tuple[float, float]:
    if isinstance(value, Mapping):
        if "start_sample" in value or "end_sample" in value:
            start = value.get("start_sample")
            end = value.get("end_sample")
            unit = "samples"
        else:
            start = value.get("start")
            end = value.get("end")
            unit = value.get("unit", "seconds")
    elif (isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2):
        start, end = value
        unit = "seconds"
    else:
        start = getattr(value, "start", None)
        end = getattr(value, "end", None)
        unit = "seconds"
    if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or
            not isinstance(end, (int, float)) or not math.isfinite(float(start)) or
            not math.isfinite(float(end)) or float(start) < 0.0 or float(end) <= float(start)):
        raise ValueError("Speech intervals require finite 0 <= start < end.")
    if unit in {"sample", "samples"}:
        return float(start) / sample_rate, float(end) / sample_rate
    if unit not in {"second", "seconds", "s"}:
        raise ValueError("Speech interval units must be seconds or samples.")
    return float(start), float(end)


def _labels_from_segments(
    segments: Any,
    *,
    sample_rate: int,
    frame_count: int,
    time_resolution: float,
):
    import torch

    if isinstance(segments, Mapping) or (isinstance(segments, Sequence) and
                                         not isinstance(segments, (str, bytes)) and len(segments) == 2 and
                                         all(isinstance(item, (int, float)) and not isinstance(item, bool)
                                             for item in segments)):
        segments = [segments]
    if isinstance(segments, (str, bytes)) or not isinstance(segments, Iterable):
        raise TypeError("`segments` must be an iterable of speech intervals.")
    labels = torch.zeros(frame_count, dtype=torch.float32)
    for segment in segments:
        start, end = _annotation_bounds(segment, sample_rate=sample_rate)
        # Exact LibriParty recipe semantics: truncate start and end to frame
        # indexes, then use an exclusive end slice.
        start_frame = min(frame_count, int(start / time_resolution))
        end_frame = min(frame_count, int(end / time_resolution))
        if end_frame > start_frame:
            labels[start_frame:end_frame] = 1.0
    return labels


def _explicit_labels(value: Any, *, frame_count: int):
    import torch

    try:
        labels = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError("SpeechBrain VAD frame labels must be tensor-like.") from error
    if labels.numel() == 1 and frame_count > 1:
        labels = labels.expand(frame_count).clone()
    if labels.numel() != frame_count:
        raise ValueError(f"SpeechBrain VAD expected {frame_count} labels, found {labels.numel()}.")
    if not torch.isfinite(labels).all() or torch.any((labels < 0.0) | (labels > 1.0)):
        raise ValueError("SpeechBrain VAD frame labels must be finite and in [0, 1].")
    return labels


def prepare_speechbrain_vad_training_batch(
    wrapper: Any,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Create padded waveforms and author-aligned 10 ms targets."""
    import torch
    from torch.nn import functional

    from voicehub.processing.waveform import load_native_audio

    if not isinstance(inputs, Mapping):
        raise TypeError("SpeechBrain VAD training inputs must be a mapping.")
    if wrapper.native_config is None:
        raise RuntimeError("SpeechBrain VAD must be loaded before preprocessing.")
    source_name = next(
        (name for name in ("input_values", "audio", "audio_path") if name in inputs),
        None,
    )
    if source_name is None:
        raise ValueError("SpeechBrain VAD training requires `audio`, `audio_path`, or `input_values`.")
    sources = _audio_batch(inputs[source_name])
    batch_size = len(sources)
    explicit_name = next(
        (name for name in ("labels", "frame_labels") if name in inputs),
        None,
    )
    explicit = None if explicit_name is None else inputs[explicit_name]
    segments = inputs.get("segments", inputs.get("speech"))
    if explicit is None and segments is None:
        raise ValueError("SpeechBrain VAD training requires aligned labels or speech intervals.")

    maximum_samples = round(wrapper.config.training_max_duration_s * wrapper.sample_rate)
    waveforms = []
    labels = []
    frame_counts = []
    for index, source in enumerate(sources):
        rate = _batch_item(
            inputs.get("sampling_rate", inputs.get("sample_rate")),
            index=index,
            batch_size=batch_size,
        )
        if hasattr(rate, "item"):
            rate = rate.item()
        if rate is None and source_name == "input_values":
            rate = wrapper.sample_rate
        materialized = load_native_audio(
            source,
            sampling_rate=rate,
            target_sampling_rate=wrapper.sample_rate,
        )
        waveform = materialized.waveform[:maximum_samples]
        if waveform.numel() < wrapper.native_config.hop_length:
            raise ValueError("Every SpeechBrain VAD example must contain at least 10 ms.")
        # Author targets omit the final frame created by centered STFT.
        frame_count = waveform.numel() // wrapper.native_config.hop_length
        target = (
            _explicit_labels(
                _batch_item(explicit, index=index, batch_size=batch_size),
                frame_count=frame_count,
            ) if explicit is not None else _labels_from_segments(
                _batch_item(segments, index=index, batch_size=batch_size),
                sample_rate=wrapper.sample_rate,
                frame_count=frame_count,
                time_resolution=wrapper.native_config.time_resolution,
            ))
        waveforms.append(waveform)
        labels.append(target)
        frame_counts.append(frame_count)

    maximum_waveform = max(item.numel() for item in waveforms)
    maximum_frames = maximum_waveform // wrapper.native_config.hop_length
    waveform_batch = torch.stack(
        [functional.pad(item, (0, maximum_waveform - item.numel())) for item in waveforms])
    label_batch = torch.zeros(batch_size, maximum_frames, dtype=torch.float32)
    mask = torch.zeros(batch_size, maximum_frames, dtype=torch.bool)
    for index, (target, count) in enumerate(zip(labels, frame_counts)):
        label_batch[index, :count] = target
        mask[index, :count] = True
    return {
        "waveforms": waveform_batch,
        "waveform_lengths": torch.tensor(
            [item.numel() for item in waveforms],
            dtype=torch.long,
        ),
        "labels": label_batch,
        "label_mask": mask,
        "positive_weight": wrapper.config.training_positive_weight,
    }


class NativeSpeechBrainVADTrainingAdapter(FrameClassificationTrainingAdapter):
    """Use model-owned BCE and portable native Safetensors export."""

    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-speechbrain-crdnn-vad-safetensors"

    def setup(self):
        super().setup()
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError("Native SpeechBrain VAD tuning must target the wrapper's exact CRDNN.")
        return self

    def create_dataset(self, records, **kwargs):
        return SpeechBrainVADTrainingDataset(records, **kwargs)

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        from voicehub.models.vad_speechbrain.native.metadata import (
            SPEECHBRAIN_TRAINING_SOURCE_REVISION,
            SPEECHBRAIN_VAD_REVISION,
        )

        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "architecture": "speechbrain-crdnn-vad",
            "checkpoint_format": "voicehub-speechbrain-crdnn-vad-v1",
            "objective": "masked-binary-cross-entropy",
            "sample_rate": self.model.sample_rate,
            "upstream_training_source_revision": (SPEECHBRAIN_TRAINING_SOURCE_REVISION),
            "published_artifact_revision": SPEECHBRAIN_VAD_REVISION,
            "exact_upstream_crdnn_recipe_available": False,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "native_architecture_family":
            "speechbrain-crdnn-vad",
            "checkpoint_format":
            "voicehub-speechbrain-crdnn-vad-v1",
            "processor_runtime":
            "voicehub-native",
            "training_reference": (
                "Author-compatible 10 ms LibriParty targets and frame "
                "BCE on the exact published CRDNN graph. The pinned "
                "upstream recipe instantiates a different GRU-only graph."),
        })
        return manifest

    def on_training_phase_end(self, context, output):
        output.metadata.update({
            "native_architecture_family": "speechbrain-crdnn-vad",
            "native_objective": "masked-binary-cross-entropy",
            "upstream_crdnn_recipe_reproduced": False,
        })
        return output

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        export = getattr(self.model, "export_native_pretrained", None)
        if not callable(export):
            raise TypeError("Native SpeechBrain VAD training requires export_native_pretrained().")
        export(Path(save_directory))


__all__ = [
    "NativeSpeechBrainVADTrainingAdapter",
    "SpeechBrainVADTrainingDataset",
    "prepare_speechbrain_vad_training_batch",
]
