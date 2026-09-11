"""Raw-audio and aligned-frame fine-tuning for native FSMN VAD."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.training.adapters import FrameClassificationTrainingAdapter
from voicehub.training.datasets import SpeechDataset


class FSMNVADTrainingDataset(SpeechDataset):
    """Validated records with raw audio and speech/PDF frame targets."""

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
                raise TypeError(f"FSMN VAD record {index} must be a mapping.")
            value = dict(record)
            if not any(name in value for name in ("audio", "audio_path", "input_values")):
                raise ValueError(f"FSMN VAD record {index} requires raw audio.")
            if not any(name in value for name in ("labels", "frame_labels", "pdf_labels", "segments")):
                raise ValueError(f"FSMN VAD record {index} requires frame or segment targets.")
            if "audio" not in value and "audio_path" in value:
                value["audio"] = value["audio_path"]
            if "labels" not in value:
                for name in ("frame_labels", "pdf_labels"):
                    if name in value:
                        value["labels"] = value[name]
                        break
            normalized.append(value)
        super().__init__(normalized, transform=transform)


def _is_numeric_sequence(value: Any) -> bool:
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
        raise ValueError("FSMN VAD audio must have shape [samples] or [batch, samples].")
    if _is_numeric_sequence(value):
        return [value]
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("FSMN VAD audio batches cannot be empty.")
        return list(value)
    return [value]


def _batch_item(
    value: Any,
    *,
    index: int,
    batch_size: int,
) -> Any:
    import torch

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim > 1 and value.shape[0] == batch_size:
            return value[index]
        if value.ndim == 1 and batch_size > 1 and value.shape[0] == batch_size:
            return value[index]
        return value
    if isinstance(value, (tuple, list)):
        if batch_size > 1 and len(value) == batch_size:
            return value[index]
    return value


def _annotation_bounds(
    value: Any,
    *,
    sample_rate: int,
) -> tuple[int, int]:
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
        raise ValueError("Speech annotations require finite 0 <= start < end.")
    if unit in {"sample", "samples"}:
        return round(float(start)), round(float(end))
    if unit not in {"second", "seconds", "s"}:
        raise ValueError("Speech annotation units must be seconds or samples.")
    return (
        round(float(start) * sample_rate),
        round(float(end) * sample_rate),
    )


def _labels_from_segments(
    segments: Any,
    *,
    sample_count: int,
    sample_rate: int,
    frame_count: int,
    frame_length: int,
    frame_shift: int,
) -> Any:
    import torch

    if isinstance(segments, Mapping):
        segments = [segments]
    elif (isinstance(segments, Sequence) and not isinstance(segments, (str, bytes)) and len(segments) == 2 and
          all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in segments)):
        segments = [segments]
    elif (isinstance(segments, Sequence) and not isinstance(segments, (str, bytes)) and len(segments) == 1 and
          isinstance(segments[0], Sequence) and not isinstance(segments[0], (str, bytes, Mapping)) and
          not _is_numeric_sequence(segments[0])):
        segments = segments[0]
    if (isinstance(segments, (str, bytes)) or not isinstance(segments, Iterable)):
        raise TypeError("`segments` must be an iterable of speech intervals.")
    sample_labels = torch.zeros(sample_count, dtype=torch.float32)
    for segment in segments:
        start, end = _annotation_bounds(
            segment,
            sample_rate=sample_rate,
        )
        start = min(sample_count, start)
        end = min(sample_count, end)
        if end > start:
            sample_labels[start:end] = 1.0
    if frame_count == 0:
        return torch.empty(0, dtype=torch.float32)
    coverage = sample_labels.unfold(
        0,
        frame_length,
        frame_shift,
    )[:frame_count].mean(dim=-1)
    return (coverage >= 0.5).to(dtype=torch.float32)


def _explicit_labels(
    value: Any,
    *,
    frame_count: int,
) -> Any:
    import torch

    try:
        labels = torch.as_tensor(value).reshape(-1)
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError("FSMN VAD frame labels must be tensor-like.") from error
    if labels.numel() == 1 and frame_count > 1:
        labels = labels.expand(frame_count).clone()
    if labels.numel() != frame_count:
        raise ValueError(f"FSMN VAD expected {frame_count} labels, "
                         f"found {labels.numel()}.")
    if labels.is_floating_point() and not torch.isfinite(labels).all():
        raise ValueError("FSMN VAD frame labels must be finite.")
    return labels


def prepare_fsmn_vad_training_batch(
    wrapper: Any,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Create padded waveforms, aligned targets, and an explicit frame mask."""
    import torch
    from torch.nn import functional

    from voicehub.processing.waveform import load_native_audio

    if not isinstance(inputs, Mapping):
        raise TypeError("FSMN VAD training inputs must be a mapping.")
    if wrapper.native_config is None:
        raise RuntimeError("FSMN VAD must be loaded before preprocessing.")
    source_name = next(
        (name for name in ("input_values", "audio", "audio_path") if name in inputs),
        None,
    )
    if source_name is None:
        raise ValueError("FSMN VAD training requires `audio`, `audio_path`, or "
                         "`input_values`.")
    sources = _audio_batch(inputs[source_name])
    batch_size = len(sources)
    explicit_name = next(
        (name for name in ("labels", "frame_labels", "pdf_labels") if name in inputs),
        None,
    )
    explicit = (None if explicit_name is None else inputs[explicit_name])
    segment_values = inputs.get("segments")
    if explicit is None and segment_values is None:
        raise ValueError("FSMN VAD training requires aligned labels or speech segments.")

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
        if rate is None and source_name != "input_values":
            rate = None
        elif rate is None:
            rate = wrapper.sample_rate
        if hasattr(rate, "item"):
            rate = rate.item()
        materialized = load_native_audio(
            source,
            sampling_rate=rate,
            target_sampling_rate=wrapper.sample_rate,
        )
        waveform = materialized.waveform[:maximum_samples]
        if waveform.numel() < wrapper.native_config.frame_length_samples:
            raise ValueError("Every FSMN VAD training sample must contain at least 25 ms "
                             "of audio.")
        frame_count = wrapper.model.frame_count(waveform.numel())
        target = (
            _explicit_labels(
                _batch_item(
                    explicit,
                    index=index,
                    batch_size=batch_size,
                ),
                frame_count=frame_count,
            ) if explicit is not None else _labels_from_segments(
                _batch_item(
                    segment_values,
                    index=index,
                    batch_size=batch_size,
                ),
                sample_count=waveform.numel(),
                sample_rate=wrapper.sample_rate,
                frame_count=frame_count,
                frame_length=wrapper.native_config.frame_length_samples,
                frame_shift=wrapper.native_config.frame_shift_samples,
            ))
        waveforms.append(waveform)
        labels.append(target)
        frame_counts.append(frame_count)

    maximum_waveform = max(item.numel() for item in waveforms)
    maximum_frames = wrapper.model.frame_count(maximum_waveform)
    waveform_batch = torch.stack(
        [functional.pad(item, (0, maximum_waveform - item.numel())) for item in waveforms])
    use_pdf_labels = (
        explicit_name == "pdf_labels" or any((
            not item.is_floating_point() and item.dtype != torch.bool and bool((
                (item < 0) | (item > 1)).any())) for item in labels))
    label_dtype = torch.long if use_pdf_labels else torch.float32
    label_batch = torch.zeros(
        batch_size,
        maximum_frames,
        dtype=label_dtype,
    )
    mask = torch.zeros(batch_size, maximum_frames, dtype=torch.bool)
    for index, (target, count) in enumerate(zip(labels, frame_counts)):
        label_batch[index, :count] = target.to(dtype=label_dtype)
        mask[index, :count] = True
    return {
        "waveforms": waveform_batch,
        "waveform_lengths": torch.tensor(
            [item.numel() for item in waveforms],
            dtype=torch.long,
        ),
        "labels": label_batch,
        "label_mask": mask,
        "target_kind": "pdf" if use_pdf_labels else "binary",
    }


class NativeFSMNVADTrainingAdapter(FrameClassificationTrainingAdapter):
    """Use the model-owned objective and portable Safetensors export."""

    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-fsmn-vad-safetensors"

    def setup(self):
        super().setup()
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError("Native FSMN VAD tuning must target the wrapper's exact model.")
        return self

    def create_dataset(self, records, **kwargs):
        return FSMNVADTrainingDataset(records, **kwargs)

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        from voicehub.models.vad_funasr.native.metadata import FUNASR_HF_REVISION, FUNASR_SOURCE_REVISION

        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "architecture": "fsmn-vad",
            "checkpoint_format": "voicehub-fsmn-vad-v1",
            "objectives": (
                "grouped-binary-nll",
                "pdf-cross-entropy",
            ),
            "sample_rate": self.model.sample_rate,
            "upstream_source_revision": FUNASR_SOURCE_REVISION,
            "published_artifact_revision": FUNASR_HF_REVISION,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "native_architecture_family":
            "fsmn-vad",
            "checkpoint_format":
            "voicehub-fsmn-vad-v1",
            "processor_runtime":
            "voicehub-native",
            "training_reference": (
                "VoiceHub checkpoint-compatible PDF cross-entropy and "
                "grouped binary VAD objective; the upstream private "
                "training recipe is unpublished."),
        })
        return manifest

    def on_training_phase_end(self, context, output):
        output.metadata.update({
            "native_architecture_family": "fsmn-vad",
            "native_objective": getattr(output, "objective", None) or "model-owned",
            "upstream_recipe_reproduced": False,
        })
        return output

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        export = getattr(self.model, "export_native_pretrained", None)
        if not callable(export):
            raise TypeError("Native FSMN VAD training requires "
                            "export_native_pretrained().")
        export(Path(save_directory))


__all__ = [
    "FSMNVADTrainingDataset",
    "NativeFSMNVADTrainingAdapter",
    "prepare_fsmn_vad_training_batch",
]
