"""Portable fine-tuning for VoiceHub-native TEN and Silero VAD.

TEN does not publish the training recipe used for its released graph.
VoiceHub therefore keeps the released graph and reviewed Sherpa frontend
exact, while declaring this window-level binary-cross-entropy recipe as
a reconstruction.  This module must not be presented as source-recipe
parity.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.training.adapters import FrameClassificationTrainingAdapter
from voicehub.training.datasets import SpeechDataset


class TENVADTrainingDataset(SpeechDataset):
    """Validated audio records with frame labels or speech intervals."""

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
                raise TypeError(f"TEN VAD record {index} must be a mapping.")
            value = dict(record)
            if not any(name in value for name in ("audio", "audio_path", "waveforms", "input_values")):
                raise ValueError(f"TEN VAD record {index} requires raw audio.")
            if not any(name in value for name in ("labels", "frame_labels", "targets", "segments")):
                raise ValueError(f"TEN VAD record {index} requires frame labels or segments.")
            if "audio" not in value and "audio_path" in value:
                value["audio"] = value["audio_path"]
            if "labels" not in value:
                for name in ("frame_labels", "targets"):
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
            return list(value.unbind(0))
        raise ValueError("TEN VAD audio must have shape [samples] or [batch, samples].")
    if _is_numeric_sequence(value):
        return [value]
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("TEN VAD audio batches cannot be empty.")
        return list(value)
    return [value]


def _batch_item(
    value: Any,
    *,
    index: int,
    batch_size: int,
    nested_annotations: bool = False,
) -> Any:
    import torch

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        if batch_size > 1 and value.shape[0] == batch_size:
            return value[index]
        return value
    if isinstance(value, (tuple, list)):
        if batch_size > 1 and len(value) == batch_size:
            return value[index]
        if (nested_annotations and batch_size == 1 and len(value) == 1 and isinstance(value[0],
                                                                                      (tuple, list))):
            return value[0]
    return value


def _sampling_rate(
    inputs: Mapping[str, Any],
    *,
    index: int,
    batch_size: int,
    preprocessed: bool,
    target_rate: int,
) -> int | None:
    value = inputs.get("sampling_rate", inputs.get("sample_rate"))
    value = _batch_item(value, index=index, batch_size=batch_size)
    if value is None:
        return target_rate if preprocessed else None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("TEN training sample rates must be integers.")
    return value


def _declared_length(
    inputs: Mapping[str, Any],
    *,
    index: int,
    batch_size: int,
) -> int | None:
    value = inputs.get(
        "waveform_lengths",
        inputs.get("input_lengths", inputs.get("audio_lengths")),
    )
    value = _batch_item(value, index=index, batch_size=batch_size)
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("TEN training waveform lengths must be positive integers.")
    return value


def _frame_labels(value: Any, *, frame_count: int):
    import torch

    if value is None:
        return None
    try:
        labels = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError("TEN VAD frame labels must be numeric tensor-like values.") from error
    if labels.numel() == 1 and frame_count > 1:
        labels = labels.expand(frame_count).clone()
    if labels.numel() != frame_count:
        raise ValueError(f"TEN VAD expected {frame_count} frame labels, "
                         f"found {labels.numel()}.")
    if not torch.isfinite(labels).all() or torch.any((labels < 0) | (labels > 1)):
        raise ValueError("TEN VAD frame labels must be finite and in [0, 1].")
    return labels


def _annotation_bounds(
    annotation: Any,
    *,
    sample_rate: int,
) -> tuple[int, int]:
    if isinstance(annotation, Mapping):
        if "start_sample" in annotation or "end_sample" in annotation:
            start = annotation.get("start_sample")
            end = annotation.get("end_sample")
            unit = "samples"
        else:
            start = annotation.get("start")
            end = annotation.get("end")
            unit = annotation.get("unit", "seconds")
    elif (isinstance(annotation, Sequence) and not isinstance(annotation, (str, bytes)) and
          len(annotation) == 2):
        start, end = annotation
        unit = "seconds"
    else:
        start = getattr(annotation, "start", None)
        end = getattr(annotation, "end", None)
        unit = "seconds"
    if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or
            not isinstance(end, (int, float)) or not math.isfinite(float(start)) or
            not math.isfinite(float(end)) or float(start) < 0 or float(end) <= float(start)):
        raise ValueError("Speech intervals require finite values satisfying 0 <= start < end.")
    if unit in {"sample", "samples"}:
        return round(float(start)), round(float(end))
    if unit not in {"second", "seconds", "s"}:
        raise ValueError("Speech interval units must be seconds or samples.")
    return (
        round(float(start) * sample_rate),
        round(float(end) * sample_rate),
    )


def _labels_from_segments(
    annotations: Any,
    *,
    valid_samples: int,
    padded_samples: int,
    sample_rate: int,
    frame_size: int,
    threshold: float,
):
    import torch

    if isinstance(annotations, Mapping):
        annotations = [annotations]
    elif (isinstance(annotations, Sequence) and not isinstance(annotations,
                                                               (str, bytes)) and len(annotations) == 2 and
          all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in annotations)):
        annotations = [annotations]
    if (isinstance(annotations, (str, bytes)) or not isinstance(annotations, Iterable)):
        raise TypeError("TEN VAD `segments` must be an iterable.")
    speech = torch.zeros(padded_samples, dtype=torch.float32)
    for annotation in annotations:
        start, end = _annotation_bounds(
            annotation,
            sample_rate=sample_rate,
        )
        start = min(start, valid_samples)
        end = min(end, valid_samples)
        if end > start:
            speech[start:end] = 1.0
    coverage = speech.reshape(-1, frame_size).mean(dim=1)
    return (coverage >= threshold).to(dtype=torch.float32)


def prepare_ten_vad_training_batch(
    wrapper: Any,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build padded raw-audio sequences and aligned window-level targets."""
    import torch

    from voicehub.processing.waveform import load_native_audio

    if not isinstance(inputs, Mapping):
        raise TypeError("TEN VAD training inputs must be a mapping.")
    if wrapper.native_config is None:
        raise RuntimeError("TEN VAD must be loaded before preprocessing.")
    source_name = next(
        (name for name in ("waveforms", "input_values", "audio", "audio_path") if name in inputs),
        None,
    )
    if source_name is None:
        raise ValueError("TEN VAD training requires `audio`, `waveforms`, or `input_values`.")
    sources = _audio_batch(inputs[source_name])
    batch_size = len(sources)
    preprocessed = source_name in {"waveforms", "input_values"}
    explicit_values = next(
        (inputs[name] for name in ("labels", "frame_labels", "targets") if name in inputs),
        None,
    )
    annotation_values = inputs.get("segments")
    if explicit_values is None and annotation_values is None:
        raise ValueError("TEN VAD training requires frame labels or speech intervals.")

    frame_size = wrapper.native_config.window_size
    maximum_frames = max(
        1,
        math.floor(wrapper.config.training_max_duration_s * wrapper.sample_rate / frame_size),
    )
    waveforms = []
    lengths = []
    label_rows = []
    mask_rows = []
    supplied_masks = inputs.get("label_mask", inputs.get("loss_mask"))

    for index, source in enumerate(sources):
        declared = _declared_length(
            inputs,
            index=index,
            batch_size=batch_size,
        )
        if declared is not None:
            try:
                source = source[:declared]
            except (TypeError, IndexError) as error:
                raise TypeError("A declared TEN waveform length requires sliceable audio.") from error
        rate = _sampling_rate(
            inputs,
            index=index,
            batch_size=batch_size,
            preprocessed=preprocessed,
            target_rate=wrapper.sample_rate,
        )
        if rate is None and not isinstance(source, (str, Path)):
            raise ValueError(
                "Tensor/array TEN training audio requires `sampling_rate`; "
                "`waveforms` and `input_values` are assumed preprocessed.")
        materialized = load_native_audio(
            source,
            sampling_rate=rate,
            target_sampling_rate=wrapper.sample_rate,
        )
        waveform = materialized.waveform[:maximum_frames * frame_size]
        valid_samples = waveform.numel()
        frame_count = math.ceil(valid_samples / frame_size)
        padded_samples = frame_count * frame_size
        if padded_samples > valid_samples:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, padded_samples - valid_samples),
            )

        label_value = _batch_item(
            explicit_values,
            index=index,
            batch_size=batch_size,
        )
        labels = _frame_labels(label_value, frame_count=frame_count)
        if labels is None:
            annotations = _batch_item(
                annotation_values,
                index=index,
                batch_size=batch_size,
                nested_annotations=True,
            )
            labels = _labels_from_segments(
                annotations,
                valid_samples=valid_samples,
                padded_samples=padded_samples,
                sample_rate=wrapper.sample_rate,
                frame_size=frame_size,
                threshold=wrapper.config.training_label_threshold,
            )

        mask = torch.ones(frame_count, dtype=torch.bool)
        supplied = _batch_item(
            supplied_masks,
            index=index,
            batch_size=batch_size,
        )
        if supplied is not None:
            supplied = _frame_labels(supplied, frame_count=frame_count)
            mask &= supplied > 0
        if not mask.any():
            raise ValueError("Each TEN VAD training item must retain at least one target frame.")

        waveforms.append(waveform)
        lengths.append(valid_samples)
        label_rows.append(labels)
        mask_rows.append(mask)

    waveform_batch = torch.nn.utils.rnn.pad_sequence(
        waveforms,
        batch_first=True,
        padding_value=0.0,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        label_rows,
        batch_first=True,
        padding_value=0.0,
    )
    label_mask = torch.nn.utils.rnn.pad_sequence(
        mask_rows,
        batch_first=True,
        padding_value=False,
    )
    return {
        "waveforms": waveform_batch.to(dtype=torch.float32),
        "waveform_lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": labels.to(dtype=torch.float32),
        "label_mask": label_mask.to(dtype=torch.bool),
        "positive_weight": wrapper.config.training_positive_weight,
    }


class NativeTENVADTrainingAdapter(FrameClassificationTrainingAdapter):
    """Reconstructed masked-BCE tuning for the exact released TEN graph."""

    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-ten-vad-safetensors"

    def _validate_configured_training_artifact(self) -> None:
        identifier = str(getattr(getattr(self.model, "config", None), "name_or_path", "")).lower()
        if identifier.endswith(".onnx"):
            if not getattr(self.model, "_trust_onnx_checkpoint", False):
                raise ValueError(
                    "TEN ONNX fine-tuning requires reviewed one-time conversion; "
                    "pass `trust_onnx_checkpoint=True`.")
            return
        super()._validate_configured_training_artifact()

    def setup(self) -> NativeTENVADTrainingAdapter:
        super().setup()
        if getattr(self.model.config, "model_family", None) != "ten":
            raise ValueError("Native TEN training requires `model_family='ten'`.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError("Native TEN tuning must target the wrapper's exact graph.")
        return self

    def create_dataset(self, records, **kwargs):
        return TENVADTrainingDataset(records, **kwargs)

    def compute_phase_objective(
        self,
        predictions,
        labels,
        context,
    ):
        from voicehub.models.vad_ten.native.objective import ten_vad_binary_cross_entropy

        self._require_predictions_and_labels(predictions, labels)
        return ten_vad_binary_cross_entropy(
            predictions,
            labels,
            mask=context.inputs.get("label_mask"),
            positive_weight=context.inputs.get("positive_weight"),
        )

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format": "voicehub-native-ten-vad-v1",
            "objective": "masked-window-binary-cross-entropy",
            "sample_rate": self.model.sample_rate,
            "source_recipe_published": False,
            "recipe_status": "voicehub-reconstructed",
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "native_architecture_family":
            "ten-vad",
            "checkpoint_format":
            "voicehub-native-ten-vad-v1",
            "processor_runtime":
            "voicehub-native",
            "training_reference":
            ("VoiceHub reconstructed masked window BCE; TEN's source "
             "training recipe is unpublished."),
            "source_recipe_parity":
            False,
        })
        return manifest

    def on_training_phase_end(self, context, output):
        output.metadata.update({
            "native_architecture_family": "ten-vad",
            "native_objective": "masked-window-binary-cross-entropy",
            "source_recipe_published": False,
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


def create_sherpa_native_vad_training_adapter(model, spec):
    """Dispatch the compatibility provider to its native family recipe."""
    if getattr(getattr(model, "config", None), "model_family", None) == "silero":
        from dataclasses import replace

        from voicehub.models.vad_silero.training_vad_silero import NativeSileroVADTrainingAdapter

        phase = replace(
            spec.phases[0],
            forward_method="frame_probabilities",
            required_inputs=("input_values", "labels"),
        )
        return NativeSileroVADTrainingAdapter(
            model,
            replace(spec, phases=(phase, )),
        )
    return NativeTENVADTrainingAdapter(model, spec)


__all__ = [
    "NativeTENVADTrainingAdapter",
    "TENVADTrainingDataset",
    "create_sherpa_native_vad_training_adapter",
    "prepare_ten_vad_training_batch",
]
