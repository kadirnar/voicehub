"""Native Silero VAD dataset preprocessing and fine-tuning recipe.

The defaults mirror the official v6.2.1 tuning code: fixed 8/16 kHz
frames, sequence-local recurrent state, a binary target selected by
greater-than-50% speech coverage, half-weighted non-speech loss,
decoder-only optimization, and frame-level binary cross entropy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.training.adapters import FrameClassificationTrainingAdapter
from voicehub.training.datasets import SpeechDataset


class SileroVADTrainingDataset(SpeechDataset):
    """Validated portable records for Silero's native frame recipe.

    VoiceHub accepts its common ``audio``/``segments`` names and the
    official tuner names ``audio_path``/``speech_ts``. Explicit
    ``frame_labels`` or ``labels`` can replace timestamp annotations.
    """

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
                raise TypeError(f"Silero VAD record {index} must be a mapping.")
            value = dict(record)
            if "audio" not in value and "input_values" not in value:
                if "audio_path" in value:
                    value["audio"] = value["audio_path"]
                else:
                    raise ValueError(
                        f"Silero VAD record {index} requires `audio`, "
                        "`input_values`, or official `audio_path`.")
            if not any(name in value for name in (
                    "frame_labels",
                    "labels",
                    "targets",
                    "segments",
                    "speech_ts",
            )):
                raise ValueError(
                    f"Silero VAD record {index} requires frame labels or "
                    "speech timestamp annotations.")
            if "segments" not in value and "speech_ts" in value:
                value["segments"] = value["speech_ts"]
            normalized.append(value)
        super().__init__(normalized, transform=transform)


def _is_numeric_sequence(value: Any) -> bool:
    if not isinstance(value, (tuple, list)) or not value:
        return False
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)


def _audio_batch(value: Any, *, frame_size: int) -> list[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        if value.ndim <= 1:
            return [value]
        if value.ndim == 2:
            return [row for row in value]
        if value.ndim == 3 and value.shape[-1] == frame_size:
            return [row.reshape(-1) for row in value]
        raise ValueError(
            "`input_values` must have shape [samples], [batch, samples], "
            "or [batch, frames, frame_size].")
    if _is_numeric_sequence(value):
        return [value]
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("A Silero VAD audio batch cannot be empty.")
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
        raise ValueError("Training `sampling_rate` values must be integers.")
    return value


def _frame_labels(
    value: Any,
    *,
    frame_count: int,
) -> Any:
    import torch

    if value is None:
        return None
    try:
        labels = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError("Silero frame labels must be a numeric tensor-like value.") from error
    if labels.numel() == 1 and frame_count != 1:
        labels = labels.expand(frame_count).clone()
    if labels.numel() != frame_count:
        raise ValueError(f"Silero training expected {frame_count} frame labels, found "
                         f"{labels.numel()}.")
    if not torch.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any():
        raise ValueError("Silero frame labels must be finite values in [0, 1].")
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
        raise ValueError("Speech annotations require finite non-negative `start` and a "
                         "larger `end`.")
    if unit in {"sample", "samples"}:
        return round(float(start)), round(float(end))
    if unit not in {"second", "seconds", "s"}:
        raise ValueError("Speech annotation `unit` must be 'seconds' or 'samples'.")
    return round(float(start) * sample_rate), round(float(end) * sample_rate)


def _labels_from_segments(
    annotations: Any,
    *,
    valid_samples: int,
    padded_samples: int,
    sample_rate: int,
    frame_size: int,
    threshold: float,
) -> Any:
    import torch

    if annotations is None:
        return None
    if isinstance(annotations, Mapping):
        annotations = [annotations]
    if isinstance(annotations, (str, bytes)) or not isinstance(
            annotations,
            Iterable,
    ):
        raise TypeError("`segments`/`speech_ts` must be an iterable.")
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
    return (coverage > threshold).to(dtype=torch.float32)


def prepare_silero_vad_training_batch(
    wrapper: Any,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert portable records into padded waveform/label/mask sequences."""
    import torch

    from voicehub.processing.waveform import load_native_audio

    if not isinstance(inputs, Mapping):
        raise TypeError("Silero VAD training inputs must be a mapping.")
    if wrapper.native_config is None:
        raise RuntimeError("Silero VAD must be loaded before preprocessing.")
    source_name = (
        "input_values" if "input_values" in inputs else
        "audio" if "audio" in inputs else "audio_path" if "audio_path" in inputs else None)
    if source_name is None:
        raise ValueError("Silero VAD training requires `audio`, `input_values`, or "
                         "`audio_path`.")
    frame_size = wrapper.native_config.frame_size
    sources = _audio_batch(
        inputs[source_name],
        frame_size=frame_size,
    )
    batch_size = len(sources)
    preprocessed = source_name == "input_values"
    explicit_values = next(
        (inputs[name] for name in ("frame_labels", "labels", "targets") if name in inputs),
        None,
    )
    annotation_values = inputs.get(
        "segments",
        inputs.get("speech_ts"),
    )
    if explicit_values is None and annotation_values is None:
        raise ValueError("Silero VAD training requires frame labels or speech timestamps.")

    maximum_frames = max(
        1,
        math.floor(wrapper.config.training_max_duration_s * wrapper.sample_rate / frame_size),
    )
    waveforms = []
    label_rows = []
    weight_rows = []
    for index, source in enumerate(sources):
        length_values = inputs.get(
            "input_lengths",
            inputs.get("audio_lengths"),
        )
        declared_length = _batch_item(
            length_values,
            index=index,
            batch_size=batch_size,
        )
        if declared_length is not None:
            item = getattr(declared_length, "item", None)
            if callable(item):
                declared_length = item()
            if (isinstance(declared_length, bool) or not isinstance(declared_length, int) or
                    declared_length <= 0):
                raise ValueError("Silero training input lengths must be positive integers.")
            try:
                source = source[:declared_length]
            except (TypeError, IndexError) as error:
                raise TypeError("A declared Silero input length requires sliceable audio.") from error
        rate = _sampling_rate(
            inputs,
            index=index,
            batch_size=batch_size,
            preprocessed=preprocessed,
            target_rate=wrapper.sample_rate,
        )
        if rate is None and not isinstance(source, (str, Path)):
            raise ValueError(
                "Tensor/array training audio requires an explicit "
                "`sampling_rate`; `input_values` are assumed preprocessed.")
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

        labels_value = _batch_item(
            explicit_values,
            index=index,
            batch_size=batch_size,
        )
        labels = _frame_labels(
            labels_value,
            frame_count=frame_count,
        )
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
        if labels is None:  # pragma: no cover - guarded above
            raise RuntimeError("Silero labels could not be constructed.")

        weights = torch.where(
            labels > 0,
            torch.ones_like(labels),
            torch.full_like(labels, wrapper.config.training_noise_loss),
        )
        supplied_weights = inputs.get(
            "loss_mask",
            inputs.get("weights"),
        )
        supplied_weights = _batch_item(
            supplied_weights,
            index=index,
            batch_size=batch_size,
        )
        if supplied_weights is not None:
            supplied_weights = _frame_labels(
                supplied_weights,
                frame_count=frame_count,
            )
            weights = weights * supplied_weights

        waveforms.append(waveform)
        label_rows.append(labels)
        weight_rows.append(weights)

    input_values = torch.nn.utils.rnn.pad_sequence(
        waveforms,
        batch_first=True,
        padding_value=0.0,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        label_rows,
        batch_first=True,
        padding_value=0.0,
    )
    loss_mask = torch.nn.utils.rnn.pad_sequence(
        weight_rows,
        batch_first=True,
        padding_value=0.0,
    )
    return {
        "input_values": input_values.to(dtype=torch.float32),
        "labels": labels.to(dtype=torch.float32),
        "loss_mask": loss_mask.to(dtype=torch.float32),
    }


class NativeSileroVADTrainingAdapter(FrameClassificationTrainingAdapter):
    """Official decoder tuning recipe on the VoiceHub-owned Silero graph."""

    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-silero-vad-safetensors"

    def setup(self) -> NativeSileroVADTrainingAdapter:
        super().setup()
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError("Native Silero tuning must target the wrapper's exact model.")
        self.primary_model.set_encoder_trainable(self.model.config.training_train_encoder)
        self.primary_model.stft_conv.weight.requires_grad_(False)
        return self

    def create_dataset(self, records, **kwargs):
        return SileroVADTrainingDataset(records, **kwargs)

    def compute_phase_objective(
        self,
        predictions,
        labels,
        context,
    ):
        from voicehub.models.vad_silero.native.objective import silero_vad_binary_cross_entropy

        self._require_predictions_and_labels(predictions, labels)
        return silero_vad_binary_cross_entropy(
            predictions,
            labels,
            weights=context.inputs.get("loss_mask"),
            from_logits=True,
        )

    def create_optimizer(
        self,
        name: str,
        parameters: list[tuple[str, Any]],
        training_args: Any,
    ) -> Any:
        import torch

        if name != "model":
            raise ValueError(f"Silero VAD declares only the 'model' optimizer, found {name!r}.")
        values = [parameter for _, parameter in parameters if parameter.requires_grad]
        if not values:
            raise ValueError("Silero VAD has no trainable parameters.")
        return torch.optim.Adam(
            values,
            lr=training_args.learning_rate,
            betas=(
                training_args.adam_beta1,
                training_args.adam_beta2,
            ),
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
        )

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format": "voicehub-native-silero-vad-v1",
            "objective": "frame-binary-cross-entropy",
            "official_learning_rate": 5e-4,
            "official_optimizer": "adam",
            "sample_rate": self.model.sample_rate,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "native_architecture_family":
            "silero-vad",
            "checkpoint_format":
            "voicehub-native-silero-vad-v1",
            "processor_runtime":
            "voicehub-native",
            "training_reference":
            ("silero-vad v6.2.1 tuning recipe at "
             "7e30209a3e901f9842f81b225f3e93d8199902b1"),
        })
        return manifest

    def on_training_phase_end(self, context, output):
        output.metadata.update({
            "native_architecture_family": "silero-vad",
            "native_objective": "frame-binary-cross-entropy",
            "encoder_trainable": (self.model.config.training_train_encoder),
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


__all__ = [
    "NativeSileroVADTrainingAdapter",
    "SileroVADTrainingDataset",
    "prepare_silero_vad_training_batch",
]
