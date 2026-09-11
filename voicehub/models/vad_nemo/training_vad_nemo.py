"""Raw-audio and aligned-frame fine-tuning for native MarbleNet VAD."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.training.adapters import FrameClassificationTrainingAdapter
from voicehub.training.datasets import SpeechDataset


class MarbleNetVADTrainingDataset(SpeechDataset):
    """Validated records with raw audio and frame or segment targets."""

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
                raise TypeError(f"MarbleNet VAD record {index} must be a mapping.")
            value = dict(record)
            if not any(name in value for name in ("audio", "audio_path", "input_values")):
                raise ValueError(f"MarbleNet VAD record {index} requires raw audio.")
            if not any(name in value for name in ("labels", "frame_labels", "segments")):
                raise ValueError(f"MarbleNet VAD record {index} requires frame or segment targets.")
            if "audio" not in value and "audio_path" in value:
                value["audio"] = value["audio_path"]
            if "labels" not in value and "frame_labels" in value:
                value["labels"] = value["frame_labels"]
            normalized.append(value)
        super().__init__(normalized, transform=transform)


def _is_numeric_sequence(value: Any) -> bool:
    return (
        isinstance(value, (tuple, list)) and bool(value) and
        all(isinstance(item, Real) and not isinstance(item, bool) for item in value))


def _audio_batch(value: Any) -> list[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            return [value]
        if value.ndim == 2:
            return [row for row in value]
        raise ValueError("MarbleNet audio must have shape [samples] or [batch, samples].")
    if _is_numeric_sequence(value):
        return [value]
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("MarbleNet audio batches cannot be empty.")
        return list(value)
    return [value]


def _batch_item(
    value: Any,
    *,
    index: int,
    batch_size: int,
    annotations: bool = False,
) -> Any:
    import torch

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim > 1 and value.shape[0] == batch_size:
            return value[index]
        return value
    if isinstance(value, (tuple, list)):
        if batch_size > 1 and len(value) == batch_size:
            return value[index]
        if (annotations and batch_size == 1 and len(value) == 1 and isinstance(value[0],
                                                                               (tuple, list, Mapping))):
            return value[0]
    return value


def _annotation_bounds(value: Any, *, sample_rate: int) -> tuple[int, int]:
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
    if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, Real) or
            not isinstance(end, Real) or not isfinite(float(start)) or not isfinite(float(end)) or
            float(start) < 0 or float(end) <= float(start)):
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
    frame_hop: int,
    threshold: float,
):
    import torch
    from torch.nn import functional

    if isinstance(segments, Mapping) or (isinstance(segments, Sequence) and
                                         not isinstance(segments, (str, bytes)) and len(segments) == 2 and
                                         all(isinstance(item, Real) and not isinstance(item, bool)
                                             for item in segments)):
        segments = [segments]
    if (isinstance(segments, (str, bytes)) or not isinstance(segments, Iterable)):
        raise TypeError("`segments` must be an iterable of speech intervals.")
    sample_labels = torch.zeros(sample_count, dtype=torch.float32)
    for segment in segments:
        start, end = _annotation_bounds(segment, sample_rate=sample_rate)
        start = min(sample_count, start)
        end = min(sample_count, end)
        if end > start:
            sample_labels[start:end] = 1
    padded_count = frame_count * frame_hop
    if sample_labels.numel() < padded_count:
        sample_labels = functional.pad(
            sample_labels,
            (0, padded_count - sample_labels.numel()),
        )
    coverage = sample_labels[:padded_count].reshape(frame_count, frame_hop).mean(-1)
    return (coverage >= threshold).to(dtype=torch.long)


def _explicit_labels(value: Any, *, frame_count: int):
    import torch

    try:
        labels = torch.as_tensor(value).reshape(-1)
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError("MarbleNet frame labels must be tensor-like.") from error
    if labels.numel() == 1 and frame_count > 1:
        labels = labels.expand(frame_count).clone()
    if labels.numel() != frame_count:
        raise ValueError(f"MarbleNet expected {frame_count} frame labels, "
                         f"found {labels.numel()}.")
    if labels.is_floating_point():
        if not torch.isfinite(labels).all() or torch.any((labels < 0) | (labels > 1)):
            raise ValueError("Floating VAD labels must be finite and in [0, 1].")
    elif torch.any((labels < 0) | (labels > 1)):
        raise ValueError("Class VAD labels must contain only 0 or 1.")
    return labels


def _uniform(
    minimum: float,
    maximum: float,
    *,
    device: Any,
) -> Any:
    import torch

    return minimum + (maximum - minimum) * torch.rand((), device=device)


def _augment_waveform(
    waveform: Any,
    *,
    wrapper: Any,
    noise_source: Any | None,
    noise_sampling_rate: Any | None,
) -> Any:
    """Apply the released white-noise, gain, and optional noise-mix recipe."""
    import torch

    values = waveform.clone()
    config = wrapper.config
    if torch.rand((), device=values.device) < config.training_white_noise_probability:
        minimum = round(config.training_white_noise_min_db)
        maximum = round(config.training_white_noise_max_db)
        if minimum == maximum:
            level = torch.tensor(float(minimum), device=values.device)
        else:
            level = torch.randint(
                minimum,
                maximum,
                (),
                device=values.device,
            ).float()
        values = values + torch.randn_like(values) * torch.pow(
            values.new_tensor(10.0),
            level / 20.0,
        )
    if torch.rand((), device=values.device) < config.training_gain_probability:
        gain = _uniform(
            config.training_gain_min_db,
            config.training_gain_max_db,
            device=values.device,
        )
        values = values * torch.pow(values.new_tensor(10.0), gain / 20.0)
    should_mix_noise = False
    if noise_source is not None:
        should_mix_noise = bool(torch.rand((), device=values.device) < config.training_noise_probability)
    if should_mix_noise:
        from voicehub.processing.waveform import load_native_audio

        noise = load_native_audio(
            noise_source,
            sampling_rate=noise_sampling_rate,
            target_sampling_rate=wrapper.sample_rate,
        ).waveform.to(
            device=values.device, dtype=values.dtype)
        if noise.numel() == 0:
            raise ValueError("Noise augmentation audio cannot be empty.")
        if noise.numel() < values.numel():
            repeats = (values.numel() + noise.numel() - 1) // noise.numel()
            noise = noise.repeat(repeats)
        if noise.numel() > values.numel():
            maximum_start = noise.numel() - values.numel()
            start = int(torch.randint(
                0,
                maximum_start + 1,
                (),
                device=noise.device,
            ).item())
            noise = noise[start:start + values.numel()]
        clean_rms = values.square().mean().sqrt()
        noise_rms = noise.square().mean().sqrt()
        if clean_rms > 0 and noise_rms > 0:
            snr = _uniform(
                config.training_noise_min_snr_db,
                config.training_noise_max_snr_db,
                device=values.device,
            )
            scale = clean_rms / noise_rms * torch.pow(
                values.new_tensor(10.0),
                -snr / 20.0,
            )
            values = values + noise * scale
    return values


def prepare_marblenet_vad_training_batch(
    wrapper: Any,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Create padded waveforms, aligned labels, and a strict frame mask."""
    import torch
    from torch.nn import functional

    from voicehub.processing.waveform import load_native_audio

    if not isinstance(inputs, Mapping):
        raise TypeError("MarbleNet training inputs must be a mapping.")
    if wrapper.native_config is None:
        raise RuntimeError("MarbleNet must be loaded before preprocessing.")
    source_name = next(
        (name for name in ("waveforms", "input_values", "audio", "audio_path") if name in inputs),
        None,
    )
    if source_name is None:
        raise ValueError("MarbleNet training requires `audio`, `audio_path`, or `input_values`.")
    sources = _audio_batch(inputs[source_name])
    batch_size = len(sources)
    explicit_name = next(
        (name for name in ("labels", "frame_labels") if name in inputs),
        None,
    )
    explicit = None if explicit_name is None else inputs[explicit_name]
    segment_values = inputs.get("segments")
    if explicit is None and segment_values is None:
        raise ValueError("MarbleNet training requires aligned frame labels or speech segments.")

    waveforms = []
    labels = []
    frame_counts = []
    maximum_samples = round(wrapper.config.training_max_duration_s * wrapper.sample_rate)
    for index, source in enumerate(sources):
        sample_rate = _batch_item(
            inputs.get("sampling_rate", inputs.get("sample_rate")),
            index=index,
            batch_size=batch_size,
        )
        materialized = load_native_audio(
            source,
            sampling_rate=sample_rate,
            target_sampling_rate=wrapper.sample_rate,
        )
        waveform = materialized.waveform
        if waveform.numel() < wrapper.native_config.window_length:
            raise ValueError("MarbleNet training audio must be at least 25 ms.")
        if waveform.numel() > maximum_samples:
            raise ValueError(
                "MarbleNet training audio exceeds `training_max_duration_s`; "
                "crop it explicitly during data preparation.")
        noise_source = _batch_item(
            inputs.get("noise_audio", inputs.get("noise")),
            index=index,
            batch_size=batch_size,
        )
        noise_sampling_rate = _batch_item(
            inputs.get("noise_sampling_rate"),
            index=index,
            batch_size=batch_size,
        )
        waveform = _augment_waveform(
            waveform,
            wrapper=wrapper,
            noise_source=noise_source,
            noise_sampling_rate=noise_sampling_rate,
        )
        frontend_frames = waveform.numel() // wrapper.native_config.hop_length + 1
        frame_count = (frontend_frames + 1) // 2
        item_labels = (
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
                    annotations=True,
                ),
                sample_count=waveform.numel(),
                sample_rate=wrapper.sample_rate,
                frame_count=frame_count,
                frame_hop=wrapper.native_config.output_frame_hop_samples,
                threshold=wrapper.config.training_label_threshold,
            ))
        waveforms.append(waveform)
        labels.append(item_labels)
        frame_counts.append(frame_count)

    max_samples = max(value.numel() for value in waveforms)
    max_frames = max(frame_counts)
    waveform_batch = torch.stack(
        [functional.pad(value, (0, max_samples - value.numel())) for value in waveforms])
    floating = any(value.is_floating_point() for value in labels)
    label_dtype = torch.float32 if floating else torch.long
    label_batch = torch.zeros(
        batch_size,
        max_frames,
        dtype=label_dtype,
    )
    label_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    for index, (value, count) in enumerate(zip(labels, frame_counts)):
        label_batch[index, :count] = value.to(dtype=label_dtype)
        label_mask[index, :count] = True
    return {
        "waveforms": waveform_batch,
        "waveform_lengths": torch.tensor(
            [value.numel() for value in waveforms],
            dtype=torch.long,
        ),
        "labels": label_batch,
        "label_mask": label_mask,
    }


class NativeMarbleNetVADTrainingAdapter(FrameClassificationTrainingAdapter):
    """Published SGD/polynomial recipe on the exact native graph."""

    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-marblenet-vad-safetensors"

    def setup(self):
        super().setup()
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError("Native MarbleNet tuning must target the wrapper's exact model.")
        return self

    def create_dataset(self, records, **kwargs):
        return MarbleNetVADTrainingDataset(records, **kwargs)

    def create_optimizer(
        self,
        name: str,
        parameters: list[tuple[str, Any]],
        training_args: Any,
    ):
        import torch

        if name not in {"default", "model"}:
            raise ValueError("MarbleNet VAD declares only the `model` optimizer, "
                             f"found {name!r}.")
        values = [parameter for _, parameter in parameters if parameter.requires_grad]
        if not values:
            raise ValueError("MarbleNet VAD has no trainable parameters.")
        return torch.optim.SGD(
            values,
            lr=training_args.learning_rate,
            momentum=0.9,
            weight_decay=0.001,
        )

    def create_scheduler(
        self,
        name: str,
        optimizer: Any,
        num_training_steps: int,
        training_args: Any,
    ):
        import torch

        del training_args
        if name not in {"default", "model"}:
            raise ValueError("MarbleNet VAD declares only the `model` scheduler, "
                             f"found {name!r}.")
        if num_training_steps <= 0:
            raise ValueError("`num_training_steps` must be positive.")
        warmup_steps = int(0.05 * num_training_steps)
        hold_until = warmup_steps + int(0.15 * num_training_steps)
        initial_lr = float(optimizer.param_groups[0]["lr"])
        minimum_factor = min(1.0, 1e-8 / initial_lr)
        decay_steps = max(1, num_training_steps - max(warmup_steps, hold_until))

        def schedule(step: int) -> float:
            if warmup_steps > 0 and step <= warmup_steps:
                return (step + 1) / (warmup_steps + 1)
            if warmup_steps <= step < hold_until:
                return 1.0
            if step > num_training_steps:
                return minimum_factor
            progress = min(
                max(step - hold_until, 0),
                decay_steps,
            ) / decay_steps
            return minimum_factor + (1.0 - minimum_factor) * (1.0 - progress)**2

        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        from voicehub.models.vad_nemo.native.metadata import MARBLENET_VAD_REVISION, NEMO_SOURCE_REVISION

        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "architecture": "marblenet-vad",
            "checkpoint_format": "voicehub-marblenet-vad-v1",
            "objective": "frame-cross-entropy",
            "official_optimizer": "sgd",
            "official_learning_rate": 0.01,
            "official_momentum": 0.9,
            "official_weight_decay": 0.001,
            "official_scheduler": "polynomial-hold-decay",
            "official_warmup_ratio": 0.05,
            "official_hold_ratio": 0.15,
            "sample_rate": self.model.sample_rate,
            "upstream_source_revision": NEMO_SOURCE_REVISION,
            "published_artifact_revision": MARBLENET_VAD_REVISION,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "native_architecture_family":
            "marblenet-vad",
            "checkpoint_format":
            "voicehub-marblenet-vad-v1",
            "processor_runtime":
            "voicehub-native",
            "training_reference": (
                "NeMo 2.1.0rc0 multilingual Frame-VAD config: frame "
                "cross-entropy, SGD, PolynomialHoldDecay, waveform/noise "
                "and spectrogram augmentations."),
        })
        return manifest

    def on_training_phase_end(self, context, output):
        output.metadata.update({
            "native_architecture_family": "marblenet-vad",
            "native_objective": "frame-cross-entropy",
            "upstream_recipe_reproduced": True,
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


__all__ = [
    "MarbleNetVADTrainingDataset",
    "NativeMarbleNetVADTrainingAdapter",
    "prepare_marblenet_vad_training_batch",
]
