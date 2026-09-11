"""Raw-audio fine-tuning recipe for the native ESPnet Transformer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.training.adapters import SpeechSeq2SeqTrainingAdapter
from voicehub.training.datasets import ASRDataset, SpeechDataset

_RAW_FIELDS = frozenset({
    "audio",
    "audio_lengths",
    "audio_path",
    "sample_rate",
    "sampling_rate",
    "text",
    "transcript",
    "transcription",
})
_PREPARED_WAVEFORM_FIELDS = frozenset({
    "waveforms",
    "waveform_lengths",
    "labels",
    "label_lengths",
})
_PREPARED_FEATURE_FIELDS = frozenset({
    "features",
    "feature_lengths",
    "labels",
    "label_lengths",
})


class ESPnetASRTrainingDataset(SpeechDataset):
    """Validated audio/transcript records for native ESPnet fine-tuning."""

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
                raise TypeError(f"ESPnet ASR record {index} must be a mapping.")
            value = dict(record)
            if (_PREPARED_WAVEFORM_FIELDS <= set(value) or _PREPARED_FEATURE_FIELDS <= set(value)):
                normalized.append(value)
                continue
            if "audio" not in value and "audio_path" in value:
                value["audio"] = value["audio_path"]
            if "text" not in value:
                for alias in ("transcription", "transcript"):
                    if alias in value:
                        value["text"] = value[alias]
                        break
            if "audio" not in value and "features" not in value:
                raise ValueError(
                    f"ESPnet ASR record {index} requires `audio`, `audio_path`, "
                    "or precomputed `features`.")
            text = value.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"ESPnet ASR record {index} requires a non-empty transcript.")
            normalized.append(value)
        super().__init__(normalized, transform=transform)


def _is_numeric_waveform(value: Any) -> bool:
    return (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and bool(value) and
        all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value))


def _audio_rows(value: Any) -> tuple[Any, ...]:
    import torch

    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            return (value, )
        if value.ndim == 2:
            return tuple(value[index] for index in range(value.shape[0]))
        raise ValueError("ESPnet audio must have shape [samples] or [batch, samples].")
    if _is_numeric_waveform(value):
        return (value, )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError("ESPnet audio batches cannot be empty.")
        return tuple(value)
    return (value, )


def _batch_values(
    value: Any,
    *,
    batch_size: int,
    name: str,
) -> tuple[Any, ...]:
    import torch

    if value is None or isinstance(value, (str, bytes)):
        return (value, ) * batch_size
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return (value.item(), ) * batch_size
        if value.ndim != 1:
            raise ValueError(f"`{name}` must be scalar or one-dimensional.")
        rows = tuple(value.tolist())
    elif isinstance(value, Sequence):
        rows = tuple(value)
    else:
        return (value, ) * batch_size
    if len(rows) != batch_size:
        raise ValueError(f"`{name}` contains {len(rows)} values for batch size {batch_size}.")
    return rows


def _transcripts(inputs: Mapping[str, Any]) -> tuple[str, ...]:
    value = inputs.get(
        "text",
        inputs.get("transcription", inputs.get("transcript")),
    )
    if isinstance(value, str):
        rows = (value, )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = tuple(value)
    else:
        raise TypeError("ESPnet training requires `text`, `transcription`, or `transcript`.")
    if not rows or any(not isinstance(text, str) or not text.strip() for text in rows):
        raise ValueError("ESPnet transcripts must be non-empty strings.")
    return rows


def _prepare_labels(wrapper: Any, texts: tuple[str, ...]):
    import torch

    encoded = tuple(wrapper.tokenizer.encode_as_ids(text) for text in texts)
    if any(not row for row in encoded):
        raise ValueError("Every ESPnet transcript must produce at least one token.")
    lengths = torch.tensor(
        [len(row) for row in encoded],
        dtype=torch.long,
    )
    labels = torch.full(
        (len(encoded), int(lengths.max().item())),
        wrapper.native_config.ignore_token_id,
        dtype=torch.long,
    )
    for index, row in enumerate(encoded):
        labels[index, :len(row)] = torch.tensor(row, dtype=torch.long)
    return labels, lengths


def prepare_espnet_training_batch(
    wrapper: Any,
    inputs: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Convert raw audio/transcripts or features/text into model inputs."""
    import torch
    from torch.nn import functional

    from voicehub.processing.waveform import load_native_audio

    if not isinstance(inputs, Mapping):
        raise TypeError("ESPnet training inputs must be a mapping.")
    if not isinstance(phase, str) or not phase:
        raise ValueError("ESPnet training `phase` must be non-empty.")
    if wrapper.native_config is None or wrapper.tokenizer is None:
        raise RuntimeError("ESPnet must be loaded before preprocessing.")
    if (_PREPARED_WAVEFORM_FIELDS <= set(inputs) or _PREPARED_FEATURE_FIELDS <= set(inputs)):
        return dict(inputs)

    texts = _transcripts(inputs)
    labels, label_lengths = _prepare_labels(wrapper, texts)
    prepared: dict[str, Any] = {
        "labels": labels,
        "label_lengths": label_lengths,
    }
    if "features" in inputs:
        features = torch.as_tensor(
            inputs["features"],
            dtype=torch.float32,
        )
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if (features.ndim != 3 or features.shape[0] != len(texts) or
                features.shape[-1] != wrapper.native_config.n_mels):
            raise ValueError(
                "ESPnet features must have shape "
                f"[batch, frames, {wrapper.native_config.n_mels}].")
        raw_lengths = inputs.get("feature_lengths")
        feature_lengths = (
            torch.full(
                (features.shape[0], ),
                features.shape[1],
                dtype=torch.long,
            ) if raw_lengths is None else torch.as_tensor(raw_lengths, dtype=torch.long))
        if (feature_lengths.ndim != 1 or feature_lengths.shape[0] != features.shape[0] or
                torch.any(feature_lengths < wrapper.native_config.minimum_feature_frames) or
                torch.any(feature_lengths > features.shape[1])):
            raise ValueError(
                "ESPnet feature lengths must fit the padded batch and contain "
                f"at least {wrapper.native_config.minimum_feature_frames} "
                "frames for conv2d6.")
        prepared.update({
            "features": features,
            "feature_lengths": feature_lengths,
        })
    else:
        if "audio" in inputs:
            raw_audio = inputs["audio"]
        elif "audio_path" in inputs:
            raw_audio = inputs["audio_path"]
        else:
            raise ValueError("ESPnet training requires `audio`, `audio_path`, or `features`.")
        sources = _audio_rows(raw_audio)
        if len(sources) != len(texts):
            raise ValueError("ESPnet training requires one transcript per waveform.")
        raw_lengths = _batch_values(
            inputs.get("audio_lengths"),
            batch_size=len(sources),
            name="audio_lengths",
        )
        sample_rates = _batch_values(
            inputs.get("sampling_rate", inputs.get("sample_rate")),
            batch_size=len(sources),
            name="sampling_rate",
        )
        maximum_samples = round(wrapper.config.training_max_duration_s * wrapper.sample_rate)
        minimum_samples = wrapper.native_config.minimum_waveform_samples
        waveforms = []
        for index, (source, length, sampling_rate) in enumerate(zip(sources, raw_lengths, sample_rates)):
            if length is not None:
                if (isinstance(length, bool) or not isinstance(length, Integral) or length <= 0):
                    raise ValueError("`audio_lengths` must contain positive integers.")
            materialized = load_native_audio(
                source,
                sampling_rate=sampling_rate,
                target_sampling_rate=wrapper.sample_rate,
                num_samples=(None if length is None else int(length)),
            )
            waveform = materialized.waveform
            if waveform.numel() > maximum_samples:
                raise ValueError(
                    f"ESPnet example {index} is longer than "
                    f"{wrapper.config.training_max_duration_s:g} seconds. "
                    "Segment it with aligned text instead of truncating it.")
            if waveform.numel() < minimum_samples:
                waveform = functional.pad(
                    waveform,
                    (0, minimum_samples - waveform.numel()),
                )
            waveforms.append(waveform)
        waveform_lengths = torch.tensor(
            [waveform.numel() for waveform in waveforms],
            dtype=torch.long,
        )
        maximum = int(waveform_lengths.max().item())
        prepared.update({
            "waveforms":
            torch.stack(
                [functional.pad(
                    waveform,
                    (0, maximum - waveform.numel()),
                ) for waveform in waveforms]),
            "waveform_lengths":
            waveform_lengths,
        })
    for name, value in inputs.items():
        if name not in _RAW_FIELDS and name not in prepared:
            prepared[name] = value
    return prepared


class NativeESPnetASRTrainingAdapter(SpeechSeq2SeqTrainingAdapter):
    """Train and export the exact hybrid CTC/attention objective."""

    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-espnet-safetensors"

    def setup(self) -> NativeESPnetASRTrainingAdapter:
        super().setup()
        if getattr(self.model, "architecture_family", None) != "speech-seq2seq":
            raise ValueError("Native ESPnet fine-tuning requires the speech-seq2seq runtime.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError("Native ESPnet fine-tuning must target the wrapper's exact graph.")
        return self

    def create_dataset(self, records, **kwargs):
        if isinstance(records, ASRDataset) and not kwargs:
            return records
        return ESPnetASRTrainingDataset(
            records,
            transform=kwargs.get("transform"),
        )

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
        accepted = {
            "apply_augmentation",
            "feature_lengths",
            "features",
            "label_lengths",
            "labels",
            "waveform_lengths",
            "waveforms",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        values = dict(super().recipe_resume_configuration())
        values.update({
            "checkpoint_format": ("voicehub-espnet-librispeech-transformer-e18-v1"),
            "gradient_accumulation_steps": 6,
            "gradient_clip_norm": 5.0,
            "learning_rate": 0.002,
            "objective": "0.3-ctc-plus-0.7-label-smoothed-seq2seq",
            "optimizer": "adam",
            "sample_rate": 16_000,
            "scheduler": "warmuplr",
            "warmup_steps": 25_000,
        })
        return values

    def create_optimizer(
        self,
        name: str,
        parameters: list[tuple[str, Any]],
        training_args: Any,
    ):
        import torch

        if name not in {"default", "model"}:
            raise ValueError(f"Native ESPnet declares only the `model` optimizer, found {name!r}.")
        trainable = [parameter for _, parameter in parameters if parameter.requires_grad]
        if not trainable:
            raise ValueError("Native ESPnet has no trainable parameters.")
        return torch.optim.Adam(
            trainable,
            lr=training_args.learning_rate,
            betas=(
                training_args.adam_beta1,
                training_args.adam_beta2,
            ),
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
        )

    def create_scheduler(
        self,
        name: str,
        optimizer: Any,
        num_training_steps: int,
        training_args: Any,
    ):
        import torch

        del num_training_steps
        if name not in {"default", "model"}:
            raise ValueError(f"Native ESPnet declares only the `model` scheduler, found {name!r}.")
        warmup_steps = (training_args.warmup_steps if training_args.warmup_steps > 0 else 25_000)
        scale = warmup_steps**0.5

        def schedule(current_step: int) -> float:
            step = current_step + 1
            return scale * min(
                step**-0.5,
                step * warmup_steps**-1.5,
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": ("voicehub-espnet-librispeech-transformer-e18-v1"),
            "native_architecture_family": ("espnet-librispeech-transformer-e18"),
            "native_objective": "hybrid-ctc-attention",
            "processor_runtime": "voicehub-native",
        })
        return manifest

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


__all__ = [
    "ESPnetASRTrainingDataset",
    "NativeESPnetASRTrainingAdapter",
    "prepare_espnet_training_batch",
]
