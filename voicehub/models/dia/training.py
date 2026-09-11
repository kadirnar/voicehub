"""Native data preparation and full fine-tuning support for Dia.

This module stays import-light. PyTorch and the model graph are resolved
only when a runtime is loaded, while dataset validation and adapter
discovery remain available to tooling without importing a provider
framework.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from voicehub.training.adapters import Seq2SeqTrainingAdapter

_PREPARED_INPUT_KEYS = frozenset({
    "input_ids",
    "decoder_input_ids",
    "labels",
})
_PROCESSOR_CONTROL_KEYS = frozenset({
    "audio",
    "generation",
    "output_labels",
    "padding",
    "return_tensors",
    "text",
})


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _processor_sample_rate(processor: Any) -> int:
    direct = getattr(processor, "sampling_rate", None)
    if direct is not None:
        return int(direct)
    feature_extractor = getattr(processor, "feature_extractor", None)
    return int(getattr(feature_extractor, "sampling_rate", 44_100))


def resolve_dia_dtype(
    torch_or_dtype: Any,
    dtype_name: str | None = None,
    device: str | None = None,
) -> Any:
    """Resolve Dia's configured dtype.

    The three-argument form is retained for callers of the former
    provider loader. New code should pass ``(dtype_name, device)``.
    """
    if isinstance(torch_or_dtype, str):
        resolved_name = torch_or_dtype
        resolved_device = dtype_name
    else:
        resolved_name = dtype_name
        resolved_device = device
    if not isinstance(resolved_name, str) or not isinstance(resolved_device, str):
        raise TypeError("resolve_dia_dtype requires a dtype name and device.")
    from voicehub.models.dia.native.runtime import resolve_dia_dtype as resolve_native_dtype

    return resolve_native_dtype(resolved_name, resolved_device)


def freeze_dia_audio_tokenizer(processor: Any) -> Any:
    """Freeze the native DAC tokenizer used for labels and reconstruction."""
    freeze = getattr(processor, "freeze_audio_tokenizer", None)
    if callable(freeze):
        return freeze()
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is None:
        raise TypeError("DiaProcessor must expose the native DAC audio tokenizer.")
    requires_grad = getattr(audio_tokenizer, "requires_grad_", None)
    if callable(requires_grad):
        requires_grad(False)
    else:
        parameters = getattr(audio_tokenizer, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                parameter.requires_grad = False
    evaluate = getattr(audio_tokenizer, "eval", None)
    if callable(evaluate):
        evaluate()
    return audio_tokenizer


def _normalize_audio(audio: Any, *, sample_rate: int) -> Any:
    if isinstance(audio, Mapping):
        source_rate = audio.get("sampling_rate")
        if source_rate is not None and int(source_rate) != sample_rate:
            raise ValueError(
                "Dia training audio must be resampled to "
                f"{sample_rate} Hz; received {source_rate} Hz.")
        if "array" not in audio and "path" not in audio:
            raise ValueError("Dia audio mappings require an 'array' or 'path' field.")
        return dict(audio)
    if isinstance(audio, (str, PathLike)):
        if not str(audio).strip():
            raise ValueError("Dia audio paths must be non-empty.")
        return audio
    ndim = getattr(audio, "ndim", None)
    if ndim is not None and int(ndim) != 1:
        raise ValueError("Dia training waveforms must be mono rank-1 values.")
    return audio


def _normalize_record(
    record: Mapping[str, Any],
    *,
    index: int,
    sample_rate: int,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("Every Dia training record must be a mapping.")
    missing = [name for name in ("text", "audio") if name not in record]
    if missing:
        raise ValueError(f"Dia training record {index} is missing: {', '.join(missing)}.")
    text = record["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Dia training record {index} requires non-empty text.")
    return {
        "text": text,
        "audio": _normalize_audio(record["audio"], sample_rate=sample_rate),
    }


@dataclass
class DiaTrainingCollator:
    """Create delayed decoder inputs and masked channel-major labels."""

    processor: Any
    sample_rate: int | None = None
    processor_kwargs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.sample_rate = int(
            self.sample_rate if self.sample_rate is not None else _processor_sample_rate(self.processor))
        self.processor_kwargs = dict(self.processor_kwargs or {})
        collisions = sorted(_PROCESSOR_CONTROL_KEYS.intersection(self.processor_kwargs))
        if collisions:
            raise ValueError(
                "Dia processor_kwargs cannot override training controls: " + ", ".join(collisions))

    def __call__(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not _is_sequence(records) or not records:
            raise ValueError("DiaTrainingCollator requires at least one training record.")
        normalized = [
            _normalize_record(
                record,
                index=index,
                sample_rate=self.sample_rate,
            ) for index, record in enumerate(records)
        ]
        prepared = self.processor(
            text=[record["text"] for record in normalized],
            audio=[record["audio"] for record in normalized],
            generation=False,
            output_labels=True,
            padding=True,
            return_tensors="pt",
            **self.processor_kwargs,
        )
        if not isinstance(prepared, Mapping):
            raise TypeError("DiaProcessor must return a mapping.")
        output = dict(prepared)
        required = (
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "labels",
        )
        missing = [name for name in required if name not in output]
        if missing:
            raise RuntimeError("DiaProcessor did not return required training fields: " + ", ".join(missing))
        return output

    def resume_fingerprint(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "processor_kwargs": dict(self.processor_kwargs),
        }


class DiaSFTDataset:
    """Raw dialogue/audio records encoded by the native processor on batch."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        processor: Any,
        sample_rate: int | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not _is_sequence(records) or not records:
            raise ValueError("DiaSFTDataset requires at least one record.")
        self.records = tuple(records)
        self.collator = DiaTrainingCollator(
            processor,
            sample_rate=sample_rate,
            processor_kwargs=processor_kwargs,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _normalize_record(
            self.records[index],
            index=index,
            sample_rate=self.collator.sample_rate,
        )

    def collate_fn(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return self.collator(records)

    def resume_fingerprint(self) -> dict[str, Any]:
        return {"collator": self.collator.resume_fingerprint()}


def _columnar_records(inputs: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = inputs.get("records")
    if records is not None:
        if not _is_sequence(records):
            raise TypeError("Dia 'records' must be a sequence of mappings.")
        return list(records)
    text = inputs.get("text")
    audio = inputs.get("audio")
    if _is_sequence(text) and _is_sequence(audio) and len(text) == len(audio):
        return [{"text": item_text, "audio": item_audio} for item_text, item_audio in zip(text, audio)]
    return [inputs]


def prepare_dia_training_inputs(
    processor: Any,
    inputs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    sample_rate: int | None = None,
    processor_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pass model-ready tensors through or encode raw text/audio records."""
    if isinstance(inputs, Mapping):
        if _PREPARED_INPUT_KEYS.issubset(inputs):
            return dict(inputs)
        records = _columnar_records(inputs)
    elif _is_sequence(inputs):
        records = list(inputs)
    else:
        raise TypeError("Dia training inputs must be a mapping or record sequence.")
    return DiaTrainingCollator(
        processor,
        sample_rate=sample_rate,
        processor_kwargs=processor_kwargs,
    )(records)


@dataclass
class DiaTrainingBackend:
    """Small compatibility facade around a native Dia model and processor."""

    model: Any
    processor: Any
    sample_rate: int
    runtime: Any | None = None
    transformers_major_version: int | None = None

    def prepare_for_training(self):
        train = getattr(self.model, "train", None)
        if callable(train):
            train()
        freeze_dia_audio_tokenizer(self.processor)
        return self

    def prepare_for_inference(self):
        evaluate = getattr(self.model, "eval", None)
        if callable(evaluate):
            evaluate()
        freeze_dia_audio_tokenizer(self.processor)
        return self

    def create_collator(
        self,
        *,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> DiaTrainingCollator:
        return DiaTrainingCollator(
            self.processor,
            sample_rate=self.sample_rate,
            processor_kwargs=processor_kwargs,
        )

    def prepare_inputs(
        self,
        inputs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return prepare_dia_training_inputs(
            self.processor,
            inputs,
            sample_rate=self.sample_rate,
            processor_kwargs=processor_kwargs,
        )

    @staticmethod
    def scalar_loss(outputs: Any) -> Any:
        loss = (outputs.get("loss") if isinstance(outputs, Mapping) else getattr(outputs, "loss", None))
        if loss is None:
            raise RuntimeError("Native Dia returned no loss. Prepare the batch with "
                               "output_labels=True.")
        numel = getattr(loss, "numel", None)
        if callable(numel) and int(numel()) != 1:
            raise ValueError("Native Dia must return exactly one loss value.")
        reshape = getattr(loss, "reshape", None)
        return reshape(()) if callable(reshape) else loss

    def forward_loss(
        self,
        inputs: Mapping[str, Any] | None = None,
        **model_inputs: Any,
    ) -> Any:
        if inputs is not None:
            if model_inputs:
                raise ValueError("Pass Dia inputs as a mapping or keywords, not both.")
            model_inputs = dict(inputs)
        return self.scalar_loss(self.model(**model_inputs))

    def save_pretrained(self, save_directory: str | Path) -> Path:
        if self.runtime is not None:
            return self.runtime.save_pretrained(save_directory)
        save = getattr(self.model, "save_pretrained", None)
        if not callable(save):
            raise TypeError(
                "This compatibility backend is not attached to a portable "
                "native Dia runtime.")
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        save(destination)
        return destination


def load_dia_native_backend(
    model_name_or_path: str | Path,
    *,
    device: str,
    compute_dtype: str = "bfloat16",
    for_training: bool = False,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
):
    """Load the strict VoiceHub-native Dia runtime."""
    from voicehub.models.dia.native.runtime import load_dia_runtime

    return load_dia_runtime(
        model_name_or_path,
        device=device,
        compute_dtype=compute_dtype,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        for_training=for_training,
    )


def load_dia_transformers_backend(*args: Any, **kwargs: Any):
    """Compatibility alias for the now-native loader.

    No Transformers module is imported or executed.
    """
    return load_dia_native_backend(*args, **kwargs)


class _DiaAdapterCollator:

    def __init__(self, adapter: DiaTrainingAdapter) -> None:
        self.adapter = adapter

    def __call__(self, records):
        self.adapter.setup()
        backend = self.adapter.backend
        return DiaTrainingCollator(
            backend.processor,
            sample_rate=backend.sample_rate,
        )(records)

    def resume_fingerprint(self) -> dict[str, Any]:
        config = getattr(self.adapter.model, "config", None)
        return {
            "sample_rate": getattr(config, "sample_rate", None),
            'base_model': getattr(config, "name_or_path", None),
        }


class DiaTrainingAdapter(Seq2SeqTrainingAdapter):
    """Train every Dia parameter through its channel-major CE objective."""

    supports_custom_recipe = True
    native_export_semantics = "inference-export"

    def __init__(self, model: Any, spec: Any) -> None:
        super().__init__(model, spec)
        self.data_collator = _DiaAdapterCollator(self)

    @property
    def backend(self):
        backend = getattr(self.model, "training_backend", None)
        if backend is None:
            raise RuntimeError(
                "Dia's native training backend is not loaded. Call setup() "
                "or load_for_training() first.")
        return backend

    def setup(self):
        super().setup()
        backend = self.backend
        if backend.model is not self.primary_model:
            raise RuntimeError(
                "Dia's training profile did not resolve the native "
                "DiaForConditionalGeneration graph.")
        backend.prepare_for_training()
        return self

    def create_dataset(self, records, **kwargs):
        self.setup()
        processor_kwargs = dict(kwargs.pop("processor_kwargs", {}) or {})
        duplicate = sorted(set(processor_kwargs).intersection(kwargs))
        if duplicate:
            raise ValueError("Dia processor options were passed twice: " + ", ".join(duplicate))
        processor_kwargs.update(kwargs)
        dataset = DiaSFTDataset(
            records,
            processor=self.backend.processor,
            sample_rate=self.backend.sample_rate,
            processor_kwargs=processor_kwargs,
        )
        self.data_collator = dataset.collate_fn
        return dataset

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        self.backend.save_pretrained(save_directory)


__all__ = [
    "DiaSFTDataset",
    "DiaTrainingAdapter",
    "DiaTrainingBackend",
    "DiaTrainingCollator",
    "freeze_dia_audio_tokenizer",
    "load_dia_native_backend",
    "load_dia_transformers_backend",
    "prepare_dia_training_inputs",
    "resolve_dia_dtype",
]
