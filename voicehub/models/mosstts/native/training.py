"""Raw-audio and pre-encoded fine-tuning for every native MOSS-TTS graph."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.models.mosstts.native.processing import MossProcessorBatch, MossTTSProcessor
from voicehub.models.mosstts.native.runtime import MossTTSRuntime
from voicehub.training.adapters import CausalLMTrainingAdapter
from voicehub.training.contracts import TrainingContext


def _validate_record(
    record: Mapping[str, Any],
    *,
    index: int,
    processor: MossTTSProcessor,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"MOSS-TTS training record {index} must be a mapping.")
    if "text" not in record:
        raise ValueError(f"MOSS-TTS training record {index} requires `text`.")
    text = record["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"MOSS-TTS training record {index} requires non-empty text.")
    raw_keys = [key for key in ("audio", "waveform", "audio_path") if record.get(key) is not None]
    has_codes = record.get("speech_tokens") is not None
    if int(has_codes) + len(raw_keys) != 1:
        raise ValueError(
            f"MOSS-TTS training record {index} requires exactly one of "
            "`speech_tokens`, `audio`, `waveform`, or `audio_path`.")
    speech_tokens = (
        processor._codes(
            record["speech_tokens"],
            name=f"records[{index}].speech_tokens",
        ).detach() if has_codes else None)
    references = record.get("reference_codes", ())
    if references is None:
        references = ()
    if isinstance(references, Tensor):
        references = (references, )
    if (not isinstance(references, Sequence) or isinstance(references, (str, bytes, bytearray))):
        raise TypeError(
            f"MOSS-TTS training record {index} `reference_codes` must be a "
            "sequence of code matrices.")
    normalized_references = tuple(
        processor._codes(
            value,
            name=f"records[{index}].reference_codes[{reference_index}]",
        ).detach() for reference_index, value in enumerate(references))
    output = dict(record)
    output["text"] = text
    if speech_tokens is not None:
        output["speech_tokens"] = speech_tokens
    if normalized_references or "reference_codes" in record:
        output["reference_codes"] = normalized_references
    return output


class MossPreencodedDataset(Sequence[dict[str, Any]]):
    """Validated raw-waveform or text/RVQ records.

    The compatibility class name is retained for existing callers.  When
    a runtime is supplied, raw PCM WAVE, tensor, mapping, and
    ``NativeAudio`` records are encoded lazily by the frozen native MOSS
    codec.
    """

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        processor: MossTTSProcessor,
        runtime: MossTTSRuntime | None = None,
    ) -> None:
        if isinstance(records, (str, bytes, bytearray, Mapping)):
            raise TypeError("MOSS-TTS `records` must be an iterable of mappings.")
        if not isinstance(processor, MossTTSProcessor):
            raise TypeError("`processor` must be MossTTSProcessor.")
        if runtime is not None and not isinstance(runtime, MossTTSRuntime):
            raise TypeError("`runtime` must be MossTTSRuntime or None.")
        self.processor = processor
        self.runtime = runtime
        self._records = tuple(
            _validate_record(
                record,
                index=index,
                processor=processor,
            ) for index, record in enumerate(records))
        if not self._records:
            raise ValueError("MOSS-TTS fine-tuning requires at least one record.")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [dict(record) for record in self._records[index]]
        return dict(self._records[index])

    def collate_fn(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        if self.runtime is not None:
            return self.runtime.prepare_training_batch(records).to_dict()
        if any(record.get("speech_tokens") is None for record in records):
            raise RuntimeError("Raw MOSS training records require a loaded native runtime.")
        return self.processor.collate_training(records).to_dict()

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(key for record in self._records for key in record))


class NativeMossTTSTrainingAdapter(CausalLMTrainingAdapter):
    """One adapter for Delay, Local, Local v1.5, and Realtime SFT."""

    supports_custom_recipe = True
    native_export_semantics = "inference-reloadable-safetensors"

    def _runtime(self) -> MossTTSRuntime:
        runtime = getattr(self.model, "training_backend", None)
        if runtime is None:
            runtime = getattr(self.model, "_mosstts_runtime", None)
        if not isinstance(runtime, MossTTSRuntime):
            raise TypeError(
                "MOSS-TTS fine-tuning requires the native runtime loaded "
                "through `load_for_training()`.")
        return runtime

    def setup(self) -> NativeMossTTSTrainingAdapter:
        super().setup()
        runtime = self._runtime()
        if self.primary_model is not runtime.model:
            raise TypeError("MOSS-TTS training did not resolve the native semantic graph.")
        runtime.prepare_for_training()
        return self

    def create_dataset(
        self,
        records: Iterable[Mapping[str, Any]],
        **kwargs: Any,
    ) -> MossPreencodedDataset:
        if kwargs:
            raise ValueError(
                "MOSS-TTS pre-encoded dataset does not accept options: " + ", ".join(sorted(kwargs)) + ".")
        self.setup()
        runtime = self._runtime()
        return MossPreencodedDataset(
            records,
            processor=runtime.processor,
            runtime=runtime,
        )

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: TrainingContext,
    ) -> Mapping[str, Any]:
        del context
        if not isinstance(inputs, Mapping):
            raise TypeError("MOSS-TTS training inputs must be a mapping.")
        if {"input_ids", "attention_mask", "labels"} <= set(inputs):
            return {
                "input_ids":
                inputs["input_ids"],
                "attention_mask":
                inputs["attention_mask"],
                "labels":
                inputs["labels"],
                **({
                    "channelwise_loss_weight": inputs["channelwise_loss_weight"],
                } if "channelwise_loss_weight" in inputs else {}),
            }
        records = inputs.get("records")
        if records is None:
            # A single uncollated record remains useful in custom trainer
            # loops; the runtime materializes raw audio before processing.
            records = (inputs, )
        if (not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray))):
            raise TypeError("MOSS-TTS `records` must be a sequence of mappings.")
        return self._runtime().prepare_training_batch(records).to_dict()

    def create_optimizer(
        self,
        name: str,
        parameters: list[tuple[str, Any]],
        training_args: Any,
    ):
        del name
        decay: list[Tensor] = []
        no_decay: list[Tensor] = []
        for parameter_name, parameter in parameters:
            target = (
                no_decay if parameter_name.endswith(".bias") or "norm" in parameter_name.lower() else decay)
            target.append(parameter)
        groups = []
        if decay:
            groups.append({
                "params": decay,
                "weight_decay": training_args.weight_decay,
            })
        if no_decay:
            groups.append({
                "params": no_decay,
                "weight_decay": 0.0,
            })
        config = self.model.config
        return torch.optim.AdamW(
            groups,
            lr=training_args.learning_rate,
            betas=(
                float(getattr(config, "training_adam_beta1", 0.9)),
                float(getattr(config, "training_adam_beta2", 0.95)),
            ),
            eps=float(getattr(config, "training_adam_epsilon", 1e-4)),
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self._runtime().save_pretrained(save_directory)

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        runtime = (self._runtime() if self.is_ready else getattr(self.model, "_mosstts_runtime", None))
        variant = (
            runtime.config.variant if isinstance(runtime, MossTTSRuntime) else getattr(
                self.model.config, "variant", "auto"))
        manifest.update({
            "checkpoint_format": "voicehub-mosstts-v1",
            "variant": variant,
            "training_scope": "raw-or-preencoded-rvq-full-semantic-model",
            "raw_audio_fine_tuning": True,
            "objective": "multichannel-next-token-cross-entropy",
            "frozen_components": ["codec"],
            "inference_reloadable": True,
        })
        return manifest


__all__ = [
    "MossPreencodedDataset",
    "MossTTSDataset",
    "NativeMossTTSTrainingAdapter",
]

MossTTSDataset = MossPreencodedDataset
