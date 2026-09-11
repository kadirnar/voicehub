"""Source-faithful OuteTTS V3 completion-only fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from voicehub.models.outetts.native.metadata import OUTETTS_TRAINING_SOURCE, OUTETTS_TRAINING_SOURCE_REVISION
from voicehub.models.outetts.native.prompting import OuteTTSPromptProcessor, SpeakerProfile
from voicehub.training.data import CausalTokenCollator
from voicehub.training.recipes import CodecCausalLMTrainingAdapter


def _integer_sequence(
    value: Any,
    *,
    name: str,
    allow_ignore_index: bool,
) -> list[int]:
    if (isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value):
        raise ValueError(f"OuteTTS `{name}` must be a non-empty sequence.")
    output = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"OuteTTS `{name}` must contain integers.")
        if item < 0 and not (allow_ignore_index and item == -100):
            raise ValueError(f"OuteTTS `{name}` contains invalid token ID {item}.")
        output.append(item)
    return output


class OuteTTSSFTDataset:
    """Build exact V3 causal-LM examples from aligned speaker profiles.

    Raw audio is deliberately not accepted. OuteTTS's author recipe
    relies on word-level timestamps, two DAC codebooks, and per-word
    acoustic features; silently deriving only a subset would train a
    different objective.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        interface=None,
        runtime=None,
        tokenizer=None,
        completion_only: bool = True,
        max_length: int | None = None,
        prompt_word_count: int | None = None,
        whisper_model: str | None = None,
        whisper_device: str | None = None,
    ) -> None:
        del whisper_model, whisper_device
        if isinstance(records, (str, bytes)) or not isinstance(
                records,
                Sequence,
        ):
            raise TypeError("`records` must be a sequence of mappings.")
        normalized = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"OuteTTS record {index} must be a mapping.")
            normalized.append(dict(record))
        if not normalized:
            raise ValueError("OuteTTSSFTDataset requires at least one record.")
        if not isinstance(completion_only, bool):
            raise TypeError("`completion_only` must be a boolean.")
        if max_length is not None and (isinstance(max_length, bool) or not isinstance(max_length, int) or
                                       max_length < 2):
            raise ValueError("`max_length` must be an integer of at least two or None.")
        if prompt_word_count is not None and (isinstance(
                prompt_word_count, bool) or not isinstance(prompt_word_count, int) or prompt_word_count < 0):
            raise ValueError("`prompt_word_count` must be non-negative or None.")
        selected_runtime = runtime if runtime is not None else interface
        processor = getattr(selected_runtime, "prompt_processor", None)
        if processor is None and tokenizer is not None:
            processor = OuteTTSPromptProcessor(tokenizer)
        if not isinstance(processor, OuteTTSPromptProcessor):
            raise TypeError("OuteTTS fine-tuning requires the native V3 prompt processor.")
        self.records = tuple(normalized)
        self.prompt_processor = processor
        self.tokenizer = processor.tokenizer
        self.completion_only = completion_only
        self.max_length = max_length
        self.prompt_word_count = prompt_word_count
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("OuteTTS tokenizer must define a pad or EOS token.")
        self.collate_fn = CausalTokenCollator(pad_token_id=int(pad_token_id))

    def __len__(self) -> int:
        return len(self.records)

    def _prepared(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        if "input_ids" not in record and "labels" not in record:
            return None
        if "input_ids" not in record or "labels" not in record:
            raise ValueError("Prepared OuteTTS records require both `input_ids` and "
                             "`labels`.")
        input_ids = _integer_sequence(
            record["input_ids"],
            name="input_ids",
            allow_ignore_index=False,
        )
        labels = _integer_sequence(
            record["labels"],
            name="labels",
            allow_ignore_index=True,
        )
        if len(input_ids) != len(labels):
            raise ValueError("Prepared OuteTTS `input_ids` and `labels` lengths differ.")
        if not any(value != -100 for value in labels):
            raise ValueError("Prepared OuteTTS labels contain no trainable tokens.")
        return {"input_ids": input_ids, "labels": labels}

    @staticmethod
    def _profile(
        record: Mapping[str, Any],
        *,
        index: int,
    ) -> SpeakerProfile:
        if any(name in record for name in ("audio", "audio_values", "waveform")):
            raise ValueError(
                f"OuteTTS record {index} contains raw audio. Prepare an "
                "author-compatible V3 speaker profile with word timestamps, "
                "DAC c1/c2 codes, and acoustic features.")
        value = record.get(
            "speaker_profile",
            record.get("speaker", record.get("profile")),
        )
        if value is None and {
                "text",
                "words",
                "global_features",
        } <= set(record):
            value = record
        if not isinstance(value, Mapping):
            raise ValueError(
                f"OuteTTS record {index} requires `speaker_profile` or "
                "prepared `input_ids`/`labels`.")
        profile = SpeakerProfile.from_mapping(value)
        declared_text = record.get("text")
        if declared_text is not None and declared_text != profile.text:
            raise ValueError(
                f"OuteTTS record {index} `text` must exactly match the "
                "aligned speaker-profile text.")
        return profile

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        prepared = self._prepared(record)
        if prepared is not None:
            result = prepared
        else:
            profile = self._profile(record, index=index)
            prompt = self.prompt_processor.training_prompt(profile)
            input_ids = self.prompt_processor.encode(prompt)
            labels = list(input_ids)
            word_count = record.get(
                "prompt_word_count",
                self.prompt_word_count,
            )
            if word_count is not None:
                prefix = self.prompt_processor.training_prefix(
                    profile,
                    prompt_word_count=word_count,
                )
                completion_start = len(self.prompt_processor.encode(prefix))
                labels[:completion_start] = [-100] * completion_start
            elif self.completion_only:
                audio_start_id = self.tokenizer.convert_tokens_to_ids(OuteTTSPromptProcessor.AUDIO_START)
                try:
                    completion_start = input_ids.index(audio_start_id) + 1
                except ValueError as error:
                    raise ValueError(
                        "OuteTTS training prompt is missing its audio-start "
                        "token.") from error
                labels[:completion_start] = [-100] * completion_start
            result = {
                "input_ids": input_ids,
                "labels": labels,
            }
        if self.max_length is not None and len(result["input_ids"]) > self.max_length:
            raise ValueError(
                f"OuteTTS record {index} produces "
                f"{len(result['input_ids'])} tokens, exceeding "
                f"max_length={self.max_length}. Pre-segment the aligned "
                "profile rather than truncating codec frames.")
        return result


class OuteTTSTrainingAdapter(CodecCausalLMTrainingAdapter):
    """Fine-tune the full native LM while keeping DAC strictly frozen."""

    native_export_semantics = "inference-export"

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        export = getattr(self.model, "export_native_pretrained", None)
        if not callable(export):
            raise TypeError("Native OuteTTS training requires a wrapper with "
                            "export_native_pretrained().")
        export(save_directory)

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-outetts-v1",
            "native_architecture_family": "outetts-v3",
            "objective": OUTETTS_TRAINING_SOURCE["objective"],
            "objective_author_verified": True,
            "training_scope": "full-language-model",
            "frozen_components": ["dac"],
            "raw_audio_preprocessing": "unsupported-requires-word-alignment",
            "prepared_inputs": [
                "v3-speaker-profile",
                "input_ids-and-labels",
            ],
            "inference_reloadable": True,
            "source_revision": OUTETTS_TRAINING_SOURCE_REVISION,
        })
        return manifest


def build_training_dataset(model, records, **kwargs) -> OuteTTSSFTDataset:
    """Build the source-native dataset declared by the training registry."""
    return OuteTTSSFTDataset(
        records,
        runtime=model.model,
        completion_only=bool(kwargs.get("completion_only", True)),
        max_length=kwargs.get("max_length"),
        prompt_word_count=kwargs.get("prompt_word_count"),
        whisper_model=kwargs.get("whisper_model"),
        whisper_device=kwargs.get("whisper_device"),
    )


__all__ = [
    "OUTETTS_TRAINING_SOURCE",
    "OUTETTS_TRAINING_SOURCE_REVISION",
    "OuteTTSSFTDataset",
    "OuteTTSTrainingAdapter",
    "build_training_dataset",
]
