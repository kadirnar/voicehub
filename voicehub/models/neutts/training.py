"""Source-faithful native data preparation for NeuTTS-Air fine-tuning."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.models.neutts.native.metadata import NEUTTS_SOURCE_REVISION, NEUTTS_TRAINING_SOURCE
from voicehub.models.neutts.native.tokenization import (
    SPEECH_CODEBOOK_SIZE,
    SPEECH_GENERATION_END,
    SPEECH_GENERATION_START,
    TEXT_PROMPT_END,
    TEXT_PROMPT_START,
    normalize_neutts_text,
)
from voicehub.processing.waveform import load_native_audio
from voicehub.training.data import CausalTokenCollator
from voicehub.training.recipes import CodecCausalLMTrainingAdapter

NEUTTS_TRAINING_SOURCE_REVISION = NEUTTS_SOURCE_REVISION


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return value


class NeuTTSSFTDataset:
    """Build completion-only examples from the pinned NeuTTS-Air recipe.

    Records require ``text`` and either safe precomputed ``codes`` /
    ``audio_codes`` or raw audio accepted by
    :func:`voicehub.processing.waveform.load_native_audio`. Phoneme variants
    additionally require a ``phonemes`` field or an explicitly injected
    phonemizer. VoiceHub does not install or invoke eSpeak implicitly.
    """

    SPEECH_START = SPEECH_GENERATION_START
    SPEECH_END = SPEECH_GENERATION_END

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        runtime,
        max_length: int = 2_048,
        phonemizer: Callable[[str], str] | Any | None = None,
        truncate: bool = True,
    ) -> None:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("`records` must be a sequence of mappings.")
        normalized = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"NeuTTS record {index} must be a mapping.")
            normalized.append(dict(record))
        if not normalized:
            raise ValueError("NeuTTSSFTDataset requires at least one record.")
        if not isinstance(truncate, bool):
            raise TypeError("`truncate` must be a boolean.")
        if (phonemizer is not None and not callable(phonemizer) and
                not callable(getattr(phonemizer, "phonemize", None))):
            raise TypeError("`phonemizer` must be callable or expose phonemize().")
        tokenizer = getattr(runtime, "tokenizer", None)
        codec = getattr(runtime, "codec", None)
        if tokenizer is None:
            raise ValueError("NeuTTS SFT requires the native tokenizer.")
        if getattr(runtime, "input_format", None) not in {"phonemes", "BPE"}:
            raise ValueError("NeuTTS runtime does not declare a supported input format.")
        self.records = tuple(normalized)
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.codec = codec
        self.input_format = runtime.input_format
        self.max_length = _positive_integer(max_length, name="max_length")
        if self.max_length < 2:
            raise ValueError("`max_length` must be at least two.")
        self.phonemizer = phonemizer
        self.truncate = truncate
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("NeuTTS tokenizer must define a pad or EOS ID.")
        self.collate_fn = CausalTokenCollator(pad_token_id=int(pad_token_id), )
        if codec is not None:
            codec.eval()
            for parameter in codec.parameters():
                parameter.requires_grad_(False)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _normalize_codes(codes: Any) -> list[int]:
        if isinstance(codes, Tensor):
            values = codes.detach()
        else:
            try:
                values = torch.as_tensor(codes)
            except (TypeError, ValueError, RuntimeError) as error:
                raise TypeError("NeuTTS audio codes must be an integer tensor or sequence.") from error
        if values.numel() == 0:
            raise ValueError("NeuTTS audio codes cannot be empty.")
        if (values.dtype == torch.bool or values.is_floating_point() or values.is_complex()):
            raise TypeError("NeuTTS audio codes must use an integer dtype.")
        if values.ndim > 1 and any(size != 1 for size in values.shape[:-1]):
            raise ValueError(
                "Each NeuTTS record must contain one code sequence, "
                "optionally shaped [1, frames] or [1, 1, frames].")
        flattened = values.reshape(-1).to(dtype=torch.long, device="cpu")
        if bool((flattened < 0).any()) or bool((flattened >= SPEECH_CODEBOOK_SIZE).any()):
            raise ValueError("NeuTTS NeuCodec codes must be in [0, 65535].")
        return [int(value) for value in flattened.tolist()]

    def _load_waveform(self, record: Mapping[str, Any]) -> Tensor:
        if "audio_values" in record:
            audio = record["audio_values"]
        elif "waveform" in record:
            audio = record["waveform"]
        elif "audio" in record:
            audio = record["audio"]
        else:
            raise ValueError(
                "NeuTTS records require `codes`, `audio_codes`, `audio`, "
                "`waveform`, or `audio_values`.")
        source_rate = record.get(
            "sampling_rate",
            record.get("sample_rate"),
        )
        input_rate = (
            getattr(self.codec, "input_sampling_rate", 16_000) if self.codec is not None else 16_000)
        return load_native_audio(
            audio,
            sampling_rate=(
                source_rate if source_rate is not None else
                (None if isinstance(audio, (str, Path, Mapping)) else input_rate)),
            target_sampling_rate=input_rate,
        ).waveform

    def _speech_ids(self, record: Mapping[str, Any]) -> list[int]:
        codes = record.get("audio_codes", record.get("codes"))
        if codes is not None:
            return self._normalize_codes(codes)
        if self.codec is None:
            raise ValueError(
                "Raw-audio NeuTTS records require a loaded native NeuCodec. "
                "Provide precomputed `codes` for offline preparation.")
        waveform = self._load_waveform(record)
        device = next(self.codec.parameters()).device
        self.codec.eval()
        with torch.inference_mode():
            encoded = self.codec.encode_code(
                waveform.to(device).unsqueeze(0),
                sample_rate=self.codec.input_sampling_rate,
            )
        return self._normalize_codes(encoded)

    def _phonemes(self, text: str, record: Mapping[str, Any]) -> str:
        explicit = record.get("phonemes")
        return self.runtime._resolve_phonemes(
            text,
            explicit=explicit,
            phonemizer=self.phonemizer,
            name="phonemes",
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"NeuTTS record {index} requires non-empty `text`.")
        content = (
            self._phonemes(text, record) if self.input_format == "phonemes" else normalize_neutts_text(text))
        prompt = (
            "user: Convert the text to speech:"
            f"{TEXT_PROMPT_START}{content}{TEXT_PROMPT_END}\n"
            f"assistant:{self.SPEECH_START}")
        prompt_ids = list(self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
        ).input_ids)
        speech_start_id = self.tokenizer.convert_tokens_to_ids(self.SPEECH_START)
        if not prompt_ids or prompt_ids[-1] != speech_start_id:
            raise RuntimeError("NeuTTS tokenizer did not preserve the terminal "
                               "speech-generation marker.")
        completion = [self.tokenizer.speech_code_to_token_id(code) for code in self._speech_ids(record)]
        completion.append(self.tokenizer.convert_tokens_to_ids(self.SPEECH_END))
        input_ids = prompt_ids + completion
        if len(input_ids) > self.max_length:
            if not self.truncate:
                raise ValueError(
                    f"NeuTTS record {index} produces {len(input_ids)} tokens, "
                    f"exceeding max_length={self.max_length}.")
            input_ids = input_ids[:self.max_length]
        completion_start = len(prompt_ids) - 1
        if completion_start >= len(input_ids):
            raise ValueError(
                "The NeuTTS sequence limit removed the speech-generation "
                "start token; shorten or pre-segment this record.")
        labels = [-100] * completion_start + input_ids[completion_start:]
        if not any(label != -100 for label in labels):
            raise ValueError(f"NeuTTS record {index} has no trainable completion tokens.")
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class NeuTTSTrainingAdapter(CodecCausalLMTrainingAdapter):
    """Export the complete LM/tokenizer/NeuCodec native runtime."""

    native_export_semantics = "inference-export"

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        export = getattr(self.model, "export_native_pretrained", None)
        if not callable(export):
            raise TypeError("Native NeuTTS training requires a wrapper with "
                            "export_native_pretrained().")
        export(save_directory)

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-neutts-v1",
            "native_architecture_family": "neutts-air",
            "objective": "completion-only-causal-language-modeling",
            "objective_author_verified": True,
            "training_scope": "full-language-model",
            "frozen_components": ["neucodec"],
            "raw_audio_preprocessing": "optional-frozen-native-neucodec",
            "phonemization": "precomputed-or-explicitly-injected",
            "inference_reloadable": True,
            "source_revision": NEUTTS_TRAINING_SOURCE_REVISION,
        })
        return manifest


def build_training_dataset(model, records, **kwargs) -> NeuTTSSFTDataset:
    """Build the source-native dataset declared by the training registry."""
    return NeuTTSSFTDataset(
        records,
        runtime=model.model,
        max_length=int(kwargs.get("max_length", 2_048)),
        phonemizer=kwargs.get("phonemizer"),
        truncate=kwargs.get("truncate", True),
    )


__all__ = [
    "NEUTTS_TRAINING_SOURCE",
    "NEUTTS_TRAINING_SOURCE_REVISION",
    "NeuTTSSFTDataset",
    "NeuTTSTrainingAdapter",
    "build_training_dataset",
]
