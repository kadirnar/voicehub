"""Native Granite Speech prompt, waveform, and label preparation."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.hub import write_json_file
from voicehub.models.asr_granite_speech.native.configuration import GraniteSpeechArchitectureConfig
from voicehub.models.asr_granite_speech.native.frontend import SAMPLE_RATE, GraniteSpeechFeatureExtractor
from voicehub.models.asr_granite_speech.native.tokenization import AUDIO_TOKEN, GraniteSpeechTokenizer

DEFAULT_TRANSCRIPTION_PROMPT = ("Please transcribe the following audio to text<|audio|>")


def _broadcast(
    value: Any,
    batch_size: int,
    *,
    name: str,
    default: Any = None,
) -> tuple[Any, ...]:
    if value is None:
        return (default, ) * batch_size
    if isinstance(value, (str, bytes, Path)):
        return (value, ) * batch_size
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return (value.item(), ) * batch_size
        values = tuple(value.detach().cpu().tolist())
    elif isinstance(value, Sequence):
        values = tuple(value)
    else:
        return (value, ) * batch_size
    if len(values) != batch_size:
        raise ValueError(f"`{name}` contains {len(values)} values for a batch of "
                         f"{batch_size}.")
    return values


class GraniteSpeechProcessor:
    """Create multimodal causal batches without Transformers or torchaudio."""

    def __init__(
        self,
        config: GraniteSpeechArchitectureConfig,
        tokenizer: GraniteSpeechTokenizer,
        *,
        preprocessor_config_path: Path | None = None,
        processor_config_path: Path | None = None,
    ) -> None:
        if not isinstance(config, GraniteSpeechArchitectureConfig):
            raise TypeError("`config` must be GraniteSpeechArchitectureConfig.")
        if not isinstance(tokenizer, GraniteSpeechTokenizer):
            raise TypeError("`tokenizer` must be GraniteSpeechTokenizer.")
        if tokenizer.audio_token_id != config.audio_token_index:
            raise ValueError("Granite Speech tokenizer and model audio token IDs disagree.")
        if tokenizer.token_id_space_size != config.text_config.vocab_size:
            raise ValueError(
                "Granite Speech tokenizer ID space does not match the "
                "language-model vocabulary.")
        self.config = config
        self.tokenizer = tokenizer
        self.preprocessor_config_path = preprocessor_config_path
        self.processor_config_path = processor_config_path
        self.feature_extractor = GraniteSpeechFeatureExtractor(
            sampling_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=config.encoder_config.input_dim // 2,
            projector_window_size=config.window_size,
            projector_downsample_rate=config.downsample_rate,
        )
        if self.feature_extractor.input_dim != config.encoder_config.input_dim:
            raise ValueError(
                "Granite Speech encoder input width must be twice the "
                "configured mel-bin count.")

    @property
    def sample_rate(self) -> int:
        return self.feature_extractor.sampling_rate

    @staticmethod
    def instruction_prompt(prompt: str | None) -> str:
        instruction = (DEFAULT_TRANSCRIPTION_PROMPT if prompt is None else prompt)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Granite Speech prompt must be a non-empty string.")
        instruction = instruction.strip()
        if AUDIO_TOKEN not in instruction:
            instruction = f"{AUDIO_TOKEN}{instruction}"
        if instruction.count(AUDIO_TOKEN) != 1:
            raise ValueError(
                "Each Granite Speech prompt must contain exactly one "
                f"{AUDIO_TOKEN!r} placeholder.")
        return instruction

    @staticmethod
    def _with_hotwords(
        instruction: str,
        hotwords: str | Sequence[str] | None,
    ) -> str:
        if hotwords is None:
            return instruction
        values = ((hotwords, ) if isinstance(hotwords, str) else tuple(hotwords))
        if not values or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("Granite Speech hotwords must be non-empty strings.")
        return (f"{instruction.rstrip()} Keywords: " + ", ".join(value.strip() for value in values))

    @staticmethod
    def render_instruction(instruction: str) -> str:
        """Render the pinned Granite chat template without a Jinja runtime."""
        return f"USER: {instruction}\n ASSISTANT:"

    def build_prompt(
        self,
        *,
        prompt: str | None,
        audio_tokens: int,
        hotwords: str | Sequence[str] | None = None,
    ) -> str:
        if (isinstance(audio_tokens, bool) or not isinstance(audio_tokens, int) or audio_tokens <= 0):
            raise ValueError("`audio_tokens` must be a positive integer.")
        instruction = self._with_hotwords(
            self.instruction_prompt(prompt),
            hotwords,
        )
        instruction = instruction.replace(
            AUDIO_TOKEN,
            AUDIO_TOKEN * audio_tokens,
            1,
        )
        return self.render_instruction(instruction)

    @staticmethod
    def _left_pad(
        rows: Sequence[Sequence[int]],
        *,
        pad_token_id: int,
    ) -> tuple[Tensor, Tensor]:
        width = max(len(row) for row in rows)
        input_ids = torch.full(
            (len(rows), width),
            pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(rows), width),
            dtype=torch.bool,
        )
        for index, row in enumerate(rows):
            values = torch.tensor(tuple(row), dtype=torch.long)
            input_ids[index, width - len(row):] = values
            attention_mask[index, width - len(row):] = True
        return input_ids, attention_mask

    def prepare_inference_batch(
        self,
        audios: Any,
        *,
        sampling_rates: int | None | Sequence[int | None],
        prompts: str | None | Sequence[str | None] = None,
        hotwords: (str
                   | Sequence[str]
                   | Sequence[str | Sequence[str] | None]
                   | None) = None,
    ) -> dict[str, Tensor]:
        materialized = self.feature_extractor.materialize(
            audios,
            sampling_rates=sampling_rates,
        )
        features = self.feature_extractor.extract(
            materialized,
            sampling_rates=tuple(item.sampling_rate for item in materialized),
        )
        batch_size = len(materialized)
        prompt_rows = _broadcast(
            prompts,
            batch_size,
            name="prompts",
            default=None,
        )
        if (batch_size == 1 and hotwords is not None and
            (isinstance(hotwords, str) or
             (isinstance(hotwords, Sequence) and all(isinstance(item, str) for item in hotwords)))):
            hotword_rows = (hotwords, )
        else:
            hotword_rows = _broadcast(
                hotwords,
                batch_size,
                name="hotwords",
                default=None,
            )
        texts = tuple(
            self.build_prompt(
                prompt=prompt,
                audio_tokens=int(size.item()),
                hotwords=keywords,
            ) for prompt, keywords, size in zip(
                prompt_rows,
                hotword_rows,
                features["audio_embed_sizes"],
            ))
        encoded = tuple(self.tokenizer.encode_prompt(text).input_ids for text in texts)
        input_ids, attention_mask = self._left_pad(
            encoded,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": features["input_features"],
            "input_features_mask": features["input_features_mask"],
            "audio_lengths": features["audio_lengths"],
        }

    def prepare_training_batch(
        self,
        audios: Any,
        transcripts: Sequence[str],
        *,
        sampling_rates: int | None | Sequence[int | None],
        prompts: str | None | Sequence[str | None] = None,
    ) -> dict[str, Tensor]:
        materialized = self.feature_extractor.materialize(
            audios,
            sampling_rates=sampling_rates,
        )
        texts = tuple(transcripts)
        if len(texts) != len(materialized):
            raise ValueError("Granite Speech requires one transcript per waveform.")
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Granite Speech transcripts must be non-empty strings.")
        features = self.feature_extractor.extract(
            materialized,
            sampling_rates=tuple(item.sampling_rate for item in materialized),
        )
        prompt_rows = _broadcast(
            prompts,
            len(materialized),
            name="prompts",
            default=None,
        )
        encoded_prompts = tuple(
            self.tokenizer.encode_prompt(self.build_prompt(
                prompt=prompt,
                audio_tokens=int(size.item()),
            )).input_ids for prompt, size in zip(
                prompt_rows,
                features["audio_embed_sizes"],
            ))
        encoded_targets = tuple((
            *self.tokenizer.encode_transcript(text).input_ids,
            self.tokenizer.eos_token_id,
        ) for text in texts)
        prompt_width = max(len(row) for row in encoded_prompts)
        target_width = max(len(row) for row in encoded_targets)
        total_width = prompt_width + target_width
        input_ids = torch.full(
            (len(texts), total_width),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(texts), total_width),
            dtype=torch.bool,
        )
        labels = torch.full(
            (len(texts), total_width),
            -100,
            dtype=torch.long,
        )
        for index, (prompt_ids, target_ids) in enumerate(zip(encoded_prompts, encoded_targets)):
            prompt_start = prompt_width - len(prompt_ids)
            prompt_tensor = torch.tensor(
                prompt_ids,
                dtype=torch.long,
            )
            target_tensor = torch.tensor(
                target_ids,
                dtype=torch.long,
            )
            input_ids[index, prompt_start:prompt_width] = prompt_tensor
            attention_mask[index, prompt_start:prompt_width] = True
            input_ids[
                index,
                prompt_width:prompt_width + len(target_ids),
            ] = target_tensor
            attention_mask[
                index,
                prompt_width:prompt_width + len(target_ids),
            ] = True
            labels[
                index,
                prompt_width:prompt_width + len(target_ids),
            ] = target_tensor
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": features["input_features"],
            "input_features_mask": features["input_features_mask"],
            "audio_lengths": features["audio_lengths"],
        }

    def save_pretrained(self, directory: str | Path) -> Path:
        target = self.tokenizer.save_pretrained(directory)
        if self.preprocessor_config_path is not None:
            destination = target / "preprocessor_config.json"
            if self.preprocessor_config_path != destination.resolve():
                shutil.copyfile(
                    self.preprocessor_config_path,
                    destination,
                )
        else:
            write_json_file(
                target / "preprocessor_config.json",
                {
                    "feature_extractor_type": ("GraniteSpeechFeatureExtractor"),
                    "melspec_kwargs": {
                        "hop_length": self.feature_extractor.hop_length,
                        "n_fft": self.feature_extractor.n_fft,
                        "n_mels": self.feature_extractor.n_mels,
                        "sample_rate": self.feature_extractor.sampling_rate,
                        "win_length": self.feature_extractor.win_length,
                    },
                    "processor_class": "GraniteSpeechProcessor",
                    "projector_downsample_rate": (self.feature_extractor.projector_downsample_rate),
                    "projector_window_size": (self.feature_extractor.projector_window_size),
                    "sampling_rate": self.feature_extractor.sampling_rate,
                },
            )
        if self.processor_config_path is not None:
            destination = target / "processor_config.json"
            if self.processor_config_path != destination.resolve():
                shutil.copyfile(
                    self.processor_config_path,
                    destination,
                )
        else:
            write_json_file(
                target / "processor_config.json",
                {
                    "audio_token": AUDIO_TOKEN,
                    "processor_class": "GraniteSpeechProcessor",
                },
            )
        return target


__all__ = [
    "DEFAULT_TRANSCRIPTION_PROMPT",
    "GraniteSpeechProcessor",
]
