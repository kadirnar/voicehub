# Copyright 2026 The Alibaba Qwen team and VoiceHub contributors.
# SPDX-License-Identifier: Apache-2.0
"""Dataset and collation for the official Qwen3-TTS 12 Hz SFT recipe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from voicehub.models.qwen3tts.native.runtime import qwen3_tts_speaker_mel
from voicehub.processing import load_native_audio


class Qwen3TTSSFTDataset:
    """Prepare single-speaker examples for Qwen3-TTS Base fine-tuning.

    Records contain ``text``, pre-extracted ``audio_codes`` with shape
    ``[frames, 16]``, and a 24 kHz ``ref_audio`` path. The same
    reference is recommended for every record by the upstream recipe.
    """

    def __init__(self, records: Sequence[Mapping[str, Any]], processor, config):
        normalized_records: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"Qwen3-TTS record {index} must be a mapping.")
            normalized_records.append(dict(record))
        self.records = tuple(normalized_records)
        self.processor = processor
        self.config = config
        if not self.records:
            raise ValueError("Qwen3TTSSFTDataset requires at least one record.")
        if not callable(processor):
            raise TypeError("Qwen3-TTS dataset processor must be callable.")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load_audio(path: str):
        audio = load_native_audio(
            path,
            target_sampling_rate=24_000,
        )
        return audio.waveform, audio.sampling_rate

    @staticmethod
    def _assistant_prompt(text: str) -> str:
        return (f"<|im_start|>assistant\n{text}<|im_end|>\n"
                "<|im_start|>assistant\n")

    def _extract_reference_mel(self, audio, sample_rate: int):
        if sample_rate != 24_000:
            raise ValueError(
                "Qwen3-TTS SFT reference audio must be 24 kHz. Resample it "
                "during dataset preparation.")
        with torch.inference_mode():
            return qwen3_tts_speaker_mel(audio)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        required = ("audio_codes", "text", "ref_audio")
        missing = [name for name in required if name not in record]
        if missing:
            raise ValueError(f"Qwen3-TTS record {index} is missing: {', '.join(missing)}.")

        text = record["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Qwen3-TTS record {index} requires non-empty `text`.")
        tokenized = self.processor(
            text=self._assistant_prompt(text),
            return_tensors="pt",
            padding=True,
        )
        text_ids = tokenized["input_ids"]
        if text_ids.ndim == 1:
            text_ids = text_ids.unsqueeze(0)
        if text_ids.shape[1] <= 5:
            raise ValueError(f"Qwen3-TTS record {index} produced an invalid assistant prompt.")

        audio_codes = torch.as_tensor(record["audio_codes"], dtype=torch.long)
        if (audio_codes.ndim != 2 or audio_codes.shape[0] == 0 or audio_codes.shape[-1] != 16):
            raise ValueError(
                "Qwen3-TTS 12 Hz audio_codes must have shape [frames, 16], "
                f"received {tuple(audio_codes.shape)} for record {index}.")
        codebook_size = int(self.config.talker_config.code_predictor_config.vocab_size)
        if bool(((audio_codes < 0) | (audio_codes >= codebook_size)).any()):
            raise ValueError(
                f"Qwen3-TTS record {index} contains audio codes outside "
                f"[0, {codebook_size}).")

        reference, sample_rate = self._load_audio(str(record["ref_audio"]))
        return {
            "text_ids": text_ids[:, :-5].contiguous(),
            "audio_codes": audio_codes.contiguous(),
            "ref_mel": self._extract_reference_mel(
                reference,
                sample_rate,
            ),
        }

    def collate_fn(self, batch: list[Mapping[str, Any]]) -> dict[str, Any]:
        """Apply Qwen's two-channel prompt and 16-codebook delay layout."""
        if not batch:
            raise ValueError("Cannot collate an empty Qwen3-TTS batch.")
        item_lengths = [item["text_ids"].shape[1] + item["audio_codes"].shape[0] for item in batch]
        max_length = max(item_lengths) + 8
        batch_size = len(batch)
        codebook_count = 16

        input_ids = torch.zeros(
            (batch_size, max_length, 2),
            dtype=torch.long,
        )
        codec_ids = torch.zeros(
            (batch_size, max_length, codebook_count),
            dtype=torch.long,
        )
        text_embedding_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.bool,
        )
        codec_embedding_mask = torch.zeros_like(text_embedding_mask)
        codec_mask = torch.zeros_like(text_embedding_mask)
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
        )
        codec_0_labels = torch.full(
            (batch_size, max_length),
            -100,
            dtype=torch.long,
        )

        talker = self.config.talker_config
        codec_prefix = torch.tensor(
            [
                talker.codec_nothink_id,
                talker.codec_think_bos_id,
                talker.codec_think_eos_id,
                0,
                talker.codec_pad_id,
            ],
            dtype=torch.long,
        )
        for row, item in enumerate(batch):
            text_ids = item["text_ids"]
            audio_codes = item["audio_codes"]
            text_length = int(text_ids.shape[1])
            codec_length = int(audio_codes.shape[0])
            if audio_codes.shape[-1] != codebook_count:
                raise ValueError("Every Qwen3-TTS batch item must have 16 codebooks.")

            input_ids[row, :3, 0] = text_ids[0, :3]
            input_ids[row, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[row, 7, 0] = self.config.tts_bos_token_id
            input_ids[row, 8:8 + text_length - 3, 0] = text_ids[0, 3:]
            input_ids[row, 8 + text_length - 3, 0] = self.config.tts_eos_token_id
            input_ids[
                row,
                8 + text_length - 2:8 + text_length + codec_length,
                0,
            ] = self.config.tts_pad_token_id
            text_embedding_mask[row, :8 + text_length + codec_length] = True

            input_ids[row, 3:8, 1] = codec_prefix
            input_ids[row, 8:8 + text_length - 2, 1] = talker.codec_pad_id
            input_ids[row, 8 + text_length - 2, 1] = talker.codec_bos_id
            codec_start = 8 + text_length - 1
            codec_end = codec_start + codec_length
            input_ids[row, codec_start:codec_end, 1] = audio_codes[:, 0]
            input_ids[row, codec_end, 1] = talker.codec_eos_token_id

            codec_0_labels[row, codec_start:codec_end] = audio_codes[:, 0]
            codec_0_labels[row, codec_end] = talker.codec_eos_token_id
            codec_ids[row, codec_start:codec_end, :] = audio_codes
            codec_embedding_mask[row, 3:codec_end + 1] = True
            codec_embedding_mask[row, 6] = False
            codec_mask[row, codec_start:codec_end] = True
            attention_mask[row, :codec_end + 1] = 1

        reference_mels = [item["ref_mel"] for item in batch]
        reference_lengths = {int(mel.shape[1]) for mel in reference_mels}
        if len(reference_lengths) != 1:
            raise ValueError(
                "Qwen3-TTS reference mels must have equal lengths within a "
                "batch. Use the same reference audio or bucket by length.")
        return {
            "input_ids": input_ids,
            "ref_mels": torch.cat(reference_mels, dim=0),
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
        }
