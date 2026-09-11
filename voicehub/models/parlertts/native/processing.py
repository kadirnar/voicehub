"""Native FLAN-T5 SentencePiece frontend for Parler-TTS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from voicehub.tokenization import SentencePieceUnigramTokenizer


@dataclass(frozen=True, slots=True)
class ParlerTextBatch:
    input_ids: Tensor
    attention_mask: Tensor

    def to(self, device: torch.device | str) -> ParlerTextBatch:
        return ParlerTextBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
        )


class ParlerTextTokenizer:
    """T5 tokenizer behavior without SentencePiece or Transformers runtimes."""

    def __init__(
        self,
        sentencepiece: SentencePieceUnigramTokenizer,
        *,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
        model_vocabulary_size: int = 32_128,
    ) -> None:
        if not isinstance(sentencepiece, SentencePieceUnigramTokenizer):
            raise TypeError("`sentencepiece` must be a native tokenizer.")
        for name, value in (
            ("eos_token_id", eos_token_id),
            ("pad_token_id", pad_token_id),
            ("model_vocabulary_size", model_vocabulary_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"`{name}` must be a non-negative integer.")
        if sentencepiece.vocabulary_size > model_vocabulary_size:
            raise ValueError("SentencePiece vocabulary exceeds the T5 embedding table.")
        self.sentencepiece = sentencepiece
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.model_vocabulary_size = model_vocabulary_size

    @classmethod
    def from_model_file(
        cls,
        path: str | Path,
        **kwargs,
    ) -> ParlerTextTokenizer:
        return cls(
            SentencePieceUnigramTokenizer.from_model_file(path),
            **kwargs,
        )

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode one string and append T5's ``</s>`` template token."""
        if not isinstance(text, str):
            raise TypeError("Parler-TTS text inputs must be strings.")
        ids = tuple(self.sentencepiece.encode_as_ids(text))
        return ids + (self.eos_token_id, )

    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        device: torch.device | str | None = None,
    ) -> ParlerTextBatch:
        if isinstance(texts, str):
            source = (texts, )
        elif isinstance(texts, Sequence) and not isinstance(texts, bytes):
            source = tuple(texts)
        else:
            raise TypeError("Parler-TTS text input must be a string or sequence of strings.")
        if not source:
            raise ValueError("At least one text string is required.")
        if any(not isinstance(text, str) for text in source):
            raise TypeError("Parler-TTS text inputs must be strings.")
        rows = tuple(self.encode(text) for text in source)
        maximum = max(len(row) for row in rows)
        input_ids = torch.full(
            (len(rows), maximum),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(rows), maximum),
            dtype=torch.long,
            device=device,
        )
        for index, row in enumerate(rows):
            input_ids[index, :len(row)] = torch.tensor(
                row,
                dtype=torch.long,
                device=device,
            )
            attention_mask[index, :len(row)] = 1
        return ParlerTextBatch(input_ids, attention_mask)


__all__ = ["ParlerTextBatch", "ParlerTextTokenizer"]
