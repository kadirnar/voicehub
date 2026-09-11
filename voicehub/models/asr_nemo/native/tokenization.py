"""Dependency-free character tokenization and CTC decoding for QuartzNet."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from voicehub.models.asr_nemo.native.configuration import QUARTZNET15X5_VOCABULARY

_WHITESPACE = re.compile(r"\s+")
_APOSTROPHES = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u02bc": "'",
    "\uff07": "'",
})


@dataclass(frozen=True, slots=True)
class CTCCharacterSpan:
    token: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class CTCWordSpan:
    word: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class CTCDecodedText:
    text: str
    characters: tuple[CTCCharacterSpan, ...]
    words: tuple[CTCWordSpan, ...]


class NeMoCharacterTokenizer:
    """Strict character tokenizer with NeMo's trailing CTC blank."""

    def __init__(
        self,
        vocabulary: tuple[str, ...] = QUARTZNET15X5_VOCABULARY,
    ) -> None:
        self.vocabulary = tuple(vocabulary)
        if len(self.vocabulary) < 2:
            raise ValueError("`vocabulary` must contain at least two characters.")
        if any(not isinstance(token, str) or len(token) != 1 for token in self.vocabulary):
            raise ValueError("Every vocabulary entry must be one character.")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("`vocabulary` cannot contain duplicate characters.")
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        self.blank_id = len(self.vocabulary)

    @staticmethod
    def normalize(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Transcription text must be a string.")
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.translate(_APOSTROPHES).lower()
        return _WHITESPACE.sub(" ", normalized).strip()

    def encode(
        self,
        text: str,
        *,
        reject_unknown: bool = True,
    ) -> tuple[int, ...]:
        normalized = self.normalize(text)
        if not normalized:
            raise ValueError("A CTC transcription cannot be empty.")
        unknown = sorted(set(normalized) - self.token_to_id.keys())
        if unknown and reject_unknown:
            printable = ", ".join(repr(value) for value in unknown)
            raise ValueError("The QuartzNet character vocabulary cannot encode: "
                             f"{printable}.")
        return tuple(self.token_to_id[character] for character in normalized if character in self.token_to_id)

    def decode_ctc(self, token_ids: list[int] | tuple[int, ...]) -> CTCDecodedText:
        characters: list[CTCCharacterSpan] = []
        previous = self.blank_id
        for offset, raw_token_id in enumerate(token_ids):
            if isinstance(raw_token_id, bool) or not isinstance(raw_token_id, int):
                raise TypeError("CTC token IDs must be integers.")
            token_id = int(raw_token_id)
            if not 0 <= token_id <= self.blank_id:
                raise ValueError(f"CTC token ID {token_id} is outside the vocabulary.")
            if token_id != self.blank_id and token_id != previous:
                characters.append(
                    CTCCharacterSpan(
                        token=self.vocabulary[token_id],
                        start_offset=offset,
                        end_offset=offset + 1,
                    ))
            previous = token_id

        words: list[CTCWordSpan] = []
        current: list[CTCCharacterSpan] = []
        for character in characters:
            if character.token == " ":
                if current:
                    words.append(
                        CTCWordSpan(
                            word="".join(item.token for item in current),
                            start_offset=current[0].start_offset,
                            end_offset=current[-1].end_offset,
                        ))
                    current = []
                continue
            current.append(character)
        if current:
            words.append(
                CTCWordSpan(
                    word="".join(item.token for item in current),
                    start_offset=current[0].start_offset,
                    end_offset=current[-1].end_offset,
                ))
        return CTCDecodedText(
            text=" ".join(word.word for word in words),
            characters=tuple(characters),
            words=tuple(words),
        )


__all__ = [
    "CTCCharacterSpan",
    "CTCDecodedText",
    "CTCWordSpan",
    "NeMoCharacterTokenizer",
]
