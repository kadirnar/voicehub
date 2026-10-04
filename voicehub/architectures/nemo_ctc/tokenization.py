"""Dependency-free character tokenization and CTC decoding for QuartzNet."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from voicehub.architectures.nemo_ctc.configuration import QUARTZNET15X5_VOCABULARY

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
        # NeMo `extract_punctuation_from_vocab`: Unicode punctuation labels
        # (the apostrophe for QuartzNet15x5) attach to the preceding word.
        self.punctuation = frozenset(
            token for token in self.vocabulary if unicodedata.category(token).startswith("P"))
        self._space_before_punctuation = re.compile(
            r"(\s)(" + "|".join(re.escape(token) for token in sorted(self.punctuation)) + ")")

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
        """Greedy-collapse frame labels like NeMo's character CTC decoding.

        Text, punctuation spacing, and character/word frame offsets
        follow ``CTCDecoding`` with ``compute_timestamps`` (NeMo 2.5): a
        character spans from the previous emission frame (one frame
        before the first emission for the first character) to its own
        emission frame, and words are grouped by ``get_words_offsets``.
        """
        emissions: list[tuple[int, int]] = []
        previous = self.blank_id
        for offset, raw_token_id in enumerate(token_ids):
            if isinstance(raw_token_id, bool) or not isinstance(raw_token_id, int):
                raise TypeError("CTC token IDs must be integers.")
            token_id = int(raw_token_id)
            if not 0 <= token_id <= self.blank_id:
                raise ValueError(f"CTC token ID {token_id} is outside the vocabulary.")
            if token_id != self.blank_id and token_id != previous:
                emissions.append((token_id, offset))
            previous = token_id

        characters: list[CTCCharacterSpan] = []
        for index, (token_id, offset) in enumerate(emissions):
            token = self.vocabulary[token_id]
            start = emissions[index - 1][1] if index else max(0, offset - 1)
            # NeMo `_refine_timestamps`: punctuation after the first
            # character ends where it starts.
            end = start if index and token in self.punctuation else offset
            characters.append(CTCCharacterSpan(token=token, start_offset=start, end_offset=end))
        text = "".join(character.token for character in characters)
        if self.punctuation:
            text = self._space_before_punctuation.sub(r"\2", text)
        return CTCDecodedText(
            text=_WHITESPACE.sub(" ", text).strip(),
            characters=tuple(characters),
            words=self._words(characters),
        )

    def _words(self, characters: list[CTCCharacterSpan]) -> tuple[CTCWordSpan, ...]:
        """Port of NeMo ``get_words_offsets`` for a character vocabulary."""
        words: list[list] = []
        built: list[str] = []
        first_index = 0
        for index, character in enumerate(characters):
            token = character.token
            is_punctuation = token in self.punctuation and token != " "
            following = None
            cursor = index
            while not following and cursor < len(characters) - 1:
                cursor += 1
                following = characters[cursor].token
                following = following if following != " " else None
            if token == " " and following not in self.punctuation and not is_punctuation:
                if built:
                    words.append([
                        "".join(built),
                        characters[first_index].start_offset,
                        characters[index - 1].end_offset,
                    ])
                built = []
            elif is_punctuation and not built and words:
                words[-1][2] = character.end_offset
                if words[-1][0].endswith(" "):
                    words[-1][0] = words[-1][0][:-1]
                words[-1][0] += token
            elif is_punctuation and built:
                if built[-1] == " ":
                    built.pop()
                built.append(token)
            else:
                if not built:
                    first_index = index
                built.append(token)
        if words:
            words[0][1] = characters[0].start_offset
            if built:
                words.append([
                    "".join(built),
                    characters[first_index].start_offset,
                    characters[-1].end_offset,
                ])
        elif built:
            words.append([
                "".join(built),
                characters[0].start_offset,
                characters[-1].end_offset,
            ])
        return tuple(CTCWordSpan(word=word, start_offset=start, end_offset=end) for word, start, end in words)


__all__ = [
    "CTCCharacterSpan",
    "CTCDecodedText",
    "CTCWordSpan",
    "NeMoCharacterTokenizer",
]
