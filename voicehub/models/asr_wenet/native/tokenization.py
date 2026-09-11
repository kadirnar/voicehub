"""Tokenizer bridge for the released GigaSpeech U2++ symbol table."""

from __future__ import annotations

import re
from pathlib import Path

from voicehub.tokenization import SentencePieceUnigramTokenizer
from voicehub.tokenization.assets import read_bounded_asset

_WHITESPACE = "\u2581"
_CJK_CHARACTER = re.compile(r"([\u4e00-\u9fa5])")


def _gigaspeech_bpe_preprocess(text: str) -> str:
    """Reproduce the released GigaSpeech word-boundary normalization."""
    text = re.sub(r"[a-z]", lambda match: match.group(0).upper(), text)
    text = re.sub(r"([A-Z])[ ]+", rf"\1{_WHITESPACE}", text)
    text = re.sub(rf"([^A-Z]){_WHITESPACE}", r"\1 ", text)
    text = re.sub(rf"{_WHITESPACE}([^A-Z])", r" \1", text)
    text = re.sub(rf"{_WHITESPACE}$", "", text)
    return (text.replace(" ", "").replace("\xEF\xBB\xBF", "").replace("\ufeff", ""))


class WeNetGigaSpeechTokenizer:
    """Map SentencePiece output pieces onto WeNet's sorted unit IDs."""

    def __init__(
        self,
        sentencepiece: SentencePieceUnigramTokenizer,
        units: tuple[str, ...],
    ) -> None:
        if len(units) < 4 or len(set(units)) != len(units):
            raise ValueError("WeNet units must be a non-empty unique sequence.")
        if units[0] != "<blank>" or units[1] != "<unk>":
            raise ValueError("WeNet units must begin with <blank> and <unk>.")
        if units[-1] != "<sos/eos>":
            raise ValueError("WeNet units must end with <sos/eos>.")
        self.sentencepiece = sentencepiece
        self.units = units
        self._piece_to_id = {piece: index for index, piece in enumerate(units)}
        self.blank_token_id = 0
        self.unknown_token_id = 1
        self.sos_eos_token_id = len(units) - 1

    @classmethod
    def from_files(
        cls,
        model_path: str | Path,
        units_path: str | Path,
    ) -> WeNetGigaSpeechTokenizer:
        sentencepiece = SentencePieceUnigramTokenizer.from_model_file(model_path)
        payload = read_bounded_asset(units_path, max_bytes=4 * 1024 * 1024)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("WeNet units are not valid UTF-8.") from error
        values: dict[int, str] = {}
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                piece, raw_index = line.rsplit(maxsplit=1)
                index = int(raw_index)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid WeNet unit at line {line_number}.") from error
            if index < 0 or index in values:
                raise ValueError(f"Invalid or duplicate WeNet unit ID {index}.")
            values[index] = piece
        expected = set(range(len(values)))
        if set(values) != expected:
            raise ValueError("WeNet unit IDs must be contiguous from zero.")
        return cls(
            sentencepiece,
            tuple(values[index] for index in range(len(values))),
        )

    @property
    def vocabulary_size(self) -> int:
        return len(self.units)

    def encode_as_pieces(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")
        normalized = _gigaspeech_bpe_preprocess(text)
        segments = [value for value in _CJK_CHARACTER.split(normalized) if value.strip()]
        pieces = []
        for segment in segments:
            for word in segment.strip().split(_WHITESPACE):
                if word.encode("utf-8").isalpha():
                    pieces.extend(self.sentencepiece.encode_as_pieces(word))
                else:
                    pieces.append(word)
        return pieces

    def encode_as_ids(self, text: str) -> list[int]:
        return [self._piece_to_id.get(piece, self.unknown_token_id) for piece in self.encode_as_pieces(text)]

    def decode_ids(self, token_ids: list[int] | tuple[int, ...]) -> str:
        pieces = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("Token IDs must be integers.")
            if token_id < 0 or token_id >= len(self.units):
                raise ValueError(f"Token ID {token_id} is outside the vocabulary.")
            if token_id in {self.blank_token_id, self.sos_eos_token_id}:
                continue
            piece = self.units[token_id]
            if piece == "<unk>":
                piece = "\u2047"
            pieces.append(piece)
        return "".join(pieces).replace(_WHITESPACE, " ").strip()


__all__ = ["WeNetGigaSpeechTokenizer"]
