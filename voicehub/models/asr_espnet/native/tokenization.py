"""Native SentencePiece-to-ESPnet vocabulary bridge."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from voicehub.tokenization import SentencePieceUnigramTokenizer
from voicehub.tokenization.assets import read_bounded_asset

_WHITESPACE = "\u2581"
_UNKNOWN_SURFACE = "\u2047"


def load_espnet_token_list(
    path: str | Path,
    *,
    maximum_bytes: int = 2 * 1024 * 1024,
    maximum_tokens: int = 100_000,
) -> tuple[str, ...]:
    """Read a bounded one-token-per-line ESPnet vocabulary."""
    payload = read_bounded_asset(path, max_bytes=maximum_bytes)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("ESPnet token list is not valid UTF-8.") from error
    tokens = tuple(text.splitlines())
    if not tokens or len(tokens) > maximum_tokens:
        raise ValueError("ESPnet token list has an invalid number of entries.")
    if any(not token for token in tokens):
        raise ValueError("ESPnet token list cannot contain empty entries.")
    if len(set(tokens)) != len(tokens):
        raise ValueError("ESPnet token list contains duplicate entries.")
    return tokens


class ESPnetLibriSpeechTokenizer:
    """Map native SentencePiece pieces to the recipe-specific token order."""

    def __init__(
        self,
        sentencepiece: SentencePieceUnigramTokenizer,
        tokens: tuple[str, ...],
        *,
        strict_release: bool = True,
    ) -> None:
        if not isinstance(sentencepiece, SentencePieceUnigramTokenizer):
            raise TypeError("`sentencepiece` must use VoiceHub's native runtime.")
        if not isinstance(tokens, tuple):
            tokens = tuple(tokens)
        if len(tokens) < 4 or len(tokens) != len(set(tokens)):
            raise ValueError("ESPnet tokens must be a unique non-empty sequence.")
        if tokens[0] != "<blank>" or tokens[1] != "<unk>":
            raise ValueError("ESPnet tokens must start with <blank> and <unk>.")
        if tokens[-1] != "<sos/eos>":
            raise ValueError("ESPnet tokens must end with <sos/eos>.")
        self.sentencepiece = sentencepiece
        self.tokens = tokens
        self._piece_to_id = {piece: token_id for token_id, piece in enumerate(tokens)}
        self.blank_token_id = 0
        self.unknown_token_id = 1
        self.sos_eos_token_id = len(tokens) - 1
        if strict_release:
            self._validate_release()

    @classmethod
    def from_files(
        cls,
        tokenizer_model: str | Path,
        token_list: str | Path,
        *,
        strict_release: bool = True,
    ) -> ESPnetLibriSpeechTokenizer:
        return cls(
            SentencePieceUnigramTokenizer.from_model_file(tokenizer_model),
            load_espnet_token_list(token_list),
            strict_release=strict_release,
        )

    def _validate_release(self) -> None:
        if len(self.tokens) != 5_000:
            raise ValueError("The audited ESPnet LibriSpeech release requires 5,000 tokens.")
        if self.sentencepiece.vocabulary_size != 5_000:
            raise ValueError("The audited ESPnet tokenizer requires 5,000 SentencePiece entries.")
        sentencepiece_values = {
            self.sentencepiece.id_to_piece(index)
            for index in range(self.sentencepiece.vocabulary_size)
        }
        expected_values = set(self.tokens)
        if (expected_values - sentencepiece_values != {"<blank>", "<sos/eos>"} or
                sentencepiece_values - expected_values != {"<s>", "</s>"}):
            raise ValueError(
                "ESPnet token list and SentencePiece vocabulary do not "
                "describe the audited remapping.")

    @property
    def vocabulary_size(self) -> int:
        return len(self.tokens)

    def id_to_piece(self, token_id: int) -> str:
        if (isinstance(token_id, bool) or not isinstance(token_id, int) or
                not 0 <= token_id < len(self.tokens)):
            raise ValueError("ESPnet token ID is outside the vocabulary.")
        return self.tokens[token_id]

    def piece_to_id(self, piece: str) -> int:
        if not isinstance(piece, str):
            raise TypeError("ESPnet pieces must be strings.")
        return self._piece_to_id.get(piece, self.unknown_token_id)

    @staticmethod
    def normalize_transcript(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("ESPnet transcripts must be strings.")
        return " ".join(text.split()).upper()

    def encode_as_pieces(self, text: str) -> tuple[str, ...]:
        normalized = self.normalize_transcript(text)
        return tuple(self.sentencepiece.encode_as_pieces(normalized))

    def encode_as_ids(self, text: str) -> tuple[int, ...]:
        return tuple(
            self._piece_to_id.get(piece, self.unknown_token_id) for piece in self.encode_as_pieces(text))

    def decode_ids(self, token_ids: Iterable[int]) -> str:
        pieces = []
        for raw_id in token_ids:
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise TypeError("ESPnet token IDs must be integers.")
            piece = self.id_to_piece(raw_id)
            if raw_id in {self.blank_token_id, self.sos_eos_token_id}:
                continue
            pieces.append(_UNKNOWN_SURFACE if raw_id == self.unknown_token_id else piece)
        return "".join(pieces).replace(_WHITESPACE, " ").strip()

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        tokenizer_filename: str = "tokenizer.model",
        tokens_filename: str = "tokens.txt",
    ) -> tuple[Path, Path]:
        for name, value in (
            ("tokenizer_filename", tokenizer_filename),
            ("tokens_filename", tokens_filename),
        ):
            if not isinstance(value, str) or not value or Path(value).name != value:
                raise ValueError(f"`{name}` must be a plain non-empty filename.")
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer_path = self.sentencepiece.save_pretrained(
            destination,
            filename=tokenizer_filename,
        )
        tokens_path = destination / tokens_filename
        tokens_path.write_text(
            "\n".join(self.tokens) + "\n",
            encoding="utf-8",
        )
        return tokenizer_path, tokens_path


__all__ = [
    "ESPnetLibriSpeechTokenizer",
    "load_espnet_token_list",
]
