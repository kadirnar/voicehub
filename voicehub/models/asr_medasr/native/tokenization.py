"""Dependency-free LASR Unigram tokenization and CTC decoding."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voicehub.tokenization import SentencePieceUnigramAssets, SentencePieceUnigramPiece, SentencePieceUnigramTokenizer
from voicehub.tokenization.assets import read_bounded_asset

_NORMAL = 1
_UNKNOWN = 2
_CONTROL = 3
_EXPECTED_SPECIALS = {
    "<epsilon>": 0,
    "<s>": 1,
    "</s>": 2,
    "<unk>": 3,
}


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(read_bounded_asset(path).decode("utf-8"), )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"Invalid MedASR tokenizer JSON: {error}.") from error
    if not isinstance(value, dict):
        raise ValueError("MedASR tokenizer JSON must contain an object.")
    return value


def _validate_pipeline(document: Mapping[str, Any]) -> None:
    if document.get("normalizer") is not None:
        raise ValueError("MedASR requires the released tokenizer with no normalizer.")
    pretokenizer = document.get("pre_tokenizer")
    if (not isinstance(pretokenizer, Mapping) or pretokenizer.get("type") != "Sequence"):
        raise ValueError("MedASR requires the WhitespaceSplit/Metaspace pipeline.")
    children = pretokenizer.get("pretokenizers")
    if not isinstance(children, list) or len(children) != 2:
        raise ValueError("MedASR requires exactly two pre-tokenizer stages.")
    whitespace, metaspace = children
    if (not isinstance(whitespace, Mapping) or whitespace.get("type") != "WhitespaceSplit" or
            not isinstance(metaspace, Mapping) or metaspace.get("type") != "Metaspace" or
            metaspace.get("replacement") != "\u2581" or metaspace.get("prepend_scheme") != "always" or
            metaspace.get("split") is not True):
        raise ValueError("MedASR tokenizer pre-tokenizer semantics are unsupported.")
    decoder = document.get("decoder")
    if (not isinstance(decoder, Mapping) or decoder.get("type") != "Metaspace" or
            decoder.get("replacement") != "\u2581" or decoder.get("prepend_scheme") != "always" or
            decoder.get("split") is not True):
        raise ValueError("MedASR tokenizer decoder semantics are unsupported.")


def _unigram_assets(
    document: Mapping[str, Any],
    *,
    original: bytes,
) -> SentencePieceUnigramAssets:
    model = document.get("model")
    if not isinstance(model, Mapping) or model.get("type") != "Unigram":
        raise ValueError("MedASR tokenizer must use a Unigram model.")
    if model.get("unk_id") != 3:
        raise ValueError("MedASR tokenizer unknown token must use ID 3.")
    if model.get("byte_fallback", False):
        raise ValueError("MedASR tokenizer must not enable byte fallback.")
    raw_vocabulary = model.get("vocab")
    if not isinstance(raw_vocabulary, list) or len(raw_vocabulary) < 4:
        raise ValueError("MedASR tokenizer vocabulary must contain at least four entries.")
    pieces = []
    for token_id, record in enumerate(raw_vocabulary):
        if (not isinstance(record, list) or len(record) != 2 or not isinstance(record[0], str) or
                not record[0] or isinstance(record[1], bool) or not isinstance(record[1], (int, float))):
            raise ValueError("MedASR tokenizer vocabulary contains an invalid record.")
        spelling, score = record
        expected = _EXPECTED_SPECIALS.get(spelling)
        if expected is not None and expected != token_id:
            raise ValueError(f"MedASR token {spelling!r} must use ID {expected}.")
        piece_type = (_UNKNOWN if token_id == 3 else _CONTROL if token_id < 3 else _NORMAL)
        pieces.append(SentencePieceUnigramPiece(
            spelling,
            float(score),
            piece_type,
        ))
    for spelling, token_id in _EXPECTED_SPECIALS.items():
        if (token_id >= len(pieces) or pieces[token_id].text != spelling):
            raise ValueError(f"MedASR tokenizer is missing {spelling!r} at ID "
                             f"{token_id}.")
    if len({piece.text for piece in pieces}) != len(pieces):
        raise ValueError("MedASR tokenizer vocabulary contains duplicate tokens.")
    return SentencePieceUnigramAssets(
        pieces=tuple(pieces),
        unk_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        unk_surface=" \u2047 ",
        byte_fallback=False,
        normalizer_name="identity",
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        escape_whitespaces=True,
        has_precompiled_normalizer=False,
        original_model=original,
    )


@dataclass(frozen=True, slots=True)
class MedASRTokenSpan:
    token_id: int
    text: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class MedASRDecodedText:
    text: str
    tokens: tuple[MedASRTokenSpan, ...]


class MedASRTokenizer:
    """Checkpoint-bound 512-piece Unigram tokenizer for LASR CTC."""

    def __init__(
        self,
        tokenizer: SentencePieceUnigramTokenizer,
        *,
        tokenizer_json_path: Path,
        tokenizer_config_path: Path | None = None,
    ) -> None:
        if not isinstance(tokenizer, SentencePieceUnigramTokenizer):
            raise TypeError("`tokenizer` must use VoiceHub's native Unigram runtime.")
        self._tokenizer = tokenizer
        self.tokenizer_json_path = tokenizer_json_path
        self.tokenizer_config_path = tokenizer_config_path
        self.blank_token_id = 0
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    @classmethod
    def from_files(
        cls,
        tokenizer_json: str | Path,
        *,
        tokenizer_config: str | Path | None = None,
        expected_vocabulary_size: int | None = None,
    ) -> MedASRTokenizer:
        path = Path(tokenizer_json).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MedASR tokenizer was not found: {path}.")
        payload = read_bounded_asset(path)
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid MedASR tokenizer JSON: {error}.") from error
        if not isinstance(document, Mapping):
            raise ValueError("MedASR tokenizer JSON must contain an object.")
        _validate_pipeline(document)
        tokenizer = SentencePieceUnigramTokenizer(_unigram_assets(document, original=payload), )
        if (expected_vocabulary_size is not None and tokenizer.vocabulary_size != expected_vocabulary_size):
            raise ValueError(
                "MedASR tokenizer/model vocabulary mismatch: tokenizer has "
                f"{tokenizer.vocabulary_size} entries, model expects "
                f"{expected_vocabulary_size}.")
        config_path = (
            Path(tokenizer_config).expanduser().resolve() if tokenizer_config is not None else None)
        if config_path is not None:
            values = _json_object(config_path)
            tokens = {
                "pad_token": "<epsilon>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            }
            for name, expected in tokens.items():
                if values.get(name) != expected:
                    raise ValueError(f"MedASR tokenizer `{name}` must be "
                                     f"{expected!r}.")
        return cls(
            tokenizer,
            tokenizer_json_path=path,
            tokenizer_config_path=config_path,
        )

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.vocabulary_size

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("MedASR transcripts must be non-empty strings.")
        encoded = tuple(self._tokenizer.encode(text).input_ids)
        if not encoded:
            raise ValueError("MedASR transcript produced no tokenizer IDs.")
        if any(token_id == self.blank_token_id for token_id in encoded):
            raise ValueError("MedASR transcripts cannot encode the CTC blank token.")
        return encoded

    def decode_ctc(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> MedASRDecodedText:
        groups: list[tuple[int, int, int]] = []
        for offset, raw_token_id in enumerate(token_ids):
            if isinstance(raw_token_id, bool) or not isinstance(
                    raw_token_id,
                    int,
            ):
                raise TypeError("MedASR token IDs must be integers.")
            token_id = int(raw_token_id)
            if not 0 <= token_id < self.vocabulary_size:
                raise ValueError(f"MedASR token ID {token_id} is outside the vocabulary.")
            if groups and groups[-1][0] == token_id:
                previous, start, _ = groups[-1]
                groups[-1] = (previous, start, offset + 1)
            else:
                groups.append((token_id, offset, offset + 1))
        spans = []
        decoded_ids = []
        special_ids = {
            self.blank_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }
        if skip_special_tokens:
            special_ids.add(self.unk_token_id)
        for token_id, start, end in groups:
            if token_id == self.blank_token_id:
                continue
            if token_id in special_ids:
                continue
            spelling = self._tokenizer.id_to_piece(token_id)
            decoded_ids.append(token_id)
            spans.append(
                MedASRTokenSpan(
                    token_id=token_id,
                    text=spelling,
                    start_offset=start,
                    end_offset=end,
                ))
        text = self._tokenizer.decode(
            decoded_ids,
            skip_special_tokens=skip_special_tokens,
        )
        return MedASRDecodedText(
            text=text.strip(),
            tokens=tuple(spans),
        )

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return self.decode_ctc(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        ).text

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer_target = destination / "tokenizer.json"
        if tokenizer_target.resolve() != self.tokenizer_json_path:
            shutil.copy2(self.tokenizer_json_path, tokenizer_target)
        if self.tokenizer_config_path is not None:
            config_target = destination / "tokenizer_config.json"
            if config_target.resolve() != self.tokenizer_config_path:
                shutil.copy2(
                    self.tokenizer_config_path,
                    config_target,
                )
        return destination


__all__ = [
    "MedASRDecodedText",
    "MedASRTokenSpan",
    "MedASRTokenizer",
]
