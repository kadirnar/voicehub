"""Dependency-free Qwen2 byte-BPE tokenizer assets for Qwen3-ASR."""

from __future__ import annotations

import json
import shutil
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from voicehub.tokenization.assets import (
    DEFAULT_MAX_ASSET_BYTES,
    DEFAULT_MAX_MERGES,
    DEFAULT_MAX_TOKENS,
    TokenizerAssetError,
    decode_gpt2_token,
    read_bounded_asset,
)
from voicehub.tokenization.base import BatchEncoding, Encoding
from voicehub.tokenization.byte_bpe import ByteBPETokenizer

END_OF_TEXT = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"
AUDIO_PAD = "<|audio_pad|>"
ASR_TEXT = "<asr_text>"

EXPECTED_TOKEN_IDS = {
    END_OF_TEXT: 151_643,
    IM_START: 151_644,
    IM_END: 151_645,
    AUDIO_START: 151_669,
    AUDIO_END: 151_670,
    AUDIO_PAD: 151_676,
    ASR_TEXT: 151_704,
}

_CONTRACTIONS = ("re", "ve", "ll", "s", "t", "m", "d")


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_number(character: str) -> bool:
    return unicodedata.category(character).startswith("N")


def _contraction_end(text: str, index: int) -> int | None:
    if index >= len(text) or text[index] != "'":
        return None
    lowered = text[index + 1:].lower()
    for suffix in _CONTRACTIONS:
        if lowered.startswith(suffix):
            return index + 1 + len(suffix)
    return None


def qwen2_pretokenize(text: str) -> tuple[str, ...]:
    """Implement Qwen2's Unicode pre-tokenization without ``regex``.

    In particular, numbers are emitted one Unicode number at a time.
    That differs from GPT-2's grouped-number rule and is required for
    exact Qwen2 token IDs.
    """
    if not isinstance(text, str):
        raise TypeError("`text` must be a string.")
    pieces: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        contraction_end = _contraction_end(text, index)
        if contraction_end is not None:
            pieces.append(text[index:contraction_end])
            index = contraction_end
            continue

        character = text[index]
        if _is_number(character):
            pieces.append(character)
            index += 1
            continue

        # Qwen's word alternative accepts one leading non-newline,
        # non-letter, non-number character (commonly a single space).
        word_start = index
        letter_start = index
        if (character not in "\r\n" and not _is_letter(character) and not _is_number(character) and
                index + 1 < length and _is_letter(text[index + 1])):
            letter_start += 1
        if letter_start < length and _is_letter(text[letter_start]):
            end = letter_start + 1
            while end < length and _is_letter(text[end]):
                end += 1
            pieces.append(text[word_start:end])
            index = end
            continue

        # Optional ASCII space plus a run of punctuation/symbols, with any
        # immediately following line endings.
        punctuation_start = index
        punctuation_index = index
        if (character == " " and index + 1 < length and not text[index + 1].isspace() and
                not _is_letter(text[index + 1]) and not _is_number(text[index + 1])):
            punctuation_index += 1
        if (punctuation_index < length and not text[punctuation_index].isspace() and
                not _is_letter(text[punctuation_index]) and not _is_number(text[punctuation_index])):
            end = punctuation_index + 1
            while (end < length and not text[end].isspace() and not _is_letter(text[end]) and
                   not _is_number(text[end])):
                end += 1
            while end < length and text[end] in "\r\n":
                end += 1
            pieces.append(text[punctuation_start:end])
            index = end
            continue

        # Qwen's remaining alternatives are:
        #
        #   \s*[\r\n]+ | \s+(?!\S) | \s+
        #
        # The newline branch consumes through the final line ending in one
        # contiguous whitespace run, leaving any trailing horizontal
        # whitespace for the next match.  Without a line ending, the negative
        # lookahead branch consumes all but the final whitespace character
        # when a non-whitespace character follows.  Reprocessing that final
        # character lets the earlier word branch attach tabs and other
        # Unicode whitespace to a following word exactly like the reference
        # regular expression.
        if character.isspace():
            end = index + 1
            while end < length and text[end].isspace():
                end += 1
            last_line_ending = max(
                (position for position in range(index, end) if text[position] in "\r\n"),
                default=-1,
            )
            if last_line_ending >= index:
                pieces.append(text[index:last_line_ending + 1])
                index = last_line_ending + 1
            elif end < length and end - index > 1:
                pieces.append(text[index:end - 1])
                index = end - 1
            else:
                pieces.append(text[index:end])
                index = end
            continue

        pieces.append(character)
        index += 1
    if "".join(pieces) != text:
        raise RuntimeError("Qwen2 pre-tokenization did not preserve the input.")
    return tuple(piece for piece in pieces if piece)


def _json_document(path: Path, *, max_bytes: int) -> Any:
    payload = read_bounded_asset(path, max_bytes=max_bytes)
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise TokenizerAssetError(f"Invalid tokenizer JSON in {path.name}: {error}.") from error


def _load_vocabulary(
    path: Path,
    *,
    max_bytes: int,
    max_tokens: int,
) -> dict[bytes, int]:
    document = _json_document(path, max_bytes=max_bytes)
    if not isinstance(document, dict) or not document:
        raise TokenizerAssetError("Qwen vocabulary must be a non-empty object.")
    if len(document) > max_tokens:
        raise TokenizerAssetError(f"Qwen vocabulary contains more than {max_tokens} tokens.")
    output: dict[bytes, int] = {}
    ids: set[int] = set()
    for spelling, token_id in document.items():
        if (not isinstance(spelling, str) or isinstance(token_id, bool) or not isinstance(token_id, int) or
                token_id < 0):
            raise TokenizerAssetError("Qwen vocabulary contains an invalid record.")
        token = decode_gpt2_token(spelling)
        if not token or token in output or token_id in ids:
            raise TokenizerAssetError("Qwen vocabulary contains duplicate or empty tokens.")
        output[token] = token_id
        ids.add(token_id)
    return output


def _load_merges(
    path: Path,
    *,
    max_bytes: int,
    max_merges: int,
) -> tuple[tuple[bytes, bytes], ...]:
    payload = read_bounded_asset(path, max_bytes=max_bytes)
    output: list[tuple[bytes, bytes]] = []
    seen: set[tuple[bytes, bytes]] = set()
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line or raw_line.startswith(b"#"):
            continue
        if len(output) >= max_merges:
            raise TokenizerAssetError(f"Qwen merges contain more than {max_merges} records.")
        try:
            fields = raw_line.decode("utf-8").split(" ")
        except UnicodeDecodeError as error:
            raise TokenizerAssetError(f"Invalid UTF-8 merge on line {line_number}.") from error
        if len(fields) != 2 or not all(fields):
            raise TokenizerAssetError(f"Invalid Qwen merge on line {line_number}.")
        pair = (
            decode_gpt2_token(fields[0]),
            decode_gpt2_token(fields[1]),
        )
        if pair in seen:
            raise TokenizerAssetError(f"Duplicate Qwen merge on line {line_number}.")
        output.append(pair)
        seen.add(pair)
    if not output:
        raise TokenizerAssetError("Qwen merge asset is empty.")
    return tuple(output)


def _load_added_tokens(
    path: Path,
    *,
    max_bytes: int,
) -> tuple[dict[str, int], dict[str, int], Mapping[str, Any]]:
    document = _json_document(path, max_bytes=max_bytes)
    if not isinstance(document, dict):
        raise TokenizerAssetError("Qwen tokenizer configuration must be an object.")
    records = document.get("added_tokens_decoder")
    if not isinstance(records, dict) or not records:
        raise TokenizerAssetError("Qwen tokenizer configuration has no added-token decoder.")
    special: dict[str, int] = {}
    added: dict[str, int] = {}
    for raw_id, record in records.items():
        if not isinstance(record, dict):
            raise TokenizerAssetError("Qwen added-token records must be objects.")
        try:
            token_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise TokenizerAssetError("Qwen added-token ID is invalid.") from error
        content = record.get("content")
        if not isinstance(content, str) or not content:
            raise TokenizerAssetError("Qwen added token must be non-empty.")
        if record.get("lstrip") or record.get("rstrip"):
            raise TokenizerAssetError("Whitespace-stripping Qwen added tokens are unsupported.")
        destination = special if record.get("special") is True else added
        if content in destination or token_id in destination.values():
            raise TokenizerAssetError("Qwen added tokens contain duplicates.")
        destination[content] = token_id
    for spelling, expected_id in EXPECTED_TOKEN_IDS.items():
        actual = special.get(spelling, added.get(spelling))
        if actual != expected_id:
            raise TokenizerAssetError(
                f"Qwen token {spelling!r} must use ID {expected_id}; "
                f"found {actual!r}.")
    return special, added, document


class Qwen3ASRTokenizer:
    """Qwen3-ASR tokenizer with immutable declarative source assets."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        vocab_path: Path,
        merges_path: Path,
        tokenizer_config_path: Path,
        tokenizer_config: Mapping[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        self.tokenizer_config_path = tokenizer_config_path
        self.tokenizer_config = dict(tokenizer_config)

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        tokenizer_config_path: str | Path,
        *,
        max_asset_bytes: int = DEFAULT_MAX_ASSET_BYTES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_merges: int = DEFAULT_MAX_MERGES,
    ) -> Qwen3ASRTokenizer:
        vocab = Path(vocab_path).expanduser().resolve()
        merges = Path(merges_path).expanduser().resolve()
        tokenizer_config = Path(tokenizer_config_path).expanduser().resolve()
        vocabulary = _load_vocabulary(
            vocab,
            max_bytes=max_asset_bytes,
            max_tokens=max_tokens,
        )
        merge_pairs = _load_merges(
            merges,
            max_bytes=max_asset_bytes,
            max_merges=max_merges,
        )
        special, added, values = _load_added_tokens(
            tokenizer_config,
            max_bytes=max_asset_bytes,
        )
        tokenizer = ByteBPETokenizer(
            vocabulary,
            merges=merge_pairs,
            special_tokens=special,
            added_tokens=added,
            pad_token_id=EXPECTED_TOKEN_IDS[END_OF_TEXT],
            add_prefix_space=bool(values.get("add_prefix_space", False)),
            use_regex=True,
            pretokenizer=qwen2_pretokenize,
            padding_side="left",
        )
        if tokenizer.token_id_space_size > 151_936:
            raise TokenizerAssetError("Qwen tokenizer IDs exceed the published decoder vocabulary.")
        return cls(
            tokenizer,
            vocab_path=vocab,
            merges_path=merges,
            tokenizer_config_path=tokenizer_config,
            tokenizer_config=values,
        )

    @property
    def pad_token_id(self) -> int:
        return EXPECTED_TOKEN_IDS[END_OF_TEXT]

    @property
    def eos_token_id(self) -> int:
        return EXPECTED_TOKEN_IDS[IM_END]

    @property
    def audio_token_id(self) -> int:
        return EXPECTED_TOKEN_IDS[AUDIO_PAD]

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.vocabulary_size

    @property
    def token_id_space_size(self) -> int:
        return self._tokenizer.token_id_space_size

    def encode(self, text: str) -> Encoding:
        return self._tokenizer.encode(
            text,
            allowed_special="all",
            disallowed_special=(),
        )

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        padding: bool | str = True,
    ) -> BatchEncoding:
        return self._tokenizer.encode_batch(
            texts,
            padding=padding,
            allowed_special="all",
            disallowed_special=(),
        )

    def decode(
        self,
        token_ids: Iterable[int] | Encoding,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        tolist = getattr(token_ids, "tolist", None)
        if callable(tolist):
            token_ids = tolist()
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            errors=str(self.tokenizer_config.get("errors", "replace")),
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        for source, filename in (
            (self.vocab_path, "vocab.json"),
            (self.merges_path, "merges.txt"),
            (self.tokenizer_config_path, "tokenizer_config.json"),
        ):
            destination = target / filename
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        return target


__all__ = [
    "ASR_TEXT",
    "AUDIO_END",
    "AUDIO_PAD",
    "AUDIO_START",
    "END_OF_TEXT",
    "EXPECTED_TOKEN_IDS",
    "IM_END",
    "IM_START",
    "Qwen3ASRTokenizer",
    "qwen2_pretokenize",
]
