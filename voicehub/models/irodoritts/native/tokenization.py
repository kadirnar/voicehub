"""Exact, dependency-free tokenizer runtime for Irodori-TTS."""

from __future__ import annotations

import json
import math
import shutil
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voicehub.tokenization.assets import read_bounded_asset

_EXPECTED_SPECIALS = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<MASK|LLM-jp>": 3,
    "<PAD|LLM-jp>": 4,
    "<CLS|LLM-jp>": 5,
    "<SEP|LLM-jp>": 6,
    "<EOD|LLM-jp>": 7,
}
_WHITESPACE_MARKER = "\u2581"


def _read_json(path: Path) -> tuple[bytes, Mapping[str, Any]]:
    payload = read_bounded_asset(path, max_bytes=16 * 1024 * 1024)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"Invalid Irodori tokenizer JSON: {error}.") from error
    if not isinstance(value, Mapping):
        raise ValueError("Irodori tokenizer JSON must contain an object.")
    return payload, value


def _validate_pipeline(document: Mapping[str, Any]) -> None:
    normalizer = document.get("normalizer")
    if not isinstance(normalizer, Mapping) or normalizer.get("type") != "Sequence":
        raise ValueError("Irodori requires the released two-stage tokenizer normalizer.")
    stages = normalizer.get("normalizers")
    if not isinstance(stages, list) or len(stages) != 2:
        raise ValueError("Irodori tokenizer normalizer must contain exactly two stages.")
    expected_patterns = ("(?<!\\n)^", " ")
    for stage, pattern in zip(stages, expected_patterns):
        if (not isinstance(stage, Mapping) or stage.get("type") != "Replace" or
                stage.get("content") != _WHITESPACE_MARKER):
            raise ValueError("Irodori tokenizer Replace normalizer is unsupported.")
        raw_pattern = stage.get("pattern")
        if not isinstance(raw_pattern, Mapping) or raw_pattern.get("Regex") != pattern:
            raise ValueError("Irodori tokenizer regex normalizer differs from the release.")
    if document.get("pre_tokenizer") is not None:
        raise ValueError("Irodori's released tokenizer has no pre-tokenizer.")
    decoder = document.get("decoder")
    if not isinstance(decoder, Mapping) or decoder.get("type") != "Sequence":
        raise ValueError("Irodori requires the released byte-fallback decoder.")
    expected_decoders = [
        {
            "type": "ByteFallback",
        },
        {
            "type": "Replace",
            "pattern": {
                "Regex": _WHITESPACE_MARKER,
            },
            "content": " ",
        },
        {
            "type": "Fuse",
        },
        {
            "type": "Replace",
            "pattern": {
                "Regex": "(?<!\\n)^ ",
            },
            "content": "",
        },
    ]
    if decoder.get("decoders") != expected_decoders:
        raise ValueError("Irodori tokenizer decoder pipeline differs from the release.")


class IrodoriTokenizer:
    """99,574-piece llm-jp Unigram tokenizer with UTF-8 byte fallback."""

    def __init__(
        self,
        vocabulary: tuple[tuple[str, float], ...],
        *,
        tokenizer_json_path: Path,
        tokenizer_config_path: Path | None = None,
        max_input_chars: int = 1_000_000,
    ) -> None:
        if not vocabulary:
            raise ValueError("Irodori tokenizer vocabulary cannot be empty.")
        if (isinstance(max_input_chars, bool) or not isinstance(max_input_chars, int) or
                max_input_chars <= 0):
            raise ValueError("`max_input_chars` must be a positive integer.")
        self._vocabulary = vocabulary
        self._piece_to_id = MappingProxyType({
            piece: token_id
            for token_id, (piece, _) in enumerate(vocabulary)
        })
        if len(self._piece_to_id) != len(vocabulary):
            raise ValueError("Irodori tokenizer vocabulary contains duplicate pieces.")
        self.tokenizer_json_path = tokenizer_json_path
        self.tokenizer_config_path = tokenizer_config_path
        self.max_input_chars = int(max_input_chars)
        self.unk_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 4
        self._byte_ids = {}
        for value in range(256):
            spelling = f"<0x{value:02X}>"
            token_id = self._piece_to_id.get(spelling)
            if token_id is None:
                raise ValueError(f"Irodori tokenizer is missing byte token {spelling}.")
            self._byte_ids[value] = token_id
        trie: dict[str | None, Any] = {}
        for token_id, (piece, _) in enumerate(vocabulary):
            if token_id < 8 or piece.startswith("<0x"):
                continue
            node = trie
            for character in piece:
                node = node.setdefault(character, {})
            node.setdefault(None, []).append(token_id)
        self._trie = trie
        self._minimum_score = min(score for _, score in vocabulary)

    @classmethod
    def from_files(
        cls,
        tokenizer_json: str | Path,
        *,
        tokenizer_config: str | Path | None = None,
        expected_vocabulary_size: int | None = 99_574,
    ) -> IrodoriTokenizer:
        path = Path(tokenizer_json).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Irodori tokenizer was not found: {path}.")
        _, document = _read_json(path)
        _validate_pipeline(document)
        model = document.get("model")
        if (not isinstance(model, Mapping) or model.get("type") != "Unigram" or model.get("unk_id") != 0 or
                model.get("byte_fallback") is not True):
            raise ValueError("Irodori requires its released byte-fallback Unigram model.")
        raw_vocabulary = model.get("vocab")
        if not isinstance(raw_vocabulary, list):
            raise ValueError("Irodori tokenizer vocabulary must be a list.")
        vocabulary = []
        for record in raw_vocabulary:
            if (not isinstance(record, list) or len(record) != 2 or not isinstance(record[0], str) or
                    not record[0] or isinstance(record[1], bool) or not isinstance(record[1], (int, float)) or
                    not math.isfinite(float(record[1]))):
                raise ValueError("Irodori tokenizer vocabulary contains an invalid record.")
            vocabulary.append((record[0], float(record[1])))
        if (expected_vocabulary_size is not None and len(vocabulary) != expected_vocabulary_size):
            raise ValueError(
                "Irodori tokenizer/model vocabulary mismatch: "
                f"{len(vocabulary)} != {expected_vocabulary_size}.")
        for spelling, expected_id in _EXPECTED_SPECIALS.items():
            if expected_id >= len(vocabulary) or vocabulary[expected_id][0] != spelling:
                raise ValueError(f"Irodori tokenizer requires {spelling!r} at ID {expected_id}.")
        config_path = (None if tokenizer_config is None else Path(tokenizer_config).expanduser().resolve())
        if config_path is not None:
            _, configuration = _read_json(config_path)
            if configuration.get("tokenizer_class") not in {
                    "PreTrainedTokenizerFast",
                    "LlamaTokenizerFast",
            }:
                raise ValueError("Irodori tokenizer configuration has an unsupported class.")
        return cls(
            tuple(vocabulary),
            tokenizer_json_path=path,
            tokenizer_config_path=config_path,
        )

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocabulary)

    @property
    def vocabulary(self) -> Mapping[str, int]:
        return self._piece_to_id

    def id_to_piece(self, token_id: int) -> str:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("Irodori token IDs must be integers.")
        if not 0 <= token_id < self.vocabulary_size:
            raise ValueError(f"Irodori token ID {token_id} is outside the vocabulary.")
        return self._vocabulary[token_id][0]

    def _matches(self, text: str, start: int) -> tuple[tuple[int, int], ...]:
        node = self._trie
        matches = []
        for position in range(start, len(text)):
            child = node.get(text[position])
            if child is None:
                break
            node = child
            for token_id in node.get(None, ()):
                matches.append((position + 1, token_id))
        return tuple(matches)

    @staticmethod
    def _add_float32(left: float, right: float) -> float:
        return struct.unpack("<f", struct.pack("<f", left + right))[0]

    def _tokenize_normalized(self, text: str) -> tuple[int, ...]:
        if not text:
            return ()
        size = len(text)
        best = [float("-inf")] * (size + 1)
        previous: list[tuple[int, int] | None] = [None] * (size + 1)
        best[0] = 0.0
        unknown_score = self._minimum_score - 10.0
        for start in range(size):
            if best[start] == float("-inf"):
                continue
            matches = self._matches(text, start)
            for end, token_id in matches:
                score = self._add_float32(best[start], self._vocabulary[token_id][1])
                if score > best[end]:
                    best[end] = score
                    previous[end] = (start, token_id)
            if not any(end == start + 1 for end, _ in matches):
                score = self._add_float32(best[start], unknown_score)
                if score > best[start + 1]:
                    best[start + 1] = score
                    previous[start + 1] = (start, self.unk_token_id)
        if previous[size] is None:
            raise RuntimeError("Irodori Unigram graph could not cover normalized text.")
        nodes = []
        position = size
        while position:
            edge = previous[position]
            if edge is None:
                raise RuntimeError("Irodori Unigram Viterbi backtrace is incomplete.")
            start, token_id = edge
            nodes.append((token_id, text[start:position]))
            position = start
        output = []
        for token_id, surface in reversed(nodes):
            if token_id == self.unk_token_id:
                output.extend(self._byte_ids[value] for value in surface.encode("utf-8"))
            else:
                output.append(token_id)
        return tuple(output)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        max_length: int | None = None,
    ) -> tuple[int, ...]:
        if not isinstance(text, str) or not text:
            raise ValueError("Irodori text must be a non-empty string.")
        if len(text) > self.max_input_chars:
            raise ValueError(f"Input contains more than {self.max_input_chars} characters.")
        normalized = _WHITESPACE_MARKER + text.replace(" ", _WHITESPACE_MARKER)
        token_ids = self._tokenize_normalized(normalized)
        if add_bos:
            token_ids = (self.bos_token_id, *token_ids)
        if max_length is not None:
            if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
                raise ValueError("`max_length` must be a positive integer.")
            token_ids = token_ids[:max_length]
        if not token_ids:
            raise ValueError("Irodori text produced no tokenizer IDs.")
        return tuple(token_ids)

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_bos: bool = True,
        max_length: int | None = None,
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[bool, ...], ...]]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("Irodori tokenizer batch input must be a sequence of strings.")
        if not texts:
            raise ValueError("Irodori tokenizer batch input cannot be empty.")
        rows = tuple(self.encode(text, add_bos=add_bos, max_length=max_length) for text in texts)
        maximum = max((len(row) for row in rows), default=0)
        ids = []
        masks = []
        for row in rows:
            padding = maximum - len(row)
            ids.append(row + (self.pad_token_id, ) * padding)
            masks.append((True, ) * len(row) + (False, ) * padding)
        return tuple(ids), tuple(masks)

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        pieces: list[str] = []
        pending_bytes = bytearray()

        def flush_bytes() -> None:
            if pending_bytes:
                pieces.append(bytes(pending_bytes).decode("utf-8", errors="replace"))
                pending_bytes.clear()

        for token_id in token_ids:
            piece = self.id_to_piece(token_id)
            if piece.startswith("<0x") and piece.endswith(">"):
                pending_bytes.append(int(piece[3:5], 16))
                continue
            flush_bytes()
            if skip_special_tokens and token_id < 8:
                continue
            pieces.append(piece)
        flush_bytes()
        decoded = "".join(pieces).replace(_WHITESPACE_MARKER, " ")
        return decoded[1:] if decoded.startswith(" ") else decoded

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        for source in (self.tokenizer_json_path, self.tokenizer_config_path):
            if source is None:
                continue
            target = destination / source.name
            if target.resolve() != source:
                shutil.copy2(source, target)
        return destination


__all__ = ["IrodoriTokenizer"]
