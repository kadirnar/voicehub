"""Dependency-free tokenizer for NVIDIA Parakeet TDT checkpoints."""

from __future__ import annotations

import base64
import binascii
import json
import re
import struct
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voicehub.tokenization.assets import (
    DEFAULT_MAX_ASSET_BYTES,
    DEFAULT_MAX_MERGES,
    DEFAULT_MAX_TOKEN_BYTES,
    DEFAULT_MAX_TOKENS,
    TokenizerAssetError,
    read_bounded_asset,
)
from voicehub.tokenization.base import BatchEncoding, Encoding

_MAX_TOKEN_ID = 2**31 - 1
_MULTIPLE_SPACES = re.compile(r" {2,}")
_BYTE_TOKEN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def _token_id(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TokenizerAssetError(f"{context} must be an integer.")
    if not 0 <= value <= _MAX_TOKEN_ID:
        raise TokenizerAssetError(f"{context} is outside the supported ID range.")
    return value


def _json(path: str | Path, *, max_bytes: int) -> dict[str, Any]:
    payload = read_bounded_asset(path, max_bytes=max_bytes)

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r} is forbidden.")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TokenizerAssetError(f"Tokenizer JSON contains duplicate key {key!r}.")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except TokenizerAssetError:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise TokenizerAssetError(f"Invalid tokenizer JSON: {error}.") from error
    if not isinstance(value, dict):
        raise TokenizerAssetError("Tokenizer JSON root must be an object.")
    return value


class PrecompiledCharsMap:
    """SentencePiece precompiled normalization map backed by a Darts trie."""

    def __init__(self, encoded: str) -> None:
        if not isinstance(encoded, str) or not encoded:
            raise TokenizerAssetError("Precompiled normalizer map must be non-empty base64.")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise TokenizerAssetError("Invalid base64 precompiled normalizer map.") from error
        if len(payload) < 8:
            raise TokenizerAssetError("Precompiled normalizer map is truncated.")
        trie_bytes = struct.unpack_from("<I", payload)[0]
        if (trie_bytes == 0 or trie_bytes % 4 or trie_bytes > len(payload) - 4):
            raise TokenizerAssetError("Precompiled normalizer trie length is invalid.")
        unit_count = trie_bytes // 4
        self._trie = struct.unpack_from(
            f"<{unit_count}I",
            payload,
            4,
        )
        self._normalized = payload[4 + trie_bytes:]
        if not self._normalized:
            raise TokenizerAssetError("Precompiled normalizer replacement table is empty.")

    @staticmethod
    def _offset(unit: int) -> int:
        return (unit >> 10) << ((unit & (1 << 9)) >> 6)

    @staticmethod
    def _label(unit: int) -> int:
        return unit & ((1 << 31) | 0xFF)

    @staticmethod
    def _has_leaf(unit: int) -> bool:
        return bool((unit >> 8) & 1)

    @staticmethod
    def _value(unit: int) -> int:
        return unit & ((1 << 31) - 1)

    def transform(self, value: str) -> str | None:
        """Return the first normalization mapping for a UTF-8 key."""
        encoded = value.encode("utf-8")
        node = self._offset(self._trie[0])
        replacement_offset = None
        for byte in encoded:
            node ^= byte
            if not 0 <= node < len(self._trie):
                return None
            unit = self._trie[node]
            if self._label(unit) != byte:
                return None
            node ^= self._offset(unit)
            if not 0 <= node < len(self._trie):
                return None
            if self._has_leaf(unit):
                replacement_offset = self._value(self._trie[node])
                break
        if replacement_offset is None:
            return None
        if not 0 <= replacement_offset < len(self._normalized):
            raise TokenizerAssetError("Precompiled normalizer replacement offset is invalid.")
        end = self._normalized.find(b"\0", replacement_offset)
        if end < 0:
            raise TokenizerAssetError("Precompiled normalizer replacement is not terminated.")
        try:
            return self._normalized[replacement_offset:end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise TokenizerAssetError("Precompiled normalizer replacement is not UTF-8.") from error

    def normalize(self, text: str) -> str:
        fragments = []
        for grapheme in _graphemes(text):
            transformed = (self.transform(grapheme) if len(grapheme.encode("utf-8")) < 6 else None)
            if transformed is not None:
                fragments.append(transformed)
                continue
            for character in grapheme:
                fragments.append(self.transform(character) or character)
        return "".join(fragments)


def _is_extend(character: str) -> bool:
    codepoint = ord(character)
    return (
        unicodedata.category(character) in {"Mn", "Mc", "Me"} or 0xFE00 <= codepoint <= 0xFE0F or
        0xE0100 <= codepoint <= 0xE01EF or 0x1F3FB <= codepoint <= 0x1F3FF)


def _is_regional_indicator(character: str) -> bool:
    return 0x1F1E6 <= ord(character) <= 0x1F1FF


def _graphemes(text: str) -> tuple[str, ...]:
    """Segment the grapheme cases relevant to SentencePiece normalization."""
    if not text:
        return ()
    result: list[str] = []
    current = text[0]
    regional_count = 1 if _is_regional_indicator(text[0]) else 0
    join_next = text[0] == "\u200d"
    for character in text[1:]:
        attach = (
            _is_extend(character) or current.endswith("\u200d") or character == "\u200d" or
            (current == "\r" and character == "\n") or (
                _is_regional_indicator(character) and regional_count % 2 == 1 and
                _is_regional_indicator(current[-1])) or join_next)
        if attach:
            current += character
            regional_count = (regional_count + 1 if _is_regional_indicator(character) else regional_count)
        else:
            result.append(current)
            current = character
            regional_count = 1 if _is_regional_indicator(character) else 0
        join_next = character == "\u200d"
    result.append(current)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ParakeetTokenizerAssets:
    vocabulary: Mapping[str, int]
    id_to_token: Mapping[int, str]
    merge_ranks: Mapping[tuple[str, str], int]
    added_tokens: Mapping[str, int]
    special_ids: frozenset[int]
    unk_token_id: int
    pad_token_id: int
    eos_token_id: int
    blank_token_id: int
    normalizer: PrecompiledCharsMap
    replacement: str
    original_document: Mapping[str, Any]
    original_config: Mapping[str, Any]


def load_parakeet_tokenizer(
    tokenizer_path: str | Path,
    tokenizer_config_path: str | Path,
    *,
    blank_token: str = "<blank>",
    max_bytes: int = DEFAULT_MAX_ASSET_BYTES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_merges: int = DEFAULT_MAX_MERGES,
    max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES,
) -> ParakeetTokenizerAssets:
    """Strictly parse the published Tokenizers graph."""
    document = _json(tokenizer_path, max_bytes=max_bytes)
    tokenizer_config = _json(tokenizer_config_path, max_bytes=max_bytes)
    if document.get("version") != "1.0":
        raise TokenizerAssetError("Unsupported Parakeet tokenizer version.")
    if document.get("truncation") is not None or document.get("padding") is not None:
        raise TokenizerAssetError("Tokenizer-level truncation and padding must be disabled.")
    model = document.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise TokenizerAssetError("Parakeet tokenizer must declare BPE.")
    expected_options = {
        "dropout": None,
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": True,
        "byte_fallback": True,
        "ignore_merges": False,
    }
    for name, expected in expected_options.items():
        if model.get(name) != expected:
            raise TokenizerAssetError(f"Unsupported Parakeet BPE option {name}={model.get(name)!r}.")
    raw_vocabulary = model.get("vocab")
    if not isinstance(raw_vocabulary, dict) or not raw_vocabulary:
        raise TokenizerAssetError("Parakeet BPE vocabulary must be non-empty.")
    if len(raw_vocabulary) > max_tokens:
        raise TokenizerAssetError("Parakeet vocabulary exceeds the safety limit.")
    vocabulary: dict[str, int] = {}
    id_to_token: dict[int, str] = {}
    for token, raw_id in raw_vocabulary.items():
        if (not isinstance(token, str) or len(token.encode("utf-8")) > max_token_bytes):
            raise TokenizerAssetError("Invalid Parakeet vocabulary token.")
        token_id = _token_id(raw_id, context="Vocabulary token ID")
        if token_id in id_to_token and id_to_token[token_id] != token:
            raise TokenizerAssetError(f"Parakeet vocabulary ID {token_id} is duplicated.")
        vocabulary[token] = token_id
        id_to_token[token_id] = token
    raw_merges = model.get("merges")
    if not isinstance(raw_merges, list) or len(raw_merges) > max_merges:
        raise TokenizerAssetError("Invalid or oversized Parakeet BPE merges.")
    merge_ranks: dict[tuple[str, str], int] = {}
    for index, value in enumerate(raw_merges):
        parts = value if isinstance(value, list) else str(value).split(" ")
        if (len(parts) != 2 or any(not isinstance(part, str) or not part for part in parts)):
            raise TokenizerAssetError(f"Invalid Parakeet BPE merge {index}.")
        pair = (parts[0], parts[1])
        if pair in merge_ranks or pair[0] + pair[1] not in vocabulary:
            raise TokenizerAssetError(f"Incoherent Parakeet BPE merge {index}.")
        merge_ranks[pair] = index

    raw_added = document.get("added_tokens")
    if not isinstance(raw_added, list) or len(raw_added) > max_tokens:
        raise TokenizerAssetError("Invalid or oversized Parakeet added tokens.")
    added_tokens: dict[str, int] = {}
    special_ids: set[int] = set()
    for index, record in enumerate(raw_added):
        if not isinstance(record, dict):
            raise TokenizerAssetError(f"Added token {index} must be an object.")
        content = record.get("content")
        if not isinstance(content, str) or not content:
            raise TokenizerAssetError(f"Added token {index} has invalid content.")
        if (record.get("single_word") is not False or record.get("lstrip") is not False or
                record.get("rstrip") is not False or record.get("normalized") is not False):
            raise TokenizerAssetError("Parakeet added-token matching policy is unsupported.")
        token_id = _token_id(record.get("id"), context="Added token ID")
        previous = id_to_token.get(token_id)
        if previous is not None and previous != content:
            raise TokenizerAssetError(f"Added token {content!r} conflicts with ID {token_id}.")
        id_to_token[token_id] = content
        added_tokens[content] = token_id
        if record.get("special") is True:
            special_ids.add(token_id)

    def configured_id(config_name: str, default_token: str) -> int:
        token = tokenizer_config.get(config_name, default_token)
        if not isinstance(token, str) or token not in added_tokens:
            raise TokenizerAssetError(f"Parakeet tokenizer {config_name} is missing.")
        return added_tokens[token]

    unk_token = model.get("unk_token")
    if not isinstance(unk_token, str) or unk_token not in vocabulary:
        raise TokenizerAssetError("Parakeet BPE unknown token is invalid.")
    unk_token_id = vocabulary[unk_token]
    pad_token_id = configured_id("pad_token", "<pad>")
    eos_token_id = configured_id("eos_token", "<|endoftext|>")
    if blank_token not in added_tokens:
        raise TokenizerAssetError(f"Parakeet tokenizer is missing blank token {blank_token!r}.")
    blank_token_id = added_tokens[blank_token]
    special_ids.update((unk_token_id, pad_token_id, eos_token_id, blank_token_id))

    normalizer = document.get("normalizer")
    if (not isinstance(normalizer, dict) or normalizer.get("type") != "Sequence" or
            not isinstance(normalizer.get("normalizers"), list) or len(normalizer["normalizers"]) != 3):
        raise TokenizerAssetError("Parakeet tokenizer requires its three-stage normalizer.")
    precompiled, strip, replace = normalizer["normalizers"]
    if (not isinstance(precompiled, dict) or precompiled.get("type") != "Precompiled" or
            not isinstance(precompiled.get("precompiled_charsmap"), str)):
        raise TokenizerAssetError("Invalid Parakeet precompiled normalizer.")
    if strip != {"type": "Strip", "strip_left": False, "strip_right": True}:
        raise TokenizerAssetError("Unsupported Parakeet Strip normalizer.")
    expected_replace = {
        "type": "Replace",
        "pattern": {
            "Regex": " {2,}"
        },
        "content": "▁",
    }
    if replace != expected_replace:
        raise TokenizerAssetError("Unsupported Parakeet whitespace normalizer.")
    metaspace = {
        "type": "Metaspace",
        "replacement": "▁",
        "prepend_scheme": "always",
        "split": True,
    }
    if document.get("pre_tokenizer") != metaspace:
        raise TokenizerAssetError("Unsupported Parakeet pre-tokenizer.")
    if document.get("decoder") != metaspace:
        raise TokenizerAssetError("Unsupported Parakeet decoder.")
    post = document.get("post_processor")
    if (not isinstance(post, dict) or post.get("type") != "TemplateProcessing" or
            post.get("single") != [{"Sequence": {"id": "A", "type_id": 0}}] or
            post.get("special_tokens") != {}):
        raise TokenizerAssetError("Unsupported Parakeet post-processor.")

    return ParakeetTokenizerAssets(
        vocabulary=MappingProxyType(vocabulary),
        id_to_token=MappingProxyType(id_to_token),
        merge_ranks=MappingProxyType(merge_ranks),
        added_tokens=MappingProxyType(added_tokens),
        special_ids=frozenset(special_ids),
        unk_token_id=unk_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        blank_token_id=blank_token_id,
        normalizer=PrecompiledCharsMap(precompiled["precompiled_charsmap"]),
        replacement="▁",
        original_document=MappingProxyType(document),
        original_config=MappingProxyType(tokenizer_config),
    )


class ParakeetTokenizer:
    """SentencePiece-style BPE used by Parakeet TDT v3."""

    def __init__(
        self,
        assets: ParakeetTokenizerAssets,
        *,
        max_input_chars: int = 1_000_000,
    ) -> None:
        if not isinstance(assets, ParakeetTokenizerAssets):
            raise TypeError("`assets` must be ParakeetTokenizerAssets.")
        if (isinstance(max_input_chars, bool) or not isinstance(max_input_chars, int) or max_input_chars < 1):
            raise ValueError("`max_input_chars` must be a positive integer.")
        self.assets = assets
        self.pad_token_id = assets.pad_token_id
        self.eos_token_id = assets.eos_token_id
        self.unk_token_id = assets.unk_token_id
        self.blank_token_id = assets.blank_token_id
        self.max_input_chars = max_input_chars
        self._recognized_added = tuple(sorted(
            assets.added_tokens,
            key=lambda token: (-len(token), token),
        ))

    @classmethod
    def from_files(
        cls,
        tokenizer_path: str | Path,
        tokenizer_config_path: str | Path,
        *,
        blank_token: str = "<blank>",
    ) -> ParakeetTokenizer:
        return cls(load_parakeet_tokenizer(
            tokenizer_path,
            tokenizer_config_path,
            blank_token=blank_token,
        ))

    @property
    def vocabulary_size(self) -> int:
        return len(self.assets.vocabulary)

    @property
    def token_id_space_size(self) -> int:
        return max(self.assets.id_to_token) + 1

    def _plain_segments(self, text: str) -> tuple[tuple[bool, str], ...]:
        result: list[tuple[bool, str]] = []
        start = 0
        index = 0
        while index < len(text):
            match = next(
                (token for token in self._recognized_added if text.startswith(token, index)),
                None,
            )
            if match is None:
                index += 1
                continue
            if index > start:
                result.append((False, text[start:index]))
            result.append((True, match))
            index += len(match)
            start = index
        if start < len(text) or not result:
            result.append((False, text[start:]))
        return tuple(result)

    def _normalize(self, text: str) -> str:
        normalized = self.assets.normalizer.normalize(text).rstrip()
        normalized = _MULTIPLE_SPACES.sub(self.assets.replacement, normalized)
        normalized = normalized.replace(" ", self.assets.replacement)
        if normalized.startswith(self.assets.replacement):
            return normalized
        return self.assets.replacement + normalized

    def _pretokenize(self, normalized: str) -> tuple[str, ...]:
        marker = self.assets.replacement
        if not normalized:
            return ()
        starts = [index for index, value in enumerate(normalized) if value == marker]
        if not starts:
            return (normalized, )
        result = []
        for offset, start in enumerate(starts):
            end = starts[offset + 1] if offset + 1 < len(starts) else len(normalized)
            if end > start:
                result.append(normalized[start:end])
        return tuple(result)

    def _initial_pieces(self, text: str) -> list[str]:
        pieces: list[str] = []
        unknown = self.assets.id_to_token[self.unk_token_id]
        for character in text:
            if character in self.assets.vocabulary:
                pieces.append(character)
                continue
            byte_pieces = tuple(f"<0x{value:02X}>" for value in character.encode("utf-8"))
            if all(piece in self.assets.vocabulary for piece in byte_pieces):
                pieces.extend(byte_pieces)
            elif not pieces or pieces[-1] != unknown:
                pieces.append(unknown)
        return pieces

    def _merge(self, pieces: list[str]) -> tuple[int, ...]:
        while len(pieces) > 1:
            candidate: tuple[int, int] | None = None
            for index in range(len(pieces) - 1):
                rank = self.assets.merge_ranks.get((pieces[index], pieces[index + 1]))
                if rank is not None and (candidate is None or rank < candidate[0]):
                    candidate = (rank, index)
            if candidate is None:
                break
            index = candidate[1]
            pieces[index:index + 2] = [pieces[index] + pieces[index + 1]]
        return tuple(self.assets.vocabulary.get(piece, self.unk_token_id) for piece in pieces)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> Encoding:
        del add_special_tokens  # The official identity template adds no IDs.
        if not isinstance(text, str):
            raise TypeError("Parakeet tokenizer input must be a string.")
        if len(text) > self.max_input_chars:
            raise ValueError(f"Input contains more than {self.max_input_chars} characters.")
        ids: list[int] = []
        for is_added, segment in self._plain_segments(text):
            if is_added:
                ids.append(self.assets.added_tokens[segment])
                continue
            if not segment:
                continue
            normalized = self._normalize(segment)
            for word in self._pretokenize(normalized):
                ids.extend(self._merge(self._initial_pieces(word)))
        return Encoding(
            input_ids=tuple(ids),
            attention_mask=tuple(1 for _ in ids),
        )

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
        pad: bool = False,
    ) -> BatchEncoding:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("`texts` must be a sequence of strings.")
        encodings = tuple(self.encode(text, add_special_tokens=add_special_tokens) for text in texts)
        maximum = (max((len(value.input_ids) for value in encodings), default=0) if pad else None)
        ids = []
        masks = []
        specials = []
        for value in encodings:
            padding = 0 if maximum is None else maximum - len(value.input_ids)
            ids.append(value.input_ids + (self.pad_token_id, ) * padding)
            masks.append(value.attention_mask + (0, ) * padding)
            specials.append(value.special_tokens_mask + (1, ) * padding)
        return BatchEncoding(
            input_ids=tuple(ids),
            attention_mask=tuple(masks),
            special_tokens_mask=tuple(specials),
        )

    def token_piece(self, token_id: int) -> str:
        token_id = _token_id(token_id, context="Token ID")
        try:
            return self.assets.id_to_token[token_id]
        except KeyError as error:
            raise ValueError(f"Token ID {token_id} is absent from the Parakeet tokenizer.") from error

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        pieces = []
        for token_id in token_ids:
            normalized_id = _token_id(token_id, context="Token ID")
            if skip_special_tokens and normalized_id in self.assets.special_ids:
                continue
            pieces.append(self.token_piece(normalized_id))
        fragments: list[str] = []
        buffered = bytearray()

        def flush() -> None:
            if buffered:
                fragments.append(buffered.decode("utf-8", errors="replace"))
                buffered.clear()

        for piece in pieces:
            match = _BYTE_TOKEN.match(piece)
            if match:
                buffered.append(int(match.group(1), 16))
            else:
                flush()
                fragments.append(piece)
        flush()
        text = "".join(fragments).replace(self.assets.replacement, " ")
        if text.startswith(" "):
            text = text[1:]
        return text

    def batch_decode(
        self,
        sequences: Iterable[Iterable[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return [self.decode(value, skip_special_tokens=skip_special_tokens) for value in sequences]

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer_path = destination / "tokenizer.json"
        tokenizer_config_path = destination / "tokenizer_config.json"
        tokenizer_path.write_text(
            json.dumps(
                dict(self.assets.original_document),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        tokenizer_config_path.write_text(
            json.dumps(
                dict(self.assets.original_config),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return tokenizer_path


__all__ = [
    "ParakeetTokenizer",
    "ParakeetTokenizerAssets",
    "PrecompiledCharsMap",
    "load_parakeet_tokenizer",
]
