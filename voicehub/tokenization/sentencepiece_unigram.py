"""Dependency-free SentencePiece unigram model reader and tokenizer.

SentencePiece stores ``.model`` files as a small protobuf document.  Pulling
in a protobuf runtime and the SentencePiece C++ extension merely to read that
document would make model architecture code depend on two unrelated
libraries.  This module implements the bounded wire-format subset and the
unigram Viterbi decoder needed by published speech checkpoints.

Models that carry a precompiled normalization table (``nmt_nfkc`` and
``nfkc`` models normally do) are normalized with that table exactly as
SentencePiece's C++ ``Normalizer`` does: longest-prefix matches in the
Darts-clone trie, whitespace collapsing on the normalized output, and the
dummy prefix. Models without a table fall back to the Python standard
library's NFKC implementation.
"""

from __future__ import annotations

import math
import struct
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voicehub.tokenization.assets import (
    DEFAULT_MAX_ASSET_BYTES,
    DEFAULT_MAX_TOKEN_BYTES,
    DEFAULT_MAX_TOKENS,
    TokenizerAssetError,
    read_bounded_asset,
)
from voicehub.tokenization.base import BatchEncoding, Encoding

_UNKNOWN = 2
_CONTROL = 3
_UNUSED = 5
_BYTE = 6
_MODEL_UNIGRAM = 1
_MODEL_BPE = 2
_MODEL_CHAR = 4
_WHITESPACE_MARKER = "\u2581"
_REPLACEMENT_CHARACTER = "\u2047"


@dataclass(frozen=True, slots=True)
class SentencePieceUnigramPiece:
    """One validated vocabulary entry from ``ModelProto.pieces``."""

    text: str
    score: float
    piece_type: int = 1


@dataclass(frozen=True, slots=True)
class SentencePieceUnigramAssets:
    """Immutable data required for unigram tokenization."""

    pieces: tuple[SentencePieceUnigramPiece, ...]
    unk_token_id: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    unk_surface: str
    byte_fallback: bool
    normalizer_name: str
    add_dummy_prefix: bool
    remove_extra_whitespaces: bool
    escape_whitespaces: bool
    has_precompiled_normalizer: bool
    original_model: bytes
    precompiled_charsmap: bytes = b""


def _read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(payload):
            raise TokenizerAssetError("Truncated protobuf varint.")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise TokenizerAssetError("Protobuf varint exceeds 64 bits.")


def _wire_fields(payload: bytes) -> tuple[tuple[int, int, Any], ...]:
    """Parse one bounded protobuf message without a protobuf dependency."""
    offset = 0
    fields: list[tuple[int, int, Any]] = []
    while offset < len(payload):
        key, offset = _read_varint(payload, offset)
        number = key >> 3
        wire_type = key & 7
        if number < 1:
            raise TokenizerAssetError("Protobuf field numbers must be positive.")
        if wire_type == 0:
            value, offset = _read_varint(payload, offset)
        elif wire_type == 1:
            if offset + 8 > len(payload):
                raise TokenizerAssetError("Truncated fixed64 protobuf field.")
            value = payload[offset:offset + 8]
            offset += 8
        elif wire_type == 2:
            size, offset = _read_varint(payload, offset)
            if size > len(payload) - offset:
                raise TokenizerAssetError("Truncated length-delimited protobuf field.")
            value = payload[offset:offset + size]
            offset += size
        elif wire_type == 5:
            if offset + 4 > len(payload):
                raise TokenizerAssetError("Truncated fixed32 protobuf field.")
            value = payload[offset:offset + 4]
            offset += 4
        else:
            raise TokenizerAssetError(f"Unsupported protobuf wire type {wire_type}.")
        fields.append((number, wire_type, value))
    return tuple(fields)


def _single_field(
    fields: tuple[tuple[int, int, Any], ...],
    number: int,
    wire_type: int,
    *,
    default: Any,
) -> Any:
    values = [
        value for field_number, field_wire, value in fields
        if field_number == number and field_wire == wire_type
    ]
    if len(values) > 1:
        raise TokenizerAssetError(f"SentencePiece protobuf field {number} appears more than once.")
    return default if not values else values[0]


def _decode_utf8(value: bytes, *, context: str) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TokenizerAssetError(f"{context} is not valid UTF-8.") from error


def _parse_piece(payload: bytes, *, max_token_bytes: int) -> SentencePieceUnigramPiece:
    fields = _wire_fields(payload)
    raw_text = _single_field(fields, 1, 2, default=None)
    raw_score = _single_field(fields, 2, 5, default=None)
    piece_type = _single_field(fields, 3, 0, default=1)
    if raw_text is None or raw_score is None:
        raise TokenizerAssetError("Every SentencePiece unigram entry requires text and score.")
    if len(raw_text) > max_token_bytes:
        raise TokenizerAssetError(f"SentencePiece token exceeds {max_token_bytes} UTF-8 bytes.")
    text = _decode_utf8(raw_text, context="SentencePiece token")
    score = float(struct.unpack("<f", raw_score)[0])
    if not text:
        raise TokenizerAssetError("SentencePiece tokens cannot be empty.")
    if not math.isfinite(score):
        raise TokenizerAssetError("SentencePiece token scores must be finite.")
    if piece_type not in {1, _UNKNOWN, _CONTROL, 4, _UNUSED, _BYTE}:
        raise TokenizerAssetError(f"Unsupported SentencePiece token type {piece_type}.")
    return SentencePieceUnigramPiece(text, score, piece_type)


def _parse_normalizer(payload: bytes, ) -> tuple[str, bool, bool, bool, bytes]:
    fields = _wire_fields(payload)
    raw_name = _single_field(fields, 1, 2, default=b"")
    precompiled = _single_field(fields, 2, 2, default=b"")
    add_dummy_prefix = bool(_single_field(fields, 3, 0, default=1))
    remove_extra_whitespaces = bool(_single_field(fields, 4, 0, default=1))
    escape_whitespaces = bool(_single_field(fields, 5, 0, default=1))
    return (
        _decode_utf8(raw_name, context="SentencePiece normalizer name"),
        add_dummy_prefix,
        remove_extra_whitespaces,
        escape_whitespaces,
        precompiled,
    )


def _signed_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def _parse_trainer(payload: bytes, ) -> tuple[int, int, int, int, str, bool]:
    fields = _wire_fields(payload)
    return (
        _signed_int32(_single_field(fields, 40, 0, default=0)),
        _signed_int32(_single_field(fields, 41, 0, default=1)),
        _signed_int32(_single_field(fields, 42, 0, default=2)),
        _signed_int32(_single_field(fields, 43, 0, default=-1)),
        _decode_utf8(
            _single_field(
                fields,
                44,
                2,
                default=f" {_REPLACEMENT_CHARACTER} ".encode(),
            ),
            context="SentencePiece unknown surface",
        ),
        bool(_single_field(fields, 35, 0, default=0)),
    )


def _parse_model_type(payload: bytes, ) -> int:
    """Return the SentencePiece ``TrainerSpec.model_type`` enum value."""
    fields = _wire_fields(payload)
    model_type = _single_field(fields, 3, 0, default=_MODEL_UNIGRAM)
    if model_type not in {_MODEL_UNIGRAM, _MODEL_BPE, _MODEL_CHAR}:
        raise TokenizerAssetError(
            "VoiceHub supports SentencePiece UNIGRAM, BPE, and CHAR model "
            f"types, not enum value {model_type}.")
    return model_type


def load_sentencepiece_unigram(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_ASSET_BYTES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES,
) -> SentencePieceUnigramAssets:
    """Read a SentencePiece unigram ``ModelProto`` with strict bounds."""
    for name, value in (
        ("max_tokens", max_tokens),
        ("max_token_bytes", max_token_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"`{name}` must be an integer.")
        if value <= 0:
            raise ValueError(f"`{name}` must be greater than zero.")
    payload = read_bounded_asset(path, max_bytes=max_bytes)
    fields = _wire_fields(payload)
    raw_pieces = [value for number, wire_type, value in fields if number == 1 and wire_type == 2]
    if not raw_pieces:
        raise TokenizerAssetError("SentencePiece model contains no vocabulary pieces.")
    if len(raw_pieces) > max_tokens:
        raise TokenizerAssetError(f"SentencePiece model contains more than {max_tokens} tokens.")
    pieces = tuple(_parse_piece(value, max_token_bytes=max_token_bytes) for value in raw_pieces)
    if len({piece.text for piece in pieces}) != len(pieces):
        raise TokenizerAssetError("SentencePiece vocabulary contains duplicate token text.")
    unknown_ids = [index for index, piece in enumerate(pieces) if piece.piece_type == _UNKNOWN]
    if len(unknown_ids) != 1:
        raise TokenizerAssetError("SentencePiece unigram models require exactly one unknown token.")
    raw_trainers = [value for number, wire_type, value in fields if number == 2 and wire_type == 2]
    if len(raw_trainers) > 1:
        raise TokenizerAssetError("SentencePiece model contains multiple trainer specifications.")
    model_type = (_MODEL_UNIGRAM if not raw_trainers else _parse_model_type(raw_trainers[0]))
    if model_type == _MODEL_BPE:
        raise TokenizerAssetError(
            "SentencePiece model declares BPE; use "
            "`load_sentencepiece_model_bpe` instead of the unigram loader.")
    if model_type == _MODEL_CHAR and any(
            len(piece.text) != 1 for piece in pieces
            if piece.piece_type not in {_UNKNOWN, _CONTROL, _UNUSED}):
        # A CHAR model splits normalized text into code points. With only
        # single-code-point pieces the unigram lattice has exactly that one
        # path, so it reproduces SentencePiece's CHAR segmentation
        # (including fused adjacent unknowns). Multi-character
        # user-defined symbols would need greedy prefix matching instead.
        raise TokenizerAssetError(
            "SentencePiece CHAR models with multi-character pieces are not "
            "supported.")
    trainer = ((0, 1, 2, -1, f" {_REPLACEMENT_CHARACTER} ",
                False) if not raw_trainers else _parse_trainer(raw_trainers[0]))
    if trainer[0] != unknown_ids[0]:
        raise TokenizerAssetError(
            "SentencePiece trainer `unk_id` does not identify the UNKNOWN "
            "vocabulary piece.")
    for name, token_id in (
        ("bos_id", trainer[1]),
        ("eos_id", trainer[2]),
        ("pad_id", trainer[3]),
    ):
        if token_id < -1 or token_id >= len(pieces):
            raise TokenizerAssetError(f"SentencePiece trainer `{name}` is outside the vocabulary.")
    raw_normalizers = [value for number, wire_type, value in fields if number == 3 and wire_type == 2]
    if len(raw_normalizers) > 1:
        raise TokenizerAssetError("SentencePiece model contains multiple normalizer specifications.")
    normalizer = (("", True, True, True,
                   b"") if not raw_normalizers else _parse_normalizer(raw_normalizers[0]))
    return SentencePieceUnigramAssets(
        pieces=pieces,
        unk_token_id=trainer[0],
        bos_token_id=trainer[1],
        eos_token_id=trainer[2],
        pad_token_id=trainer[3],
        unk_surface=trainer[4],
        byte_fallback=trainer[5],
        normalizer_name=normalizer[0],
        add_dummy_prefix=normalizer[1],
        remove_extra_whitespaces=normalizer[2],
        escape_whitespaces=normalizer[3],
        has_precompiled_normalizer=bool(normalizer[4]),
        original_model=payload,
        precompiled_charsmap=normalizer[4],
    )


def _decode_utf8_length(data: bytes, start: int) -> int:
    """Return the byte length of one valid UTF-8 character, or 0.

    Mirrors SentencePiece ``string_util::IsValidDecodeUTF8``: overlong forms,
    surrogates, and code points above U+10FFFF are malformed.
    """
    first = data[start]
    if first < 0x80:
        return 1
    remaining = len(data) - start
    for length, mask, marker, minimum in ((2, 0xE0, 0xC0, 0x80), (3, 0xF0, 0xE0, 0x800),
                                          (4, 0xF8, 0xF0, 0x10000)):
        if first & mask != marker:
            continue
        if remaining < length:
            return 0
        codepoint = first & (0xFF >> (length + 1))
        for byte in data[start + 1:start + length]:
            if byte & 0xC0 != 0x80:
                return 0
            codepoint = (codepoint << 6) | (byte & 0x3F)
        valid = codepoint < 0xD800 or 0xE000 <= codepoint <= 0x10FFFF
        return length if codepoint >= minimum and valid else 0
    return 0


class _PrecompiledNormalizer:
    """SentencePiece ``Normalizer`` driven by a precompiled charsmap.

    The blob is ``<uint32 trie size><Darts-clone double array><replacement
    strings>``; each trie value is the offset of a NUL-terminated replacement.
    """

    _MAX_TRIE_RESULTS = 32

    def __init__(self, blob: bytes) -> None:
        if len(blob) < 4:
            raise TokenizerAssetError("SentencePiece precompiled charsmap is truncated.")
        trie_size = struct.unpack_from("<I", blob)[0]
        if trie_size == 0 or trie_size % 4 or trie_size > len(blob) - 4:
            raise TokenizerAssetError("SentencePiece precompiled charsmap trie size is invalid.")
        self._units = struct.unpack_from(f"<{trie_size // 4}I", blob, 4)
        self._replacements = blob[4 + trie_size:]

    def _replacement(self, offset: int) -> bytes:
        end = self._replacements.find(b"\0", offset)
        if offset >= len(self._replacements) or end < 0:
            raise TokenizerAssetError("SentencePiece precompiled charsmap value is invalid.")
        return self._replacements[offset:end]

    def _normalize_prefix(self, data: bytes, start: int) -> tuple[bytes, int]:
        units = self._units
        unit = units[0]
        node = (unit >> 10) << ((unit & (1 << 9)) >> 6)
        longest_length = 0
        longest_value = 0
        results = 0
        for position in range(start, len(data)):
            byte = data[position]
            node ^= byte
            if node >= len(units):
                break
            unit = units[node]
            if unit & ((1 << 31) | 0xFF) != byte:
                break
            node ^= (unit >> 10) << ((unit & (1 << 9)) >> 6)
            if (unit >> 8) & 1:
                if results < self._MAX_TRIE_RESULTS and position + 1 - start > longest_length:
                    longest_length = position + 1 - start
                    longest_value = units[node] & ((1 << 31) - 1)
                results += 1
        if longest_length:
            return self._replacement(longest_value), longest_length
        length = _decode_utf8_length(data, start)
        if not length:
            # Malformed UTF-8 consumes one byte and emits U+FFFD.
            return "\ufffd".encode(), 1
        return data[start:start + length], length

    def normalize(
        self,
        text: str,
        *,
        add_dummy_prefix: bool,
        remove_extra_whitespaces: bool,
        escape_whitespaces: bool,
    ) -> str:
        data = text.encode("utf-8", "surrogatepass")
        position = 0
        if remove_extra_whitespaces:
            while position < len(data):
                replacement, consumed = self._normalize_prefix(data, position)
                if replacement != b" ":
                    break
                position += consumed
        if position >= len(data):
            return ""
        space = _WHITESPACE_MARKER.encode() if escape_whitespaces else b" "
        output = bytearray(space if add_dummy_prefix else b"")
        previous_space = remove_extra_whitespaces
        while position < len(data):
            replacement, consumed = self._normalize_prefix(data, position)
            position += consumed
            if previous_space:
                replacement = replacement.lstrip(b" ")
            if replacement:
                output += (replacement.replace(b" ", space) if escape_whitespaces else replacement)
                previous_space = replacement.endswith(b" ")
            if not remove_extra_whitespaces:
                previous_space = False
        if remove_extra_whitespaces:
            while output.endswith(space):
                del output[-len(space):]
        return output.decode("utf-8")


class SentencePieceUnigramTokenizer:
    """Pure-Python Viterbi tokenizer for SentencePiece unigram models."""

    def __init__(
        self,
        assets: SentencePieceUnigramAssets,
        *,
        max_input_chars: int = 1_000_000,
    ) -> None:
        if not isinstance(assets, SentencePieceUnigramAssets):
            raise TypeError("`assets` must be SentencePieceUnigramAssets.")
        if (isinstance(max_input_chars, bool) or not isinstance(max_input_chars, int) or
                max_input_chars <= 0):
            raise ValueError("`max_input_chars` must be a positive integer.")
        if assets.normalizer_name not in {"", "identity", "nmt_nfkc", "nfkc"}:
            raise TokenizerAssetError(
                "VoiceHub supports identity, NFKC, and nmt_nfkc "
                f"SentencePiece normalizers, not {assets.normalizer_name!r}.")
        if assets.byte_fallback:
            raise TokenizerAssetError(
                "SentencePiece unigram byte fallback is not implemented; "
                "refusing to replace byte pieces with unknown tokens.")
        self._assets = assets
        self._pieces = assets.pieces
        self._id_to_piece = tuple(piece.text for piece in assets.pieces)
        self._piece_to_id = MappingProxyType({piece.text: index for index, piece in enumerate(assets.pieces)})
        self._unknown_id = assets.unk_token_id
        self._control_ids = frozenset(
            index for index, piece in enumerate(assets.pieces) if piece.piece_type == _CONTROL)
        self._unused_ids = frozenset(
            index for index, piece in enumerate(assets.pieces) if piece.piece_type == _UNUSED)
        trie: dict[str, Any] = {}
        for token_id, piece in enumerate(assets.pieces):
            if piece.piece_type in {_UNKNOWN, _CONTROL, _UNUSED, _BYTE}:
                continue
            node = trie
            for character in piece.text:
                node = node.setdefault(character, {})
            node.setdefault(None, []).append(token_id)
        self._trie = trie
        self._precompiled_normalizer = (
            _PrecompiledNormalizer(assets.precompiled_charsmap) if assets.precompiled_charsmap else None)
        self.max_input_chars = max_input_chars

    @classmethod
    def from_model_file(
        cls,
        path: str | Path,
        **limits: Any,
    ) -> SentencePieceUnigramTokenizer:
        return cls(load_sentencepiece_unigram(path, **limits))

    @property
    def vocabulary_size(self) -> int:
        return len(self._pieces)

    @property
    def unk_token_id(self) -> int:
        return self._unknown_id

    @property
    def bos_token_id(self) -> int:
        return self._assets.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._assets.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._assets.pad_token_id

    @property
    def vocabulary(self) -> MappingProxyType:
        return self._piece_to_id

    @property
    def special_tokens(self) -> MappingProxyType:
        values = {
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
        return MappingProxyType(values)

    def get_piece_size(self) -> int:
        """SentencePieceProcessor-compatible vocabulary-size method."""
        return self.vocabulary_size

    def piece_to_id(self, piece: str) -> int:
        if not isinstance(piece, str):
            raise TypeError("`piece` must be a string.")
        return self._piece_to_id.get(piece, self._unknown_id)

    def id_to_piece(self, token_id: int) -> str:
        if (isinstance(token_id, bool) or not isinstance(token_id, int) or
                not 0 <= token_id < len(self._pieces)):
            raise ValueError("Token ID is outside the SentencePiece vocabulary.")
        return self._id_to_piece[token_id]

    def _normalize(self, text: str) -> str:
        if self._precompiled_normalizer is not None:
            return self._precompiled_normalizer.normalize(
                text,
                add_dummy_prefix=self._assets.add_dummy_prefix,
                remove_extra_whitespaces=self._assets.remove_extra_whitespaces,
                escape_whitespaces=self._assets.escape_whitespaces,
            )
        if self._assets.normalizer_name not in {"", "identity"}:
            text = unicodedata.normalize("NFKC", text)
        if self._assets.remove_extra_whitespaces:
            text = " ".join(text.split())
        else:
            text = "".join(" " if character.isspace() else character for character in text)
        if not text:
            return ""
        if self._assets.add_dummy_prefix:
            text = " " + text
        if self._assets.escape_whitespaces:
            text = text.replace(" ", _WHITESPACE_MARKER)
        return text

    def _matches(
        self,
        text: str,
        start: int,
    ) -> tuple[tuple[int, int], ...]:
        node = self._trie
        matches: list[tuple[int, int]] = []
        for position in range(start, len(text)):
            child = node.get(text[position])
            if child is None:
                break
            node = child
            for token_id in node.get(None, ()):
                matches.append((position + 1, token_id))
        return tuple(matches)

    def _encode_path(
        self,
        normalized: str,
    ) -> tuple[tuple[int, str], ...]:
        if not normalized:
            return ()
        size = len(normalized)
        best = [float("-inf")] * (size + 1)
        previous: list[tuple[int, int] | None] = [None] * (size + 1)
        best[0] = 0.0
        minimum_score = min(piece.score for piece in self._pieces)
        unknown_score = minimum_score - 10.0

        def add_score(left: float, right: float) -> float:
            # The reference lattice stores path scores as C++ ``float``.
            # Rounding after each edge matters for otherwise identical
            # repeated-piece segmentations.
            return struct.unpack("<f", struct.pack("<f", left + right))[0]

        for start in range(size):
            if best[start] == float("-inf"):
                continue
            matches = self._matches(normalized, start)
            for end, token_id in matches:
                candidate = add_score(
                    best[start],
                    self._pieces[token_id].score,
                )
                if candidate > best[end]:
                    best[end] = candidate
                    previous[end] = (start, token_id)
            if not any(end == start + 1 for end, _ in matches):
                end = start + 1
                candidate = add_score(best[start], unknown_score)
                if candidate > best[end]:
                    best[end] = candidate
                    previous[end] = (start, self._unknown_id)
        if previous[size] is None:
            raise RuntimeError("SentencePiece unigram graph could not cover normalized text.")
        reversed_nodes: list[tuple[int, str]] = []
        position = size
        while position:
            edge = previous[position]
            if edge is None:
                raise RuntimeError("SentencePiece Viterbi backtrace is incomplete.")
            start, token_id = edge
            reversed_nodes.append((token_id, normalized[start:position]), )
            position = start
        nodes = list(reversed(reversed_nodes))
        # SentencePiece fuses adjacent unknown character nodes into one piece
        # while retaining their original normalized surface for ``out_type``
        # piece strings.
        fused: list[tuple[int, str]] = []
        for token_id, surface in nodes:
            if (token_id == self._unknown_id and fused and fused[-1][0] == token_id):
                fused[-1] = (token_id, fused[-1][1] + surface)
            else:
                fused.append((token_id, surface))
        return tuple(fused)

    def _encode_ids(self, normalized: str) -> tuple[int, ...]:
        return tuple(token_id for token_id, _ in self._encode_path(normalized))

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> Encoding:
        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")
        if len(text) > self.max_input_chars:
            raise ValueError(f"Input contains more than {self.max_input_chars} characters.")
        if add_special_tokens:
            raise ValueError(
                "This SentencePiece model declares no automatic special-token "
                "template; prepend or append task tokens explicitly.")
        ids = self._encode_ids(self._normalize(text))
        return Encoding(
            input_ids=ids,
            attention_mask=tuple(1 for _ in ids),
        )

    def encode_as_ids(self, text: str) -> list[int]:
        """SentencePieceProcessor-compatible convenience method."""
        return list(self.encode(text).input_ids)

    def encode_as_pieces(self, text: str) -> list[str]:
        """Return recognized piece strings with unknown surfaces preserved."""
        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")
        if len(text) > self.max_input_chars:
            raise ValueError(f"Input contains more than {self.max_input_chars} characters.")
        return [(surface if token_id == self._unknown_id else self._id_to_piece[token_id])
                for token_id, surface in self._encode_path(self._normalize(text))]

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = False,
        pad: bool = False,
        pad_token_id: int = 0,
    ) -> BatchEncoding:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("`texts` must be a sequence of strings.")
        rows = tuple(self.encode(text, add_special_tokens=add_special_tokens) for text in texts)
        maximum = max((len(row.input_ids) for row in rows), default=0) if pad else None
        input_ids = []
        masks = []
        for row in rows:
            amount = 0 if maximum is None else maximum - len(row.input_ids)
            input_ids.append(row.input_ids + (pad_token_id, ) * amount)
            masks.append(row.attention_mask + (0, ) * amount)
        return BatchEncoding(
            input_ids=tuple(input_ids),
            attention_mask=tuple(masks),
            special_tokens_mask=tuple(tuple(0 if visible else 1 for visible in mask) for mask in masks),
        )

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        pieces: list[str] = []
        for raw_id in token_ids:
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise TypeError("Token IDs must be integers.")
            if not 0 <= raw_id < len(self._pieces):
                raise ValueError(f"Token ID {raw_id} is outside the SentencePiece vocabulary.")
            if skip_special_tokens and raw_id in self._control_ids:
                continue
            if raw_id == self._unknown_id:
                pieces.append(self._assets.unk_surface)
            elif raw_id not in self._unused_ids:
                pieces.append(self._id_to_piece[raw_id])
        decoded = "".join(pieces)
        if self._assets.escape_whitespaces:
            decoded = decoded.replace(_WHITESPACE_MARKER, " ")
        if self._assets.add_dummy_prefix and decoded.startswith(" "):
            decoded = decoded[1:]
        return decoded

    def decode_ids(self, token_ids: Iterable[int]) -> str:
        """SentencePieceProcessor-compatible convenience method."""
        return self.decode(token_ids)

    def batch_decode(
        self,
        sequences: Iterable[Iterable[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return [self.decode(
            sequence,
            skip_special_tokens=skip_special_tokens,
        ) for sequence in sequences]

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        filename: str = "tokenizer.model",
    ) -> Path:
        if not isinstance(filename, str) or not filename:
            raise ValueError("`filename` must be a non-empty string.")
        if Path(filename).name != filename:
            raise ValueError("`filename` must not contain path separators.")
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / filename
        output.write_bytes(self._assets.original_model)
        return output


__all__ = [
    "SentencePieceUnigramAssets",
    "SentencePieceUnigramPiece",
    "SentencePieceUnigramTokenizer",
    "load_sentencepiece_unigram",
]
