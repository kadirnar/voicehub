"""Bounded, dependency-free SentencePiece ``ModelProto`` BPE runtime.

SentencePiece serializes both unigram and BPE vocabularies in the same
protobuf container.  The merge table for a BPE model is implicit: every
normal vocabulary piece contributes candidate splits, ordered by its
published score.  This module reconstructs that table without protobuf,
SentencePiece, or Hugging Face Tokenizers.

The standard-library NFKC normalizer is used for the declared ``nfkc``
and ``nmt_nfkc`` profiles.  A model-specific processor may instead call
``encode_normalized`` after reproducing its own audited normalization
graph. Opaque precompiled character maps are recorded but never executed
as code.
"""

from __future__ import annotations

import heapq
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
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
from voicehub.tokenization.sentencepiece_unigram import (
    _CONTROL,
    _UNKNOWN,
    _UNUSED,
    _WHITESPACE_MARKER,
    SentencePieceUnigramPiece,
    _parse_model_type,
    _parse_normalizer,
    _parse_piece,
    _parse_trainer,
    _wire_fields,
)

_NORMAL = 1
_USER_DEFINED = 4


@dataclass(frozen=True, slots=True)
class SentencePieceModelBPEAssets:
    """Immutable BPE vocabulary and normalization metadata."""

    pieces: tuple[SentencePieceUnigramPiece, ...]
    unk_token_id: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    unk_surface: str
    normalizer_name: str
    add_dummy_prefix: bool
    remove_extra_whitespaces: bool
    escape_whitespaces: bool
    has_precompiled_normalizer: bool
    original_model: bytes


def load_sentencepiece_model_bpe(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_ASSET_BYTES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES,
) -> SentencePieceModelBPEAssets:
    """Read and validate a SentencePiece BPE ``ModelProto``."""
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
        raise TokenizerAssetError("SentencePiece BPE model contains no vocabulary pieces.")
    if len(raw_pieces) > max_tokens:
        raise TokenizerAssetError(f"SentencePiece model contains more than {max_tokens} tokens.")
    pieces = tuple(_parse_piece(value, max_token_bytes=max_token_bytes) for value in raw_pieces)
    if len({piece.text for piece in pieces}) != len(pieces):
        raise TokenizerAssetError("SentencePiece vocabulary contains duplicate token text.")

    raw_trainers = [value for number, wire_type, value in fields if number == 2 and wire_type == 2]
    if len(raw_trainers) != 1:
        raise TokenizerAssetError("SentencePiece BPE requires exactly one trainer specification.")
    if _parse_model_type(raw_trainers[0]) != 2:
        raise TokenizerAssetError(
            "SentencePiece model declares UNIGRAM or CHAR; use "
            "`load_sentencepiece_unigram` instead of the BPE loader.")
    trainer = _parse_trainer(raw_trainers[0])
    unknown_ids = [index for index, piece in enumerate(pieces) if piece.piece_type == _UNKNOWN]
    if unknown_ids != [trainer[0]]:
        raise TokenizerAssetError(
            "SentencePiece trainer `unk_id` must identify the sole UNKNOWN "
            "vocabulary piece.")
    if trainer[5]:
        raise TokenizerAssetError(
            "SentencePiece BPE byte fallback is not yet a verified runtime "
            "contract.")
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
                   False) if not raw_normalizers else _parse_normalizer(raw_normalizers[0]))
    return SentencePieceModelBPEAssets(
        pieces=pieces,
        unk_token_id=trainer[0],
        bos_token_id=trainer[1],
        eos_token_id=trainer[2],
        pad_token_id=trainer[3],
        unk_surface=trainer[4],
        normalizer_name=normalizer[0],
        add_dummy_prefix=normalizer[1],
        remove_extra_whitespaces=normalizer[2],
        escape_whitespaces=normalizer[3],
        has_precompiled_normalizer=normalizer[4],
        original_model=payload,
    )


class SentencePieceModelBPETokenizer:
    """Score-ordered SentencePiece BPE with bounded asset parsing."""

    def __init__(
        self,
        assets: SentencePieceModelBPEAssets,
        *,
        max_input_chars: int = 1_000_000,
    ) -> None:
        if not isinstance(assets, SentencePieceModelBPEAssets):
            raise TypeError("`assets` must be SentencePieceModelBPEAssets.")
        if (isinstance(max_input_chars, bool) or not isinstance(max_input_chars, int) or
                max_input_chars <= 0):
            raise ValueError("`max_input_chars` must be a positive integer.")
        if assets.normalizer_name not in {
                "",
                "identity",
                "nfkc",
                "nmt_nfkc",
        }:
            raise TokenizerAssetError(
                "VoiceHub supports identity, NFKC, and nmt_nfkc "
                f"SentencePiece normalizers, not {assets.normalizer_name!r}.")
        user_defined = tuple(piece.text for piece in assets.pieces if piece.piece_type == _USER_DEFINED)
        if user_defined:
            raise TokenizerAssetError(
                "SentencePiece BPE user-defined symbols require a "
                "model-specific protected-token contract.")

        self._assets = assets
        self._pieces = assets.pieces
        self._id_to_piece = tuple(piece.text for piece in assets.pieces)
        self._piece_to_id = MappingProxyType({piece.text: index for index, piece in enumerate(assets.pieces)})
        self._normal_piece_ids = frozenset(
            index for index, piece in enumerate(assets.pieces) if piece.piece_type == _NORMAL)
        self._control_ids = frozenset(
            index for index, piece in enumerate(assets.pieces) if piece.piece_type == _CONTROL)
        self._unused_ids = frozenset(
            index for index, piece in enumerate(assets.pieces) if piece.piece_type == _UNUSED)
        self._merge_ranks = self._build_merge_ranks()
        self.max_input_chars = max_input_chars

    @classmethod
    def from_model_file(
        cls,
        path: str | Path,
        **limits: Any,
    ) -> SentencePieceModelBPETokenizer:
        return cls(load_sentencepiece_model_bpe(path, **limits))

    @property
    def vocabulary_size(self) -> int:
        return len(self._pieces)

    @property
    def vocabulary(self) -> Mapping[str, int]:
        return self._piece_to_id

    @property
    def unk_token_id(self) -> int:
        return self._assets.unk_token_id

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
    def has_precompiled_normalizer(self) -> bool:
        return self._assets.has_precompiled_normalizer

    def get_piece_size(self) -> int:
        return self.vocabulary_size

    def piece_to_id(self, piece: str) -> int:
        if not isinstance(piece, str):
            raise TypeError("`piece` must be a string.")
        return self._piece_to_id.get(piece, self.unk_token_id)

    def id_to_piece(self, token_id: int) -> str:
        if (isinstance(token_id, bool) or not isinstance(token_id, int) or
                not 0 <= token_id < len(self._pieces)):
            raise ValueError("Token ID is outside the SentencePiece vocabulary.")
        return self._id_to_piece[token_id]

    def _build_merge_ranks(self) -> Mapping[tuple[str, str], int]:
        candidates: list[tuple[float, int, int, int, str, str]] = []
        for result_id, piece in enumerate(self._pieces):
            if piece.piece_type != _NORMAL:
                continue
            for split in range(1, len(piece.text)):
                left = piece.text[:split]
                right = piece.text[split:]
                left_id = self._piece_to_id.get(left)
                right_id = self._piece_to_id.get(right)
                if (left_id in self._normal_piece_ids and right_id in self._normal_piece_ids):
                    candidates.append((
                        piece.score,
                        len(left),
                        len(right),
                        result_id,
                        left,
                        right,
                    ))
        # This reproduces SentencePiece's score ordering and Transformers'
        # conversion rule. The final fields make equal-score custom models
        # deterministic without altering published unique-score models.
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                -item[3],
            ),
            reverse=True,
        )
        ranks: dict[tuple[str, str], int] = {}
        for rank, (*_metadata, left, right) in enumerate(candidates):
            ranks.setdefault((left, right), rank)
        return MappingProxyType(ranks)

    def _normalize(self, text: str) -> str:
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

    def _initial_symbols(
        self,
        normalized: str,
    ) -> tuple[tuple[str | None, int], ...]:
        symbols: list[tuple[str | None, int]] = []
        for character in normalized:
            token_id = self._piece_to_id.get(character)
            if token_id not in self._normal_piece_ids:
                if symbols and symbols[-1][0] is None:
                    continue
                symbols.append((None, self.unk_token_id))
            else:
                symbols.append((character, token_id))
        return tuple(symbols)

    def _encode_normalized_ids(self, normalized: str) -> tuple[int, ...]:
        initial = self._initial_symbols(normalized)
        if not initial:
            return ()
        texts = [text for text, _ in initial]
        token_ids = [token_id for _, token_id in initial]
        previous = [index - 1 for index in range(len(initial))]
        following = [index + 1 if index + 1 < len(initial) else -1 for index in range(len(initial))]
        alive = [True] * len(initial)
        versions = [0] * len(initial)
        queue: list[tuple[int, int, int, int, int]] = []

        def enqueue(left: int) -> None:
            if left < 0 or not alive[left]:
                return
            right = following[left]
            if right < 0 or not alive[right]:
                return
            left_text = texts[left]
            right_text = texts[right]
            if left_text is None or right_text is None:
                return
            rank = self._merge_ranks.get((left_text, right_text))
            if rank is not None:
                heapq.heappush(
                    queue,
                    (
                        rank,
                        left,
                        right,
                        versions[left],
                        versions[right],
                    ),
                )

        for index in range(len(initial) - 1):
            enqueue(index)
        while queue:
            rank, left, right, left_version, right_version = heapq.heappop(queue)
            del rank
            if (not alive[left] or not alive[right] or following[left] != right or
                    versions[left] != left_version or versions[right] != right_version):
                continue
            merged = (texts[left] or "") + (texts[right] or "")
            token_id = self._piece_to_id.get(merged)
            if token_id not in self._normal_piece_ids:
                raise RuntimeError("SentencePiece BPE merge table produced an unknown piece.")
            texts[left] = merged
            token_ids[left] = token_id
            versions[left] += 1
            alive[right] = False
            versions[right] += 1
            successor = following[right]
            following[left] = successor
            if successor >= 0:
                previous[successor] = left
            enqueue(previous[left])
            enqueue(left)

        result: list[int] = []
        index = 0
        while index >= 0:
            if alive[index]:
                result.append(token_ids[index])
                index = following[index]
            else:  # pragma: no cover - the first node is never removed
                index += 1
                if index >= len(alive):
                    break
        return tuple(result)

    def encode_normalized(
        self,
        normalized: str,
        *,
        add_special_tokens: bool = False,
    ) -> Encoding:
        """Encode text that already contains its required whitespace marks."""
        if not isinstance(normalized, str):
            raise TypeError("`normalized` must be a string.")
        if len(normalized) > self.max_input_chars:
            raise ValueError(f"Input contains more than {self.max_input_chars} characters.")
        if add_special_tokens:
            raise ValueError(
                "SentencePiece ModelProto has no task-specific postprocessor; "
                "add language/BOS/EOS tokens in the model processor.")
        token_ids = self._encode_normalized_ids(normalized)
        return Encoding(
            input_ids=token_ids,
            attention_mask=tuple(1 for _ in token_ids),
        )

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
        return self.encode_normalized(
            self._normalize(text),
            add_special_tokens=add_special_tokens,
        )

    def encode_as_ids(self, text: str) -> list[int]:
        return list(self.encode(text).input_ids)

    def encode_as_pieces(self, text: str) -> list[str]:
        return [self.id_to_piece(token_id) for token_id in self.encode(text).input_ids]

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = False,
        pad: bool = False,
        pad_token_id: int | None = None,
    ) -> BatchEncoding:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("`texts` must be a sequence of strings.")
        rows = tuple(self.encode(text, add_special_tokens=add_special_tokens) for text in texts)
        maximum = max(
            (len(row.input_ids) for row in rows),
            default=0,
        ) if pad else None
        padding_id = self.pad_token_id if pad_token_id is None else pad_token_id
        if pad and padding_id < 0:
            raise ValueError("Padding requires an explicit non-negative token ID.")
        input_ids = []
        masks = []
        for row in rows:
            amount = 0 if maximum is None else maximum - len(row.input_ids)
            input_ids.append(row.input_ids + (padding_id, ) * amount)
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
        fragments: list[str] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("Token IDs must be integers.")
            if not 0 <= token_id < len(self._pieces):
                raise ValueError(f"Token ID {token_id} is outside the vocabulary.")
            if skip_special_tokens and token_id in self._control_ids:
                continue
            if token_id == self.unk_token_id:
                fragments.append(self._assets.unk_surface)
            elif token_id not in self._unused_ids:
                fragments.append(self._id_to_piece[token_id])
        decoded = "".join(fragments)
        if self._assets.escape_whitespaces:
            decoded = decoded.replace(_WHITESPACE_MARKER, " ")
        if self._assets.add_dummy_prefix and decoded.startswith(" "):
            decoded = decoded[1:]
        return decoded

    def decode_ids(self, token_ids: Iterable[int]) -> str:
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
    "SentencePieceModelBPEAssets",
    "SentencePieceModelBPETokenizer",
    "load_sentencepiece_model_bpe",
]
