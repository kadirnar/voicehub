"""A deterministic, dependency-free byte-level BPE tokenizer."""

from __future__ import annotations

import codecs
import heapq
import threading
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Collection, Iterable, Mapping
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from voicehub.tokenization.assets import (
    DEFAULT_MAX_ASSET_BYTES,
    DEFAULT_MAX_MERGES,
    DEFAULT_MAX_TOKEN_BYTES,
    DEFAULT_MAX_TOKENS,
    load_huggingface_byte_bpe,
    load_tiktoken_ranks,
)
from voicehub.tokenization.base import (
    BatchEncoding,
    Encoding,
    PaddingStrategy,
    SpecialTokenSelection,
    TruncationStrategy,
    pad_encodings,
)

_CONTRACTIONS = ("re", "ve", "ll", "s", "t", "m", "d")
_APOSTROPHES = frozenset({"'"})
_NORMALIZATIONS = frozenset({"NFC", "NFD", "NFKC", "NFKD"})


class TokenizationError(ValueError):
    """Base error for text that cannot be represented by a tokenizer."""


class SpecialTokenError(TokenizationError):
    """Raised when input contains a special token that was not allowed."""


def pretokenize(text: str) -> tuple[str, ...]:
    """Split text using GPT/Whisper-like Unicode-aware boundaries.

    Python's standard :mod:`re` does not provide Unicode ``Letter`` and
    ``Number`` properties. This scanner uses
    :func:`unicodedata.category` instead, preserving contractions, one
    optional ASCII space before a word, and exact source text for
    lossless byte encoding.
    """
    if not isinstance(text, str):
        raise TypeError("`text` must be a string.")
    pieces: list[str] = []
    index = 0
    while index < len(text):
        contraction_end = _contraction_end(text, index)
        if contraction_end is not None:
            pieces.append(text[index:contraction_end])
            index = contraction_end
            continue

        start = index
        if (text[index] == " " and index + 1 < len(text) and not text[index + 1].isspace()):
            index += 1

        character_kind = _character_kind(text[index])
        if character_kind == "space":
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            # GPT-style tokenization attaches the final literal space in a run
            # to the following word while retaining preceding whitespace.
            if end < len(text) and end - start > 1:
                pieces.append(text[start:end - 1])
                index = end - 1
                continue
            pieces.append(text[start:end])
            index = end
            continue

        end = index + 1
        while end < len(text):
            if _character_kind(text[end]) != character_kind:
                break
            end += 1
        pieces.append(text[start:end])
        index = end
    return tuple(pieces)


class ByteBPETokenizer:
    """Byte-level BPE compatible with TikToken and tokenizer.json assets.

    The vocabulary maps raw byte strings to token IDs. For TikToken
    assets, token IDs also define merge priority. Hugging Face BPE
    assets instead provide an explicit ordered merge table.
    """

    def __init__(
        self,
        mergeable_ranks: Mapping[bytes, int],
        *,
        merges: Iterable[tuple[bytes, bytes]] | None = None,
        special_tokens: Mapping[str, int] | None = None,
        added_tokens: Mapping[str, int] | None = None,
        unk_token_id: int | None = None,
        pad_token_id: int | None = None,
        prefix_token_ids: Iterable[int] = (),
        suffix_token_ids: Iterable[int] = (),
        add_prefix_space: bool = False,
        use_regex: bool = True,
        pretokenizer: Callable[[str], Iterable[str]] | None = None,
        normalization: Literal["NFC", "NFD", "NFKC", "NFKD"] | None = None,
        max_input_chars: int = 1_000_000,
        max_piece_bytes: int = 1_000_000,
        cache_capacity: int = 4096,
        padding_side: Literal["left", "right"] = "right",
    ) -> None:
        vocabulary = _validate_vocabulary(mergeable_ranks)
        normalized_special = _validate_special_tokens(special_tokens or {})
        normalized_added = _validate_added_tokens(added_tokens or {})
        overlapping_spellings = normalized_special.keys() & normalized_added.keys()
        if overlapping_spellings:
            raise ValueError(
                "Tokens cannot be both special and non-special added tokens: "
                f"{sorted(overlapping_spellings)!r}.")
        id_to_bytes = {token_id: token for token, token_id in vocabulary.items()}
        id_to_special: dict[int, str] = {}
        for token, token_id in normalized_special.items():
            previous = id_to_special.get(token_id)
            if previous is not None and previous != token:
                raise ValueError(f"Special tokens {previous!r} and {token!r} share ID {token_id}.")
            regular = id_to_bytes.get(token_id)
            if regular is not None and regular != token.encode("utf-8"):
                raise ValueError(f"Special token {token!r} conflicts with regular token ID {token_id}.")
            id_to_special[token_id] = token
        id_to_added: dict[int, str] = {}
        for token, token_id in normalized_added.items():
            special = id_to_special.get(token_id)
            if special is not None:
                raise ValueError(
                    f"Added token {token!r} shares special-token ID {token_id} "
                    f"with {special!r}.")
            previous = id_to_added.get(token_id)
            if previous is not None and previous != token:
                raise ValueError(f"Added tokens {previous!r} and {token!r} share ID "
                                 f"{token_id}.")
            regular = id_to_bytes.get(token_id)
            if regular is not None and regular != token.encode("utf-8"):
                raise ValueError(f"Added token {token!r} conflicts with regular token ID "
                                 f"{token_id}.")
            id_to_added[token_id] = token

        normalized_merges = tuple(merges) if merges is not None else None
        pair_ranks: dict[tuple[bytes, bytes], int] | None = None
        if normalized_merges is not None:
            pair_ranks = {}
            for rank, pair in enumerate(normalized_merges):
                if (not isinstance(pair, tuple) or len(pair) != 2 or
                        not all(isinstance(item, bytes) and item for item in pair)):
                    raise TypeError("Each BPE merge must be a pair of non-empty bytes.")
                if pair in pair_ranks:
                    raise ValueError(f"Duplicate BPE merge pair: {pair!r}.")
                if pair[0] + pair[1] not in vocabulary:
                    raise ValueError(f"BPE merge {pair!r} produces a token absent from the vocabulary.")
                pair_ranks[pair] = rank

        self._vocabulary = MappingProxyType(vocabulary)
        self._id_to_bytes = MappingProxyType(id_to_bytes)
        self._special_tokens = MappingProxyType(normalized_special)
        self._id_to_special = MappingProxyType(id_to_special)
        self._added_tokens = MappingProxyType(normalized_added)
        self._id_to_added = MappingProxyType(id_to_added)
        self._pair_ranks = (MappingProxyType(pair_ranks) if pair_ranks is not None else None)
        self._special_trie = _build_special_trie(normalized_special)
        self._added_trie = _build_special_trie(normalized_added)
        known_ids = set(id_to_bytes) | set(id_to_special) | set(id_to_added)
        self._unk_token_id = _optional_known_id(
            unk_token_id,
            name="unk_token_id",
            known_ids=known_ids,
        )
        self._pad_token_id = _optional_known_id(
            pad_token_id,
            name="pad_token_id",
            known_ids=known_ids,
        )
        self._prefix_token_ids = _known_ids(
            prefix_token_ids,
            name="prefix_token_ids",
            known_ids=known_ids,
        )
        self._suffix_token_ids = _known_ids(
            suffix_token_ids,
            name="suffix_token_ids",
            known_ids=known_ids,
        )
        self._configured_special_ids = frozenset(
            set(id_to_special)
            | set(self._prefix_token_ids)
            | set(self._suffix_token_ids))
        if not isinstance(add_prefix_space, bool):
            raise TypeError("`add_prefix_space` must be a boolean.")
        if not isinstance(use_regex, bool):
            raise TypeError("`use_regex` must be a boolean.")
        if pretokenizer is not None and not callable(pretokenizer):
            raise TypeError("`pretokenizer` must be callable or None.")
        if normalization is not None and normalization not in _NORMALIZATIONS:
            raise ValueError(f"Unsupported Unicode normalization: {normalization!r}.")
        if padding_side not in ("left", "right"):
            raise ValueError("`padding_side` must be either 'left' or 'right'.")
        self._add_prefix_space = add_prefix_space
        self._use_regex = use_regex
        self._pretokenizer = pretokenizer
        self._normalization = normalization
        self._max_input_chars = _positive_integer(
            max_input_chars,
            name="max_input_chars",
        )
        self._max_piece_bytes = _positive_integer(
            max_piece_bytes,
            name="max_piece_bytes",
        )
        self._cache_capacity = _nonnegative_integer(
            cache_capacity,
            name="cache_capacity",
        )
        self._padding_side = padding_side
        self._cache: OrderedDict[bytes, tuple[int, ...]] = OrderedDict()
        self._cache_lock = threading.RLock()

    @classmethod
    def from_tiktoken_file(
        cls,
        path: str | Path,
        *,
        special_tokens: Mapping[str, int] | None = None,
        max_asset_bytes: int = DEFAULT_MAX_ASSET_BYTES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES,
        **options: object,
    ) -> ByteBPETokenizer:
        """Construct a tokenizer from an OpenAI ``*.tiktoken`` rank file."""
        ranks = load_tiktoken_ranks(
            path,
            max_bytes=max_asset_bytes,
            max_tokens=max_tokens,
            max_token_bytes=max_token_bytes,
        )
        return cls(ranks, special_tokens=special_tokens, **options)

    @classmethod
    def from_tiktoken(
        cls,
        path: str | Path,
        **options: object,
    ) -> ByteBPETokenizer:
        """Alias for :meth:`from_tiktoken_file`."""
        return cls.from_tiktoken_file(path, **options)

    @classmethod
    def from_tokenizer_json(
        cls,
        path: str | Path,
        *,
        max_asset_bytes: int = DEFAULT_MAX_ASSET_BYTES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_merges: int = DEFAULT_MAX_MERGES,
        max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES,
        **options: object,
    ) -> ByteBPETokenizer:
        """Construct a tokenizer from a Hugging Face byte-BPE JSON asset."""
        assets = load_huggingface_byte_bpe(
            path,
            max_bytes=max_asset_bytes,
            max_tokens=max_tokens,
            max_merges=max_merges,
            max_token_bytes=max_token_bytes,
        )
        defaults: dict[str, object] = {
            "merges": assets.merges,
            "special_tokens": assets.special_tokens,
            "added_tokens": assets.added_tokens,
            "unk_token_id": assets.unk_token_id,
            "add_prefix_space": assets.add_prefix_space,
            "use_regex": assets.use_regex,
            "normalization": assets.normalization,
        }
        defaults.update(options)
        return cls(assets.vocabulary, **defaults)

    @classmethod
    def from_huggingface_tokenizer_json(
        cls,
        path: str | Path,
        **options: object,
    ) -> ByteBPETokenizer:
        """Explicit alias for :meth:`from_tokenizer_json`."""
        return cls.from_tokenizer_json(path, **options)

    @property
    def vocabulary(self) -> Mapping[bytes, int]:
        """Immutable raw-byte vocabulary."""
        return self._vocabulary

    @property
    def special_tokens(self) -> Mapping[str, int]:
        """Immutable special-token mapping."""
        return self._special_tokens

    @property
    def added_tokens(self) -> Mapping[str, int]:
        """Immutable non-special added-token mapping."""
        return self._added_tokens

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct token IDs represented by the assets."""
        return len(
            set(self._vocabulary.values())
            | set(self._special_tokens.values())
            | set(self._added_tokens.values()))

    @property
    def token_id_space_size(self) -> int:
        """Exclusive upper bound of the tokenizer's declared ID space.

        Most tokenizer vocabularies are dense, making this equal to
        :attr:`vocabulary_size`. Sparse synthetic or converted artifacts
        can retain unused model rows, so model compatibility must
        compare this bound rather than the number of represented
        spellings.
        """
        return max(
            (
                *self._vocabulary.values(),
                *self._special_tokens.values(),
                *self._added_tokens.values(),
            ),
            default=-1,
        ) + 1

    @property
    def pad_token_id(self) -> int | None:
        return self._pad_token_id

    @property
    def unk_token_id(self) -> int | None:
        return self._unk_token_id

    def encode(
        self,
        text: str,
        *,
        allowed_special: SpecialTokenSelection = "none",
        disallowed_special: SpecialTokenSelection = "all",
        max_length: int | None = None,
        truncation: TruncationStrategy = False,
    ) -> Encoding:
        """Encode text with explicit special-token and truncation policy."""
        normalized = self._prepare_text(text)
        allowed = self._resolve_special_selection(
            allowed_special,
            name="allowed_special",
        )
        disallowed = self._resolve_special_selection(
            disallowed_special,
            name="disallowed_special",
        ) - allowed
        disallowed_match = self._first_special_match(normalized, disallowed)
        if disallowed_match is not None:
            _, token = disallowed_match
            raise SpecialTokenError(
                f"Input contains special token {token!r}; add it to "
                "`allowed_special` or remove it from `disallowed_special`.")

        token_ids: list[int] = list(self._prefix_token_ids)
        special_mask: list[int] = [1] * len(self._prefix_token_ids)
        cursor = 0
        # A leftmost match found from an earlier cursor stays the leftmost
        # match until the cursor passes it, and a miss stays a miss. Reusing
        # them keeps long placeholder runs linear instead of quadratic.
        special_match: tuple[int, str] | None = None
        added_match: tuple[int, str] | None = None
        special_exhausted = added_exhausted = False
        while cursor < len(normalized):
            if not special_exhausted and (special_match is None or special_match[0] < cursor):
                special_match = self._first_special_match(
                    normalized,
                    allowed,
                    start=cursor,
                )
                special_exhausted = special_match is None
            if not added_exhausted and (added_match is None or added_match[0] < cursor):
                added_match = self._first_added_match(normalized, start=cursor)
                added_exhausted = added_match is None
            match, is_special = _first_token_match(
                special_match,
                added_match,
            )
            if match is None:
                regular_ids = self._encode_ordinary_fragment(normalized[cursor:])
                token_ids.extend(regular_ids)
                special_mask.extend((0, ) * len(regular_ids))
                break
            position, token = match
            regular_ids = self._encode_ordinary_fragment(normalized[cursor:position])
            token_ids.extend(regular_ids)
            special_mask.extend((0, ) * len(regular_ids))
            token_ids.append(self._special_tokens[token] if is_special else self._added_tokens[token])
            special_mask.append(int(is_special))
            cursor = position + len(token)
        if not normalized:
            cursor = 0
        token_ids.extend(self._suffix_token_ids)
        special_mask.extend((1, ) * len(self._suffix_token_ids))
        encoding = Encoding(
            input_ids=tuple(token_ids),
            special_tokens_mask=tuple(special_mask),
        )
        return _truncate(
            encoding,
            max_length=max_length,
            truncation=truncation,
        )

    def encode_ordinary(
        self,
        text: str,
        *,
        max_length: int | None = None,
        truncation: TruncationStrategy = False,
    ) -> Encoding:
        """Encode every character as ordinary text, including special
        strings."""
        normalized = self._prepare_text(text)
        token_ids = (
            self._prefix_token_ids + self._encode_ordinary_fragment(normalized) + self._suffix_token_ids)
        encoding = Encoding(
            input_ids=token_ids,
            special_tokens_mask=(
                (1, ) * len(self._prefix_token_ids) + (0, ) *
                (len(token_ids) - len(self._prefix_token_ids) - len(self._suffix_token_ids)) +
                (1, ) * len(self._suffix_token_ids)),
        )
        return _truncate(
            encoding,
            max_length=max_length,
            truncation=truncation,
        )

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        padding: PaddingStrategy = False,
        max_length: int | None = None,
        truncation: TruncationStrategy = False,
        pad_to_multiple_of: int | None = None,
        allowed_special: SpecialTokenSelection = "none",
        disallowed_special: SpecialTokenSelection = "all",
    ) -> BatchEncoding:
        """Encode a batch and optionally pad to its longest or requested
        width."""
        try:
            text_values = tuple(texts)
        except TypeError as error:
            raise TypeError("`texts` must be an iterable of strings.") from error
        encodings = tuple(
            self.encode(
                text,
                allowed_special=allowed_special,
                disallowed_special=disallowed_special,
                max_length=max_length,
                truncation=truncation,
            ) for text in text_values)
        if padding is False:
            if pad_to_multiple_of is not None:
                raise ValueError("`pad_to_multiple_of` requires an enabled padding strategy.")
            return BatchEncoding(
                input_ids=tuple(item.input_ids for item in encodings),
                attention_mask=tuple(item.attention_mask for item in encodings),
                special_tokens_mask=tuple(item.special_tokens_mask for item in encodings),
            )
        if padding not in (True, "longest", "max_length"):
            raise ValueError("`padding` must be False, True, 'longest', or 'max_length'.")
        if self._pad_token_id is None:
            raise ValueError("Padding requires a configured `pad_token_id`.")
        target_length: int | None = None
        if padding == "max_length":
            if max_length is None:
                raise ValueError("`padding='max_length'` requires `max_length`.")
            target_length = _nonnegative_integer(max_length, name="max_length")
        return pad_encodings(
            encodings,
            pad_token_id=self._pad_token_id,
            length=target_length,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=self._padding_side,
        )

    def decode(
        self,
        token_ids: Iterable[int] | Encoding,
        *,
        skip_special_tokens: bool = False,
        errors: str = "replace",
    ) -> str:
        """Decode IDs, joining byte fragments before applying UTF-8
        decoding."""
        if not isinstance(skip_special_tokens, bool):
            raise TypeError("`skip_special_tokens` must be a boolean.")
        codecs.lookup_error(errors)
        values = token_ids.input_ids if isinstance(token_ids, Encoding) else token_ids
        try:
            normalized_ids = tuple(values)
        except TypeError as error:
            raise TypeError("`token_ids` must be an iterable of integers.") from error

        output: list[str] = []
        byte_buffer = bytearray()
        skippable_ids = set(self._configured_special_ids)
        if self._pad_token_id is not None:
            skippable_ids.add(self._pad_token_id)
        for token_id in normalized_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, Integral):
                raise TypeError("`token_ids` must contain only integers.")
            normalized_id = int(token_id)
            special = self._id_to_special.get(normalized_id)
            if special is not None:
                if skip_special_tokens:
                    continue
                if byte_buffer:
                    output.append(bytes(byte_buffer).decode("utf-8", errors=errors))
                    byte_buffer.clear()
                output.append(special)
                continue
            added = self._id_to_added.get(normalized_id)
            if added is not None:
                if byte_buffer:
                    output.append(bytes(byte_buffer).decode("utf-8", errors=errors))
                    byte_buffer.clear()
                output.append(added)
                continue
            if skip_special_tokens and normalized_id in skippable_ids:
                continue
            token = self._id_to_bytes.get(normalized_id)
            if token is None:
                raise TokenizationError(f"Unknown token ID: {normalized_id}.")
            byte_buffer.extend(token)
        if byte_buffer:
            output.append(bytes(byte_buffer).decode("utf-8", errors=errors))
        return "".join(output)

    def _prepare_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")
        if len(text) > self._max_input_chars:
            raise TokenizationError(
                f"Input contains {len(text)} characters; the configured limit is "
                f"{self._max_input_chars}.")
        normalized = (
            unicodedata.normalize(self._normalization, text) if self._normalization is not None else text)
        if len(normalized) > self._max_input_chars:
            raise TokenizationError(
                "Unicode normalization expanded the input beyond the configured "
                f"{self._max_input_chars}-character limit.")
        if (self._add_prefix_space and normalized and not normalized[0].isspace()):
            normalized = " " + normalized
        return normalized

    def _encode_ordinary_fragment(self, text: str) -> tuple[int, ...]:
        token_ids: list[int] = []
        if self._pretokenizer is not None:
            try:
                pieces = tuple(self._pretokenizer(text))
            except TypeError as error:
                raise TokenizationError(
                    "The configured pretokenizer did not return an iterable "
                    "of strings.") from error
            if any(not isinstance(piece, str) for piece in pieces):
                raise TokenizationError("The configured pretokenizer must return only strings.")
            if "".join(pieces) != text:
                raise TokenizationError(
                    "The configured pretokenizer must preserve the exact "
                    "source text.")
        else:
            pieces = pretokenize(text) if self._use_regex else ((text, ) if text else ())
        for piece in pieces:
            encoded = piece.encode("utf-8")
            if len(encoded) > self._max_piece_bytes:
                raise TokenizationError(
                    f"Pretokenized piece contains {len(encoded)} bytes; the configured "
                    f"limit is {self._max_piece_bytes}.")
            token_ids.extend(self._encode_piece(encoded))
        return tuple(token_ids)

    def _encode_piece(self, piece: bytes) -> tuple[int, ...]:
        if self._cache_capacity:
            with self._cache_lock:
                cached = self._cache.get(piece)
                if cached is not None:
                    self._cache.move_to_end(piece)
                    return cached
        encoded = self._encode_piece_uncached(piece)
        if self._cache_capacity and len(piece) <= 4096:
            with self._cache_lock:
                self._cache[piece] = encoded
                self._cache.move_to_end(piece)
                while len(self._cache) > self._cache_capacity:
                    self._cache.popitem(last=False)
        return encoded

    def _encode_piece_uncached(self, piece: bytes) -> tuple[int, ...]:
        parts = [bytes((byte, )) for byte in piece]
        if any(part not in self._vocabulary for part in parts):
            if self._unk_token_id is None:
                missing = next(part for part in parts if part not in self._vocabulary)
                raise TokenizationError(f"Vocabulary does not contain byte 0x{missing[0]:02x}.")
            return tuple(self._vocabulary.get(part, self._unk_token_id) for part in parts)
        if len(parts) < 2:
            return tuple(self._vocabulary[part] for part in parts)

        previous = [index - 1 for index in range(len(parts))]
        following = [index + 1 for index in range(len(parts))]
        following[-1] = -1
        alive = [True] * len(parts)
        versions = [0] * len(parts)
        candidates: list[tuple[int, int, int, int, int]] = []

        def add_candidate(left: int) -> None:
            if left < 0 or not alive[left]:
                return
            right = following[left]
            if right < 0 or not alive[right]:
                return
            rank = self._merge_rank(parts[left], parts[right])
            if rank is not None:
                heapq.heappush(
                    candidates,
                    (rank, left, right, versions[left], versions[right]),
                )

        for index in range(len(parts) - 1):
            add_candidate(index)
        while candidates:
            rank, left, right, left_version, right_version = heapq.heappop(candidates)
            if (not alive[left] or not alive[right] or following[left] != right or
                    versions[left] != left_version or versions[right] != right_version or
                    self._merge_rank(parts[left], parts[right]) != rank):
                continue
            parts[left] += parts[right]
            versions[left] += 1
            alive[right] = False
            versions[right] += 1
            next_index = following[right]
            following[left] = next_index
            if next_index >= 0:
                previous[next_index] = left
            add_candidate(previous[left])
            add_candidate(left)

        result: list[int] = []
        index = 0
        while index >= 0:
            token_id = self._vocabulary.get(parts[index])
            if token_id is None:
                if self._unk_token_id is None:
                    raise TokenizationError(
                        f"Merged byte token is absent from the vocabulary: "
                        f"{parts[index]!r}.")
                token_id = self._unk_token_id
            result.append(token_id)
            index = following[index]
        return tuple(result)

    def _merge_rank(self, left: bytes, right: bytes) -> int | None:
        if self._pair_ranks is not None:
            return self._pair_ranks.get((left, right))
        return self._vocabulary.get(left + right)

    def _resolve_special_selection(
        self,
        selection: SpecialTokenSelection,
        *,
        name: str,
    ) -> frozenset[str]:
        if selection == "all":
            return frozenset(self._special_tokens)
        if selection == "none":
            return frozenset()
        if isinstance(selection, str) or not isinstance(selection, Collection):
            raise TypeError(f"`{name}` must be 'all', 'none', or a collection of token strings.")
        normalized = frozenset(selection)
        if any(not isinstance(token, str) for token in normalized):
            raise TypeError(f"`{name}` must contain only strings.")
        unknown = normalized - self._special_tokens.keys()
        if unknown:
            raise ValueError(f"`{name}` contains unknown special tokens: {sorted(unknown)!r}.")
        return normalized

    def _first_special_match(
        self,
        text: str,
        selected: Collection[str],
        *,
        start: int = 0,
    ) -> tuple[int, str] | None:
        if not selected or not self._special_trie:
            return None
        for position in range(start, len(text)):
            node = self._special_trie
            matches: list[str] = []
            cursor = position
            while cursor < len(text):
                child = node.get(text[cursor])
                if not isinstance(child, dict):
                    break
                node = child
                token = node.get(None)
                if isinstance(token, str) and token in selected:
                    matches.append(token)
                cursor += 1
            if matches:
                return position, max(matches, key=len)
        return None

    def _first_added_match(
        self,
        text: str,
        *,
        start: int = 0,
    ) -> tuple[int, str] | None:
        return _first_trie_match(
            text,
            self._added_trie,
            self._added_tokens,
            start=start,
        )


def _character_kind(character: str) -> str:
    if character.isspace():
        return "space"
    category = unicodedata.category(character)
    if category.startswith("L"):
        return "letter"
    if category.startswith("N"):
        return "number"
    return "other"


def _contraction_end(text: str, index: int) -> int | None:
    if text[index] not in _APOSTROPHES:
        return None
    remaining = text[index + 1:]
    for suffix in _CONTRACTIONS:
        if remaining.startswith(suffix):
            return index + 1 + len(suffix)
    return None


def _build_special_trie(tokens: Mapping[str, int]) -> dict[object, object]:
    root: dict[object, object] = {}
    for token in tokens:
        node = root
        for character in token:
            child = node.setdefault(character, {})
            if not isinstance(child, dict):
                raise AssertionError("Invalid special-token trie state.")
            node = child
        node[None] = token
    return root


def _first_trie_match(
    text: str,
    trie: Mapping[object, object],
    selected: Collection[str],
    *,
    start: int,
) -> tuple[int, str] | None:
    if not selected or not trie:
        return None
    for position in range(start, len(text)):
        node = trie
        matches: list[str] = []
        cursor = position
        while cursor < len(text):
            child = node.get(text[cursor])
            if not isinstance(child, dict):
                break
            node = child
            token = node.get(None)
            if isinstance(token, str) and token in selected:
                matches.append(token)
            cursor += 1
        if matches:
            return position, max(matches, key=len)
    return None


def _first_token_match(
    special: tuple[int, str] | None,
    added: tuple[int, str] | None,
) -> tuple[tuple[int, str] | None, bool]:
    if special is None:
        return added, False
    if added is None:
        return special, True
    if special[0] < added[0]:
        return special, True
    if added[0] < special[0]:
        return added, False
    if len(special[1]) >= len(added[1]):
        return special, True
    return added, False


def _validate_vocabulary(value: Mapping[bytes, int]) -> dict[bytes, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("`mergeable_ranks` must be a non-empty mapping.")
    normalized: dict[bytes, int] = {}
    seen_ids: set[int] = set()
    for token, token_id in value.items():
        if not isinstance(token, bytes) or not token:
            raise TypeError("Vocabulary tokens must be non-empty bytes.")
        normalized_id = _token_id(token_id, name="vocabulary token ID")
        if token in normalized:
            raise ValueError(f"Duplicate vocabulary token: {token!r}.")
        if normalized_id in seen_ids:
            raise ValueError(f"Duplicate vocabulary token ID: {normalized_id}.")
        normalized[token] = normalized_id
        seen_ids.add(normalized_id)
    return normalized


def _validate_special_tokens(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("`special_tokens` must be a mapping.")
    normalized: dict[str, int] = {}
    seen_ids: set[int] = set()
    for token, token_id in value.items():
        if not isinstance(token, str) or not token:
            raise TypeError("Special-token keys must be non-empty strings.")
        normalized_id = _token_id(token_id, name="special token ID")
        if normalized_id in seen_ids:
            raise ValueError(f"Duplicate special token ID: {normalized_id}.")
        normalized[token] = normalized_id
        seen_ids.add(normalized_id)
    return normalized


def _validate_added_tokens(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("`added_tokens` must be a mapping.")
    normalized: dict[str, int] = {}
    seen_ids: set[int] = set()
    for token, token_id in value.items():
        if not isinstance(token, str) or not token:
            raise TypeError("Added-token keys must be non-empty strings.")
        normalized_id = _token_id(token_id, name="added token ID")
        if normalized_id in seen_ids:
            raise ValueError(f"Duplicate added token ID: {normalized_id}.")
        normalized[token] = normalized_id
        seen_ids.add(normalized_id)
    return normalized


def _known_ids(
    values: Iterable[int],
    *,
    name: str,
    known_ids: set[int],
) -> tuple[int, ...]:
    try:
        normalized = tuple(_token_id(value, name=name) for value in values)
    except TypeError as error:
        raise TypeError(f"`{name}` must be an iterable of token IDs.") from error
    unknown = set(normalized) - known_ids
    if unknown:
        raise ValueError(f"`{name}` contains unknown token IDs: {sorted(unknown)!r}.")
    return normalized


def _optional_known_id(
    value: int | None,
    *,
    name: str,
    known_ids: set[int],
) -> int | None:
    if value is None:
        return None
    normalized = _token_id(value, name=name)
    if normalized not in known_ids:
        raise ValueError(f"`{name}` references unknown token ID {normalized}.")
    return normalized


def _token_id(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"`{name}` must be non-negative.")
    return normalized


def _positive_integer(value: object, *, name: str) -> int:
    normalized = _token_id(value, name=name)
    if normalized == 0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return normalized


def _nonnegative_integer(value: object, *, name: str) -> int:
    return _token_id(value, name=name)


def _truncate(
    encoding: Encoding,
    *,
    max_length: int | None,
    truncation: TruncationStrategy,
) -> Encoding:
    if max_length is None:
        if truncation not in (False, True, "left", "right"):
            raise ValueError("`truncation` must be False, True, 'left', or 'right'.")
        return encoding
    normalized_length = _nonnegative_integer(max_length, name="max_length")
    if truncation not in (False, True, "left", "right"):
        raise ValueError("`truncation` must be False, True, 'left', or 'right'.")
    if len(encoding) <= normalized_length:
        return encoding
    if truncation is False:
        raise ValueError(
            f"Encoding length {len(encoding)} exceeds `max_length={normalized_length}`; "
            "enable truncation explicitly.")
    selection = (
        slice(len(encoding) -
              normalized_length, None) if truncation == "left" else slice(None, normalized_length))
    return Encoding(
        input_ids=encoding.input_ids[selection],
        attention_mask=encoding.attention_mask[selection],
        special_tokens_mask=encoding.special_tokens_mask[selection],
    )
