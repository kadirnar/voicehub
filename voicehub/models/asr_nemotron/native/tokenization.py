"""Dependency-free SentencePiece-BPE tokenization for Nemotron 3.5 ASR."""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voicehub.tokenization.assets import (
    DEFAULT_MAX_ASSET_BYTES,
    DEFAULT_MAX_JSON_DEPTH,
    DEFAULT_MAX_JSON_NODES,
    DEFAULT_MAX_MERGES,
    DEFAULT_MAX_TOKEN_BYTES,
    DEFAULT_MAX_TOKENS,
    TokenizerAssetError,
    read_bounded_asset,
)
from voicehub.tokenization.base import BatchEncoding, Encoding

METASPACE = "\u2581"
PAD_TOKEN = "<pad>"
PUBLISHED_BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"
PUBLISHED_PAD_TOKEN_ID = 13087
PUBLISHED_TOKENIZER_BLANK_ID = 13088

_MULTIPLE_SPACES = re.compile(r" {2,}")
_MAX_TOKEN_ID = 2**31 - 1


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is forbidden.")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise TokenizerAssetError(f"Tokenizer JSON contains duplicate key {key!r}.")
        output[key] = value
    return output


def _validate_json_bounds(
    value: Any,
    *,
    max_depth: int,
    max_nodes: int,
) -> None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    count = 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > max_nodes:
            raise TokenizerAssetError(f"Tokenizer JSON contains more than {max_nodes} values.")
        if depth > max_depth:
            raise TokenizerAssetError("Tokenizer JSON nesting exceeds its configured bound.")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)


def _token_id(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TokenizerAssetError(f"{context} must be an integer.")
    if not 0 <= value <= _MAX_TOKEN_ID:
        raise TokenizerAssetError(f"{context} must be between zero and {_MAX_TOKEN_ID}.")
    return value


def _merge_pair(value: Any, *, index: int) -> tuple[str, str]:
    if isinstance(value, str):
        parts = value.split(" ")
    elif isinstance(value, list):
        parts = value
    else:
        raise TokenizerAssetError(f"Nemotron BPE merge {index} must be a string or pair.")
    if len(parts) != 2 or any(not isinstance(part, str) or not part for part in parts):
        raise TokenizerAssetError(f"Nemotron BPE merge {index} is invalid.")
    return parts[0], parts[1]


@dataclass(frozen=True, slots=True)
class NemotronTokenizerAssets:
    vocabulary: Mapping[str, int]
    merges: tuple[tuple[str, str], ...]
    special_tokens: Mapping[str, int]
    unk_token_id: int
    byte_fallback: bool
    fuse_unk: bool
    original_document: Mapping[str, Any]


def load_nemotron_tokenizer_assets(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_ASSET_BYTES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_merges: int = DEFAULT_MAX_MERGES,
    max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES,
    max_json_depth: int = DEFAULT_MAX_JSON_DEPTH,
    max_json_nodes: int = DEFAULT_MAX_JSON_NODES,
) -> NemotronTokenizerAssets:
    """Read and validate the declarative Nemotron tokenizer graph."""
    payload = read_bounded_asset(path, max_bytes=max_bytes)
    try:
        document = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except TokenizerAssetError:
        raise
    except (
            UnicodeDecodeError,
            ValueError,
            json.JSONDecodeError,
            RecursionError,
    ) as error:
        raise TokenizerAssetError(f"Invalid Nemotron tokenizer JSON: {error}.") from error
    _validate_json_bounds(
        document,
        max_depth=max_json_depth,
        max_nodes=max_json_nodes,
    )
    if not isinstance(document, dict):
        raise TokenizerAssetError("Tokenizer JSON root must be an object.")
    if document.get("truncation") is not None or document.get("padding") is not None:
        raise TokenizerAssetError("Nemotron tokenizer-level truncation/padding is unsupported.")

    model = document.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise TokenizerAssetError("Nemotron tokenizer must declare a BPE model.")
    for name in (
            "dropout",
            "continuing_subword_prefix",
            "end_of_word_suffix",
    ):
        if model.get(name) not in (None, ""):
            raise TokenizerAssetError(f"Unsupported Nemotron BPE option `{name}`.")
    if model.get("ignore_merges") not in (None, False):
        raise TokenizerAssetError("Nemotron `ignore_merges=True` is unsupported.")
    byte_fallback = model.get("byte_fallback")
    fuse_unk = model.get("fuse_unk")
    if byte_fallback is not True or fuse_unk is not True:
        raise TokenizerAssetError("Nemotron requires byte fallback and fused unknown tokens.")

    raw_vocabulary = model.get("vocab")
    if not isinstance(raw_vocabulary, dict) or not raw_vocabulary:
        raise TokenizerAssetError("Nemotron BPE vocabulary must be a non-empty object.")
    if len(raw_vocabulary) > max_tokens:
        raise TokenizerAssetError(f"Nemotron vocabulary exceeds {max_tokens} tokens.")
    vocabulary: dict[str, int] = {}
    ids: dict[int, str] = {}
    for token, raw_id in raw_vocabulary.items():
        if not isinstance(token, str):
            raise TokenizerAssetError("Nemotron vocabulary tokens must be strings.")
        if len(token.encode("utf-8")) > max_token_bytes:
            raise TokenizerAssetError(f"Vocabulary token exceeds {max_token_bytes} bytes.")
        token_id = _token_id(raw_id, context="Vocabulary token ID")
        if token_id in ids and ids[token_id] != token:
            raise TokenizerAssetError(f"Vocabulary ID {token_id} is assigned twice.")
        vocabulary[token] = token_id
        ids[token_id] = token

    raw_merges = model.get("merges")
    if not isinstance(raw_merges, list):
        raise TokenizerAssetError("Nemotron BPE merges must be an array.")
    if len(raw_merges) > max_merges:
        raise TokenizerAssetError(f"Nemotron merges exceed {max_merges} records.")
    merges: list[tuple[str, str]] = []
    seen_merges: set[tuple[str, str]] = set()
    for index, value in enumerate(raw_merges):
        pair = _merge_pair(value, index=index)
        if pair in seen_merges:
            raise TokenizerAssetError(f"Duplicate Nemotron merge at index {index}.")
        if pair[0] + pair[1] not in vocabulary:
            raise TokenizerAssetError(f"Nemotron merge {index} produces an unknown token.")
        merges.append(pair)
        seen_merges.add(pair)

    raw_added = document.get("added_tokens")
    if not isinstance(raw_added, list):
        raise TokenizerAssetError("Nemotron added tokens must be an array.")
    special_tokens: dict[str, int] = {}
    for index, record in enumerate(raw_added):
        if not isinstance(record, dict):
            raise TokenizerAssetError(f"Added token {index} must be an object.")
        content = record.get("content")
        if (not isinstance(content, str) or not content or record.get("special") is not True or
                record.get("normalized") not in (None, False) or
                record.get("single_word") not in (None, False) or record.get("lstrip") not in (None, False) or
                record.get("rstrip") not in (None, False)):
            raise TokenizerAssetError(f"Unsupported Nemotron added-token record {index}.")
        token_id = _token_id(
            record.get("id"),
            context=f"Added token {index} ID",
        )
        previous = ids.get(token_id)
        if previous is not None and previous != content:
            raise TokenizerAssetError(f"Added token {content!r} conflicts with ID {token_id}.")
        special_tokens[content] = token_id
        ids[token_id] = content

    unk_token = model.get("unk_token")
    if unk_token != UNK_TOKEN or vocabulary.get(UNK_TOKEN) != 0:
        raise TokenizerAssetError("Nemotron unknown token must be '<unk>' at ID 0.")
    if special_tokens.get(PAD_TOKEN) != PUBLISHED_PAD_TOKEN_ID:
        raise TokenizerAssetError("Nemotron tokenizer must declare '<pad>' at ID 13087.")
    if (special_tokens.get(PUBLISHED_BLANK_TOKEN) != PUBLISHED_TOKENIZER_BLANK_ID):
        raise TokenizerAssetError("Nemotron tokenizer must declare '<blank>' at ID 13088.")

    normalizer = document.get("normalizer")
    stages = (
        normalizer.get("normalizers")
        if isinstance(normalizer, dict) and normalizer.get("type") == "Sequence" else None)
    if (not isinstance(stages, list) or len(stages) != 3 or
        [stage.get("type")
         for stage in stages if isinstance(stage, dict)] != ["Precompiled", "Strip", "Replace"]):
        raise TokenizerAssetError("Nemotron requires its published three-stage normalizer.")
    if (stages[1].get("strip_left") is not False or stages[1].get("strip_right") is not True or
            stages[2].get("pattern") != {"Regex": " {2,}"} or stages[2].get("content") != METASPACE):
        raise TokenizerAssetError("Nemotron whitespace normalization does not match the checkpoint.")
    charsmap = stages[0].get("precompiled_charsmap")
    if not isinstance(charsmap, str) or not charsmap:
        raise TokenizerAssetError("Nemotron precompiled normalization table is missing.")

    expected_metaspace = {
        "type": "Metaspace",
        "replacement": METASPACE,
        "prepend_scheme": "always",
        "split": True,
    }
    if document.get("pre_tokenizer") != expected_metaspace:
        raise TokenizerAssetError("Nemotron pre-tokenizer differs from the published Metaspace graph.")
    if document.get("decoder") != expected_metaspace:
        raise TokenizerAssetError("Nemotron decoder differs from the published Metaspace graph.")
    post_processor = document.get("post_processor")
    if (not isinstance(post_processor, dict) or post_processor.get("type") != "TemplateProcessing" or
            post_processor.get("single") != [{"Sequence": {"id": "A", "type_id": 0}}] or
            post_processor.get("special_tokens") != {}):
        raise TokenizerAssetError("Nemotron tokenizer post-processor is unsupported.")

    return NemotronTokenizerAssets(
        vocabulary=MappingProxyType(vocabulary),
        merges=tuple(merges),
        special_tokens=MappingProxyType(special_tokens),
        unk_token_id=0,
        byte_fallback=byte_fallback,
        fuse_unk=fuse_unk,
        original_document=MappingProxyType(document),
    )


class NemotronASRTokenizer:
    """Native ordered BPE with SentencePiece-compatible Metaspace rules."""

    def __init__(
        self,
        assets: NemotronTokenizerAssets,
        *,
        tokenizer_json_path: Path,
        max_input_chars: int = 1_000_000,
    ) -> None:
        if not isinstance(assets, NemotronTokenizerAssets):
            raise TypeError("`assets` must be NemotronTokenizerAssets.")
        if (isinstance(max_input_chars, bool) or not isinstance(max_input_chars, int) or
                max_input_chars <= 0):
            raise ValueError("`max_input_chars` must be a positive integer.")
        self._assets = assets
        self.tokenizer_json_path = tokenizer_json_path
        self._id_to_token = MappingProxyType({
            token_id: token
            for token, token_id in (
                *assets.vocabulary.items(),
                *assets.special_tokens.items(),
            )
        })
        self._merge_ranks = MappingProxyType({pair: rank for rank, pair in enumerate(assets.merges)})
        self._special_ids = frozenset(assets.special_tokens.values())
        self._special_spellings = tuple(
            sorted(
                assets.special_tokens,
                key=lambda value: (-len(value), value),
            ))
        self.max_input_chars = max_input_chars

    @classmethod
    def from_tokenizer_json(
        cls,
        path: str | Path,
        **limits: Any,
    ) -> NemotronASRTokenizer:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Nemotron tokenizer was not found: {source}.")
        return cls(
            load_nemotron_tokenizer_assets(source, **limits),
            tokenizer_json_path=source,
        )

    @property
    def vocabulary_size(self) -> int:
        return len(set(self._id_to_token))

    @property
    def token_id_space_size(self) -> int:
        return max(self._id_to_token) + 1

    @property
    def pad_token_id(self) -> int:
        return PUBLISHED_PAD_TOKEN_ID

    @property
    def blank_token_id(self) -> int:
        return PUBLISHED_TOKENIZER_BLANK_ID

    @property
    def special_token_ids(self) -> frozenset[int]:
        return self._special_ids

    @property
    def special_tokens(self) -> Mapping[str, int]:
        return self._assets.special_tokens

    def token_for_id(self, token_id: int) -> str:
        """Return one validated vocabulary spelling."""
        resolved = _token_id(token_id, context="Token ID")
        try:
            return self._id_to_token[resolved]
        except KeyError as error:
            raise ValueError(f"Token ID {resolved} is absent from the tokenizer.") from error

    @staticmethod
    def normalize(text: str) -> str:
        """Apply the checkpoint's NFKC/strip/Metaspace-compatible rules."""
        if not isinstance(text, str):
            raise TypeError("Nemotron text must be a string.")
        normalized = unicodedata.normalize("NFKC", text)
        normalized = "".join(" " if character.isspace() else character for character in normalized).rstrip()
        normalized = _MULTIPLE_SPACES.sub(METASPACE, normalized)
        normalized = normalized.replace(" ", METASPACE)
        if normalized.startswith(METASPACE):
            return normalized
        return METASPACE + normalized

    def _initial_tokens(self, text: str) -> list[str]:
        pieces: list[str] = []
        unknown = self._id_to_token[self._assets.unk_token_id]
        for character in self.normalize(text):
            if character in self._assets.vocabulary:
                pieces.append(character)
                continue
            fallback = tuple(f"<0x{byte:02X}>" for byte in character.encode("utf-8"))
            if all(token in self._assets.vocabulary for token in fallback):
                pieces.extend(fallback)
            elif not (self._assets.fuse_unk and pieces and pieces[-1] == unknown):
                pieces.append(unknown)
        return pieces

    def _merge(self, pieces: list[str]) -> tuple[int, ...]:
        while len(pieces) > 1:
            best_rank: int | None = None
            best_index: int | None = None
            for index in range(len(pieces) - 1):
                rank = self._merge_ranks.get((pieces[index], pieces[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_index is None:
                break
            pieces[best_index:best_index + 2] = [pieces[best_index] + pieces[best_index + 1]]
        return tuple(self._assets.vocabulary.get(
            piece,
            self._assets.unk_token_id,
        ) for piece in pieces)

    def _segments(
        self,
        text: str,
    ) -> tuple[tuple[bool, str], ...]:
        segments: list[tuple[bool, str]] = []
        start = 0
        index = 0
        while index < len(text):
            match = next(
                (token for token in self._special_spellings if text.startswith(token, index)),
                None,
            )
            if match is None:
                index += 1
                continue
            if index > start:
                segments.append((False, text[start:index]))
            segments.append((True, match))
            index += len(match)
            start = index
        if start < len(text) or not segments:
            segments.append((False, text[start:]))
        return tuple(segments)

    def encode(
        self,
        text: str,
        *,
        allow_special_tokens: bool = False,
    ) -> Encoding:
        if not isinstance(text, str):
            raise TypeError("Nemotron text must be a string.")
        if not text:
            raise ValueError("Nemotron transcripts must be non-empty.")
        if len(text) > self.max_input_chars:
            raise ValueError(f"Input exceeds {self.max_input_chars} characters.")
        ids: list[int] = []
        for is_special, segment in self._segments(text):
            if is_special:
                if not allow_special_tokens:
                    raise ValueError(f"Transcript contains reserved token {segment!r}.")
                ids.append(self._assets.special_tokens[segment])
            elif segment:
                ids.extend(self._merge(self._initial_tokens(segment)))
        if not ids:
            raise ValueError("Nemotron transcript produced no tokens.")
        return Encoding(tuple(ids))

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        padding: bool = True,
        allow_special_tokens: bool = False,
    ) -> BatchEncoding:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("`texts` must be a sequence of strings.")
        rows = tuple(self.encode(
            text,
            allow_special_tokens=allow_special_tokens,
        ) for text in texts)
        width = max((len(row) for row in rows), default=0) if padding else None
        ids: list[tuple[int, ...]] = []
        masks: list[tuple[int, ...]] = []
        specials: list[tuple[int, ...]] = []
        for row in rows:
            amount = 0 if width is None else width - len(row)
            ids.append(row.input_ids + (self.pad_token_id, ) * amount)
            masks.append(row.attention_mask + (0, ) * amount)
            specials.append(row.special_tokens_mask + (1, ) * amount)
        return BatchEncoding(
            input_ids=tuple(ids),
            attention_mask=tuple(masks),
            special_tokens_mask=tuple(specials),
        )

    @staticmethod
    def _byte_value(token: str) -> int | None:
        if (len(token) != 6 or not token.startswith("<0x") or token[-1] != ">"):
            return None
        try:
            return int(token[3:5], 16)
        except ValueError:
            return None

    def decode(
        self,
        token_ids: Iterable[int] | Encoding,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        values = (token_ids.input_ids if isinstance(token_ids, Encoding) else token_ids)
        tokens: list[str] = []
        for raw_id in values:
            token_id = _token_id(raw_id, context="Token ID")
            token = self.token_for_id(token_id)
            if skip_special_tokens and token_id in self._special_ids:
                continue
            tokens.append(token)

        fragments: list[str] = []
        buffered = bytearray()

        def flush_bytes() -> None:
            if buffered:
                fragments.append(buffered.decode("utf-8", errors="replace"))
                buffered.clear()

        for token in tokens:
            byte_value = self._byte_value(token)
            if byte_value is not None:
                buffered.append(byte_value)
            else:
                flush_bytes()
                fragments.append(token)
        flush_bytes()
        decoded = "".join(fragments).replace(METASPACE, " ")
        return decoded[1:] if decoded.startswith(" ") else decoded

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

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "tokenizer.json"
        if self.tokenizer_json_path != target.resolve():
            shutil.copyfile(self.tokenizer_json_path, target)
        return target


__all__ = [
    "METASPACE",
    "NemotronASRTokenizer",
    "NemotronTokenizerAssets",
    "PAD_TOKEN",
    "PUBLISHED_BLANK_TOKEN",
    "PUBLISHED_PAD_TOKEN_ID",
    "PUBLISHED_TOKENIZER_BLANK_ID",
    "UNK_TOKEN",
    "load_nemotron_tokenizer_assets",
]
