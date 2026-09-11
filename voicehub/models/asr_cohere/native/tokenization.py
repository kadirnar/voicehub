"""Dependency-free 16k tokenizer for Cohere Transcribe."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voicehub.models.asr_parakeet_tdt.native.tokenization import PrecompiledCharsMap
from voicehub.tokenization.assets import (
    DEFAULT_MAX_ASSET_BYTES,
    DEFAULT_MAX_MERGES,
    DEFAULT_MAX_TOKEN_BYTES,
    DEFAULT_MAX_TOKENS,
    TokenizerAssetError,
    read_bounded_asset,
)
from voicehub.tokenization.base import BatchEncoding, Encoding

_BYTE_TOKEN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
_MAX_TOKEN_ID = 2**31 - 1
_WHITESPACE_MARKER = "▁"


def _token_id(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TokenizerAssetError(f"{context} must be an integer.")
    if not 0 <= value <= _MAX_TOKEN_ID:
        raise TokenizerAssetError(f"{context} is outside the supported range.")
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
        raise TokenizerAssetError(f"Invalid Cohere tokenizer JSON: {error}.") from error
    if not isinstance(value, dict):
        raise TokenizerAssetError("Cohere tokenizer JSON root must be an object.")
    return value


@dataclass(frozen=True, slots=True)
class CohereTokenizerAssets:
    """Validated declarative tokenizer graph."""

    vocabulary: Mapping[str, int]
    id_to_token: Mapping[int, str]
    merge_ranks: Mapping[tuple[str, str], int]
    special_tokens: Mapping[str, int]
    special_ids: frozenset[int]
    unk_token_id: int
    pad_token_id: int
    eos_token_id: int
    bos_token_id: int
    normalizer: PrecompiledCharsMap
    original_document: Mapping[str, Any]
    original_config: Mapping[str, Any]


def load_cohere_tokenizer(
    tokenizer_path: str | Path,
    tokenizer_config_path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_ASSET_BYTES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_merges: int = DEFAULT_MAX_MERGES,
    max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES,
) -> CohereTokenizerAssets:
    """Parse the published Tokenizers JSON without executing remote code."""
    document = _json(tokenizer_path, max_bytes=max_bytes)
    tokenizer_config = _json(
        tokenizer_config_path,
        max_bytes=max_bytes,
    )
    if document.get("version") != "1.0":
        raise TokenizerAssetError("Unsupported Cohere tokenizer format version.")
    if document.get("truncation") is not None:
        raise TokenizerAssetError("Cohere tokenizer-level truncation must be disabled.")
    if document.get("padding") is not None:
        raise TokenizerAssetError("Cohere tokenizer-level padding must be disabled.")

    model = document.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise TokenizerAssetError("Cohere tokenizer must declare BPE.")
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
            raise TokenizerAssetError(f"Unsupported Cohere BPE option {name}={model.get(name)!r}.")
    raw_vocabulary = model.get("vocab")
    if not isinstance(raw_vocabulary, dict) or not raw_vocabulary:
        raise TokenizerAssetError("Cohere BPE vocabulary must be non-empty.")
    if len(raw_vocabulary) > max_tokens:
        raise TokenizerAssetError("Cohere BPE vocabulary exceeds the safety limit.")
    vocabulary: dict[str, int] = {}
    id_to_token: dict[int, str] = {}
    for token, raw_id in raw_vocabulary.items():
        if (not isinstance(token, str) or not token or len(token.encode("utf-8")) > max_token_bytes):
            raise TokenizerAssetError("Cohere vocabulary contains an invalid token.")
        token_id = _token_id(raw_id, context="Vocabulary token ID")
        if token_id in id_to_token:
            raise TokenizerAssetError(f"Cohere vocabulary duplicates token ID {token_id}.")
        vocabulary[token] = token_id
        id_to_token[token_id] = token
    if set(id_to_token) != set(range(len(id_to_token))):
        raise TokenizerAssetError("Cohere vocabulary IDs must form one contiguous range.")

    raw_merges = model.get("merges")
    if not isinstance(raw_merges, list) or len(raw_merges) > max_merges:
        raise TokenizerAssetError("Cohere BPE merges are invalid or oversized.")
    merge_ranks: dict[tuple[str, str], int] = {}
    for index, value in enumerate(raw_merges):
        parts = value if isinstance(value, list) else str(value).split(" ")
        if (len(parts) != 2 or any(not isinstance(part, str) or not part for part in parts)):
            raise TokenizerAssetError(f"Invalid Cohere BPE merge at index {index}.")
        pair = (parts[0], parts[1])
        if pair in merge_ranks or pair[0] + pair[1] not in vocabulary:
            raise TokenizerAssetError(f"Incoherent Cohere BPE merge at index {index}.")
        merge_ranks[pair] = index

    raw_added = document.get("added_tokens")
    if not isinstance(raw_added, list) or len(raw_added) > max_tokens:
        raise TokenizerAssetError("Cohere added-token table is invalid or oversized.")
    specials: dict[str, int] = {}
    special_ids: set[int] = set()
    for index, record in enumerate(raw_added):
        if not isinstance(record, dict):
            raise TokenizerAssetError(f"Cohere added token {index} must be an object.")
        content = record.get("content")
        if not isinstance(content, str) or content not in vocabulary:
            raise TokenizerAssetError(f"Cohere added token {index} is absent from the vocabulary.")
        token_id = _token_id(
            record.get("id"),
            context=f"Added token {index} ID",
        )
        if vocabulary[content] != token_id:
            raise TokenizerAssetError(f"Cohere added token {content!r} has an inconsistent ID.")
        if (record.get("single_word") is not False or record.get("lstrip") is not False or
                record.get("rstrip") is not False or record.get("normalized") is not False):
            raise TokenizerAssetError("Unsupported Cohere added-token matching policy.")
        if record.get("special") is not True:
            raise TokenizerAssetError("Published Cohere added tokens must be special.")
        specials[content] = token_id
        special_ids.add(token_id)

    def configured_token(name: str, default: str) -> int:
        value = tokenizer_config.get(name, default)
        if not isinstance(value, str) or value not in vocabulary:
            raise TokenizerAssetError(f"Cohere tokenizer is missing {name!r}.")
        return vocabulary[value]

    unk_token = model.get("unk_token")
    if not isinstance(unk_token, str) or unk_token not in vocabulary:
        raise TokenizerAssetError("Cohere tokenizer unknown token is invalid.")
    unk_token_id = vocabulary[unk_token]
    pad_token_id = configured_token("pad_token", "<pad>")
    eos_token_id = configured_token("eos_token", "<|endoftext|>")
    bos_token_id = configured_token(
        "bos_token",
        "<|startoftranscript|>",
    )
    special_ids.update((unk_token_id, pad_token_id, eos_token_id, bos_token_id))

    normalizer = document.get("normalizer")
    if (not isinstance(normalizer, dict) or normalizer.get("type") != "Sequence"):
        raise TokenizerAssetError("Cohere tokenizer requires a normalizer sequence.")
    stages = normalizer.get("normalizers")
    if not isinstance(stages, list) or len(stages) != 3:
        raise TokenizerAssetError("Cohere tokenizer requires exactly three normalizer stages.")
    precompiled, prepend, replace = stages
    if (not isinstance(precompiled, dict) or precompiled.get("type") != "Precompiled" or
            not isinstance(precompiled.get("precompiled_charsmap"), str)):
        raise TokenizerAssetError("Cohere tokenizer has an invalid precompiled normalizer.")
    if prepend != {"type": "Prepend", "prepend": _WHITESPACE_MARKER}:
        raise TokenizerAssetError("Cohere tokenizer has an unsupported prefix normalizer.")
    expected_replace = {
        "type": "Replace",
        "pattern": {
            "String": " "
        },
        "content": _WHITESPACE_MARKER,
    }
    if replace != expected_replace:
        raise TokenizerAssetError("Cohere tokenizer has an unsupported whitespace normalizer.")
    if document.get("pre_tokenizer") is not None:
        raise TokenizerAssetError("Cohere tokenizer must not declare a pre-tokenizer.")
    post = document.get("post_processor")
    if (not isinstance(post, dict) or post.get("type") != "TemplateProcessing" or
            post.get("single") != [{"Sequence": {"id": "A", "type_id": 0}}] or
            post.get("special_tokens") != {}):
        raise TokenizerAssetError("Cohere tokenizer post-processor must be the identity template.")
    expected_decoder = {
        "type":
        "Sequence",
        "decoders": [
            {
                "type": "Replace",
                "pattern": {
                    "String": _WHITESPACE_MARKER
                },
                "content": " ",
            },
            {
                "type": "ByteFallback"
            },
            {
                "type": "Fuse"
            },
            {
                "type": "Strip",
                "content": " ",
                "start": 1,
                "stop": 0,
            },
        ],
    }
    if document.get("decoder") != expected_decoder:
        raise TokenizerAssetError("Cohere tokenizer decoder graph is unsupported.")
    return CohereTokenizerAssets(
        vocabulary=MappingProxyType(vocabulary),
        id_to_token=MappingProxyType(id_to_token),
        merge_ranks=MappingProxyType(merge_ranks),
        special_tokens=MappingProxyType(specials),
        special_ids=frozenset(special_ids),
        unk_token_id=unk_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        normalizer=PrecompiledCharsMap(precompiled["precompiled_charsmap"]),
        original_document=MappingProxyType(document),
        original_config=MappingProxyType(tokenizer_config),
    )


class CohereAsrTokenizer:
    """SentencePiece-style byte-fallback BPE used by Cohere Transcribe."""

    def __init__(
        self,
        assets: CohereTokenizerAssets,
        *,
        max_input_chars: int = 1_000_000,
    ) -> None:
        if not isinstance(assets, CohereTokenizerAssets):
            raise TypeError("`assets` must be CohereTokenizerAssets.")
        if (isinstance(max_input_chars, bool) or not isinstance(max_input_chars, int) or max_input_chars < 1):
            raise ValueError("`max_input_chars` must be a positive integer.")
        self.assets = assets
        self.max_input_chars = max_input_chars
        self.unk_token_id = assets.unk_token_id
        self.pad_token_id = assets.pad_token_id
        self.eos_token_id = assets.eos_token_id
        self.bos_token_id = assets.bos_token_id
        self._special_strings = tuple(
            sorted(
                assets.special_tokens,
                key=lambda token: (-len(token), token),
            ))

    @classmethod
    def from_files(
        cls,
        tokenizer_path: str | Path,
        tokenizer_config_path: str | Path,
    ) -> CohereAsrTokenizer:
        return cls(load_cohere_tokenizer(
            tokenizer_path,
            tokenizer_config_path,
        ))

    @property
    def vocabulary_size(self) -> int:
        return len(self.assets.vocabulary)

    @property
    def token_id_space_size(self) -> int:
        return max(self.assets.id_to_token) + 1

    def convert_tokens_to_ids(
        self,
        tokens: str | Sequence[str],
    ) -> int | list[int]:
        single = isinstance(tokens, str)
        values = (tokens, ) if single else tuple(tokens)
        result = []
        for token in values:
            if not isinstance(token, str):
                raise TypeError("Cohere tokenizer tokens must be strings.")
            result.append(self.assets.vocabulary.get(token, self.unk_token_id))
        return result[0] if single else result

    def token_piece(self, token_id: int) -> str:
        normalized = _token_id(token_id, context="Token ID")
        try:
            return self.assets.id_to_token[normalized]
        except KeyError as error:
            raise ValueError(f"Token ID {normalized} is absent from the Cohere vocabulary.") from error

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
        add_special_tokens: bool = False,
        allow_special_tokens: bool = False,
    ) -> Encoding:
        if not isinstance(text, str):
            raise TypeError("Cohere tokenizer input must be a string.")
        if len(text) > self.max_input_chars:
            raise ValueError(f"Input contains more than {self.max_input_chars} characters.")
        if add_special_tokens:
            raise ValueError(
                "Cohere ASR uses an explicit task prompt; automatic BOS/EOS "
                "insertion is unsupported.")
        present_specials = [token for token in self._special_strings if token in text]
        if present_specials and not allow_special_tokens:
            raise ValueError(
                "Cohere transcript contains a reserved control token: "
                f"{present_specials[0]!r}.")
        if allow_special_tokens and text in self.assets.special_tokens:
            token_id = self.assets.special_tokens[text]
            return Encoding(
                input_ids=(token_id, ),
                attention_mask=(1, ),
                special_tokens_mask=(1, ),
            )
        normalized = self.assets.normalizer.normalize(text)
        normalized = _WHITESPACE_MARKER + normalized
        normalized = normalized.replace(" ", _WHITESPACE_MARKER)
        ids = self._merge(self._initial_pieces(normalized))
        return Encoding(
            input_ids=ids,
            attention_mask=tuple(1 for _ in ids),
        )

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        pad: bool = False,
    ) -> BatchEncoding:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("`texts` must be a sequence of strings.")
        rows = tuple(self.encode(text) for text in texts)
        maximum = max((len(row) for row in rows), default=0) if pad else None
        ids = []
        masks = []
        specials = []
        for row in rows:
            amount = 0 if maximum is None else maximum - len(row)
            ids.append(row.input_ids + (self.pad_token_id, ) * amount)
            masks.append(row.attention_mask + (0, ) * amount)
            specials.append(row.special_tokens_mask + (1, ) * amount)
        return BatchEncoding(
            input_ids=tuple(ids),
            attention_mask=tuple(masks),
            special_tokens_mask=tuple(specials),
        )

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        pieces = []
        for raw_id in token_ids:
            token_id = _token_id(raw_id, context="Token ID")
            if (skip_special_tokens and token_id in self.assets.special_ids):
                continue
            pieces.append(self.token_piece(token_id))
        fragments: list[str] = []
        buffered = bytearray()

        def flush() -> None:
            if buffered:
                fragments.append(buffered.decode("utf-8", errors="replace"))
                buffered.clear()

        for piece in pieces:
            match = _BYTE_TOKEN.match(piece)
            if match is not None:
                buffered.append(int(match.group(1), 16))
            else:
                flush()
                fragments.append(piece)
        flush()
        text = "".join(fragments).replace(_WHITESPACE_MARKER, " ")
        return text[1:] if text.startswith(" ") else text

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

    def save_pretrained(self, directory: str | Path) -> tuple[Path, Path]:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer_path = destination / "tokenizer.json"
        config_path = destination / "tokenizer_config.json"
        tokenizer_path.write_text(
            json.dumps(
                dict(self.assets.original_document),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        config_path.write_text(
            json.dumps(
                dict(self.assets.original_config),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return tokenizer_path, config_path


__all__ = [
    "CohereAsrTokenizer",
    "CohereTokenizerAssets",
    "load_cohere_tokenizer",
]
