"""Dependency-free Fish S2 byte-BPE tokenizer and protocol validation."""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.models.asr_qwen3.native.tokenization import qwen2_pretokenize
from voicehub.models.fishtts.native.configuration import FishS2Config
from voicehub.tokenization import ByteBPETokenizer, Encoding, TokenizerAssetError
from voicehub.tokenization.assets import read_bounded_asset

END_OF_TEXT = "<|endoftext|>"
PAD = "<|pad|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
PHONEME_START = "<|phoneme_start|>"
PHONEME_END = "<|phoneme_end|>"
MODALITY_TEXT = "<|text|>"
MODALITY_VOICE = "<|voice|>"
MODALITY_INTERLEAVE = "<|interleave|>"
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"
AUDIO_PAD = "<|audio_pad|>"
SEMANTIC_TEMPLATE = "<|semantic:{code}|>"

_REQUIRED_PROTOCOL = (
    END_OF_TEXT,
    PAD,
    IM_START,
    IM_END,
    PHONEME_START,
    PHONEME_END,
    MODALITY_TEXT,
    MODALITY_VOICE,
    MODALITY_INTERLEAVE,
    AUDIO_START,
    AUDIO_END,
    AUDIO_PAD,
)
_CONTROL_PATTERN = re.compile(r"<\|[^<>\r\n]{1,128}\|>")
_SPEAKER_PATTERN = re.compile(r"<\|speaker:\d+\|>")
_QWEN2_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
    r"\s+(?!\S)|\s+")


def normalize_fish_text(text: str) -> str:
    """Normalize user text while rejecting model-control injection.

    Fish's documented ``<|speaker:N|>`` tags are data, not registered
    tokenizer control tokens, so they remain permitted.
    """
    if not isinstance(text, str):
        raise TypeError("Fish text must be a string.")
    normalized = unicodedata.normalize("NFC", text)
    if not normalized.strip():
        raise ValueError("Fish text cannot be empty.")
    for match in _CONTROL_PATTERN.finditer(normalized):
        if _SPEAKER_PATTERN.fullmatch(match.group(0)) is None:
            raise ValueError(
                "Fish input cannot contain reserved `<|...|>` control "
                f"token {match.group(0)!r}.")
    return normalized


def _document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(read_bounded_asset(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise TokenizerAssetError(f"Invalid Fish tokenizer JSON: {error}.") from error
    if not isinstance(value, dict):
        raise TokenizerAssetError("Fish tokenizer JSON root must be an object.")
    return value


def _added_tokens(document: Mapping[str, Any]) -> dict[str, int]:
    records = document.get("added_tokens")
    if not isinstance(records, list):
        raise TokenizerAssetError("Fish tokenizer has no declarative added-token table.")
    output: dict[str, int] = {}
    ids: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TokenizerAssetError("Fish added-token records must be objects.")
        spelling = record.get("content")
        token_id = record.get("id")
        if (not isinstance(spelling, str) or not spelling or isinstance(token_id, bool) or
                not isinstance(token_id, int) or token_id < 0):
            raise TokenizerAssetError("Fish tokenizer contains an invalid added token.")
        if spelling in output or token_id in ids:
            raise TokenizerAssetError("Fish tokenizer contains duplicate added tokens.")
        if record.get("lstrip") or record.get("rstrip"):
            raise TokenizerAssetError("Fish protocol tokens cannot strip surrounding whitespace.")
        output[spelling] = token_id
        ids.add(token_id)
    return output


def _validate_pretokenizer(document: Mapping[str, Any]) -> None:
    pretokenizer = document.get("pre_tokenizer")
    candidates = (pretokenizer.get("pretokenizers") if isinstance(pretokenizer, Mapping) else None)
    if not isinstance(candidates, list):
        raise TokenizerAssetError("Fish tokenizer requires its Qwen2 pre-tokenizer sequence.")
    split_patterns = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("type") != "Split":
            continue
        pattern = candidate.get("pattern")
        if isinstance(pattern, Mapping):
            split_patterns.append(pattern.get("Regex"))
    if split_patterns != [_QWEN2_PATTERN]:
        raise TokenizerAssetError("Fish tokenizer uses an unreviewed pre-tokenization expression.")


class FishTokenizer:
    """Validated native wrapper around VoiceHub's byte-BPE engine."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        token_ids: Mapping[str, int],
        config: FishS2Config,
        tokenizer_path: Path,
        tokenizer_config_path: Path | None,
    ) -> None:
        self._tokenizer = tokenizer
        self._token_ids = dict(token_ids)
        self.config = config
        self.tokenizer_path = tokenizer_path
        self.tokenizer_config_path = tokenizer_config_path
        self.semantic_begin_id = config.semantic_begin_id
        self.semantic_end_id = config.semantic_end_id
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.end_of_text_id

    @classmethod
    def from_tokenizer_json(
        cls,
        path: str | Path,
        *,
        config: FishS2Config,
        tokenizer_config_path: str | Path | None = None,
    ) -> FishTokenizer:
        if not isinstance(config, FishS2Config):
            raise TypeError("`config` must be a FishS2Config.")
        tokenizer_path = Path(path).expanduser().resolve()
        document = _document(tokenizer_path)
        _validate_pretokenizer(document)
        tokens = _added_tokens(document)
        missing = [name for name in _REQUIRED_PROTOCOL if name not in tokens]
        if missing:
            raise TokenizerAssetError(
                "Fish tokenizer is missing protocol tokens: " + ", ".join(repr(item)
                                                                          for item in missing) + ".")
        expected = {
            END_OF_TEXT: config.end_of_text_id,
            PAD: config.pad_token_id,
            IM_START: config.im_start_id,
            IM_END: config.im_end_id,
            AUDIO_PAD: config.audio_pad_token_id,
        }
        for spelling, expected_id in expected.items():
            if tokens.get(spelling) != expected_id:
                raise TokenizerAssetError(
                    f"Fish token {spelling!r} must use ID {expected_id}; "
                    f"found {tokens.get(spelling)!r}.")
        for code in range(config.codebook_size):
            spelling = SEMANTIC_TEMPLATE.format(code=code)
            expected_id = config.semantic_begin_id + code
            if tokens.get(spelling) != expected_id:
                raise TokenizerAssetError(
                    f"Fish semantic token {spelling!r} must use ID "
                    f"{expected_id}; found {tokens.get(spelling)!r}.")
        tokenizer = ByteBPETokenizer.from_tokenizer_json(
            tokenizer_path,
            pad_token_id=config.pad_token_id,
            use_regex=False,
            pretokenizer=qwen2_pretokenize,
        )
        if tokenizer.token_id_space_size > config.text.vocab_size:
            raise TokenizerAssetError(
                "Fish tokenizer IDs exceed the model vocabulary: "
                f"{tokenizer.token_id_space_size} > "
                f"{config.text.vocab_size}.")
        config_path = (
            None if tokenizer_config_path is None else Path(tokenizer_config_path).expanduser().resolve())
        return cls(
            tokenizer,
            token_ids=tokens,
            config=config,
            tokenizer_path=tokenizer_path,
            tokenizer_config_path=config_path,
        )

    @property
    def vocab_size(self) -> int:
        return self.config.text.vocab_size

    def __len__(self) -> int:
        return self.vocab_size

    def get_token_id(self, token: str) -> int:
        if token == "<|end_of_text|>":
            token = END_OF_TEXT
        try:
            return self._token_ids[token]
        except KeyError as error:
            raise KeyError(f"Unknown Fish protocol token {token!r}.") from error

    convert_tokens_to_ids = get_token_id

    def semantic_code_to_token_id(self, code: int) -> int:
        if (isinstance(code, bool) or not isinstance(code, int) or not 0 <= code < self.config.codebook_size):
            raise ValueError(
                "Fish semantic code must be an integer in "
                f"[0, {self.config.codebook_size - 1}].")
        return self.semantic_begin_id + code

    def token_id_to_semantic_code(self, token_id: int) -> int:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("Fish semantic token ID must be an integer.")
        code = token_id - self.semantic_begin_id
        if not 0 <= code < self.config.codebook_size:
            raise ValueError(f"Token ID {token_id} is not a semantic token.")
        return code

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        allow_protocol_tokens: bool = True,
    ) -> list[int]:
        if add_special_tokens:
            raise ValueError(
                "Fish prompt assembly inserts control tokens explicitly; "
                "`add_special_tokens=True` is unsupported.")
        encoded = self._tokenizer.encode(
            text,
            allowed_special=("all" if allow_protocol_tokens else "none"),
            disallowed_special=("none" if allow_protocol_tokens else "all"),
        )
        return list(encoded.input_ids)

    def encode_user_text(self, text: str) -> Encoding:
        normalized = normalize_fish_text(text)
        return self._tokenizer.encode(
            normalized,
            allowed_special="none",
            disallowed_special="all",
        )

    def decode(
        self,
        token_ids: Sequence[int] | int,
        *,
        skip_special_tokens: bool = False,
        **unused: Any,
    ) -> str:
        del unused
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        tokenizer_target = (target / "tokenizer.json").resolve()
        if tokenizer_target != self.tokenizer_path:
            shutil.copy2(self.tokenizer_path, tokenizer_target)
        if self.tokenizer_config_path is not None:
            config_target = (target / "tokenizer_config.json").resolve()
            if config_target != self.tokenizer_config_path:
                shutil.copy2(self.tokenizer_config_path, config_target)
        return target.resolve()


__all__ = [
    "AUDIO_END",
    "AUDIO_PAD",
    "AUDIO_START",
    "END_OF_TEXT",
    "FishTokenizer",
    "IM_END",
    "IM_START",
    "MODALITY_INTERLEAVE",
    "MODALITY_TEXT",
    "MODALITY_VOICE",
    "PAD",
    "SEMANTIC_TEMPLATE",
    "normalize_fish_text",
]
