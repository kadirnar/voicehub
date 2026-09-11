"""Dependency-free byte-BPE tokenization for every safe NeuTTS variant."""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.models.asr_qwen3.native.tokenization import qwen2_pretokenize
from voicehub.tokenization import ByteBPETokenizer, Encoding, TokenizerAssetError
from voicehub.tokenization.assets import read_bounded_asset
from voicehub.tokenization.llama3 import LLAMA3_SPLIT_PATTERN, llama3_pretokenize

TEXT_REPLACE = "<|TEXT_REPLACE|>"
TEXT_PROMPT_START = "<|TEXT_PROMPT_START|>"
TEXT_PROMPT_END = "<|TEXT_PROMPT_END|>"
SPEECH_REPLACE = "<|SPEECH_REPLACE|>"
SPEECH_GENERATION_START = "<|SPEECH_GENERATION_START|>"
SPEECH_GENERATION_END = "<|SPEECH_GENERATION_END|>"
SPEECH_CODEBOOK_SIZE = 65_536
SUPPORTED_EMOTIONS = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)

QWEN2_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
    r"\s+(?!\S)|\s+")
_PROTOCOL_TOKENS = (
    TEXT_REPLACE,
    TEXT_PROMPT_START,
    TEXT_PROMPT_END,
    SPEECH_REPLACE,
    SPEECH_GENERATION_START,
    SPEECH_GENERATION_END,
)
_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^<>\r\n]{1,128}\|>")


def normalize_neutts_text(text: str) -> str:
    """Apply the BPE-model normalization used by the upstream runtime."""
    if not isinstance(text, str):
        raise TypeError("NeuTTS text must be a string.")
    normalized = unicodedata.normalize("NFKC", text.translate(_QUOTE_MAP))
    if _SPECIAL_TOKEN_PATTERN.search(normalized):
        raise ValueError("NeuTTS text cannot contain reserved `<|...|>` control tokens.")
    return normalized


def _tokenizer_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(read_bounded_asset(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TokenizerAssetError(f"Invalid NeuTTS tokenizer JSON: {error}.") from error
    if not isinstance(document, dict):
        raise TokenizerAssetError("NeuTTS tokenizer JSON root must be an object.")
    return document


def _added_token_ids(document: Mapping[str, Any]) -> dict[str, int]:
    records = document.get("added_tokens")
    if not isinstance(records, list):
        raise TokenizerAssetError("NeuTTS tokenizer has no added-token table.")
    result: dict[str, int] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        content = record.get("content")
        token_id = record.get("id")
        if (isinstance(content, str) and isinstance(token_id, int) and not isinstance(token_id, bool)):
            result[content] = token_id
    return result


def _split_pattern(document: Mapping[str, Any]) -> str:
    pre_tokenizer = document.get("pre_tokenizer")
    if not isinstance(pre_tokenizer, Mapping):
        raise TokenizerAssetError("NeuTTS tokenizer has no declarative pre-tokenizer.")
    candidates = pre_tokenizer.get("pretokenizers")
    if not isinstance(candidates, list):
        candidates = [pre_tokenizer]
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or candidate.get("type") != "Split":
            continue
        pattern = candidate.get("pattern")
        if isinstance(pattern, Mapping) and isinstance(
                pattern.get("Regex"),
                str,
        ):
            return pattern["Regex"]
    raise TokenizerAssetError("NeuTTS tokenizer has no supported declarative split expression.")


def _adds_bos(document: Mapping[str, Any]) -> tuple[bool, int | None]:
    processor = document.get("post_processor")
    processors = (processor.get("processors", ()) if isinstance(processor, Mapping) else ())
    if not isinstance(processors, list):
        return False, None
    for candidate in processors:
        if (not isinstance(candidate, Mapping) or candidate.get("type") != "TemplateProcessing"):
            continue
        single = candidate.get("single")
        specials = candidate.get("special_tokens")
        if not isinstance(single, list) or not isinstance(specials, Mapping):
            continue
        if not single or not isinstance(single[0], Mapping):
            continue
        special = single[0].get("SpecialToken")
        if not isinstance(special, Mapping):
            continue
        spelling = special.get("id")
        record = specials.get(spelling)
        if not isinstance(record, Mapping):
            continue
        ids = record.get("ids")
        if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int):
            return True, ids[0]
    return False, None


class NeuTTSTokenizer:
    """Validated protocol tokenizer backed by VoiceHub's byte-BPE engine."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        token_ids: Mapping[str, int],
        speech_token_offset: int,
        tokenizer_path: Path,
        tokenizer_config_path: Path | None,
        bos_token_id: int | None,
        eos_token_id: int | None,
        pad_token_id: int | None,
        adds_bos_token: bool,
    ) -> None:
        self._tokenizer = tokenizer
        self._token_ids = dict(token_ids)
        self.speech_token_offset = speech_token_offset
        self.tokenizer_path = tokenizer_path
        self.tokenizer_config_path = tokenizer_config_path
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = (pad_token_id if pad_token_id is not None else eos_token_id)
        self.adds_bos_token = adds_bos_token

    @classmethod
    def from_tokenizer_json(
        cls,
        path: str | Path,
        *,
        tokenizer_config_path: str | Path | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        expected_vocabulary_size: int | None = None,
    ) -> NeuTTSTokenizer:
        tokenizer_path = Path(path).expanduser().resolve()
        document = _tokenizer_document(tokenizer_path)
        token_ids = _added_token_ids(document)
        missing = [token for token in _PROTOCOL_TOKENS if token not in token_ids]
        if missing:
            raise TokenizerAssetError(
                "NeuTTS tokenizer is missing protocol tokens: " + ", ".join(repr(token)
                                                                            for token in missing) + ".")
        speech_zero = token_ids.get("<|speech_0|>")
        if speech_zero is None:
            raise TokenizerAssetError("NeuTTS tokenizer is missing <|speech_0|>.")
        for code in range(SPEECH_CODEBOOK_SIZE):
            spelling = f"<|speech_{code}|>"
            expected_id = speech_zero + code
            if token_ids.get(spelling) != expected_id:
                raise TokenizerAssetError(
                    f"NeuTTS token {spelling!r} must use ID {expected_id}; "
                    f"found {token_ids.get(spelling)!r}.")

        pattern = _split_pattern(document)
        if pattern == QWEN2_SPLIT_PATTERN:
            pretokenizer = qwen2_pretokenize
        elif pattern == LLAMA3_SPLIT_PATTERN:
            pretokenizer = llama3_pretokenize
        else:
            raise TokenizerAssetError(
                "NeuTTS tokenizer uses an unreviewed split expression; "
                "refusing approximate tokenization.")
        adds_bos, declared_bos = _adds_bos(document)
        if adds_bos:
            if bos_token_id is None:
                bos_token_id = declared_bos
            elif declared_bos != bos_token_id:
                raise TokenizerAssetError("Tokenizer BOS template conflicts with model config.")
        tokenizer = ByteBPETokenizer.from_tokenizer_json(
            tokenizer_path,
            pad_token_id=(pad_token_id if pad_token_id is not None else eos_token_id),
            use_regex=False,
            pretokenizer=pretokenizer,
        )
        if (expected_vocabulary_size is not None and
                tokenizer.token_id_space_size > expected_vocabulary_size):
            raise TokenizerAssetError(
                "NeuTTS tokenizer declares token IDs outside the model "
                f"vocabulary ({tokenizer.token_id_space_size} > "
                f"{expected_vocabulary_size}).")
        config_path = (
            None if tokenizer_config_path is None else Path(tokenizer_config_path).expanduser().resolve())
        return cls(
            tokenizer,
            token_ids=token_ids,
            speech_token_offset=speech_zero,
            tokenizer_path=tokenizer_path,
            tokenizer_config_path=config_path,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            adds_bos_token=adds_bos,
        )

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.vocabulary_size

    @property
    def token_id_space_size(self) -> int:
        return self._tokenizer.token_id_space_size

    def __len__(self) -> int:
        return self.vocabulary_size

    def convert_tokens_to_ids(self, token: str) -> int:
        if not isinstance(token, str):
            raise TypeError("`token` must be a string.")
        try:
            return self._token_ids[token]
        except KeyError as error:
            raise KeyError(f"Unknown NeuTTS protocol token {token!r}.") from error

    def speech_code_to_token_id(self, code: int) -> int:
        if isinstance(code, bool) or not isinstance(code, int):
            raise TypeError("NeuCodec codes must be integers.")
        if not 0 <= code < SPEECH_CODEBOOK_SIZE:
            raise ValueError(f"NeuCodec codes must be in [0, {SPEECH_CODEBOOK_SIZE - 1}].")
        return self.speech_token_offset + code

    def token_id_to_speech_code(self, token_id: int) -> int:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("NeuTTS token IDs must be integers.")
        code = token_id - self.speech_token_offset
        if not 0 <= code < SPEECH_CODEBOOK_SIZE:
            raise ValueError(f"Token ID {token_id} is not a NeuTTS speech token.")
        return code

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> Encoding:
        if not isinstance(add_special_tokens, bool):
            raise TypeError("`add_special_tokens` must be a boolean.")
        encoding = self._tokenizer.encode(
            text,
            allowed_special="all",
            disallowed_special="none",
        )
        if not add_special_tokens or not self.adds_bos_token:
            return encoding
        if self.bos_token_id is None:
            raise RuntimeError("Tokenizer declares BOS insertion without a BOS token ID.")
        return Encoding(
            input_ids=(self.bos_token_id, *encoding.input_ids),
            attention_mask=(1, *encoding.attention_mask),
            special_tokens_mask=(1, *encoding.special_tokens_mask),
        )

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
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
    "NeuTTSTokenizer",
    "QWEN2_SPLIT_PATTERN",
    "SPEECH_CODEBOOK_SIZE",
    "SPEECH_GENERATION_END",
    "SPEECH_GENERATION_START",
    "SPEECH_REPLACE",
    "SUPPORTED_EMOTIONS",
    "TEXT_PROMPT_END",
    "TEXT_PROMPT_START",
    "TEXT_REPLACE",
    "normalize_neutts_text",
]
