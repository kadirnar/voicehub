"""Exact dependency-free byte-BPE tokenization for OuteTTS 1.0."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from voicehub.models.asr_qwen3.native.tokenization import qwen2_pretokenize
from voicehub.tokenization import ByteBPETokenizer, Encoding, TokenizerAssetError
from voicehub.tokenization.assets import read_bounded_asset
from voicehub.tokenization.llama3 import LLAMA3_SPLIT_PATTERN, llama3_pretokenize

_PROTOCOL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|text_start|>",
    "<|text_end|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|word_start|>",
    "<|word_end|>",
    "<|features|>",
    "<|global_features_start|>",
    "<|global_features_end|>",
    "<|code|>",
)
_FAMILY_IDS = {
    "llama": {
        "<|im_start|>": 133_309,
        "<|im_end|>": 133_310,
        "<|text_start|>": 133_311,
        "<|text_end|>": 133_312,
        "<|audio_start|>": 133_317,
        "<|audio_end|>": 133_318,
        "<|word_start|>": 133_320,
        "<|word_end|>": 133_321,
        "<|features|>": 133_322,
        "<|global_features_start|>": 133_323,
        "<|global_features_end|>": 133_324,
        "c1_offset": 128_256,
        "c2_offset": 129_281,
    },
    "qwen3": {
        "<|im_start|>": 151_644,
        "<|im_end|>": 151_645,
        "<|text_start|>": 156_722,
        "<|text_end|>": 156_723,
        "<|audio_start|>": 156_728,
        "<|audio_end|>": 156_729,
        "<|word_start|>": 156_731,
        "<|word_end|>": 156_732,
        "<|features|>": 156_733,
        "<|global_features_start|>": 156_734,
        "<|global_features_end|>": 156_735,
        "c1_offset": 151_669,
        "c2_offset": 152_694,
    },
}


def _document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(read_bounded_asset(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise TokenizerAssetError(f"Invalid OuteTTS tokenizer JSON: {error}.") from error
    if not isinstance(value, dict):
        raise TokenizerAssetError("OuteTTS tokenizer JSON must be an object.")
    return value


def _added_token_ids(document: dict[str, Any]) -> dict[str, int]:
    records = document.get("added_tokens")
    if not isinstance(records, list):
        raise TokenizerAssetError("OuteTTS tokenizer has no added-token table.")
    result: dict[str, int] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TokenizerAssetError("OuteTTS added-token records must be objects.")
        spelling = record.get("content")
        token_id = record.get("id")
        if (not isinstance(spelling, str) or not spelling or isinstance(token_id, bool) or
                not isinstance(token_id, int) or token_id < 0):
            raise TokenizerAssetError("OuteTTS tokenizer contains an invalid added token.")
        if spelling in result and result[spelling] != token_id:
            raise TokenizerAssetError(f"Duplicate OuteTTS added token {spelling!r}.")
        result[spelling] = token_id
    return result


def _detect_family(token_ids: dict[str, int]) -> str:
    for family, expected in _FAMILY_IDS.items():
        if all(token_ids.get(name) == token_id for name, token_id in expected.items()
               if not name.endswith("_offset")):
            return family
    raise TokenizerAssetError("Tokenizer does not match a supported OuteTTS 1.0 Llama or Qwen "
                              "protocol.")


def _validate_audio_tokens(
    token_ids: dict[str, int],
    *,
    family: str,
) -> None:
    expected = _FAMILY_IDS[family]
    for codebook in ("c1", "c2"):
        offset = expected[f"{codebook}_offset"]
        for value in range(1_025):
            spelling = f"<|{codebook}_{value}|>"
            actual = token_ids.get(spelling)
            wanted = offset + value
            if actual != wanted:
                raise TokenizerAssetError(
                    f"OuteTTS token {spelling!r} must use ID {wanted}; "
                    f"found {actual!r}.")
    missing = [name for name in _PROTOCOL_TOKENS if name not in token_ids]
    if missing:
        raise TokenizerAssetError("OuteTTS tokenizer is missing protocol tokens: " + ", ".join(missing))


def _validate_pipeline(document: dict[str, Any], *, family: str) -> None:
    normalizer = document.get("normalizer")
    if family == "llama" and normalizer is not None:
        raise TokenizerAssetError("Llama OuteTTS requires the unnormalized Llama-3 tokenizer.")
    if family == "qwen3":
        if (not isinstance(normalizer, dict) or normalizer.get("type") != "NFC"):
            raise TokenizerAssetError("Qwen OuteTTS requires the published NFC normalizer.")
    pre_tokenizer = document.get("pre_tokenizer")
    if (not isinstance(pre_tokenizer, dict) or pre_tokenizer.get("type") != "Sequence"):
        raise TokenizerAssetError("OuteTTS requires a Split/ByteLevel pre-tokenizer sequence.")
    children = pre_tokenizer.get("pretokenizers")
    if not isinstance(children, list) or len(children) != 2:
        raise TokenizerAssetError("OuteTTS requires exactly two pre-tokenizer stages.")
    split, byte_level = children
    if (not isinstance(split, dict) or split.get("type") != "Split" or split.get("behavior") != "Isolated" or
            not isinstance(byte_level, dict) or byte_level.get("type") != "ByteLevel" or
            byte_level.get("add_prefix_space", False) or byte_level.get("use_regex", True)):
        raise TokenizerAssetError("OuteTTS tokenizer pre-tokenizer semantics are unsupported.")
    if family == "llama":
        pattern = split.get("pattern")
        regex = pattern.get("Regex") if isinstance(pattern, dict) else None
        if regex != LLAMA3_SPLIT_PATTERN:
            raise TokenizerAssetError("Llama OuteTTS uses an unexpected text split expression.")


class OuteTTSTokenizer:
    """OuteTTS protocol tokenizer backed by VoiceHub's byte-BPE engine."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        family: str,
        token_ids: dict[str, int],
        tokenizer_path: Path,
        tokenizer_config_path: Path | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self.family = family
        self._token_ids = dict(token_ids)
        self.tokenizer_path = tokenizer_path
        self.tokenizer_config_path = tokenizer_config_path
        self.bos_token_id = token_ids["<|im_start|>"]
        self.eos_token_id = token_ids["<|im_end|>"]
        self.pad_token_id = self.eos_token_id
        expected = _FAMILY_IDS[family]
        self.c1_offset = int(expected["c1_offset"])
        self.c2_offset = int(expected["c2_offset"])

    @classmethod
    def from_tokenizer_json(
        cls,
        path: str | Path,
        *,
        tokenizer_config_path: str | Path | None = None,
    ) -> OuteTTSTokenizer:
        tokenizer_path = Path(path).expanduser().resolve()
        document = _document(tokenizer_path)
        token_ids = _added_token_ids(document)
        family = _detect_family(token_ids)
        _validate_audio_tokens(token_ids, family=family)
        _validate_pipeline(document, family=family)
        tokenizer = ByteBPETokenizer.from_tokenizer_json(
            tokenizer_path,
            pad_token_id=token_ids["<|im_end|>"],
            use_regex=False,
            pretokenizer=(llama3_pretokenize if family == "llama" else qwen2_pretokenize),
            normalization=None if family == "llama" else "NFC",
        )
        config_path = (
            None if tokenizer_config_path is None else Path(tokenizer_config_path).expanduser().resolve())
        return cls(
            tokenizer,
            family=family,
            token_ids=token_ids,
            tokenizer_path=tokenizer_path,
            tokenizer_config_path=config_path,
        )

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.vocabulary_size

    @property
    def token_id_space_size(self) -> int:
        return self._tokenizer.token_id_space_size

    @property
    def special_tokens(self) -> dict[str, int]:
        return dict(self._token_ids)

    def convert_tokens_to_ids(self, spelling: str) -> int:
        if not isinstance(spelling, str):
            raise TypeError("Token spelling must be a string.")
        try:
            return self._token_ids[spelling]
        except KeyError as error:
            raise KeyError(f"Unknown OuteTTS protocol token {spelling!r}.") from error

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> Encoding:
        if not isinstance(add_special_tokens, bool):
            raise TypeError("`add_special_tokens` must be a boolean.")
        encoded = self._tokenizer.encode(
            text,
            allowed_special="all",
            disallowed_special="none",
        )
        if not add_special_tokens:
            return encoded
        return Encoding(
            input_ids=(self.bos_token_id, *encoded.input_ids),
            attention_mask=(1, *encoded.attention_mask),
            special_tokens_mask=(1, *encoded.special_tokens_mask),
        )

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def codebook_token_id(self, codebook: int, value: int) -> int:
        if codebook not in (1, 2):
            raise ValueError("OuteTTS codebook must be 1 or 2.")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("OuteTTS codec values must be integers.")
        if not 0 <= value <= 1_024:
            raise ValueError("OuteTTS codec values must be in [0, 1024].")
        return (self.c1_offset if codebook == 1 else self.c2_offset) + value

    def audio_codes_from_ids(
        self,
        token_ids,
    ) -> tuple[list[int], list[int]]:
        first: list[int] = []
        second: list[int] = []
        for raw_value in token_ids:
            value = int(raw_value)
            if self.c1_offset <= value <= self.c1_offset + 1_024:
                first.append(value - self.c1_offset)
            elif self.c2_offset <= value <= self.c2_offset + 1_024:
                second.append(value - self.c2_offset)
        frames = min(len(first), len(second))
        return first[:frames], second[:frames]

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer_destination = destination / "tokenizer.json"
        if tokenizer_destination.resolve() != self.tokenizer_path:
            shutil.copy2(self.tokenizer_path, tokenizer_destination)
        if self.tokenizer_config_path is not None:
            config_destination = destination / "tokenizer_config.json"
            if config_destination.resolve() != self.tokenizer_config_path:
                shutil.copy2(
                    self.tokenizer_config_path,
                    config_destination,
                )
        return destination


__all__ = ["OuteTTSTokenizer"]
