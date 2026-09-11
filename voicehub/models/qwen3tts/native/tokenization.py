"""Dependency-free Qwen2 byte-BPE text tokenizer for Qwen3-TTS."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from voicehub.models.asr_qwen3.native.tokenization import _load_merges, _load_vocabulary, qwen2_pretokenize
from voicehub.tokenization.assets import (
    DEFAULT_MAX_ASSET_BYTES,
    DEFAULT_MAX_MERGES,
    DEFAULT_MAX_TOKENS,
    TokenizerAssetError,
    read_bounded_asset,
)
from voicehub.tokenization.base import BatchEncoding, Encoding
from voicehub.tokenization.byte_bpe import ByteBPETokenizer

END_OF_TEXT = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TTS_PAD = "<tts_pad>"
TTS_BOS = "<tts_text_bos>"
TTS_EOS = "<tts_text_eod>"

EXPECTED_TTS_TOKEN_IDS = {
    END_OF_TEXT: 151_643,
    IM_START: 151_644,
    IM_END: 151_645,
    TTS_PAD: 151_671,
    TTS_BOS: 151_672,
    TTS_EOS: 151_673,
}


def _added_tokens(
    path: Path,
    *,
    max_bytes: int,
) -> tuple[dict[str, int], dict[str, int], Mapping[str, Any]]:
    payload = read_bounded_asset(path, max_bytes=max_bytes)
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TokenizerAssetError(f"Invalid tokenizer configuration {path.name}: {error}.") from error
    if not isinstance(document, dict):
        raise TokenizerAssetError("Qwen tokenizer configuration must be an object.")
    records = document.get("added_tokens_decoder")
    if not isinstance(records, dict) or not records:
        raise TokenizerAssetError("Qwen tokenizer has no added-token decoder.")
    special: dict[str, int] = {}
    added: dict[str, int] = {}
    seen_ids: set[int] = set()
    for raw_id, record in records.items():
        if not isinstance(record, dict):
            raise TokenizerAssetError("Qwen added-token records must be objects.")
        try:
            token_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise TokenizerAssetError("Qwen added-token ID is invalid.") from error
        content = record.get("content")
        if (not isinstance(content, str) or not content or token_id < 0 or token_id in seen_ids):
            raise TokenizerAssetError("Qwen added-token record is invalid.")
        if record.get("lstrip") or record.get("rstrip"):
            raise TokenizerAssetError("Whitespace-stripping Qwen tokens are unsupported.")
        destination = special if record.get("special") is True else added
        if content in special or content in added:
            raise TokenizerAssetError("Qwen added-token spelling is duplicated.")
        destination[content] = token_id
        seen_ids.add(token_id)
    for spelling, expected in EXPECTED_TTS_TOKEN_IDS.items():
        actual = special.get(spelling, added.get(spelling))
        if actual != expected:
            raise TokenizerAssetError(f"Qwen token {spelling!r} must use ID {expected}; found {actual!r}.")
    return special, added, document


class Qwen3TTSTextTokenizer:
    """Immutable native tokenizer for Qwen3-TTS text and control prompts."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        vocab_path: Path,
        merges_path: Path,
        tokenizer_config_path: Path,
        tokenizer_config: Mapping[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        self.tokenizer_config_path = tokenizer_config_path
        self.tokenizer_config = dict(tokenizer_config)

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        tokenizer_config_path: str | Path,
        *,
        max_asset_bytes: int = DEFAULT_MAX_ASSET_BYTES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_merges: int = DEFAULT_MAX_MERGES,
    ) -> Qwen3TTSTextTokenizer:
        vocab = Path(vocab_path).expanduser().resolve()
        merges = Path(merges_path).expanduser().resolve()
        tokenizer_config = Path(tokenizer_config_path).expanduser().resolve()
        vocabulary = _load_vocabulary(
            vocab,
            max_bytes=max_asset_bytes,
            max_tokens=max_tokens,
        )
        merge_pairs = _load_merges(
            merges,
            max_bytes=max_asset_bytes,
            max_merges=max_merges,
        )
        special, added, values = _added_tokens(
            tokenizer_config,
            max_bytes=max_asset_bytes,
        )
        tokenizer = ByteBPETokenizer(
            vocabulary,
            merges=merge_pairs,
            special_tokens=special,
            added_tokens=added,
            pad_token_id=EXPECTED_TTS_TOKEN_IDS[END_OF_TEXT],
            add_prefix_space=bool(values.get("add_prefix_space", False)),
            use_regex=True,
            pretokenizer=qwen2_pretokenize,
            padding_side="left",
        )
        if tokenizer.token_id_space_size > 151_936:
            raise TokenizerAssetError("Qwen tokenizer IDs exceed the talker text vocabulary.")
        return cls(
            tokenizer,
            vocab_path=vocab,
            merges_path=merges,
            tokenizer_config_path=tokenizer_config,
            tokenizer_config=values,
        )

    @property
    def pad_token_id(self) -> int:
        return EXPECTED_TTS_TOKEN_IDS[END_OF_TEXT]

    @property
    def eos_token_id(self) -> int:
        return EXPECTED_TTS_TOKEN_IDS[IM_END]

    @property
    def token_id_space_size(self) -> int:
        return self._tokenizer.token_id_space_size

    def encode(self, text: str) -> Encoding:
        return self._tokenizer.encode(
            text,
            allowed_special="all",
            disallowed_special=(),
        )

    def encode_tensor(
        self,
        text: str,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        return torch.tensor(
            [self.encode(text).input_ids],
            dtype=torch.long,
            device=device,
        )

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        padding: bool | str = True,
    ) -> BatchEncoding:
        return self._tokenizer.encode_batch(
            texts,
            padding=padding,
            allowed_special="all",
            disallowed_special=(),
        )

    def decode(
        self,
        token_ids: Iterable[int] | Encoding,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        tolist = getattr(token_ids, "tolist", None)
        if callable(tolist):
            token_ids = tolist()
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            errors=str(self.tokenizer_config.get("errors", "replace")),
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        for source, filename in (
            (self.vocab_path, "vocab.json"),
            (self.merges_path, "merges.txt"),
            (self.tokenizer_config_path, "tokenizer_config.json"),
        ):
            destination = target / filename
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        return target


__all__ = [
    "END_OF_TEXT",
    "EXPECTED_TTS_TOKEN_IDS",
    "IM_END",
    "IM_START",
    "Qwen3TTSTextTokenizer",
    "TTS_BOS",
    "TTS_EOS",
    "TTS_PAD",
]
