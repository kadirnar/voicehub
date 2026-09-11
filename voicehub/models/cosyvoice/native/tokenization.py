"""Dependency-free Qwen2 byte-BPE tokenizer for CosyVoice 3."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor

from voicehub.models.asr_qwen3.native.tokenization import _load_merges, _load_vocabulary, qwen2_pretokenize
from voicehub.tokenization import ByteBPETokenizer, Encoding
from voicehub.tokenization.assets import (
    DEFAULT_MAX_ASSET_BYTES,
    DEFAULT_MAX_MERGES,
    DEFAULT_MAX_TOKENS,
    TokenizerAssetError,
    read_bounded_asset,
)

END_OF_TEXT = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
END_OF_PROMPT = "<|endofprompt|>"
PUBLISHED_SPECIAL_IDS = {
    END_OF_TEXT: 151_643,
    IM_START: 151_644,
    IM_END: 151_645,
    END_OF_PROMPT: 151_646,
}


def _tokenizer_metadata(path: Path, *, max_bytes: int) -> dict:
    payload = read_bounded_asset(path, max_bytes=max_bytes)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TokenizerAssetError(f"Invalid tokenizer metadata {path}: {error}.") from error
    if not isinstance(value, dict):
        raise TokenizerAssetError("Tokenizer metadata must be a JSON object.")
    return value


class CosyVoiceTextTokenizer:
    """Qwen byte BPE plus CosyVoice control/event tokens."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        vocab_path: Path,
        merges_path: Path,
        tokenizer_config_path: Path,
        metadata: dict,
    ) -> None:
        self._tokenizer = tokenizer
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        self.tokenizer_config_path = tokenizer_config_path
        self.metadata = dict(metadata)

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        tokenizer_config_path: str | Path,
        *,
        validate_published_ids: bool = True,
        max_asset_bytes: int = DEFAULT_MAX_ASSET_BYTES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_merges: int = DEFAULT_MAX_MERGES,
    ) -> CosyVoiceTextTokenizer:
        vocab_path = Path(vocab_path).expanduser().resolve()
        merges_path = Path(merges_path).expanduser().resolve()
        tokenizer_config_path = Path(tokenizer_config_path).expanduser().resolve()
        vocabulary = _load_vocabulary(
            vocab_path,
            max_bytes=max_asset_bytes,
            max_tokens=max_tokens,
        )
        merges = _load_merges(
            merges_path,
            max_bytes=max_asset_bytes,
            max_merges=max_merges,
        )
        metadata = _tokenizer_metadata(
            tokenizer_config_path,
            max_bytes=max_asset_bytes,
        )
        special: dict[str, int] = {}
        for raw_token, token_id in vocabulary.items():
            try:
                spelling = raw_token.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if (spelling.startswith(("<|", "[", "</")) or
                    spelling in {"<strong>", "</strong>", "<laughter>", "</laughter>"}):
                special[spelling] = token_id
        records = metadata.get("added_tokens_decoder", {})
        if isinstance(records, dict):
            for raw_id, record in records.items():
                if isinstance(record, dict) and isinstance(record.get("content"), str):
                    special[record["content"]] = int(raw_id)
        if validate_published_ids:
            for spelling, expected in PUBLISHED_SPECIAL_IDS.items():
                actual = special.get(spelling)
                if actual != expected:
                    raise TokenizerAssetError(
                        f"CosyVoice token {spelling!r} must use ID "
                        f"{expected}; found {actual!r}.")
        tokenizer = ByteBPETokenizer(
            vocabulary,
            merges=merges,
            special_tokens=special,
            pad_token_id=special.get(END_OF_TEXT),
            add_prefix_space=bool(metadata.get("add_prefix_space", False)),
            use_regex=True,
            pretokenizer=qwen2_pretokenize,
            padding_side="left",
        )
        return cls(
            tokenizer,
            vocab_path=vocab_path,
            merges_path=merges_path,
            tokenizer_config_path=tokenizer_config_path,
            metadata=metadata,
        )

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def token_id_space_size(self) -> int:
        return self._tokenizer.token_id_space_size

    def encode(self, text: str) -> Encoding:
        if not isinstance(text, str) or not text:
            raise ValueError("CosyVoice text must be a non-empty string.")
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

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            errors=str(self.metadata.get("errors", "replace")),
        )

    def instruction_tokens(
        self,
        instruction: str | None,
        *,
        device: str | torch.device | None = None,
    ) -> Tensor:
        if instruction is None:
            prompt = f"You are a helpful assistant.{END_OF_PROMPT}"
        else:
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("CosyVoice instruction must be non-empty or None.")
            prompt = (
                instruction if END_OF_PROMPT in instruction else
                f"You are a helpful assistant. {instruction.strip()}{END_OF_PROMPT}")
        return self.encode_tensor(prompt, device=device)

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        for source, filename in (
            (self.vocab_path, "vocab.json"),
            (self.merges_path, "merges.txt"),
            (self.tokenizer_config_path, "tokenizer_config.json"),
        ):
            destination = target / filename
            if source != destination.resolve():
                shutil.copy2(source, destination)
        return target


__all__ = [
    "CosyVoiceTextTokenizer",
    "END_OF_PROMPT",
    "END_OF_TEXT",
    "IM_END",
    "IM_START",
    "PUBLISHED_SPECIAL_IDS",
]
