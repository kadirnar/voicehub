"""Dependency-free Qwen byte-BPE tokenizer used by every MOSS-TTS release."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path

from voicehub.models.asr_qwen3.native.tokenization import _load_merges, _load_vocabulary, qwen2_pretokenize
from voicehub.models.mosstts.native.configuration import MossTTSConfig
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
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"
AUDIO_USER_SLOT = "<|audio_user_slot|>"
AUDIO_ASSISTANT_SLOT = "<|audio_assistant_gen_slot|>"
AUDIO_DELAY_SLOT = "<|audio_assistant_delay_slot|>"
REALTIME_AUDIO_PAD = "<|audio_pad|>"
REALTIME_TEXT_PAD = "<|text_pad|>"


def _metadata(path: Path, *, max_bytes: int) -> dict[str, object]:
    payload = read_bounded_asset(path, max_bytes=max_bytes)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise TokenizerAssetError(f"Invalid MOSS tokenizer metadata {path}: {error}.") from error
    if not isinstance(value, dict):
        raise TokenizerAssetError("MOSS tokenizer metadata must be a JSON object.")
    return value


def _added_tokens(metadata: Mapping[str, object], ) -> tuple[dict[str, int], dict[str, int]]:
    records = metadata.get("added_tokens_decoder")
    if not isinstance(records, dict) or not records:
        raise TokenizerAssetError("MOSS tokenizer metadata has no added-token decoder.")
    special: dict[str, int] = {}
    added: dict[str, int] = {}
    used_ids: set[int] = set()
    for raw_id, raw_record in records.items():
        if not isinstance(raw_record, dict):
            raise TokenizerAssetError("MOSS added-token records must be objects.")
        try:
            token_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise TokenizerAssetError(f"Invalid MOSS added-token ID {raw_id!r}.") from error
        spelling = raw_record.get("content")
        if not isinstance(spelling, str) or not spelling:
            raise TokenizerAssetError("MOSS added-token spellings must be non-empty strings.")
        if raw_record.get("lstrip") or raw_record.get("rstrip"):
            raise TokenizerAssetError("Whitespace-stripping MOSS added tokens are unsupported.")
        if spelling in special or spelling in added or token_id in used_ids:
            raise TokenizerAssetError("MOSS tokenizer metadata contains duplicate added tokens.")
        destination = special if raw_record.get("special") is True else added
        destination[spelling] = token_id
        used_ids.add(token_id)
    return special, added


def _expected_control_ids(config: MossTTSConfig) -> dict[str, int]:
    expected = {
        END_OF_TEXT: config.pad_token_id,
        IM_START: config.im_start_token_id,
        IM_END: config.im_end_token_id,
    }
    if config.variant == "realtime":
        if config.reference_audio_pad_token_id is None:
            raise ValueError("Realtime MOSS config has no reference-audio pad token.")
        if config.text_pad_token_id is None:
            raise ValueError("Realtime MOSS config has no text-pad token.")
        expected.update({
            REALTIME_AUDIO_PAD: config.reference_audio_pad_token_id,
            REALTIME_TEXT_PAD: config.text_pad_token_id,
        })
        return expected
    expected.update({
        AUDIO_START: config.audio_start_token_id,
        AUDIO_END: config.audio_end_token_id,
    })
    if config.variant == "local_v1_5":
        # Local v1.5 writes numeric slot IDs directly into the multichannel
        # matrix.  Those IDs retain Qwen's vision-pad spellings in the
        # tokenizer asset and must not be reinterpreted as text controls.
        return expected
    if config.audio_user_slot_token_id is not None:
        expected[AUDIO_USER_SLOT] = config.audio_user_slot_token_id
    if config.audio_assistant_slot_token_id is not None:
        expected[AUDIO_ASSISTANT_SLOT] = config.audio_assistant_slot_token_id
    if config.audio_assistant_delay_slot_token_id is not None:
        expected[AUDIO_DELAY_SLOT] = (config.audio_assistant_delay_slot_token_id)
    return expected


class MossTextTokenizer:
    """Exact Qwen2 byte-BPE plus release-specific MOSS control tokens."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        vocab_path: Path,
        merges_path: Path,
        tokenizer_config_path: Path,
        metadata: Mapping[str, object],
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
        model_config: MossTTSConfig | None = None,
        max_asset_bytes: int = DEFAULT_MAX_ASSET_BYTES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_merges: int = DEFAULT_MAX_MERGES,
    ) -> MossTextTokenizer:
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
        values = _metadata(tokenizer_config, max_bytes=max_asset_bytes)
        special, added = _added_tokens(values)
        all_added = {**special, **added}
        if model_config is not None:
            for spelling, expected in _expected_control_ids(model_config).items():
                actual = all_added.get(spelling)
                # Older Delay releases spell the three audio slot tokens
                # differently in `added_tokens.json`, but the authoritative
                # tokenizer_config decoder retains their configured spelling.
                if actual != expected:
                    raise TokenizerAssetError(
                        f"MOSS control token {spelling!r} must use ID "
                        f"{expected}; found {actual!r}.")
            if (model_config.language_config.vocab_size < max(all_added.values(), default=-1) + 1):
                raise TokenizerAssetError("MOSS tokenizer IDs exceed the model text vocabulary.")
        pad_id = all_added.get(END_OF_TEXT)
        if pad_id is None:
            raise TokenizerAssetError("MOSS tokenizer has no end-of-text padding token.")
        tokenizer = ByteBPETokenizer(
            vocabulary,
            merges=merge_pairs,
            special_tokens=special,
            added_tokens=added,
            pad_token_id=pad_id,
            add_prefix_space=bool(values.get("add_prefix_space", False)),
            use_regex=True,
            pretokenizer=qwen2_pretokenize,
            padding_side="left",
        )
        return cls(
            tokenizer,
            vocab_path=vocab,
            merges_path=merges,
            tokenizer_config_path=tokenizer_config,
            metadata=values,
        )

    @property
    def pad_token_id(self) -> int:
        value = self._tokenizer.pad_token_id
        if value is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("MOSS tokenizer padding ID was not configured.")
        return value

    @property
    def token_id_space_size(self) -> int:
        return self._tokenizer.token_id_space_size

    def token_to_id(self, spelling: str) -> int:
        if spelling in self._tokenizer.special_tokens:
            return self._tokenizer.special_tokens[spelling]
        if spelling in self._tokenizer.added_tokens:
            return self._tokenizer.added_tokens[spelling]
        encoded = self.encode(spelling)
        if len(encoded.input_ids) != 1:
            raise KeyError(f"MOSS token {spelling!r} is not one vocabulary entry.")
        return encoded.input_ids[0]

    def id_to_token(self, token_id: int) -> str:
        for spelling, candidate in self._tokenizer.special_tokens.items():
            if candidate == token_id:
                return spelling
        for spelling, candidate in self._tokenizer.added_tokens.items():
            if candidate == token_id:
                return spelling
        return self.decode((token_id, ), skip_special_tokens=False)

    def encode(self, text: str) -> Encoding:
        if not isinstance(text, str):
            raise TypeError("MOSS text must be a string.")
        return self._tokenizer.encode(
            text,
            allowed_special="all",
            disallowed_special=(),
        )

    def encode_ids(self, text: str) -> list[int]:
        return list(self.encode(text).input_ids)

    def decode(
        self,
        token_ids: Iterable[int] | Encoding,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        values: Iterable[int]
        if isinstance(token_ids, Encoding):
            values = token_ids.input_ids
        else:
            tolist = getattr(token_ids, "tolist", None)
            values = tolist() if callable(tolist) else token_ids
        return self._tokenizer.decode(
            values,
            skip_special_tokens=skip_special_tokens,
            errors=str(self.metadata.get("errors", "replace")),
        )

    def apply_chat_template(
        self,
        *,
        role: str,
        content: str,
        add_generation_prompt: bool = False,
    ) -> str:
        if role not in {"system", "user", "assistant", "context"}:
            raise ValueError(f"Unsupported MOSS conversation role {role!r}.")
        if not isinstance(content, str):
            raise TypeError("MOSS message content must be a string.")
        rendered = f"{IM_START}{role}\n{content}{IM_END}\n"
        if add_generation_prompt:
            rendered += f"{IM_START}assistant\n"
        return rendered

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
    "AUDIO_ASSISTANT_SLOT",
    "AUDIO_DELAY_SLOT",
    "AUDIO_END",
    "AUDIO_START",
    "AUDIO_USER_SLOT",
    "END_OF_TEXT",
    "IM_END",
    "IM_START",
    "MossTextTokenizer",
    "REALTIME_AUDIO_PAD",
    "REALTIME_TEXT_PAD",
]
