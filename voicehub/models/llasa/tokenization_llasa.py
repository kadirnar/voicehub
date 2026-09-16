"""Native Llama-3 byte-BPE tokenization and the LLaSA chat protocol.

The public LLaSA checkpoints extend the Llama-3 tokenizer with eight
modality markers followed by one added token for each of XCodec2's
65,536 codes.  This module reads the declarative ``tokenizer.json``
asset directly; it never imports or executes tokenizer provider code.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from voicehub.tokenization import ByteBPETokenizer, Encoding, TokenizerAssetError
from voicehub.tokenization.assets import read_bounded_asset
from voicehub.tokenization.llama3 import LLAMA3_SPLIT_PATTERN, llama3_pretokenize

LLASA_BASE_VOCABULARY_SIZE = 128_256
LLASA_MODALITY_TOKEN_COUNT = 8
LLASA_SPEECH_CODEBOOK_SIZE = 65_536
LLASA_SPEECH_TOKEN_OFFSET = (LLASA_BASE_VOCABULARY_SIZE + LLASA_MODALITY_TOKEN_COUNT)
LLASA_VOCABULARY_SIZE = (LLASA_SPEECH_TOKEN_OFFSET + LLASA_SPEECH_CODEBOOK_SIZE)

BOS_TOKEN = "<|begin_of_text|>"
EOT_TOKEN = "<|eot_id|>"
START_HEADER_TOKEN = "<|start_header_id|>"
END_HEADER_TOKEN = "<|end_header_id|>"
TEXT_GENERATION_START = "<|TEXT_GENERATION_START|>"
TEXT_GENERATION_END = "<|TEXT_GENERATION_END|>"
TEXT_UNDERSTANDING_START = "<|TEXT_UNDERSTANDING_START|>"
TEXT_UNDERSTANDING_END = "<|TEXT_UNDERSTANDING_END|>"
SPEECH_GENERATION_START = "<|SPEECH_GENERATION_START|>"
SPEECH_GENERATION_END = "<|SPEECH_GENERATION_END|>"
SPEECH_UNDERSTANDING_START = "<|SPEECH_UNDERSTANDING_START|>"
SPEECH_UNDERSTANDING_END = "<|SPEECH_UNDERSTANDING_END|>"

_EXPECTED_TOKEN_IDS = {
    BOS_TOKEN: 128_000,
    EOT_TOKEN: 128_009,
    START_HEADER_TOKEN: 128_006,
    END_HEADER_TOKEN: 128_007,
    TEXT_GENERATION_START: 128_256,
    TEXT_GENERATION_END: 128_257,
    TEXT_UNDERSTANDING_START: 128_258,
    TEXT_UNDERSTANDING_END: 128_259,
    SPEECH_GENERATION_START: 128_260,
    SPEECH_GENERATION_END: 128_261,
    SPEECH_UNDERSTANDING_START: 128_262,
    SPEECH_UNDERSTANDING_END: 128_263,
}


def _tokenizer_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(read_bounded_asset(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TokenizerAssetError(f"Invalid LLaSA tokenizer JSON: {error}.") from error
    if not isinstance(document, dict):
        raise TokenizerAssetError("LLaSA tokenizer JSON root must be an object.")
    return document


def _added_token_ids(document: Mapping[str, Any]) -> dict[str, int]:
    records = document.get("added_tokens")
    if not isinstance(records, list):
        raise TokenizerAssetError("LLaSA tokenizer has no added-token table.")
    result: dict[str, int] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        content = record.get("content")
        token_id = record.get("id")
        if (isinstance(content, str) and isinstance(token_id, int) and not isinstance(token_id, bool)):
            result[content] = token_id
    return result


class LlasaTokenizer:
    """Exact supported subset of the published LLaSA tokenizer contract."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        tokenizer_path: Path,
        tokenizer_config_path: Path | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path
        self.tokenizer_config_path = tokenizer_config_path
        self.bos_token_id = _EXPECTED_TOKEN_IDS[BOS_TOKEN]
        self.eos_token_id = _EXPECTED_TOKEN_IDS[EOT_TOKEN]
        self.pad_token_id = self.eos_token_id

    @classmethod
    def from_tokenizer_json(
        cls,
        path: str | Path,
        *,
        tokenizer_config_path: str | Path | None = None,
    ) -> LlasaTokenizer:
        tokenizer_path = Path(path).expanduser().resolve()
        document = _tokenizer_document(tokenizer_path)
        token_ids = _added_token_ids(document)
        for token, expected_id in _EXPECTED_TOKEN_IDS.items():
            if token_ids.get(token) != expected_id:
                raise TokenizerAssetError(
                    f"LLaSA tokenizer token {token!r} must use ID "
                    f"{expected_id}; found {token_ids.get(token)!r}.")
        for code in range(LLASA_SPEECH_CODEBOOK_SIZE):
            spelling = f"<|s_{code}|>"
            expected_id = LLASA_SPEECH_TOKEN_OFFSET + code
            if token_ids.get(spelling) != expected_id:
                raise TokenizerAssetError(
                    f"LLaSA tokenizer token {spelling!r} must use ID "
                    f"{expected_id}; found {token_ids.get(spelling)!r}.")
        tokenizer = ByteBPETokenizer.from_tokenizer_json(
            tokenizer_path,
            pad_token_id=_EXPECTED_TOKEN_IDS[EOT_TOKEN],
            use_regex=False,
            pretokenizer=llama3_pretokenize,
        )
        if tokenizer.token_id_space_size != LLASA_VOCABULARY_SIZE:
            raise TokenizerAssetError(
                "LLaSA tokenizer ID space must contain exactly "
                f"{LLASA_VOCABULARY_SIZE} rows; found "
                f"{tokenizer.token_id_space_size}.")
        config_path = (
            None if tokenizer_config_path is None else Path(tokenizer_config_path).expanduser().resolve())
        return cls(
            tokenizer,
            tokenizer_path=tokenizer_path,
            tokenizer_config_path=config_path,
        )

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.vocabulary_size

    @property
    def token_id_space_size(self) -> int:
        return self._tokenizer.token_id_space_size

    def __len__(self) -> int:
        return self.vocabulary_size

    @staticmethod
    def speech_code_to_token_id(code: int) -> int:
        if isinstance(code, bool) or not isinstance(code, int):
            raise TypeError("XCodec2 codes must be integers.")
        if not 0 <= code < LLASA_SPEECH_CODEBOOK_SIZE:
            raise ValueError(f"XCodec2 codes must be in [0, "
                             f"{LLASA_SPEECH_CODEBOOK_SIZE - 1}].")
        return LLASA_SPEECH_TOKEN_OFFSET + code

    @staticmethod
    def token_id_to_speech_code(token_id: int) -> int:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("LLaSA token IDs must be integers.")
        code = token_id - LLASA_SPEECH_TOKEN_OFFSET
        if not 0 <= code < LLASA_SPEECH_CODEBOOK_SIZE:
            raise ValueError(f"Token ID {token_id} is not an LLaSA speech token.")
        return code

    def convert_tokens_to_ids(self, token: str) -> int:
        if not isinstance(token, str):
            raise TypeError("`token` must be a string.")
        known = _EXPECTED_TOKEN_IDS.get(token)
        if known is not None:
            return known
        if token.startswith("<|s_") and token.endswith("|>"):
            try:
                code = int(token[4:-2])
            except ValueError as error:
                raise ValueError(f"Malformed LLaSA speech token {token!r}.") from error
            return self.speech_code_to_token_id(code)
        raise KeyError(f"Unknown LLaSA protocol token {token!r}.")

    def convert_ids_to_tokens(
        self,
        token_ids: int | Sequence[int],
    ) -> str | list[str]:
        scalar = isinstance(token_ids, int) and not isinstance(token_ids, bool)
        values = (token_ids, ) if scalar else tuple(token_ids)
        inverse = {value: token for token, value in _EXPECTED_TOKEN_IDS.items()}
        output = []
        for token_id in values:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("Token IDs must be integers.")
            if token_id in inverse:
                output.append(inverse[token_id])
            elif LLASA_SPEECH_TOKEN_OFFSET <= token_id < LLASA_VOCABULARY_SIZE:
                output.append(f"<|s_{token_id - LLASA_SPEECH_TOKEN_OFFSET}|>")
            else:
                output.append(self._tokenizer.decode((token_id, )))
        return output[0] if scalar else output

    @staticmethod
    def _validate_messages(messages: Sequence[Mapping[str, Any]], ) -> tuple[dict[str, str], ...]:
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise TypeError("`messages` must be a sequence of mappings.")
        normalized = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(f"LLaSA message {index} must be a mapping.")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"LLaSA message {index} has unsupported role {role!r}.")
            if not isinstance(content, str):
                raise TypeError(f"LLaSA message {index} content must be a string.")
            normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError("LLaSA chat messages cannot be empty.")
        return tuple(normalized)

    @classmethod
    def format_chat(
        cls,
        messages: Sequence[Mapping[str, Any]],
        *,
        continue_final_message: bool = False,
        add_generation_prompt: bool = False,
        date_string: str | None = None,
    ) -> str:
        """Render the published template, with an optional date for replay."""
        if date_string is None:
            date_string = datetime.now().strftime("%d %b %Y")
        if not isinstance(date_string, str) or not date_string.strip():
            raise ValueError("`date_string` must be a non-empty string or None.")
        normalized = list(cls._validate_messages(messages))
        if continue_final_message and add_generation_prompt:
            raise ValueError(
                "`continue_final_message` and `add_generation_prompt` are "
                "mutually exclusive.")
        if continue_final_message and normalized[-1]["role"] != "assistant":
            raise ValueError("LLaSA continuation requires a final assistant message.")
        if normalized[0]["role"] == "system":
            system_message = normalized.pop(0)["content"].strip()
        else:
            system_message = ""
        rendered = (
            BOS_TOKEN + START_HEADER_TOKEN + "system" + END_HEADER_TOKEN + "\n\n" +
            "Cutting Knowledge Date: December 2023\n" + f"Today Date: {date_string}\n\n" + system_message +
            EOT_TOKEN)
        for message in normalized:
            rendered += (
                START_HEADER_TOKEN + message["role"] + END_HEADER_TOKEN + "\n\n" +
                message["content"].strip() + EOT_TOKEN)
        if continue_final_message:
            rendered = rendered[:-len(EOT_TOKEN)]
        elif add_generation_prompt:
            rendered += (START_HEADER_TOKEN + "assistant" + END_HEADER_TOKEN + "\n\n")
        return rendered

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

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tokenize: bool = True,
        return_tensors: str | None = None,
        continue_final_message: bool = False,
        add_generation_prompt: bool = False,
        date_string: str | None = None,
    ):
        rendered = self.format_chat(
            messages,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            date_string=date_string,
        )
        if not tokenize:
            if return_tensors is not None:
                raise ValueError("`return_tensors` requires `tokenize=True`.")
            return rendered
        ids = list(self.encode(rendered).input_ids)
        if return_tensors is None:
            return ids
        if return_tensors != "pt":
            raise ValueError("Native LLaSA tokenization supports only PyTorch tensors.")
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

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
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer_destination = (destination / "tokenizer.json").resolve()
        if tokenizer_destination != self.tokenizer_path.resolve():
            shutil.copy2(self.tokenizer_path, tokenizer_destination)
        if self.tokenizer_config_path is not None:
            config_destination = (destination / "tokenizer_config.json").resolve()
            if config_destination != self.tokenizer_config_path.resolve():
                shutil.copy2(
                    self.tokenizer_config_path,
                    config_destination,
                )
        return destination.resolve()


__all__ = [
    "BOS_TOKEN",
    "END_HEADER_TOKEN",
    "EOT_TOKEN",
    "LLAMA3_SPLIT_PATTERN",
    "LLASA_BASE_VOCABULARY_SIZE",
    "LLASA_MODALITY_TOKEN_COUNT",
    "LLASA_SPEECH_CODEBOOK_SIZE",
    "LLASA_SPEECH_TOKEN_OFFSET",
    "LLASA_VOCABULARY_SIZE",
    "LlasaTokenizer",
    "SPEECH_GENERATION_END",
    "SPEECH_GENERATION_START",
    "START_HEADER_TOKEN",
    "TEXT_UNDERSTANDING_END",
    "TEXT_UNDERSTANDING_START",
    "llama3_pretokenize",
]
