"""Native byte-BPE tokenizer boundary for Granite Speech."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from voicehub.hub import write_json_file
from voicehub.tokenization import ByteBPETokenizer, Encoding
from voicehub.tokenization.assets import read_bounded_asset
from voicehub.tokenization.llama3 import llama3_pretokenize

AUDIO_TOKEN = "<|audio|>"
PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|end_of_text|>"
DEFAULT_AUDIO_TOKEN_ID = 100_352
DEFAULT_PAD_TOKEN_ID = 100_256
DEFAULT_EOS_TOKEN_ID = 100_257


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        values = json.loads(read_bounded_asset(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid Granite Speech {name}: {error}.") from error
    if not isinstance(values, dict):
        raise ValueError(f"Granite Speech {name} must contain a JSON object.")
    return values


class GraniteSpeechTokenizer:
    """Checkpoint-bound Llama-3-style byte BPE with explicit special IDs."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        tokenizer_json_path: Path,
        tokenizer_config_path: Path | None = None,
        special_tokens_map_path: Path | None = None,
        added_tokens_path: Path | None = None,
        chat_template_path: Path | None = None,
    ) -> None:
        if not isinstance(tokenizer, ByteBPETokenizer):
            raise TypeError("`tokenizer` must be a ByteBPETokenizer.")
        self._tokenizer = tokenizer
        self.tokenizer_json_path = tokenizer_json_path
        self.tokenizer_config_path = tokenizer_config_path
        self.special_tokens_map_path = special_tokens_map_path
        self.added_tokens_path = added_tokens_path
        self.chat_template_path = chat_template_path
        declared = {
            **dict(tokenizer.special_tokens),
            **dict(tokenizer.added_tokens),
        }
        required = {
            AUDIO_TOKEN: DEFAULT_AUDIO_TOKEN_ID,
            PAD_TOKEN: DEFAULT_PAD_TOKEN_ID,
            EOS_TOKEN: DEFAULT_EOS_TOKEN_ID,
        }
        for token, expected in required.items():
            actual = declared.get(token)
            if actual != expected:
                raise ValueError(
                    f"Granite Speech tokenizer declares {token!r} as "
                    f"{actual!r}; expected {expected}.")

    @classmethod
    def from_files(
        cls,
        tokenizer_json: str | Path,
        *,
        tokenizer_config: str | Path | None = None,
        special_tokens_map: str | Path | None = None,
        added_tokens: str | Path | None = None,
        chat_template: str | Path | None = None,
    ) -> GraniteSpeechTokenizer:
        tokenizer_path = Path(tokenizer_json).expanduser().resolve()
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Granite Speech tokenizer was not found: {tokenizer_path}.")
        config_path = (
            Path(tokenizer_config).expanduser().resolve() if tokenizer_config is not None else None)
        if config_path is not None:
            values = _read_json(config_path, name="tokenizer_config.json")
            if values.get("add_bos_token", False) is not False:
                raise ValueError("Granite Speech tokenizer must not prepend a BOS token.")
            if values.get("add_prefix_space", False) is not False:
                raise ValueError("Granite Speech tokenizer must not add a prefix space.")
        tokenizer = ByteBPETokenizer.from_tokenizer_json(
            tokenizer_path,
            pad_token_id=DEFAULT_PAD_TOKEN_ID,
            padding_side="left",
            pretokenizer=llama3_pretokenize,
        )
        return cls(
            tokenizer,
            tokenizer_json_path=tokenizer_path,
            tokenizer_config_path=config_path,
            special_tokens_map_path=(
                Path(special_tokens_map).expanduser().resolve() if special_tokens_map is not None else None),
            added_tokens_path=(
                Path(added_tokens).expanduser().resolve() if added_tokens is not None else None),
            chat_template_path=(
                Path(chat_template).expanduser().resolve() if chat_template is not None else None),
        )

    @property
    def audio_token_id(self) -> int:
        return DEFAULT_AUDIO_TOKEN_ID

    @property
    def pad_token_id(self) -> int:
        return DEFAULT_PAD_TOKEN_ID

    @property
    def eos_token_id(self) -> int:
        return DEFAULT_EOS_TOKEN_ID

    @property
    def token_id_space_size(self) -> int:
        return self._tokenizer.token_id_space_size

    @property
    def special_token_ids(self) -> frozenset[int]:
        return frozenset(self._tokenizer.special_tokens.values())

    def encode_prompt(self, text: str) -> Encoding:
        return self._tokenizer.encode(
            text,
            allowed_special={AUDIO_TOKEN},
            disallowed_special="all",
        )

    def encode_transcript(self, text: str) -> Encoding:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Granite Speech transcripts must be non-empty strings.")
        return self._tokenizer.encode(
            text.strip(),
            allowed_special="none",
            disallowed_special="all",
        )

    def decode(
        self,
        token_ids: Iterable[int] | Encoding,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        assets = (
            (self.tokenizer_json_path, "tokenizer.json"),
            (self.tokenizer_config_path, "tokenizer_config.json"),
            (self.special_tokens_map_path, "special_tokens_map.json"),
            (self.added_tokens_path, "added_tokens.json"),
            (self.chat_template_path, "chat_template.jinja"),
        )
        for source, filename in assets:
            if source is None:
                continue
            if not source.is_file():
                raise FileNotFoundError(f"Granite Speech tokenizer asset was not found: {source}.")
            destination = target / filename
            if source != destination.resolve():
                shutil.copyfile(source, destination)
        if not (target / "tokenizer_config.json").is_file():
            write_json_file(
                target / "tokenizer_config.json",
                {
                    "add_bos_token": False,
                    "add_prefix_space": False,
                    "audio_token": AUDIO_TOKEN,
                    "bos_token": EOS_TOKEN,
                    "eos_token": EOS_TOKEN,
                    "pad_token": PAD_TOKEN,
                    "tokenizer_class": "VoiceHubByteBPETokenizer",
                },
            )
        if not (target / "special_tokens_map.json").is_file():
            write_json_file(
                target / "special_tokens_map.json",
                {
                    "bos_token": EOS_TOKEN,
                    "eos_token": EOS_TOKEN,
                    "pad_token": PAD_TOKEN,
                },
            )
        if not (target / "added_tokens.json").is_file():
            write_json_file(
                target / "added_tokens.json",
                {AUDIO_TOKEN: self.audio_token_id},
            )
        if not (target / "chat_template.jinja").is_file():
            (target / "chat_template.jinja").write_text(
                (
                    "{% for message in messages %}"
                    "{% if message['role'] == 'user' %}"
                    "USER: {{ message['content'] }}\n ASSISTANT:"
                    "{% elif message['role'] == 'assistant' %}"
                    "{{ message['content'] }}"
                    "{% endif %}{% endfor %}"),
                encoding="utf-8",
            )
        return target


__all__ = [
    "AUDIO_TOKEN",
    "DEFAULT_AUDIO_TOKEN_ID",
    "DEFAULT_EOS_TOKEN_ID",
    "DEFAULT_PAD_TOKEN_ID",
    "EOS_TOKEN",
    "GraniteSpeechTokenizer",
    "PAD_TOKEN",
]
