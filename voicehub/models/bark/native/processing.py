"""Dependency-free Bark WordPiece tokenization and speaker prompts."""

from __future__ import annotations

import ast
import json
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.hub import resolve_pretrained_file

_MAX_VOCAB_BYTES = 16 * 1024 * 1024
_MAX_PRESET_BYTES = 64 * 1024 * 1024
_PRESET_SHAPES = {
    "semantic_prompt": 1,
    "coarse_prompt": 2,
    "fine_prompt": 2,
}


class BarkWordPieceTokenizer:
    """BERT multilingual-cased basic tokenizer plus greedy WordPiece."""

    def __init__(
        self,
        vocabulary: Sequence[str],
        *,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        do_lower_case: bool = False,
        tokenize_chinese_chars: bool = True,
        max_input_chars_per_word: int = 100,
    ) -> None:
        if (isinstance(vocabulary, (str, bytes)) or not isinstance(vocabulary, Sequence) or not vocabulary):
            raise ValueError("Bark vocabulary must be a non-empty token sequence.")
        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError("Bark vocabulary contains duplicate tokens.")
        if any(not isinstance(token, str) or not token for token in vocabulary):
            raise ValueError("Bark vocabulary tokens must be non-empty strings.")
        self.vocabulary = tuple(vocabulary)
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        for name, token in (("unk_token", unk_token), ("pad_token", pad_token)):
            if token not in self.token_to_id:
                raise ValueError(f"Bark vocabulary is missing {name}={token!r}.")
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.unk_token_id = self.token_to_id[unk_token]
        self.pad_token_id = self.token_to_id[pad_token]
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.max_input_chars_per_word = max_input_chars_per_word

    @classmethod
    def from_vocab_file(
        cls,
        path: str | Path,
        **options: Any,
    ) -> BarkWordPieceTokenizer:
        source = Path(path)
        if source.stat().st_size > _MAX_VOCAB_BYTES:
            raise ValueError("Bark vocabulary exceeds the safe size limit.")
        payload = source.read_bytes()
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Bark vocabulary is not valid UTF-8.") from error
        tokens = text.splitlines()
        if tokens and tokens[-1] == "":
            tokens.pop()
        return cls(tokens, **options)

    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("Bark text must be a string.")
        cleaned = self._clean_text(text)
        if self.tokenize_chinese_chars:
            cleaned = self._tokenize_chinese(cleaned)
        basic_tokens: list[str] = []
        for token in cleaned.strip().split():
            if self.do_lower_case:
                token = self._strip_accents(token.lower())
            basic_tokens.extend(self._split_punctuation(token))
        pieces: list[str] = []
        for token in basic_tokens:
            pieces.extend(self._wordpiece(token))
        return pieces

    def encode(
        self,
        text: str,
        *,
        max_length: int = 256,
    ) -> tuple[list[int], list[int]]:
        if (isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0):
            raise ValueError("Bark maximum token length must be positive.")
        ids = [self.token_to_id.get(token, self.unk_token_id) for token in self.tokenize(text)][:max_length]
        attention = [1] * len(ids)
        padding = max_length - len(ids)
        ids.extend([self.pad_token_id] * padding)
        attention.extend([0] * padding)
        return ids, attention

    @staticmethod
    def _clean_text(text: str) -> str:
        output: list[str] = []
        for character in text:
            codepoint = ord(character)
            category = unicodedata.category(character)
            if codepoint in {0, 0xFFFD} or category.startswith("C"):
                continue
            output.append(" " if character.isspace() else character)
        return "".join(output)

    @staticmethod
    def _strip_accents(text: str) -> str:
        return "".join(
            character for character in unicodedata.normalize("NFD", text)
            if unicodedata.category(character) != "Mn")

    @staticmethod
    def _is_chinese(codepoint: int) -> bool:
        return (
            0x4E00 <= codepoint <= 0x9FFF or 0x3400 <= codepoint <= 0x4DBF or
            0x20000 <= codepoint <= 0x2A6DF or 0x2A700 <= codepoint <= 0x2B73F or
            0x2B740 <= codepoint <= 0x2B81F or 0x2B820 <= codepoint <= 0x2CEAF or
            0xF900 <= codepoint <= 0xFAFF or 0x2F800 <= codepoint <= 0x2FA1F)

    @classmethod
    def _tokenize_chinese(cls, text: str) -> str:
        output: list[str] = []
        for character in text:
            if cls._is_chinese(ord(character)):
                output.extend((" ", character, " "))
            else:
                output.append(character)
        return "".join(output)

    @staticmethod
    def _split_punctuation(token: str) -> list[str]:
        output: list[str] = []
        current: list[str] = []
        for character in token:
            codepoint = ord(character)
            punctuation = (
                33 <= codepoint <= 47 or 58 <= codepoint <= 64 or 91 <= codepoint <= 96 or
                123 <= codepoint <= 126 or unicodedata.category(character).startswith("P"))
            if punctuation:
                if current:
                    output.append("".join(current))
                    current = []
                output.append(character)
            else:
                current.append(character)
        if current:
            output.append("".join(current))
        return output

    def _wordpiece(self, token: str) -> list[str]:
        if len(token) > self.max_input_chars_per_word:
            return [self.unk_token]
        pieces: list[str] = []
        start = 0
        while start < len(token):
            end = len(token)
            selected = None
            while start < end:
                candidate = token[start:end]
                if start:
                    candidate = "##" + candidate
                if candidate in self.token_to_id:
                    selected = candidate
                    break
                end -= 1
            if selected is None:
                return [self.unk_token]
            pieces.append(selected)
            start = end
        return pieces


class BarkProcessor:
    """Turn text and validated voice presets into native Bark tensors."""

    def __init__(
        self,
        tokenizer: BarkWordPieceTokenizer,
        *,
        speaker_embeddings: Mapping[str, Any] | None = None,
        speaker_source: str | Path | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ) -> None:
        if not isinstance(tokenizer, BarkWordPieceTokenizer):
            raise TypeError("Bark processor requires BarkWordPieceTokenizer.")
        if speaker_embeddings is not None and not isinstance(
                speaker_embeddings,
                Mapping,
        ):
            raise TypeError("Bark speaker index must be a mapping or None.")
        self.tokenizer = tokenizer
        self.speaker_embeddings = (None if speaker_embeddings is None else dict(speaker_embeddings))
        self.speaker_source = (None if speaker_source is None else str(speaker_source))
        self.revision = revision
        self.cache_dir = cache_dir
        self.token = token
        self.local_files_only = local_files_only

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Save the declarative processor assets used by the native runtime."""
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "vocab.txt").write_text(
            "\n".join(self.tokenizer.vocabulary) + "\n",
            encoding="utf-8",
        )
        tokenizer_config = {
            "clean_up_tokenization_spaces": True,
            "do_lower_case": self.tokenizer.do_lower_case,
            "model_max_length": 512,
            "pad_token": self.tokenizer.pad_token,
            "processor_class": "BarkProcessor",
            "strip_accents": None,
            "tokenize_chinese_chars": self.tokenizer.tokenize_chinese_chars,
            "tokenizer_class": "BertTokenizer",
            "unk_token": self.tokenizer.unk_token,
        }
        (destination / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "processor_config.json").write_text(
            json.dumps(
                {
                    "processor_class": "BarkProcessor",
                    "speaker_source": self.speaker_source,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        if self.speaker_embeddings is not None:
            (destination / "speaker_embeddings_path.json").write_text(
                json.dumps(
                    self.speaker_embeddings,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
        return destination

    @classmethod
    def from_files(
        cls,
        vocab: str | Path,
        *,
        speaker_index: str | Path | None = None,
        speaker_source: str | Path | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ) -> BarkProcessor:
        embeddings = None
        if speaker_index is not None:
            path = Path(speaker_index)
            if path.stat().st_size > _MAX_VOCAB_BYTES:
                raise ValueError("Bark speaker index exceeds the safe size limit.")
            try:
                embeddings = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("Bark speaker index is invalid JSON.") from error
            if not isinstance(embeddings, dict):
                raise ValueError("Bark speaker index must be a JSON object.")
        return cls(
            BarkWordPieceTokenizer.from_vocab_file(vocab),
            speaker_embeddings=embeddings,
            speaker_source=speaker_source,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )

    @property
    def available_voice_presets(self) -> tuple[str, ...]:
        if self.speaker_embeddings is None:
            return ()
        return tuple(sorted(name for name in self.speaker_embeddings if name != "repo_or_path"))

    def __call__(
        self,
        *,
        text: str,
        voice_preset: str | Mapping[str, Any] | None = None,
        return_tensors: str = "pt",
        max_length: int = 256,
        **_: Any,
    ) -> dict[str, Tensor]:
        if return_tensors != "pt":
            raise ValueError("Native Bark processing returns PyTorch tensors only.")
        input_ids, attention_mask = self.tokenizer.encode(
            text,
            max_length=max_length,
        )
        output: dict[str, Any] = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor(
                [attention_mask],
                dtype=torch.long,
            ),
        }
        if voice_preset is not None:
            output["history_prompt"] = self.load_voice_preset(voice_preset)
        return output

    def load_voice_preset(
        self,
        value: str | Mapping[str, Any],
    ) -> dict[str, Tensor]:
        if isinstance(value, Mapping):
            return _validate_preset(value)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Bark voice preset must be a non-empty name or mapping.")
        name = value.strip()
        if self.speaker_embeddings is None or name not in self.speaker_embeddings:
            choices = ", ".join(self.available_voice_presets[:12])
            suffix = f"; available examples: {choices}" if choices else ""
            raise ValueError(f"Unknown Bark voice preset {name!r}{suffix}.")
        descriptor = self.speaker_embeddings[name]
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"Bark voice preset {name!r} has an invalid index entry.")
        source = (self.speaker_source or self.speaker_embeddings.get("repo_or_path"))
        if not isinstance(source, (str, Path)) or not str(source).strip():
            raise ValueError(f"Bark voice preset {name!r} has no artifact source.")
        tensors: dict[str, Tensor] = {}
        for key in _PRESET_SHAPES:
            filename = descriptor.get(key)
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"Bark voice preset {name!r} is missing {key!r}.")
            _validate_relative_path(filename)
            path = resolve_pretrained_file(
                source,
                filename,
                revision=self.revision,
                cache_dir=self.cache_dir,
                token=self.token,
                local_files_only=self.local_files_only,
            )
            tensors[key] = _read_npy_integer(path)
        return _validate_preset(tensors)


def _validate_relative_path(value: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe Bark voice preset path {value!r}.")


def _validate_preset(values: Mapping[str, Any]) -> dict[str, Tensor]:
    output: dict[str, Tensor] = {}
    for name, dimensions in _PRESET_SHAPES.items():
        value = values.get(name)
        if not isinstance(value, Tensor):
            try:
                value = torch.as_tensor(value)
            except (TypeError, ValueError) as error:
                raise TypeError(f"Bark {name} must be tensor-compatible.") from error
        if value.ndim != dimensions or value.numel() == 0:
            raise ValueError(f"Bark {name} must be a non-empty {dimensions}D tensor.")
        if value.dtype not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
        }:
            raise TypeError(f"Bark {name} must contain integer token IDs.")
        output[name] = value.long()
    if output["coarse_prompt"].shape[0] != 2:
        raise ValueError("Bark coarse prompt must contain exactly two codebooks.")
    if output["fine_prompt"].shape[0] != 8:
        raise ValueError("Bark fine prompt must contain exactly eight codebooks.")
    if output["coarse_prompt"].shape[1] != output["fine_prompt"].shape[1]:
        raise ValueError("Bark coarse and fine prompts must contain the same frame count.")
    if bool(((output["semantic_prompt"] < 0) | (output["semantic_prompt"] >= 10_000)).any().item()):
        raise ValueError("Bark semantic prompt tokens must be in [0, 10000).")
    for name in ("coarse_prompt", "fine_prompt"):
        if bool(((output[name] < 0) | (output[name] >= 1024)).any().item()):
            raise ValueError(f"Bark {name} codec tokens must be in [0, 1024).")
    return output


def _read_npy_integer(path: str | Path) -> Tensor:
    """Read a C-order integer NPY without importing NumPy or allowing
    pickle."""
    source = Path(path)
    if source.stat().st_size > _MAX_PRESET_BYTES:
        raise ValueError("Bark speaker prompt exceeds the safe size limit.")
    payload = source.read_bytes()
    if len(payload) < 12 or payload[:6] != b"\x93NUMPY":
        raise ValueError("Bark speaker prompt is not an NPY file.")
    major, minor = payload[6], payload[7]
    if major == 1:
        header_length = struct.unpack("<H", payload[8:10])[0]
        header_start = 10
    elif major in {2, 3}:
        header_length = struct.unpack("<I", payload[8:12])[0]
        header_start = 12
    else:
        raise ValueError(f"Unsupported NPY version {major}.{minor}.")
    header_end = header_start + header_length
    if header_end > len(payload):
        raise ValueError("Truncated Bark speaker prompt header.")
    try:
        header = ast.literal_eval(
            payload[header_start:header_end].decode("latin-1" if major < 3 else "utf-8"))
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise ValueError("Invalid Bark speaker prompt header.") from error
    if not isinstance(header, dict) or header.get("fortran_order") is not False:
        raise ValueError("Bark speaker prompts must use C-order NPY arrays.")
    descriptor = header.get("descr")
    dtype_map = {
        "|u1": (torch.uint8, 1),
        "|i1": (torch.int8, 1),
        "<i2": (torch.int16, 2),
        "<i4": (torch.int32, 4),
        "<i8": (torch.int64, 8),
    }
    if descriptor not in dtype_map:
        raise ValueError(f"Unsupported Bark speaker prompt dtype {descriptor!r}.")
    shape = header.get("shape")
    if (not isinstance(shape, tuple) or not shape or
            any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)):
        raise ValueError("Bark speaker prompt has an invalid shape.")
    dtype, item_size = dtype_map[descriptor]
    count = math_prod(shape)
    data = payload[header_end:]
    if len(data) != count * item_size:
        raise ValueError("Bark speaker prompt payload length is inconsistent.")
    return torch.frombuffer(
        bytearray(data),
        dtype=dtype,
        count=count,
    ).reshape(shape).clone()


def math_prod(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


__all__ = [
    "BarkProcessor",
    "BarkWordPieceTokenizer",
]
