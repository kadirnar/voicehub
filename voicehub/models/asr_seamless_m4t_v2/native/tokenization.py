"""Native SeamlessM4T-v2 SentencePiece BPE and language prompting."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor

from voicehub.models.asr_seamless_m4t_v2.native.languages import SEAMLESS_M4T_V2_LANGUAGE_TO_ID
from voicehub.tokenization.sentencepiece_model_bpe import SentencePieceModelBPETokenizer

_LINE_WHITESPACE = re.compile(r"[\n\r\t]")
_SPACES_BEFORE_METASPACE = re.compile(r" +▁")
_ONLY_METASPACE = re.compile(r"^▁+$")
_REPEATED_SPACES = re.compile(r" {2,}")


def _language(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A target language must be a non-empty string.")
    result = value.strip()
    if result.startswith("__") and result.endswith("__") and len(result) > 4:
        result = result[2:-2]
    # Preserve the official mixed-case script suffix while accepting common
    # lower-case public input.
    if result.lower() == "cmn_hant":
        result = "cmn_Hant"
    else:
        result = result.lower()
    if result not in SEAMLESS_M4T_V2_LANGUAGE_TO_ID:
        choices = ", ".join(SEAMLESS_M4T_V2_LANGUAGE_TO_ID)
        raise ValueError(f"Unsupported SeamlessM4T-v2 language {value!r}; choose: {choices}.")
    return result


def _normalize_for_backend(text: str) -> str:
    """Reproduce the published Tokenizers normalizer and Metaspace graph."""
    text = _LINE_WHITESPACE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.rstrip()
    text = _SPACES_BEFORE_METASPACE.sub("▁", text)
    if _ONLY_METASPACE.fullmatch(text):
        text = ""
    text = _REPEATED_SPACES.sub("▁", text)
    if not text:
        return ""
    text = text.replace(" ", "▁")
    if not text.startswith("▁"):
        text = "▁" + text
    return text


class SeamlessM4Tv2Tokenizer:
    """Official ID remap and language postprocessor over native BPE."""

    pad_token_id = 0
    unk_token_id = 1
    bos_token_id = 2
    eos_token_id = 3
    sentencepiece_offset = 1

    def __init__(
        self,
        sentencepiece: SentencePieceModelBPETokenizer,
        *,
        expected_sentencepiece_size: int | None = 256_000,
    ) -> None:
        if not isinstance(sentencepiece, SentencePieceModelBPETokenizer):
            raise TypeError("`sentencepiece` must be SentencePieceModelBPETokenizer.")
        if (expected_sentencepiece_size is not None and
                sentencepiece.vocabulary_size != expected_sentencepiece_size):
            raise ValueError(
                "The SeamlessM4T-v2 tokenizer contains "
                f"{sentencepiece.vocabulary_size} SentencePiece entries; "
                f"expected {expected_sentencepiece_size}.")
        if (sentencepiece.unk_token_id != 0 or sentencepiece.bos_token_id != 1 or
                sentencepiece.eos_token_id != 2):
            raise ValueError("SeamlessM4T-v2 SentencePiece special IDs are incoherent.")
        self.sentencepiece = sentencepiece

    @classmethod
    def from_files(
        cls,
        tokenizer_model: str | Path,
        *,
        added_tokens: str | Path | None = None,
        expected_sentencepiece_size: int | None = 256_000,
    ) -> SeamlessM4Tv2Tokenizer:
        tokenizer = cls(
            SentencePieceModelBPETokenizer.from_model_file(tokenizer_model),
            expected_sentencepiece_size=expected_sentencepiece_size,
        )
        if added_tokens is not None:
            path = Path(added_tokens)
            try:
                values = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Could not read SeamlessM4T-v2 added tokens: {path}.") from error
            expected = {
                f"__{language}__": token_id
                for language, token_id in SEAMLESS_M4T_V2_LANGUAGE_TO_ID.items()
            }
            if values != expected:
                raise ValueError(
                    "SeamlessM4T-v2 language tokens disagree with the "
                    "audited 98-language table.")
        return tokenizer

    @property
    def language_to_id(self) -> Mapping[str, int]:
        return SEAMLESS_M4T_V2_LANGUAGE_TO_ID

    def language_token_id(self, language: str) -> int:
        return SEAMLESS_M4T_V2_LANGUAGE_TO_ID[_language(language)]

    def encode_text(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")
        encoded = self.sentencepiece.encode_normalized(_normalize_for_backend(text), )
        return tuple(token_id + self.sentencepiece_offset for token_id in encoded.input_ids)

    def encode_target(
        self,
        text: str,
        *,
        language: str,
        add_special_tokens: bool = True,
    ) -> tuple[int, ...]:
        tokens = self.encode_text(text)
        if not add_special_tokens:
            return tokens
        return (
            self.eos_token_id,
            self.language_token_id(language),
            *tokens,
            self.eos_token_id,
        )

    def generation_prompt(self, language: str) -> tuple[int, ...]:
        return (self.language_token_id(language), )

    def batch_encode_targets(
        self,
        texts: Sequence[str],
        *,
        language: str,
        padding_value: int = -100,
    ) -> Tensor:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("`texts` must be a sequence of strings.")
        rows = tuple(self.encode_target(text, language=language) for text in texts)
        if not rows:
            raise ValueError("A target batch cannot be empty.")
        if isinstance(padding_value, bool) or not isinstance(padding_value, int):
            raise TypeError("`padding_value` must be an integer.")
        maximum = max(len(row) for row in rows)
        return torch.tensor(
            [(*row, *((padding_value, ) * (maximum - len(row)))) for row in rows],
            dtype=torch.long,
        )

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        raw = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("Token IDs must be integers.")
            if token_id in SEAMLESS_M4T_V2_LANGUAGE_TO_ID.values():
                if skip_special_tokens:
                    continue
                raise ValueError("Language tokens cannot be rendered as ordinary text.")
            if token_id == self.pad_token_id:
                if skip_special_tokens:
                    continue
                raise ValueError("The standalone pad token has no text form.")
            if token_id in {
                    self.bos_token_id,
                    self.eos_token_id,
            } and skip_special_tokens:
                continue
            raw_id = token_id - self.sentencepiece_offset
            if not 0 <= raw_id < self.sentencepiece.vocabulary_size:
                raise ValueError(f"Token ID {token_id} is outside the text vocabulary.")
            raw.append(raw_id)
        return self.sentencepiece.decode(
            raw,
            skip_special_tokens=skip_special_tokens,
        )

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

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        model = self.sentencepiece.save_pretrained(destination)
        (destination / "added_tokens.json").write_text(
            json.dumps(
                {
                    f"__{language}__": token_id
                    for language, token_id in SEAMLESS_M4T_V2_LANGUAGE_TO_ID.items()
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        return model


__all__ = [
    "SEAMLESS_M4T_V2_LANGUAGE_TO_ID",
    "SeamlessM4Tv2Tokenizer",
]
