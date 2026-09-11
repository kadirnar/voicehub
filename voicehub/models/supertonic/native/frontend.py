"""Dependency-free multilingual Unicode frontend and style loader."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from unicodedata import normalize

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from voicehub.models.supertonic.native.metadata import SUPERTONIC_LANGUAGES

AVAILABLE_LANGUAGES = SUPERTONIC_LANGUAGES
_MAX_INDEXER_BYTES = 8 * 1024 * 1024
_MAX_STYLE_BYTES = 4 * 1024 * 1024
_EMOJI = re.compile(
    "[\U0001f1e6-\U0001f1ff"
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf]+",
    flags=re.UNICODE,
)
_REPLACEMENTS = {
    "–": "-",
    "‑": "-",
    "—": "-",
    "_": " ",
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "´": "'",
    "`": "'",
    "[": " ",
    "]": " ",
    "|": " ",
    "/": " ",
    "#": " ",
    "→": " ",
    "←": " ",
}
_EXPRESSIONS = {
    "@": " at ",
    "e.g.,": "for example, ",
    "i.e.,": "that is, ",
}


def _json_file(path: str | Path, *, maximum_bytes: int):
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Supertonic asset was not found: {source}.")
    size = source.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"Supertonic asset {source.name!r} has unsafe size {size}.")
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Supertonic asset {source.name!r} is invalid JSON.") from error


def length_mask(lengths: Tensor, maximum: int | None = None) -> Tensor:
    """Return a float mask with shape ``[batch, 1, time]``."""
    if not isinstance(lengths, Tensor) or lengths.ndim != 1:
        raise ValueError("`lengths` must be a rank-one tensor.")
    if lengths.dtype == torch.bool or lengths.is_floating_point():
        raise TypeError("`lengths` must use an integer dtype.")
    if (lengths <= 0).any():
        raise ValueError("All sequence lengths must be positive.")
    resolved_maximum = (int(lengths.max().item()) if maximum is None else int(maximum))
    if resolved_maximum < int(lengths.max().item()):
        raise ValueError("Mask maximum is shorter than a sequence.")
    positions = torch.arange(
        resolved_maximum,
        device=lengths.device,
    )
    return (positions.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).to(dtype=torch.float32)


class SupertonicUnicodeProcessor:
    """Reproduce the released NFKD/Unicode-index frontend with PyTorch."""

    def __init__(self, indexer: tuple[int, ...]) -> None:
        if not indexer:
            raise ValueError("Supertonic Unicode indexer cannot be empty.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < -1 for value in indexer):
            raise ValueError("Supertonic Unicode indexer must contain integer IDs >= -1.")
        self.indexer = indexer

    @classmethod
    def from_file(
        cls,
        path: str | Path,
    ) -> SupertonicUnicodeProcessor:
        value = _json_file(path, maximum_bytes=_MAX_INDEXER_BYTES)
        if not isinstance(value, list):
            raise ValueError("Supertonic Unicode indexer must be an array.")
        return cls(tuple(value))

    @staticmethod
    def normalize_text(text: str, language: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Supertonic text must be non-empty.")
        if not isinstance(language, str):
            raise TypeError("Supertonic language must be a string.")
        language = language.strip().lower()
        if language not in AVAILABLE_LANGUAGES:
            supported = ", ".join(sorted(AVAILABLE_LANGUAGES))
            raise ValueError(f"Unsupported Supertonic language {language!r}. "
                             f"Supported: {supported}.")
        value = _EMOJI.sub("", normalize("NFKD", text))
        for source, replacement in _REPLACEMENTS.items():
            value = value.replace(source, replacement)
        value = re.sub(r"[♥☆♡©\\]", "", value)
        for source, replacement in _EXPRESSIONS.items():
            value = value.replace(source, replacement)
        value = re.sub(r" ([,.!?;:])", r"\1", value)
        value = re.sub(r" '", "'", value)
        for repeated, replacement in (
            ('""', '"'),
            ("''", "'"),
            ("``", "`"),
        ):
            while repeated in value:
                value = value.replace(repeated, replacement)
        value = re.sub(r"\s+", " ", value).strip()
        sentence_ending = r"[.!?;:,'\")\]}…。」』】〉》›»]$"
        if not re.search(sentence_ending, value):
            value += "."
        return f"<{language}>{value}</{language}>"

    def encode(
        self,
        texts: tuple[str, ...] | list[str],
        languages: tuple[str, ...] | list[str],
        *,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:
        texts = tuple(texts)
        languages = tuple(languages)
        if not texts or len(texts) != len(languages):
            raise ValueError("Supertonic texts and languages must have equal non-zero "
                             "lengths.")
        encoded = []
        for text, language in zip(texts, languages):
            normalized = self.normalize_text(text, language)
            ids = []
            for character in normalized:
                # The released frontend materializes Unicode values as
                # uint16 before indexing. Preserve that wraparound exactly
                # without introducing NumPy.
                codepoint = ord(character) & 0xFFFF
                if codepoint >= len(self.indexer):
                    raise ValueError(
                        f"Character U+{codepoint:04X} is outside the "
                        "released Supertonic indexer.")
                ids.append(self.indexer[codepoint])
            encoded.append(torch.tensor(
                ids,
                dtype=torch.int64,
                device=device,
            ))
        lengths = torch.tensor(
            [value.numel() for value in encoded],
            dtype=torch.int64,
            device=device,
        )
        ids = pad_sequence(
            encoded,
            batch_first=True,
            padding_value=0,
        )
        return ids, length_mask(lengths, ids.shape[1])


@dataclass(frozen=True, slots=True)
class SupertonicStyle:
    """Released text-to-latent and duration-predictor style tensors."""

    ttl: Tensor
    duration: Tensor

    def __post_init__(self) -> None:
        for name, value, shape in (
            ("ttl", self.ttl, (50, 256)),
            ("duration", self.duration, (8, 16)),
        ):
            if not isinstance(value, Tensor) or value.ndim != 3:
                raise ValueError(
                    f"Supertonic style `{name}` must have shape "
                    f"[batch, {shape[0]}, {shape[1]}].")
            if tuple(value.shape[1:]) != shape:
                raise ValueError(f"Supertonic style `{name}` has invalid shape "
                                 f"{tuple(value.shape)}.")
            if (not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError(f"Supertonic style `{name}` must contain finite floats.")
        if self.ttl.shape[0] != self.duration.shape[0]:
            raise ValueError("Supertonic style batch dimensions differ.")

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SupertonicStyle:
        return SupertonicStyle(
            ttl=self.ttl.to(device=device, dtype=dtype),
            duration=self.duration.to(device=device, dtype=dtype),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> SupertonicStyle:
        payload = _json_file(path, maximum_bytes=_MAX_STYLE_BYTES)
        if not isinstance(payload, dict):
            raise ValueError("Supertonic style must contain an object.")

        def tensor(section_name: str, shape: tuple[int, ...]) -> Tensor:
            section = payload.get(section_name)
            if not isinstance(section, dict):
                raise ValueError(f"Supertonic style is missing {section_name!r}.")
            declared = section.get("dims")
            if declared != list(shape):
                raise ValueError(
                    f"Supertonic style {section_name!r} declares invalid "
                    f"dimensions {declared!r}.")
            try:
                value = torch.tensor(
                    section["data"],
                    dtype=torch.float32,
                ).reshape(shape)
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                raise ValueError(f"Supertonic style {section_name!r} has invalid data.") from error
            if not torch.isfinite(value).all():
                raise ValueError(f"Supertonic style {section_name!r} is not finite.")
            return value

        return cls(
            ttl=tensor("style_ttl", (1, 50, 256)),
            duration=tensor("style_dp", (1, 8, 16)),
        )


__all__ = [
    "AVAILABLE_LANGUAGES",
    "SupertonicStyle",
    "SupertonicUnicodeProcessor",
    "length_mask",
]
