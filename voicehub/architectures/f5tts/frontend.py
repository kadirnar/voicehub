"""Dependency-free text frontend for F5-TTS vocabularies."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

TokenSequence = Sequence[str]
TextNormalizer = Callable[[str], TokenSequence]

_PUNCTUATION_TRANSLATION = str.maketrans({
    ";": ",",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
})


def _contains_chinese(text: str) -> bool:
    # The released frontend converts \u3100-\u9fff with pypinyin, and its
    # word segmenter also treats CJK compatibility and supplementary
    # ideographs as Chinese. None of these can be reproduced character-wise.
    return any(
        "\u3100" <= character <= "\u9fff" or "\uf900" <= character <= "\ufaff" or
        "\U00020000" <= character <= "\U0002fa1f" for character in text)


# Word segmentation used by the released F5-TTS frontend (``rjieba.cut``,
# i.e. jieba-rs with its default dictionary) restricted to text without
# Chinese characters. The released frontend inserts a space before every
# multi-character, single-byte segment unless the previous character is one
# of `` :'"``; the checkpoint was trained on, and is evaluated with, those
# spaces (``"well-known"`` becomes ``"well- known"``).
_SEGMENT_BLOCK = re.compile(r"([a-zA-Z0-9+#&._%\-]+)")
_SEGMENT_WHITESPACE = re.compile(r"(\r\n|\s)")
# jieba-rs' HMM fallback pattern; its unescaped ``.`` is intentional.
_SEGMENT_ALPHANUMERIC = re.compile(r"([a-zA-Z0-9]+(?:.\d+)?%?)")
# The only pure-ASCII entries of the default jieba dictionary, with their
# frequencies, and the dictionary's total frequency.
_SEGMENT_DICTIONARY = {"AT&T": 3, "c#": 3, "C#": 3, "c++": 3, "C++": 3}
_SEGMENT_LOG_TOTAL = math.log(60_101_967)
_NO_SPACE_BEFORE_SEGMENT = frozenset(" :'\"")


def _segment_block(block: str) -> Iterator[str]:
    """Cut one ``[a-zA-Z0-9+#&._%-]`` run like jieba's DAG + HMM path."""
    length = len(block)
    route: list[tuple[float, int]] = [(0.0, 0)] * (length + 1)
    for start in range(length - 1, -1, -1):
        ends = [start + len(word) - 1
                for word in _SEGMENT_DICTIONARY if block.startswith(word, start)] or [start]
        route[start] = max((
            math.log(_SEGMENT_DICTIONARY.get(block[start:end + 1], 1)) - _SEGMENT_LOG_TOTAL +
            route[end + 1][0],
            end,
        ) for end in ends)

    def flush(buffer: str) -> Iterator[str]:
        if len(buffer) == 1:
            yield buffer
        elif buffer in _SEGMENT_DICTIONARY:
            yield from buffer
        else:
            yield from (piece for piece in _SEGMENT_ALPHANUMERIC.split(buffer) if piece)

    buffer = ""
    start = 0
    while start < length:
        end = route[start][1] + 1
        if end - start == 1:
            buffer += block[start]
        else:
            if buffer:
                yield from flush(buffer)
                buffer = ""
            yield block[start:end]
        start = end
    if buffer:
        yield from flush(buffer)


def segment_text(text: str) -> Iterator[str]:
    """Yield the released frontend's word segments for non-Chinese text."""
    for block in _SEGMENT_BLOCK.split(text):
        if not block:
            continue
        if _SEGMENT_BLOCK.fullmatch(block):
            yield from _segment_block(block)
            continue
        for piece in _SEGMENT_WHITESPACE.split(block):
            if _SEGMENT_WHITESPACE.fullmatch(piece):
                yield piece
            else:
                yield from piece


def character_tokens(text: str) -> tuple[str, ...]:
    """Tokenize non-Chinese text exactly like ``convert_char_to_pinyin``."""
    tokens: list[str] = []
    for segment in segment_text(text):
        if (len(segment) > 1 and len(segment.encode("utf-8")) == len(segment) and tokens and
                tokens[-1] not in _NO_SPACE_BEFORE_SEGMENT):
            tokens.append(" ")
        tokens.extend(segment)
    return tuple(tokens)


class F5Vocabulary:
    """Ordered F5 vocabulary with the released unknown-token convention."""

    def __init__(self, tokens: Sequence[str]) -> None:
        resolved = tuple(tokens)
        if not resolved:
            raise ValueError("F5-TTS vocabulary cannot be empty.")
        if resolved[0] != " ":
            raise ValueError("F5-TTS vocabulary index 0 must be a space/unknown token.")
        if len(set(resolved)) != len(resolved):
            raise ValueError("F5-TTS vocabulary contains duplicate tokens.")
        self.tokens = resolved
        self.token_to_id = {token: index for index, token in enumerate(resolved)}

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_file(cls, path: str | Path) -> F5Vocabulary:
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"F5-TTS vocabulary was not found: {source}.")
        lines = source.read_text(encoding="utf-8").splitlines()
        return cls(lines)

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "".join(f"{token}\n" for token in self.tokens),
            encoding="utf-8",
        )
        return destination

    def encode(self, tokens: TokenSequence) -> list[int]:
        return [self.token_to_id.get(token, 0) for token in tokens]


class NativeF5TextFrontend:
    """F5 tokenizer with an explicit Chinese G2P boundary.

    The released multilingual vocabulary stores Chinese syllables with
    tone numbers. Reproducing jieba segmentation and tone-sandhi through
    a hidden optional dependency would make the native runtime
    inaccurate and non-reproducible. Callers may therefore inject a
    VoiceHub-owned normalizer or pass already-normalized token
    sequences. Non-Chinese text follows the released character path,
    including the word-boundary spaces it inserts (see
    :func:`character_tokens`).
    """

    def __init__(
        self,
        vocabulary: F5Vocabulary,
        *,
        normalizer: TextNormalizer | None = None,
    ) -> None:
        if not isinstance(vocabulary, F5Vocabulary):
            raise TypeError("`vocabulary` must be an F5Vocabulary.")
        if normalizer is not None and not callable(normalizer):
            raise TypeError("`normalizer` must be callable or None.")
        self.vocabulary = vocabulary
        self.normalizer = normalizer

    def normalize(
        self,
        text: str | TokenSequence,
    ) -> tuple[str, ...]:
        if isinstance(text, str):
            translated = text.translate(_PUNCTUATION_TRANSLATION)
            if self.normalizer is not None:
                tokens = tuple(self.normalizer(translated))
            else:
                if _contains_chinese(translated):
                    raise ValueError(
                        "Chinese F5-TTS input requires pinyin-with-tone tokens "
                        "or an explicit native text normalizer. VoiceHub does "
                        "not silently substitute a non-equivalent G2P.")
                tokens = character_tokens(translated)
        elif isinstance(text, Sequence) and not isinstance(text, (bytes, bytearray)):
            tokens = tuple(text)
        else:
            raise TypeError("F5-TTS text must be a string or token sequence.")
        if any(not isinstance(token, str) or not token for token in tokens):
            raise ValueError("F5-TTS token sequences must contain non-empty strings.")
        return tokens

    def encode(
        self,
        text: str | TokenSequence,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        tokens = self.normalize(text)
        return torch.tensor(
            self.vocabulary.encode(tokens),
            dtype=torch.long,
            device=device,
        )

    def encode_batch(
        self,
        texts: Sequence[str | TokenSequence],
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if not texts:
            raise ValueError("F5-TTS text batch cannot be empty.")
        encoded = [self.encode(text, device=device) for text in texts]
        return pad_sequence(encoded, batch_first=True, padding_value=-1)


__all__ = [
    "F5Vocabulary",
    "NativeF5TextFrontend",
    "TextNormalizer",
    "TokenSequence",
    "character_tokens",
    "segment_text",
]
