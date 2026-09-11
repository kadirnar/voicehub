"""Dependency-free text frontend for F5-TTS vocabularies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    return any("\u3100" <= character <= "\u9fff" for character in text)


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
    sequences. Non-Chinese character input is handled directly and
    matches the released character path.
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
                tokens = tuple(translated)
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
]
