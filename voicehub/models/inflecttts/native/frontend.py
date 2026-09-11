"""Dependency-free, explicit phoneme frontend for Inflect v2.

The released models were trained with eSpeak's ``en-us`` phoneme output.
VoiceHub does not silently replace that versioned native frontend with
an approximate grapheme-to-phoneme implementation.  Applications must
provide the phoneme string produced during data preparation, or exact
token IDs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from voicehub.models.inflecttts.native.symbols import symbols

_SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(symbols)}


class InflectFrontendError(ValueError):
    """Raised when preprocessed Inflect input is missing or incompatible."""


@dataclass(frozen=True, slots=True)
class InflectTokenBatch:
    """Padded checkpoint-native token IDs and lengths."""

    input_ids: torch.Tensor
    input_lengths: torch.Tensor


def phonemes_to_ids(
    phoneme_text: str,
    *,
    add_blank: bool = True,
) -> list[int]:
    """Map an exact eSpeak-compatible phoneme string to checkpoint IDs."""
    if not isinstance(phoneme_text, str):
        raise TypeError("`phoneme_text` must be a string.")
    if not phoneme_text:
        raise InflectFrontendError("`phoneme_text` cannot be empty.")
    unknown = sorted(set(phoneme_text) - _SYMBOL_TO_ID.keys())
    if unknown:
        rendered = ", ".join(repr(item) for item in unknown)
        raise InflectFrontendError(
            "Inflect phoneme text contains symbols outside the published "
            f"178-symbol inventory: {rendered}.")
    sequence = [_SYMBOL_TO_ID[symbol] for symbol in phoneme_text]
    if add_blank:
        expanded = [0] * (len(sequence) * 2 + 1)
        expanded[1::2] = sequence
        sequence = expanded
    return sequence


def validate_token_ids(
        values: Sequence[int] | torch.Tensor,
        *,
        vocabulary_size: int = len(symbols),
) -> torch.Tensor:
    """Validate one unbatched exact token sequence."""
    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    if tensor.dtype == torch.bool or tensor.is_floating_point():
        raise TypeError("Inflect token IDs must use an integer dtype.")
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise InflectFrontendError("Inflect token IDs must be a non-empty rank-one sequence.")
    tensor = tensor.long()
    if bool(((tensor < 0) | (tensor >= vocabulary_size)).any()):
        raise InflectFrontendError(f"Inflect token IDs must be in [0, {vocabulary_size}).")
    return tensor


def batch_token_ids(
        sequences: Sequence[Sequence[int] | torch.Tensor],
        *,
        vocabulary_size: int = len(symbols),
        device: torch.device | str | None = None,
) -> InflectTokenBatch:
    """Pad validated token sequences with the published blank/pad ID zero."""
    if not sequences:
        raise InflectFrontendError("At least one token sequence is required.")
    items = [validate_token_ids(item, vocabulary_size=vocabulary_size) for item in sequences]
    lengths = torch.tensor([item.numel() for item in items], dtype=torch.long)
    padded = torch.zeros(
        len(items),
        int(lengths.max().item()),
        dtype=torch.long,
    )
    for index, item in enumerate(items):
        padded[index, :item.numel()] = item
    if device is not None:
        padded = padded.to(device)
        lengths = lengths.to(device)
    return InflectTokenBatch(padded, lengths)


def require_preprocessed_phonemes(
    raw_text: str,
    *,
    phoneme_text: str | None,
    input_is_phonemes: bool,
) -> str:
    """Resolve explicit input without pretending raw English is phonemes."""
    if phoneme_text is not None:
        if input_is_phonemes:
            raise ValueError("Provide either `phoneme_text` or `input_is_phonemes=True`, "
                             "not both.")
        return phoneme_text
    if input_is_phonemes:
        return raw_text
    raise InflectFrontendError(
        "Inflect v2 requires checkpoint-compatible en-us phonemes because "
        "VoiceHub's native runtime does not depend on eSpeak/phonemizer. "
        "Pass `phoneme_text=...`, exact `input_ids`, or set "
        "`input_is_phonemes=True` when the positional text already contains "
        "the preprocessed phoneme string.")


__all__ = [
    "InflectFrontendError",
    "InflectTokenBatch",
    "batch_token_ids",
    "phonemes_to_ids",
    "require_preprocessed_phonemes",
    "validate_token_ids",
]
