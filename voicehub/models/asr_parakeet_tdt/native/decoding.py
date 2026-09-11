"""Duration-aware text and timestamp decoding for Parakeet TDT."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from voicehub.models.asr_parakeet_tdt.native.tokenization import ParakeetTokenizer

_PUNCTUATION = frozenset({"?", "'", "¡", "¿", "-", ":", ",", "%", "/", ".", "!"})


@dataclass(frozen=True, slots=True)
class TokenTimestamp:
    text: str
    start: float
    end: float
    token_id: int


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class DecodedParakeetSequence:
    text: str
    tokens: tuple[TokenTimestamp, ...]
    words: tuple[WordTimestamp, ...]


def _piece_text(piece: str) -> tuple[str, bool]:
    starts_word = piece.startswith("▁")
    return piece.replace("▁", " "), starts_word


def decode_tdt_sequence(
    tokenizer: ParakeetTokenizer,
    token_ids: torch.Tensor,
    durations: torch.Tensor,
    *,
    frame_seconds: float,
) -> DecodedParakeetSequence:
    """Decode one sequence without collapsing repeated TDT emissions."""
    if token_ids.ndim != 1 or durations.ndim != 1:
        raise ValueError("TDT token IDs and durations must be one-dimensional.")
    if token_ids.shape != durations.shape:
        raise ValueError("TDT token IDs and durations must have equal shape.")
    if frame_seconds <= 0:
        raise ValueError("`frame_seconds` must be positive.")
    starts = torch.cumsum(durations, dim=0) - durations
    skip = {
        tokenizer.pad_token_id,
        tokenizer.blank_token_id,
        tokenizer.eos_token_id,
        tokenizer.unk_token_id,
    }
    token_offsets: list[TokenTimestamp] = []
    word_offsets: list[WordTimestamp] = []
    active_text = ""
    active_start = 0.0
    active_end = 0.0

    def finish_word() -> None:
        nonlocal active_text
        text = active_text.strip()
        if text:
            word_offsets.append(WordTimestamp(
                text=text,
                start=active_start,
                end=active_end,
            ))
        active_text = ""

    for raw_id, raw_start, raw_duration in zip(
            token_ids.tolist(),
            starts.tolist(),
            durations.tolist(),
    ):
        token_id = int(raw_id)
        if token_id in skip:
            continue
        piece = tokenizer.token_piece(token_id)
        text, starts_word = _piece_text(piece)
        start = int(raw_start) * frame_seconds
        end = (int(raw_start) + int(raw_duration)) * frame_seconds
        stripped = text.strip()
        if stripped in _PUNCTUATION and token_offsets:
            start = token_offsets[-1].end
            end = start
        token_offsets.append(TokenTimestamp(
            text=text,
            start=start,
            end=end,
            token_id=token_id,
        ))
        if starts_word and active_text:
            finish_word()
        if not active_text:
            active_start = start
        active_text += text
        active_end = end
    finish_word()
    decoded = tokenizer.decode(token_ids.tolist())
    return DecodedParakeetSequence(
        text=decoded,
        tokens=tuple(token_offsets),
        words=tuple(word_offsets),
    )


__all__ = [
    "DecodedParakeetSequence",
    "TokenTimestamp",
    "WordTimestamp",
    "decode_tdt_sequence",
]
