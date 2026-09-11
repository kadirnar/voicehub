"""Torch-only CTC forced alignment.

The dynamic-programming recurrence follows WhisperX's BSD-licensed alignment
implementation at revision
``2cfd7b7c5c7bba144954364db747319b50e8232b``.  VoiceHub keeps the numerical
trellis and backtracking semantics while replacing NumPy, pandas, NLTK,
torchaudio, and Transformers data structures with small typed values.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral

import torch

from voicehub.neural.ctc_alignment.metadata import LANGUAGES_WITHOUT_SPACES


@dataclass(frozen=True, slots=True)
class _PathPoint:
    token_index: int
    time_index: int
    score: float


@dataclass(frozen=True, slots=True)
class _TokenSpan:
    token_index: int
    start_frame: int
    end_frame: int
    score: float


@dataclass(frozen=True, slots=True)
class AlignedCharacter:
    """One transcript character aligned to a half-open time interval."""

    character: str
    text_index: int
    start: float
    end: float
    confidence: float


@dataclass(frozen=True, slots=True)
class AlignedWord:
    """One word assembled from aligned transcript characters."""

    text: str
    start: float
    end: float
    confidence: float


@dataclass(frozen=True, slots=True)
class CTCAlignment:
    """Typed result of aligning one transcript segment."""

    words: tuple[AlignedWord, ...]
    characters: tuple[AlignedCharacter, ...]


def _validate_emission(emission: torch.Tensor) -> None:
    if not isinstance(emission, torch.Tensor):
        raise TypeError("`emission` must be a torch.Tensor.")
    if emission.ndim != 2:
        raise ValueError("CTC alignment emission must have shape [frames, vocabulary].")
    if emission.shape[0] <= 0 or emission.shape[1] <= 1:
        raise ValueError("CTC alignment emission requires frames and at least two labels.")
    if not emission.is_floating_point():
        raise TypeError("CTC alignment emission must use a floating dtype.")
    if torch.isnan(emission).any() or torch.isposinf(emission).any():
        raise ValueError("CTC alignment emission contains NaN or +inf values.")


def _validated_tokens(
    tokens: Sequence[int],
    *,
    vocabulary_size: int,
) -> tuple[int, ...]:
    normalized = []
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, Integral):
            raise TypeError("CTC transcript tokens must be integers.")
        token = int(token)
        if not 0 <= token < vocabulary_size:
            raise ValueError(
                f"CTC transcript token {token} is outside a vocabulary of "
                f"size {vocabulary_size}.")
        normalized.append(token)
    return tuple(normalized)


def build_trellis(
    emission: torch.Tensor,
    tokens: Sequence[int],
    *,
    blank_id: int = 0,
) -> torch.Tensor:
    """Build WhisperX's CTC stay/change dynamic-programming trellis."""
    _validate_emission(emission)
    if (isinstance(blank_id, bool) or not isinstance(blank_id, Integral) or
            not 0 <= int(blank_id) < emission.shape[1]):
        raise ValueError("`blank_id` must index the emission vocabulary.")
    blank_id = int(blank_id)
    normalized = _validated_tokens(
        tokens,
        vocabulary_size=emission.shape[1],
    )
    frames = emission.shape[0]
    token_count = len(normalized)
    if token_count > frames:
        raise ValueError("CTC alignment requires at least one emission frame per "
                         "transcript token.")

    trellis = emission.new_full(
        (frames + 1, token_count + 1),
        -torch.inf,
    )
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], dim=0)
    if token_count:
        # Match the reference boundary condition: these cells cannot be valid
        # starts because too few frames remain to consume every token.
        trellis[-token_count:, 0] = torch.inf
        token_tensor = torch.tensor(
            normalized,
            device=emission.device,
            dtype=torch.long,
        )
        for frame in range(frames):
            trellis[frame + 1, 1:] = torch.maximum(
                trellis[frame, 1:] + emission[frame, blank_id],
                trellis[frame, :-1] + emission[frame, token_tensor],
            )
    return trellis


def _backtrack(
    trellis: torch.Tensor,
    emission: torch.Tensor,
    tokens: tuple[int, ...],
    *,
    blank_id: int,
) -> tuple[_PathPoint, ...] | None:
    if not tokens:
        return ()
    token_index = trellis.shape[1] - 1
    start_frame = int(torch.argmax(trellis[:, token_index]).item())
    path: list[_PathPoint] = []
    for frame in range(start_frame, 0, -1):
        stayed = (trellis[frame - 1, token_index] + emission[frame - 1, blank_id])
        changed = (trellis[frame - 1, token_index - 1] + emission[frame - 1, tokens[token_index - 1]])
        did_change = bool((changed > stayed).item())
        label = tokens[token_index - 1] if did_change else blank_id
        score = float(emission[frame - 1, label].exp().item())
        path.append(
            _PathPoint(
                token_index=token_index - 1,
                time_index=frame - 1,
                score=max(0.0, min(1.0, score)),
            ))
        if did_change:
            token_index -= 1
            if token_index == 0:
                return tuple(reversed(path))
    return None


def _merge_path(path: Sequence[_PathPoint]) -> tuple[_TokenSpan, ...]:
    spans: list[_TokenSpan] = []
    cursor = 0
    while cursor < len(path):
        stop = cursor + 1
        while (stop < len(path) and path[stop].token_index == path[cursor].token_index):
            stop += 1
        points = path[cursor:stop]
        spans.append(
            _TokenSpan(
                token_index=points[0].token_index,
                start_frame=points[0].time_index,
                end_frame=points[-1].time_index + 1,
                score=sum(point.score for point in points) / len(points),
            ))
        cursor = stop
    return tuple(spans)


def _folded_vocabulary(
    vocabulary: Mapping[str, int],
    *,
    vocabulary_size: int,
) -> dict[str, int]:
    folded: dict[str, int] = {}
    for token, token_id in vocabulary.items():
        if not isinstance(token, str) or not token:
            raise ValueError("CTC vocabulary tokens must be non-empty strings.")
        if (isinstance(token_id, bool) or not isinstance(token_id, Integral) or
                not 0 <= int(token_id) < vocabulary_size):
            raise ValueError(f"CTC vocabulary token {token!r} has an invalid ID.")
        key = token.casefold()
        existing = folded.get(key)
        if existing is not None and existing != int(token_id):
            raise ValueError(
                "CTC alignment cannot case-fold vocabulary tokens "
                f"unambiguously: {token!r}.")
        folded[key] = int(token_id)
    return folded


def _word_spans(
    text: str,
    *,
    language: str,
) -> tuple[tuple[int, int], ...]:
    if language in LANGUAGES_WITHOUT_SPACES:
        return tuple((index, index + 1) for index, character in enumerate(text) if not character.isspace())
    return tuple((match.start(), match.end()) for match in re.finditer(r"\S+", text))


def align_ctc_transcript(
    emission: torch.Tensor,
    text: str,
    vocabulary: Mapping[str, int],
    *,
    blank_id: int,
    word_delimiter_token: str = "|",
    language: str = "en",
    segment_start: float = 0.0,
    segment_end: float = 0.0,
) -> CTCAlignment:
    """Force-align one transcript against frame-wise CTC log probabilities.

    ``emission`` must already be log-softmax normalized. Unknown
    transcript characters follow WhisperX's wildcard rule: their score
    is the best non-blank label at each frame.
    """
    _validate_emission(emission)
    if not isinstance(text, str):
        raise TypeError("`text` must be a string.")
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise ValueError("`vocabulary` must be a non-empty mapping.")
    if not isinstance(word_delimiter_token, str) or not word_delimiter_token:
        raise ValueError("`word_delimiter_token` must be non-empty.")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("`language` must be a non-empty string.")
    language = language.strip().lower()
    for name, value in (
        ("segment_start", segment_start),
        ("segment_end", segment_end),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"`{name}` must be a finite real number.")
        if not isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"`{name}` must be finite and non-negative.")
    segment_start = float(segment_start)
    segment_end = float(segment_end)
    if segment_end <= segment_start:
        raise ValueError("`segment_end` must be greater than `segment_start`.")
    if (isinstance(blank_id, bool) or not isinstance(blank_id, Integral) or
            not 0 <= int(blank_id) < emission.shape[1]):
        raise ValueError("`blank_id` must index the emission vocabulary.")
    blank_id = int(blank_id)

    dictionary = _folded_vocabulary(
        vocabulary,
        vocabulary_size=emission.shape[1],
    )
    delimiter = word_delimiter_token.casefold()
    clean: list[tuple[int, str]] = []
    stripped_start = len(text) - len(text.lstrip())
    stripped_stop = len(text.rstrip())
    for index, character in enumerate(text):
        if index < stripped_start or index >= stripped_stop:
            continue
        if (character.isspace() and language in LANGUAGES_WITHOUT_SPACES):
            continue
        token = (
            delimiter
            if character.isspace() and language not in LANGUAGES_WITHOUT_SPACES else character.casefold())
        clean.append((index, token))
    if not clean:
        return CTCAlignment(words=(), characters=())

    has_wildcard = any(token not in dictionary for _, token in clean)
    working_emission = emission
    wildcard_id: int | None = None
    if has_wildcard:
        non_blank = torch.ones(
            emission.shape[1],
            device=emission.device,
            dtype=torch.bool,
        )
        non_blank[blank_id] = False
        wildcard = emission[:, non_blank].amax(dim=1, keepdim=True)
        working_emission = torch.cat((emission, wildcard), dim=1)
        wildcard_id = working_emission.shape[1] - 1
    token_ids = tuple(dictionary.get(token, wildcard_id) for _, token in clean)
    if any(token_id is None for token_id in token_ids):
        raise RuntimeError("Wildcard token construction failed.")
    normalized_ids = tuple(int(token_id) for token_id in token_ids)
    if len(normalized_ids) > working_emission.shape[0]:
        return CTCAlignment(words=(), characters=())

    trellis = build_trellis(
        working_emission,
        normalized_ids,
        blank_id=blank_id,
    )
    path = _backtrack(
        trellis,
        working_emission,
        normalized_ids,
        blank_id=blank_id,
    )
    if path is None:
        return CTCAlignment(words=(), characters=())
    spans = _merge_path(path)
    if len(spans) != len(clean):
        return CTCAlignment(words=(), characters=())

    seconds_per_frame = ((segment_end - segment_start) / working_emission.shape[0])
    characters = tuple(
        AlignedCharacter(
            character=text[text_index],
            text_index=text_index,
            start=segment_start + span.start_frame * seconds_per_frame,
            end=segment_start + span.end_frame * seconds_per_frame,
            confidence=span.score,
        ) for (text_index, _), span in zip(clean, spans))
    words: list[AlignedWord] = []
    for start_index, end_index in _word_spans(text, language=language):
        aligned = tuple(
            character for character in characters
            if start_index <= character.text_index < end_index and not character.character.isspace())
        if not aligned:
            continue
        words.append(
            AlignedWord(
                text=text[start_index:end_index].strip(),
                start=aligned[0].start,
                end=aligned[-1].end,
                confidence=(sum(character.confidence for character in aligned) / len(aligned)),
            ))
    return CTCAlignment(
        words=tuple(words),
        characters=characters,
    )


__all__ = [
    "AlignedCharacter",
    "AlignedWord",
    "CTCAlignment",
    "align_ctc_transcript",
    "build_trellis",
]
