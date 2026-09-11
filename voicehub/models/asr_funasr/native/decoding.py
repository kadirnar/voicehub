"""Greedy CTC decoding and the published SenseVoice timestamp alignment."""

from __future__ import annotations

from itertools import groupby

import torch
from torch import Tensor

from voicehub.outputs import ASRWord


def ctc_greedy_tokens(
    log_probabilities: Tensor,
    length: int,
    *,
    blank_token_id: int = 0,
) -> tuple[int, ...]:
    if log_probabilities.ndim != 2:
        raise ValueError("CTC probabilities must have shape [frames, tokens].")
    if (isinstance(length, bool) or not isinstance(length, int) or
            not 0 <= length <= log_probabilities.shape[0]):
        raise ValueError("CTC length is outside the probability matrix.")
    if not 0 <= blank_token_id < log_probabilities.shape[1]:
        raise ValueError("CTC blank ID is outside the vocabulary.")
    predicted = log_probabilities[:length].argmax(dim=-1)
    collapsed = torch.unique_consecutive(predicted)
    return tuple(int(token_id) for token_id in collapsed.tolist() if token_id != blank_token_id)


def ctc_forced_align(
    log_probabilities: Tensor,
    targets: Tensor,
    *,
    blank_token_id: int = 0,
) -> Tensor:
    """Align one target sequence with the exact published Viterbi topology."""
    if log_probabilities.ndim != 2:
        raise ValueError("CTC probabilities must have shape [frames, tokens].")
    if targets.ndim != 1:
        raise ValueError("CTC targets must have shape [tokens].")
    frames = log_probabilities.shape[0]
    target_length = targets.numel()
    if frames < 1:
        raise ValueError("CTC alignment requires at least one frame.")
    if target_length < 1:
        return torch.full(
            (frames, ),
            blank_token_id,
            dtype=torch.long,
            device=log_probabilities.device,
        )
    if target_length > frames:
        raise ValueError("CTC target length cannot exceed the number of emission frames.")
    targets = targets.to(
        device=log_probabilities.device,
        dtype=torch.long,
    )
    if torch.any(targets < 0) or torch.any(targets >= log_probabilities.shape[1]):
        raise ValueError("CTC target IDs are outside the vocabulary.")
    expanded = torch.stack(
        (
            torch.full_like(targets, blank_token_id),
            targets,
        ),
        dim=-1,
    ).flatten()
    expanded = torch.cat((
        expanded,
        torch.full_like(targets[:1], blank_token_id),
    ))
    different = torch.cat((
        torch.tensor(
            [False, False],
            device=targets.device,
        ),
        expanded[2:] != expanded[:-2],
    ))
    negative_infinity = torch.tensor(
        -float("inf"),
        dtype=log_probabilities.dtype,
        device=log_probabilities.device,
    )
    padding = 2
    padded_states = padding + expanded.numel()
    scores = torch.full(
        (padded_states, ),
        negative_infinity,
        dtype=log_probabilities.dtype,
        device=log_probabilities.device,
    )
    scores[padding] = log_probabilities[0, blank_token_id]
    scores[padding + 1] = log_probabilities[0, expanded[1]]
    backpointers = torch.zeros(
        (frames, padded_states),
        dtype=torch.long,
        device=targets.device,
    )
    for frame in range(1, frames):
        previous = torch.stack((
            scores[2:],
            scores[1:-1],
            torch.where(different, scores[:-2], negative_infinity),
        ))
        best_score, best_index = previous.max(dim=0)
        scores[padding:] = (log_probabilities[frame].gather(0, expanded) + best_score)
        backpointers[frame, padding:] = best_index
    final_states = torch.stack((
        scores[padding + target_length * 2 - 1],
        scores[padding + target_length * 2],
    ))
    path = torch.zeros(
        (frames, ),
        dtype=torch.long,
        device=targets.device,
    )
    path[-1] = (padding + target_length * 2 - 1 + int(final_states.argmax().item()))
    for frame in range(frames - 1, 0, -1):
        state = path[frame]
        path[frame - 1] = state - backpointers[frame, state]
    return expanded.gather(0, (path - padding).clamp_min(0))


def _piece_intervals(
    alignment: Tensor,
    pieces: tuple[str, ...],
    *,
    blank_token_id: int,
    frame_seconds: float,
    center_offset_seconds: float,
    duration: float,
) -> tuple[tuple[str, float, float], ...]:
    intervals = []
    start_frame = 0
    piece_index = 0
    for token_id, grouped in groupby(alignment.tolist()):
        end_frame = start_frame + len(tuple(grouped))
        if token_id != blank_token_id and piece_index < len(pieces):
            start = max(
                start_frame * frame_seconds - center_offset_seconds,
                0.0,
            )
            end = min(
                end_frame * frame_seconds - center_offset_seconds,
                duration,
            )
            intervals.append((
                pieces[piece_index],
                min(start, duration),
                max(min(end, duration), min(start, duration)),
            ))
            piece_index += 1
        start_frame = end_frame
    return tuple(intervals)


def sensevoice_word_timestamps(
    log_probabilities: Tensor,
    target_ids: tuple[int, ...],
    pieces: tuple[str, ...],
    *,
    duration: float,
    blank_token_id: int = 0,
    frame_seconds: float = 0.060,
    center_offset_seconds: float = 0.030,
) -> tuple[ASRWord, ...]:
    """Return published 60 ms LFR token alignment grouped into words."""
    if len(target_ids) != len(pieces):
        raise ValueError("SenseVoice target IDs and pieces must align.")
    if not target_ids:
        return ()
    targets = torch.tensor(
        target_ids,
        dtype=torch.long,
        device=log_probabilities.device,
    )
    alignment = ctc_forced_align(
        log_probabilities,
        targets,
        blank_token_id=blank_token_id,
    )
    intervals = _piece_intervals(
        alignment,
        pieces,
        blank_token_id=blank_token_id,
        frame_seconds=frame_seconds,
        center_offset_seconds=center_offset_seconds,
        duration=duration,
    )
    words: list[ASRWord] = []
    previous_piece: str | None = None
    for piece, start, end in intervals:
        if piece == "\u2581":
            previous_piece = piece
            continue
        if piece.startswith("\u2581"):
            text = piece[1:]
            if text:
                words.append(ASRWord(text=text, start=start, end=end))
        elif (words and previous_piece is not None and previous_piece.isascii() and
              previous_piece.isalpha() and piece.isascii() and piece.isalpha()):
            previous = words[-1]
            words[-1] = ASRWord(
                text=previous.text + piece,
                start=previous.start,
                end=end,
            )
        elif piece:
            words.append(ASRWord(text=piece, start=start, end=end))
        previous_piece = piece
    return tuple(words)


__all__ = [
    "ctc_forced_align",
    "ctc_greedy_tokens",
    "sensevoice_word_timestamps",
]
