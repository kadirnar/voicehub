"""Pure-PyTorch monotonic alignment utilities for native VITS training."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor
from torch.nn import functional


def sequence_mask(lengths: Tensor, maximum_length: int | None = None) -> Tensor:
    """Return a right-padded boolean mask for positive sequence lengths."""
    if not isinstance(lengths, Tensor) or lengths.ndim != 1:
        raise ValueError("`lengths` must be a one-dimensional tensor.")
    if lengths.dtype == torch.bool or lengths.is_floating_point():
        raise TypeError("`lengths` must use an integer dtype.")
    normalized = lengths.to(dtype=torch.long)
    if (normalized < 1).any():
        raise ValueError("Every sequence length must be positive.")
    if maximum_length is None:
        maximum = int(normalized.max().item())
    else:
        if isinstance(maximum_length, bool) or not isinstance(maximum_length, Integral):
            raise TypeError("`maximum_length` must be an integer or None.")
        maximum = int(maximum_length)
        if maximum < 1:
            raise ValueError("`maximum_length` must be positive.")
        if (normalized > maximum).any():
            raise ValueError("A sequence length exceeds `maximum_length`.")
    positions = torch.arange(maximum, device=normalized.device)
    return positions.unsqueeze(0) < normalized.unsqueeze(1)


def generate_path(durations: Tensor, attention_mask: Tensor) -> Tensor:
    """Expand integer token durations into a hard monotonic attention path."""
    if not isinstance(durations, Tensor) or durations.ndim != 3:
        raise ValueError("`durations` must have shape [batch, 1, text].")
    if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 4:
        raise ValueError("`attention_mask` must have shape [batch, 1, frames, text].")
    batch, channels, frames, text = attention_mask.shape
    if batch < 1 or frames < 1 or text < 1:
        raise ValueError("Alignment dimensions must be non-empty.")
    if channels != 1 or tuple(durations.shape) != (batch, 1, text):
        raise ValueError("Duration and attention-mask dimensions do not match.")
    if durations.device != attention_mask.device:
        raise ValueError("Durations and attention masks must share a device.")
    if durations.dtype == torch.bool:
        raise TypeError("`durations` cannot use a boolean dtype.")
    if durations.is_complex():
        raise TypeError("`durations` must be real-valued.")
    if durations.is_floating_point():
        duration_values = durations
    else:
        duration_values = durations.to(dtype=torch.float32)
    if not torch.isfinite(duration_values).all() or (duration_values < 0).any():
        raise ValueError("`durations` must be finite and non-negative.")
    if not torch.equal(duration_values, duration_values.round()):
        raise ValueError("`durations` must contain whole frame counts.")

    mask, frame_lengths, _ = _rectangular_alignment_mask(
        attention_mask[:, 0],
        name="attention_mask",
    )
    valid_text = mask.any(dim=1)
    if (duration_values.squeeze(1)[valid_text] < 1).any():
        raise ValueError("Every valid text token needs at least one frame.")
    if (duration_values.squeeze(1)[~valid_text] != 0).any():
        raise ValueError("Padded text tokens must have zero duration.")
    if not torch.equal(
            duration_values.sum(dim=(1, 2)).long(),
            frame_lengths,
    ):
        raise ValueError("Token durations must cover every valid acoustic frame.")

    cumulative = torch.cumsum(duration_values, dim=-1)
    cumulative = cumulative.reshape(batch * text, 1)
    positions = torch.arange(
        frames,
        dtype=duration_values.dtype,
        device=duration_values.device,
    )
    path = (positions.unsqueeze(0) < cumulative).to(dtype=duration_values.dtype)
    path = path.reshape(batch, text, frames)
    previous = functional.pad(path, (0, 0, 1, 0))[:, :-1]
    return ((path - previous).unsqueeze(1).transpose(2, 3).to(dtype=attention_mask.dtype) * attention_mask)


def _rectangular_alignment_mask(
    value: Tensor,
    *,
    name: str,
) -> tuple[Tensor, Tensor, Tensor]:
    if value.is_complex() or not ((value == 0) | (value == 1)).all():
        raise ValueError(f"`{name}` must contain only zero and one.")
    mask = value.to(dtype=torch.bool)
    frame_valid = mask.any(dim=2)
    text_valid = mask.any(dim=1)
    if not frame_valid.any(dim=1).all() or not text_valid.any(dim=1).all():
        raise ValueError("Every alignment item needs text and acoustic frames.")
    frame_gap = ((~frame_valid[:, :-1]) & frame_valid[:, 1:]).any()
    text_gap = ((~text_valid[:, :-1]) & text_valid[:, 1:]).any()
    if frame_gap or text_gap:
        raise ValueError(f"`{name}` must describe contiguous right padding.")
    expected = frame_valid.unsqueeze(2) & text_valid.unsqueeze(1)
    if not torch.equal(mask, expected):
        raise ValueError(f"`{name}` must contain one valid rectangle per item.")
    return (
        mask,
        frame_valid.sum(dim=1, dtype=torch.long),
        text_valid.sum(dim=1, dtype=torch.long),
    )


@torch.no_grad()
def maximum_path(scores: Tensor, attention_mask: Tensor) -> Tensor:
    """Compute VITS monotonic alignment search without Cython or NumPy.

    The dynamic program matches the recurrence and strict tie-breaking
    in the original VITS Cython kernel while keeping tensors on their
    current device. Alignment itself is intentionally non-
    differentiable; the resulting path is detached before the generator
    graph consumes it.
    """
    if not isinstance(scores, Tensor) or scores.ndim != 3:
        raise ValueError("`scores` must have shape [batch, frames, text].")
    if not isinstance(attention_mask, Tensor):
        raise TypeError("`attention_mask` must be a tensor.")
    if attention_mask.ndim == 4:
        if attention_mask.shape[1] != 1:
            raise ValueError("Four-dimensional alignment masks need one channel.")
        mask = attention_mask[:, 0]
    elif attention_mask.ndim == 3:
        mask = attention_mask
    else:
        raise ValueError(
            "`attention_mask` must have shape [batch, frames, text] or "
            "[batch, 1, frames, text].")
    if tuple(mask.shape) != tuple(scores.shape):
        raise ValueError("Alignment scores and mask must have equal shapes.")
    if any(dimension < 1 for dimension in scores.shape):
        raise ValueError("Alignment dimensions must be non-empty.")
    if mask.device != scores.device:
        raise ValueError("Alignment scores and masks must share a device.")
    if not scores.is_floating_point():
        raise TypeError("Alignment scores must use a floating-point dtype.")
    if not torch.isfinite(scores).all():
        raise ValueError("Alignment scores cannot contain NaN or infinity.")
    mask, frame_lengths, text_lengths = _rectangular_alignment_mask(
        mask,
        name="attention_mask",
    )
    if (text_lengths > frame_lengths).any():
        raise ValueError("Monotonic alignment requires at least one frame per text token.")

    batch, frame_count, text_count = scores.shape
    unreachable_value = torch.tensor(
        -1e9,
        dtype=scores.dtype,
        device=scores.device,
    )
    values = torch.full(
        (batch, text_count),
        unreachable_value,
        dtype=scores.dtype,
        device=scores.device,
    )
    choices = torch.zeros(
        (batch, frame_count, text_count),
        dtype=torch.bool,
        device=scores.device,
    )
    values[:, 0] = scores[:, 0, 0]
    values = torch.where(mask[:, 0], values, unreachable_value)
    text_positions = torch.arange(
        text_count,
        device=scores.device,
    ).unsqueeze(0)

    for frame in range(1, frame_count):
        stay = torch.where(
            text_positions == frame,
            unreachable_value,
            values,
        )
        advance = functional.pad(
            values[:, :-1],
            (1, 0),
            value=-1e9,
        )
        choose_advance = advance > stay
        updated = torch.maximum(stay, advance) + scores[:, frame]
        lower_bound = (text_lengths + frame - frame_lengths).unsqueeze(1)
        reachable = (mask[:, frame] & (text_positions <= frame) & (text_positions >= lower_bound))
        values = torch.where(reachable, updated, unreachable_value)
        choices[:, frame] = choose_advance

    path = torch.zeros_like(scores)
    batch_indices = torch.arange(batch, device=scores.device)
    text_indices = text_lengths - 1
    for frame in range(frame_count - 1, -1, -1):
        active = frame < frame_lengths
        path[batch_indices, frame, text_indices] = active.to(dtype=path.dtype)
        if frame:
            selected = choices[batch_indices, frame, text_indices]
            must_advance = (text_indices == frame) | selected
            decrement = active & (text_indices > 0) & must_advance
            text_indices = text_indices - decrement.to(dtype=torch.long)
    if (text_indices != 0).any():
        raise RuntimeError("Monotonic alignment backtracking failed.")
    return path * mask.to(dtype=path.dtype)


__all__ = ["generate_path", "maximum_path", "sequence_mask"]
