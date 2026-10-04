"""Composable, row-wise stopping criteria."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class StoppingCriterion(Protocol):
    """Protocol for model- or task-specific row-wise stopping logic."""

    def __call__(self, sequences: Tensor, next_tokens: Tensor, step_index: int) -> Tensor:
        """Return one boolean stop decision for every batch row."""
        ...


def tokens_match_any(token_ids: Tensor, candidates: Sequence[int]) -> Tensor:
    """Return a mask indicating which tokens match any candidate ID."""
    if token_ids.ndim != 1:
        raise ValueError("`token_ids` must have shape [batch].")
    candidates = tuple(candidates)
    if not candidates:
        return torch.zeros_like(token_ids, dtype=torch.bool)
    # Compare against Python scalars: building a candidate tensor would copy
    # it from host memory on every decoding step, which synchronizes CUDA.
    matches = token_ids == candidates[0]
    for candidate in candidates[1:]:
        matches |= token_ids == candidate
    return matches


@dataclass(frozen=True, slots=True)
class EosStoppingCriterion:
    """Stop individual rows when one of their terminal tokens is emitted."""

    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("`token_ids` must contain at least one terminal token.")
        if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in self.token_ids):
            raise TypeError("Every terminal token ID must be an integer.")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("Terminal token IDs cannot be negative.")

    def __call__(self, sequences: Tensor, next_tokens: Tensor, step_index: int) -> Tensor:
        del sequences, step_index
        return tokens_match_any(next_tokens, self.token_ids)


def evaluate_stopping_criteria(
    criteria: Sequence[StoppingCriterion],
    sequences: Tensor,
    next_tokens: Tensor,
    step_index: int,
) -> Tensor:
    """Combine stopping decisions and validate every criterion's contract."""
    stopped = torch.zeros(
        next_tokens.shape[0],
        dtype=torch.bool,
        device=next_tokens.device,
    )
    for criterion in criteria:
        decision = criterion(sequences, next_tokens, step_index)
        if not isinstance(decision, Tensor):
            raise TypeError("A stopping criterion must return a PyTorch tensor.")
        if decision.shape != stopped.shape:
            raise ValueError(
                "A stopping criterion must return shape "
                f"{tuple(stopped.shape)}, found {tuple(decision.shape)}.")
        if decision.device != stopped.device:
            raise ValueError("Stopping decisions must use the token tensor's device.")
        if decision.dtype != torch.bool:
            raise TypeError("A stopping criterion must return a boolean tensor.")
        stopped |= decision
    return stopped
