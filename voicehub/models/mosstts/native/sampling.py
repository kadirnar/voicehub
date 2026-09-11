"""Validated sampling primitives shared by native MOSS-TTS generators."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _validate_sampling(
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> None:
    if not math.isfinite(float(temperature)) or float(temperature) < 0.0:
        raise ValueError("`temperature` must be finite and non-negative.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("`top_k` must be a non-negative integer.")
    if not math.isfinite(float(top_p)) or not 0.0 < float(top_p) <= 1.0:
        raise ValueError("`top_p` must be in (0, 1].")
    if (not math.isfinite(float(repetition_penalty)) or float(repetition_penalty) <= 0.0):
        raise ValueError("`repetition_penalty` must be positive.")


def apply_repetition_penalty(
    scores: Tensor,
    previous_token_ids: Tensor | None,
    penalty: float,
) -> Tensor:
    if previous_token_ids is None or float(penalty) == 1.0:
        return scores
    if previous_token_ids.ndim == 1:
        previous_token_ids = previous_token_ids.unsqueeze(0)
    if previous_token_ids.ndim != 2 or previous_token_ids.shape[0] != scores.shape[0]:
        raise ValueError("Previous token IDs must have shape [batch, sequence].")
    updated = scores.clone()
    for batch_index in range(updated.shape[0]):
        token_ids = torch.unique(previous_token_ids[batch_index])
        token_ids = token_ids[(token_ids >= 0) & (token_ids < updated.shape[-1])]
        selected = updated[batch_index].index_select(0, token_ids)
        selected = torch.where(
            selected < 0,
            selected * float(penalty),
            selected / float(penalty),
        )
        updated[batch_index].scatter_(0, token_ids, selected)
    return updated


def filter_logits(
    scores: Tensor,
    *,
    top_k: int,
    top_p: float,
) -> Tensor:
    if top_k and top_k < scores.shape[-1]:
        threshold = torch.topk(scores, top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(
            scores,
            descending=True,
            dim=-1,
        )
        cumulative = torch.softmax(
            sorted_scores,
            dim=-1,
            dtype=torch.float32,
        ).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, remove)
        scores = scores.masked_fill(mask, -torch.inf)
    return scores


def sample_token(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float = 1.0,
    previous_token_ids: Tensor | None = None,
) -> Tensor:
    """Greedy-decode at zero temperature, otherwise multinomial-sample."""
    _validate_sampling(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    scores = apply_repetition_penalty(
        logits.float(),
        previous_token_ids,
        repetition_penalty,
    )
    if temperature == 0.0:
        return torch.argmax(scores, dim=-1)
    scores = filter_logits(
        scores / float(temperature),
        top_k=top_k,
        top_p=top_p,
    )
    probabilities = torch.softmax(scores, dim=-1)
    if not bool(torch.isfinite(probabilities).all()):
        raise RuntimeError("MOSS-TTS sampling produced non-finite probabilities.")
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


__all__ = [
    "apply_repetition_penalty",
    "filter_logits",
    "sample_token",
]
