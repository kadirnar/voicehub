"""Differentiable PyTorch-only RNN-T objective."""

from __future__ import annotations

import torch
from torch import Tensor


def rnnt_loss(
    logits: Tensor,
    targets: Tensor,
    logit_lengths: Tensor,
    target_lengths: Tensor,
    blank_token_id: int,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Compute the exact RNN-T negative log likelihood.

    This implementation intentionally favors a small, auditable dynamic
    program over a fused provider-specific kernel.  It is suitable for
    fine-tuning and correctness tests; an optimization backend may
    replace it later without changing the model/trainer contract.
    """
    if logits.ndim != 4:
        raise ValueError("RNN-T logits must have shape [batch, time, labels + 1, vocab].")
    if targets.ndim != 2:
        raise ValueError("RNN-T targets must have shape [batch, labels].")
    batch, maximum_time, maximum_states, vocabulary = logits.shape
    if targets.shape[0] != batch:
        raise ValueError("RNN-T logits and targets use different batches.")
    if maximum_states != targets.shape[1] + 1:
        raise ValueError("RNN-T logits must contain target width + 1 label states.")
    if (isinstance(blank_token_id, bool) or not isinstance(blank_token_id, int) or
            not 0 <= blank_token_id < vocabulary):
        raise ValueError("RNN-T blank token is outside the vocabulary.")
    lengths = torch.as_tensor(
        logit_lengths,
        dtype=torch.long,
        device=logits.device,
    )
    label_lengths = torch.as_tensor(
        target_lengths,
        dtype=torch.long,
        device=logits.device,
    )
    if lengths.shape != (batch, ) or label_lengths.shape != (batch, ):
        raise ValueError("RNN-T lengths must have shape [batch].")
    if torch.any(lengths <= 0) or torch.any(lengths > maximum_time):
        raise ValueError("RNN-T logit lengths are outside the lattice.")
    if (torch.any(label_lengths < 0) or torch.any(label_lengths >= maximum_states)):
        raise ValueError("RNN-T target lengths are outside the lattice.")
    if reduction not in {"none", "sum", "mean", "mean_batch"}:
        raise ValueError("RNN-T reduction must be none, sum, mean, or mean_batch.")

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    losses: list[Tensor] = []
    negative_infinity = logits.new_tensor(float("-inf"), dtype=torch.float32)
    for batch_index in range(batch):
        time_steps = int(lengths[batch_index])
        target_count = int(label_lengths[batch_index])
        target = targets[batch_index, :target_count].to(
            device=logits.device,
            dtype=torch.long,
        )
        if torch.any(target < 0) or torch.any(target >= vocabulary):
            raise ValueError("RNN-T target contains an invalid token ID.")
        if torch.any(target == blank_token_id):
            raise ValueError("RNN-T target labels cannot contain the blank token.")

        previous: list[Tensor] = [negative_infinity] * (target_count + 1)
        previous[0] = logits.new_zeros((), dtype=torch.float32)
        for time_index in range(time_steps):
            current: list[Tensor] = [negative_infinity] * (target_count + 1)
            if time_index == 0:
                current[0] = previous[0]
            else:
                current[0] = (previous[0] + log_probs[
                    batch_index,
                    time_index - 1,
                    0,
                    blank_token_id,
                ])
            for label_index in range(1, target_count + 1):
                emit = (
                    current[label_index - 1] + log_probs[
                        batch_index,
                        time_index,
                        label_index - 1,
                        target[label_index - 1],
                    ])
                if time_index == 0:
                    current[label_index] = emit
                else:
                    advance = (
                        previous[label_index] + log_probs[
                            batch_index,
                            time_index - 1,
                            label_index,
                            blank_token_id,
                        ])
                    current[label_index] = torch.logaddexp(advance, emit)
            previous = current
        terminal = (
            previous[target_count] + log_probs[
                batch_index,
                time_steps - 1,
                target_count,
                blank_token_id,
            ])
        losses.append(-terminal)

    stacked = torch.stack(losses)
    if reduction == "none":
        return stacked
    if reduction == "sum":
        return stacked.sum()
    if reduction == "mean_batch":
        return stacked.mean()
    denominators = label_lengths.clamp(min=1).to(
        device=stacked.device,
        dtype=stacked.dtype,
    )
    return (stacked / denominators).mean()


__all__ = ["rnnt_loss"]
