"""Differentiable Token-and-Duration Transducer objective.

This is a native PyTorch port of NeMo's anti-diagonal TDT dynamic
program, cross-checked against Transformers revision
``af71155683b4d34dd92d8f037392fa6bf334035e``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def tdt_loss(
    token_logits: torch.Tensor,
    duration_logits: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_token_id: int,
    durations: Sequence[int],
    *,
    sigma: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute exact TDT negative log likelihood.

    Shapes are ``[batch, time, labels + 1, vocabulary]`` for token
    logits, ``[batch, time, labels + 1, durations]`` for duration
    logits, and ``[batch, labels]`` for targets.
    """
    valid_reductions = {"mean_volume", "mean_batch", "mean", "sum", "none"}
    if reduction not in valid_reductions:
        choices = ", ".join(sorted(valid_reductions))
        raise ValueError(f"Unsupported TDT reduction {reduction!r}; choose {choices}.")
    if token_logits.ndim != 4 or duration_logits.ndim != 4:
        raise ValueError("TDT token and duration logits must be four-dimensional.")
    if token_logits.shape[:3] != duration_logits.shape[:3]:
        raise ValueError("TDT token and duration lattice dimensions must match.")
    if targets.ndim != 2:
        raise ValueError("TDT targets must have shape [batch, labels].")
    if targets.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
    }:
        raise TypeError("TDT targets must contain integer token IDs.")
    batch_size, maximum_time, maximum_u, vocabulary_size = token_logits.shape
    device = token_logits.device
    duration_values = tuple(durations)
    if duration_logits.shape[-1] != len(duration_values):
        raise ValueError("Duration-logit width does not match configured durations.")
    if targets.shape[0] != batch_size:
        raise ValueError("TDT targets and logits use different batch sizes.")
    if maximum_u != targets.shape[1] + 1:
        raise ValueError("TDT lattice must contain target_length + 1 label states.")
    if not 0 <= blank_token_id < vocabulary_size:
        raise ValueError("TDT blank token ID is outside the token vocabulary.")
    if (logit_lengths.shape != (batch_size, ) or target_lengths.shape != (batch_size, )):
        raise ValueError("TDT lengths must have shape [batch].")
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if (logit_lengths.dtype not in integer_dtypes or target_lengths.dtype not in integer_dtypes):
        raise TypeError("TDT logit and target lengths must be integer tensors.")
    if not duration_values or duration_values[0] != 0:
        raise ValueError("TDT durations must begin with zero.")
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in duration_values) or
            tuple(sorted(set(duration_values))) != duration_values):
        raise ValueError("TDT durations must be unique, non-negative, increasing integers.")
    if torch.any(logit_lengths <= 0) or torch.any(logit_lengths > maximum_time):
        raise ValueError("TDT logit lengths are outside the lattice.")
    if torch.any(target_lengths <= 0) or torch.any(target_lengths >= maximum_u):
        raise ValueError("TDT target lengths must be positive and fit the lattice.")
    device_targets = targets.to(device=device, dtype=torch.long)
    if torch.any(device_targets < 0) or torch.any(device_targets >= vocabulary_size):
        raise ValueError("TDT targets contain IDs outside the token vocabulary.")
    supervised = (
        torch.arange(targets.shape[1], device=device)[None, :]
        < target_lengths.to(device=device, dtype=torch.long)[:, None])
    if torch.any(device_targets.masked_select(supervised) == blank_token_id):
        raise ValueError("TDT supervised targets cannot contain the transducer blank ID.")

    token_log_probs = torch.log_softmax(token_logits.float(), dim=-1) - float(sigma)
    duration_log_probs = torch.log_softmax(duration_logits.float(), dim=-1)
    log_alpha = torch.full(
        (batch_size, maximum_time, maximum_u),
        float("-inf"),
        device=device,
    )
    log_alpha[:, 0, 0] = 0.0
    blank_log_probs = token_log_probs[..., blank_token_id]
    expanded_targets = device_targets.unsqueeze(1)
    expanded_targets = expanded_targets.expand(-1, maximum_time, -1)
    label_log_probs = torch.gather(
        token_log_probs[:, :, :maximum_u - 1, :],
        dim=3,
        index=expanded_targets.unsqueeze(-1),
    ).squeeze(-1)
    negative_infinity = torch.tensor(float("-inf"), device=device)

    for diagonal in range(1, maximum_time + maximum_u - 1):
        u_start = max(0, diagonal - maximum_time + 1)
        u_end = min(diagonal + 1, maximum_u)
        u_indices = torch.arange(u_start, u_end, device=device)
        t_indices = diagonal - u_indices
        candidates: list[torch.Tensor] = []
        for duration_index, duration in enumerate(duration_values):
            previous_t = t_indices - duration
            valid_t = previous_t >= 0
            if not torch.any(valid_t):
                continue
            source_t = previous_t.clamp(min=0)
            if duration > 0:
                blank = (
                    log_alpha[:, source_t, u_indices] + blank_log_probs[:, source_t, u_indices] +
                    duration_log_probs[:, source_t, u_indices, duration_index])
                candidates.append(torch.where(valid_t.unsqueeze(0), blank, negative_infinity))
            valid_label = valid_t & (u_indices > 0)
            if torch.any(valid_label):
                source_u = (u_indices - 1).clamp(min=0)
                label = (
                    log_alpha[:, source_t, source_u] + label_log_probs[:, source_t, source_u] +
                    duration_log_probs[:, source_t, source_u, duration_index])
                candidates.append(torch.where(
                    valid_label.unsqueeze(0),
                    label,
                    negative_infinity,
                ))
        if candidates:
            log_alpha[:, t_indices, u_indices] = torch.logsumexp(
                torch.stack(candidates),
                dim=0,
            )

    batch_indices = torch.arange(batch_size, device=device)
    terminal_log_probs = torch.full(
        (batch_size, ),
        float("-inf"),
        device=device,
    )
    device_logit_lengths = logit_lengths.to(device=device, dtype=torch.long)
    device_target_lengths = target_lengths.to(device=device, dtype=torch.long)
    for duration_index, duration in enumerate(duration_values):
        if duration == 0:
            continue
        final_time = device_logit_lengths - duration
        valid = final_time >= 0
        if not torch.any(valid):
            continue
        source_t = final_time.clamp(min=0)
        terminal = (
            log_alpha[batch_indices, source_t, device_target_lengths] + token_log_probs[
                batch_indices,
                source_t,
                device_target_lengths,
                blank_token_id,
            ] + duration_log_probs[
                batch_indices,
                source_t,
                device_target_lengths,
                duration_index,
            ])
        combined = torch.logsumexp(
            torch.stack((terminal_log_probs, terminal)),
            dim=0,
        )
        terminal_log_probs = torch.where(valid, combined, terminal_log_probs)

    losses = -terminal_log_probs
    length_scale = device_target_lengths.float()
    if reduction == "mean_volume":
        return losses.sum() / length_scale.sum()
    if reduction == "mean_batch":
        return losses.mean()
    if reduction == "mean":
        return (losses / length_scale).mean()
    if reduction == "sum":
        return losses.sum()
    return losses


__all__ = ["tdt_loss"]
