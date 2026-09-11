"""Teacher-forced ZONOS2 objective reconstructed from the published graph.

Zyphra publishes ``loss_softcap`` and the causal multi-codebook
architecture, but not its original optimizer, data pipeline, or training
loop.  VoiceHub's objective is therefore intentionally labelled
*reconstructed*: each position predicts the next delayed audio row and
cross-entropy is averaged over valid codebook targets.  It is
mathematically verified in VoiceHub tests, but it is not represented as
an author-published recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(slots=True)
class Zonos2TrainingOutput:
    loss: Tensor
    per_codebook_loss: Tensor
    token_count: Tensor


def zonos2_causal_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    audio_pad_id: int,
    loss_mask: Tensor | None = None,
    ignore_index: int = -100,
) -> Zonos2TrainingOutput:
    """Compute next-row cross-entropy for delayed ZONOS2 audio streams.

    Args:
        logits: ``[batch, time, codebooks, audio_vocab]`` model logits.
        labels: ``[batch, time, codebooks]`` token rows aligned with the model
            inputs. The function performs the causal one-position shift.
        audio_pad_id: Padding token excluded from the objective.
        loss_mask: Optional ``[batch, time]`` boolean mask. This is useful for
            restricting loss to the audio completion after a text prompt.
        ignore_index: Additional sentinel excluded from the objective.
    """
    if logits.ndim != 4:
        raise ValueError("ZONOS2 logits must have shape [batch, time, codebooks, vocab].")
    if labels.shape != logits.shape[:-1]:
        raise ValueError(
            f"ZONOS2 labels must have shape {tuple(logits.shape[:-1])}, "
            f"received {tuple(labels.shape)}.")
    if labels.dtype == torch.bool or labels.is_floating_point():
        raise TypeError("ZONOS2 labels must use an integer dtype.")
    if logits.shape[1] < 2:
        raise ValueError("ZONOS2 causal loss requires at least two time steps.")
    if not 0 <= audio_pad_id < logits.shape[-1]:
        raise ValueError("`audio_pad_id` is outside the logits vocabulary.")

    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:].long()
    valid = ((shifted_labels != ignore_index)
             & (shifted_labels != audio_pad_id)
             & (shifted_labels >= 0)
             & (shifted_labels < logits.shape[-1]))
    if loss_mask is not None:
        if loss_mask.shape != labels.shape[:2]:
            raise ValueError("ZONOS2 loss mask must have shape [batch, time].")
        valid = valid & loss_mask[:, 1:].to(
            device=valid.device,
            dtype=torch.bool,
        ).unsqueeze(-1)

    safe_labels = shifted_labels.masked_fill(~valid, 0)
    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
        safe_labels.reshape(-1),
        reduction="none",
    ).view_as(safe_labels)
    weighted = losses * valid.to(dtype=losses.dtype)
    counts = valid.sum(dim=(0, 1))
    totals = weighted.sum(dim=(0, 1))
    per_codebook = torch.where(
        counts > 0,
        totals / counts.clamp_min(1).to(dtype=totals.dtype),
        torch.zeros_like(totals),
    )
    token_count = counts.sum()
    loss = weighted.sum() / token_count.clamp_min(1).to(dtype=weighted.dtype)
    return Zonos2TrainingOutput(
        loss=loss,
        per_codebook_loss=per_codebook,
        token_count=token_count,
    )


__all__ = [
    "Zonos2TrainingOutput",
    "zonos2_causal_cross_entropy",
]
