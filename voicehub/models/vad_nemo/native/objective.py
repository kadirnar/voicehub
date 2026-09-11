"""Fine-tuning objective for native MarbleNet frame classification."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def marblenet_vad_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    label_mask: Tensor | None = None,
) -> Tensor:
    """Compute masked hard- or soft-label frame cross entropy."""
    if not isinstance(logits, Tensor) or logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("`logits` must have shape [batch, frames, 2].")
    targets = torch.as_tensor(labels, device=logits.device)
    if targets.ndim == 3 and targets.shape == logits.shape:
        if not targets.is_floating_point():
            targets = targets.float()
        if not torch.isfinite(targets).all() or torch.any(targets < 0):
            raise ValueError("Soft VAD labels must be finite and non-negative.")
        totals = targets.sum(dim=-1)
        if not torch.allclose(totals, torch.ones_like(totals), atol=1e-5):
            raise ValueError("Soft VAD label rows must sum to one.")
        losses = -(targets * logits.log_softmax(dim=-1)).sum(dim=-1)
    elif targets.ndim == 2 and targets.shape == logits.shape[:2]:
        if targets.is_floating_point():
            if not torch.isfinite(targets).all() or torch.any((targets < 0) | (targets > 1)):
                raise ValueError("Binary VAD targets must be finite and in [0, 1].")
            probabilities = torch.stack((1.0 - targets, targets), dim=-1)
            losses = -(probabilities * logits.log_softmax(dim=-1)).sum(dim=-1)
        else:
            targets = targets.long()
            if torch.any((targets < 0) | (targets >= 2)):
                raise ValueError("Class VAD labels must contain only 0 or 1.")
            losses = functional.cross_entropy(
                logits.transpose(1, 2),
                targets,
                reduction="none",
            )
    else:
        raise ValueError("`labels` must have shape [batch, frames] or [batch, frames, 2].")

    if label_mask is None:
        return losses.mean()
    mask = torch.as_tensor(
        label_mask,
        dtype=torch.bool,
        device=logits.device,
    )
    if mask.shape != losses.shape:
        raise ValueError("`label_mask` must have shape [batch, frames].")
    selected = losses.masked_select(mask)
    if selected.numel() == 0:
        raise ValueError("`label_mask` must select at least one frame.")
    return selected.mean()


__all__ = ["marblenet_vad_loss"]
