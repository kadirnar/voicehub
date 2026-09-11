"""Training objective for the released TEN VAD graph."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def ten_vad_binary_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    mask: Tensor | None = None,
    positive_weight: float | Tensor | None = None,
) -> Tensor:
    """Masked frame BCE used by VoiceHub's reconstructed tuning recipe."""
    if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("TEN VAD logits and labels must be tensors.")
    if logits.shape != labels.shape:
        raise ValueError("TEN VAD logits and labels must have identical shapes.")
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    if not torch.isfinite(labels).all() or torch.any((labels < 0) | (labels > 1)):
        raise ValueError("TEN VAD labels must be finite and in [0, 1].")
    pos_weight = None
    if positive_weight is not None:
        pos_weight = torch.as_tensor(
            positive_weight,
            dtype=logits.dtype,
            device=logits.device,
        )
        if pos_weight.numel() != 1 or not torch.isfinite(pos_weight).all() or pos_weight <= 0:
            raise ValueError("`positive_weight` must be finite and positive.")
    losses = functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
        pos_weight=pos_weight,
    )
    if mask is None:
        return losses.mean()
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if mask.shape != logits.shape:
        raise ValueError("TEN VAD label mask must match the logits shape.")
    if not mask.any():
        raise ValueError("TEN VAD label mask must select at least one frame.")
    return losses.masked_select(mask).mean()


__all__ = ["ten_vad_binary_cross_entropy"]
