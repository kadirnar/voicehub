"""Frame objective for the VoiceHub-native SpeechBrain VAD."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def speechbrain_vad_binary_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    label_mask: Tensor | None = None,
    positive_weight: float | Tensor | None = None,
) -> Tensor:
    """Compute the author-compatible masked binary cross entropy.

    The LibriParty target recipe produces ``floor(samples / hop)``
    labels, while centered STFT produces one additional frame.  The
    published training code explicitly slices predictions to target
    length; this implementation preserves that behavior.
    """
    if not isinstance(logits, Tensor) or logits.ndim not in {2, 3}:
        raise ValueError("`logits` must have shape [batch, frames] or [batch, frames, 1].")
    if logits.ndim == 3:
        if logits.shape[-1] != 1:
            raise ValueError("Three-dimensional logits must have one output channel.")
        logits = logits.squeeze(-1)
    targets = torch.as_tensor(labels, device=logits.device)
    if targets.ndim == 3 and targets.shape[-1] == 1:
        targets = targets.squeeze(-1)
    if targets.ndim != 2 or targets.shape[0] != logits.shape[0]:
        raise ValueError("`labels` must have shape [batch, target_frames].")
    if targets.shape[1] > logits.shape[1] or logits.shape[1] - targets.shape[1] > 1:
        raise ValueError(
            "SpeechBrain VAD labels must match logits or omit only the final centered-STFT frame.")
    logits = logits[:, :targets.shape[1]]
    targets = targets.to(dtype=logits.dtype)
    if not torch.isfinite(targets).all() or torch.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("Binary VAD labels must be finite and in [0, 1].")
    weight = None
    if positive_weight is not None:
        weight = torch.as_tensor(
            positive_weight,
            dtype=logits.dtype,
            device=logits.device,
        )
        if weight.numel() != 1 or not torch.isfinite(weight).all() or weight.item() <= 0:
            raise ValueError("`positive_weight` must be one finite positive value.")
    losses = functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=weight,
    )
    if label_mask is None:
        return losses.mean()
    mask = torch.as_tensor(
        label_mask,
        dtype=torch.bool,
        device=logits.device,
    )
    if mask.shape != losses.shape:
        raise ValueError("`label_mask` must match the target frame shape.")
    selected = losses.masked_select(mask)
    if selected.numel() == 0:
        raise ValueError("`label_mask` must select at least one frame.")
    return selected.mean()


__all__ = ["speechbrain_vad_binary_cross_entropy"]
