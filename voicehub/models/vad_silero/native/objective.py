"""Supervised frame objective for the native Silero VAD graph."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

Reduction = Literal["mean", "none", "sum"]


def _target_tensor(name: str, value: Tensor, *, shape: torch.Size) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"`{name}` must be a PyTorch tensor.")
    if value.shape != shape:
        raise ValueError(f"`{name}` must have shape {tuple(shape)}.")
    if value.dtype == torch.bool or not value.is_floating_point():
        raise TypeError(f"`{name}` must use a floating-point dtype.")
    if not torch.isfinite(value).all():
        raise ValueError(f"`{name}` cannot contain NaN or infinite values.")
    return value


def silero_vad_binary_cross_entropy(
    predictions: Tensor,
    targets: Tensor,
    *,
    weights: Tensor | None = None,
    from_logits: bool = False,
    reduction: Reduction = "mean",
) -> Tensor:
    """Compute the official frame-level binary objective.

    ``weights`` follows the tuning recipe's mask semantics.  Mean
    reduction divides by the number of frames, not by the sum of
    weights, matching ``(binary_cross_entropy * mask).mean()`` in the
    released training code.
    """
    if not isinstance(predictions, Tensor):
        raise TypeError("`predictions` must be a PyTorch tensor.")
    if predictions.ndim < 1:
        raise ValueError("`predictions` must contain at least one dimension.")
    if not predictions.is_floating_point():
        raise TypeError("`predictions` must use a floating-point dtype.")
    if not torch.isfinite(predictions).all():
        raise ValueError("`predictions` cannot contain NaN or infinite values.")
    if not isinstance(from_logits, bool):
        raise TypeError("`from_logits` must be a boolean.")
    if reduction not in ("mean", "none", "sum"):
        raise ValueError("`reduction` must be 'mean', 'none', or 'sum'.")

    targets = _target_tensor(
        "targets",
        targets,
        shape=predictions.shape,
    ).to(
        device=predictions.device, dtype=predictions.dtype)
    if ((targets < 0.0) | (targets > 1.0)).any():
        raise ValueError("`targets` must be probabilities in the interval [0, 1].")

    if from_logits:
        losses = functional.binary_cross_entropy_with_logits(
            predictions,
            targets,
            reduction="none",
        )
    else:
        if ((predictions < 0.0) | (predictions > 1.0)).any():
            raise ValueError("`predictions` must be probabilities when `from_logits` is False.")
        losses = functional.binary_cross_entropy(
            predictions,
            targets,
            reduction="none",
        )

    if weights is not None:
        weights = _target_tensor(
            "weights",
            weights,
            shape=predictions.shape,
        ).to(
            device=predictions.device, dtype=losses.dtype)
        if (weights < 0.0).any():
            raise ValueError("`weights` cannot contain negative values.")
        losses = losses * weights

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


class SileroVADBinaryCrossEntropyLoss(nn.Module):
    """Module wrapper for supervised Silero VAD fine-tuning."""

    def __init__(
        self,
        *,
        from_logits: bool = False,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        if not isinstance(from_logits, bool):
            raise TypeError("`from_logits` must be a boolean.")
        if reduction not in ("mean", "none", "sum"):
            raise ValueError("`reduction` must be 'mean', 'none', or 'sum'.")
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        weights: Tensor | None = None,
    ) -> Tensor:
        return silero_vad_binary_cross_entropy(
            predictions,
            targets,
            weights=weights,
            from_logits=self.from_logits,
            reduction=self.reduction,
        )


__all__ = [
    "Reduction",
    "SileroVADBinaryCrossEntropyLoss",
    "silero_vad_binary_cross_entropy",
]
