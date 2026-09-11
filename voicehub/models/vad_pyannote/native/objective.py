"""Native supervised objectives for PyanNet segmentation families."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from voicehub.models.vad_pyannote.native.configuration import PyanNetConfig


def _positive_scale(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"`{name}` must be finite and positive.")
    return result


def _validated_labels(
    labels: Tensor,
    *,
    batch_size: int,
    frame_count: int,
) -> Tensor:
    if not isinstance(labels, Tensor):
        raise TypeError("`labels` must be a PyTorch tensor.")
    if labels.shape[0] != batch_size or labels.shape[1] != frame_count:
        raise ValueError("`labels` batch and frame dimensions must match PyanNet output.")
    if not torch.isfinite(labels).all():
        raise ValueError("`labels` cannot contain NaN or infinite values.")
    return labels


def _weights(
    frame_weights: Tensor | None,
    *,
    reference: Tensor,
) -> Tensor | None:
    if frame_weights is None:
        return None
    if not isinstance(frame_weights, Tensor):
        raise TypeError("`frame_weights` must be a PyTorch tensor or None.")
    if frame_weights.ndim == 2:
        frame_weights = frame_weights.unsqueeze(-1)
    if frame_weights.shape != reference.shape[:2] + (1, ):
        raise ValueError("`frame_weights` must have shape [batch, frames] or "
                         "[batch, frames, 1].")
    frame_weights = frame_weights.to(
        device=reference.device,
        dtype=reference.dtype,
    )
    if not torch.isfinite(frame_weights).all() or (frame_weights < 0).any():
        raise ValueError("`frame_weights` must be finite and non-negative.")
    return frame_weights


def _weighted_mean(
    values: Tensor,
    weights: Tensor | None,
    *,
    allow_empty: bool = False,
) -> Tensor:
    if weights is None:
        return values.mean()
    expanded = weights.expand_as(values)
    denominator = expanded.sum()
    if denominator <= 0:
        if allow_empty:
            # Keep the zero connected to the prediction graph so an
            # all-silence batch remains valid for backward().
            return values.sum() * 0.0
        raise ValueError("At least one training frame must have positive weight.")
    return (values * expanded).sum() / denominator


def pyannet_loss(
    *,
    config: PyanNetConfig | Mapping[str, Any],
    logits: Tensor,
    probabilities: Tensor,
    labels: Tensor,
    frame_weights: Tensor | None = None,
    snr_loss_scale: float = 1.0,
    c50_loss_scale: float = 1.0,
) -> dict[str, Tensor]:
    """Compute the pinned upstream objective without its trainer framework."""
    resolved = PyanNetConfig.coerce(config)
    if not isinstance(logits, Tensor) or not isinstance(probabilities, Tensor):
        raise TypeError("`logits` and `probabilities` must be tensors.")
    if logits.shape != probabilities.shape or logits.ndim != 3:
        raise ValueError("PyanNet logits and probabilities must share "
                         "[batch, frames, outputs] shape.")
    labels = _validated_labels(
        labels,
        batch_size=logits.shape[0],
        frame_count=logits.shape[1],
    ).to(device=logits.device)
    weights = _weights(frame_weights, reference=logits)

    if resolved.is_powerset:
        if labels.ndim == 3:
            if labels.shape[-1] != resolved.output_size:
                raise ValueError("One-hot powerset labels have the wrong class dimension.")
            labels = labels.argmax(dim=-1)
        if labels.ndim != 2:
            raise ValueError("Powerset labels must contain class IDs with shape "
                             "[batch, frames].")
        if labels.dtype == torch.bool or labels.is_floating_point():
            raise TypeError("Powerset class labels must use an integer dtype.")
        losses = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.long().reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        if weights is not None:
            weights = weights.squeeze(-1)
        return {"loss": _weighted_mean(losses, weights)}

    labels = labels.to(dtype=probabilities.dtype)
    if labels.shape != probabilities.shape:
        raise ValueError("Multi-label and Brouhaha targets must match output shape.")
    if resolved.is_brouhaha:
        if ((labels[..., 0] < 0) | (labels[..., 0] > 1)).any():
            raise ValueError("Brouhaha VAD labels must be in [0, 1].")
        loss_vad = _weighted_mean(
            functional.binary_cross_entropy(
                probabilities[..., :1],
                labels[..., :1],
                reduction="none",
            ),
            weights,
        )
        speech_weights = labels[..., :1]
        if weights is not None:
            speech_weights = speech_weights * weights
        loss_snr = _weighted_mean(
            functional.mse_loss(
                probabilities[..., 1:2],
                labels[..., 1:2],
                reduction="none",
            ),
            speech_weights,
            allow_empty=True,
        )
        loss_c50 = _weighted_mean(
            functional.mse_loss(
                probabilities[..., 2:3],
                labels[..., 2:3],
                reduction="none",
            ),
            weights,
        )
        snr_scale = _positive_scale("snr_loss_scale", snr_loss_scale)
        c50_scale = _positive_scale("c50_loss_scale", c50_loss_scale)
        return {
            "loss": loss_vad + loss_snr / snr_scale + loss_c50 / c50_scale,
            "loss_vad": loss_vad,
            "loss_snr": loss_snr,
            "loss_c50": loss_c50,
        }

    if ((labels < 0) | (labels > 1)).any():
        raise ValueError("Segmentation labels must be probabilities in [0, 1].")
    losses = functional.binary_cross_entropy(
        probabilities,
        labels,
        reduction="none",
    )
    return {"loss": _weighted_mean(losses, weights)}


__all__ = ["pyannet_loss"]
