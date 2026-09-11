"""Checkpoint-compatible objectives for native FSMN VAD fine-tuning."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional


def fsmn_vad_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    silence_pdf_ids: tuple[int, ...] = (0, ),
    label_mask: Tensor | None = None,
    target_kind: str = "auto",
) -> tuple[Tensor, str]:
    """Compute PDF cross-entropy or grouped speech/silence NLL.

    Integer targets containing values outside ``{0, 1}`` select the
    exact 248-PDF objective. Binary integer, boolean, and floating-point
    targets optimize the released decoder's grouped silence/speech
    probability.
    """
    if target_kind not in {"auto", "binary", "pdf"}:
        raise ValueError("`target_kind` must be 'auto', 'binary', or 'pdf'.")
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError("`logits` must have shape [batch, frames, pdfs].")
    if (not isinstance(silence_pdf_ids, tuple) or not silence_pdf_ids or
            any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item >= logits.shape[-1]
                for item in silence_pdf_ids)):
        raise ValueError("`silence_pdf_ids` must identify valid output classes.")
    if not isinstance(labels, Tensor):
        try:
            labels = torch.as_tensor(labels, device=logits.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise TypeError("`labels` must be tensor-like.") from error
    labels = labels.to(device=logits.device)
    if labels.ndim == 3 and labels.shape[-1] == 1:
        labels = labels.squeeze(-1)
    if labels.ndim != 2 or tuple(labels.shape) != tuple(logits.shape[:2]):
        raise ValueError("`labels` must have shape [batch, frames] matching `logits`.")
    if label_mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        if not isinstance(label_mask, Tensor):
            try:
                label_mask = torch.as_tensor(
                    label_mask,
                    device=logits.device,
                )
            except (TypeError, ValueError, RuntimeError) as error:
                raise TypeError("`label_mask` must be tensor-like.") from error
        if tuple(label_mask.shape) != tuple(labels.shape):
            raise ValueError("`label_mask` must match `labels`.")
        mask = label_mask.to(device=logits.device, dtype=torch.bool)
    if not mask.any():
        return logits.sum() * 0.0, "empty"

    valid_labels = labels[mask]
    inferred_pdf_targets = (
        not labels.is_floating_point() and labels.dtype != torch.bool and bool(
            ((valid_labels < 0) | (valid_labels > 1)).any()))
    uses_pdf_targets = (target_kind == "pdf" or (target_kind == "auto" and inferred_pdf_targets))
    if uses_pdf_targets:
        if labels.is_floating_point() or labels.dtype == torch.bool:
            raise TypeError("PDF targets must use an integer tensor dtype.")
        targets = labels.to(dtype=torch.long)
        if ((targets[mask] < 0) | (targets[mask] >= logits.shape[-1])).any():
            raise ValueError(f"PDF labels must be in [0, {logits.shape[-1] - 1}].")
        losses = functional.cross_entropy(
            logits.transpose(1, 2),
            targets,
            reduction="none",
        )
        return losses[mask].mean(), "pdf-cross-entropy"

    binary = labels.to(dtype=logits.dtype)
    if not torch.isfinite(binary[mask]).all():
        raise ValueError("Binary VAD labels must be finite.")
    if ((binary[mask] < 0.0) | (binary[mask] > 1.0)).any():
        raise ValueError("Binary VAD labels must be in [0, 1].")
    probabilities = logits.softmax(dim=-1)
    silence = probabilities[..., list(silence_pdf_ids)].sum(dim=-1)
    speech = (1.0 - silence).clamp_min(torch.finfo(logits.dtype).eps)
    silence = silence.clamp_min(torch.finfo(logits.dtype).eps)
    losses = -(binary * speech.log() + (1.0 - binary) * silence.log())
    return losses[mask].mean(), "grouped-binary-nll"


__all__ = ["fsmn_vad_loss"]
