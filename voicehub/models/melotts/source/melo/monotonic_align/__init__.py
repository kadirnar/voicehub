"""Torch-only maximum monotonic alignment for the MeloTTS graph."""

from __future__ import annotations

import torch

from voicehub.models.vits.native.alignment import maximum_path as _maximum_path


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the maximum-score path in MeloTTS's ``[B, frames, text]`` layout."""
    if not isinstance(neg_cent, torch.Tensor) or neg_cent.ndim != 3:
        raise ValueError("MeloTTS alignment scores must have shape [batch, frames, text].")
    if not isinstance(mask, torch.Tensor) or mask.shape != neg_cent.shape:
        raise ValueError("MeloTTS alignment mask must match the score tensor.")
    if not neg_cent.is_floating_point() or not mask.is_floating_point():
        raise TypeError("MeloTTS alignment scores and masks must be floating-point tensors.")
    return _maximum_path(
        neg_cent.detach(),
        mask,
    ).to(dtype=neg_cent.dtype)


__all__ = ["maximum_path"]
