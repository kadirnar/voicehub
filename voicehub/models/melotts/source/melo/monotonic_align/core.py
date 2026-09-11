"""Compatibility entry point for the removed Numba alignment kernel."""

from __future__ import annotations

import torch

from voicehub.models.vits.native.alignment import maximum_path


def maximum_path_jit(
    paths: torch.Tensor,
    values: torch.Tensor,
    frame_lengths: torch.Tensor,
    text_lengths: torch.Tensor,
) -> None:
    """Fill ``paths`` in place while retaining the historical callable name."""
    paths = torch.as_tensor(paths)
    values = torch.as_tensor(values)
    frame_lengths = torch.as_tensor(frame_lengths)
    text_lengths = torch.as_tensor(text_lengths)
    if paths.shape != values.shape or paths.ndim != 3:
        raise ValueError("MeloTTS alignment paths and values must have equal rank-three shapes.")
    if frame_lengths.shape != text_lengths.shape or frame_lengths.numel() != paths.shape[0]:
        raise ValueError("MeloTTS alignment lengths must contain one value per batch item.")
    paths.zero_()
    for index in range(paths.shape[0]):
        frames = int(frame_lengths[index])
        text = int(text_lengths[index])
        if frames < text or text < 1:
            raise ValueError("MeloTTS monotonic alignment requires frames >= text >= 1.")
        scores = values[index, :frames, :text].unsqueeze(0)
        mask = torch.ones_like(scores)
        paths[index, :frames, :text].copy_(maximum_path(scores, mask)[0].to(dtype=paths.dtype))


__all__ = ["maximum_path_jit"]
