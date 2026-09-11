"""Pure-PyTorch monotonic alignment used by the StyleTTS 2 graph.

The API mirrors Resemble AI's MIT-licensed ``monotonic_align``
extension, whose Cython sources are preserved under
``source/third_party/monotonic_align``. This fallback keeps VoiceHub
wheels self-contained and avoids a Git/pip build.
"""

from __future__ import annotations

import torch

from voicehub.models.vits.native.alignment import maximum_path as _native_maximum_path


def maximum_path_c(paths, values, text_lengths, frame_lengths) -> None:
    """Fill tensor- or array-backed ``paths`` with alignments in place.

    ``torch.as_tensor`` shares storage with supported CPU array inputs.
    This preserves the historical Cython-shaped API without requiring
    NumPy or a compiled extension.
    """
    path_tensor = torch.as_tensor(paths)
    value_tensor = torch.as_tensor(values)
    text_length_tensor = torch.as_tensor(text_lengths)
    frame_length_tensor = torch.as_tensor(frame_lengths)
    if path_tensor.ndim != 3 or value_tensor.ndim != 3:
        raise ValueError("Alignment paths and values must have rank three.")
    if path_tensor.shape != value_tensor.shape:
        raise ValueError("Alignment paths and values must have equal shapes.")
    if text_length_tensor.shape != frame_length_tensor.shape:
        raise ValueError("Text and frame length tensors must have equal shapes.")
    if text_length_tensor.numel() != value_tensor.shape[0]:
        raise ValueError("Alignment lengths must contain one item per batch.")

    path_tensor.zero_()
    for batch_index in range(value_tensor.shape[0]):
        text_length = int(text_length_tensor[batch_index].item())
        frame_length = int(frame_length_tensor[batch_index].item())
        if text_length <= 0 or frame_length <= 0:
            continue
        if frame_length < text_length:
            raise ValueError("A monotonic alignment needs at least as many frames as "
                             "text tokens.")
        scores = value_tensor[
            batch_index,
            :text_length,
            :frame_length,
        ].transpose(0, 1).unsqueeze(0)
        mask = torch.ones_like(scores)
        aligned = _native_maximum_path(scores.float(), mask)
        path_tensor[
            batch_index,
            :text_length,
            :frame_length,
        ].copy_(aligned[0].transpose(0, 1).to(dtype=path_tensor.dtype))


def mask_from_lens(similarity, symbol_lens, mel_lens):
    """Create a broadcast alignment mask from token and frame lengths."""
    _, symbols, frames = similarity.size()
    symbol_mask = (torch.arange(symbols, device=symbol_lens.device)[None, :] < symbol_lens[:, None])
    frame_mask = (torch.arange(frames, device=mel_lens.device)[None, :] < mel_lens[:, None])
    return (symbol_mask.unsqueeze(2) * frame_mask.unsqueeze(1)).to(similarity)


def maximum_path(value, mask=None):
    """Return a maximum-score monotonic path as a Torch tensor."""
    if mask is None:
        mask = torch.ones_like(value)
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError("Alignment values must have shape [batch, text, frames].")
    if not isinstance(mask, torch.Tensor) or mask.shape != value.shape:
        raise ValueError("Alignment mask must match the value tensor.")
    scores = (value * mask).detach().transpose(1, 2)
    native_mask = mask.transpose(1, 2)
    return _native_maximum_path(
        scores,
        native_mask,
    ).transpose(1, 2).to(dtype=value.dtype)
