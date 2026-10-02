"""Rotary position embeddings shared by Llama/Qwen-style speech models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


def rotate_half(inputs: Tensor) -> Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    """Compute float32 rotary cosines/sines for arbitrary position IDs."""

    def __init__(
        self,
        dimension: int,
        *,
        base: float = 10_000.0,
        scaling: Mapping[str, Any] | None = None,
        device=None,
    ) -> None:
        super().__init__()
        if (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0 or dimension % 2):
            raise ValueError("Rotary embedding dimension must be positive and even.")
        if base <= 1.0:
            raise ValueError("Rotary embedding base must be greater than one.")
        self.dimension = dimension
        self.base = float(base)
        # Always evaluate the frequencies on the CPU (as the Hugging Face
        # reference implementations do) and move them afterwards: CUDA `pow`
        # rounds some of these float32 values differently, which shifts RoPE
        # angles at long positions and changes bfloat16 outputs.
        inverse_frequency = 1.0 / (
            self.base**(torch.arange(
                0,
                dimension,
                2,
                dtype=torch.float32,
                device="cpu",
            ) / dimension))
        if scaling is not None:
            rope_type = scaling.get("rope_type", scaling.get("type", "default"))
            if rope_type == "llama3":
                factor = float(scaling["factor"])
                low_frequency_factor = float(scaling["low_freq_factor"])
                high_frequency_factor = float(scaling["high_freq_factor"])
                original_context = int(scaling["original_max_position_embeddings"])
                wavelength = 2 * math.pi / inverse_frequency
                low_wavelength = original_context / low_frequency_factor
                high_wavelength = original_context / high_frequency_factor
                scaled_frequency = torch.where(
                    wavelength > low_wavelength,
                    inverse_frequency / factor,
                    inverse_frequency,
                )
                smooth = (original_context / wavelength -
                          low_frequency_factor) / (high_frequency_factor - low_frequency_factor)
                interpolated = ((1 - smooth) * scaled_frequency / factor + smooth * scaled_frequency)
                medium = (~(wavelength < high_wavelength) & ~(wavelength > low_wavelength))
                inverse_frequency = torch.where(
                    medium,
                    interpolated,
                    scaled_frequency,
                )
        self.register_buffer(
            "inverse_frequency",
            inverse_frequency.to(device=device),
            persistent=False,
        )

    def forward(
        self,
        position_ids: Tensor,
        *,
        dtype: Any | None = None,
    ) -> tuple[Tensor, Tensor]:
        if (not isinstance(position_ids, Tensor) or position_ids.ndim not in (1, 2) or
                position_ids.dtype == torch.bool or position_ids.is_floating_point() or
                position_ids.is_complex()):
            raise TypeError("Rotary position IDs must be a rank-one or rank-two integer tensor.")
        if (position_ids < 0).any():
            raise ValueError("Rotary position IDs cannot be negative.")
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        angles = position_ids.float().unsqueeze(-1) * self.inverse_frequency
        embeddings = torch.cat((angles, angles), dim=-1)
        output_dtype = dtype or self.inverse_frequency.dtype
        return embeddings.cos().to(output_dtype), embeddings.sin().to(output_dtype)


def apply_rotary_embedding(
    query: Tensor,
    key: Tensor,
    cosine: Tensor,
    sine: Tensor,
    *,
    rotary_dimension: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Rotate the configured prefix of query/key head dimensions."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("Rotary query/key tensors must have rank four.")
    dimension = query.shape[-1] if rotary_dimension is None else rotary_dimension
    if (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0 or dimension % 2 or
            dimension > min(query.shape[-1], key.shape[-1])):
        raise ValueError("Invalid rotary dimension for query/key head sizes.")
    if cosine.ndim != 3 or sine.shape != cosine.shape:
        raise ValueError("Rotary cosine/sine tensors must have shape [batch, time, dimension].")
    cosine = cosine[..., :dimension].unsqueeze(1)
    sine = sine[..., :dimension].unsqueeze(1)

    def rotate(inputs: Tensor) -> Tensor:
        rotated = (inputs[..., :dimension] * cosine + rotate_half(inputs[..., :dimension]) * sine)
        return torch.cat((rotated, inputs[..., dimension:]), dim=-1)

    return rotate(query), rotate(key)


__all__ = [
    "RotaryEmbedding",
    "apply_rotary_embedding",
    "rotate_half",
]
