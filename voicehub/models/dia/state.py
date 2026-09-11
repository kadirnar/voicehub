"""Small compatibility state objects for the native Dia implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from voicehub.models.dia.native.modeling import DiaConditionalGenerationOutput, DiaEncoderOutput


def create_attn_mask(
    query_padding_mask: Tensor,
    key_padding_mask: Tensor,
    device: torch.device,
    is_causal: bool = False,
) -> Tensor:
    """Return the boolean padding mask used by legacy integrations."""
    if query_padding_mask.ndim != 2 or key_padding_mask.ndim != 2:
        raise ValueError("Dia padding masks must have shape [batch, sequence].")
    query = query_padding_mask.to(device=device, dtype=torch.bool)[:, :, None]
    key = key_padding_mask.to(device=device, dtype=torch.bool)[:, None, :]
    mask = (query & key) | (~query & ~key)
    if is_causal:
        causal = torch.ones(
            query.shape[1],
            key.shape[2],
            dtype=torch.bool,
            device=device,
        ).tril()
        mask &= causal
    return mask[:, None]


class KVCache(torch.nn.Module):
    """Generic key/value cache available to optional optimization
    strategies."""

    def __init__(self, key: Tensor, value: Tensor) -> None:
        super().__init__()
        if key.shape != value.shape:
            raise ValueError("Dia cache key and value shapes must match.")
        self.register_buffer("k", key)
        self.register_buffer("v", value)

    @classmethod
    def from_kv(cls, key: Tensor, value: Tensor) -> KVCache:
        return cls(key, value)

    def update(self, key: Tensor, value: Tensor, position: int) -> tuple[Tensor, Tensor]:
        if key.shape != value.shape:
            raise ValueError("Dia cache key and value shapes must match.")
        width = key.shape[-2]
        end = position + width
        if position < 0 or end > self.k.shape[-2]:
            raise IndexError("Dia cache update exceeds its allocated sequence.")
        self.k[..., position:end, :] = key
        self.v[..., position:end, :] = value
        return self.k, self.v


@dataclass
class DecoderOutput:
    generated_tokens: Tensor
    prefill_steps: list[int]


EncoderInferenceState = DiaEncoderOutput
DecoderInferenceState = DiaConditionalGenerationOutput

__all__ = [
    "DecoderInferenceState",
    "DecoderOutput",
    "EncoderInferenceState",
    "KVCache",
    "create_attn_mask",
]
