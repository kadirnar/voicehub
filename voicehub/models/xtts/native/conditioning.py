"""XTTS v2 conditioning encoder and Perceiver resampler."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GroupNorm32(nn.GroupNorm):

    def forward(self, value: Tensor) -> Tensor:
        return super().forward(value.float()).to(value.dtype)


def _normalization(channels: int) -> GroupNorm32:
    groups = 8 if channels <= 16 else 16 if channels <= 64 else 32
    while channels % groups:
        groups //= 2
    if groups <= 2:
        raise ValueError("XTTS conditioning width is too small for group normalization.")
    return GroupNorm32(groups, channels)


class QKVAttention(nn.Module):

    def __init__(self, heads: int) -> None:
        super().__init__()
        self.n_heads = heads

    def forward(self, qkv: Tensor) -> Tensor:
        batch, width, length = qkv.shape
        channels = width // (3 * self.n_heads)
        query, key, value = qkv.reshape(
            batch * self.n_heads,
            channels * 3,
            length,
        ).split(
            channels, dim=1)
        scale = 1.0 / math.sqrt(math.sqrt(channels))
        weight = torch.einsum("bct,bcs->bts", query * scale, key * scale)
        weight = F.softmax(weight.float(), dim=-1).to(weight.dtype)
        output = torch.einsum("bts,bcs->bct", weight, value)
        return output.reshape(batch, -1, length)


class AttentionBlock(nn.Module):

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.norm = _normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(heads)
        self.x_proj = nn.Identity()
        self.proj_out = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, value: Tensor) -> Tensor:
        normalized = self.norm(value)
        return self.x_proj(normalized) + self.proj_out(self.attention(self.qkv(normalized)), )


class ConditioningEncoder(nn.Module):

    def __init__(
        self,
        spec_dim: int,
        embedding_dim: int,
        attn_blocks: int = 6,
        num_attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        self.attn = nn.Sequential(
            *[AttentionBlock(embedding_dim, num_attn_heads) for _ in range(attn_blocks)])
        self.dim = embedding_dim

    def forward(self, value: Tensor) -> Tensor:
        return self.attn(self.init(value))


class RMSNorm(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, value: Tensor) -> Tensor:
        return F.normalize(value, dim=-1) * self.scale * self.gamma


class GEGLU(nn.Module):

    def forward(self, value: Tensor) -> Tensor:
        value, gate = value.chunk(2, dim=-1)
        return F.gelu(gate) * value


class PerceiverAttention(nn.Module):

    def __init__(self, dim: int, *, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.heads = heads
        inner = dim_head * heads
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def forward(self, latents: Tensor, context: Tensor) -> Tensor:
        context = torch.cat((latents, context), dim=-2)
        query = self._heads(self.to_q(latents))
        key, value = (self._heads(item) for item in self.to_kv(context).chunk(2, dim=-1))
        output = F.scaled_dot_product_attention(query, key, value)
        output = output.transpose(1, 2).contiguous().flatten(2)
        return self.to_out(output)

    def _heads(self, value: Tensor) -> Tensor:
        return value.view(*value.shape[:-1], self.heads, -1).transpose(1, 2)


class PerceiverFeedForward(nn.Sequential):

    def __init__(self, dim: int, mult: int = 4) -> None:
        inner = int(dim * mult * 2 / 3)
        super().__init__(
            nn.Linear(dim, inner * 2),
            GEGLU(),
            nn.Linear(inner, dim),
        )


class PerceiverResampler(nn.Module):

    def __init__(
        self,
        *,
        dim: int,
        depth: int = 2,
        num_latents: int = 32,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.proj_context = nn.Identity()
        self.latents = nn.Parameter(torch.empty(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim, dim_head=dim_head, heads=heads),
                PerceiverFeedForward(dim, ff_mult),
            ]) for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, context: Tensor) -> Tensor:
        context = self.proj_context(context)
        latents = self.latents.unsqueeze(0).expand(context.shape[0], -1, -1)
        for attention, feed_forward in self.layers:
            latents = attention(latents, context) + latents
            latents = feed_forward(latents) + latents
        return self.norm(latents)


__all__ = ["ConditioningEncoder", "PerceiverResampler"]
