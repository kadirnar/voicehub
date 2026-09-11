"""PyTorch-only F5-TTS DiT building blocks.

The module hierarchy intentionally mirrors the released F5-TTS graph so
official ``ema_model.transformer.*`` tensors load without name
rewriting.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from voicehub.kernels import KernelBackend, fused_bias_gelu
from voicehub.kernels.diffusion import DiffusionModulationKernelOptimizable
from voicehub.neural.backends import FlashAttention4Policy, flash_attention4_or_sdpa


class SinusPositionEmbedding(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, value: torch.Tensor, scale: float = 1_000.0) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = math.log(10_000.0) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=value.device, dtype=torch.float32) * -exponent)
        angles = scale * value.unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class TimestepEmbedding(nn.Module):

    def __init__(self, dim: int, frequency_embedding_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = SinusPositionEmbedding(frequency_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        hidden = self.time_embed(timestep).to(timestep.dtype)
        return self.time_mlp(hidden)


class ConvPositionEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        kernel_size: int = 31,
        groups: int = 16,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("F5-TTS position-convolution kernel must be odd.")
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                dim,
                dim,
                kernel_size,
                groups=groups,
                padding=kernel_size // 2,
            ),
            nn.Mish(),
            nn.Conv1d(
                dim,
                dim,
                kernel_size,
                groups=groups,
                padding=kernel_size // 2,
            ),
            nn.Mish(),
        )
        self.layer_need_mask_idx = tuple(
            index for index, layer in enumerate(self.conv1d) if isinstance(layer, nn.Conv1d))

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        channel_states = hidden_states.transpose(1, 2)
        channel_mask = mask.unsqueeze(1) if mask is not None else None
        if channel_mask is not None:
            channel_states = channel_states.masked_fill(~channel_mask, 0.0)
        for index, layer in enumerate(self.conv1d):
            channel_states = layer(channel_states)
            if channel_mask is not None and index in self.layer_need_mask_idx:
                channel_states = channel_states.masked_fill(~channel_mask, 0.0)
        return channel_states.transpose(1, 2)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10_000.0,
    theta_rescale_factor: float = 1.0,
) -> torch.Tensor:
    theta *= theta_rescale_factor**(dim / (dim - 2))
    frequencies = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=torch.float32)[:dim // 2] / dim))
    positions = torch.arange(end, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


class GRN(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        response = torch.norm(hidden_states, p=2, dim=1, keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (hidden_states * normalized) + self.beta + hidden_states


class ConvNeXtV2Block(nn.Module):

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * 3
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            padding=padding,
            groups=dim,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.grn(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        return residual + hidden_states


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        source_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states.to(source_dtype) * self.weight


class AdaLayerNorm(DiffusionModulationKernelOptimizable, nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self._initialize_diffusion_kernel_backend()

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        modulation = self.linear(self.silu(embedding))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (modulation.chunk(6, dim=1))
        normalized = self._diffusion_modulate(
            self.norm(hidden_states),
            shift_msa[:, None],
            scale_msa[:, None],
        )
        return normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(DiffusionModulationKernelOptimizable, nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self._initialize_diffusion_kernel_backend()

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = self.linear(self.silu(embedding)).chunk(2, dim=1)
        return self._diffusion_modulate(
            self.norm(hidden_states),
            shift[:, None, :],
            scale[:, None, :],
        )


class FeedForward(nn.Module):

    def __init__(
        self,
        dim: int,
        *,
        mult: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.kernel_backend = KernelBackend.TORCH
        self.ff = nn.Sequential(
            nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.GELU(approximate="tanh"),
            ),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def set_kernel_backend(self, backend: KernelBackend | str) -> None:
        """Select a registered fused bias/GELU implementation."""
        self.kernel_backend = KernelBackend.coerce(backend)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projection = self.ff[0][0]
        if not isinstance(projection, nn.Linear) or projection.bias is None:
            raise RuntimeError("F5 feed-forward input projection must be a biased Linear.")
        hidden_states = F.linear(
            hidden_states,
            projection.weight,
            bias=None,
        )
        hidden_states = fused_bias_gelu(
            hidden_states,
            projection.bias,
            backend=self.kernel_backend,
        )
        return self.ff[2](self.ff[1](hidden_states))


def _rotate_half_interleaved(hidden_states: torch.Tensor) -> torch.Tensor:
    pairs = hidden_states.reshape(*hidden_states.shape[:-1], -1, 2)
    first, second = pairs.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def apply_rotary_position_embedding(
    hidden_states: torch.Tensor,
    frequencies: torch.Tensor,
    scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Match x-transformers' adjacent-pair rotary convention."""
    rotary_dim = frequencies.shape[-1]
    if rotary_dim > hidden_states.shape[-1]:
        raise ValueError("Rotary dimension exceeds the attention head dimension.")
    rotary = hidden_states[..., :rotary_dim]
    remainder = hidden_states[..., rotary_dim:]
    frequencies = frequencies[-rotary.shape[-2]:].to(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    while frequencies.ndim < rotary.ndim:
        frequencies = frequencies.unsqueeze(0)
    rotated = (
        rotary * frequencies.cos() * scale + _rotate_half_interleaved(rotary) * frequencies.sin() * scale)
    return torch.cat((rotated, remainder), dim=-1).to(hidden_states.dtype)


class RotaryEmbedding(nn.Module):
    """Minimal state-compatible subset of x-transformers RotaryEmbedding."""

    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward_from_seq_len(
        self,
        sequence_length: int,
    ) -> tuple[torch.Tensor, float]:
        positions = torch.arange(
            sequence_length,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        angles = torch.einsum("i,j->ij", positions, self.inv_freq)
        frequencies = torch.stack((angles, angles), dim=-1).flatten(-2)
        return frequencies, 1.0


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        dropout: float,
        qk_norm: str | None,
        pe_attn_head: int | None,
        attn_mask_enabled: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.inner_dim = heads * dim_head
        self.dropout = dropout
        self.pe_attn_head = pe_attn_head
        self.attn_mask_enabled = attn_mask_enabled
        self.flash_attention4_policy = FlashAttention4Policy.DISABLED
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        if qk_norm == "rms_norm":
            self.q_norm: nn.Module | None = RMSNorm(dim_head, eps=1e-6)
            self.k_norm: nn.Module | None = RMSNorm(dim_head, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None
        self.to_out = nn.ModuleList((nn.Linear(self.inner_dim, dim), nn.Dropout(dropout)))

    def set_flash_attention4_policy(
        self,
        policy: FlashAttention4Policy | str,
    ) -> None:
        """Select disabled, automatic, or required FlashAttention-4."""
        self.flash_attention4_policy = FlashAttention4Policy.coerce(policy)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, float] | None = None,
    ) -> torch.Tensor:
        batch, sequence_length, _ = hidden_states.shape
        head_dim = self.inner_dim // self.heads

        def project(layer: nn.Linear) -> torch.Tensor:
            return (layer(hidden_states).view(batch, sequence_length, self.heads, head_dim).transpose(1, 2))

        query = project(self.to_q)
        key = project(self.to_k)
        value = project(self.to_v)
        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)
        if rope is not None:
            frequencies, xpos_scale = rope
            positional_heads = self.heads if self.pe_attn_head is None else self.pe_attn_head
            query_rotary = apply_rotary_position_embedding(
                query[:, :positional_heads],
                frequencies,
                xpos_scale,
            )
            key_rotary = apply_rotary_position_embedding(
                key[:, :positional_heads],
                frequencies,
                xpos_scale**-1.0,
            )
            if positional_heads == self.heads:
                query, key = query_rotary, key_rotary
            else:
                query = torch.cat((query_rotary, query[:, positional_heads:]), dim=1)
                key = torch.cat((key_rotary, key[:, positional_heads:]), dim=1)

        attention_mask = None
        if self.attn_mask_enabled and mask is not None:
            attention_mask = mask[:, None, None, :].expand(
                batch,
                self.heads,
                sequence_length,
                sequence_length,
            )
        attended = flash_attention4_or_sdpa(
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            policy=self.flash_attention4_policy,
        )
        attended = attended.transpose(1, 2).reshape(
            batch,
            sequence_length,
            self.inner_dim,
        )
        attended = self.to_out[1](self.to_out[0](attended))
        if mask is not None:
            attended = attended.masked_fill(~mask.unsqueeze(-1), 0.0)
        return attended


class DiTBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        ff_mult: float,
        dropout: float,
        qk_norm: str | None,
        pe_attn_head: int | None,
        attn_mask_enabled: bool,
    ) -> None:
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
            pe_attn_head=pe_attn_head,
            attn_mask_enabled=attn_mask_enabled,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_embedding: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, float] | None = None,
    ) -> torch.Tensor:
        normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(
            hidden_states,
            time_embedding,
        )
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * self.attn(
            normalized,
            mask=mask,
            rope=rope,
        )
        normalized = self.attn_norm._diffusion_modulate(
            self.ff_norm(hidden_states),
            shift_mlp[:, None],
            scale_mlp[:, None],
        )
        return hidden_states + gate_mlp.unsqueeze(1) * self.ff(normalized)


class TextEmbedding(nn.Module):

    def __init__(
        self,
        text_num_embeds: int,
        text_dim: int,
        *,
        mask_padding: bool,
        average_upsampling: bool,
        conv_layers: int,
        conv_mult: int,
    ) -> None:
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.mask_padding = mask_padding
        self.average_upsampling = average_upsampling
        if average_upsampling and not mask_padding:
            raise ValueError("Average text upsampling requires padding masks.")
        self.extra_modeling = conv_layers > 0
        if self.extra_modeling:
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(text_dim, 8_192),
                persistent=False,
            )
            self.text_blocks = nn.Sequential(
                *(ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)))

    @staticmethod
    def _average_upsample(
        hidden_states: torch.Tensor,
        text_mask: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for batch_index in range(hidden_states.shape[0]):
            valid = torch.where(text_mask[batch_index])[0]
            text_length = int(valid.numel())
            audio_length = int(target_lengths[batch_index].item())
            if text_length == 0 or audio_length <= 0:
                continue
            base_repeat, remainder = divmod(audio_length, text_length)
            repeats = torch.full(
                (text_length, ),
                base_repeat,
                device=hidden_states.device,
                dtype=torch.long,
            )
            if remainder:
                repeats[-remainder:] += 1
            indices = torch.repeat_interleave(
                torch.arange(text_length, device=hidden_states.device),
                repeats,
            )[:audio_length]
            output[batch_index, :audio_length] = hidden_states[
                batch_index,
                valid[indices],
            ]
        return output

    def forward(
        self,
        token_ids: torch.Tensor,
        sequence_length: int | torch.Tensor,
        *,
        drop_text: bool = False,
    ) -> torch.Tensor:
        token_ids = token_ids + 1
        valid_positions = None
        if isinstance(sequence_length, torch.Tensor):
            target_lengths = sequence_length.to(
                device=token_ids.device,
                dtype=torch.long,
            )
            maximum_length = int(target_lengths.max().item())
        else:
            maximum_length = int(sequence_length)
            target_lengths = torch.full(
                (token_ids.shape[0], ),
                maximum_length,
                device=token_ids.device,
                dtype=torch.long,
            )
        token_ids = token_ids[:, :maximum_length]
        token_ids = F.pad(
            token_ids,
            (0, maximum_length - token_ids.shape[1]),
            value=0,
        )
        if isinstance(sequence_length, torch.Tensor):
            positions = torch.arange(maximum_length, device=token_ids.device)
            valid_positions = positions.unsqueeze(0) < target_lengths.unsqueeze(1)
            token_ids = token_ids.masked_fill(~valid_positions, 0)
        padding_mask = token_ids == 0
        if drop_text:
            token_ids = torch.zeros_like(token_ids)
        hidden_states = self.text_embed(token_ids)
        if valid_positions is not None:
            hidden_states = hidden_states.masked_fill(
                ~valid_positions.unsqueeze(-1),
                0.0,
            )
        if self.extra_modeling:
            positions = self.freqs_cis[:maximum_length]
            if valid_positions is not None:
                positions = (positions.unsqueeze(0) * valid_positions.unsqueeze(-1).to(positions.dtype))
            hidden_states = hidden_states + positions
            if self.mask_padding:
                mask = padding_mask.unsqueeze(-1)
                hidden_states = hidden_states.masked_fill(mask, 0.0)
                for block in self.text_blocks:
                    hidden_states = block(hidden_states).masked_fill(mask, 0.0)
            else:
                hidden_states = self.text_blocks(hidden_states)
        if self.average_upsampling:
            hidden_states = self._average_upsample(
                hidden_states,
                ~padding_mask,
                target_lengths,
            )
        return hidden_states


class InputEmbedding(nn.Module):

    def __init__(self, mel_dim: int, text_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        text_embedding: torch.Tensor,
        *,
        drop_audio_cond: bool = False,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if drop_audio_cond:
            conditioning = torch.zeros_like(conditioning)
        projected = self.proj(torch.cat((hidden_states, conditioning, text_embedding), dim=-1))
        return self.conv_pos_embed(projected, mask=audio_mask) + projected


__all__ = [
    "AdaLayerNorm",
    "AdaLayerNormFinal",
    "Attention",
    "ConvNeXtV2Block",
    "ConvPositionEmbedding",
    "DiTBlock",
    "FeedForward",
    "GRN",
    "InputEmbedding",
    "RMSNorm",
    "RotaryEmbedding",
    "SinusPositionEmbedding",
    "TextEmbedding",
    "TimestepEmbedding",
    "apply_rotary_position_embedding",
    "precompute_freqs_cis",
]
