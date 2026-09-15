# SPDX-License-Identifier: Apache-2.0

# This file contains portions adapted from:
#   • Descript Audio Codec (DAC) — MIT License (full text appended below)
#   • Fish-Speech S1 DAC Autoencoder — reference implementation (Apache-2.0 / CC-BY-NC),
#     rewritten here in a single-file Torch module for interoperability and transparency.
#
# OVERALL LICENSE (this file): Apache-2.0, except where explicitly marked:
#     # SPDX-License-Identifier: MIT
# Keep these notices and the embedded MIT text if you redistribute this file.

# NOTE
# Self-contained autoencoder implementation of Fish-S1-DAC (inlining DAC code to avoid dependencies).
# Code in this module has been largely copy-and-pasted from the Fish-S1-DAC and DAC repositories,
# and refactored with help from ChatGPT/Claude (these models also helped with licensing).
# Thus, it differs stylistically from the rest of the codebase (and is likely internally inconsistent as well).

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from voicehub.kernels.codecs import CodecSnakeKernelOptimizable

# --------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------


def find_multiple(n: int, k: int) -> int:
    return n if n % k == 0 else n + k - (n % k)


def unpad1d(x: Tensor, paddings: tuple[int, int]) -> Tensor:
    """Remove padding from x, handling properly zero padding.

    Only for 1d!
    """
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def get_extra_padding_for_conv1d(x: Tensor, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    """See pad_for_conv1d; enough right pad so striding evenly covers
    length."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(
    x: Tensor,
    paddings: tuple[int, int],
    mode: str = "zeros",
    value: float = 0.0,
) -> Tensor:
    """Reflect‑safe 1D pad: if reflect would underflow on small inputs, insert
    temporary right zero-pad before reflecting."""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, (padding_left, padding_right), mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, (padding_left, padding_right), mode, value)


# --------------------------------------------------------------------
# DAC Layers (adapted) — MIT
# Original: https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/layers.py
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


@torch.jit.script
def snake(x: Tensor, alpha: Tensor) -> Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(CodecSnakeKernelOptimizable, nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))
        self._initialize_codec_kernel_backend()

    def forward(self, x: Tensor) -> Tensor:
        return self._codec_snake(x, self.alpha)


# --------------------------------------------------------------------
# DAC Vector Quantize (adapted) — MIT
# Original: https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/quantize.py
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------


class VectorQuantize(nn.Module):
    """VQ with factorized, l2-normalized codes (ViT‑VQGAN style).

    I/O in (B, D, T).
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z: Tensor):
        z_e = self.in_proj(z)  # (B, D, T)
        z_q, indices = self.decode_latents(z_e)
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
        z_q = z_e + (z_q - z_e).detach()  # straight‑through
        z_q = self.out_proj(z_q)
        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id: Tensor) -> Tensor:
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: Tensor) -> Tensor:
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, _, time_steps = latents.shape
        encodings = latents.permute(0, 2, 1).reshape(
            batch_size * time_steps,
            self.codebook_dim,
        )
        codebook = self.codebook.weight
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)
        dist = (
            encodings.pow(2).sum(1, keepdim=True) - 2 * encodings @ codebook.t() +
            codebook.pow(2).sum(1, keepdim=True).t())
        indices = (-dist).max(1)[1].reshape(batch_size, time_steps)
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """SoundStream-style residual VQ stack."""

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int | list[int] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [VectorQuantize(input_dim, codebook_size, codebook_dim[i]) for i in range(n_codebooks)])
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z: Tensor, n_quantizers: int | None = None):
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0], )) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0], ))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commit_i, codebk_i, indices_i, z_e_i = quantizer(residual)

            mask = (torch.full((z.shape[0], ), fill_value=i, device=z.device) < n_quantizers)
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commitment_loss += (commit_i * mask).mean()
            codebook_loss += (codebk_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)
            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z_q = 0
        z_p = []
        codes = []
        dimensions = [0]
        for quantizer in self.quantizers:
            next_dimension = dimensions[-1] + quantizer.codebook_dim
            if next_dimension > latents.shape[1]:
                break
            dimensions.append(next_dimension)
        n_codebooks = len(dimensions) - 1
        if n_codebooks == 0:
            raise ValueError("Latent tensor does not contain one complete codebook "
                             "projection.")
        for i in range(n_codebooks):
            j, k = dimensions[i], dimensions[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)
            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


# --------------------------------------------------------------------
# S1 DAC rvq
# --------------------------------------------------------------------


@dataclass
class VQResult:
    z: Tensor
    codes: Tensor
    latents: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor
    semantic_distill_z: Tensor | None = None


class CausalConvNet(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        groups=1,
        padding=None,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def forward(self, x: Tensor) -> Tensor:
        pad = self.padding
        extra = get_extra_padding_for_conv1d(x, self.kernel_size, self.stride, pad)
        x = pad1d(x, (pad, extra), mode="constant", value=0)
        return self.conv(x).contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


class CausalTransConvNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, padding=None):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = unpad1d(x, (padding_left, padding_right))
        return x.contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


def CausalWNConv1d(*args, **kwargs):
    return CausalConvNet(*args, **kwargs).weight_norm()


def CausalWNConvTranspose1d(*args, **kwargs):
    return CausalTransConvNet(*args, **kwargs).weight_norm()


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block (1D).

    DwConv -> (N, C, L) → (N, L, C) -> LN -> Linear -> GELU -> Linear ->
    (N, C, L) with residual
    """

    def __init__(
        self,
        dim: int,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()
        convnet_type = CausalConvNet
        self.dwconv = convnet_type(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
            dilation=dilation,
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value *
                         torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None)

    def forward(self, x: Tensor, apply_residual: bool = True) -> Tensor:
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        if apply_residual:
            x = inp + x
        return x


class DownsampleResidualVectorQuantize(nn.Module):

    def __init__(
        self,
        input_dim: int = 1024,
        n_codebooks: int = 9,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 4096,
        downsample_factor: tuple[int, ...] = (2, 2),
        downsample_dims: tuple[int, ...] | None = None,
        pre_module: nn.Module | None = None,
        post_module: nn.Module | None = None,
        semantic_predictor_module: nn.Module | None = None,
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = tuple(input_dim for _ in range(len(downsample_factor)))

        all_dims = (input_dim, ) + tuple(downsample_dims)

        self.semantic_quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=1,
            codebook_size=semantic_codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=0.0,
        )

        self.quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        convnet_type = CausalConvNet
        transconvnet_type = CausalTransConvNet

        self.downsample = nn.Sequential(
            *[
                nn.Sequential(
                    convnet_type(all_dims[idx], all_dims[idx + 1], kernel_size=factor, stride=factor),
                    ConvNeXtBlock(dim=all_dims[idx + 1]),
                ) for idx, factor in enumerate(downsample_factor)
            ])

        self.upsample = nn.Sequential(
            *[
                nn.Sequential(
                    transconvnet_type(all_dims[idx + 1], all_dims[idx], kernel_size=factor, stride=factor),
                    ConvNeXtBlock(dim=all_dims[idx]),
                ) for idx, factor in reversed(list(enumerate(downsample_factor)))
            ])

        self.apply(self._init_weights)
        self.pre_module = pre_module if pre_module is not None else nn.Identity()
        self.post_module = post_module if post_module is not None else nn.Identity()
        self.semantic_predictor_module = (
            semantic_predictor_module if semantic_predictor_module is not None else nn.Identity())

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
            self, z: Tensor, n_quantizers: int | None = None, semantic_len: Tensor | None = None, **kwargs):
        # z: (B, D, T)
        original_shape = z.shape
        if semantic_len is None:
            semantic_len = torch.LongTensor([z.shape[-1]])

        z = self.downsample(z)
        z = self.pre_module(
            z)  # (B, D, T) or (B, T, D) depending on module; original uses channels-first in/out

        semantic_z, semantic_codes, semantic_latents, semantic_commitment_loss, semantic_codebook_loss =\
            self.semantic_quantizer(z)
        residual_z = z - semantic_z
        residual_z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            residual_z, n_quantizers=n_quantizers)
        z = semantic_z + residual_z
        commitment_loss = commitment_loss + semantic_commitment_loss
        codebook_loss = codebook_loss + semantic_codebook_loss
        codes = torch.cat([semantic_codes, codes], dim=1)
        latents = torch.cat([semantic_latents, latents], dim=1)
        z = self.post_module(z)
        z = self.upsample(z)

        # Pad or crop z to match original shape (time dimension)
        diff = original_shape[-1] - z.shape[-1]
        right = 0
        left = abs(diff) - right
        if diff > 0:
            z = F.pad(z, (left, right))
        elif diff < 0:
            z = z[..., left:]

        return VQResult(
            z=z,
            codes=codes,
            latents=latents,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
        )

    def decode(self, indices: Tensor) -> Tensor:
        new_indices = torch.zeros_like(indices)
        new_indices[:, 0] = torch.clamp(indices[:, 0], max=self.semantic_quantizer.codebook_size - 1)
        new_indices[:, 1:] = torch.clamp(indices[:, 1:], max=self.quantizer.codebook_size - 1)

        z_q_semantic = self.semantic_quantizer.from_codes(new_indices[:, :1])[0]
        z_q_residual = self.quantizer.from_codes(new_indices[:, 1:])[0]
        z_q = z_q_semantic + z_q_residual
        z_q = self.post_module(z_q)
        z_q = self.upsample(z_q)
        return z_q


# --------------------------------------------------------------------
# Transformer stack
# --------------------------------------------------------------------


@dataclass
class ModelArgs:
    block_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    dim: int = 512
    intermediate_size: int = 1536
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1
    channels_first: bool = True  # to be compatible with conv1d input/output
    pos_embed_type: str = "rope"  # "rope" or "conformer"
    max_relative_position: int = 128

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        assert self.pos_embed_type in ["rope", "conformer"]


class KVCache(nn.Module):

    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return (
            k_out[:, :, :input_pos.max() + 1, :],
            v_out[:, :, :input_pos.max() + 1, :],
        )

    def clear_cache(self, prompt_len: int):
        self.k_cache[:, :, prompt_len:, :].fill_(0)
        self.v_cache[:, :, prompt_len:, :].fill_(0)


class Transformer(nn.Module):

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if config.pos_embed_type == "rope":
            freqs_cis = precompute_freqs_cis(
                self.config.block_size, self.config.head_dim, self.config.rope_base)
            self.register_buffer("freqs_cis", freqs_cis)
        else:
            self.register_buffer("freqs_cis", None)

        causal_mask = torch.tril(torch.ones(self.config.block_size, self.config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask)

        self.max_batch_size = -1
        self.max_seq_length = -1
        self.use_kv_cache = False

    def setup_caches(self, max_batch_size, max_seq_length):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.norm.weight.dtype
        device = self.norm.weight.device

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype).to(device)

        self.use_kv_cache = True

    def forward(self, x: Tensor, input_pos: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        if self.config.pos_embed_type == "rope":
            assert self.freqs_cis is not None
            freqs_cis = self.freqs_cis[input_pos]
        else:
            freqs_cis = None

        if mask is None:
            if not self.training and self.use_kv_cache:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., :input_pos.max() + 1]
            else:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., input_pos]

        for layer in self.layers:
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_layer_scale = LayerScale(config.dim, inplace=True)
        self.ffn_layer_scale = LayerScale(config.dim, inplace=True)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention_layer_scale(self.attention(self.attention_norm(x), freqs_cis, mask, input_pos))
        out = h + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(h)))
        return out


class Attention(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.attn_dropout_rate = config.attn_dropout_rate
        self.pos_embed_type = config.pos_embed_type

        if self.pos_embed_type == "conformer":
            self.max_relative_position = config.max_relative_position
            num_pos_embeddings = 2 * config.max_relative_position + 1
            self.rel_pos_embeddings = nn.Parameter(torch.zeros(num_pos_embeddings, self.head_dim))
            nn.init.normal_(self.rel_pos_embeddings, mean=0.0, std=0.02)

    def _compute_conformer_pos_scores(self, q: Tensor, seqlen: int) -> Tensor:
        positions = torch.arange(seqlen, device=q.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)  # [S, S]
        relative_positions = torch.clamp(
            relative_positions + self.max_relative_position, 0, 2 * self.max_relative_position)
        rel_embeddings = self.rel_pos_embeddings[relative_positions]  # [S, S, D]
        q = q.transpose(1, 2)  # [B, S, H, D]
        rel_logits = torch.matmul(q, rel_embeddings.transpose(-2, -1))  # [B, S, H, S]
        rel_logits = rel_logits.transpose(1, 2)  # [B, H, S, S]
        return rel_logits

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Tensor | None = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
        context_seqlen = seqlen

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

        if self.pos_embed_type == "rope":
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if self.pos_embed_type == "conformer":
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            rel_scores = self._compute_conformer_pos_scores(q, seqlen)
            scores = scores + rel_scores
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            if self.attn_dropout_rate > 0 and self.training:
                attn = F.dropout(attn, p=self.attn_dropout_rate)
            y = torch.matmul(attn, v)
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout_rate if self.training else 0.0,
                attn_mask=mask,
            )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.head_dim * self.n_head)
        y = self.wo(y)
        return y


class FeedForward(nn.Module):

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LayerScale(nn.Module):

    def __init__(self, dim: int, init_values: float | Tensor = 1e-2, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class WindowLimitedTransformer(Transformer):
    """Transformer with window-limited causal attention."""

    def __init__(
        self,
        config: ModelArgs,
        input_dim: int = 512,
        window_size: int | None = None,
        causal: bool = True,
        look_ahead_conv: nn.Module | None = None,
    ):
        super().__init__(config)
        self.window_size = window_size
        self.causal = causal
        self.channels_first = config.channels_first
        self.look_ahead_conv = look_ahead_conv if look_ahead_conv is not None else nn.Identity()
        self.input_proj = nn.Linear(input_dim, config.dim) if input_dim != config.dim else nn.Identity()
        self.output_proj = nn.Linear(config.dim, input_dim) if input_dim != config.dim else nn.Identity()

    def make_window_limited_mask(self, max_length: int, x_lens: Tensor | None = None) -> Tensor:
        if self.causal:
            mask = torch.tril(torch.ones(max_length, max_length))
            row_indices = torch.arange(max_length).view(-1, 1)
            window_size = self.window_size or max_length
            valid_range = (row_indices - window_size + 1).clamp(min=0)
            column_indices = torch.arange(max_length)
            mask = (column_indices >= valid_range) & mask.bool()
        else:
            raise NotImplementedError
        mask = mask.bool()[None, None]
        return mask

    def make_mask(self, max_length: int, x_lens: Tensor | None = None) -> Tensor:
        if self.causal:
            mask = torch.tril(torch.ones(max_length, max_length))
        else:
            mask = torch.ones(max_length, max_length)
            mask = mask.bool()[None, None]
            for i, x_len in enumerate(x_lens):
                mask[:x_len, i] = 0
        mask = mask.bool()[None, None]
        return mask

    def forward(self, x: Tensor, x_lens: Tensor | None = None) -> Tensor:
        if self.channels_first:
            x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.look_ahead_conv(x)
        input_pos = torch.arange(x.shape[1], device=x.device)
        max_length = x.shape[1]
        if self.window_size is not None:
            mask = self.make_window_limited_mask(max_length, x_lens)
        else:
            mask = self.make_mask(max_length, x_lens)
        mask = mask.to(x.device)
        x = super().forward(x, input_pos, mask)
        x = self.output_proj(x)
        if self.channels_first:
            x = x.transpose(1, 2)
        return x


def precompute_freqs_cis(
        seq_len: int, n_elem: int, base: int = 10000, dtype: torch.dtype = torch.bfloat16) -> Tensor:
    freqs = 1.0 / (base**(torch.arange(0, n_elem, 2)[:(n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


# --------------------------------------------------------------------
# Top-level AE
# --------------------------------------------------------------------


class EncoderBlock(nn.Module):

    def __init__(
        self,
        dim: int = 16,
        stride: int = 1,
        causal: bool = False,
        n_t_layer: int = 0,
        transformer_general_config=None,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        transformer_module = (
            nn.Identity() if n_t_layer == 0 else WindowLimitedTransformer(
                causal=causal,
                input_dim=dim,
                window_size=512,
                config=transformer_general_config(
                    n_layer=n_t_layer,
                    n_head=dim // 64,
                    dim=dim,
                    intermediate_size=dim * 3,
                ),
            ))
        self.block = nn.Sequential(
            # three multi‑receptive‑field residual units
            ResidualUnit(dim // 2, dilation=1, causal=causal),
            ResidualUnit(dim // 2, dilation=3, causal=causal),
            ResidualUnit(dim // 2, dilation=9, causal=causal),
            Snake1d(dim // 2),
            conv_class(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
            transformer_module,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualUnit(nn.Module):

    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            conv_class(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            conv_class(dim, dim, kernel_size=1),
        )
        self.causal = causal

    def forward(self, x: Tensor) -> Tensor:
        y = self.block(x)
        pad = x.shape[-1] - y.shape[-1]
        if pad > 0:
            if self.causal:
                x = x[..., :-pad]
            else:
                x = x[..., pad // 2:-pad // 2]
        return x + y


class Encoder(nn.Module):

    def __init__(
        self,
        d_model: int = 64,
        strides: list[int] = [2, 4, 8, 8],
        d_latent: int = 64,
        n_transformer_layers: list[int] = [0, 0, 4, 4],
        transformer_general_config: ModelArgs | None = None,
        causal: bool = False,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        layers: list[nn.Module] = [conv_class(1, d_model, kernel_size=7, padding=3)]
        for stride, n_t_layer in zip(strides, n_transformer_layers):
            d_model *= 2
            layers.append(
                EncoderBlock(
                    d_model,
                    stride=stride,
                    causal=causal,
                    n_t_layer=n_t_layer,
                    transformer_general_config=transformer_general_config,
                ))
        layers += [Snake1d(d_model), conv_class(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*layers)
        self.enc_dim = d_model

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        causal: bool = False,
        n_t_layer: int = 0,
        transformer_general_config=None,
    ):
        super().__init__()
        conv_trans_class = CausalWNConvTranspose1d if causal else WNConvTranspose1d
        # The published Fish S1 DAC decoder never attaches the transformer
        # constructed by the original factory. Its checkpoint starts with
        # Snake at block.0, followed by convolution and residual units.
        del n_t_layer, transformer_general_config
        self.block = nn.Sequential(
            Snake1d(input_dim),
            conv_trans_class(
                input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
            ResidualUnit(output_dim, dilation=1, causal=causal),
            ResidualUnit(output_dim, dilation=3, causal=causal),
            ResidualUnit(output_dim, dilation=9, causal=causal),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Decoder(nn.Module):

    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: list[int],
        d_out: int = 1,
        causal: bool = False,
        n_transformer_layers: list[int] = [0, 0, 0, 0],
        transformer_general_config=None,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        layers: list[nn.Module] = [conv_class(input_channel, channels, kernel_size=7, padding=3)]
        for i, (stride, n_t_layer) in enumerate(zip(rates, n_transformer_layers)):
            input_dim = channels // 2**i
            output_dim = channels // 2**(i + 1)
            layers.append(
                DecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    causal=causal,
                    n_t_layer=n_t_layer,
                    transformer_general_config=transformer_general_config,
                ))
        layers += [Snake1d(output_dim), conv_class(output_dim, d_out, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class DAC(nn.Module):

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list[int] = [2, 4, 8, 8],
        latent_dim: int | None = None,
        decoder_dim: int = 1536,
        decoder_rates: list[int] = [8, 8, 4, 2],
        quantizer: nn.Module | None = None,
        sample_rate: int = 44100,
        causal: bool = True,
        encoder_transformer_layers: list[int] = [0, 0, 0, 0],
        decoder_transformer_layers: list[int] = [0, 0, 0, 0],
        transformer_general_config=None,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2**len(encoder_rates))
        self.latent_dim = latent_dim

        self.hop_length = math.prod(encoder_rates)
        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            latent_dim,
            causal=causal,
            n_transformer_layers=encoder_transformer_layers,
            transformer_general_config=transformer_general_config,
        )
        self.quantizer = quantizer
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            causal=causal,
            n_transformer_layers=decoder_transformer_layers,
            transformer_general_config=transformer_general_config,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()
        self.frame_length = self.hop_length * 4

    def get_output_length(self, input_length: int) -> int:
        length = input_length
        for stride in self.encoder_rates:
            length = math.ceil(length / stride)
        return length

    def get_delay(self) -> int:
        l_out = self.get_output_length(0)
        L = l_out

        layers = [layer for layer in self.modules() if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d))]
        for layer in reversed(layers):
            d = layer.dilation[0]
            k = layer.kernel_size[0]
            s = layer.stride[0]
            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                L = (L - 1) * s + d * (k - 1) + 1
            L = math.ceil(L)

        l_in = L
        return (l_in - l_out) // 2

    def preprocess(self, audio_data: Tensor, sample_rate: int | None) -> Tensor:
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = F.pad(audio_data, (0, right_pad))
        return audio_data

    def encode(
        self,
        audio_data: Tensor,
        audio_lengths: Tensor | None = None,
        n_quantizers: int | None = None,
        **kwargs,
    ):
        """Encode audio to quantized code indices."""
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.frame_length) * self.frame_length - length
        audio_data = F.pad(audio_data, (0, right_pad))
        if audio_lengths is None:
            audio_lengths = torch.LongTensor([length + right_pad]).to(audio_data.device)

        z = self.encoder(audio_data)
        vq_results = self.quantizer(z, n_quantizers, **kwargs)
        indices = vq_results.codes
        indices_lens = torch.ceil(audio_lengths / self.frame_length).long()
        return indices, indices_lens

    def decode(self, indices: Tensor, feature_lengths: Tensor):
        """Decode code indices to audio."""
        if indices.ndim == 2:
            indices = indices[None]
        z = self.quantizer.decode(indices)
        audio_lengths = feature_lengths * self.frame_length
        return self.decoder(z), audio_lengths

    def encode_to_codes(
            self, audio: Tensor, audio_lengths: Tensor | None = None, n_quantizers: int | None = None, **kw):
        return self.encode(audio, audio_lengths, n_quantizers, **kw)

    def decode_codes(self, indices: Tensor, feature_lengths: Tensor):
        return self.decode(indices, feature_lengths)

    @torch.no_grad()
    def encode_zq(self, audio_data: Tensor) -> Tensor:
        indices, _ = self.encode(audio_data)
        new_indices = torch.zeros_like(indices)
        new_indices[:, 0] = torch.clamp(
            indices[:, 0], max=self.quantizer.semantic_quantizer.codebook_size - 1)
        new_indices[:, 1:] = torch.clamp(indices[:, 1:], max=self.quantizer.quantizer.codebook_size - 1)

        z_q_semantic = self.quantizer.semantic_quantizer.from_codes(new_indices[:, :1])[0]
        z_q_residual = self.quantizer.quantizer.from_codes(new_indices[:, 1:])[0]
        z_q = z_q_semantic + z_q_residual
        return z_q

    @torch.no_grad()
    def decode_zq(self, z_q: Tensor) -> Tensor:
        z_q = self.quantizer.post_module(z_q)
        z_q = self.quantizer.upsample(z_q)
        return self.decoder(z_q)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


# --------------------------------------------------------------------
# Build helpers
# --------------------------------------------------------------------


def build_ae(**cfg) -> DAC:
    """Factory used by external loaders."""
    # Shared transformer config for the RVQ pre/post modules
    q_config = ModelArgs(
        block_size=4096,
        n_layer=8,
        n_head=16,
        dim=1024,
        intermediate_size=3072,
        head_dim=64,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True)

    def make_transformer():
        return WindowLimitedTransformer(causal=True, window_size=128, input_dim=1024, config=q_config)

    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024,
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.5,
        downsample_factor=(2, 2),
        semantic_codebook_size=4096,
        pre_module=make_transformer(),
        post_module=make_transformer(),
    )

    def transformer_general_config(**kw):
        return ModelArgs(
            block_size=kw.get("block_size", 16384),
            n_layer=kw.get("n_layer", 8),
            n_head=kw.get("n_head", 8),
            dim=kw.get("dim", 512),
            intermediate_size=kw.get("intermediate_size", 1536),
            n_local_heads=kw.get("n_local_heads", -1),
            head_dim=kw.get("head_dim", 64),
            rope_base=kw.get("rope_base", 10000),
            norm_eps=kw.get("norm_eps", 1e-5),
            dropout_rate=kw.get("dropout_rate", 0.1),
            attn_dropout_rate=kw.get("attn_dropout_rate", 0.1),
            channels_first=kw.get("channels_first", True),
        )

    dac = DAC(
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        latent_dim=1024,
        decoder_dim=1536,
        decoder_rates=[8, 8, 4, 2],
        quantizer=quantizer,
        sample_rate=44100,
        causal=True,
        encoder_transformer_layers=[0, 0, 0, 4],
        decoder_transformer_layers=[4, 0, 0, 0],
        transformer_general_config=transformer_general_config,
    )
    return dac


__all__ = [
    "DAC",
    "build_ae",
    "VectorQuantize",
    "ResidualVectorQuantize",
    "DownsampleResidualVectorQuantize",
]

# ----- BEGIN DAC MIT LICENSE -----
# MIT License
# Copyright (c) 2023-present, Descript
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----- END DAC MIT LICENSE -----
