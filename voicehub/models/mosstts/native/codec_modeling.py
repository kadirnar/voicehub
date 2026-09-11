# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native PyTorch MOSS Audio Tokenizer model."""

from __future__ import annotations

import copy
import logging
import math
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec_configuration import MossAudioTokenizerConfig


def auto_docstring(*args, **kwargs):
    """Compatibility no-op for doc decorators used in the published source."""
    del kwargs
    if len(args) == 1 and callable(args[0]):
        return args[0]

    def decorator(obj):
        return obj

    return decorator


logger = logging.getLogger(__name__)
flash_attn_varlen_func = None
HAS_FLASH_ATTN = False

SUPPORTED_ATTENTION_IMPLEMENTATIONS = {"sdpa"}
SUPPORTED_COMPUTE_DTYPES = {
    "fp32": None,
    "float32": None,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
SUPPORTED_CODEC_WEIGHT_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
CANONICAL_CODEC_WEIGHT_DTYPES = {
    "fp32": "fp32",
    "float32": "fp32",
    "bf16": "bf16",
    "bfloat16": "bf16",
}


def resolve_compute_dtype(compute_dtype: str) -> torch.dtype | None:
    if compute_dtype not in SUPPORTED_COMPUTE_DTYPES:
        raise ValueError(
            f"Unsupported compute_dtype={compute_dtype!r}. Expected one of {sorted(SUPPORTED_COMPUTE_DTYPES)}."
        )
    return SUPPORTED_COMPUTE_DTYPES[compute_dtype]


def canonicalize_codec_weight_dtype(codec_weight_dtype: str) -> str:
    key = str(codec_weight_dtype).lower()
    if key not in CANONICAL_CODEC_WEIGHT_DTYPES:
        raise ValueError(
            "Unsupported codec_weight_dtype="
            f"{codec_weight_dtype!r}. Expected one of {sorted(CANONICAL_CODEC_WEIGHT_DTYPES)}.")
    return CANONICAL_CODEC_WEIGHT_DTYPES[key]


def resolve_codec_weight_dtype(codec_weight_dtype: str) -> torch.dtype:
    key = str(codec_weight_dtype).lower()
    if key not in SUPPORTED_CODEC_WEIGHT_DTYPES:
        raise ValueError(
            "Unsupported codec_weight_dtype="
            f"{codec_weight_dtype!r}. Expected one of {sorted(SUPPORTED_CODEC_WEIGHT_DTYPES)}.")
    return SUPPORTED_CODEC_WEIGHT_DTYPES[key]


@contextmanager
def disable_cuda_autocast():
    with torch.autocast(device_type="cuda", enabled=False):
        yield


# =============================================================================
# Output Classes
# =============================================================================


@dataclass
@auto_docstring
class MossAudioTokenizerEncoderOutput:
    r"""audio_codes (`torch.LongTensor` of shape `(num_quantizers, batch_size,
    sequence_length)`, *optional*):

    Discrete audio codes computed using the encoder and quantizer.
    audio_codes_lengths (`torch.LongTensor` of shape `(batch_size,)`,
    *optional*):     Valid lengths for each sample's audio codes.
    encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size,
    hidden_size, sequence_length)`, *optional*):     Hidden states from
    the encoder before quantization.
    """

    audio_codes: torch.Tensor | None = None
    audio_codes_lengths: torch.Tensor | None = None
    encoder_hidden_states: torch.Tensor | None = None


@dataclass
@auto_docstring
class MossAudioTokenizerDecoderOutput:
    r"""Audio (`torch.FloatTensor` of shape `(batch_size, channels,
    sequence_length)`, *optional*):

    Decoded audio waveform. audio_lengths (`torch.LongTensor` of shape
    `(batch_size,)`, *optional*):     Valid lengths for each sample's
    audio.
    """

    audio: torch.Tensor | None = None
    audio_lengths: torch.Tensor | None = None


@dataclass
@auto_docstring
class MossAudioTokenizerOutput:
    r"""Audio (`torch.FloatTensor` of shape `(batch_size, channels,
    sequence_length)`, *optional*):

    Decoded audio waveform. audio_lengths (`torch.LongTensor` of shape
    `(batch_size,)`, *optional*):     Valid lengths for each sample's
    audio. audio_codes (`torch.LongTensor` of shape `(num_quantizers,
    batch_size, sequence_length)`, *optional*):     Discrete audio codes
    computed using the encoder and quantizer. audio_codes_lengths
    (`torch.LongTensor` of shape `(batch_size,)`, *optional*):     Valid
    lengths for each sample's audio codes.
    """

    audio: torch.Tensor | None = None
    audio_lengths: torch.Tensor | None = None
    audio_codes: torch.Tensor | None = None
    audio_codes_lengths: torch.Tensor | None = None


# =============================================================================
# Streaming Module Base Classes
# =============================================================================


@dataclass
class StreamingState:
    """Base state for streaming modules."""

    batch_size: int
    device: torch.device

    def __post_init__(self):
        self.exec_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

    def set_exec_mask(self, exec_mask: torch.Tensor):
        self.exec_mask[:] = exec_mask

    def reset(self, reset_mask: torch.Tensor) -> None:
        self.exec_mask[:] = torch.where(reset_mask, torch.ones_like(self.exec_mask), self.exec_mask)

    def __enter__(self):
        # ExitStack expects a context manager; returning self is conventional and useful for debugging.
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


class StreamingModule(nn.Module):
    """Base class for streaming components."""

    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: StreamingState | None = None
        self._streaming_detached: bool = False
        self._cached_children: list[tuple[str, StreamingModule]] | None = None

    @property
    def is_streaming(self):
        return self._streaming_state is not None

    def _apply_named_streaming(self, fn):

        def _handle_module(prefix: str, module: nn.Module):
            if isinstance(module, StreamingModule):
                if module._streaming_detached and prefix != "":
                    return
                if self._cached_children is None:
                    raise RuntimeError(
                        "Internal error: _cached_children should be initialized before traversal.")
                self._cached_children.append((prefix, module))
            for name, child in module.named_children():
                new_prefix = f"{prefix}.{name}" if prefix else name
                _handle_module(new_prefix, child)

        if self._cached_children is None:
            self._cached_children = []
            _handle_module("", self)
        for name, child in self._cached_children:
            fn(name, child)

    def _start_streaming(self, batch_size: int, exit_stack: ExitStack):

        def _start_streaming_fn(name: str, module: StreamingModule):
            if module._streaming_state is not None:
                raise RuntimeError(f"{name} is already streaming!")
            state = module._init_streaming_state(batch_size)
            exit_stack.enter_context(state)
            module._streaming_state = state

        self._apply_named_streaming(_start_streaming_fn)

    def _stop_streaming(self) -> None:

        def _stop_streaming_fn(name: str, module: StreamingModule):
            module._streaming_state = None

        self._apply_named_streaming(_stop_streaming_fn)

    def _init_streaming_state(self, batch_size: int) -> StreamingState:
        device = next(iter(self.parameters())).device
        return StreamingState(batch_size, device)

    def streaming(self, batch_size: int) -> ExitStack:
        """Context manager to enter streaming mode."""
        exit_stack = ExitStack()
        self._start_streaming(batch_size, exit_stack)
        exit_stack.callback(self._stop_streaming)
        return exit_stack


class StreamingContainer(StreamingModule):
    """Container for streaming modules."""

    pass


# =============================================================================
# Normalization Layers
# =============================================================================


class MossAudioTokenizerRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: torch.dtype | None = None,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.alpha = nn.Parameter(
            torch.full((1, 1, dim), 1.0, requires_grad=True, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        if self.dtype is not None:
            x = x.to(self.dtype)
        var = self.eps + torch.mean(x**2, dim=-1, keepdim=True)
        alpha = self.alpha.to(var)
        if x.dim() == 2:
            alpha = alpha.view(1, -1)
        y = (x * (alpha * torch.rsqrt(var))).to(x_dtype)
        return y


class MossAudioTokenizerLayerScale(nn.Module):
    """Layer scale from Touvron et al.

    2021.
    """

    def __init__(
        self,
        channels: int,
        init: float = 1e-4,
        channel_last: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(
            torch.full((channels, ), init, requires_grad=True, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    """Create normalization module."""
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    elif norm_type in {"rms_norm"}:
        return MossAudioTokenizerRMSNorm(dim, eps=1e-5, **kwargs)
    elif norm_type in {"rms_norm_f32"}:
        kwargs.pop("dtype", None)
        return MossAudioTokenizerRMSNorm(dim, eps=1e-8, dtype=torch.float, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


# =============================================================================
# Rotary Position Embedding
# =============================================================================


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    offset: torch.Tensor,
    max_period: float = 10_000,
    time_before_heads: bool = False,
):
    """Apply rotary position embedding."""
    if time_before_heads:
        B, T, H, D = q.shape
    else:
        B, H, T, D = q.shape
    if k.shape != q.shape:
        raise ValueError(f"Expected k.shape == q.shape, got k={tuple(k.shape)} q={tuple(q.shape)}")
    if D <= 0 or (D % 2) != 0:
        raise ValueError(f"RoPE requires an even last dimension, got D={D}")

    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = offset.float().view(-1, 1) + torch.arange(T, device=q.device, dtype=torch.float32)

    if time_before_heads:
        ts = ts.view(B, -1, 1, 1)
    else:
        ts = ts.view(B, 1, -1, 1)

    dims = q.shape[:-1]
    q = q.view(*dims, D // 2, 2)
    k = k.view(*dims, D // 2, 2)

    qr, qi = q[..., 0].float(), q[..., 1].float()
    kr, ki = k[..., 0].float(), k[..., 1].float()

    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)

    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)

    return qo.view(*dims, D), ko.view(*dims, D)


def apply_rope_with_positions(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    max_period: float = 10_000,
):
    """Apply rotary position embedding to packed `[N, H, D]` tensors."""
    N, H, D = q.shape
    if k.shape != q.shape:
        raise ValueError(f"Expected k.shape == q.shape, got k={tuple(k.shape)} q={tuple(q.shape)}")
    if D <= 0 or (D % 2) != 0:
        raise ValueError(f"RoPE requires an even last dimension, got D={D}")

    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = positions.to(torch.float32).view(N, 1, 1)

    qr = q.float().view(N, H, D // 2, 2)[..., 0]
    qi = q.float().view(N, H, D // 2, 2)[..., 1]
    kr = k.float().view(N, H, D // 2, 2)[..., 0]
    ki = k.float().view(N, H, D // 2, 2)[..., 1]

    rotr = torch.cos(ts * freqs.view(1, 1, -1))
    roti = torch.sin(ts * freqs.view(1, 1, -1))

    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)
    return qo.view(N, H, D), ko.view(N, H, D)


class MossAudioTokenizerRotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE)."""

    def __init__(self, max_period: float = 10000.0):
        super().__init__()
        self.max_period = max_period

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: torch.Tensor,
        time_before_heads: bool = False,
    ):
        return apply_rope(q, k, offset, self.max_period, time_before_heads)


# =============================================================================
# Gating Modules
# =============================================================================


class MossAudioTokenizerActivationGating(nn.Module):
    """Gating FFN layer with activation."""

    def __init__(self, dim: int, dim_feedforward: int, activation, **factory_kwargs):
        super().__init__()
        if dim_feedforward == 4 * dim:
            hidden = (21 * dim) // 8
        else:
            hidden = (2 * dim_feedforward) // 3

        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False, **factory_kwargs)
        self.linear_out = nn.Linear(hidden, dim, bias=False, **factory_kwargs)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        x = self.linear_in(x)
        B, T, _ = x.shape
        x = x.view(B, T, 2, -1)
        x = self.activation(x[..., 0, :]) * x[..., 1, :]
        x = self.linear_out(x)
        return x


def _get_activation(name: str):
    if name in ["sigmoid", "tanh", "relu"]:
        return getattr(torch, name)
    elif name in ["leaky_relu", "elu", "gelu", "silu", "mish", "softsign"]:
        return getattr(F, name)
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation {name}")


def make_gating(name: str, dim: int, dim_feedforward: int, **factory_kwargs) -> nn.Module:
    return MossAudioTokenizerActivationGating(dim, dim_feedforward, _get_activation(name), **factory_kwargs)


# =============================================================================
# Positional Embeddings
# =============================================================================


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create sinusoidal positional embedding with shape [..., C]."""
    if dim % 2 != 0:
        raise ValueError(f"Sinusoidal embedding requires even dim, got dim={dim}")
    half_dim = dim // 2
    if half_dim <= 1:
        raise ValueError(f"Sinusoidal embedding requires dim >= 4, got dim={dim}")
    if positions.dim() == 0:
        positions = positions.view(1)
    positions = positions.to(dtype).unsqueeze(-1)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)
    phase = positions / (max_period_tensor**(adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def pack_padded_sequence(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a padded `[B, T, D]` tensor into `[N, D]` plus metadata."""
    batch_size, max_seqlen, _ = x.shape
    positions = torch.arange(max_seqlen, device=x.device, dtype=torch.long)
    valid_mask = positions.view(1, max_seqlen) < input_lengths.view(batch_size, 1)
    packed_x = x[valid_mask]
    cu_seqlens = torch.zeros(batch_size + 1, device=x.device, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(input_lengths.to(torch.int32), dim=0)
    position_ids = positions.view(1, max_seqlen).expand(batch_size, -1)[valid_mask]
    return packed_x, valid_mask, cu_seqlens, position_ids


def unpack_packed_sequence(
    packed_x: torch.Tensor,
    valid_mask: torch.Tensor,
    batch_size: int,
    max_seqlen: int,
) -> torch.Tensor:
    """Unpack a packed `[N, D]` tensor back into `[B, T, D]`."""
    output = packed_x.new_zeros((batch_size, max_seqlen, packed_x.shape[-1]))
    output[valid_mask] = packed_x
    return output


# =============================================================================
# KV Cache for Attention
# =============================================================================


class KVCacheResult:
    """Container for KV cache results that supports tuple unpacking."""

    __slots__ = ("keys", "values", "positions")

    def __init__(self, keys: torch.Tensor, values: torch.Tensor, positions: torch.Tensor):
        self.keys = keys
        self.values = values
        self.positions = positions

    def __iter__(self):
        """Allow unpacking as (keys, values, positions)."""
        return iter((self.keys, self.values, self.positions))

    @staticmethod
    def from_kv(keys: torch.Tensor, values: torch.Tensor) -> KVCacheResult:
        B, H, T, D = keys.shape
        positions = torch.arange(T, device=keys.device, dtype=torch.long)
        return KVCacheResult(keys, values, positions.expand(B, -1))


class RingKVCache:
    """Efficient streaming KVCache compatible with CUDA Graph."""

    def __init__(
            self,
            batch_size: int,
            num_heads: int,
            dim_per_head: int,
            capacity: int,
            respect_exec_mask: bool = True,
            device: torch.device = torch.device("cuda"),
            dtype: torch.dtype = torch.bfloat16,
    ):
        self.capacity = capacity
        self.cache = torch.zeros(
            (2, batch_size, num_heads, capacity, dim_per_head),
            device=device,
            dtype=dtype,
        )
        self.respect_exec_mask = respect_exec_mask
        if self.respect_exec_mask:
            self.end_offset = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            self.end_offset = torch.zeros(1, device=device, dtype=torch.long)

    def reset(self, reset_mask: torch.Tensor) -> None:
        self.end_offset[:] = torch.where(reset_mask, torch.zeros_like(self.end_offset), self.end_offset)

    def complete(self, k: torch.Tensor, v: torch.Tensor, exec_mask: torch.Tensor) -> KVCacheResult:
        B, H, T, D = k.shape
        if T <= 0:
            raise ValueError(f"Expected T > 0, got T={T}")

        indexes = torch.arange(T, device=self.end_offset.device, dtype=self.end_offset.dtype)
        indexes = indexes + self.end_offset.view(-1, 1)
        indexes = indexes % self.capacity

        if self.respect_exec_mask:
            this_indexes = indexes.view(B, 1, T, 1).expand(-1, H, T, D)
            self.cache[0].scatter_(2, this_indexes, k)
            self.cache[1].scatter_(2, this_indexes, v)
        else:
            self.cache[0].index_copy_(2, indexes[0], k)
            self.cache[1].index_copy_(2, indexes[0], v)

        keys = self.cache[0]
        values = self.cache[1]

        indexes = torch.arange(self.capacity, device=self.end_offset.device, dtype=torch.long)
        last_offset = self.end_offset.view(-1, 1) + T - 1
        end_index = last_offset % self.capacity
        delta = indexes - end_index

        positions = torch.where(
            delta <= 0,
            last_offset + delta,
            last_offset + delta - self.capacity,
        )

        if self.respect_exec_mask:
            self.end_offset[:] = torch.where(exec_mask, self.end_offset + T, self.end_offset)
        else:
            self.end_offset.add_(T)

        invalid = indexes >= self.end_offset.view(-1, 1)
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)

        return KVCacheResult(keys, values, positions)


# =============================================================================
# Multi-Head Attention
# =============================================================================


@dataclass
class MHAState(StreamingState):
    cached_keys: torch.Tensor | None
    cached_values: torch.Tensor | None
    cached_positions: torch.Tensor | None
    offset: torch.Tensor

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.offset[:] = torch.where(reset_mask, torch.zeros_like(self.offset), self.offset)
        if self.cached_positions is not None:
            self.cached_positions[reset_mask] = -1
        if self.cached_keys is not None:
            self.cached_keys[reset_mask] = 0
        if self.cached_values is not None:
            self.cached_values[reset_mask] = 0


def apply_weights_per_step(
    modules: nn.ModuleList,
    schedule: list[int] | None,
    x: torch.Tensor,
    offset: int | None,
) -> torch.Tensor:
    """Apply different weights for each time step."""
    if len(modules) == 1:
        return modules[0](x)

    if offset is None:
        raise ValueError("offset must be provided when using per-step weights (len(modules) > 1).")
    if x.dim() != 3:
        raise ValueError(
            f"Per-step weights require a dense `[B, T, C]` tensor when len(modules) > 1, got shape {tuple(x.shape)}."
        )
    ys = []
    B, T, C = x.shape
    for t in range(T):
        module_index = t + offset
        if schedule is not None:
            if module_index >= len(schedule) or module_index < 0:
                raise ValueError(
                    f"weights_per_step_schedule is too short for module_index={module_index} (len={len(schedule)})."
                )
            module_index = schedule[module_index]
        if module_index >= len(modules) or module_index < 0:
            raise ValueError(f"module_index={module_index} out of range for len(modules)={len(modules)}.")
        y = modules[module_index](x[:, t:t + 1])
        ys.append(y)
    return torch.cat(ys, 1)


class MossAudioTokenizerMultiheadAttention(StreamingModule):
    """Multi-head attention with streaming support."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: int | None = None,
        rope: MossAudioTokenizerRotaryEmbedding | None = None,
        attention_implementation: str = "sdpa",
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.causal = causal
        self.context = context
        self.rope = rope
        self.num_heads = num_heads
        if attention_implementation not in SUPPORTED_ATTENTION_IMPLEMENTATIONS:
            raise ValueError(
                f"Unsupported attention_implementation={attention_implementation!r}. "
                f"Expected one of {sorted(SUPPORTED_ATTENTION_IMPLEMENTATIONS)}.")
        self.attention_implementation = attention_implementation
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)

        self._register_load_state_dict_pre_hook(self._load_hook, with_module=True)

    def set_attention_implementation(self, attention_implementation: str) -> None:
        if attention_implementation not in SUPPORTED_ATTENTION_IMPLEMENTATIONS:
            raise ValueError(
                f"Unsupported attention_implementation={attention_implementation!r}. "
                f"Expected one of {sorted(SUPPORTED_ATTENTION_IMPLEMENTATIONS)}.")
        self.attention_implementation = attention_implementation

    @staticmethod
    def _load_hook(module, state_dict, prefix, *_):
        mappings = {
            "in_proj_weight": "in_proj.weight",
            "in_projs.0.weight": "in_proj.weight",
            "out_projs.0.weight": "out_proj.weight",
        }
        for suffix in ["", "_scb"]:
            for source, target in mappings.items():
                this_source = prefix + source + suffix
                if this_source in state_dict:
                    state_dict[prefix + target + suffix] = state_dict.pop(this_source)

    def _init_streaming_state(self, batch_size: int) -> MHAState:
        device = cast(torch.device, self.in_proj.weight.device)
        return MHAState(
            batch_size,
            device,
            cached_keys=None,
            cached_values=None,
            cached_positions=None,
            offset=torch.zeros(batch_size, device=cast(torch.device, device), dtype=torch.long),
        )

    def _supports_flash_attention(self, device: torch.device, dtype: torch.dtype) -> bool:
        return HAS_FLASH_ATTN and device.type == "cuda" and dtype == torch.bfloat16

    def _get_backend_check_dtype(self, x: torch.Tensor) -> torch.dtype:
        if x.device.type != "cuda":
            return x.dtype
        try:
            autocast_enabled = torch.is_autocast_enabled("cuda")
        except TypeError:
            autocast_enabled = torch.is_autocast_enabled()
        if not autocast_enabled:
            return x.dtype
        try:
            return torch.get_autocast_dtype("cuda")
        except TypeError:
            return torch.get_autocast_gpu_dtype()

    def resolve_attention_implementation(self, x: torch.Tensor, is_streaming: bool) -> str:
        if self.attention_implementation == "sdpa":
            return "sdpa"
        backend_dtype = self._get_backend_check_dtype(x)
        if self._supports_flash_attention(x.device, backend_dtype):
            return "flash_attention_2"
        if self.attention_implementation == "flash_attention_2":
            logger.warning(
                "Falling back to SDPA because flash_attention_2 is unavailable for device=%s dtype=%s "
                "(HAS_FLASH_ATTN=%s).",
                x.device,
                backend_dtype,
                HAS_FLASH_ATTN,
            )
        return "sdpa"

    def _project_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim_per_head = self.embed_dim // self.num_heads
        if x.dim() == 3:
            projected = self.in_proj(x)
            projected = projected.reshape(x.shape[0], x.shape[1], 3, self.num_heads,
                                          dim_per_head).permute(2, 0, 3, 1, 4)
            return projected[0], projected[1], projected[2]
        if x.dim() == 2:
            projected = self.in_proj(x)
            projected = projected.view(x.shape[0], 3, self.num_heads, dim_per_head)
            return projected[:, 0], projected[:, 1], projected[:, 2]
        raise ValueError(f"Expected a 2D or 3D tensor, got shape {tuple(x.shape)}")

    def _apply_dense_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None:
            return q, k
        offset = torch.zeros(q.shape[0], device=q.device, dtype=torch.long)
        return self.rope(q, k, offset, time_before_heads=False)

    def _apply_packed_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None:
            return q, k
        return apply_rope_with_positions(q, k, position_ids, max_period=self.rope.max_period)

    def _ensure_streaming_cache(
        self,
        state: MHAState,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head_dim = self.embed_dim // self.num_heads
        cache_length = 0 if self.context is None else self.context
        if state.cached_keys is None or state.cached_values is None or state.cached_positions is None:
            state.cached_keys = torch.zeros(
                (batch_size, self.num_heads, cache_length, head_dim),
                device=device,
                dtype=dtype,
            )
            state.cached_values = torch.zeros_like(state.cached_keys)
            state.cached_positions = torch.full(
                (batch_size, cache_length),
                -1,
                device=device,
                dtype=torch.long,
            )
        else:
            if state.cached_keys.device != device or state.cached_keys.dtype != dtype:
                state.cached_keys = state.cached_keys.to(device=device, dtype=dtype)
            if state.cached_values.device != device or state.cached_values.dtype != dtype:
                state.cached_values = state.cached_values.to(device=device, dtype=dtype)
            if state.cached_positions.device != device:
                state.cached_positions = state.cached_positions.to(device=device)
        return state.cached_keys, state.cached_values, state.cached_positions

    def _build_streaming_kv(
        self,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        cached_pos: torch.Tensor,
        k_cur: torch.Tensor,
        v_cur: torch.Tensor,
        pos_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_all = torch.cat([cached_k, k_cur], dim=2)
        v_all = torch.cat([cached_v, v_cur], dim=2)
        pos_k = torch.cat([cached_pos, pos_q], dim=1)
        return k_all, v_all, pos_k

    def _update_streaming_cache(
        self,
        state: MHAState,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        cached_pos: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        pos_k: torch.Tensor,
    ) -> None:
        exec_mask = state.exec_mask.view(-1, 1, 1, 1)
        exec_mask_pos = state.exec_mask.view(-1, 1)
        if self.context is None:
            if not bool(state.exec_mask.all().item()):
                raise RuntimeError("Streaming exec_mask with context=None is not supported.")
            state.cached_keys = k_all.contiguous()
            state.cached_values = v_all.contiguous()
            state.cached_positions = pos_k.contiguous()
            return

        new_cached_k = k_all[:, :, -self.context:, :].contiguous()
        new_cached_v = v_all[:, :, -self.context:, :].contiguous()
        new_cached_pos = pos_k[:, -self.context:].contiguous()
        state.cached_keys = torch.where(exec_mask, new_cached_k, cached_k)
        state.cached_values = torch.where(exec_mask, new_cached_v, cached_v)
        state.cached_positions = torch.where(exec_mask_pos, new_cached_pos, cached_pos)

    def _build_streaming_sdpa_bias(self, pos_q: torch.Tensor, pos_k: torch.Tensor) -> torch.Tensor:
        delta = pos_q[:, :, None] - pos_k[:, None, :]
        attn_bias = (pos_k[:, None, :] >= 0) & (delta >= 0)
        if self.context is not None:
            attn_bias = attn_bias & (delta < self.context)
        return attn_bias[:, None, :, :]

    def _build_non_streaming_sdpa_bias(
        self,
        input_lengths: torch.Tensor,
        max_seqlen: int,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(max_seqlen, device=device, dtype=torch.long)
        valid_k = positions.view(1, 1, max_seqlen) < input_lengths.view(-1, 1, 1)
        if not self.causal and self.context is None:
            return valid_k[:, None, :, :].expand(-1, 1, max_seqlen, -1)
        delta = positions.view(1, max_seqlen, 1) - positions.view(1, 1, max_seqlen)
        attn_bias = torch.ones((1, max_seqlen, max_seqlen), device=device, dtype=torch.bool)
        if self.causal:
            attn_bias = attn_bias & (delta >= 0)
        if self.context is not None:
            attn_bias = attn_bias & (delta < self.context)
        return (attn_bias & valid_k)[:, None, :, :]

    def _run_flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        if flash_attn_varlen_func is None:
            raise RuntimeError("flash-attn is not installed.")
        window_size = (self.context, 0) if (self.context is not None and self.causal) else (-1, -1)
        return flash_attn_varlen_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=self.causal,
            window_size=window_size,
        )

    def _forward_streaming_sdpa(self, x: torch.Tensor, state: MHAState) -> torch.Tensor:
        batch_size, chunk_length, _ = x.shape
        q, k_cur, v_cur = self._project_qkv(x)
        if self.rope is not None:
            q, k_cur = self.rope(q, k_cur, state.offset, time_before_heads=False)
        pos_q = state.offset.view(-1, 1) + torch.arange(
            chunk_length, device=x.device, dtype=torch.long).view(1, -1)
        cached_k, cached_v, cached_pos = self._ensure_streaming_cache(
            state, batch_size, k_cur.device, k_cur.dtype)
        k_all, v_all, pos_k = self._build_streaming_kv(cached_k, cached_v, cached_pos, k_cur, v_cur, pos_q)
        attn_bias = self._build_streaming_sdpa_bias(pos_q, pos_k)
        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_bias, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(batch_size, chunk_length, self.embed_dim)

        self._update_streaming_cache(state, cached_k, cached_v, cached_pos, k_all, v_all, pos_k)
        state.offset[:] = torch.where(state.exec_mask, state.offset + chunk_length, state.offset)
        return out

    def _forward_streaming_flash(self, x: torch.Tensor, state: MHAState) -> torch.Tensor:
        batch_size, chunk_length, _ = x.shape
        q, k_cur, v_cur = self._project_qkv(x)
        if self.rope is not None:
            q, k_cur = self.rope(q, k_cur, state.offset, time_before_heads=False)
        pos_q = state.offset.view(-1, 1) + torch.arange(
            chunk_length, device=x.device, dtype=torch.long).view(1, -1)
        cached_k, cached_v, cached_pos = self._ensure_streaming_cache(
            state, batch_size, k_cur.device, k_cur.dtype)
        k_all, v_all, pos_k = self._build_streaming_kv(cached_k, cached_v, cached_pos, k_cur, v_cur, pos_q)

        q_chunks = []
        k_chunks = []
        v_chunks = []
        cu_q = [0]
        cu_k = [0]
        max_kv_len = 0

        for batch_idx in range(batch_size):
            valid_k = pos_k[batch_idx] >= 0
            q_i = q[batch_idx].transpose(0, 1).contiguous()
            k_i = k_all[batch_idx, :, valid_k, :].transpose(0, 1).contiguous()
            v_i = v_all[batch_idx, :, valid_k, :].transpose(0, 1).contiguous()
            q_chunks.append(q_i)
            k_chunks.append(k_i)
            v_chunks.append(v_i)
            cu_q.append(cu_q[-1] + q_i.shape[0])
            cu_k.append(cu_k[-1] + k_i.shape[0])
            max_kv_len = max(max_kv_len, int(k_i.shape[0]))

        out_flat = self._run_flash_attention(
            torch.cat(q_chunks, dim=0),
            torch.cat(k_chunks, dim=0),
            torch.cat(v_chunks, dim=0),
            torch.tensor(cu_q, device=x.device, dtype=torch.int32),
            torch.tensor(cu_k, device=x.device, dtype=torch.int32),
            max_seqlen_q=chunk_length,
            max_seqlen_k=max_kv_len,
        )

        outputs = []
        start = 0
        for _ in range(batch_size):
            outputs.append(out_flat[start:start + chunk_length].transpose(0, 1).contiguous())
            start += chunk_length
        out = torch.stack(outputs, dim=0)
        out = out.transpose(1, 2).reshape(batch_size, chunk_length, self.embed_dim)

        self._update_streaming_cache(state, cached_k, cached_v, cached_pos, k_all, v_all, pos_k)
        state.offset[:] = torch.where(state.exec_mask, state.offset + chunk_length, state.offset)
        return out

    def _forward_non_streaming_sdpa(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, max_seqlen, _ = x.shape
        q, k, v = self._project_qkv(x)
        q, k = self._apply_dense_rope(q, k)

        # The native SDPA path accepts the full local-causal mask, but on long
        # codec sequences that materializes a large T x T bias and can also hit
        # backend-specific silent errors. Query chunking keeps the exact same
        # non-streaming attention pattern while bounding each SDPA call.
        query_chunk_size = 1500
        if max_seqlen <= query_chunk_size:
            attn_bias = self._build_non_streaming_sdpa_bias(input_lengths, max_seqlen, x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
        else:
            out = torch.empty_like(q)
            all_positions = torch.arange(max_seqlen, device=x.device, dtype=torch.long)
            for q_start in range(0, max_seqlen, query_chunk_size):
                q_end = min(q_start + query_chunk_size, max_seqlen)
                if self.causal:
                    k_end = q_end
                else:
                    k_end = max_seqlen

                if self.context is not None:
                    k_start = max(0, q_start - self.context + 1)
                else:
                    k_start = 0

                q_positions = all_positions[q_start:q_end]
                k_positions = all_positions[k_start:k_end]
                valid_k = k_positions.view(1, 1, -1) < input_lengths.view(-1, 1, 1)
                if not self.causal and self.context is None:
                    attn_bias = valid_k[:, None, :, :].expand(-1, 1, q_end - q_start, -1)
                else:
                    delta = q_positions.view(1, -1, 1) - k_positions.view(1, 1, -1)
                    attn_bias = torch.ones(
                        (1, q_end - q_start, k_end - k_start),
                        device=x.device,
                        dtype=torch.bool,
                    )
                    if self.causal:
                        attn_bias = attn_bias & (delta >= 0)
                    if self.context is not None:
                        attn_bias = attn_bias & (delta < self.context)
                    attn_bias = (attn_bias & valid_k)[:, None, :, :]

                out[:, :, q_start:q_end, :] = F.scaled_dot_product_attention(
                    q[:, :, q_start:q_end, :],
                    k[:, :, k_start:k_end, :],
                    v[:, :, k_start:k_end, :],
                    attn_bias,
                    dropout_p=0.0,
                )
        valid_q = (torch.arange(max_seqlen, device=x.device).view(1, max_seqlen)
                   < input_lengths.view(-1, 1)).view(batch_size, 1, max_seqlen, 1)
        # Some SDPA backends return NaNs for fully-masked padded query rows in local-causal attention.
        # Multiplying by zero is not sufficient because NaN * 0 is still NaN; use torch.where so padded
        # rows are materialized as exact zeros before they can leak into later layers as masked K/V values.
        out = torch.where(valid_q, out, torch.zeros((), device=out.device, dtype=out.dtype))
        return out.transpose(1, 2).reshape(batch_size, max_seqlen, self.embed_dim)

    def _forward_non_streaming_flash(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self._project_qkv(x)
        q, k = self._apply_packed_rope(q, k, position_ids)
        out = self._run_flash_attention(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
        return out.reshape(x.shape[0], self.embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        position_ids: torch.Tensor | None = None,
        input_lengths: torch.Tensor | None = None,
    ):
        state = cast(MHAState | None, self._streaming_state)
        backend = self.resolve_attention_implementation(query, is_streaming=state is not None)

        if state is not None:
            if query.dim() != 3:
                raise ValueError(f"Streaming attention expects a 3D tensor, got shape {tuple(query.shape)}")
            out = (
                self._forward_streaming_flash(query, state)
                if backend == "flash_attention_2" else self._forward_streaming_sdpa(query, state))
            return self.out_proj(out)

        if backend == "flash_attention_2":
            if query.dim() != 2:
                raise ValueError(
                    f"Packed flash attention expects a 2D tensor, got shape {tuple(query.shape)}")
            if cu_seqlens is None or max_seqlen is None or position_ids is None:
                raise ValueError("Packed flash attention requires cu_seqlens, max_seqlen, and position_ids.")
            out = self._forward_non_streaming_flash(query, cu_seqlens, max_seqlen, position_ids)
            return self.out_proj(out)

        if query.dim() != 3:
            raise ValueError(f"Non-streaming SDPA expects a 3D tensor, got shape {tuple(query.shape)}")
        if input_lengths is None:
            raise ValueError("Non-streaming SDPA requires input_lengths.")
        out = self._forward_non_streaming_sdpa(query, input_lengths)
        return self.out_proj(out)


# =============================================================================
# Transformer Layer
# =============================================================================


@dataclass
class LayerState(StreamingState):
    pass


class MossAudioTokenizerTransformerLayer(StreamingModule):
    """Transformer layer with streaming support."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        causal: bool = False,
        context: int | None = None,
        rope: MossAudioTokenizerRotaryEmbedding | None = None,
        attention_implementation: str = "sdpa",
        norm: str = "layer_norm",
        layer_scale: float | None = None,
        gating: str = "none",
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.self_attn = MossAudioTokenizerMultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            causal=causal,
            context=context,
            rope=rope,
            attention_implementation=attention_implementation,
            **factory_kwargs,
        )
        self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)
        if gating == "none":
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward, bias=False, **factory_kwargs),
                nn.GELU(),
                nn.Linear(dim_feedforward, d_model, bias=False, **factory_kwargs),
            )
        else:
            self.ffn = make_gating(gating, d_model, dim_feedforward, **factory_kwargs)

        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = MossAudioTokenizerLayerScale(
                channels=d_model,
                init=layer_scale,
                channel_last=True,
                **cast(dict[str, object], factory_kwargs))
            self.layer_scale_2 = MossAudioTokenizerLayerScale(
                channels=d_model,
                init=layer_scale,
                channel_last=True,
                **cast(dict[str, object], factory_kwargs))

        self._register_load_state_dict_pre_hook(self._load_hook, with_module=True)

    @staticmethod
    def _load_hook(module, state_dict, prefix, *_):
        mappings = {
            "linear1.weight": "ffn.0.weight",
            "linear2.weight": "ffn.2.weight",
            "linear1.bias": "ffn.0.bias",
            "linear2.bias": "ffn.2.bias",
        }
        for source, target in mappings.items():
            this_source = prefix + source
            if this_source in state_dict:
                state_dict[prefix + target] = state_dict.pop(this_source)

    def _init_streaming_state(self, batch_size: int) -> LayerState:
        device = next(iter(self.parameters())).device
        return LayerState(batch_size, device)

    def forward(self, x: torch.Tensor, **kwargs):
        residual = x
        x = self.norm1(x)
        x = residual.to(x) + self.layer_scale_1(self.self_attn(x, **kwargs))
        residual = x
        x = self.norm2(x)
        x = residual.to(x) + self.layer_scale_2(self.ffn(x))
        return x


# =============================================================================
# Streaming Transformer
# =============================================================================


@dataclass
class TransformerState(StreamingState):
    offsets: torch.Tensor

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.offsets[:] = torch.where(reset_mask, torch.zeros_like(self.offsets), self.offsets)


class MossAudioTokenizerTransformer(StreamingModule):
    """Transformer with streaming/causal support."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        causal: bool = False,
        context: int | None = None,
        positional_embedding: str = "sin",
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        attention_implementation: str = "sdpa",
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, got d_model={d_model}, num_heads={num_heads}")

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale

        self.rope: MossAudioTokenizerRotaryEmbedding | None = None
        if positional_embedding in {"rope", "sin_rope"}:
            self.rope = MossAudioTokenizerRotaryEmbedding(max_period=max_period)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                MossAudioTokenizerTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    rope=self.rope,
                    attention_implementation=attention_implementation,
                    device=device,
                    dtype=dtype,
                    **kwargs,
                ))

    def _init_streaming_state(self, batch_size: int) -> TransformerState:
        device = next(self.parameters()).device
        return TransformerState(
            batch_size,
            device,
            offsets=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def resolve_attention_implementation(self, x: torch.Tensor) -> str:
        if len(self.layers) == 0:
            return "sdpa"
        first_layer = cast(MossAudioTokenizerTransformerLayer, self.layers[0])
        return first_layer.self_attn.resolve_attention_implementation(
            x, is_streaming=self._streaming_state is not None)

    def set_attention_implementation(self, attention_implementation: str) -> None:
        for layer in self.layers:
            cast(MossAudioTokenizerTransformerLayer,
                 layer).self_attn.set_attention_implementation(attention_implementation)

    def forward(self, x: torch.Tensor, **kwargs):
        C = x.shape[-1]
        state = self._streaming_state
        if x.dim() == 3:
            _, T, _ = x.shape
            offsets = (
                torch.zeros(1, dtype=torch.long, device=x.device) if state is None else (
                    state.offsets if isinstance(state, TransformerState) else torch.zeros(
                        1, dtype=torch.long, device=x.device)))
        else:
            T = 0
            offsets = None

        if self.positional_embedding in {"sin", "sin_rope"}:
            if x.dim() == 3:
                positions = torch.arange(
                    T, device=x.device).view(1, -1) + cast(torch.Tensor, offsets).view(-1, 1)
            else:
                position_ids = kwargs.get("position_ids")
                if position_ids is None:
                    raise ValueError(
                        "Packed transformer inputs require position_ids when using sinusoidal embeddings.")
                positions = position_ids
            pos_emb = create_sin_embedding(positions, C, max_period=self.max_period, dtype=x.dtype)
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = layer(x, **kwargs)

        if state is not None and x.dim() == 3:
            assert isinstance(state, TransformerState)
            state.offsets[:] = torch.where(state.exec_mask, state.offsets + T, state.offsets)
        return x


class MossAudioTokenizerProjectedTransformer(StreamingContainer):
    """Transformer with input/output projections."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        d_model: int,
        *,
        conv_layout: bool = False,
        module_type: str,
        **kwargs,
    ):
        super().__init__()
        self.module_type = module_type
        self.downsample_ratio: int = 1
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.input_proj = nn.Linear(input_dimension, d_model, bias=False)
        self.transformer = MossAudioTokenizerTransformer(d_model=d_model, **kwargs)
        self.conv_layout = conv_layout
        self.output_proj = nn.Linear(d_model, output_dimension, bias=False)

    def set_attention_implementation(self, attention_implementation: str) -> None:
        self.transformer.set_attention_implementation(attention_implementation)

    def forward(self, x, input_lengths, **kwargs):
        x = self.input_proj(x.transpose(1, 2))  # (B, D, T) -> (B, T, D)
        if not self.is_streaming and self.transformer.resolve_attention_implementation(
                x) == "flash_attention_2":
            batch_size, max_seqlen, _ = x.shape
            if max_seqlen > 0 and bool(input_lengths.any().item()):
                max_valid_seqlen = int(input_lengths.max().item())
                packed_x, valid_mask, cu_seqlens, position_ids = pack_padded_sequence(x, input_lengths)
                packed_x = self.transformer(
                    packed_x,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_valid_seqlen,
                    position_ids=position_ids,
                    input_lengths=input_lengths,
                    **kwargs,
                )
                x = unpack_packed_sequence(packed_x, valid_mask, batch_size, max_seqlen)
            else:
                x = x.new_zeros(x.shape)
        else:
            x = self.transformer(x, input_lengths=input_lengths, **kwargs)
        x = self.output_proj(x).transpose(1, 2)  # (B, T, D) -> (B, D, T)
        return x, input_lengths


# =============================================================================
# Patched Pretransform Module
# =============================================================================


class MossAudioTokenizerPatchedPretransform(nn.Module):
    """Patching module for downsampling/upsampling."""

    def __init__(self, patch_size: int, is_downsample: bool, module_type: str, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.downsample_ratio: int = patch_size
        self.is_downsample = is_downsample
        self.module_type = module_type

    def encode(self, x, input_lengths):
        b, d, _ = x.shape
        h = self.patch_size
        x = x.reshape(b, d, -1, h).permute(0, 1, 3, 2).reshape(b, d * h, -1)
        # We pad the input waveform to a multiple of `downsample_rate` before applying the encoder.
        # Use a ceil division to match that padding and avoid dropping the last (partially padded) frame.
        output_lengths = input_lengths // self.patch_size
        return x, output_lengths

    def decode(self, x, input_lengths):
        b, dh, length = x.shape
        h = self.patch_size
        d = dh // h
        x = (x.reshape(b, d, h, length).permute(0, 1, 3, 2).reshape(b, d, length * h))
        output_lengths = input_lengths * self.patch_size
        return x, output_lengths

    def forward(self, x, input_lengths):
        if self.is_downsample:
            return self.encode(x, input_lengths)
        else:
            return self.decode(x, input_lengths)


# =============================================================================
# Vector Quantization
# =============================================================================


def WNConv1d(*args, **kwargs):
    """Weight-normalized Conv1d."""
    return nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


def remap_weight_norm_state_dict_keys(state_dict: dict[str, torch.Tensor], prefix: str) -> None:
    replacements = (
        (".weight_g", ".parametrizations.weight.original0"),
        (".weight_v", ".parametrizations.weight.original1"),
    )
    for key in list(state_dict.keys()):
        if not key.startswith(prefix):
            continue
        new_key = key
        for source, target in replacements:
            new_key = new_key.replace(source, target)
        if new_key != key:
            state_dict[new_key] = state_dict.pop(key)


class MossAudioTokenizerVectorQuantize(nn.Module):
    """Single codebook vector quantization (inference only)."""

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        if input_dim != codebook_dim:
            self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
            self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Input tensor of shape (B, D, T)
        Returns:
            z_q: Quantized tensor of shape (B, D, T)
            indices: Code indices of shape (B, T)
            z_e: Encoded tensor before quantization
        """
        z = z.float()
        z_e = self.in_proj(z).float()

        encodings = z_e.transpose(1, 2).reshape(-1, z_e.shape[1])

        codebook_weight = self.codebook.weight
        dist = (
            encodings.pow(2).sum(1, keepdim=True) - 2 * encodings @ codebook_weight.float().t() +
            codebook_weight.float().pow(2).sum(1, keepdim=True).t())

        indices = (-dist).max(1)[1]
        indices = indices.reshape(z.size(0), -1)

        z_q = self.decode_code(indices)
        z_q = self.out_proj(z_q).float()

        return z_q, indices, z_e

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """Decode code indices to embeddings."""
        return self.codebook(embed_id).transpose(1, 2).float()


class MossAudioTokenizerLFQ(nn.Module):
    """LFQ (inference-only) used by ResidualLFQ."""

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        if self.input_dim != self.codebook_dim:
            self.in_proj = WNConv1d(self.input_dim, self.codebook_dim, kernel_size=1)
            self.out_proj = WNConv1d(self.codebook_dim, self.input_dim, kernel_size=1)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize z into codebook vectors."""
        z = z.float()
        z_e = self.in_proj(z).float()
        z_q, indices = self.decode_latents(z_e)
        z_q = (z_e + (z_q - z_e).detach()).float()
        z_q = self.out_proj(z_q).float()
        return z_q, indices, z_e

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code_wo_out_proj(self, embed_id: torch.Tensor) -> torch.Tensor:
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        z_q = self.decode_code_wo_out_proj(embed_id).float()
        z_q = self.out_proj(z_q).float()
        return z_q

    def decode_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Match training LFQ: L2-normalize then argmin squared distance."""
        encodings = latents.transpose(1, 2).reshape(-1, latents.shape[1]).float()
        codebook = self.codebook.weight.float()

        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        dist = (
            encodings.pow(2).sum(1, keepdim=True) - 2 * encodings @ codebook.t() +
            codebook.pow(2).sum(1, keepdim=True).t())
        indices = (-dist).max(1)[1]
        indices = indices.reshape(latents.size(0), -1)
        z_q = self.decode_code_wo_out_proj(indices).float()
        return z_q, indices


class MossAudioTokenizerResidualVQ(nn.Module):
    """Residual Vector Quantization (inference only)."""

    def __init__(
        self,
        input_dim: int = 1024,
        rvq_dim: int | None = None,
        output_dim: int | None = None,
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rvq_dim = rvq_dim or input_dim
        self.output_dim = output_dim or input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.input_proj = (
            WNConv1d(input_dim, self.rvq_dim, kernel_size=1) if input_dim != self.rvq_dim else nn.Identity())
        self.output_proj = (
            WNConv1d(self.rvq_dim, self.output_dim, kernel_size=1)
            if self.rvq_dim != self.output_dim else nn.Identity())

        self.quantizers = nn.ModuleList([
            MossAudioTokenizerVectorQuantize(
                input_dim=self.rvq_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                **kwargs,
            ) for _ in range(num_quantizers)
        ])
        self._register_load_state_dict_pre_hook(self._load_hook, with_module=True)

    @staticmethod
    def _load_hook(module, state_dict, prefix, *_):
        remap_weight_norm_state_dict_keys(state_dict, prefix)

    @torch.no_grad()
    def forward(
        self,
        z: torch.Tensor,
        input_length: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Input tensor of shape (B, D, T)
            input_length: Valid lengths for each sample (B,)
            n_quantizers: Number of quantizers to use
        Returns:
            quantized_out: Quantized output (B, D, T)
            all_indices: All code indices (N, B, T)
            output_length: Output lengths (B,)
        """
        with disable_cuda_autocast():
            z = self.input_proj(z).float()

            batch_size, _, max_time = z.shape
            mask = torch.arange(
                max_time, device=z.device).expand(batch_size, max_time) < input_length.unsqueeze(1)

            quantized_out = torch.zeros_like(z, dtype=torch.float32)
            residual = z.clone().float()
            all_indices = []

            n_quantizers = n_quantizers or self.num_quantizers

            for i, quantizer in enumerate(self.quantizers):
                if i >= n_quantizers:
                    break

                masked_residual = residual * mask.unsqueeze(1)
                z_q_i, indices_i, _ = quantizer(masked_residual.float())

                update_mask = mask.unsqueeze(1)
                quantized_out = quantized_out + z_q_i * update_mask
                residual = residual - z_q_i * update_mask
                all_indices.append(indices_i)

            all_indices = torch.stack(all_indices)  # (N, B, T)
            quantized_out = self.output_proj(quantized_out.float()).float()

        return quantized_out, all_indices, input_length

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes from multiple quantizers to embeddings."""
        with disable_cuda_autocast():
            nq, B, T = codes.shape
            emb = torch.zeros(B, self.rvq_dim, T, device=codes.device, dtype=torch.float32)

            for i, quantizer in enumerate(self.quantizers[:nq]):
                quantizer = cast(MossAudioTokenizerVectorQuantize, quantizer)
                quantized_i = quantizer.decode_code(codes[i]).float()
                emb += quantized_i

            emb = self.output_proj(emb.float()).float()
        return emb


class MossAudioTokenizerResidualLFQ(nn.Module):
    """Residual LFQ (inference only)."""

    def __init__(
        self,
        input_dim: int = 1024,
        rvq_dim: int | None = None,
        output_dim: int | None = None,
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rvq_dim = rvq_dim or input_dim
        self.output_dim = output_dim or input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.input_proj = (
            WNConv1d(input_dim, self.rvq_dim, kernel_size=1) if input_dim != self.rvq_dim else nn.Identity())
        self.output_proj = (
            WNConv1d(self.rvq_dim, self.output_dim, kernel_size=1)
            if self.rvq_dim != self.output_dim else nn.Identity())

        self.quantizers = nn.ModuleList([
            MossAudioTokenizerLFQ(
                input_dim=self.rvq_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                **kwargs,
            ) for _ in range(num_quantizers)
        ])
        self._register_load_state_dict_pre_hook(self._load_hook, with_module=True)

    @staticmethod
    def _load_hook(module, state_dict, prefix, *_):
        remap_weight_norm_state_dict_keys(state_dict, prefix)

    @torch.no_grad()
    def forward(
        self,
        z: torch.Tensor,
        input_length: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference quantization."""
        with disable_cuda_autocast():
            z = self.input_proj(z).float()

            batch_size, _, max_time = z.shape
            mask = torch.arange(
                max_time, device=z.device).expand(batch_size, max_time) < input_length.unsqueeze(1)

            quantized_out = torch.zeros_like(z, dtype=torch.float32)
            residual = z.clone().float()
            all_indices = []

            n_quantizers = n_quantizers or self.num_quantizers
            for i, quantizer in enumerate(self.quantizers):
                if i >= n_quantizers:
                    break

                masked_residual = residual * mask.unsqueeze(1)
                z_q_i, indices_i, _ = quantizer(masked_residual.float())

                update_mask = mask.unsqueeze(1)
                quantized_out = quantized_out + z_q_i * update_mask
                residual = residual - z_q_i * update_mask
                all_indices.append(indices_i)

            all_indices = (
                torch.stack(all_indices) if all_indices else torch.empty(
                    0, batch_size, max_time, device=z.device, dtype=torch.long))
            quantized_out = self.output_proj(quantized_out.float()).float()
        return quantized_out, all_indices, input_length

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        with disable_cuda_autocast():
            nq, B, T = codes.shape
            emb = torch.zeros(B, self.rvq_dim, T, device=codes.device, dtype=torch.float32)
            for i, quantizer in enumerate(self.quantizers[:nq]):
                quantizer = cast(MossAudioTokenizerLFQ, quantizer)
                emb += quantizer.decode_code(codes[i]).float()
            emb = self.output_proj(emb.float()).float()
        return emb


# =============================================================================
# Main Model Classes
# =============================================================================


@auto_docstring
class MossAudioTokenizerPreTrainedModel(nn.Module):
    """Base class for MossAudioTokenizer models."""

    config_class = MossAudioTokenizerConfig
    base_model_prefix = ""
    main_input_name = "input_values"
    input_modalities = "audio"
    supports_gradient_checkpointing = False
    _no_split_modules = [
        "MossAudioTokenizerTransformerLayer",
        "MossAudioTokenizerResidualVQ",
        "MossAudioTokenizerResidualLFQ",
    ]

    def __init__(self, config: MossAudioTokenizerConfig) -> None:
        super().__init__()
        self.config = config

    def post_init(self) -> None:
        """Preserve the upstream construction hook without Transformers."""


class MossAudioTokenizerModel(MossAudioTokenizerPreTrainedModel):
    """MossAudioTokenizer model for audio tokenization and synthesis.

    This model can encode audio waveforms into discrete tokens and
    decode tokens back into audio waveforms.
    """

    def __init__(self, config: MossAudioTokenizerConfig):
        super().__init__(config)

        self.config = config
        _ = config.version
        self.sampling_rate = config.sampling_rate
        self.downsample_rate = config.downsample_rate
        self.number_channels = config.number_channels
        self.enable_channel_interleave = getattr(config, "enable_channel_interleave", True)
        self.attention_implementation = config.attention_implementation
        self.compute_dtype_name = config.compute_dtype
        self.compute_dtype = resolve_compute_dtype(config.compute_dtype)
        self.codec_weight_dtype_name = canonicalize_codec_weight_dtype(
            getattr(config, "codec_weight_dtype", "fp32"))

        encoder_context_durations = [
            float(module_kwargs.get("context_duration", config.causal_transformer_context_duration))
            for module_kwargs in config.encoder_kwargs if module_kwargs["module_type"] == "Transformer"
        ]
        self.causal_transformer_context_duration = (
            min(encoder_context_durations)
            if encoder_context_durations else config.causal_transformer_context_duration)

        # Build encoder
        channel_interleave_factor = (
            self.number_channels if self.enable_channel_interleave and self.number_channels > 1 else 1)
        current_frame_rate: float = float(self.sampling_rate * channel_interleave_factor)
        self.encoder = nn.ModuleList()

        for encoder_kwargs_i in config.encoder_kwargs:
            encoder_kwargs_i = dict(encoder_kwargs_i)  # Make a copy
            if encoder_kwargs_i["module_type"] == "PatchedPretransform":
                self.encoder.append(
                    MossAudioTokenizerPatchedPretransform(**encoder_kwargs_i, is_downsample=True))
            elif encoder_kwargs_i["module_type"] == "Transformer":
                context_duration = float(
                    encoder_kwargs_i.pop("context_duration", self.causal_transformer_context_duration))
                self.encoder.append(
                    MossAudioTokenizerProjectedTransformer(
                        **encoder_kwargs_i,
                        context=int(round(current_frame_rate * context_duration)),
                        attention_implementation=self.attention_implementation,
                    ))
            current_frame_rate /= self.encoder[-1].downsample_ratio

        # Build quantizer
        quantizer_kwargs = dict(config.quantizer_kwargs)
        quantizer_type = quantizer_kwargs.get("quantizer_type", getattr(config, "quantizer_type", "rvq"))
        if quantizer_type in {"rvq", "spec_rvq"}:
            self.quantizer = MossAudioTokenizerResidualVQ(**quantizer_kwargs)
        elif quantizer_type in {"rlfq", "random_prefix_rlfq"}:
            self.quantizer = MossAudioTokenizerResidualLFQ(**quantizer_kwargs)
        else:
            raise ValueError(f"Unsupported quantizer_type: {quantizer_type}")

        # Build decoder
        decoder_kwargs_list = copy.deepcopy(config.decoder_kwargs)
        self.decoder = nn.ModuleList()

        for decoder_kwargs_i in decoder_kwargs_list:
            decoder_kwargs_i = dict(decoder_kwargs_i)
            if decoder_kwargs_i["module_type"] == "PatchedPretransform":
                self.decoder.append(
                    MossAudioTokenizerPatchedPretransform(**decoder_kwargs_i, is_downsample=False))
            elif decoder_kwargs_i["module_type"] == "Transformer":
                context_duration = float(
                    decoder_kwargs_i.pop("context_duration", self.causal_transformer_context_duration))
                self.decoder.append(
                    MossAudioTokenizerProjectedTransformer(
                        **decoder_kwargs_i,
                        context=int(round(current_frame_rate * context_duration)),
                        attention_implementation=self.attention_implementation,
                    ))
            current_frame_rate *= self.decoder[-1].downsample_ratio

        expected_output_frame_rate = float(self.sampling_rate * channel_interleave_factor)
        if int(round(current_frame_rate)) != int(round(expected_output_frame_rate)):
            raise ValueError(
                "Decoder stack does not invert the encoder frame rate correctly: "
                f"got current_frame_rate={current_frame_rate}, expected={expected_output_frame_rate}.")

        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if model_args:
            raise TypeError("Native MOSS codec loading does not accept positional model arguments.")
        from voicehub.models.mosstts.native.codec_checkpoint import load_moss_audio_tokenizer

        codec_weight_dtype = kwargs.pop("codec_weight_dtype", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        if codec_weight_dtype is not None:
            resolved = resolve_codec_weight_dtype(codec_weight_dtype)
            if torch_dtype is not None and torch_dtype != resolved:
                raise ValueError("`torch_dtype` and `codec_weight_dtype` request different dtypes.")
            torch_dtype = resolved
        loaded = load_moss_audio_tokenizer(
            pretrained_model_name_or_path,
            encoder_decoder_dtype=torch_dtype,
            **kwargs,
        )
        if not isinstance(loaded.model, cls):
            raise ValueError("The checkpoint does not contain a MOSS codec v2 graph.")
        return loaded.model

    def _start_streaming(self, batch_size: int):
        """Start streaming mode for all modules."""

        def _start(module):
            if isinstance(module, StreamingModule):
                module._streaming_state = module._init_streaming_state(batch_size)

        self.apply(_start)

    def _stop_streaming(self):
        """Stop streaming mode for all modules."""

        def _stop(module):
            if isinstance(module, StreamingModule):
                module._streaming_state = None

        self.apply(_stop)

    @contextmanager
    def streaming(self, batch_size: int = 1):
        """Context manager for streaming mode."""
        self._start_streaming(batch_size)
        try:
            yield
        finally:
            self._stop_streaming()

    def _set_streaming_exec_mask(self, exec_mask: torch.Tensor) -> None:
        exec_mask = exec_mask.to(torch.bool)

        def _set_exec_mask(module: nn.Module):
            if isinstance(module, StreamingModule) and module._streaming_state is not None:
                module._streaming_state.set_exec_mask(exec_mask.to(module._streaming_state.device))

        self.apply(_set_exec_mask)

    def _plan_batch_stream_step(
        self,
        remaining: torch.Tensor,
        max_step_length: int,
        alignment: int,
    ) -> tuple[int, torch.Tensor]:
        positive_mask = remaining > 0
        if not bool(positive_mask.any().item()):
            raise RuntimeError("Cannot plan a streaming step when no samples remain.")

        if max_step_length > 0:
            full_step_mask = remaining >= max_step_length
            if bool(full_step_mask.any().item()):
                return max_step_length, full_step_mask

        positive_remaining = remaining[positive_mask]
        min_remaining = int(positive_remaining.min().item())

        if alignment > 1:
            aligned_step = (min_remaining // alignment) * alignment
            if aligned_step > 0:
                return aligned_step, remaining >= aligned_step
            return min_remaining, remaining == min_remaining

        step_length = min_remaining
        if max_step_length > 0:
            step_length = min(step_length, max_step_length)
        return step_length, remaining >= step_length

    def _infer_num_quantizers(
            self, codes_chunks: list[list[torch.Tensor]], requested_num_quantizers: int | None) -> int:
        if requested_num_quantizers is not None:
            return requested_num_quantizers
        for chunks_i in codes_chunks:
            if chunks_i:
                return int(chunks_i[0].shape[0])
        num_quantizers = getattr(self.quantizer, "num_quantizers", None)
        if num_quantizers is None:
            raise RuntimeError("Unable to infer the number of quantizers from empty streaming output.")
        return int(num_quantizers)

    def _infer_waveform_dtype(self, wav_chunks: list[list[torch.Tensor]]) -> torch.dtype:
        for chunks_i in wav_chunks:
            if chunks_i:
                return chunks_i[0].dtype
        return torch.float32

    @contextmanager
    def _codec_inference_autocast(self):
        device = next(self.parameters()).device
        if device.type == "cuda" and self.compute_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
                yield
        elif device.type == "cpu" and self.compute_dtype is torch.bfloat16:
            with torch.autocast(device_type="cpu", dtype=self.compute_dtype):
                yield
        else:
            yield

    def set_attention_implementation(self, attention_implementation: str) -> None:
        self.attention_implementation = attention_implementation
        for module in self.modules():
            if isinstance(module, MossAudioTokenizerProjectedTransformer):
                module.set_attention_implementation(attention_implementation)

    def set_compute_dtype(self, compute_dtype: str) -> None:
        self.compute_dtype_name = compute_dtype
        self.compute_dtype = resolve_compute_dtype(compute_dtype)
        self.config.compute_dtype = compute_dtype

    def set_codec_weight_dtype(self, codec_weight_dtype: str) -> None:
        codec_weight_dtype = canonicalize_codec_weight_dtype(codec_weight_dtype)
        weight_dtype = resolve_codec_weight_dtype(codec_weight_dtype)

        self.encoder.to(dtype=weight_dtype)
        self.decoder.to(dtype=weight_dtype)

        # Quantizer decode/encode intentionally disables autocast and builds fp32 intermediates.
        # Keeping it fp32 avoids fp32-input/bf16-bias mismatches and preserves codebook numerics.
        self.quantizer.to(dtype=torch.float32)

        self.codec_weight_dtype_name = codec_weight_dtype
        self.config.codec_weight_dtype = codec_weight_dtype
        if codec_weight_dtype != "fp32" and self.compute_dtype is None:
            self.set_compute_dtype(codec_weight_dtype)

    def get_codec_dtype_summary(self) -> dict[str, str]:

        def _first_param_dtype(module: nn.Module) -> str:
            for param in module.parameters():
                return str(param.dtype)
            return "no_params"

        return {
            "encoder": _first_param_dtype(self.encoder),
            "decoder": _first_param_dtype(self.decoder),
            "quantizer": _first_param_dtype(self.quantizer),
            "compute_dtype": self.compute_dtype_name,
            "codec_weight_dtype": self.codec_weight_dtype_name,
        }

    def _prepare_waveform_batch(
        self,
        wav_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(wav_list) == 0:
            raise ValueError("`wav_list` must contain at least one waveform.")

        device = wav_list[0].device
        dtype = wav_list[0].dtype
        batch_size = len(wav_list)
        lengths = torch.zeros(batch_size, device=device, dtype=torch.long)

        normalized_wavs: list[torch.Tensor] = []
        for i, wav in enumerate(wav_list):
            if self.number_channels == 1:
                if wav.dim() == 1:
                    wav_i = wav.unsqueeze(0)
                elif wav.dim() == 2 and wav.shape[0] == 1:
                    wav_i = wav
                else:
                    raise ValueError(
                        f"Expected wav_list[{i}] to have shape `(T,)` or `(1, T)` for a mono model, got {tuple(wav.shape)}."
                    )
            else:
                if wav.dim() != 2 or wav.shape[0] != self.number_channels:
                    raise ValueError(
                        f"Expected wav_list[{i}] to have shape `({self.number_channels}, T)`, got {tuple(wav.shape)}."
                    )
                wav_i = wav

            normalized_wavs.append(wav_i)
            lengths[i] = wav_i.shape[-1]

        max_length = int(lengths.max().item()) if batch_size > 0 else 0
        input_values = torch.zeros(batch_size, self.number_channels, max_length, device=device, dtype=dtype)
        for i, wav_i in enumerate(normalized_wavs):
            input_values[i, :, :wav_i.shape[-1]] = wav_i
        return input_values, lengths

    def _prepare_codes_batch(
        self,
        codes_list: list[torch.Tensor],
        num_quantizers: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if len(codes_list) == 0:
            raise ValueError("`codes_list` must contain at least one code tensor.")

        batch_size = len(codes_list)
        device = codes_list[0].device
        nqs = [codes.shape[0] for codes in codes_list]
        if num_quantizers is None:
            num_quantizers = nqs[0]
            if any(nq != num_quantizers for nq in nqs):
                raise ValueError(
                    "All elements in `codes_list` must have the same number of quantizers when `num_quantizers` is None. "
                    "Pass `num_quantizers=...` to decode a common prefix.")
        elif min(nqs) < num_quantizers:
            raise ValueError(
                "`num_quantizers` must be <= the number of quantizers for every element in `codes_list`. "
                f"Got num_quantizers={num_quantizers}, min(codes.shape[0])={min(nqs)}.")

        lengths = torch.tensor([codes.shape[-1] for codes in codes_list], device=device, dtype=torch.long)
        max_length = int(lengths.max().item()) if batch_size > 0 else 0
        audio_codes = torch.zeros(num_quantizers, batch_size, max_length, device=device, dtype=torch.long)

        for i, codes in enumerate(codes_list):
            codes_i = codes[:num_quantizers]
            audio_codes[:, i, :codes_i.shape[-1]] = codes_i
        return audio_codes, lengths, num_quantizers

    def _flatten_channels_for_codec(
        self,
        input_values: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_values.dim() != 3:
            raise ValueError(
                f"Expected `input_values` with shape `(B, C, T)`, got {tuple(input_values.shape)}.")
        if input_values.shape[1] != self.number_channels:
            raise ValueError(
                f"Expected `input_values.shape[1] == {self.number_channels}`, got {input_values.shape[1]}.")

        if input_values.shape[-1] % self.downsample_rate != 0:
            pad_length = self.downsample_rate - (input_values.shape[-1] % self.downsample_rate)
            input_values = F.pad(input_values, (0, pad_length))

        if self.number_channels > 1 and self.enable_channel_interleave:
            input_values = input_values.transpose(1, 2).contiguous().view(input_values.shape[0], 1, -1)
            input_lengths = input_lengths * self.number_channels
        return input_values, input_lengths

    def _restore_channels_from_codec(
        self,
        output_values: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.number_channels == 1 or not self.enable_channel_interleave:
            return output_values.float(), output_lengths

        output_values = (
            output_values.squeeze(1).contiguous().view(
                output_values.shape[0], -1, self.number_channels).transpose(1, 2).contiguous().float())
        output_lengths = torch.div(output_lengths, self.number_channels, rounding_mode="floor")
        return output_values, output_lengths

    def _stack_hidden_states(
        self,
        hidden_chunks: list[list[torch.Tensor]],
        lengths: torch.Tensor,
    ) -> torch.Tensor | None:
        hidden_dim = None
        for chunks_i in hidden_chunks:
            if chunks_i:
                hidden_dim = chunks_i[0].shape[0]
                break
        if hidden_dim is None:
            return None

        batch_size = len(hidden_chunks)
        max_length = int(lengths.max().item()) if batch_size > 0 else 0
        device = lengths.device
        hidden_states = torch.zeros(batch_size, hidden_dim, max_length, device=device, dtype=torch.float32)
        for i, chunks_i in enumerate(hidden_chunks):
            if not chunks_i:
                continue
            hidden_i = torch.cat(chunks_i, dim=-1).float()
            hidden_states[i, :, :hidden_i.shape[-1]] = hidden_i
        return hidden_states

    @torch.no_grad()
    def _encode_frame(
        self,
        input_values: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        n_quantizers: int | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        if input_values.dim() == 1:
            input_values = input_values.view(1, 1, -1)
        elif input_values.dim() == 2:
            if self.number_channels == 1:
                input_values = input_values.unsqueeze(1)
            else:
                input_values = input_values.unsqueeze(0)

        batch_size, _, time = input_values.shape
        device = input_values.device
        if input_lengths is None:
            input_lengths = torch.full((batch_size, ), time, device=device, dtype=torch.long)

        input_values, input_lengths = self._flatten_channels_for_codec(input_values, input_lengths)

        with self._codec_inference_autocast():
            encoder_hidden_states, encoder_hidden_lengths = input_values, input_lengths
            for encoder_module in self.encoder:
                encoder_hidden_states, encoder_hidden_lengths = encoder_module(
                    encoder_hidden_states,
                    encoder_hidden_lengths,
                )

        quantizer = cast(MossAudioTokenizerResidualVQ | MossAudioTokenizerResidualLFQ, self.quantizer)
        _, audio_codes, audio_codes_lengths = quantizer(
            encoder_hidden_states.float(), encoder_hidden_lengths, n_quantizers)
        max_valid_length = int(audio_codes_lengths.max().item()) if audio_codes_lengths.numel() > 0 else 0
        audio_codes = audio_codes[:, :, :max_valid_length]
        encoder_hidden_states = encoder_hidden_states[:, :, :max_valid_length]

        return MossAudioTokenizerEncoderOutput(
            audio_codes=audio_codes,
            audio_codes_lengths=audio_codes_lengths,
            encoder_hidden_states=encoder_hidden_states.float(),
        )

    @torch.no_grad()
    def _decode_frame(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor | None = None,
    ) -> MossAudioTokenizerDecoderOutput:
        _, batch_size, time = codes.shape
        device = codes.device
        if codes_lengths is None:
            codes_lengths = torch.full((batch_size, ), time, device=device, dtype=torch.long)

        quantizer = cast(MossAudioTokenizerResidualVQ | MossAudioTokenizerResidualLFQ, self.quantizer)
        decoder_hidden_states = quantizer.decode_codes(codes).float()

        with self._codec_inference_autocast():
            audio, audio_lengths = decoder_hidden_states, codes_lengths
            for decoder_module in self.decoder:
                audio, audio_lengths = decoder_module(audio, audio_lengths)

        audio, audio_lengths = self._restore_channels_from_codec(audio, audio_lengths)
        return MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=audio_lengths)

    @torch.no_grad()
    def batch_encode(
        self,
        wav_list: list[torch.Tensor],
        num_quantizers: int | None = None,
        chunk_duration: float | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        input_values, input_lengths = self._prepare_waveform_batch(wav_list)
        batch_size = len(wav_list)
        device = input_values.device

        if chunk_duration is None:
            return self._encode_frame(input_values, input_lengths, n_quantizers=num_quantizers)

        if chunk_duration <= 0:
            raise ValueError("`chunk_duration` must be > 0 when provided.")

        chunk_length = int(round(chunk_duration * self.sampling_rate))
        if chunk_length <= 0:
            raise ValueError("`chunk_duration` is too small and results in chunk_length <= 0.")
        if chunk_length % self.downsample_rate != 0:
            raise ValueError(
                "`chunk_duration * config.sampling_rate` must be divisible by `config.downsample_rate`. "
                f"Got chunk_length={chunk_length}, downsample_rate={self.downsample_rate}.")

        cursors = torch.zeros_like(input_lengths)
        codes_chunks: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        hidden_chunks: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        with self.streaming(batch_size=batch_size):
            while bool((cursors < input_lengths).any().item()):
                remaining = input_lengths - cursors
                step_length, active_mask = self._plan_batch_stream_step(
                    remaining=remaining,
                    max_step_length=chunk_length,
                    alignment=self.downsample_rate,
                )
                x_step = torch.zeros(
                    batch_size,
                    self.number_channels,
                    step_length,
                    device=device,
                    dtype=input_values.dtype,
                )
                input_lengths_step = torch.zeros(batch_size, device=device, dtype=torch.long)
                active_indices = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()

                for i in active_indices:
                    start = int(cursors[i].item())
                    end = start + step_length
                    x_step[i] = input_values[i, :, start:end]
                    input_lengths_step[i] = step_length

                self._set_streaming_exec_mask(active_mask)
                result = self._encode_frame(x_step, input_lengths_step, n_quantizers=num_quantizers)
                assert result.audio_codes is not None
                assert result.audio_codes_lengths is not None

                for i in active_indices:
                    codes_length_i = int(result.audio_codes_lengths[i].item())
                    if codes_length_i > 0:
                        codes_chunks[i].append(result.audio_codes[:, i, :codes_length_i].clone())
                        if result.encoder_hidden_states is not None:
                            hidden_chunks[i].append(
                                result.encoder_hidden_states[i, :, :codes_length_i].clone())
                    cursors[i] += step_length

        num_quantizers_used = self._infer_num_quantizers(codes_chunks, num_quantizers)
        empty_codes = torch.empty((num_quantizers_used, 0), device=device, dtype=torch.long)
        codes_list = [
            torch.cat(chunks_i, dim=-1) if chunks_i else empty_codes.clone() for chunks_i in codes_chunks
        ]
        audio_codes, audio_codes_lengths, _ = self._prepare_codes_batch(
            codes_list, num_quantizers=num_quantizers_used)
        encoder_hidden_states = self._stack_hidden_states(hidden_chunks, audio_codes_lengths)
        return MossAudioTokenizerEncoderOutput(
            audio_codes=audio_codes,
            audio_codes_lengths=audio_codes_lengths,
            encoder_hidden_states=encoder_hidden_states,
        )

    @torch.no_grad()
    def batch_decode(
        self,
        codes_list: list[torch.Tensor],
        num_quantizers: int | None = None,
        chunk_duration: float | None = None,
    ) -> MossAudioTokenizerDecoderOutput:
        audio_codes, audio_codes_lengths, num_quantizers_used = self._prepare_codes_batch(
            codes_list,
            num_quantizers=num_quantizers,
        )
        batch_size = len(codes_list)
        device = audio_codes.device

        if chunk_duration is None:
            return self._decode_frame(audio_codes, audio_codes_lengths)

        if chunk_duration <= 0:
            raise ValueError("`chunk_duration` must be > 0 when provided.")

        chunk_length = int(round(chunk_duration * self.sampling_rate))
        if chunk_length <= 0:
            raise ValueError("`chunk_duration` is too small and results in chunk_length <= 0.")
        if chunk_length % self.downsample_rate != 0:
            raise ValueError(
                "`chunk_duration * config.sampling_rate` must be divisible by `config.downsample_rate`. "
                f"Got chunk_length={chunk_length}, downsample_rate={self.downsample_rate}.")

        chunk_frame_length = chunk_length // self.downsample_rate
        cursors = torch.zeros_like(audio_codes_lengths)
        wav_chunks: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        with self.streaming(batch_size=batch_size):
            while bool((cursors < audio_codes_lengths).any().item()):
                remaining = audio_codes_lengths - cursors
                step_frames, active_mask = self._plan_batch_stream_step(
                    remaining=remaining,
                    max_step_length=chunk_frame_length,
                    alignment=1,
                )
                codes_step = torch.zeros(
                    num_quantizers_used,
                    batch_size,
                    step_frames,
                    device=device,
                    dtype=torch.long,
                )
                codes_lengths_step = torch.zeros(batch_size, device=device, dtype=torch.long)
                active_indices = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()

                for i in active_indices:
                    start = int(cursors[i].item())
                    end = start + step_frames
                    codes_step[:, i, :] = audio_codes[:, i, start:end]
                    codes_lengths_step[i] = step_frames

                self._set_streaming_exec_mask(active_mask)
                result = self._decode_frame(codes_step, codes_lengths_step)
                assert result.audio is not None
                assert result.audio_lengths is not None

                for i in active_indices:
                    audio_length_i = int(result.audio_lengths[i].item())
                    if audio_length_i > 0:
                        wav_chunks[i].append(result.audio[i, :, :audio_length_i].clone())
                    cursors[i] += step_frames

        wav_dtype = self._infer_waveform_dtype(wav_chunks)
        audio_lengths = torch.tensor(
            [sum(chunk.shape[-1] for chunk in chunks_i) for chunks_i in wav_chunks],
            device=device,
            dtype=torch.long,
        )
        max_audio_length = int(audio_lengths.max().item()) if batch_size > 0 else 0
        audio = torch.zeros(
            batch_size, self.number_channels, max_audio_length, device=device, dtype=wav_dtype)
        for i, chunks_i in enumerate(wav_chunks):
            if not chunks_i:
                continue
            wav_i = torch.cat(chunks_i, dim=-1)
            audio[i, :, :wav_i.shape[-1]] = wav_i
        return MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=audio_lengths)

    def encode(  # type: ignore[override]
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        num_quantizers: int | None = None,
        return_dict: bool | None = None,
        chunk_duration: float | None = None,
    ):
        """Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to indicate valid audio samples.
            num_quantizers (`int`, *optional*):
                Number of quantizers to use. By default, all quantizers are used.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            chunk_duration (`float`, *optional*):
                If provided, encode the input waveform in successive chunks of `chunk_duration` seconds while keeping a
                streaming KV cache for the causal transformers.

                `chunk_duration` must be <= `config.causal_transformer_context_duration`, and
                `chunk_duration * config.sampling_rate` must be divisible by `config.downsample_rate`.

        Returns:
            `MossAudioTokenizerEncoderOutput` or tuple containing audio codes and lengths.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        wav_list: list[torch.Tensor]
        if input_values.dim() == 1:
            wav_list = [input_values]
        elif input_values.dim() == 2:
            if self.number_channels == 1:
                lengths = (
                    padding_mask.sum(dim=-1).long() if padding_mask is not None and padding_mask.dim() == 2
                    else torch.full((input_values.shape[0], ),
                                    input_values.shape[-1],
                                    device=input_values.device,
                                    dtype=torch.long))
                wav_list = [input_values[i, :int(lengths[i].item())] for i in range(input_values.shape[0])]
            else:
                length = (
                    int(padding_mask.sum().item())
                    if padding_mask is not None and padding_mask.dim() == 1 else int(input_values.shape[-1]))
                wav_list = [input_values[:, :length]]
        elif input_values.dim() == 3:
            if input_values.shape[1] != self.number_channels:
                raise ValueError(
                    f"Expected `input_values.shape[1] == {self.number_channels}`, got {input_values.shape[1]}."
                )
            lengths = (
                padding_mask.sum(dim=-1).long() if padding_mask is not None else torch.full(
                    (input_values.shape[0], ),
                    input_values.shape[-1],
                    device=input_values.device,
                    dtype=torch.long))
            wav_list = [input_values[i, :, :int(lengths[i].item())] for i in range(input_values.shape[0])]
        else:
            raise ValueError(f"Unsupported `input_values` shape: {tuple(input_values.shape)}")

        encoder_output = self.batch_encode(
            wav_list, num_quantizers=num_quantizers, chunk_duration=chunk_duration)

        if not return_dict:
            assert encoder_output.audio_codes is not None
            assert encoder_output.audio_codes_lengths is not None
            return (
                cast(torch.Tensor, encoder_output.audio_codes),
                cast(torch.Tensor, encoder_output.audio_codes_lengths),
            )
        return encoder_output

    def decode(  # type: ignore[override]
        self,
        audio_codes: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
        chunk_duration: float | None = None,
        num_quantizers: int | None = None,
    ):
        """Decodes the given codes into an output audio waveform.

        Args:
            audio_codes (`torch.LongTensor` of shape `(num_quantizers, batch_size, sequence_length)`):
                Discrete code embeddings computed using `model.encode`.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to indicate valid code positions.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            chunk_duration (`float`, *optional*):
                If provided, decode the input codes in successive chunks of `chunk_duration` seconds while keeping a
                streaming KV cache for the causal transformers.

            num_quantizers (`int`, *optional*):
                Number of quantizers to use. By default, all quantizers in `audio_codes` are used.

                `chunk_duration` must be <= `config.causal_transformer_context_duration`, and
                `chunk_duration * config.sampling_rate` must be divisible by `config.downsample_rate`.

        Returns:
            `MossAudioTokenizerDecoderOutput` or tuple containing decoded audio.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if audio_codes.dim() == 2:
            codes_list = [audio_codes[:num_quantizers] if num_quantizers is not None else audio_codes]
        elif audio_codes.dim() == 3:
            if num_quantizers is not None and num_quantizers > audio_codes.shape[0]:
                raise ValueError(
                    f"`num_quantizers` ({num_quantizers}) must be <= audio_codes.shape[0] ({audio_codes.shape[0]})."
                )
            codes_lengths = (
                padding_mask.sum(dim=-1).long() if padding_mask is not None else torch.full(
                    (audio_codes.shape[1], ),
                    audio_codes.shape[-1],
                    device=audio_codes.device,
                    dtype=torch.long))
            codes_list = []
            for index in range(audio_codes.shape[1]):
                stop = int(codes_lengths[index].item())
                item = audio_codes[:, index, :stop]
                if num_quantizers is not None:
                    item = item[:num_quantizers]
                codes_list.append(item)
        else:
            raise ValueError(f"Unsupported `audio_codes` shape: {tuple(audio_codes.shape)}")

        decoder_output = self.batch_decode(
            codes_list, num_quantizers=num_quantizers, chunk_duration=chunk_duration)

        if not return_dict:
            assert decoder_output.audio is not None
            return (cast(torch.Tensor, decoder_output.audio), )
        return decoder_output

    @auto_docstring
    def forward(
        self,
        input_values: torch.FloatTensor | None = None,
        padding_mask: torch.BoolTensor | None = None,
        audio_codes: torch.Tensor | None = None,
        num_quantizers: int | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor
               | None] | MossAudioTokenizerOutput:  # type: ignore[override]
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
            Raw audio input converted to Float.
        padding_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid computing on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_codes (`torch.LongTensor` of shape `(num_quantizers, batch_size, sequence_length)`, *optional*):
            Discrete code embeddings computed using `model.encode`.
        num_quantizers (`int`, *optional*):
            Number of quantizers (codebooks) to use. By default, all quantizers are used.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Examples:

        ```python
        >>> import torch
        >>> from voicehub.models.mosstts.native import MossAudioTokenizerV2Model

        >>> model = MossAudioTokenizerV2Model.from_pretrained(
        ...     "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
        ... )

        >>> # Create dummy audio input
        >>> audio = torch.randn(1, 2, 48000)  # 1 second of audio at 48kHz stereo

        >>> outputs = model(input_values=audio)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        output_audio_codes: torch.Tensor | None = None
        output_audio_codes_lengths: torch.Tensor | None = None
        output_audio: torch.Tensor | None = None
        output_audio_lengths: torch.Tensor | None = None
        decoded_from_encoded_codes = False

        # Encode if input_values provided
        if input_values is not None:
            encoder_output = self.encode(input_values, padding_mask, num_quantizers, return_dict=True)
            encoder_output = cast(MossAudioTokenizerEncoderOutput, encoder_output)
            output_audio_codes = encoder_output.audio_codes
            output_audio_codes_lengths = encoder_output.audio_codes_lengths

            # If codes not provided separately, use encoded codes for decoding
            if audio_codes is None:
                audio_codes = output_audio_codes
                decoded_from_encoded_codes = True

        # Decode if codes available
        if audio_codes is not None:
            # If we're decoding the codes we just produced, use the computed lengths so we don't decode padded garbage.
            if decoded_from_encoded_codes and output_audio_codes_lengths is not None:
                decoder_output = self._decode_frame(audio_codes, output_audio_codes_lengths)
            else:
                decoder_output = self.decode(
                    audio_codes,
                    padding_mask=padding_mask,
                    return_dict=True,
                    num_quantizers=num_quantizers,
                )
                decoder_output = cast(MossAudioTokenizerDecoderOutput, decoder_output)
            output_audio = decoder_output.audio
            output_audio_lengths = decoder_output.audio_lengths

        if not return_dict:
            return (output_audio_codes, output_audio, output_audio_lengths)

        return MossAudioTokenizerOutput(
            audio=output_audio,
            audio_lengths=output_audio_lengths,
            audio_codes=output_audio_codes,
            audio_codes_lengths=output_audio_codes_lengths,
        )


MossAudioTokenizerV2Model = MossAudioTokenizerModel

__all__ = [
    "MossAudioTokenizerModel",
    "MossAudioTokenizerPreTrainedModel",
    "MossAudioTokenizerV2Model",
]
