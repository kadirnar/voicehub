"""PyTorch-native MiniCPM-4 decoder used by VoxCPM2."""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.voxcpm.native.configuration import VoxCPMTransformerConfig


def _rms_norm(hidden: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    source_dtype = hidden.dtype
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    return (hidden * torch.rsqrt(variance + epsilon)).to(source_dtype) * weight


class MiniCPMRMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        *,
        epsilon: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.variance_epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return _rms_norm(hidden_states, self.weight, self.variance_epsilon)


def _rotate_half(inputs: Tensor) -> Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    queries: Tensor,
    keys: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> tuple[Tensor, Tensor]:
    source_dtype = queries.dtype
    queries_f = queries.float()
    keys_f = keys.float()
    return (
        (queries_f * cosine + _rotate_half(queries_f) * sine).to(source_dtype),
        (keys_f * cosine + _rotate_half(keys_f) * sine).to(source_dtype),
    )


class MiniCPMLongRoPE(nn.Module):
    """Source-compatible MiniCPM long-context rotary embedding."""

    def __init__(
        self,
        config: VoxCPMTransformerConfig,
        *,
        device=None,
    ) -> None:
        super().__init__()
        self.dimension = config.head_dim
        self.base = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.short_factor = config.rope_scaling.short_factor
        self.long_factor = config.rope_scaling.long_factor
        self.original_max_position_embeddings = (config.rope_scaling.original_max_position_embeddings)
        scale = (self.max_position_embeddings / self.original_max_position_embeddings)
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        resolved_device = torch.device("cpu" if device is None else device)
        if resolved_device.type == "meta":
            inverse = torch.empty(
                self.dimension // 2,
                device=resolved_device,
            )
            cosine = torch.empty(
                self.max_position_embeddings,
                self.dimension,
                device=resolved_device,
            )
            sine = torch.empty_like(cosine)
        else:
            inverse = 1.0 / (
                self.base**(
                    torch.arange(
                        0,
                        self.dimension,
                        2,
                        device=resolved_device,
                        dtype=torch.float32,
                    ) / self.dimension))
            cosine, sine = self._make_cache(
                inverse,
                sequence_length=self.max_position_embeddings,
                device=resolved_device,
                dtype=torch.float32,
            )
        self.register_buffer("inv_freq", inverse, persistent=False)
        self.register_buffer("cos_cached", cosine, persistent=False)
        self.register_buffer("sin_cached", sine, persistent=False)

    def _make_cache(
        self,
        inverse: Tensor,
        *,
        sequence_length: int,
        device,
        dtype,
    ) -> tuple[Tensor, Tensor]:
        positions = torch.arange(
            sequence_length,
            device=device,
            dtype=inverse.dtype,
        )
        factors = torch.tensor(
            self.long_factor
            if sequence_length > self.original_max_position_embeddings else self.short_factor,
            device=device,
            dtype=torch.float32,
        )
        frequencies = torch.outer(
            positions,
            1.0 / factors,
        ) * inverse.to(
            device=device, dtype=dtype)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return (
            embedding.cos().to(dtype) * self.scaling_factor,
            embedding.sin().to(dtype) * self.scaling_factor,
        )

    def materialize(self, device) -> None:
        exponents = torch.arange(
            0,
            self.dimension,
            2,
            device=device,
            dtype=torch.float32,
        ) / self.dimension
        inverse = 1.0 / self.base**exponents
        cosine, sine = self._make_cache(
            inverse,
            sequence_length=self.max_position_embeddings,
            device=device,
            dtype=torch.float32,
        )
        self.inv_freq = inverse
        self.cos_cached = cosine
        self.sin_cached = sine

    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


class StaticKVCache(nn.Module):
    """Fixed cache kept outside the persistent checkpoint namespace."""

    def __init__(self) -> None:
        super().__init__()
        self.keys: list[Tensor] = []
        self.values: list[Tensor] = []
        self.position = 0

    def setup(
        self,
        config: VoxCPMTransformerConfig,
        *,
        batch_size: int,
        max_length: int,
        device,
        dtype,
    ) -> None:
        shape = (
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.head_dim,
        )
        self.keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.values = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)
        ]
        self.position = 0

    def clear(self) -> None:
        self.keys.clear()
        self.values.clear()
        self.position = 0

    def fill(self, values: list[tuple[Tensor, Tensor]]) -> None:
        if len(values) != len(self.keys):
            raise ValueError("MiniCPM cache layer count does not match prefill output.")
        for index, (keys, states) in enumerate(values):
            length = keys.shape[2]
            self.keys[index][..., :length, :].copy_(keys)
            self.values[index][..., :length, :].copy_(states)
        self.position = values[0][0].shape[2] if values else 0

    def layer(self, index: int) -> tuple[Tensor, Tensor]:
        if not self.keys:
            raise RuntimeError("MiniCPM generation cache has not been configured.")
        return self.keys[index], self.values[index]


class MiniCPMAttention(nn.Module):

    def __init__(
        self,
        config: VoxCPMTransformerConfig,
        layer_index: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def _attention(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        *,
        is_causal: bool = False,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        options = {
            "attn_mask": attention_mask,
            "is_causal": is_causal,
        }
        try:
            return functional.scaled_dot_product_attention(
                queries.contiguous(),
                keys.contiguous(),
                values.contiguous(),
                enable_gqa=True,
                **options,
            )
        except TypeError:
            keys = keys.repeat_interleave(
                self.num_key_value_groups,
                dim=1,
            )
            values = values.repeat_interleave(
                self.num_key_value_groups,
                dim=1,
            )
            return functional.scaled_dot_product_attention(
                queries.contiguous(),
                keys.contiguous(),
                values.contiguous(),
                **options,
            )

    def forward(
        self,
        hidden_states: Tensor,
        position_embedding: tuple[Tensor, Tensor] | None,
        *,
        is_causal: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, length, _ = hidden_states.shape
        queries = self.q_proj(hidden_states).view(
            batch,
            length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(
            batch,
            length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        values = self.v_proj(hidden_states).view(
            batch,
            length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        if position_embedding is not None:
            queries, keys = _apply_rotary(
                queries,
                keys,
                *position_embedding,
            )
        output = self._attention(
            queries,
            keys,
            values,
            is_causal=is_causal,
        )
        output = output.transpose(1, 2).reshape(
            batch,
            length,
            self.num_heads * self.head_dim,
        )
        return self.o_proj(output), (keys, values)

    def forward_step(
        self,
        hidden_states: Tensor,
        position_embedding: tuple[Tensor, Tensor] | None,
        position: int,
        cache: tuple[Tensor, Tensor],
    ) -> Tensor:
        batch = hidden_states.shape[0]
        queries = self.q_proj(hidden_states).view(
            batch,
            1,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(
            batch,
            1,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        values = self.v_proj(hidden_states).view(
            batch,
            1,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        if position_embedding is not None:
            queries, keys = _apply_rotary(
                queries,
                keys,
                *position_embedding,
            )
        key_cache, value_cache = cache
        key_cache[:, :, position:position + 1, :].copy_(keys)
        value_cache[:, :, position:position + 1, :].copy_(values)
        mask = (torch.arange(key_cache.shape[2], device=key_cache.device) <= position).view(1, 1, 1, -1)
        output = self._attention(
            queries,
            key_cache,
            value_cache,
            attention_mask=mask,
        )
        output = output.transpose(1, 2).reshape(
            batch,
            self.num_heads * self.head_dim,
        )
        return self.o_proj(output)


class MiniCPMMLP(nn.Module):

    def __init__(
        self,
        config: VoxCPMTransformerConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.act_fn = nn.SiLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))


class MiniCPMDecoderLayer(nn.Module):

    def __init__(
        self,
        config: VoxCPMTransformerConfig,
        layer_index: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.self_attn = MiniCPMAttention(
            config,
            layer_index,
            device=device,
            dtype=dtype,
        )
        self.mlp = MiniCPMMLP(config, device=device, dtype=dtype)
        self.input_layernorm = MiniCPMRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.post_attention_layernorm = MiniCPMRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.residual_scale = (
            config.scale_depth / math.sqrt(config.num_hidden_layers) if config.use_mup else 1.0)

    def forward(
        self,
        hidden_states: Tensor,
        position_embedding: tuple[Tensor, Tensor] | None,
        *,
        is_causal: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        residual = hidden_states
        attention, cache = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embedding,
            is_causal=is_causal,
        )
        hidden_states = residual + attention * self.residual_scale
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)) * self.residual_scale
        return hidden_states, cache

    def forward_step(
        self,
        hidden_states: Tensor,
        position_embedding: tuple[Tensor, Tensor] | None,
        position: int,
        cache: tuple[Tensor, Tensor],
    ) -> Tensor:
        residual = hidden_states
        hidden_states = residual + self.self_attn.forward_step(
            self.input_layernorm(hidden_states),
            position_embedding,
            position,
            cache,
        ) * self.residual_scale
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states)) * self.residual_scale


class MiniCPMModel(nn.Module):
    """Checkpoint-compatible MiniCPM-4 decoder."""

    def __init__(
        self,
        config: VoxCPMTransformerConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens: nn.Module
        if config.vocab_size:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                device=device,
                dtype=dtype,
            )
        else:
            self.embed_tokens = nn.Identity()
        self.layers = nn.ModuleList([
            MiniCPMDecoderLayer(
                config,
                layer_index,
                device=device,
                dtype=dtype,
            ) for layer_index in range(config.num_hidden_layers)
        ])
        self.norm = MiniCPMRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.rope_emb = (None if config.no_rope else MiniCPMLongRoPE(config, device=device))
        self.kv_cache = StaticKVCache()

    def forward(
        self,
        inputs_embeds: Tensor,
        *,
        is_causal: bool = True,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        position_embedding = None
        if self.rope_emb is not None:
            positions = torch.arange(
                inputs_embeds.shape[1],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            position_embedding = self.rope_emb(positions)
        hidden_states = inputs_embeds
        cache = []
        for layer in self.layers:
            hidden_states, layer_cache = layer(
                hidden_states,
                position_embedding,
                is_causal=is_causal,
            )
            cache.append(layer_cache)
        return self.norm(hidden_states), cache

    def setup_cache(
        self,
        *,
        batch_size: int,
        max_length: int,
        device,
        dtype,
    ) -> None:
        self.kv_cache.setup(
            self.config,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            dtype=dtype,
        )

    def forward_step(
        self,
        inputs_embeds: Tensor,
        position_id: int,
    ) -> Tensor:
        position_embedding = None
        if self.rope_emb is not None:
            position_embedding = self.rope_emb(
                torch.tensor(
                    [position_id],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                ))
        hidden_states = inputs_embeds
        for index, layer in enumerate(self.layers):
            hidden_states = layer.forward_step(
                hidden_states,
                position_embedding,
                position_id,
                self.kv_cache.layer(index),
            )
        return self.norm(hidden_states)

    def materialize_runtime_buffers(self, device) -> None:
        if self.rope_emb is not None:
            self.rope_emb.materialize(device)


def local_transformer_config(
    base: VoxCPMTransformerConfig,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_hidden_layers: int,
    kv_channels: int | None,
    no_rope: bool = False,
) -> VoxCPMTransformerConfig:
    """Create source-equivalent local encoder/DiT/RALM configs."""
    head_dim = (hidden_size // num_attention_heads if kv_channels is None else kv_channels)
    factors = tuple(1.0 for _ in range(head_dim // 2))
    rope = replace(
        base.rope_scaling,
        long_factor=factors,
        short_factor=factors,
    )
    return replace(
        base,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        kv_channels=kv_channels,
        vocab_size=0,
        no_rope=no_rope,
        rope_scaling=rope,
    )


__all__ = [
    "MiniCPMLongRoPE",
    "MiniCPMModel",
    "MiniCPMRMSNorm",
    "StaticKVCache",
    "local_transformer_config",
]
