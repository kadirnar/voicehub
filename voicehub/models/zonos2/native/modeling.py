"""Differentiable PyTorch implementation of the ZONOS2 acoustic LM.

The module names and tensor shapes intentionally match the published
checkpoint.  Unlike the upstream serving graph, this implementation has
a normal autograd path for teacher-forced fine-tuning and a separate KV-
cache path for autoregressive inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicehub.models.zonos2.native.configuration import Zonos2ArchitectureConfig


class Zonos2RMSNorm(nn.Module):
    """RMS normalization with the checkpoint-compatible ``weight`` name."""

    def __init__(self, dimension: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = float(eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.rms_norm(
            hidden_states,
            (hidden_states.shape[-1], ),
            self.weight,
            self.eps,
        )


class TensorLinear(nn.Module):
    """Linear projection stored as ``[chunks, output, input]``."""

    def __init__(
        self,
        chunks: int,
        output_per_chunk: int,
        input_features: int,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(chunks, output_per_chunk, input_features))
        nn.init.kaiming_uniform_(self.weight.flatten(0, 1), a=5**0.5)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.linear(hidden_states, self.weight.flatten(0, 1))


class MultiEmbedding(nn.Module):
    """Sum one embedding table per audio codebook plus the text stream."""

    def __init__(self, config: Zonos2ArchitectureConfig) -> None:
        super().__init__()
        self.embedders = nn.ModuleList(
            [nn.Embedding(config.audio_vocab_size, config.dim)
             for _ in range(config.n_codebooks)] + [nn.Embedding(config.text_vocab + 1, config.dim)])

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 3:
            raise ValueError("ZONOS2 input IDs must have shape [batch, time, streams].")
        if input_ids.shape[-1] != len(self.embedders):
            raise ValueError(
                f"ZONOS2 expected {len(self.embedders)} token streams, "
                f"received {input_ids.shape[-1]}.")
        result = self.embedders[0](input_ids[..., 0].long())
        for index, embedder in enumerate(self.embedders[1:], start=1):
            result = result + embedder(input_ids[..., index].long())
        return result


@dataclass(slots=True)
class Zonos2LayerKVCache:
    """Preallocated K/V state for one decoder layer."""

    key: Tensor
    value: Tensor
    length: int = 0

    @classmethod
    def allocate(
        cls,
        *,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Zonos2LayerKVCache:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("KV cache batch size and length must be positive.")
        shape = (
            batch_size,
            num_key_value_heads,
            max_length,
            head_dim,
        )
        return cls(
            key=torch.empty(shape, device=device, dtype=dtype),
            value=torch.empty(shape, device=device, dtype=dtype),
        )

    @property
    def capacity(self) -> int:
        return self.key.shape[2]

    def reset(self) -> None:
        self.length = 0


def _interleaved_rope(
    hidden_states: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> Tensor:
    pairs = hidden_states.reshape(
        *hidden_states.shape[:-1],
        hidden_states.shape[-1] // 2,
        2,
    )
    even, odd = pairs.unbind(dim=-1)
    rotated = torch.stack(
        (even * cosine - odd * sine, odd * cosine + even * sine),
        dim=-1,
    )
    return rotated.flatten(-2)


class Zonos2Attention(nn.Module):
    """Grouped-query causal attention with QK norm and headwise gating."""

    def __init__(self, config: Zonos2ArchitectureConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.max_seqlen = config.max_seqlen
        self.rope_theta = config.rope_theta
        self.wq = nn.Linear(
            config.dim,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.wkv = TensorLinear(
            2,
            self.num_key_value_heads * self.head_dim,
            config.dim,
        )
        self.wo = nn.Linear(
            self.num_heads * self.head_dim,
            config.dim,
            bias=False,
        )
        self.temp = nn.Parameter(torch.ones(1, self.num_heads, 1))
        self.gater = nn.Linear(config.dim, self.num_heads, bias=False)
        self.register_buffer(
            "_rope_inverse_frequency",
            self._inverse_frequency(device=None),
            persistent=False,
        )

    def _inverse_frequency(
        self,
        *,
        device: torch.device | str | None,
    ) -> Tensor:
        exponents = torch.arange(
            0,
            self.head_dim,
            2,
            dtype=torch.float32,
            device=device,
        ) / self.head_dim
        return 1.0 / torch.pow(self.rope_theta, exponents)

    def materialize_buffers(self, device: torch.device | str) -> None:
        self._rope_inverse_frequency = self._inverse_frequency(device=device)

    def _project(
        self,
        hidden_states: Tensor,
        *,
        position_offset: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, query_length, _ = hidden_states.shape
        query = self.wq(hidden_states).view(
            batch,
            query_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key_value = self.wkv(hidden_states).view(
            batch,
            query_length,
            2,
            self.num_key_value_heads,
            self.head_dim,
        )
        key = key_value[:, :, 0].transpose(1, 2)
        value = key_value[:, :, 1].transpose(1, 2)
        query = F.rms_norm(query, (self.head_dim, ), eps=1e-6)
        query = query * self.temp.abs().to(
            device=query.device,
            dtype=query.dtype,
        ).unsqueeze(-1)
        key = F.rms_norm(key, (self.head_dim, ), eps=1e-6)

        positions = torch.arange(
            position_offset,
            position_offset + query_length,
            dtype=torch.float32,
            device=query.device,
        )
        inverse = self._rope_inverse_frequency.to(device=query.device)
        angles = torch.outer(positions, inverse)
        cosine = angles.cos().to(dtype=query.dtype).view(1, 1, query_length, self.head_dim // 2)
        sine = angles.sin().to(dtype=query.dtype).view(1, 1, query_length, self.head_dim // 2)
        query = _interleaved_rope(query, cosine, sine)
        key = _interleaved_rope(key, cosine, sine)
        gate = torch.sigmoid(self.gater(hidden_states)).transpose(1, 2)
        return query, key, value, gate

    def _expanded_key_value(
        self,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        groups = self.num_heads // self.num_key_value_heads
        if groups == 1:
            return key, value
        return (
            key.repeat_interleave(groups, dim=1),
            value.repeat_interleave(groups, dim=1),
        )

    @staticmethod
    def _causal_mask(
        *,
        batch_size: int,
        query_length: int,
        key_length: int,
        position_offset: int,
        attention_mask: Tensor | None,
        device: torch.device,
    ) -> Tensor:
        query_positions = (torch.arange(query_length, device=device) + position_offset)
        key_positions = torch.arange(key_length, device=device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        allowed = allowed.view(1, 1, query_length, key_length)
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, key_length):
                raise ValueError("ZONOS2 attention mask must have shape "
                                 f"[{batch_size}, {key_length}].")
            allowed = allowed & attention_mask.to(
                device=device,
                dtype=torch.bool,
            ).view(batch_size, 1, 1, key_length)
        return allowed

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cache: Zonos2LayerKVCache | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, query_length, _ = hidden_states.shape
        position_offset = 0 if cache is None else cache.length
        if position_offset + query_length > self.max_seqlen:
            raise ValueError("ZONOS2 attention sequence exceeds configured maximum "
                             f"{self.max_seqlen}.")
        query, key, value, gate = self._project(
            hidden_states,
            position_offset=position_offset,
        )
        if cache is not None:
            if cache.key.shape[0] != batch_size:
                raise ValueError("ZONOS2 KV cache batch size does not match input.")
            end = position_offset + query_length
            if end > cache.capacity:
                raise ValueError(f"ZONOS2 KV cache capacity is {cache.capacity}, need {end}.")
            cache.key[:, :, position_offset:end].copy_(key)
            cache.value[:, :, position_offset:end].copy_(value)
            cache.length = end
            key = cache.key[:, :, :end]
            value = cache.value[:, :, :end]
        key, value = self._expanded_key_value(key, value)
        key_length = key.shape[2]

        # An explicit boolean mask handles full-sequence training, padded
        # batches, and multi-token cache continuation with the same semantics.
        causal_mask = self._causal_mask(
            batch_size=batch_size,
            query_length=query_length,
            key_length=key_length,
            position_offset=position_offset,
            attention_mask=attention_mask,
            device=query.device,
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended * gate.unsqueeze(-1)
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            query_length,
            self.num_heads * self.head_dim,
        )
        return self.wo(attended)


class Zonos2DenseFeedForward(nn.Module):
    """Dense SwiGLU projection used outside the MoE layer interval."""

    def __init__(self, config: Zonos2ArchitectureConfig) -> None:
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.w_in = TensorLinear(2, self.intermediate_size, config.dim)
        self.w_out = nn.Linear(
            self.intermediate_size,
            config.dim,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        projected = self.w_in(hidden_states)
        up, gate = projected.split(self.intermediate_size, dim=-1)
        return self.w_out(up * F.silu(gate))


class Zonos2SonicExperts(nn.Module):
    """Packed Sonic MoE weights with differentiable grouped dispatch."""

    def __init__(self, config: Zonos2ArchitectureConfig) -> None:
        super().__init__()
        self.num_experts = config.moe_n_experts
        self.intermediate_size = config.intermediate_size
        self.w13 = nn.Parameter(torch.empty(
            self.num_experts,
            2 * self.intermediate_size,
            config.dim,
        ))
        self.w2 = nn.Parameter(torch.empty(
            self.num_experts,
            config.dim,
            self.intermediate_size,
        ))
        for expert in range(self.num_experts):
            nn.init.kaiming_uniform_(self.w13[expert], a=5**0.5)
            nn.init.kaiming_uniform_(self.w2[expert], a=5**0.5)

    def _expert(self, hidden_states: Tensor, expert_index: int) -> Tensor:
        projected = F.linear(hidden_states, self.w13[expert_index])
        # Sonic stores gate/up rows interleaved in its published checkpoint.
        gate = projected[..., 0::2]
        up = projected[..., 1::2]
        return F.linear(F.silu(gate) * up, self.w2[expert_index])

    def forward(
        self,
        hidden_states: Tensor,
        route_probabilities: Tensor,
        expert_indices: Tensor,
    ) -> Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("Sonic expert input must have shape [tokens, hidden].")
        top_k = expert_indices.shape[-1]
        flattened_experts = expert_indices.reshape(-1)
        flattened_weights = route_probabilities.reshape(-1)
        output = torch.zeros_like(hidden_states)
        for expert_index in range(self.num_experts):
            assignments = torch.nonzero(
                flattened_experts == expert_index,
                as_tuple=False,
            ).flatten()
            if assignments.numel() == 0:
                continue
            token_indices = torch.div(
                assignments,
                top_k,
                rounding_mode="floor",
            )
            selected = hidden_states.index_select(0, token_indices)
            expert_output = self._expert(selected, expert_index)
            weights = flattened_weights.index_select(0, assignments).to(dtype=expert_output.dtype)
            output = output.index_add(
                0,
                token_indices,
                expert_output * weights.unsqueeze(-1),
            )
        return output


class Zonos2Router(nn.Module):
    """Sonic router, including expert-decision alignment (EDA) state."""

    def __init__(
        self,
        config: Zonos2ArchitectureConfig,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.down_proj = nn.Linear(
            config.dim,
            config.moe_router_dim,
            bias=True,
        )
        self.router_mlp = nn.Sequential(
            nn.Linear(
                config.moe_router_dim,
                config.moe_router_dim,
                bias=True,
            ),
            nn.GELU(),
            nn.Linear(
                config.moe_router_dim,
                config.moe_router_dim,
                bias=True,
            ),
            nn.GELU(),
            nn.Linear(
                config.moe_router_dim,
                config.moe_n_experts,
                bias=False,
            ),
        )
        self.rmsnorm_eda = Zonos2RMSNorm(
            config.moe_router_dim,
            config.norm_eps,
        )
        self.use_eda = layer_index != config.moe_start_from_layer
        if self.use_eda:
            self.router_states_scale = nn.Parameter(torch.ones(config.moe_router_dim))
        else:
            self.register_parameter("router_states_scale", None)
        # This value affects discrete expert selection, not the differentiable
        # route probability. Preserve it in checkpoints but do not optimize it.
        self.balancing_biases = nn.Parameter(
            torch.zeros(config.moe_n_experts),
            requires_grad=False,
        )
        self.top_k = config.top_k_for_layer(layer_index)

    def forward(
        self,
        hidden_states: Tensor,
        previous_router_states: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        projected = self.down_proj(hidden_states)
        if self.use_eda and previous_router_states is not None:
            projected = (projected + previous_router_states * self.router_states_scale)
        next_router_states = projected
        probabilities = torch.softmax(
            self.router_mlp(self.rmsnorm_eda(projected)).float(),
            dim=-1,
        )
        with torch.no_grad():
            expert_indices = torch.topk(
                probabilities + self.balancing_biases.float(),
                self.top_k,
                dim=-1,
            ).indices
        route_probabilities = probabilities.gather(-1, expert_indices)
        return route_probabilities, expert_indices, next_router_states


class Zonos2MoEFeedForward(nn.Module):

    def __init__(
        self,
        config: Zonos2ArchitectureConfig,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.router = Zonos2Router(config, layer_index)
        self.experts = Zonos2SonicExperts(config)

    def forward(
        self,
        hidden_states: Tensor,
        router_states: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        original_shape = hidden_states.shape
        flattened = hidden_states.reshape(-1, original_shape[-1])
        flattened_previous = (
            None if router_states is None else router_states.reshape(-1, router_states.shape[-1]))
        route_probabilities, expert_indices, next_router_states = self.router(
            flattened,
            flattened_previous,
        )
        output = self.experts(
            flattened,
            route_probabilities,
            expert_indices,
        )
        return (
            output.view(original_shape),
            next_router_states.view(*original_shape[:-1], -1),
        )


class Zonos2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Zonos2ArchitectureConfig,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.attention = Zonos2Attention(config)
        self.attention_norm = Zonos2RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = Zonos2RMSNorm(config.dim, config.norm_eps)
        self.is_moe = config.is_moe_layer(layer_index)
        self.feed_forward: nn.Module = (
            Zonos2MoEFeedForward(config, layer_index) if self.is_moe else Zonos2DenseFeedForward(config))

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cache: Zonos2LayerKVCache | None,
        attention_mask: Tensor | None,
        router_states: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states),
            cache=cache,
            attention_mask=attention_mask,
        )
        normalized = self.ffn_norm(hidden_states)
        if self.is_moe:
            feed_forward, router_states = self.feed_forward(
                normalized,
                router_states,
            )
        else:
            feed_forward = self.feed_forward(normalized)
            router_states = None
        return hidden_states + feed_forward, router_states


@dataclass(slots=True)
class Zonos2ForCausalLMOutput:
    """Output shared by inference and teacher-forced fine-tuning."""

    logits: Tensor
    loss: Tensor | None = None
    per_codebook_loss: Tensor | None = None
    token_count: Tensor | None = None
    hidden_states: Tensor | None = None


class Zonos2ForCausalLM(nn.Module):
    """Native ZONOS2 transformer with full-sequence and cached execution."""

    def __init__(self, config: Zonos2ArchitectureConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.multi_embedder = MultiEmbedding(config)
        if config.speaker_enabled and config.speaker_lda_dim is not None:
            self.speaker_lda_projection: nn.Linear | None = nn.Linear(
                config.speaker_embedding_dim,
                config.speaker_lda_dim,
                bias=True,
            )
            speaker_input_size = config.speaker_lda_dim
        else:
            self.speaker_lda_projection = None
            speaker_input_size = config.speaker_embedding_dim
        self.speaker_projection: nn.Linear | None = (
            nn.Linear(speaker_input_size, config.dim, bias=True) if config.speaker_enabled else None)
        self.layers = nn.ModuleList(
            [Zonos2DecoderLayer(config, layer_index) for layer_index in range(config.n_layers)])
        self.out_norm = Zonos2RMSNorm(config.dim, config.norm_eps)
        self.multi_output = nn.Linear(
            config.dim,
            config.n_codebooks * config.audio_vocab_size,
            bias=False,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def materialize_runtime_buffers(
        self,
        device: torch.device | str,
    ) -> None:
        for layer in self.layers:
            layer.attention.materialize_buffers(device)

    def create_kv_cache(
        self,
        *,
        batch_size: int,
        max_length: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[Zonos2LayerKVCache]:
        if max_length > self.config.max_seqlen:
            raise ValueError(
                f"KV cache length {max_length} exceeds "
                f"max_seqlen={self.config.max_seqlen}.")
        cache_device = self.device if device is None else device
        cache_dtype = self.dtype if dtype is None else dtype
        return [
            Zonos2LayerKVCache.allocate(
                batch_size=batch_size,
                max_length=max_length,
                num_key_value_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                device=cache_device,
                dtype=cache_dtype,
            ) for _ in self.layers
        ]

    def _project_speaker(self, speaker_embedding: Tensor) -> Tensor:
        if self.speaker_projection is None:
            raise ValueError(
                "Speaker conditioning was supplied to a speaker-disabled "
                "ZONOS2 configuration.")
        projected = speaker_embedding
        if self.speaker_lda_projection is not None:
            projected = self.speaker_lda_projection(projected)
        return self.speaker_projection(projected)

    def _embed(
        self,
        input_ids: Tensor,
        *,
        speaker_embedding: Tensor | None,
        speaker_position: int | None,
    ) -> Tensor:
        hidden_states = self.multi_embedder(input_ids)
        if speaker_embedding is not None:
            if speaker_position is None:
                raise ValueError("`speaker_position` is required with a speaker embedding.")
            if not 0 <= speaker_position < hidden_states.shape[1]:
                raise ValueError("`speaker_position` is outside the input sequence.")
            speaker_embedding = speaker_embedding.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if speaker_embedding.ndim == 1:
                speaker_embedding = speaker_embedding.unsqueeze(0)
            if speaker_embedding.shape[0] not in (1, hidden_states.shape[0]):
                raise ValueError("Speaker embedding batch must be one or match input batch.")
            projected = self._project_speaker(speaker_embedding)
            if projected.shape[0] == 1 and hidden_states.shape[0] > 1:
                projected = projected.expand(hidden_states.shape[0], -1)
            position_mask = torch.zeros(
                hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
            position_mask[:, speaker_position] = True
            hidden_states = torch.where(
                position_mask.unsqueeze(-1),
                projected.unsqueeze(1),
                hidden_states,
            )
        return F.rms_norm(
            hidden_states,
            (hidden_states.shape[-1], ),
            weight=None,
            eps=self.config.norm_eps,
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        kv_cache: list[Zonos2LayerKVCache] | None = None,
        speaker_embedding: Tensor | None = None,
        speaker_position: int | None = None,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
        return_hidden_states: bool = False,
        **_: Any,
    ) -> Zonos2ForCausalLMOutput:
        if input_ids.dtype == torch.bool or input_ids.is_floating_point():
            raise TypeError("ZONOS2 input IDs must use an integer dtype.")
        if kv_cache is not None and len(kv_cache) != len(self.layers):
            raise ValueError(f"ZONOS2 requires {len(self.layers)} KV caches, "
                             f"received {len(kv_cache)}.")
        if kv_cache is not None and labels is not None:
            raise ValueError("Teacher-forced labels cannot be used with KV cache.")
        hidden_states = self._embed(
            input_ids,
            speaker_embedding=speaker_embedding,
            speaker_position=speaker_position,
        )
        router_states = None
        for index, layer in enumerate(self.layers):
            hidden_states, router_states = layer(
                hidden_states,
                cache=None if kv_cache is None else kv_cache[index],
                attention_mask=attention_mask,
                router_states=router_states,
            )
        normalized = self.out_norm(hidden_states)
        logits = self.multi_output(normalized).view(
            *normalized.shape[:-1],
            self.config.n_codebooks,
            self.config.audio_vocab_size,
        )
        if self.config.loss_softcap > 0:
            cap = self.config.loss_softcap
            logits = cap * torch.tanh(logits / cap)

        loss = per_codebook_loss = token_count = None
        if labels is not None:
            from voicehub.models.zonos2.native.objective import zonos2_causal_cross_entropy

            objective = zonos2_causal_cross_entropy(
                logits,
                labels,
                audio_pad_id=self.config.audio_pad_id,
                loss_mask=loss_mask,
            )
            loss = objective.loss
            per_codebook_loss = objective.per_codebook_loss
            token_count = objective.token_count
        return Zonos2ForCausalLMOutput(
            logits=logits,
            loss=loss,
            per_codebook_loss=per_codebook_loss,
            token_count=token_count,
            hidden_states=hidden_states if return_hidden_states else None,
        )


__all__ = [
    "MultiEmbedding",
    "TensorLinear",
    "Zonos2Attention",
    "Zonos2DecoderLayer",
    "Zonos2ForCausalLM",
    "Zonos2ForCausalLMOutput",
    "Zonos2LayerKVCache",
    "Zonos2MoEFeedForward",
    "Zonos2RMSNorm",
    "Zonos2Router",
    "Zonos2SonicExperts",
]
