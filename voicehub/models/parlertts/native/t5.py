"""PyTorch-only T5 encoder used by the released Parler-TTS checkpoints.

The module hierarchy intentionally matches ``T5EncoderModel`` so
official Safetensors keys load without a lossy name translation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicehub.models.parlertts.native.configuration import T5EncoderConfig


def gelu_new(value: Tensor) -> Tensor:
    """Exact tanh GELU selected by the FLAN-T5 checkpoint."""
    coefficient = math.sqrt(2.0 / math.pi)
    return 0.5 * value * (1.0 + torch.tanh(coefficient * (value + 0.044715 * value.pow(3))))


class T5LayerNorm(nn.Module):
    """T5 RMS normalization, which deliberately does not subtract the mean."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in {torch.float16, torch.bfloat16}:
            normalized = normalized.to(self.weight.dtype)
        return self.weight * normalized


class T5DenseGatedActDense(nn.Module):
    """Gated feed-forward projection from FLAN-T5."""

    def __init__(self, config: T5EncoderConfig) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.activation = gelu_new if config.dense_act_fn == "gelu_new" else F.gelu

    def forward(self, hidden_states: Tensor) -> Tensor:
        activated = self.activation(self.wi_0(hidden_states))
        hidden_gated = activated * self.wi_1(hidden_states)
        hidden_gated = self.dropout(hidden_gated)
        if hidden_gated.dtype != self.wo.weight.dtype:
            hidden_gated = hidden_gated.to(self.wo.weight.dtype)
        return self.wo(hidden_gated)


class T5LayerFF(nn.Module):

    def __init__(self, config: T5EncoderConfig) -> None:
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(config)
        self.layer_norm = T5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: Tensor) -> Tensor:
        forwarded = self.DenseReluDense(self.layer_norm(hidden_states))
        return hidden_states + self.dropout(forwarded)


class T5Attention(nn.Module):
    """T5 self-attention including the published relative-position buckets."""

    def __init__(
        self,
        config: T5EncoderConfig,
        *,
        has_relative_attention_bias: bool = False,
        attention_implementation: str = "eager",
    ) -> None:
        super().__init__()
        if attention_implementation not in {"eager", "sdpa"}:
            raise ValueError("T5 attention implementation must be 'eager' or 'sdpa'.")
        self.attention_implementation = attention_implementation
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = (config.relative_attention_num_buckets)
        self.relative_attention_max_distance = (config.relative_attention_max_distance)
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets,
                self.n_heads,
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: Tensor,
        *,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> Tensor:
        relative_buckets: Tensor | int = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets = ((relative_position > 0).to(torch.long) * num_buckets)
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.minimum(
                relative_position,
                torch.zeros_like(relative_position),
            )
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) *
            (num_buckets - max_exact)).to(torch.long)
        relative_if_large = torch.minimum(
            relative_if_large,
            torch.full_like(relative_if_large, num_buckets - 1),
        )
        return relative_buckets + torch.where(
            is_small,
            relative_position,
            relative_if_large,
        )

    def compute_bias(
        self,
        query_length: int,
        key_length: int,
        *,
        device: torch.device,
    ) -> Tensor:
        context = torch.arange(query_length, device=device)[:, None]
        memory = torch.arange(key_length, device=device)[None, :]
        buckets = self._relative_position_bucket(
            memory - context,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(buckets)
        return values.permute(2, 0, 1).unsqueeze(0)

    def _shape(self, states: Tensor) -> Tensor:
        batch_size = states.shape[0]
        return states.view(
            batch_size,
            -1,
            self.n_heads,
            self.key_value_proj_dim,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        mask: Tensor | None = None,
        position_bias: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        query_states = self._shape(self.q(hidden_states))
        key_states = self._shape(self.k(hidden_states))
        value_states = self._shape(self.v(hidden_states))
        query_length = hidden_states.shape[1]
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(
                    query_length,
                    key_states.shape[2],
                    device=query_states.device,
                )
            else:
                position_bias = torch.zeros(
                    1,
                    self.n_heads,
                    query_length,
                    key_states.shape[2],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
            if mask is not None:
                position_bias = position_bias + mask
        if (self.attention_implementation == "sdpa" and not output_attentions):
            attended = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=position_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=1.0,
            )
            attended = attended.transpose(1, 2).contiguous().view(
                hidden_states.shape[0],
                query_length,
                self.inner_dim,
            )
            return self.o(attended), position_bias, None
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        scores = scores + position_bias
        attention = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        attention = F.dropout(
            attention,
            p=self.dropout,
            training=self.training,
        )
        attended = torch.matmul(attention, value_states)
        attended = attended.transpose(1, 2).contiguous().view(
            hidden_states.shape[0],
            query_length,
            self.inner_dim,
        )
        output = self.o(attended)
        return output, position_bias, attention if output_attentions else None


class T5LayerSelfAttention(nn.Module):

    def __init__(
        self,
        config: T5EncoderConfig,
        *,
        has_relative_attention_bias: bool = False,
        attention_implementation: str = "eager",
    ) -> None:
        super().__init__()
        self.SelfAttention = T5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            attention_implementation=attention_implementation,
        )
        self.layer_norm = T5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor,
        position_bias: Tensor | None,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        attention, position_bias, weights = self.SelfAttention(
            self.layer_norm(hidden_states),
            mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        return (
            hidden_states + self.dropout(attention),
            position_bias,
            weights,
        )


class T5Block(nn.Module):

    def __init__(
        self,
        config: T5EncoderConfig,
        *,
        has_relative_attention_bias: bool = False,
        attention_implementation: str = "eager",
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList((
            T5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias,
                attention_implementation=attention_implementation,
            ),
            T5LayerFF(config),
        ))

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor,
        position_bias: Tensor | None,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        hidden_states, position_bias, attention = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias, attention


class T5Stack(nn.Module):

    def __init__(
        self,
        config: T5EncoderConfig,
        embed_tokens: nn.Embedding,
        *,
        attention_implementation: str = "eager",
    ) -> None:
        super().__init__()
        self.config = config
        # Keep the shared embedding registered only at model root, exactly as
        # Hugging Face does for tied module serialization.
        object.__setattr__(self, "embed_tokens", embed_tokens)
        self.block = nn.ModuleList(
            T5Block(
                config,
                has_relative_attention_bias=index == 0,
                attention_implementation=attention_implementation,
            ) for index in range(config.num_layers))
        self.final_layer_norm = T5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        output_attentions: bool = False,
    ) -> T5EncoderOutput:
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise ValueError("T5 `input_ids` must have shape [batch, sequence].")
        if input_ids.dtype == torch.bool or input_ids.is_floating_point():
            raise TypeError("T5 `input_ids` must use an integer dtype.")
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        batch_size, sequence_length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=input_ids.device,
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError("T5 attention mask must match `input_ids`.")
        minimum = torch.finfo(hidden_states.dtype).min
        additive_mask = (~attention_mask.to(dtype=torch.bool))[:, None, None, :].to(
            hidden_states.dtype) * minimum
        position_bias = None
        attentions: list[Tensor] = []
        for block in self.block:
            hidden_states, position_bias, attention = block(
                hidden_states,
                attention_mask=additive_mask,
                position_bias=position_bias,
                output_attentions=output_attentions,
            )
            if attention is not None:
                attentions.append(attention)
        hidden_states = self.dropout(self.final_layer_norm(hidden_states))
        return T5EncoderOutput(
            last_hidden_state=hidden_states,
            attentions=tuple(attentions),
        )


@dataclass(frozen=True, slots=True)
class T5EncoderOutput:
    last_hidden_state: Tensor
    attentions: tuple[Tensor, ...] = ()

    def __getitem__(self, index: int) -> Tensor | tuple[Tensor, ...]:
        values = (self.last_hidden_state, self.attentions)
        return values[index]


class NativeT5EncoderModel(nn.Module):
    """T5 encoder graph with the official checkpoint namespace."""

    def __init__(
        self,
        config: T5EncoderConfig,
        *,
        attention_implementation: str = "eager",
    ) -> None:
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )
        self.encoder = T5Stack(
            config,
            self.shared,
            attention_implementation=attention_implementation,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        output_attentions: bool = False,
    ) -> T5EncoderOutput:
        return self.encoder(
            input_ids,
            attention_mask,
            output_attentions=output_attentions,
        )


__all__ = [
    "NativeT5EncoderModel",
    "T5Attention",
    "T5EncoderOutput",
    "T5LayerNorm",
    "gelu_new",
]
