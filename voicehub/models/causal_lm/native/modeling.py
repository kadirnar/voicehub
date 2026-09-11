"""VoiceHub-owned Granite, Llama, Qwen2, and Qwen3 decoder architectures.

The graph follows the official dense model semantics reviewed in
Transformers at revision ``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``:
pre-norm decoder blocks, RMSNorm, grouped-query attention, RoPE, and
gated SwiGLU MLPs.  Qwen2 projection biases and Qwen3 head-wise Q/K
normalization are represented explicitly.  Only PyTorch and VoiceHub
runtime components are used.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.generation.config import GenerationConfig
from voicehub.generation.engine import (
    AutoregressiveGenerator,
    GenerationOutput,
    GenerationStepInput,
    GenerationStepOutput,
)
from voicehub.generation.stopping import StoppingCriterion
from voicehub.models.causal_lm.native.configuration import (
    CausalLMConfig,
    GraniteConfig,
    LlamaConfig,
    Qwen2Config,
    Qwen3Config,
)
from voicehub.neural.cache import DynamicKVCache
from voicehub.neural.normalization import RMSNorm
from voicehub.neural.rotary import RotaryEmbedding, apply_rotary_embedding
from voicehub.objectives.sequence import sequence_cross_entropy


@dataclass(frozen=True)
class CausalLMModelOutput:
    """Hidden states and cache returned by the native decoder backbone."""

    last_hidden_state: Tensor
    past_key_values: DynamicKVCache | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass(frozen=True)
class CausalLMOutput:
    """Logits, optional shifted-token loss, and decoder diagnostics."""

    logits: Tensor
    loss: Tensor | None = None
    past_key_values: DynamicKVCache | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


def _expand_key_values(value: Tensor, groups: int) -> Tensor:
    if groups == 1:
        return value
    batch_size, heads, time, head_dim = value.shape
    return (
        value[:, :, None, :, :].expand(batch_size, heads, groups, time,
                                       head_dim).reshape(batch_size, heads * groups, time, head_dim))


def _normalize_position_ids(
    position_ids: Tensor | None,
    *,
    attention_mask: Tensor | None,
    batch_size: int,
    query_length: int,
    past_length: int,
    max_position_embeddings: int,
    device: torch.device,
) -> Tensor:
    if position_ids is None:
        if attention_mask is not None and attention_mask.ndim == 2:
            positions = attention_mask.to(device=device, dtype=torch.long)
            positions = positions.cumsum(dim=-1).sub(1).clamp_min(0)
            position_ids = positions[:, -query_length:]
        else:
            position_ids = torch.arange(
                past_length,
                past_length + query_length,
                device=device,
            ).unsqueeze(0)
    if not isinstance(position_ids, Tensor):
        raise TypeError("`position_ids` must be a PyTorch tensor.")
    if (position_ids.dtype == torch.bool or position_ids.is_floating_point() or position_ids.is_complex()):
        raise TypeError("`position_ids` must use an integer dtype.")
    if position_ids.ndim == 1:
        if position_ids.shape[0] != query_length:
            raise ValueError("Rank-one `position_ids` must match the query length.")
        position_ids = position_ids.unsqueeze(0)
    if (position_ids.ndim != 2 or position_ids.shape[0] not in (1, batch_size) or
            position_ids.shape[1] != query_length):
        raise ValueError("`position_ids` must have shape [1|batch, query_length].")
    position_ids = position_ids.to(device=device, dtype=torch.long)
    if position_ids.shape[0] == 1 and batch_size != 1:
        position_ids = position_ids.expand(batch_size, -1)
    if (position_ids < 0).any():
        raise ValueError("`position_ids` cannot contain negative values.")
    if (position_ids.numel() and int(position_ids.max().item()) >= max_position_embeddings):
        raise ValueError(
            "`position_ids` exceed `max_position_embeddings`; scaled or "
            "dynamic RoPE must be represented by a future compatible config.")
    return position_ids


def _attention_bias(
    attention_mask: Tensor | None,
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    past_length: int,
    device: torch.device,
) -> Tensor:
    query_positions = (past_length + torch.arange(query_length, device=device))
    key_positions = torch.arange(key_length, device=device)
    allowed = key_positions.view(1, 1, 1, key_length) <= query_positions.view(
        1,
        1,
        query_length,
        1,
    )
    allowed = allowed.expand(batch_size, 1, query_length, key_length)
    additive: Tensor | None = None

    if attention_mask is not None:
        if not isinstance(attention_mask, Tensor):
            raise TypeError("`attention_mask` must be a PyTorch tensor.")
        if attention_mask.device != device:
            raise ValueError("`attention_mask` must be on the same device as model inputs.")
        if attention_mask.ndim == 2:
            if tuple(attention_mask.shape) != (batch_size, key_length):
                raise ValueError(
                    "Rank-two `attention_mask` must have shape "
                    f"{(batch_size, key_length)!r}; found "
                    f"{tuple(attention_mask.shape)!r}.")
            key_allowed = attention_mask.to(dtype=torch.bool)
            allowed = allowed & key_allowed[:, None, None, :]
        elif attention_mask.ndim == 4:
            if (attention_mask.shape[0] not in (1, batch_size) or attention_mask.shape[1] not in (1, ) or
                    attention_mask.shape[2] not in (1, query_length) or
                    attention_mask.shape[3] != key_length):
                raise ValueError(
                    "Rank-four `attention_mask` is not broadcast-compatible "
                    "with [batch, heads, query, key].")
            if attention_mask.dtype == torch.bool:
                allowed = allowed & attention_mask
            elif attention_mask.is_floating_point():
                additive = attention_mask.float()
            else:
                allowed = allowed & attention_mask.to(dtype=torch.bool)
        else:
            raise ValueError("`attention_mask` must have rank two or four.")

    # A padded query can otherwise receive an all-masked row and produce NaNs.
    # Allowing that query to attend to its own slot keeps the row finite while
    # the original key mask still prevents it from influencing real tokens.
    has_key = allowed.any(dim=-1, keepdim=True)
    if not bool(has_key.all()):
        fallback = torch.zeros_like(allowed)
        diagonal = (past_length + torch.arange(query_length, device=device))
        fallback.scatter_(
            -1,
            diagonal.view(1, 1, query_length, 1).expand(
                batch_size,
                1,
                query_length,
                1,
            ),
            True,
        )
        allowed = torch.where(has_key, allowed, fallback)

    bias = torch.zeros(
        allowed.shape,
        dtype=torch.float32,
        device=device,
    )
    bias.masked_fill_(~allowed, torch.finfo(torch.float32).min)
    if additive is not None:
        if torch.isnan(additive).any() or torch.isposinf(additive).any():
            raise ValueError("An additive `attention_mask` cannot contain NaN or +inf.")
        bias = (bias + additive).clamp_min(torch.finfo(torch.float32).min)
    return bias


class CausalSelfAttention(nn.Module):
    """Family-aware grouped-query self-attention with an explicit KV cache."""

    def __init__(
        self,
        config: CausalLMConfig,
        layer_index: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (config.num_attention_heads // config.num_key_value_heads)
        self.scaling = config.attention_multiplier
        self.attention_dropout = config.attention_dropout

        query_size = config.num_attention_heads * self.head_dim
        key_value_size = config.num_key_value_heads * self.head_dim
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        self.q_proj = nn.Linear(
            config.hidden_size,
            query_size,
            bias=config.qkv_bias,
            **factory_kwargs,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            key_value_size,
            bias=config.qkv_bias,
            **factory_kwargs,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            key_value_size,
            bias=config.qkv_bias,
            **factory_kwargs,
        )
        self.o_proj = nn.Linear(
            query_size,
            config.hidden_size,
            bias=config.attention_output_bias,
            **factory_kwargs,
        )
        self.q_norm = (
            RMSNorm(
                self.head_dim,
                epsilon=config.rms_norm_eps,
                **factory_kwargs,
            ) if config.uses_qk_norm else None)
        self.k_norm = (
            RMSNorm(
                self.head_dim,
                epsilon=config.rms_norm_eps,
                **factory_kwargs,
            ) if config.uses_qk_norm else None)
        self.rotary = RotaryEmbedding(
            self.head_dim,
            base=config.rope_theta,
            scaling=config.rope_scaling,
            device=device,
        )

    def _shape(self, value: Tensor, heads: int) -> Tensor:
        batch_size, time, _ = value.shape
        return value.view(
            batch_size,
            time,
            heads,
            self.head_dim,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        position_ids: Tensor,
        cache: DynamicKVCache | None,
        use_cache: bool,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None, DynamicKVCache | None]:
        batch_size, query_length, _ = hidden_states.shape
        query = self._shape(
            self.q_proj(hidden_states),
            self.num_attention_heads,
        )
        key = self._shape(
            self.k_proj(hidden_states),
            self.num_key_value_heads,
        )
        value = self._shape(
            self.v_proj(hidden_states),
            self.num_key_value_heads,
        )
        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        cosine, sine = self.rotary(position_ids, dtype=query.dtype)
        query, key = apply_rotary_embedding(
            query,
            key,
            cosine,
            sine,
        )

        existing = None if cache is None else cache.get(self.layer_index)
        past_length = 0 if existing is None else existing.sequence_length
        if use_cache:
            if cache is None:
                cache = DynamicKVCache()
            entry = cache.update(self.layer_index, key, value)
            key, value = entry.key, entry.value
        elif existing is not None:
            key = torch.cat((existing.key, key), dim=-2)
            value = torch.cat((existing.value, value), dim=-2)

        key_length = key.shape[-2]
        if attention_mask is None and not output_attentions:
            # An unpadded prefill is the ordinary top-left causal case, while
            # a cached one-token decode may attend to every stored key.  Keep
            # grouped K/V heads compact so PyTorch can select its fused GQA
            # SDPA kernel instead of materializing repeated heads.
            chunk_mask = None
            is_causal = past_length == 0 and query_length > 1
            if past_length and query_length > 1:
                query_positions = (past_length + torch.arange(
                    query_length,
                    device=hidden_states.device,
                ))
                key_positions = torch.arange(
                    key_length,
                    device=hidden_states.device,
                )
                chunk_mask = (
                    key_positions.view(1, 1, 1, key_length) <= query_positions.view(1, 1, query_length, 1))
            attended = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=chunk_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scaling,
                enable_gqa=self.num_key_value_groups != 1,
            )
            probabilities = None
        else:
            # Preserve the explicit float32 path for arbitrary padding and
            # additive masks, and whenever callers request attention weights.
            key = _expand_key_values(key, self.num_key_value_groups)
            value = _expand_key_values(value, self.num_key_value_groups)
            bias = _attention_bias(
                attention_mask,
                batch_size=batch_size,
                query_length=query_length,
                key_length=key_length,
                past_length=past_length,
                device=hidden_states.device,
            )
            scores = torch.matmul(
                query.float(),
                key.float().transpose(-1, -2),
            )
            scores.mul_(self.scaling)
            scores.add_(bias)
            probabilities = functional.softmax(scores, dim=-1)
            probabilities = probabilities.to(dtype=query.dtype)
            probabilities = functional.dropout(
                probabilities,
                p=self.attention_dropout if self.training else 0.0,
                training=self.training,
            )
            attended = torch.matmul(probabilities, value)
        attended = (
            attended.transpose(1, 2).contiguous().view(
                batch_size,
                query_length,
                self.num_attention_heads * self.head_dim,
            ))
        output = self.o_proj(attended)
        return (
            output,
            probabilities if output_attentions else None,
            cache if use_cache else None,
        )


class GatedMLP(nn.Module):
    """SwiGLU feed-forward block with official checkpoint tensor names."""

    def __init__(
        self,
        config: CausalLMConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        bias = config.mlp_bias if config.model_type in {"granite", "llama"} else False
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=bias,
            **factory_kwargs,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=bias,
            **factory_kwargs,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=bias,
            **factory_kwargs,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class CausalLMDecoderLayer(nn.Module):
    """Pre-norm decoder layer shared by the dense model families."""

    def __init__(
        self,
        config: CausalLMConfig,
        layer_index: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        self.self_attn = CausalSelfAttention(
            config,
            layer_index,
            **factory_kwargs,
        )
        self.mlp = GatedMLP(config, **factory_kwargs)
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            **factory_kwargs,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            **factory_kwargs,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        position_ids: Tensor,
        cache: DynamicKVCache | None,
        use_cache: bool,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None, DynamicKVCache | None]:
        residual = hidden_states
        attention_output, attention, cache = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attention_output * self.config.residual_multiplier
        residual = hidden_states
        hidden_states = (
            residual +
            self.mlp(self.post_attention_layernorm(hidden_states)) * self.config.residual_multiplier)
        return hidden_states, attention, cache


class CausalLMModel(nn.Module):
    """Embedding, decoder stack, and final norm for a native dense LM."""

    def __init__(
        self,
        config: CausalLMConfig | dict[str, Any],
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = CausalLMConfig.coerce(config)
        self.padding_idx = self.config.pad_token_id
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            self.padding_idx,
            **factory_kwargs,
        )
        self.layers = nn.ModuleList(
            CausalLMDecoderLayer(
                self.config,
                layer_index,
                **factory_kwargs,
            ) for layer_index in range(self.config.num_hidden_layers))
        self.norm = RMSNorm(
            self.config.hidden_size,
            epsilon=self.config.rms_norm_eps,
            **factory_kwargs,
        )
        self.gradient_checkpointing = False
        if initialize:
            self.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        inputs_embeds: Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> CausalLMModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")
        if input_ids is not None:
            if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
                raise ValueError("`input_ids` must have shape [batch, sequence].")
            if (input_ids.dtype == torch.bool or input_ids.is_floating_point() or input_ids.is_complex()):
                raise TypeError("`input_ids` must use an integer dtype.")
            if input_ids.numel() == 0:
                raise ValueError("`input_ids` cannot be empty.")
            if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
                raise ValueError("An input token ID is outside the vocabulary.")
            hidden_states = self.embed_tokens(input_ids)
        else:
            if (not isinstance(inputs_embeds, Tensor) or inputs_embeds.ndim != 3 or
                    inputs_embeds.shape[-1] != self.config.hidden_size):
                raise ValueError(
                    "`inputs_embeds` must have shape "
                    f"[batch, sequence, {self.config.hidden_size}].")
            if inputs_embeds.shape[0] == 0 or inputs_embeds.shape[1] == 0:
                raise ValueError("`inputs_embeds` cannot have an empty batch or sequence.")
            hidden_states = inputs_embeds
        hidden_states = hidden_states * self.config.embedding_multiplier
        batch_size, query_length, _ = hidden_states.shape

        if past_key_values is not None and not isinstance(
                past_key_values,
                DynamicKVCache,
        ):
            raise TypeError("`past_key_values` must be a DynamicKVCache.")
        past_length = (0 if past_key_values is None else past_key_values.sequence_length())
        if past_key_values is not None:
            lengths = {
                past_key_values.sequence_length(layer_index)
                for layer_index in range(self.config.num_hidden_layers) if layer_index in past_key_values
            }
            if lengths and lengths != {past_length}:
                raise ValueError(
                    "Every populated decoder cache layer must have the same "
                    "sequence length.")

        use_cache = self.config.use_cache if use_cache is None else use_cache
        if not isinstance(use_cache, bool):
            raise TypeError("`use_cache` must be a boolean.")
        if self.gradient_checkpointing and self.training and use_cache:
            raise ValueError(
                "Gradient checkpointing and KV-cache mutation cannot be used "
                "in the same training forward pass.")
        if use_cache and past_key_values is None:
            past_key_values = DynamicKVCache()

        position_ids = _normalize_position_ids(
            position_ids,
            attention_mask=attention_mask,
            batch_size=batch_size,
            query_length=query_length,
            past_length=past_length,
            max_position_embeddings=self.config.max_position_embeddings,
            device=hidden_states.device,
        )

        hidden_history: list[Tensor] | None = ([] if output_hidden_states else None)
        attention_history: list[Tensor] | None = ([] if output_attentions else None)
        for layer in self.layers:
            if hidden_history is not None:
                hidden_history.append(hidden_states)
            if (self.gradient_checkpointing and self.training and not output_attentions):

                def custom_forward(
                    states: Tensor,
                    current_layer: CausalLMDecoderLayer = layer,
                ) -> Tensor:
                    result, _, _ = current_layer(
                        states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        cache=None,
                        use_cache=False,
                        output_attentions=False,
                    )
                    return result

                hidden_states = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    use_reentrant=False,
                )
                attention = None
            else:
                hidden_states, attention, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            if attention_history is not None:
                if attention is None:
                    raise RuntimeError("A decoder layer did not return requested attentions.")
                attention_history.append(attention)
        hidden_states = self.norm(hidden_states)
        if hidden_history is not None:
            hidden_history.append(hidden_states)
        return CausalLMModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=(tuple(hidden_history) if hidden_history is not None else None),
            attentions=(tuple(attention_history) if attention_history is not None else None),
        )


class CausalLMForCausalLM(nn.Module):
    """Trainable decoder plus vocabulary projection and native generation."""

    expected_model_type: str | None = None

    def __init__(
        self,
        config: CausalLMConfig | dict[str, Any],
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = CausalLMConfig.coerce(config)
        if (self.expected_model_type is not None and self.config.model_type != self.expected_model_type):
            raise ValueError(
                f"{type(self).__name__} requires model_type "
                f"{self.expected_model_type!r}, found "
                f"{self.config.model_type!r}.")
        self.model = CausalLMModel(
            self.config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if initialize:
            nn.init.normal_(
                self.lm_head.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
        if self.config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.model.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ) -> CausalLMOutput:
        if (labels is not None and isinstance(past_key_values, DynamicKVCache) and
                past_key_values.sequence_length()):
            raise ValueError("Causal-LM loss cannot be computed from a partial cached "
                             "sequence.")
        if labels is not None and use_cache is None:
            use_cache = False
        decoder_output = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = (self.lm_head(decoder_output.last_hidden_state) / self.config.logits_scaling).float()
        loss = None
        if labels is not None:
            if not isinstance(labels, Tensor) or labels.ndim != 2:
                raise ValueError("`labels` must have shape [batch, sequence].")
            if tuple(labels.shape) != tuple(logits.shape[:2]):
                raise ValueError("`labels` must match the input batch and sequence length.")
            if labels.shape[1] < 2:
                raise ValueError("Causal-LM loss requires a sequence of at least two tokens.")
            loss_mask = None
            if attention_mask is not None:
                if attention_mask.ndim != 2:
                    raise ValueError("Loss masking requires a rank-two `attention_mask`.")
                loss_mask = attention_mask[:, 1:]
            loss = sequence_cross_entropy(
                logits[:, :-1, :],
                labels[:, 1:],
                attention_mask=loss_mask,
                ignore_index=ignore_index,
                label_smoothing=label_smoothing,
            )
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=decoder_output.past_key_values,
            hidden_states=decoder_output.hidden_states,
            attentions=decoder_output.attentions,
        )

    def generate(
            self,
            input_ids: Tensor,
            *,
            attention_mask: Tensor | None = None,
            generation_config: GenerationConfig | None = None,
            stopping_criteria: Sequence[StoppingCriterion] = (),
    ) -> GenerationOutput:
        """Run VoiceHub's model-neutral cache-aware generation engine."""
        if (attention_mask is not None and (not isinstance(attention_mask, Tensor) or
                                            tuple(attention_mask.shape) != tuple(input_ids.shape))):
            raise ValueError("`attention_mask` must have the same shape as `input_ids`.")
        config = generation_config or GenerationConfig(
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
            use_cache=self.config.use_cache,
        )
        if not isinstance(config, GenerationConfig):
            raise TypeError("`generation_config` must be a VoiceHub GenerationConfig.")
        updates: dict[str, Any] = {}
        if config.eos_token_id is None and self.config.eos_token_id is not None:
            updates["eos_token_id"] = self.config.eos_token_id
        if config.pad_token_id is None and self.config.pad_token_id is not None:
            updates["pad_token_id"] = self.config.pad_token_id
        if updates:
            config = config.with_updates(**updates)
        prompt_mask = (None if attention_mask is None else attention_mask.to(device=input_ids.device))
        # TTS adapters commonly provide an explicit all-ones mask for their
        # unbatched prompt.  Collapse it once at request setup so every decoder
        # layer and generated token can use fused, mask-free SDPA.
        if prompt_mask is not None and bool(prompt_mask.all()):
            prompt_mask = None
        prompt_length = input_ids.shape[1]

        def decoder_step(step: GenerationStepInput) -> GenerationStepOutput:
            past_length = (step.cache.sequence_length() if isinstance(step.cache, DynamicKVCache) else 0)
            key_length = past_length + step.token_ids.shape[1]
            if key_length < prompt_length:
                raise RuntimeError("Decoder cache length is shorter than the prompt mask.")
            generated = key_length - prompt_length
            step_mask = prompt_mask
            if generated and prompt_mask is not None:
                step_mask = torch.cat(
                    (
                        prompt_mask,
                        torch.ones(
                            prompt_mask.shape[0],
                            generated,
                            dtype=prompt_mask.dtype,
                            device=prompt_mask.device,
                        ),
                    ),
                    dim=-1,
                )
            output = self(
                step.token_ids,
                attention_mask=step_mask,
                past_key_values=step.cache,
                use_cache=step.use_cache,
            )
            return GenerationStepOutput(
                logits=output.logits,
                cache=output.past_key_values,
            )

        return AutoregressiveGenerator().generate(
            decoder_step,
            input_ids,
            config,
            stopping_criteria=stopping_criteria,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        """Write a native/Hugging-Face-compatible local Safetensors
        artifact."""
        from voicehub.checkpointing import save_safetensors

        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        config_path = target / "config.json"
        config_path.write_text(
            json.dumps(
                self.config.to_dict(),
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        state = dict(self.state_dict())
        if self.config.tie_word_embeddings:
            state.pop("lm_head.weight", None)
        save_safetensors(
            state,
            target / "model.safetensors",
            metadata={
                "format": "pt",
                "architecture": self.config.model_type,
                "producer": "voicehub",
            },
        )
        return target

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        device: Any = "cpu",
        dtype: torch.dtype | None = None,
        strict: bool = True,
    ) -> CausalLMForCausalLM:
        """Strictly load one local single- or multi-shard HF artifact."""
        from voicehub.models.causal_lm.native.checkpoint import (
            HuggingFaceCausalLMCheckpointAdapter,
            open_causal_lm_tensor_source,
        )

        root = Path(directory).expanduser().resolve()
        config_path = root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Causal-LM config was not found: {config_path}")
        try:
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not parse causal-LM config {config_path}: {error}.") from error
        config = CausalLMConfig.from_dict(raw_config)
        model = cls(
            config,
            initialize=False,
            device=device,
            dtype=dtype,
        )
        with open_causal_lm_tensor_source(root) as source:
            HuggingFaceCausalLMCheckpointAdapter().load_streaming(
                model,
                source,
                config.to_dict(),
                strict=strict,
            )
        if config.tie_word_embeddings:
            model.tie_weights()
        return model


class LlamaForCausalLM(CausalLMForCausalLM):
    expected_model_type = "llama"


class GraniteForCausalLM(CausalLMForCausalLM):
    expected_model_type = "granite"


class Qwen2ForCausalLM(CausalLMForCausalLM):
    expected_model_type = "qwen2"


class Qwen3ForCausalLM(CausalLMForCausalLM):
    expected_model_type = "qwen3"


LlamaModel = CausalLMModel
GraniteModel = CausalLMModel
Qwen2Model = CausalLMModel
Qwen3Model = CausalLMModel

__all__ = [
    "CausalLMDecoderLayer",
    "CausalLMForCausalLM",
    "CausalLMModel",
    "CausalLMModelOutput",
    "CausalLMOutput",
    "CausalSelfAttention",
    "GatedMLP",
    "GraniteConfig",
    "GraniteForCausalLM",
    "GraniteModel",
    "LlamaConfig",
    "LlamaForCausalLM",
    "LlamaModel",
    "Qwen2Config",
    "Qwen2ForCausalLM",
    "Qwen2Model",
    "Qwen3Config",
    "Qwen3ForCausalLM",
    "Qwen3Model",
]
