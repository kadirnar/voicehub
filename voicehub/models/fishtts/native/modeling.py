"""VoiceHub-native Fish Speech S2 semantic transformer.

This module is a direct PyTorch implementation of the released Dual-AR graph:

* a 36-layer slow transformer predicts text and semantic-codebook tokens;
* a 4-layer fast transformer predicts all ten codec codebooks at each
  semantic position;
* text and codec embeddings are combined only at semantic-token positions.

The implementation deliberately retains the checkpoint's fused QKV projection
and parameter names.  No provider model class or Transformers runtime is
involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from voicehub.models.fishtts.native.configuration import FishS2Config, FishTransformerConfig
from voicehub.optimization.protocols import OptimizationCompileTarget


def _causal_attention_mask(
    query_positions: Tensor,
    key_length: int,
    *,
    device: torch.device,
) -> Tensor:
    key_positions = torch.arange(key_length, device=device)
    return key_positions.view(1, 1, 1, -1) <= query_positions.view(
        1,
        1,
        -1,
        1,
    )


def _rotary_frequencies(
    positions: Tensor,
    dimension: int,
    *,
    base: float,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    inverse = 1.0 / (
        base**(torch.arange(
            0,
            dimension,
            2,
            device=positions.device,
            dtype=torch.float32,
        ) / dimension))
    angles = positions.to(torch.float32).unsqueeze(-1) * inverse
    return (
        angles.cos().to(dtype=dtype),
        angles.sin().to(dtype=dtype),
    )


def _apply_rotary(
    values: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> Tensor:
    source_dtype = values.dtype
    pairs = values.float().reshape(*values.shape[:-1], -1, 2)
    cosine = cosine.float().view(1, values.shape[1], 1, -1)
    sine = sine.float().view(1, values.shape[1], 1, -1)
    real = pairs[..., 0]
    imaginary = pairs[..., 1]
    rotated = torch.stack(
        (
            real * cosine - imaginary * sine,
            imaginary * cosine + real * sine,
        ),
        dim=-1,
    )
    return rotated.flatten(-2).to(dtype=source_dtype)


class FishRMSNorm(nn.Module):
    """RMS normalization with the source graph's float32 accumulation."""

    def __init__(self, hidden_size: int, *, epsilon: float) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return normalized.to(dtype=hidden_states.dtype) * self.weight


class FishKVCache(nn.Module):
    """Non-persistent serving cache; never part of exported checkpoints."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_heads: int,
        sequence_length: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        shape = (batch_size, num_heads, sequence_length, head_dim)
        self.register_buffer(
            "key",
            torch.zeros(shape, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "value",
            torch.zeros(shape, device=device, dtype=dtype),
            persistent=False,
        )

    def update(
        self,
        positions: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if positions.ndim != 1 or positions.shape[0] != key.shape[2]:
            raise ValueError("Fish cache positions must have one entry per query token.")
        if key.shape != value.shape:
            raise ValueError("Fish key and value cache updates must align.")
        if key.shape[0] > self.key.shape[0]:
            raise ValueError("Fish cache batch capacity was exceeded.")
        if int(positions.max().item()) >= self.key.shape[2]:
            raise ValueError("Fish cache sequence capacity was exceeded.")
        self.key[:key.shape[0], :, positions] = key
        self.value[:value.shape[0], :, positions] = value
        return (
            self.key[:key.shape[0]],
            self.value[:value.shape[0]],
        )

    def clear(self) -> None:
        self.key.zero_()
        self.value.zero_()


class FishAttention(nn.Module):
    """Fused-QKV grouped-query attention used by both AR stacks."""

    def __init__(
        self,
        config: FishTransformerConfig,
        *,
        manual_attention: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.manual_attention = manual_attention
        query_width = config.num_attention_heads * config.head_dim
        key_value_width = config.num_key_value_heads * config.head_dim
        self.wqkv = nn.Linear(
            config.hidden_size,
            query_width + 2 * key_value_width,
            bias=config.attention_qkv_bias,
        )
        self.wo = nn.Linear(
            query_width,
            config.hidden_size,
            bias=config.attention_o_bias,
        )
        if config.attention_qk_norm:
            self.q_norm = FishRMSNorm(
                config.head_dim,
                epsilon=config.rms_norm_eps,
            )
            self.k_norm = FishRMSNorm(
                config.head_dim,
                epsilon=config.rms_norm_eps,
            )
        self.kv_cache: FishKVCache | None = None

    def _manual_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(query.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(
                ~mask,
                torch.finfo(scores.dtype).min,
            )
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        if self.config.dropout and self.training:
            weights = F.dropout(
                weights,
                p=float(self.config.dropout),
                training=True,
            )
        return torch.matmul(weights, value)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        positions: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query_width = (self.config.num_attention_heads * self.config.head_dim)
        key_value_width = (self.config.num_key_value_heads * self.config.head_dim)
        query, key, value = self.wqkv(hidden_states).split(
            (query_width, key_value_width, key_value_width),
            dim=-1,
        )
        query = query.view(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.config.head_dim,
        )
        key = key.view(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        value = value.view(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        if self.config.attention_qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        cosine, sine = _rotary_frequencies(
            positions,
            self.config.head_dim,
            base=self.config.rope_theta,
            dtype=query.dtype,
        )
        query = _apply_rotary(query, cosine, sine).transpose(1, 2)
        key = _apply_rotary(key, cosine, sine).transpose(1, 2)
        value = value.transpose(1, 2)
        if self.kv_cache is not None:
            key, value = self.kv_cache.update(
                positions,
                key,
                value,
            )
            if attention_mask is None:
                attention_mask = _causal_attention_mask(
                    positions,
                    key.shape[-2],
                    device=query.device,
                )
        repeat = (self.config.num_attention_heads // self.config.num_key_value_heads)
        if repeat != 1:
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        if self.manual_attention:
            attended = self._manual_attention(
                query,
                key,
                value,
                attention_mask,
            )
        else:
            is_causal = attention_mask is None and self.kv_cache is None
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=(float(self.config.dropout) if self.training else 0.0),
                is_causal=is_causal,
            )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            query_width,
        )
        return self.wo(attended)


class FishFeedForward(nn.Module):

    def __init__(self, config: FishTransformerConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.w3 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.w2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class FishTransformerBlock(nn.Module):

    def __init__(
        self,
        config: FishTransformerConfig,
        *,
        manual_attention: bool,
    ) -> None:
        super().__init__()
        self.attention = FishAttention(
            config,
            manual_attention=manual_attention,
        )
        self.feed_forward = FishFeedForward(config)
        self.ffn_norm = FishRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )
        self.attention_norm = FishRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states),
            positions=positions,
            attention_mask=attention_mask,
        )
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


@dataclass(slots=True)
class FishSemanticOutput:
    """Slow-token and fast-codebook logits returned by training."""

    token_logits: Tensor
    codebook_logits: Tensor
    hidden_states: Tensor

    @property
    def logits(self) -> Tensor:
        return self.token_logits


@dataclass(slots=True)
class FishSlowOutput:
    logits: Tensor
    hidden_states: Tensor


class FishS2ForConditionalGeneration(nn.Module):
    """Exact differentiable Fish S2 semantic topology."""

    def __init__(
        self,
        config: FishS2Config,
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(config, FishS2Config):
            raise TypeError("`config` must be a FishS2Config.")
        self.config = config
        slow = config.text
        fast = config.audio_decoder
        self.embeddings = nn.Embedding(slow.vocab_size, slow.hidden_size)
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks,
            slow.hidden_size,
        )
        self.layers = nn.ModuleList(
            FishTransformerBlock(slow, manual_attention=False) for _ in range(slow.num_hidden_layers))
        self.norm = FishRMSNorm(
            slow.hidden_size,
            epsilon=slow.rms_norm_eps,
        )
        if not slow.tie_word_embeddings:
            self.output = nn.Linear(
                slow.hidden_size,
                slow.vocab_size,
                bias=False,
            )
        self.fast_embeddings = nn.Embedding(
            config.codebook_size,
            fast.hidden_size,
        )
        self.fast_layers = nn.ModuleList(
            FishTransformerBlock(fast, manual_attention=True) for _ in range(fast.num_hidden_layers))
        self.fast_norm = FishRMSNorm(
            fast.hidden_size,
            epsilon=fast.rms_norm_eps,
        )
        self.fast_output = nn.Linear(
            fast.hidden_size,
            config.codebook_size,
            bias=False,
        )
        self.max_batch_size = -1
        self.max_sequence_length = -1
        if initialize:
            self.apply(self._initialize_weights)

    @property
    def max_seq_len(self) -> int:
        return self.config.text.max_position_embeddings

    def _initialize_weights(self, module: nn.Module) -> None:
        standard_deviation = self.config.text.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0,
                std=standard_deviation,
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def setup_caches(
        self,
        *,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype | None = None,
    ) -> None:
        if (isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size <= 0):
            raise ValueError("`max_batch_size` must be a positive integer.")
        if (isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len <= 0 or
                max_seq_len > self.max_seq_len):
            raise ValueError("`max_seq_len` must be positive and no larger than the "
                             "configured context.")
        parameter = next(self.parameters())
        cache_dtype = parameter.dtype if dtype is None else dtype
        device = parameter.device
        if device.type == "meta":
            raise RuntimeError("Fish caches cannot be created before weights are materialized.")
        self.clear_caches()
        for block in self.layers:
            block.attention.kv_cache = FishKVCache(
                batch_size=max_batch_size,
                num_heads=self.config.text.num_key_value_heads,
                sequence_length=max_seq_len,
                head_dim=self.config.text.head_dim,
                device=device,
                dtype=cache_dtype,
            )
        for block in self.fast_layers:
            block.attention.kv_cache = FishKVCache(
                batch_size=max_batch_size,
                num_heads=self.config.audio_decoder.num_key_value_heads,
                sequence_length=self.config.num_codebooks,
                head_dim=self.config.audio_decoder.head_dim,
                device=device,
                dtype=cache_dtype,
            )
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_seq_len

    def clear_caches(self) -> None:
        for block in (*self.layers, *self.fast_layers):
            block.attention.kv_cache = None
        self.max_batch_size = -1
        self.max_sequence_length = -1

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Declare the Dual-AR boundaries used by each execution mode."""
        if mode == "inference":
            return (
                OptimizationCompileTarget(
                    "semantic.forward_generate",
                    self,
                    "forward_generate",
                ),
                OptimizationCompileTarget(
                    "semantic.forward_generate_fast",
                    self,
                    "forward_generate_fast",
                ),
            )
        if mode == "training":
            return (OptimizationCompileTarget(
                "semantic.forward",
                self,
                "forward",
            ), )
        raise ValueError("Fish compile targets require 'inference' or 'training' mode.")

    def reset_fast_caches(self) -> None:
        for block in self.fast_layers:
            cache = block.attention.kv_cache
            if cache is not None:
                cache.clear()

    def _validate_inputs(self, input_ids: Tensor) -> None:
        if input_ids.ndim != 3:
            raise ValueError("Fish inputs must have shape "
                             "[batch, num_codebooks + 1, time].")
        if input_ids.shape[1] != self.config.num_codebooks + 1:
            raise ValueError("Fish input channel count does not match the configured "
                             "codebooks.")
        if input_ids.shape[-1] > self.max_seq_len:
            raise ValueError("Fish input exceeds the configured context.")
        primary = input_ids[:, 0]
        if primary.numel() and (int(primary.min().item()) < 0 or
                                int(primary.max().item()) >= self.config.text.vocab_size):
            raise ValueError("Fish primary token ID is outside the vocabulary.")
        codebooks = input_ids[:, 1:]
        if codebooks.numel() and (int(codebooks.min().item()) < 0 or
                                  int(codebooks.max().item()) >= self.config.codebook_size):
            raise ValueError("Fish codec token ID is outside its codebook.")

    def embed(
        self,
        input_ids: Tensor,
        *,
        scale_semantic: bool,
    ) -> Tensor:
        self._validate_inputs(input_ids)
        primary = input_ids[:, 0]
        codebooks = input_ids[:, 1:].long()
        offsets = (
            torch.arange(
                self.config.num_codebooks,
                device=input_ids.device,
                dtype=codebooks.dtype,
            ) * self.config.codebook_size)
        codec_embeddings = self.codebook_embeddings(codebooks + offsets.view(1, -1, 1)).sum(dim=1)
        semantic_mask = primary.ge(self.config.semantic_begin_id) & primary.le(self.config.semantic_end_id)
        codec_embeddings = codec_embeddings.masked_fill(
            ~semantic_mask.unsqueeze(-1),
            0,
        )
        hidden_states = self.embeddings(primary) + codec_embeddings
        if scale_semantic and self.config.scale_codebook_embeddings:
            scale = math.sqrt(self.config.num_codebooks + 1)
            hidden_states = torch.where(
                semantic_mask.unsqueeze(-1),
                hidden_states / scale,
                hidden_states,
            )
        return hidden_states

    def _run_slow(
        self,
        hidden_states: Tensor,
        *,
        positions: Tensor,
        attention_mask: Tensor | None,
    ) -> FishSlowOutput:
        for layer in self.layers:
            if (self.config.text.gradient_checkpointing and self.training and
                    layer.attention.kv_cache is None):
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    positions,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    positions,
                    attention_mask,
                )
        normalized = self.norm(hidden_states)
        logits = (
            F.linear(normalized, self.embeddings.weight)
            if self.config.text.tie_word_embeddings else self.output(normalized))
        fast_input = (normalized if self.config.norm_fastlayer_input else hidden_states)
        return FishSlowOutput(logits=logits, hidden_states=fast_input)

    def forward_slow(
        self,
        input_ids: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
    ) -> FishSlowOutput:
        hidden_states = self.embed(input_ids, scale_semantic=False)
        sequence_length = input_ids.shape[-1]
        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        )
        attention_mask = None
        if key_padding_mask is not None:
            if (key_padding_mask.ndim != 2 or
                    tuple(key_padding_mask.shape) != (input_ids.shape[0], sequence_length)):
                raise ValueError("Fish key-padding mask must have shape [batch, time].")
            causal = _causal_attention_mask(
                positions,
                sequence_length,
                device=input_ids.device,
            )
            attention_mask = causal & ~key_padding_mask.bool().view(
                input_ids.shape[0],
                1,
                1,
                sequence_length,
            )
        return self._run_slow(
            hidden_states,
            positions=positions,
            attention_mask=attention_mask,
        )

    def _run_fast(self, hidden_states: Tensor) -> Tensor:
        sequence_length = hidden_states.shape[1]
        positions = torch.arange(
            sequence_length,
            device=hidden_states.device,
            dtype=torch.long,
        )
        mask = _causal_attention_mask(
            positions,
            sequence_length,
            device=hidden_states.device,
        )
        for layer in self.fast_layers:
            if (self.config.audio_decoder.gradient_checkpointing and self.training and
                    layer.attention.kv_cache is None):
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    positions,
                    mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, positions, mask)
        return self.fast_output(self.fast_norm(hidden_states))

    def forward(
        self,
        inp: Tensor | None = None,
        *,
        input_ids: Tensor | None = None,
        labels: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        **unused: Any,
    ) -> FishSemanticOutput:
        del unused
        if inp is not None and input_ids is not None:
            raise ValueError("Pass either `inp` or `input_ids`, not both.")
        values = input_ids if input_ids is not None else inp
        if values is None:
            raise ValueError("Fish forward requires `inp` or `input_ids`.")
        if labels is None:
            raise ValueError("Fish Dual-AR training forward requires aligned `labels`.")
        if labels.shape != values.shape:
            raise ValueError("Fish labels must have the same shape as inputs.")
        slow = self.forward_slow(
            values,
            key_padding_mask=key_padding_mask,
        )
        primary_labels = labels[:, 0]
        semantic_mask = primary_labels.ge(self.config.semantic_begin_id) & primary_labels.le(
            self.config.semantic_end_id)
        slow_hidden = slow.hidden_states[semantic_mask]
        if slow_hidden.shape[0] == 0:
            # Preserve a connected, finite graph while the objective rejects
            # a batch with no supervised semantic positions.
            slow_hidden = slow.hidden_states.reshape(
                -1,
                slow.hidden_states.shape[-1],
            )[:1]
            teacher_codes = torch.zeros(
                (
                    slow_hidden.shape[0],
                    self.config.num_codebooks - 1,
                ),
                device=values.device,
                dtype=torch.long,
            )
        else:
            semantic_codes = labels[:, 1:].permute(0, 2, 1)[semantic_mask]
            invalid_codes = semantic_codes.ne(-100) & (
                semantic_codes.lt(0)
                | semantic_codes.ge(self.config.codebook_size))
            if invalid_codes.any():
                raise ValueError("Fish supervised codebook labels are out of range.")
            teacher_codes = semantic_codes[:, :-1].masked_fill(
                semantic_codes[:, :-1].eq(-100),
                0,
            ).long()
        fast_input = torch.cat(
            (
                slow_hidden.unsqueeze(1),
                self.fast_embeddings(teacher_codes),
            ),
            dim=1,
        )
        codebook_logits = self._run_fast(fast_input)
        return FishSemanticOutput(
            token_logits=slow.logits,
            codebook_logits=codebook_logits,
            hidden_states=slow.hidden_states,
        )

    def forward_generate(
        self,
        input_ids: Tensor,
        positions: Tensor,
        *,
        return_all: bool = False,
    ) -> FishSlowOutput:
        if positions.ndim != 1 or positions.shape[0] != input_ids.shape[-1]:
            raise ValueError("Fish generation positions must align with input time.")
        hidden_states = self.embed(input_ids, scale_semantic=True)
        mask = _causal_attention_mask(
            positions,
            self.max_sequence_length,
            device=input_ids.device,
        )
        slow = self._run_slow(
            hidden_states,
            positions=positions,
            attention_mask=mask,
        )
        if return_all or slow.logits.shape[1] == 1:
            return slow
        return FishSlowOutput(
            logits=slow.logits[:, -1:],
            hidden_states=slow.hidden_states[:, -1:],
        )

    def forward_generate_fast(
        self,
        hidden_states: Tensor,
        positions: Tensor,
    ) -> Tensor:
        if positions.ndim != 1 or positions.shape[0] != hidden_states.shape[1]:
            raise ValueError("Fish fast-generation positions must align with input time.")
        mask = _causal_attention_mask(
            positions,
            self.config.num_codebooks,
            device=hidden_states.device,
        )
        for layer in self.fast_layers:
            hidden_states = layer(hidden_states, positions, mask)
        return self.fast_output(self.fast_norm(hidden_states))

    def save_pretrained(self, directory: str | Any) -> Any:
        from voicehub.models.fishtts.native.checkpoint import save_fish_semantic_pretrained

        return save_fish_semantic_pretrained(self, directory)


# Compatibility name used by the published source and existing adapters.
DualARTransformer = FishS2ForConditionalGeneration

__all__ = [
    "DualARTransformer",
    "FishAttention",
    "FishKVCache",
    "FishRMSNorm",
    "FishS2ForConditionalGeneration",
    "FishSemanticOutput",
    "FishSlowOutput",
    "FishTransformerBlock",
]
