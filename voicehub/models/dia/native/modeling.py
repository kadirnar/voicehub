"""VoiceHub-native Dia encoder, decoder, objective, and generation.

This is a clean PyTorch port of the Apache-2.0 Dia implementation
released by Nari Labs and its Hugging Face conversion.  Module names
deliberately match ``nari-labs/Dia-1.6B-0626`` so the published
Safetensors load without a runtime translation layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicehub.models.dia.native.configuration import DiaArchitectureConfig, DiaDecoderConfig, DiaEncoderConfig


def _finite_number(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"`{name}` must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}.")
    return result


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"`{name}` must be a positive integer.")
    return int(value)


def _repeat_key_values(hidden_states: Tensor, repeats: int) -> Tensor:
    if repeats == 1:
        return hidden_states
    batch, heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None].expand(
        batch,
        heads,
        repeats,
        sequence,
        head_dim,
    )
    return expanded.reshape(batch, heads * repeats, sequence, head_dim)


def _rotate_half(values: Tensor) -> Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    query: Tensor,
    key: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> tuple[Tensor, Tensor]:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


def _padding_attention_bias(
    attention_mask: Tensor | None,
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    dtype: torch.dtype,
    device: torch.device,
    causal: bool,
) -> Tensor | None:
    """Build the additive mask used by the reference eager attention path."""
    allowed: Tensor | None = None
    if causal:
        allowed = torch.ones(
            query_length,
            key_length,
            dtype=torch.bool,
            device=device,
        ).tril(diagonal=key_length - query_length)
        allowed = allowed[None, None].expand(batch_size, 1, -1, -1)
    if attention_mask is not None:
        if not isinstance(attention_mask, Tensor):
            raise TypeError("Attention masks must be PyTorch tensors.")
        if attention_mask.ndim != 2:
            raise ValueError("Attention masks must have shape [batch, sequence].")
        if attention_mask.shape != (batch_size, key_length):
            raise ValueError(
                "Attention mask shape does not match the attention keys: "
                f"expected {(batch_size, key_length)}, found "
                f"{tuple(attention_mask.shape)}.")
        key_allowed = attention_mask.to(device=device, dtype=torch.bool)
        key_allowed = key_allowed[:, None, None, :].expand(
            batch_size,
            1,
            query_length,
            key_length,
        )
        allowed = key_allowed if allowed is None else allowed & key_allowed
    if allowed is None:
        return None
    bias = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return bias.masked_fill(~allowed, torch.finfo(dtype).min)


@dataclass
class DiaEncoderOutput:
    last_hidden_state: Tensor

    def __getitem__(self, index: int) -> Tensor:
        if index != 0:
            raise IndexError(index)
        return self.last_hidden_state


@dataclass
class DiaModelOutput:
    last_hidden_state: Tensor
    encoder_last_hidden_state: Tensor

    def __getitem__(self, index: int) -> Tensor:
        values = (self.last_hidden_state, self.encoder_last_hidden_state)
        return values[index]


@dataclass
class DiaConditionalGenerationOutput:
    logits: Tensor
    loss: Tensor | None = None
    encoder_last_hidden_state: Tensor | None = None

    def __getitem__(self, index: int) -> Tensor | None:
        values = (
            self.logits,
            self.loss,
            self.encoder_last_hidden_state,
        )
        return values[index]


class DiaMultiChannelEmbedding(nn.Module):
    """Sum one independent embedding table per delayed DAC channel."""

    def __init__(self, config: DiaDecoderConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(
            config.vocab_size * config.num_channels,
            config.hidden_size,
        )
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        # Keep non-persistent protocol buffers materialized even when the
        # parameter graph is constructed on ``meta`` for streaming loading.
        offsets = torch.arange(
            config.num_channels,
            dtype=torch.long,
            device="cpu",
        )
        offsets = offsets * config.vocab_size
        self.register_buffer("offsets", offsets, persistent=False)

    def forward(self, audio_codes: Tensor) -> Tensor:
        if audio_codes.ndim != 3:
            raise ValueError("Dia decoder IDs must have shape [batch, sequence, channels].")
        if audio_codes.shape[-1] != self.num_channels:
            raise ValueError(
                f"Dia expects {self.num_channels} audio channels; found "
                f"{audio_codes.shape[-1]}.")
        tokens = audio_codes.long() + self.offsets.to(audio_codes.device)
        embedded = self.embed(tokens)
        return embedded.sum(dim=2)


class DiaMLP(nn.Module):

    def __init__(
        self,
        config: DiaEncoderConfig | DiaDecoderConfig,
    ) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, values = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * values)


class DiaRMSNorm(nn.Module):
    """Reference-compatible float32 RMS normalization."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


class DiaRotaryEmbedding(nn.Module):

    def __init__(
        self,
        config: DiaEncoderConfig | DiaDecoderConfig,
    ) -> None:
        super().__init__()
        frequencies = torch.arange(
            0,
            config.head_dim,
            2,
            dtype=torch.float32,
            device="cpu",
        )
        inverse = 1.0 / (config.rope_theta**(frequencies / config.head_dim))
        self.register_buffer("inv_freq", inverse, persistent=False)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if position_ids.ndim != 2:
            raise ValueError("Dia position IDs must have shape [batch, sequence].")
        frequencies = torch.einsum(
            "bi,j->bij",
            position_ids.float(),
            self.inv_freq.to(position_ids.device).float(),
        )
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return (
            embeddings.cos().to(dtype=hidden_states.dtype),
            embeddings.sin().to(dtype=hidden_states.dtype),
        )


class DiaSelfAttention(nn.Module):

    def __init__(
        self,
        config: DiaEncoderConfig | DiaDecoderConfig,
        layer_idx: int,
        *,
        is_causal: bool,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (self.num_heads // self.num_key_value_heads)
        self.head_dim = config.head_dim
        self.is_causal = is_causal
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch,
            sequence,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch,
            sequence,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        query, key = _apply_rotary(
            query,
            key,
            *position_embeddings,
        )
        key = _repeat_key_values(key, self.num_key_value_groups)
        value = _repeat_key_values(value, self.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(-1, -2))
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=query.dtype)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(
            batch,
            sequence,
            self.num_heads * self.head_dim,
        )
        return self.o_proj(attended)


class DiaCrossAttention(nn.Module):

    def __init__(self, config: DiaDecoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.cross_hidden_size = config.cross_hidden_size
        self.num_heads = config.cross_num_attention_heads
        self.num_key_value_heads = config.cross_num_key_value_heads
        self.num_key_value_groups = (self.num_heads // self.num_key_value_heads)
        self.head_dim = config.cross_head_dim
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.cross_hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.cross_hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        batch, query_length, _ = hidden_states.shape
        key_length = encoder_hidden_states.shape[1]
        query = self.q_proj(hidden_states).view(
            batch,
            query_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = self.k_proj(encoder_hidden_states).view(
            batch,
            key_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = self.v_proj(encoder_hidden_states).view(
            batch,
            key_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = _repeat_key_values(key, self.num_key_value_groups)
        value = _repeat_key_values(value, self.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(-1, -2))
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=query.dtype)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(
            batch,
            query_length,
            self.num_heads * self.head_dim,
        )
        return self.o_proj(attended)


class DiaEncoderLayer(nn.Module):

    def __init__(self, config: DiaEncoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.pre_sa_norm = DiaRMSNorm(config.hidden_size, config.norm_eps)
        self.self_attention = DiaSelfAttention(
            config,
            layer_idx,
            is_causal=False,
        )
        self.post_sa_norm = DiaRMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = DiaMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        hidden_states = hidden_states + self.self_attention(
            self.pre_sa_norm(hidden_states),
            position_embeddings,
            attention_mask,
        )
        return hidden_states + self.mlp(self.post_sa_norm(hidden_states))


class DiaEncoder(nn.Module):

    def __init__(self, config: DiaEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DiaEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers))
        self.norm = DiaRMSNorm(config.hidden_size, config.norm_eps)
        self.rotary_emb = DiaRotaryEmbedding(config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> DiaEncoderOutput:
        if input_ids.ndim != 2:
            raise ValueError("Dia text IDs must have shape [batch, sequence].")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Dia text exceeds the configured maximum sequence length.")
        hidden_states = self.embedding(input_ids.long())
        positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        )[None].expand(input_ids.shape[0], -1)
        bias = _padding_attention_bias(
            attention_mask,
            batch_size=input_ids.shape[0],
            query_length=input_ids.shape[1],
            key_length=input_ids.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            causal=False,
        )
        position_embeddings = self.rotary_emb(hidden_states, positions)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                bias,
            )
        return DiaEncoderOutput(self.norm(hidden_states))


class DiaDecoderLayer(nn.Module):

    def __init__(self, config: DiaDecoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attention = DiaSelfAttention(
            config,
            layer_idx,
            is_causal=True,
        )
        self.cross_attention = DiaCrossAttention(config, layer_idx)
        self.pre_sa_norm = DiaRMSNorm(config.hidden_size, config.norm_eps)
        self.pre_ca_norm = DiaRMSNorm(config.hidden_size, config.norm_eps)
        self.pre_mlp_norm = DiaRMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = DiaMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None,
    ) -> Tensor:
        hidden_states = hidden_states + self.self_attention(
            self.pre_sa_norm(hidden_states),
            position_embeddings,
            attention_mask,
        )
        hidden_states = hidden_states + self.cross_attention(
            self.pre_ca_norm(hidden_states),
            encoder_hidden_states,
            encoder_attention_mask,
        )
        return hidden_states + self.mlp(self.pre_mlp_norm(hidden_states))


class DiaDecoder(nn.Module):

    def __init__(self, config: DiaDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.num_channels = config.num_channels
        self.vocab_size = config.vocab_size
        self.embeddings = DiaMultiChannelEmbedding(config)
        self.layers = nn.ModuleList(
            DiaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers))
        self.norm = DiaRMSNorm(config.hidden_size, config.norm_eps)
        self.rotary_emb = DiaRotaryEmbedding(config)

    def forward(
        self,
        input_ids: Tensor,
        *,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        if input_ids.ndim != 3:
            raise ValueError("Dia decoder IDs must have shape [batch, sequence, channels].")
        batch, sequence, _ = input_ids.shape
        if sequence > self.config.max_position_embeddings:
            raise ValueError("Dia audio tokens exceed the configured decoder length.")
        hidden_states = self.embeddings(input_ids)
        if position_ids is None:
            position_ids = torch.arange(
                sequence,
                device=input_ids.device,
            )[None].expand(batch, -1)
        self_bias = _padding_attention_bias(
            attention_mask,
            batch_size=batch,
            query_length=sequence,
            key_length=sequence,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            causal=True,
        )
        cross_bias = _padding_attention_bias(
            encoder_attention_mask,
            batch_size=batch,
            query_length=sequence,
            key_length=encoder_hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            causal=False,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                self_bias,
                encoder_hidden_states,
                cross_bias,
            )
        return self.norm(hidden_states)


class DiaModel(nn.Module):
    """Bare byte encoder and delayed-codebook decoder."""

    def __init__(
        self,
        config: DiaArchitectureConfig | dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = DiaArchitectureConfig.coerce(config)
        self.encoder = DiaEncoder(self.config.encoder_config)
        self.decoder = DiaDecoder(self.config.decoder_config)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        decoder_position_ids: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        encoder_outputs: DiaEncoderOutput | Tensor | tuple[Tensor, ...] | None = None,
        **_: Any,
    ) -> DiaModelOutput:
        if encoder_outputs is None:
            if input_ids is None:
                raise ValueError("Dia requires text IDs or cached encoder outputs.")
            encoded = self.encoder(input_ids, attention_mask)
            encoder_hidden_states = encoded.last_hidden_state
        elif isinstance(encoder_outputs, DiaEncoderOutput):
            encoder_hidden_states = encoder_outputs.last_hidden_state
        elif isinstance(encoder_outputs, Tensor):
            encoder_hidden_states = encoder_outputs
        else:
            encoder_hidden_states = encoder_outputs[0]

        batch_size = encoder_hidden_states.shape[0]
        channels = self.config.decoder_config.num_channels
        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (batch_size, 1, channels),
                self.config.decoder_config.bos_token_id,
                dtype=torch.long,
                device=encoder_hidden_states.device,
            )
        elif decoder_input_ids.ndim == 2:
            if decoder_input_ids.shape[0] != batch_size * channels:
                raise ValueError("Flattened Dia decoder IDs require batch * channels rows.")
            decoder_input_ids = decoder_input_ids.reshape(
                batch_size,
                channels,
                -1,
            ).transpose(1, 2)
        decoded = self.decoder(
            decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            position_ids=decoder_position_ids,
        )
        return DiaModelOutput(decoded, encoder_hidden_states)


def _top_k_mask(scores: Tensor, top_k: int) -> Tensor:
    top_k = min(top_k, scores.shape[-1])
    threshold = torch.topk(scores, top_k, dim=-1).values[..., -1, None]
    return scores.masked_fill(scores < threshold, -torch.inf)


def _top_p_mask(scores: Tensor, top_p: float) -> Tensor:
    if top_p >= 1.0:
        return scores
    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    probabilities = F.softmax(sorted_scores.float(), dim=-1)
    remove = probabilities.cumsum(dim=-1) > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
    return torch.full_like(scores, -torch.inf).scatter(
        -1,
        sorted_indices,
        sorted_scores,
    )


def _apply_eos_channel_filter(
    scores: Tensor,
    *,
    eos_token_id: int,
) -> Tensor:
    """Match Dia's channel-zero EOS and special-token constraint."""
    result = scores.clone()
    result[:, 1:, eos_token_id:] = -torch.inf
    result[:, 0, eos_token_id + 1:] = -torch.inf
    flat = result.reshape(-1, result.shape[-1])
    highest = flat.argmax(dim=-1)
    force = highest == eos_token_id
    flat[force, :eos_token_id] = -torch.inf
    flat[~force, eos_token_id] = -torch.inf
    return flat.reshape_as(result)


class DiaForConditionalGeneration(nn.Module):
    """Published Dia graph with native teacher forcing and generation."""

    def __init__(
        self,
        config: DiaArchitectureConfig | dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = DiaArchitectureConfig.coerce(config)
        self.model = DiaModel(self.config)
        decoder = self.config.decoder_config
        self.num_channels = decoder.num_channels
        self.vocab_size = decoder.vocab_size
        self.logits_dense = nn.Linear(
            decoder.hidden_size,
            self.num_channels * self.vocab_size,
            bias=False,
        )
        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset_parameters(self) -> None:
        standard_deviation = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=standard_deviation,
                )
            elif isinstance(module, DiaRMSNorm):
                nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        decoder_position_ids: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        encoder_outputs: DiaEncoderOutput | Tensor | tuple[Tensor, ...] | None = None,
        labels: Tensor | None = None,
        **kwargs: Any,
    ) -> DiaConditionalGenerationOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_position_ids=decoder_position_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )
        hidden = outputs.last_hidden_state
        batch_size = hidden.shape[0]
        logits = self.logits_dense(hidden)
        logits = logits.view(
            batch_size,
            -1,
            self.num_channels,
            self.vocab_size,
        )
        logits = logits.transpose(1, 2).contiguous().view(
            batch_size * self.num_channels,
            -1,
            self.vocab_size,
        )
        loss = None
        if labels is not None:
            if labels.shape != logits.shape[:-1]:
                raise ValueError(
                    "Dia labels must have shape [batch * channels, sequence]; "
                    f"expected {tuple(logits.shape[:-1])}, found "
                    f"{tuple(labels.shape)}.")
            loss = F.cross_entropy(
                logits.float().reshape(-1, self.vocab_size),
                labels.long().reshape(-1),
                ignore_index=-100,
            )
        return DiaConditionalGenerationOutput(
            logits=logits,
            loss=loss,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        )

    @staticmethod
    def apply_delay_mask(
        input_ids: Tensor,
        pad_token_id: int,
        delay_mask: Tensor | None,
    ) -> Tensor:
        if delay_mask is None:
            return input_ids
        mask_length = min(input_ids.shape[1], delay_mask.shape[1])
        result = input_ids.clone()
        fixed = delay_mask[:, :mask_length].to(result.device)
        result[:, :mask_length] = torch.where(
            fixed == pad_token_id,
            result[:, :mask_length],
            fixed,
        )
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        *,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 1.8,
        top_k: int | None = 50,
        top_p: float = 0.9,
        guidance_scale: float | None = 3.0,
        **kwargs: Any,
    ) -> Tensor:
        """Generate delayed DAC tokens using the released Dia sampling rules.

        The native implementation intentionally recomputes the decoder
        prefix instead of depending on a framework cache object.  This
        preserves exact logits and keeps cache optimisation an optional
        runtime strategy.
        """
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported native Dia generation options: {names}.")
        max_new_tokens = _positive_integer("max_new_tokens", max_new_tokens)
        if not isinstance(do_sample, bool):
            raise TypeError("`do_sample` must be a boolean.")
        temperature = max(
            1.0,
            _finite_number("temperature", temperature, minimum=0.0),
        )
        if top_k is not None:
            top_k = _positive_integer("top_k", top_k)
        top_p = _finite_number("top_p", top_p, minimum=0.0)
        if not 0.0 < top_p <= 1.0:
            raise ValueError("`top_p` must be in the interval (0, 1].")
        if guidance_scale is not None:
            guidance_scale = _finite_number(
                "guidance_scale",
                guidance_scale,
                minimum=0.0,
            )
        use_cfg = guidance_scale is not None and guidance_scale != 1.0
        if use_cfg and guidance_scale <= 1.0:
            raise ValueError("Dia classifier-free guidance requires `guidance_scale > 1`.")

        if input_ids.ndim != 2:
            raise ValueError("Dia generation text IDs must be rank two.")
        batch_size = input_ids.shape[0]
        device = input_ids.device
        decoder = self.config.decoder_config
        if decoder_input_ids is None:
            delay_mask = torch.full(
                (batch_size, 1, self.num_channels),
                decoder.bos_token_id,
                dtype=torch.long,
                device=device,
            )
            decoder_attention_mask = torch.ones(
                batch_size,
                1,
                dtype=torch.long,
                device=device,
            )
        else:
            if decoder_input_ids.ndim != 3:
                raise ValueError(
                    "Dia generation decoder IDs must have shape "
                    "[batch, sequence, channels].")
            if decoder_input_ids.shape[0] != batch_size:
                raise ValueError("Dia text and decoder batch sizes must match.")
            delay_mask = decoder_input_ids.long().to(device)
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones(
                batch_size,
                delay_mask.shape[1],
                dtype=torch.long,
                device=device,
            )
        else:
            decoder_attention_mask = decoder_attention_mask.long().to(device)

        channel_zero_padding = (delay_mask[:, :, 0] == decoder.pad_token_id).sum(dim=-1)
        valid_length = delay_mask.shape[1] - int(channel_zero_padding.max().item())
        sequences = delay_mask[:, :valid_length].clone()
        sequences = self.apply_delay_mask(
            sequences,
            decoder.pad_token_id,
            delay_mask,
        )

        conditioned = self.model.encoder(
            input_ids,
            attention_mask,
        ).last_hidden_state
        if use_cfg:
            unconditioned = self.model.encoder(
                torch.zeros_like(input_ids),
                attention_mask,
            ).last_hidden_state
        else:
            unconditioned = None

        active = torch.zeros(batch_size, dtype=torch.bool, device=device)
        remaining_delays = torch.tensor(
            self.config.delay_pattern,
            dtype=torch.long,
            device=device,
        )[None].expand(batch_size, -1).clone()
        generated_after_eos = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=device,
        )
        maximum_steps = max_new_tokens + max(self.config.delay_pattern)

        for step in range(maximum_steps):
            forced_sequence = self.apply_delay_mask(
                sequences,
                decoder.pad_token_id,
                delay_mask,
            )
            sequence_mask = torch.ones(
                batch_size,
                forced_sequence.shape[1],
                dtype=torch.long,
                device=device,
            )
            conditional_output = self(
                attention_mask=attention_mask,
                decoder_input_ids=forced_sequence,
                decoder_attention_mask=sequence_mask,
                encoder_outputs=conditioned,
            )
            conditional_scores = conditional_output.logits[:, -1]
            conditional_scores = conditional_scores.reshape(
                batch_size,
                self.num_channels,
                self.vocab_size,
            )

            if use_cfg:
                unconditional_output = self(
                    attention_mask=attention_mask,
                    decoder_input_ids=forced_sequence,
                    decoder_attention_mask=sequence_mask,
                    encoder_outputs=unconditioned,
                )
                unconditional_scores = unconditional_output.logits[:, -1]
                unconditional_scores = unconditional_scores.reshape_as(conditional_scores)
                guided = conditional_scores + (conditional_scores - unconditional_scores) * guidance_scale
                if top_k is not None:
                    guided_flat = guided.reshape(-1, self.vocab_size)
                    indices = torch.topk(
                        guided_flat,
                        min(top_k, self.vocab_size),
                        dim=-1,
                    ).indices
                    keep = torch.zeros_like(guided_flat, dtype=torch.bool)
                    keep.scatter_(1, indices, True)
                    scores = conditional_scores.reshape(
                        -1,
                        self.vocab_size,
                    ).masked_fill(~keep, -torch.inf)
                    scores = scores.reshape_as(conditional_scores)
                else:
                    scores = guided
            else:
                scores = conditional_scores

            scores = scores / temperature
            scores = _apply_eos_channel_filter(
                scores,
                eos_token_id=decoder.eos_token_id,
            )
            if not use_cfg and top_k is not None:
                scores = _top_k_mask(scores, top_k)
            scores = _top_p_mask(scores, top_p)

            predicted_zero = scores[:, 0].argmax(dim=-1)
            new_eos = predicted_zero == decoder.eos_token_id
            if step + 1 >= max_new_tokens:
                new_eos = torch.ones_like(new_eos)
            active |= new_eos
            force_eos = active[:, None] & (remaining_delays == 0)
            if force_eos.any():
                rows, channels = force_eos.nonzero(as_tuple=True)
                scores[rows, channels] = -torch.inf
                scores[rows, channels, decoder.eos_token_id] = 0.0

            flat_scores = scores.reshape(-1, self.vocab_size)
            if do_sample:
                probabilities = F.softmax(flat_scores.float(), dim=-1)
                next_tokens = torch.multinomial(probabilities, 1).squeeze(-1)
            else:
                next_tokens = flat_scores.argmax(dim=-1)
            next_tokens = next_tokens.reshape(batch_size, self.num_channels)
            sequences = torch.cat(
                (sequences, next_tokens[:, None]),
                dim=1,
            )
            remaining_delays -= active[:, None].long()
            generated_after_eos += active.long()
            if bool((active & (generated_after_eos > max(self.config.delay_pattern))).all()):
                break

        return self.apply_delay_mask(
            sequences,
            decoder.pad_token_id,
            delay_mask,
        )


__all__ = [
    "DiaConditionalGenerationOutput",
    "DiaDecoder",
    "DiaDecoderLayer",
    "DiaEncoder",
    "DiaEncoderLayer",
    "DiaEncoderOutput",
    "DiaForConditionalGeneration",
    "DiaMLP",
    "DiaModel",
    "DiaModelOutput",
    "DiaMultiChannelEmbedding",
    "DiaRMSNorm",
    "DiaRotaryEmbedding",
]
