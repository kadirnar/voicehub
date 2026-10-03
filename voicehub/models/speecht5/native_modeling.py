"""VoiceHub-owned PyTorch implementation of SpeechT5 text-to-speech.

The module mirrors the published Hugging Face SpeechT5 checkpoint
namespace while depending only on PyTorch and the standard library.  It
intentionally implements the public TTS graph, its autoregressive cache,
supervised spectrogram objective, and the paired HiFi-GAN vocoder; ASR
and voice conversion graphs are separate model families and are not
implied here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicehub.models.speecht5.native_configuration import NativeSpeechT5Config, NativeSpeechT5HifiGanConfig
from voicehub.optimization.protocols import OptimizationCompileTarget


def _activation(name: str, values: Tensor) -> Tensor:
    if name == "gelu":
        return F.gelu(values)
    if name == "gelu_new":
        return 0.5 * values * (
            1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (values + 0.044715 * values.pow(3))))
    if name == "relu":
        return F.relu(values)
    if name == "selu":
        return F.selu(values)
    if name == "silu":
        return F.silu(values)
    raise ValueError(f"Unsupported SpeechT5 activation {name!r}.")


def _expand_attention_mask(
    mask: Tensor,
    dtype: torch.dtype,
    *,
    target_length: int,
) -> Tensor:
    if mask.ndim != 2:
        raise ValueError("SpeechT5 attention masks must have shape [batch, time].")
    batch_size, source_length = mask.shape
    expanded = mask[:, None, None, :].expand(
        batch_size,
        1,
        target_length,
        source_length,
    ).to(dtype=dtype)
    inverted = 1.0 - expanded
    return inverted.masked_fill(inverted.bool(), torch.finfo(dtype).min)


def _causal_attention_mask(
    batch_size: int,
    target_length: int,
    past_length: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    if target_length <= 1:
        return None
    mask = torch.full(
        (target_length, target_length),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    positions = torch.arange(target_length, device=device)
    mask.masked_fill_(positions < (positions + 1).view(-1, 1), 0.0)
    if past_length:
        mask = torch.cat(
            (
                torch.zeros(
                    target_length,
                    past_length,
                    dtype=dtype,
                    device=device,
                ),
                mask,
            ),
            dim=-1,
        )
    return mask[None, None].expand(
        batch_size,
        1,
        target_length,
        target_length + past_length,
    )


def shift_spectrograms_right(
    input_values: Tensor,
    reduction_factor: int = 1,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Apply SpeechT5 teacher-forcing shift and frame reduction."""
    if input_values.ndim != 3:
        raise ValueError("SpeechT5 target spectrograms must have shape [batch, frames, mel].")
    if reduction_factor > 1:
        input_values = input_values[:, reduction_factor - 1::reduction_factor]
        if attention_mask is not None:
            attention_mask = attention_mask[:, reduction_factor - 1::reduction_factor]
    shifted = input_values.new_zeros(input_values.shape)
    shifted[:, 1:] = input_values[:, :-1].clone()
    shifted.masked_fill_(shifted == -100.0, 0.0)
    return shifted, attention_mask


@dataclass(slots=True)
class SpeechT5EncoderOutput:
    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor | None, ...] | None = None

    def __getitem__(self, index: int) -> Any:
        return (
            self.last_hidden_state,
            self.hidden_states,
            self.attentions,
        )[index]


@dataclass(slots=True)
class SpeechT5DecoderOutput:
    last_hidden_state: Tensor
    past_key_values: SpeechT5DecoderCache | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor | None, ...] | None = None
    cross_attentions: tuple[Tensor | None, ...] | None = None

    def __getitem__(self, index: int) -> Any:
        return (
            self.last_hidden_state,
            self.past_key_values,
            self.hidden_states,
            self.attentions,
            self.cross_attentions,
        )[index]


@dataclass(slots=True)
class SpeechT5Seq2SeqOutput:
    last_hidden_state: Tensor
    past_key_values: SpeechT5DecoderCache | None
    decoder_hidden_states: tuple[Tensor, ...] | None
    decoder_attentions: tuple[Tensor | None, ...] | None
    cross_attentions: tuple[Tensor | None, ...] | None
    encoder_last_hidden_state: Tensor
    encoder_hidden_states: tuple[Tensor, ...] | None
    encoder_attentions: tuple[Tensor | None, ...] | None

    def __getitem__(self, index: int) -> Any:
        return (
            self.last_hidden_state,
            self.past_key_values,
            self.decoder_hidden_states,
            self.decoder_attentions,
            self.cross_attentions,
            self.encoder_last_hidden_state,
            self.encoder_hidden_states,
            self.encoder_attentions,
        )[index]


@dataclass(slots=True)
class SpeechT5SpectrogramOutput:
    loss: Tensor | None
    spectrogram: Tensor
    past_key_values: SpeechT5DecoderCache | None = None
    decoder_hidden_states: tuple[Tensor, ...] | None = None
    decoder_attentions: tuple[Tensor | None, ...] | None = None
    cross_attentions: tuple[Tensor | None, ...] | None = None
    encoder_last_hidden_state: Tensor | None = None
    encoder_hidden_states: tuple[Tensor, ...] | None = None
    encoder_attentions: tuple[Tensor | None, ...] | None = None
    spectrogram_loss: Tensor | None = None
    stop_loss: Tensor | None = None
    guided_attention_loss: Tensor | None = None
    losses: dict[str, Tensor] | None = None

    def __getitem__(self, index: int) -> Any:
        values = (
            self.loss,
            self.spectrogram,
            self.past_key_values,
            self.decoder_hidden_states,
            self.decoder_attentions,
            self.cross_attentions,
            self.encoder_last_hidden_state,
            self.encoder_hidden_states,
            self.encoder_attentions,
        )
        return values[index]


class SpeechT5DecoderCache:
    """Per-layer self- and cross-attention key/value cache."""

    def __init__(self, number_of_layers: int) -> None:
        if isinstance(number_of_layers, bool) or number_of_layers <= 0:
            raise ValueError("SpeechT5 cache requires at least one decoder layer.")
        self.self_keys: list[Tensor | None] = [None] * number_of_layers
        self.self_values: list[Tensor | None] = [None] * number_of_layers
        self.cross_keys: list[Tensor | None] = [None] * number_of_layers
        self.cross_values: list[Tensor | None] = [None] * number_of_layers

    @property
    def sequence_length(self) -> int:
        for keys in self.self_keys:
            if keys is not None:
                return int(keys.shape[-2])
        return 0

    def update(
        self,
        layer_index: int,
        keys: Tensor,
        values: Tensor,
        *,
        cross_attention: bool,
    ) -> tuple[Tensor, Tensor]:
        key_cache = self.cross_keys if cross_attention else self.self_keys
        value_cache = self.cross_values if cross_attention else self.self_values
        cached_keys = key_cache[layer_index]
        if cached_keys is not None:
            if cross_attention:
                return cached_keys, value_cache[layer_index]  # type: ignore[return-value]
            keys = torch.cat((cached_keys, keys), dim=-2)
            values = torch.cat(
                (value_cache[layer_index], values),  # type: ignore[arg-type]
                dim=-2,
            )
        key_cache[layer_index] = keys
        value_cache[layer_index] = values
        return keys, values


class SpeechT5ScaledPositionalEncoding(nn.Module):

    def __init__(self, dropout: float, dimension: int, max_length: int) -> None:
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        divisors = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.int64).float() * -(math.log(10_000.0) / dimension))
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(positions.float() * divisors)
        encoding[:, 1::2] = torch.cos(positions.float() * divisors)
        # The checkpoint published with Transformers 4.28 contains these two
        # buffers, and 4.28 uses the stored values. They differ from this
        # recomputed table by up to 1.2e-4, so loading them reproduces 4.28;
        # newer Transformers releases recompute the table and omit it on save.
        self.register_buffer("pe", encoding.unsqueeze(0), persistent=True)
        self.dropout = nn.Dropout(dropout)
        self.dim = dimension
        self.max_len = max_length
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.shape[1] > self.max_len:
            raise ValueError(
                "SpeechT5 sequence exceeds the checkpoint positional table: "
                f"{embeddings.shape[1]} > {self.max_len}.")
        return self.dropout(
            embeddings + self.alpha * self.pe[:, :embeddings.shape[1]].to(dtype=embeddings.dtype, ))


class SpeechT5RelativePositionalEncoding(nn.Module):

    def __init__(self, dimension: int, max_length: int) -> None:
        super().__init__()
        self.dim = dimension
        self.max_length = max_length
        self.pe_k = nn.Embedding(2 * max_length, dimension)

    def forward(self, hidden_states: Tensor) -> Tensor:
        sequence_length = hidden_states.shape[1]
        positions = torch.arange(
            sequence_length,
            device=hidden_states.device,
            dtype=torch.long,
        )
        relative = positions[:, None] - positions[None, :]
        relative = relative.clamp(
            min=-self.max_length,
            max=self.max_length - 1,
        )
        return self.pe_k(relative + self.max_length)


class SpeechT5TextEncoderPrenet(nn.Module):

    def __init__(self, config: NativeSpeechT5Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.encode_positions = SpeechT5ScaledPositionalEncoding(
            config.positional_dropout,
            config.hidden_size,
            config.max_text_positions,
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("SpeechT5 input IDs must have shape [batch, tokens].")
        if input_ids.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
        }:
            raise TypeError("SpeechT5 input IDs must use an integer dtype.")
        return self.encode_positions(self.embed_tokens(input_ids.long()))


class SpeechT5SpeechDecoderPrenet(nn.Module):

    def __init__(self, config: NativeSpeechT5Config) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            nn.Linear(
                config.num_mel_bins if index == 0 else config.speech_decoder_prenet_units,
                config.speech_decoder_prenet_units,
            ) for index in range(config.speech_decoder_prenet_layers)
        ])
        self.final_layer = nn.Linear(
            config.speech_decoder_prenet_units,
            config.hidden_size,
        )
        self.encode_positions = SpeechT5ScaledPositionalEncoding(
            config.positional_dropout,
            config.hidden_size,
            config.max_speech_positions,
        )
        self.speaker_embeds_layer = nn.Linear(
            config.speaker_embedding_dim + config.hidden_size,
            config.hidden_size,
        )

    @staticmethod
    def _consistent_dropout(embeddings: Tensor, probability: float) -> Tensor:
        # SpeechT5 intentionally applies prenet dropout during inference and
        # shares one mask across the batch. This matches the current official
        # implementation and keeps seeded generation reproducible.
        mask = torch.bernoulli(
            embeddings[0],
            p=probability,
        )
        masks = mask.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        return torch.where(masks == 1, embeddings, 0) / (1.0 - probability)

    def forward(
        self,
        input_values: Tensor,
        speaker_embeddings: Tensor | None = None,
    ) -> Tensor:
        if (input_values.ndim != 3 or input_values.shape[-1] != self.config.num_mel_bins):
            raise ValueError(
                "SpeechT5 decoder inputs must have shape "
                f"[batch, frames, {self.config.num_mel_bins}].")
        embeddings = input_values
        for layer in self.layers:
            embeddings = F.relu(layer(embeddings))
            embeddings = self._consistent_dropout(
                embeddings,
                self.config.speech_decoder_prenet_dropout,
            )
        embeddings = self.encode_positions(self.final_layer(embeddings))
        if speaker_embeddings is not None:
            if (speaker_embeddings.ndim != 2 or speaker_embeddings.shape[0] != embeddings.shape[0] or
                    speaker_embeddings.shape[1] != self.config.speaker_embedding_dim):
                raise ValueError(
                    "SpeechT5 speaker embeddings must have shape "
                    f"[batch, {self.config.speaker_embedding_dim}].")
            speaker_embeddings = F.normalize(speaker_embeddings)
            speaker_embeddings = speaker_embeddings.unsqueeze(1).expand(
                -1,
                embeddings.shape[1],
                -1,
            )
            embeddings = F.relu(
                self.speaker_embeds_layer(torch.cat((embeddings, speaker_embeddings), dim=-1)))
        return embeddings


class SpeechT5Attention(nn.Module):

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        *,
        dropout: float,
        layer_index: int,
    ) -> None:
        super().__init__()
        if embedding_dimension % number_of_heads:
            raise ValueError("SpeechT5 attention dimensions are not divisible.")
        self.embed_dim = embedding_dimension
        self.num_heads = number_of_heads
        self.dropout = dropout
        self.head_dim = embedding_dimension // number_of_heads
        self.scaling = self.head_dim**-0.5
        self.layer_idx = layer_index
        self.k_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.v_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.q_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.out_proj = nn.Linear(embedding_dimension, embedding_dimension)

    def _split_heads(self, values: Tensor) -> Tensor:
        return values.view(
            values.shape[0],
            values.shape[1],
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        key_value_states: Tensor | None = None,
        cache: SpeechT5DecoderCache | None = None,
        attention_mask: Tensor | None = None,
        position_bias: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        cross_attention = key_value_states is not None
        batch_size, target_length, _ = hidden_states.shape
        query_states = self._split_heads(self.q_proj(hidden_states) * self.scaling)
        current_states = (key_value_states if cross_attention else hidden_states)
        if current_states is None:  # pragma: no cover - type narrowing
            raise RuntimeError("SpeechT5 attention received no source states.")
        if (cache is not None and cross_attention and cache.cross_keys[self.layer_idx] is not None):
            key_states = cache.cross_keys[self.layer_idx]
            value_states = cache.cross_values[self.layer_idx]
        else:
            key_states = self._split_heads(self.k_proj(current_states))
            value_states = self._split_heads(self.v_proj(current_states))
            if cache is not None:
                key_states, value_states = cache.update(
                    self.layer_idx,
                    key_states,
                    value_states,
                    cross_attention=cross_attention,
                )
        if key_states is None or value_states is None:  # pragma: no cover
            raise RuntimeError("SpeechT5 attention cache is incomplete.")
        source_length = key_states.shape[-2]
        queries = query_states.reshape(
            batch_size * self.num_heads,
            target_length,
            self.head_dim,
        )
        keys = key_states.reshape(
            batch_size * self.num_heads,
            source_length,
            self.head_dim,
        )
        values = value_states.reshape(
            batch_size * self.num_heads,
            source_length,
            self.head_dim,
        )
        attention_weights = torch.bmm(queries, keys.transpose(1, 2))
        if position_bias is not None:
            reshaped_queries = queries.transpose(0, 1)
            relative_bias = torch.matmul(
                reshaped_queries,
                position_bias.transpose(-2, -1),
            )
            relative_bias = relative_bias.transpose(0, 1).reshape(
                batch_size * self.num_heads,
                target_length,
                source_length,
            )
            attention_weights = attention_weights + relative_bias
        if attention_mask is not None:
            expected = (batch_size, 1, target_length, source_length)
            if tuple(attention_mask.shape) != expected:
                raise ValueError(
                    "SpeechT5 attention mask has shape "
                    f"{tuple(attention_mask.shape)}, expected {expected}.")
            attention_weights = attention_weights.view(
                batch_size,
                self.num_heads,
                target_length,
                source_length,
            )
            attention_weights = attention_weights + attention_mask
            attention_weights = attention_weights.view(
                batch_size * self.num_heads,
                target_length,
                source_length,
            )
        attention_weights = F.softmax(attention_weights, dim=-1)
        returned_attention = (
            attention_weights.view(
                batch_size,
                self.num_heads,
                target_length,
                source_length,
            ) if output_attentions else None)
        probabilities = F.dropout(
            attention_weights,
            p=self.dropout,
            training=self.training,
        )
        output = torch.bmm(probabilities, values)
        output = output.view(
            batch_size,
            self.num_heads,
            target_length,
            self.head_dim,
        ).transpose(1, 2).reshape(
            batch_size,
            target_length,
            self.embed_dim,
        )
        return self.out_proj(output), returned_attention


class SpeechT5FeedForward(nn.Module):

    def __init__(
        self,
        config: NativeSpeechT5Config,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.activation_name = config.hidden_act
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.intermediate_dense = nn.Linear(
            config.hidden_size,
            intermediate_size,
        )
        self.output_dense = nn.Linear(intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = _activation(self.activation_name, hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        return self.output_dropout(hidden_states)


class SpeechT5EncoderLayer(nn.Module):

    def __init__(
        self,
        config: NativeSpeechT5Config,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.attention = SpeechT5Attention(
            config.hidden_size,
            config.encoder_attention_heads,
            dropout=config.attention_dropout,
            layer_index=layer_index,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.feed_forward = SpeechT5FeedForward(
            config,
            config.encoder_ffn_dim,
        )
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        position_bias: Tensor,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None]:
        residual = hidden_states
        hidden_states, attention = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = residual + self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        return self.final_layer_norm(hidden_states), attention


class SpeechT5DecoderLayer(nn.Module):

    def __init__(
        self,
        config: NativeSpeechT5Config,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.self_attn = SpeechT5Attention(
            config.hidden_size,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            layer_index=layer_index,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.encoder_attn = SpeechT5Attention(
            config.hidden_size,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            layer_index=layer_index,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.feed_forward = SpeechT5FeedForward(
            config,
            config.decoder_ffn_dim,
        )
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None,
        cache: SpeechT5DecoderCache | None,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        residual = hidden_states
        hidden_states, self_attention = self.self_attn(
            hidden_states,
            cache=cache,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_layer_norm(residual + self.dropout(hidden_states))
        residual = hidden_states
        hidden_states, cross_attention = self.encoder_attn(
            hidden_states,
            key_value_states=encoder_hidden_states,
            cache=cache,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.encoder_attn_layer_norm(residual + self.dropout(hidden_states))
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states, self_attention, cross_attention


class SpeechT5Encoder(nn.Module):

    def __init__(self, config: NativeSpeechT5Config) -> None:
        super().__init__()
        self.config = config
        self.layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layerdrop = config.encoder_layerdrop
        self.layers = nn.ModuleList(
            [SpeechT5EncoderLayer(config, index) for index in range(config.encoder_layers)])
        self.embed_positions = SpeechT5RelativePositionalEncoding(
            config.hidden_size // config.encoder_attention_heads,
            config.encoder_max_relative_position,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        *,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SpeechT5EncoderOutput:
        output_attentions = (
            self.config.output_attentions if output_attentions is None else output_attentions)
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states)
        additive_mask = (
            None if attention_mask is None else _expand_attention_mask(
                attention_mask,
                hidden_states.dtype,
                target_length=hidden_states.shape[1],
            ))
        hidden_states = self.dropout(self.layer_norm(hidden_states))
        position_bias = self.embed_positions(hidden_states)
        all_hidden_states: list[Tensor] | None = ([] if output_hidden_states else None)
        all_attentions: list[Tensor | None] | None = ([] if output_attentions else None)
        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            skip = (
                self.training and self.layerdrop > 0.0 and bool(
                    (torch.rand((), device=hidden_states.device) < self.layerdrop).item()))
            if skip:
                attention = None
            else:
                hidden_states, attention = layer(
                    hidden_states,
                    attention_mask=additive_mask,
                    position_bias=position_bias,
                    output_attentions=output_attentions,
                )
            if all_attentions is not None:
                all_attentions.append(attention)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        return SpeechT5EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=(tuple(all_hidden_states) if all_hidden_states is not None else None),
            attentions=(tuple(all_attentions) if all_attentions is not None else None),
        )


class SpeechT5EncoderWithTextPrenet(nn.Module):

    def __init__(self, config: NativeSpeechT5Config) -> None:
        super().__init__()
        self.prenet = SpeechT5TextEncoderPrenet(config)
        self.wrapped_encoder = SpeechT5Encoder(config)

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None = None,
        *,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SpeechT5EncoderOutput:
        return self.wrapped_encoder(
            self.prenet(input_values),
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class SpeechT5Decoder(nn.Module):

    def __init__(self, config: NativeSpeechT5Config) -> None:
        super().__init__()
        self.config = config
        self.layerdrop = config.decoder_layerdrop
        self.layers = nn.ModuleList(
            [SpeechT5DecoderLayer(config, index) for index in range(config.decoder_layers)])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        *,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: SpeechT5DecoderCache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SpeechT5DecoderOutput:
        output_attentions = (
            self.config.output_attentions if output_attentions is None else output_attentions)
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states)
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = SpeechT5DecoderCache(len(self.layers))
        cache = past_key_values if use_cache else None
        past_length = 0 if cache is None else cache.sequence_length
        causal_mask = _causal_attention_mask(
            hidden_states.shape[0],
            hidden_states.shape[1],
            past_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if attention_mask is not None:
            expanded = _expand_attention_mask(
                attention_mask,
                hidden_states.dtype,
                target_length=hidden_states.shape[1],
            )
            causal_mask = (expanded if causal_mask is None else causal_mask + expanded)
        cross_mask = (
            None if encoder_attention_mask is None else _expand_attention_mask(
                encoder_attention_mask,
                hidden_states.dtype,
                target_length=hidden_states.shape[1],
            ))
        all_hidden_states: list[Tensor] | None = ([] if output_hidden_states else None)
        self_attentions: list[Tensor | None] | None = ([] if output_attentions else None)
        cross_attentions: list[Tensor | None] | None = ([] if output_attentions else None)
        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            skip = (
                self.training and self.layerdrop > 0.0 and bool(
                    (torch.rand((), device=hidden_states.device) < self.layerdrop).item()))
            if skip:
                self_attention = None
                cross_attention = None
            else:
                hidden_states, self_attention, cross_attention = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=cross_mask,
                    cache=cache,
                    output_attentions=output_attentions,
                )
            if self_attentions is not None:
                self_attentions.append(self_attention)
                cross_attentions.append(cross_attention)  # type: ignore[union-attr]
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        return SpeechT5DecoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=cache,
            hidden_states=(tuple(all_hidden_states) if all_hidden_states is not None else None),
            attentions=(tuple(self_attentions) if self_attentions is not None else None),
            cross_attentions=(tuple(cross_attentions) if cross_attentions is not None else None),
        )


class SpeechT5DecoderWithSpeechPrenet(nn.Module):

    def __init__(self, config: NativeSpeechT5Config) -> None:
        super().__init__()
        self.prenet = SpeechT5SpeechDecoderPrenet(config)
        self.wrapped_decoder = SpeechT5Decoder(config)

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None = None,
        *,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None = None,
        speaker_embeddings: Tensor | None = None,
        past_key_values: SpeechT5DecoderCache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SpeechT5DecoderOutput:
        return self.wrapped_decoder(
            self.prenet(input_values, speaker_embeddings),
            attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class SpeechT5Model(nn.Module):

    def __init__(self, config: NativeSpeechT5Config) -> None:
        super().__init__()
        self.config = config
        self.encoder = SpeechT5EncoderWithTextPrenet(config)
        self.decoder = SpeechT5DecoderWithSpeechPrenet(config)

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None,
        decoder_input_values: Tensor,
        decoder_attention_mask: Tensor | None,
        *,
        encoder_outputs: SpeechT5EncoderOutput | None = None,
        past_key_values: SpeechT5DecoderCache | None = None,
        use_cache: bool | None = None,
        speaker_embeddings: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SpeechT5Seq2SeqOutput:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_values,
                attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        decoder_outputs = self.decoder(
            decoder_input_values,
            decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            speaker_embeddings=speaker_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return SpeechT5Seq2SeqOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class SpeechT5BatchNormConvLayer(nn.Module):

    def __init__(
        self,
        config: NativeSpeechT5Config,
        layer_index: int,
    ) -> None:
        super().__init__()
        input_channels = (config.num_mel_bins if layer_index == 0 else config.speech_decoder_postnet_units)
        output_channels = (
            config.num_mel_bins if layer_index == config.speech_decoder_postnet_layers -
            1 else config.speech_decoder_postnet_units)
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            config.speech_decoder_postnet_kernel,
            padding=(config.speech_decoder_postnet_kernel - 1) // 2,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(output_channels)
        self.activation = (nn.Tanh() if layer_index < config.speech_decoder_postnet_layers - 1 else None)
        self.dropout = nn.Dropout(config.speech_decoder_postnet_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.batch_norm(self.conv(hidden_states))
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        return self.dropout(hidden_states)


class SpeechT5SpeechDecoderPostnet(nn.Module):

    def __init__(self, config: NativeSpeechT5Config) -> None:
        super().__init__()
        self.config = config
        self.feat_out = nn.Linear(
            config.hidden_size,
            config.num_mel_bins * config.reduction_factor,
        )
        self.prob_out = nn.Linear(
            config.hidden_size,
            config.reduction_factor,
        )
        self.layers = nn.ModuleList([
            SpeechT5BatchNormConvLayer(config, index)
            for index in range(config.speech_decoder_postnet_layers)
        ])

    def postnet(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states.transpose(1, 2)
        for layer in self.layers:
            residual = layer(residual)
        return hidden_states + residual.transpose(1, 2)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        before = self.feat_out(hidden_states).view(
            hidden_states.shape[0],
            -1,
            self.config.num_mel_bins,
        )
        after = self.postnet(before)
        logits = self.prob_out(hidden_states).view(hidden_states.shape[0], -1)
        return before, after, logits


@dataclass(frozen=True, slots=True)
class SpeechT5Losses:
    total: Tensor
    spectrogram: Tensor
    stop: Tensor
    guided_attention: Tensor | None


class SpeechT5GuidedMultiheadAttentionLoss(nn.Module):

    def __init__(self, config: NativeSpeechT5Config) -> None:
        super().__init__()
        self.sigma = config.guided_attention_loss_sigma
        self.scale = config.guided_attention_loss_scale

    @staticmethod
    def _mask(
        input_length: Tensor,
        output_length: Tensor,
        sigma: float,
        device: torch.device,
    ) -> Tensor:
        input_positions = torch.arange(input_length, device=device)
        output_positions = torch.arange(output_length, device=device)
        output_grid, input_grid = torch.meshgrid(
            output_positions,
            input_positions,
            indexing="ij",
        )
        input_grid = input_grid.float() / input_length
        output_grid = output_grid.float() / output_length
        return 1.0 - torch.exp(-((input_grid - output_grid)**2) / (2.0 * sigma**2))

    def forward(
        self,
        attentions: Tensor,
        input_masks: Tensor,
        output_masks: Tensor,
    ) -> Tensor:
        guided_masks = attentions.new_zeros(
            input_masks.shape[0],
            output_masks.shape[1],
            input_masks.shape[1],
        )
        for index, (input_length, output_length) in enumerate(zip(input_masks.sum(-1), output_masks.sum(-1))):
            if int(input_length) <= 0 or int(output_length) <= 0:
                raise ValueError("Guided attention requires non-empty input and output sequences.")
            guided_masks[index, :output_length, :input_length] = self._mask(
                input_length,
                output_length,
                self.sigma,
                attentions.device,
            )
        valid = (output_masks.unsqueeze(-1)
                 & input_masks.unsqueeze(-2)).unsqueeze(1).to(device=attentions.device)
        selected = (guided_masks.unsqueeze(1) * attentions).masked_select(valid)
        if selected.numel() == 0:
            raise ValueError("Guided attention selected no valid alignment cells.")
        return self.scale * selected.mean()


class SpeechT5SpectrogramLoss(nn.Module):

    def __init__(self, config: NativeSpeechT5Config) -> None:
        super().__init__()
        self.config = config
        self.guided = (
            SpeechT5GuidedMultiheadAttentionLoss(config) if config.use_guided_attention_loss else None)

    def forward(
        self,
        attention_mask: Tensor,
        outputs_before_postnet: Tensor,
        outputs_after_postnet: Tensor,
        logits: Tensor,
        labels: Tensor,
        cross_attentions: tuple[Tensor | None, ...] | None,
    ) -> SpeechT5Losses:
        if labels.shape != outputs_before_postnet.shape:
            raise ValueError(
                "SpeechT5 predictions and labels must have the same shape; "
                f"found {tuple(outputs_before_postnet.shape)} and "
                f"{tuple(labels.shape)}.")
        padding_mask = labels != -100.0
        frame_mask = padding_mask[:, :, 0]
        if not bool((padding_mask == frame_mask.unsqueeze(-1)).all().item()):
            raise ValueError("SpeechT5 label padding must mask complete mel frames.")
        selected_labels = labels.masked_select(padding_mask)
        if selected_labels.numel() == 0:
            raise ValueError("SpeechT5 labels contain no supervised mel values.")
        spectrogram_loss = (
            F.l1_loss(
                outputs_after_postnet.masked_select(padding_mask),
                selected_labels,
            ) + F.l1_loss(
                outputs_before_postnet.masked_select(padding_mask),
                selected_labels,
            ))
        stop_labels = torch.cat(
            (
                (~frame_mask).to(dtype=logits.dtype),
                torch.ones(
                    frame_mask.shape[0],
                    1,
                    dtype=logits.dtype,
                    device=logits.device,
                ),
            ),
            dim=1,
        )[:, 1:].masked_select(frame_mask)
        selected_logits = logits.masked_select(frame_mask)
        stop_loss = F.binary_cross_entropy_with_logits(
            selected_logits,
            stop_labels,
            pos_weight=logits.new_tensor(5.0),
        )
        guided_loss = None
        if self.guided is not None:
            if not cross_attentions or any(attention is None for attention in cross_attentions):
                raise ValueError("Guided-attention training requires decoder cross attentions.")
            number_of_heads = self.config.guided_attention_loss_num_heads
            attentions = torch.cat(
                [
                    attention[:, :number_of_heads]  # type: ignore[index]
                    for attention in cross_attentions
                ],
                dim=1)
            output_masks = frame_mask
            if self.config.reduction_factor > 1:
                output_masks = output_masks[:, self.config.reduction_factor - 1::self.config.reduction_factor]
            guided_loss = self.guided(
                attentions,
                attention_mask == 1,
                output_masks,
            )
        total = spectrogram_loss + stop_loss
        if guided_loss is not None:
            total = total + guided_loss
        return SpeechT5Losses(
            total=total,
            spectrogram=spectrogram_loss,
            stop=stop_loss,
            guided_attention=guided_loss,
        )


class SpeechT5ForTextToSpeechModel(nn.Module):
    """Checkpoint-compatible SpeechT5 TTS acoustic model."""

    def __init__(self, config: NativeSpeechT5Config) -> None:
        super().__init__()
        self.config = config
        self.speecht5 = SpeechT5Model(config)
        self.speech_decoder_postnet = SpeechT5SpeechDecoderPostnet(config)
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
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                bound = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, SpeechT5ScaledPositionalEncoding):
            nn.init.ones_(module.alpha)

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose the tensor graph actually used by SpeechT5 synthesis."""
        if mode == "training":
            return (OptimizationCompileTarget(
                "acoustic_model.forward",
                self,
                "forward",
            ), )
        if mode == "inference":
            return (
                OptimizationCompileTarget(
                    "encoder.forward",
                    self.speecht5.encoder,
                    "forward",
                ),
                OptimizationCompileTarget(
                    "decoder.prenet.forward",
                    self.speecht5.decoder.prenet,
                    "forward",
                ),
                OptimizationCompileTarget(
                    "decoder.forward",
                    self.speecht5.decoder.wrapped_decoder,
                    "forward",
                ),
                OptimizationCompileTarget(
                    "postnet.forward",
                    self.speech_decoder_postnet,
                    "postnet",
                ),
            )
        raise ValueError("SpeechT5 compile targets require 'inference' or 'training' "
                         "mode.")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        decoder_input_values: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        encoder_outputs: SpeechT5EncoderOutput | None = None,
        past_key_values: SpeechT5DecoderCache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        speaker_embeddings: Tensor | None = None,
        labels: Tensor | None = None,
        stop_labels: Tensor | None = None,
    ) -> SpeechT5SpectrogramOutput:
        del stop_labels
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()
        if labels is not None:
            if labels.ndim != 3 or labels.shape[-1] != self.config.num_mel_bins:
                raise ValueError(
                    "SpeechT5 labels must have shape "
                    f"[batch, frames, {self.config.num_mel_bins}].")
            remainder = labels.shape[1] % self.config.reduction_factor
            if remainder:
                labels = F.pad(
                    labels,
                    (
                        0,
                        0,
                        0,
                        self.config.reduction_factor - remainder,
                    ),
                    value=-100.0,
                )
                if decoder_attention_mask is not None:
                    decoder_attention_mask = F.pad(
                        decoder_attention_mask,
                        (
                            0,
                            self.config.reduction_factor - remainder,
                        ),
                        value=0,
                    )
            if decoder_input_values is None:
                decoder_input_values, decoder_attention_mask = (
                    shift_spectrograms_right(
                        labels,
                        self.config.reduction_factor,
                        decoder_attention_mask,
                    ))
            if self.config.use_guided_attention_loss:
                output_attentions = True
            use_cache = False
        if decoder_input_values is None:
            raise ValueError("SpeechT5 requires decoder inputs or supervised labels.")
        outputs = self.speecht5(
            input_ids,
            attention_mask,
            decoder_input_values,
            decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            speaker_embeddings=speaker_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        before, after, logits = self.speech_decoder_postnet(outputs.last_hidden_state)
        losses = None
        if labels is not None:
            losses = SpeechT5SpectrogramLoss(self.config)(
                attention_mask,
                before,
                after,
                logits,
                labels,
                outputs.cross_attentions,
            )
        loss_mapping = None
        if losses is not None:
            loss_mapping = {
                "loss": losses.total,
                "spectrogram_loss": losses.spectrogram,
                "stop_loss": losses.stop,
            }
            if losses.guided_attention is not None:
                loss_mapping["guided_attention_loss"] = (losses.guided_attention)
        return SpeechT5SpectrogramOutput(
            loss=None if losses is None else losses.total,
            spectrogram=after,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            spectrogram_loss=(None if losses is None else losses.spectrogram),
            stop_loss=None if losses is None else losses.stop,
            guided_attention_loss=(None if losses is None else losses.guided_attention),
            losses=loss_mapping,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        speaker_embeddings: Tensor | None = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
        vocoder: nn.Module | None = None,
        output_cross_attentions: bool = False,
        return_output_lengths: bool = False,
    ):
        if speaker_embeddings is None:
            raise ValueError("SpeechT5 generation requires speaker embeddings.")
        if speaker_embeddings.shape[0] != input_ids.shape[0]:
            if speaker_embeddings.shape[0] == 1:
                speaker_embeddings = speaker_embeddings.expand(
                    input_ids.shape[0],
                    -1,
                )
            else:
                raise ValueError("Speaker embedding batch must be one or match text batch.")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()
        encoder_output = self.speecht5.encoder(
            input_ids,
            attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        encoded = encoder_output.last_hidden_state
        maximum_steps = max(
            1,
            int(encoded.shape[1] * maxlenratio / self.config.reduction_factor),
        )
        minimum_steps = max(
            0,
            int(encoded.shape[1] * minlenratio / self.config.reduction_factor),
        )
        batch_size = input_ids.shape[0]
        output_sequence = encoded.new_zeros(
            batch_size,
            1,
            self.config.num_mel_bins,
        )
        spectra: list[Tensor] = []
        attention_steps: list[Tensor] = []
        cache: SpeechT5DecoderCache | None = None
        completed: dict[int, Tensor] = {}
        for step in range(1, maximum_steps + 1):
            decoder_hidden = self.speecht5.decoder.prenet(
                output_sequence,
                speaker_embeddings,
            )
            decoder_output = self.speecht5.decoder.wrapped_decoder(
                decoder_hidden[:, -1:],
                None,
                encoder_hidden_states=encoded,
                encoder_attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                output_attentions=output_cross_attentions,
                output_hidden_states=False,
            )
            cache = decoder_output.past_key_values
            if output_cross_attentions:
                if not decoder_output.cross_attentions or any(value is None
                                                              for value in decoder_output.cross_attentions):
                    raise RuntimeError("SpeechT5 generation requested missing cross attentions.")
                attention_steps.append(
                    torch.stack(
                        [
                            value.squeeze(2)  # type: ignore[union-attr]
                            for value in decoder_output.cross_attentions
                        ],
                        dim=1))
            last_hidden = decoder_output.last_hidden_state.squeeze(1)
            spectrum = self.speech_decoder_postnet.feat_out(last_hidden).view(
                batch_size,
                self.config.reduction_factor,
                self.config.num_mel_bins,
            )
            spectra.append(spectrum)
            output_sequence = torch.cat(
                (output_sequence, spectrum[:, -1:, :]),
                dim=1,
            )
            probabilities = torch.sigmoid(self.speech_decoder_postnet.prob_out(last_hidden))
            if step < minimum_steps:
                continue
            if step < maximum_steps:
                finished = torch.where(probabilities.sum(dim=-1) >= threshold)[0].tolist()
            else:
                finished = list(range(batch_size))
            newly_finished = [index for index in finished if index not in completed]
            if newly_finished:
                stacked = torch.stack(spectra, dim=1).flatten(1, 2)
                stacked = self.speech_decoder_postnet.postnet(stacked)
                for index in newly_finished:
                    completed[index] = stacked[index]
            if len(completed) == batch_size:
                break
        spectrograms = [completed[index] for index in range(batch_size)]
        lengths = [int(value.shape[0]) for value in spectrograms]
        padded = nn.utils.rnn.pad_sequence(
            spectrograms,
            batch_first=True,
        )
        if vocoder is None:
            generated = padded if batch_size > 1 or return_output_lengths else (spectrograms[0])
            output_lengths = lengths
        else:
            waveforms = vocoder(padded)
            generated = waveforms
            factor = getattr(vocoder, "upsample_factor", None)
            if factor is None:
                factor = waveforms.shape[-1] // max(lengths)
            output_lengths = [length * int(factor) for length in lengths]
            if batch_size == 1 and not return_output_lengths:
                generated = generated[0]
        if return_output_lengths:
            result: tuple[Any, ...] = (generated, output_lengths)
        else:
            result = (generated, )
        if output_cross_attentions:
            attention = torch.stack(attention_steps, dim=3)
            result = (*result, attention)
        return result if len(result) > 1 else result[0]


class HifiGanResidualBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: tuple[int, ...],
        leaky_relu_slope: float,
    ) -> None:
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=value,
                padding=(kernel_size * value - value) // 2,
            ) for value in dilation
        ])
        self.convs2 = nn.ModuleList(
            [nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
            ) for _ in dilation])

    def forward(self, hidden_states: Tensor) -> Tensor:
        for first, second in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = F.leaky_relu(
                hidden_states,
                self.leaky_relu_slope,
            )
            hidden_states = first(hidden_states)
            hidden_states = F.leaky_relu(
                hidden_states,
                self.leaky_relu_slope,
            )
            hidden_states = second(hidden_states) + residual
        return hidden_states


class SpeechT5HifiGan(nn.Module):
    """Checkpoint-compatible SpeechT5 HiFi-GAN generator."""

    def __init__(self, config: NativeSpeechT5HifiGanConfig) -> None:
        super().__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1d(
            config.model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            padding=3,
        )
        self.upsampler = nn.ModuleList([
            nn.ConvTranspose1d(
                config.upsample_initial_channel // (2**index),
                config.upsample_initial_channel // (2**(index + 1)),
                kernel_size=kernel,
                stride=rate,
                padding=(kernel - rate) // 2,
            )
            for index, (rate, kernel) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes))
        ])
        blocks: list[HifiGanResidualBlock] = []
        for index in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2**(index + 1))
            for kernel, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                blocks.append(HifiGanResidualBlock(
                    channels,
                    kernel,
                    dilation,
                    config.leaky_relu_slope,
                ))
        self.resblocks = nn.ModuleList(blocks)
        self.conv_post = nn.Conv1d(
            channels,
            1,
            kernel_size=7,
            padding=3,
        )
        self.register_buffer("mean", torch.zeros(config.model_in_dim))
        self.register_buffer("scale", torch.ones(config.model_in_dim))

    @property
    def upsample_factor(self) -> int:
        return self.config.upsample_factor

    def forward(self, spectrogram: Tensor) -> Tensor:
        if spectrogram.ndim not in {2, 3}:
            raise ValueError("SpeechT5 HiFi-GAN expects [frames, mel] or [batch, frames, mel].")
        if spectrogram.shape[-1] != self.config.model_in_dim:
            raise ValueError("SpeechT5 HiFi-GAN mel dimension does not match its checkpoint.")
        if self.config.normalize_before:
            spectrogram = (spectrogram -
                           self.mean.to(dtype=spectrogram.dtype)) / self.scale.to(dtype=spectrogram.dtype)
        batched = spectrogram.ndim == 3
        if not batched:
            spectrogram = spectrogram.unsqueeze(0)
        hidden_states = self.conv_pre(spectrogram.transpose(1, 2))
        for index, upsampler in enumerate(self.upsampler):
            hidden_states = F.leaky_relu(
                hidden_states,
                self.config.leaky_relu_slope,
            )
            hidden_states = upsampler(hidden_states)
            residual = self.resblocks[index * self.num_kernels](hidden_states)
            for offset in range(1, self.num_kernels):
                residual = residual + self.resblocks[index * self.num_kernels + offset](hidden_states)
            hidden_states = residual / self.num_kernels
        hidden_states = F.leaky_relu(hidden_states)
        waveform = torch.tanh(self.conv_post(hidden_states)).squeeze(1)
        return waveform if batched else waveform[0]


__all__ = [
    "HifiGanResidualBlock",
    "SpeechT5DecoderCache",
    "SpeechT5ForTextToSpeechModel",
    "SpeechT5GuidedMultiheadAttentionLoss",
    "SpeechT5HifiGan",
    "SpeechT5Losses",
    "SpeechT5SpectrogramLoss",
    "SpeechT5SpectrogramOutput",
    "shift_spectrograms_right",
]
