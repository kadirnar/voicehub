"""VoiceHub-owned PyTorch SeamlessM4T-v2 speech-to-text implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from voicehub.models.asr_seamless_m4t_v2.native.configuration import SeamlessM4Tv2S2TConfig


def _right_padded_mask(
    attention_mask: Tensor | None,
    *,
    batch: int,
    length: int,
    device: torch.device,
) -> Tensor:
    if attention_mask is None:
        return torch.ones((batch, length), dtype=torch.bool, device=device)
    if not isinstance(attention_mask, Tensor):
        raise TypeError("`attention_mask` must be a PyTorch tensor.")
    if tuple(attention_mask.shape) != (batch, length):
        raise ValueError("`attention_mask` must match [batch, sequence].")
    if attention_mask.device != device:
        raise ValueError("The attention mask and inputs must share a device.")
    if not ((attention_mask == 0) | (attention_mask == 1)).all():
        raise ValueError("`attention_mask` must contain only zero and one.")
    result = attention_mask.to(dtype=torch.bool)
    if length > 1 and ((~result[:, :-1]) & result[:, 1:]).any():
        raise ValueError("`attention_mask` must describe right padding.")
    if not result.any(dim=1).all():
        raise ValueError("Every sequence must contain at least one valid item.")
    return result


def _additive_key_mask(mask: Tensor, *, dtype: torch.dtype) -> Tensor:
    return (~mask[:, None, None, :]).to(dtype=dtype) * torch.finfo(dtype).min


def _speech_attention_bias(
    mask: Tensor,
    *,
    dtype: torch.dtype,
    chunk_size: int,
    left_chunks: int,
) -> Tensor:
    batch, length = mask.shape
    invalid = ~mask[:, None, None, :]
    positions = torch.arange(length, device=mask.device)
    chunks = torch.div(positions, chunk_size, rounding_mode="floor")
    starts = torch.clamp(chunks - left_chunks, min=0) * chunk_size
    ends = torch.clamp((chunks + 1) * chunk_size, max=length)
    keys = positions.unsqueeze(0)
    chunk_invalid = ((keys < starts.unsqueeze(1)) | (keys >= ends.unsqueeze(1)))[None, None, :, :]
    invalid = invalid.expand(batch, 1, length, length) | chunk_invalid
    return invalid.to(dtype=dtype) * torch.finfo(dtype).min


def _causal_attention_bias(
    mask: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    _, length = mask.shape
    causal = torch.triu(
        torch.ones(
            (length, length),
            dtype=torch.bool,
            device=mask.device,
        ),
        diagonal=1,
    )
    invalid = (~mask[:, None, None, :]) | causal[None, None, :, :]
    return invalid.to(dtype=dtype) * torch.finfo(dtype).min


def _length_mask(
    lengths: Tensor,
    *,
    maximum: int,
) -> Tensor:
    positions = torch.arange(maximum, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


class SeamlessM4Tv2ConformerFeatureProjection(nn.Module):

    def __init__(self, config: SeamlessM4Tv2S2TConfig) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            config.feature_projection_input_dim,
            eps=config.layer_norm_eps,
        )
        self.projection = nn.Linear(
            config.feature_projection_input_dim,
            config.hidden_size,
        )
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        normalized = self.layer_norm(hidden_states.to(dtype=self.layer_norm.weight.dtype))
        return self.dropout(self.projection(normalized))


class SeamlessM4Tv2ConformerFeedForward(nn.Module):

    def __init__(
        self,
        config: SeamlessM4Tv2S2TConfig,
        *,
        activation: str | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.intermediate_dense = nn.Linear(
            config.hidden_size,
            config.speech_encoder_intermediate_size,
        )
        self.output_dense = nn.Linear(
            config.speech_encoder_intermediate_size,
            config.hidden_size,
        )
        self.intermediate_dropout = nn.Dropout(config.speech_encoder_dropout if dropout is None else dropout)
        self.output_dropout = nn.Dropout(config.speech_encoder_dropout if dropout is None else dropout)
        self.activation = (config.speech_encoder_hidden_act if activation is None else activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        if self.activation in {"swish", "silu"}:
            hidden_states = functional.silu(hidden_states)
        else:
            hidden_states = functional.relu(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        return self.output_dropout(self.output_dense(hidden_states))


class SeamlessM4Tv2ConformerConvolutionModule(nn.Module):

    def __init__(self, config: SeamlessM4Tv2S2TConfig) -> None:
        super().__init__()
        channels = config.hidden_size
        self.layer_norm = nn.LayerNorm(channels)
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            bias=False,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=config.conv_depthwise_kernel_size,
            groups=channels,
            bias=False,
        )
        self.depthwise_layer_norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        frame_mask: Tensor | None,
    ) -> Tensor:
        hidden_states = self.layer_norm(hidden_states)
        if frame_mask is not None:
            hidden_states = hidden_states.masked_fill(
                ~frame_mask.unsqueeze(-1),
                0.0,
            )
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = functional.glu(
            self.pointwise_conv1(hidden_states),
            dim=1,
        )
        hidden_states = functional.pad(
            hidden_states,
            (self.depthwise_conv.kernel_size[0] - 1, 0),
        )
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = functional.silu(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        return self.dropout(hidden_states).transpose(1, 2)


class SeamlessM4Tv2ConformerSelfAttention(nn.Module):

    def __init__(
        self,
        config: SeamlessM4Tv2S2TConfig,
        *,
        use_position_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = config.speech_encoder_attention_heads
        self.head_size = config.speech_head_dimension
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = config.speech_encoder_dropout
        self.left_max_position_embeddings = config.left_max_position_embeddings
        self.right_max_position_embeddings = config.right_max_position_embeddings
        self.distance_embedding = (
            nn.Embedding(
                self.left_max_position_embeddings + self.right_max_position_embeddings + 1,
                self.head_size,
            ) if use_position_embeddings else None)

    def _split(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.view(
            batch,
            length,
            self.num_heads,
            self.head_size,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_bias: Tensor | None,
    ) -> Tensor:
        query = self._split(self.linear_q(hidden_states))
        key = self._split(self.linear_k(hidden_states))
        value = self._split(self.linear_v(hidden_states))
        weights = torch.matmul(query, key.transpose(-2, -1))
        weights = weights / math.sqrt(self.head_size)
        if self.distance_embedding is not None:
            query_length = query.shape[2]
            key_length = key.shape[2]
            query_positions = torch.arange(
                query_length,
                device=hidden_states.device,
            ).unsqueeze(1)
            key_positions = torch.arange(
                key_length,
                device=hidden_states.device,
            ).unsqueeze(0)
            distance = torch.clamp(
                key_positions - query_positions,
                -self.left_max_position_embeddings,
                self.right_max_position_embeddings,
            )
            relative = self.distance_embedding(distance +
                                               self.left_max_position_embeddings, ).to(dtype=query.dtype)
            weights = weights + torch.einsum(
                "bhld,lrd->bhlr",
                query,
                relative,
            ) / math.sqrt(self.head_size)
        if attention_bias is not None:
            weights = weights + attention_bias
        probabilities = torch.softmax(
            weights,
            dim=-1,
            dtype=torch.float32,
        ).to(dtype=query.dtype)
        probabilities = functional.dropout(
            probabilities,
            p=self.dropout,
            training=self.training,
        )
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).contiguous().view(hidden_states.shape, )
        return self.linear_out(attended)


class SeamlessM4Tv2ConformerEncoderLayer(nn.Module):

    def __init__(self, config: SeamlessM4Tv2S2TConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.ffn1_layer_norm = nn.LayerNorm(hidden)
        self.ffn1 = SeamlessM4Tv2ConformerFeedForward(config)
        self.self_attn_layer_norm = nn.LayerNorm(hidden)
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(config)
        self.self_attn_dropout = nn.Dropout(config.speech_encoder_dropout)
        self.conv_module = SeamlessM4Tv2ConformerConvolutionModule(config)
        self.ffn2_layer_norm = nn.LayerNorm(hidden)
        self.ffn2 = SeamlessM4Tv2ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        hidden_states: Tensor,
        attention_bias: Tensor | None,
        frame_mask: Tensor | None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = residual + 0.5 * self.ffn1(self.ffn1_layer_norm(hidden_states))
        residual = hidden_states
        hidden_states = self.self_attn(
            self.self_attn_layer_norm(hidden_states),
            attention_bias=attention_bias,
        )
        hidden_states = residual + self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + self.conv_module(
            hidden_states,
            frame_mask=frame_mask,
        )
        residual = hidden_states
        hidden_states = residual + 0.5 * self.ffn2(self.ffn2_layer_norm(hidden_states))
        return self.final_layer_norm(hidden_states)


class SeamlessM4Tv2ConformerEncoder(nn.Module):

    def __init__(self, config: SeamlessM4Tv2S2TConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.speech_encoder_dropout)
        self.layers = nn.ModuleList(
            SeamlessM4Tv2ConformerEncoderLayer(config) for _ in range(config.speech_encoder_layers))
        self.layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        *,
        frame_mask: Tensor,
    ) -> Tensor:
        hidden_states = hidden_states.masked_fill(
            ~frame_mask.unsqueeze(-1),
            0.0,
        )
        attention_bias = _speech_attention_bias(
            frame_mask,
            dtype=hidden_states.dtype,
            chunk_size=self.config.speech_encoder_chunk_size,
            left_chunks=self.config.speech_encoder_left_chunk_num,
        )
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers:
            drop_layer = False
            if self.training and self.config.speech_encoder_layerdrop > 0.0:
                drop_layer = bool(
                    torch.rand(
                        (),
                        device=hidden_states.device,
                    ) < self.config.speech_encoder_layerdrop)
            if drop_layer:
                continue
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    attention_bias,
                    frame_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_bias,
                    frame_mask,
                )
        return self.layer_norm(hidden_states)


class SeamlessM4Tv2ConformerAdapterLayer(nn.Module):

    def __init__(self, config: SeamlessM4Tv2S2TConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.kernel_size = config.adaptor_kernel_size
        self.stride = config.adaptor_stride
        self.residual_layer_norm = nn.LayerNorm(hidden)
        self.residual_conv = nn.Conv1d(
            hidden,
            2 * hidden,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        self.self_attn_layer_norm = nn.LayerNorm(hidden)
        self.self_attn_conv = nn.Conv1d(
            hidden,
            2 * hidden,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(
            config,
            use_position_embeddings=False,
        )
        self.self_attn_dropout = nn.Dropout(config.adaptor_dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden)
        self.ffn = SeamlessM4Tv2ConformerFeedForward(
            config,
            activation="relu",
            dropout=config.adaptor_dropout,
        )

    def forward(
        self,
        hidden_states: Tensor,
        frame_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residual = functional.glu(
            self.residual_conv(self.residual_layer_norm(hidden_states).transpose(1, 2)),
            dim=1,
        ).transpose(1, 2)
        hidden_states = functional.glu(
            self.self_attn_conv(self.self_attn_layer_norm(hidden_states).transpose(1, 2)),
            dim=1,
        ).transpose(1, 2)
        lengths = frame_mask.sum(dim=-1, dtype=torch.long)
        padding = self.kernel_size // 2
        lengths = (lengths + 2 * padding - self.kernel_size) // self.stride + 1
        frame_mask = _length_mask(
            lengths,
            maximum=hidden_states.shape[1],
        )
        bias = _additive_key_mask(frame_mask, dtype=hidden_states.dtype)
        hidden_states = residual + self.self_attn_dropout(
            self.self_attn(
                hidden_states,
                attention_bias=bias,
            ))
        hidden_states = hidden_states + self.ffn(self.ffn_layer_norm(hidden_states))
        return hidden_states, frame_mask


class SeamlessM4Tv2ConformerAdapter(nn.Module):

    def __init__(self, config: SeamlessM4Tv2S2TConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            SeamlessM4Tv2ConformerAdapterLayer(config) for _ in range(config.num_adapter_layers))

    def forward(
        self,
        hidden_states: Tensor,
        frame_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        for layer in self.layers:
            hidden_states, frame_mask = layer(hidden_states, frame_mask)
        return hidden_states, frame_mask


class SeamlessM4Tv2SpeechEncoder(nn.Module):

    def __init__(self, config: SeamlessM4Tv2S2TConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_projection = SeamlessM4Tv2ConformerFeatureProjection(config)
        self.encoder = SeamlessM4Tv2ConformerEncoder(config)
        self.intermediate_ffn = SeamlessM4Tv2ConformerFeedForward(
            config,
            activation="relu",
            dropout=0.0,
        )
        self.adapter = (SeamlessM4Tv2ConformerAdapter(config) if config.add_adapter else None)
        self.inner_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        input_features: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(input_features, Tensor):
            raise TypeError("`input_features` must be a PyTorch tensor.")
        if input_features.ndim != 3:
            raise ValueError("`input_features` must have shape [batch, frames, channels].")
        if input_features.shape[-1] != self.config.feature_projection_input_dim:
            raise ValueError(
                "SeamlessM4T-v2 expects "
                f"{self.config.feature_projection_input_dim} stacked mel "
                f"features, found {input_features.shape[-1]}.")
        if not input_features.is_floating_point():
            raise TypeError("`input_features` must be floating-point.")
        batch, frames, _ = input_features.shape
        frame_mask = _right_padded_mask(
            attention_mask,
            batch=batch,
            length=frames,
            device=input_features.device,
        )
        hidden_states = self.feature_projection(input_features)
        hidden_states = self.encoder(
            hidden_states,
            frame_mask=frame_mask,
        )
        hidden_states = hidden_states + 0.5 * self.intermediate_ffn(hidden_states)
        if self.adapter is not None:
            hidden_states, frame_mask = self.adapter(
                hidden_states,
                frame_mask,
            )
        hidden_states = self.inner_layer_norm(hidden_states)
        return hidden_states, frame_mask


class SeamlessM4Tv2ScaledWordEmbedding(nn.Embedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        *,
        embed_scale: float,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.embed_scale = embed_scale

    def forward(self, input_ids: Tensor) -> Tensor:
        return super().forward(input_ids) * self.embed_scale


class SeamlessM4Tv2SinusoidalPositionalEmbedding(nn.Module):
    """Non-persistent fairseq-style sinusoidal decoder positions."""

    def __init__(
        self,
        num_positions: int,
        embedding_dim: int,
        padding_idx: int,
    ) -> None:
        super().__init__()
        self.offset = 2
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.register_buffer(
            "weights",
            self._weights(num_positions + self.offset),
            persistent=False,
        )

    def _weights(self, length: int) -> Tensor:
        half = self.embedding_dim // 2
        scale = math.log(10_000.0) / (half - 1)
        frequencies = torch.exp(torch.arange(half, dtype=torch.float32) * -scale)
        positions = torch.arange(
            length,
            dtype=torch.float32,
        ).unsqueeze(1)
        result = torch.cat(
            (
                torch.sin(positions * frequencies.unsqueeze(0)),
                torch.cos(positions * frequencies.unsqueeze(0)),
            ),
            dim=1,
        )
        if self.embedding_dim % 2:
            result = functional.pad(result, (0, 1))
        result[self.padding_idx] = 0
        return result

    def forward(self, input_ids: Tensor) -> Tensor:
        visible = input_ids.ne(self.padding_idx).to(dtype=torch.long)
        positions = torch.cumsum(visible, dim=1) * visible + self.padding_idx
        maximum = int(positions.max().item()) if positions.numel() else 0
        if maximum >= self.weights.shape[0]:
            self.weights = self._weights(maximum + self.offset + 1, ).to(
                device=self.weights.device,
                dtype=self.weights.dtype,
            )
        return self.weights.index_select(
            0,
            positions.reshape(-1),
        ).view(*positions.shape, self.embedding_dim).detach()


class SeamlessM4Tv2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dimension = hidden_size // num_heads
        self.scaling = self.head_dimension**-0.5
        self.dropout = dropout
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _split(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.view(
            batch,
            length,
            self.num_heads,
            self.head_dimension,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        key_value_states: Tensor | None = None,
        attention_bias: Tensor | None = None,
    ) -> Tensor:
        source = hidden_states if key_value_states is None else key_value_states
        query = self._split(self.q_proj(hidden_states)) * self.scaling
        key = self._split(self.k_proj(source))
        value = self._split(self.v_proj(source))
        scores = torch.matmul(query, key.transpose(-2, -1))
        if attention_bias is not None:
            scores = scores + attention_bias
        probabilities = torch.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        ).to(dtype=scores.dtype)
        probabilities = functional.dropout(
            probabilities,
            p=self.dropout,
            training=self.training,
        )
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).contiguous().view(hidden_states.shape, )
        return self.out_proj(attended)


class SeamlessM4Tv2FeedForwardNetwork(nn.Module):

    def __init__(
        self,
        config: SeamlessM4Tv2S2TConfig,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.activation_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dropout(functional.relu(self.fc1(hidden_states)))
        return self.fc2(hidden_states)


class SeamlessM4Tv2DecoderLayer(nn.Module):

    def __init__(self, config: SeamlessM4Tv2S2TConfig) -> None:
        super().__init__()
        self.self_attn = SeamlessM4Tv2Attention(
            config.hidden_size,
            config.decoder_attention_heads,
            config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.cross_attention = SeamlessM4Tv2Attention(
            config.hidden_size,
            config.decoder_attention_heads,
            config.attention_dropout,
        )
        self.cross_attention_layer_norm = nn.LayerNorm(config.hidden_size)
        self.ffn = SeamlessM4Tv2FeedForwardNetwork(config)
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.ffn_dropout = nn.Dropout(config.activation_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        self_attention_bias: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_bias: Tensor,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = residual + self.attn_dropout(
            self.self_attn(
                self.self_attn_layer_norm(hidden_states),
                attention_bias=self_attention_bias,
            ))
        residual = hidden_states
        hidden_states = residual + self.attn_dropout(
            self.cross_attention(
                self.cross_attention_layer_norm(hidden_states),
                key_value_states=encoder_hidden_states,
                attention_bias=encoder_attention_bias,
            ))
        residual = hidden_states
        hidden_states = residual + self.ffn_dropout(self.ffn(self.ffn_layer_norm(hidden_states)))
        return hidden_states


class SeamlessM4Tv2Decoder(nn.Module):

    def __init__(
        self,
        config: SeamlessM4Tv2S2TConfig,
        *,
        embed_tokens: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        self.embed_tokens = SeamlessM4Tv2ScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=scale,
        )
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight
        self.embed_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(SeamlessM4Tv2DecoderLayer(config) for _ in range(config.decoder_layers))
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = config.dropout
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise ValueError("`decoder_input_ids` must have shape [batch, tokens].")
        if input_ids.dtype != torch.long:
            raise TypeError("`decoder_input_ids` must use torch.long.")
        if input_ids.shape[0] != encoder_hidden_states.shape[0]:
            raise ValueError("Decoder and encoder batch sizes must match.")
        if input_ids.numel() and (input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size):
            raise ValueError("`decoder_input_ids` contains an invalid token ID.")
        decoder_mask = _right_padded_mask(
            attention_mask,
            batch=input_ids.shape[0],
            length=input_ids.shape[1],
            device=input_ids.device,
        )
        self_bias = _causal_attention_bias(
            decoder_mask,
            dtype=encoder_hidden_states.dtype,
        )
        encoder_bias = _additive_key_mask(
            encoder_attention_mask,
            dtype=encoder_hidden_states.dtype,
        )
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states + self.embed_positions(input_ids).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        hidden_states = functional.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        for layer in self.layers:
            drop_layer = False
            if self.training and self.config.decoder_layerdrop > 0.0:
                drop_layer = bool(
                    torch.rand(
                        (),
                        device=hidden_states.device,
                    ) < self.config.decoder_layerdrop)
            if drop_layer:
                continue
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    self_bias,
                    encoder_hidden_states,
                    encoder_bias,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    self_bias,
                    encoder_hidden_states,
                    encoder_bias,
                )
        return self.layer_norm(hidden_states)


@dataclass(slots=True)
class SeamlessM4Tv2S2TOutput:
    loss: Tensor | None
    logits: Tensor
    encoder_last_hidden_state: Tensor
    encoder_attention_mask: Tensor


def shift_tokens_right(
    labels: Tensor,
    *,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> Tensor:
    if not isinstance(labels, Tensor) or labels.ndim != 2:
        raise ValueError("`labels` must have shape [batch, tokens].")
    if labels.dtype != torch.long:
        raise TypeError("`labels` must use torch.long.")
    shifted = labels.new_full(labels.shape, pad_token_id)
    shifted[:, 1:] = labels[:, :-1]
    shifted[:, 0] = decoder_start_token_id
    return shifted.masked_fill(shifted == -100, pad_token_id)


class SeamlessM4Tv2ForSpeechToText(nn.Module):
    """Exact native S2T subset with tied decoder input/output embeddings."""

    def __init__(
        self,
        config: SeamlessM4Tv2S2TConfig,
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(config, SeamlessM4Tv2S2TConfig):
            raise TypeError("`config` must be SeamlessM4Tv2S2TConfig.")
        if not isinstance(initialize, bool):
            raise TypeError("`initialize` must be a boolean.")
        self.config = config
        self.shared = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.speech_encoder = SeamlessM4Tv2SpeechEncoder(config)
        self.text_decoder = SeamlessM4Tv2Decoder(
            config,
            embed_tokens=self.shared,
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.tie_weights()
        if initialize:
            self._initialize_weights()

    def tie_weights(self) -> None:
        self.text_decoder.embed_tokens.weight = self.shared.weight
        self.lm_head.weight = self.shared.weight

    @torch.no_grad()
    def _initialize_weights(self) -> None:
        for module in self.modules():
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
                    module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    bound = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.tie_weights()

    def gradient_checkpointing_enable(self) -> None:
        self.speech_encoder.encoder.gradient_checkpointing = True
        self.text_decoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.speech_encoder.encoder.gradient_checkpointing = False
        self.text_decoder.gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self) -> bool:
        return (
            self.speech_encoder.encoder.gradient_checkpointing or self.text_decoder.gradient_checkpointing)

    def encode(
        self,
        input_features: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.speech_encoder(
            input_features,
            attention_mask=attention_mask,
        )

    def decode(
        self,
        decoder_input_ids: Tensor,
        *,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
        decoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden_states = self.text_decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return self.lm_head(hidden_states)

    def forward(
        self,
        input_features: Tensor,
        *,
        attention_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> SeamlessM4Tv2S2TOutput:
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels,
                pad_token_id=self.config.pad_token_id,
                decoder_start_token_id=self.config.decoder_start_token_id,
            )
        if labels is not None and decoder_attention_mask is None:
            decoder_attention_mask = labels.ne(-100)
        if decoder_input_ids is None:
            raise ValueError("`decoder_input_ids` or `labels` is required for S2T forward.")
        encoder_hidden_states, encoder_mask = self.encode(
            input_features,
            attention_mask=attention_mask,
        )
        logits = self.decode(
            decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        loss = None
        if labels is not None:
            if tuple(labels.shape) != tuple(logits.shape[:2]):
                raise ValueError("`labels` must match [batch, decoder_tokens].")
            loss = functional.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                labels.to(device=logits.device).reshape(-1),
                ignore_index=-100,
            )
        return SeamlessM4Tv2S2TOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_attention_mask=encoder_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        input_features: Tensor,
        *,
        attention_mask: Tensor | None,
        language_token_id: int,
        max_new_tokens: int | None = None,
    ) -> Tensor:
        if self.training:
            raise RuntimeError("Call `eval()` before SeamlessM4T-v2 generation.")
        if (isinstance(language_token_id, bool) or not isinstance(language_token_id, int) or
                not 0 <= language_token_id < self.config.vocab_size):
            raise ValueError("`language_token_id` is outside the vocabulary.")
        maximum = (self.config.max_new_tokens if max_new_tokens is None else max_new_tokens)
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError("`max_new_tokens` must be a positive integer.")
        encoder_hidden, encoder_mask = self.encode(
            input_features,
            attention_mask=attention_mask,
        )
        sequences = torch.full(
            (input_features.shape[0], 1),
            language_token_id,
            dtype=torch.long,
            device=input_features.device,
        )
        finished = torch.zeros(
            input_features.shape[0],
            dtype=torch.bool,
            device=input_features.device,
        )
        for _ in range(maximum):
            logits = self.decode(
                sequences,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_mask,
            )
            next_tokens = logits[:, -1].argmax(dim=-1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, self.config.pad_token_id),
                next_tokens,
            )
            sequences = torch.cat(
                (sequences, next_tokens.unsqueeze(1)),
                dim=1,
            )
            finished = finished | next_tokens.eq(self.config.eos_token_id)
            if bool(finished.all()):
                break
        return sequences


__all__ = [
    "SeamlessM4Tv2ForSpeechToText",
    "SeamlessM4Tv2S2TOutput",
    "SeamlessM4Tv2SpeechEncoder",
    "shift_tokens_right",
]
