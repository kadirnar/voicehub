"""PyTorch-only Whisper encoder-decoder architecture owned by VoiceHub.

Architecture semantics and tensor names were independently implemented after
reviewing OpenAI Whisper ``model.py`` at revision
``04f449b8a437f1bbd3dba5c9f826aca972e7709a`` and Hugging Face Transformers'
Whisper implementation at revision
``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``.  No upstream runtime code is
imported or executed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.asr_whisper_native.native.configuration import WhisperConfig


def whisper_sinusoids(
    length: int,
    channels: int,
    *,
    max_timescale: float = 10_000.0,
) -> Tensor:
    """Build OpenAI-compatible sinusoidal encoder positions in float32."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("`length` must be a positive integer.")
    if isinstance(channels, bool) or not isinstance(channels, int) or channels < 2:
        raise ValueError("`channels` must be an integer of at least two.")
    if channels % 2:
        raise ValueError("`channels` must be even.")
    half = channels // 2
    denominator = max(half - 1, 1)
    increment = math.log(max_timescale) / denominator
    inverse_timescales = torch.exp(-increment * torch.arange(half, dtype=torch.float32))
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    scaled = positions * inverse_timescales.unsqueeze(0)
    return torch.cat((scaled.sin(), scaled.cos()), dim=1)


class Float32LayerNorm(nn.LayerNorm):
    """Layer normalization with float32 accumulation for low-precision
    inputs."""

    def forward(self, value: Tensor) -> Tensor:
        if value.dtype not in (torch.float16, torch.bfloat16):
            return super().forward(value)
        normalized = functional.layer_norm(
            value.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        )
        return normalized.to(dtype=value.dtype)


class _GELUNew(nn.Module):
    """Tanh GELU variant accepted by compatible Whisper configurations."""

    def forward(self, value: Tensor) -> Tensor:
        coefficient = math.sqrt(2.0 / math.pi)
        return 0.5 * value * (1.0 + torch.tanh(coefficient * (value + 0.044715 * value.pow(3))))


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "gelu_new":
        return _GELUNew()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported Whisper activation {name!r}.")


@dataclass(frozen=True)
class WhisperAttentionCache:
    """Projected keys and values for one attention operation."""

    key: Tensor
    value: Tensor

    def __post_init__(self) -> None:
        if self.key.ndim != 4 or self.value.ndim != 4:
            raise ValueError("Attention cache tensors must have rank four.")
        if self.key.shape != self.value.shape:
            raise ValueError("Attention cache key and value shapes must match.")

    @property
    def sequence_length(self) -> int:
        return self.key.shape[-2]


@dataclass(frozen=True)
class WhisperLayerCache:
    """Self- and cross-attention caches for one decoder layer."""

    self_attention: WhisperAttentionCache
    cross_attention: WhisperAttentionCache

    @property
    def sequence_length(self) -> int:
        return self.self_attention.sequence_length


WhisperDecoderCache = tuple[WhisperLayerCache, ...]


@dataclass(frozen=True)
class WhisperDecoderOutput:
    """Cache-friendly result of :meth:`WhisperModel.decode`."""

    logits: Tensor
    last_hidden_state: Tensor
    past_key_values: WhisperDecoderCache | None


@dataclass(frozen=True)
class WhisperOutput:
    """Native sequence-to-sequence forward result."""

    logits: Tensor
    encoder_last_hidden_state: Tensor
    decoder_last_hidden_state: Tensor
    loss: Tensor | None = None
    past_key_values: WhisperDecoderCache | None = None


def _validate_key_mask(
    mask: Tensor | None,
    *,
    batch_size: int,
    key_length: int,
    name: str,
    device: torch.device,
) -> Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, Tensor):
        raise TypeError(f"`{name}` must be a PyTorch tensor.")
    if mask.ndim != 2 or tuple(mask.shape) != (batch_size, key_length):
        raise ValueError(
            f"`{name}` must have shape [{batch_size}, {key_length}]; "
            f"found {tuple(mask.shape)}.")
    if mask.device != device:
        raise ValueError(f"`{name}` must be on the same device as the model input.")
    return mask.to(dtype=torch.bool)


class WhisperAttention(nn.Module):
    """Multi-head attention with stable probabilities and explicit caching."""

    def __init__(
        self,
        width: int,
        heads: int,
        *,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.attention_dropout = attention_dropout
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def _split_heads(self, value: Tensor) -> Tensor:
        batch, steps, _ = value.shape
        return value.reshape(
            batch,
            steps,
            self.heads,
            self.head_width,
        ).transpose(1, 2)

    def _validate_cache(
        self,
        cache: WhisperAttentionCache,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        expected_prefix = (batch_size, self.heads)
        if tuple(cache.key.shape[:2]) != expected_prefix:
            raise ValueError("Attention cache batch/head dimensions do not match the input.")
        if cache.key.shape[-1] != self.head_width:
            raise ValueError("Attention cache head width does not match the model.")
        if cache.key.device != device or cache.value.device != device:
            raise ValueError("Attention cache tensors must be on the input device.")
        if cache.key.dtype != cache.value.dtype:
            raise ValueError("Attention cache key and value dtypes must match.")

    def forward(
        self,
        hidden_states: Tensor,
        *,
        source_states: Tensor | None = None,
        key_mask: Tensor | None = None,
        causal_offset: int | None = None,
        past_key_value: WhisperAttentionCache | None = None,
        static_key_value: bool = False,
        use_cache: bool = False,
    ) -> tuple[Tensor, WhisperAttentionCache | None]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.width:
            raise ValueError("Whisper attention input must have shape [batch, time, width].")
        batch_size, query_length, _ = hidden_states.shape
        query = self._split_heads(self.query(hidden_states))

        if past_key_value is not None:
            self._validate_cache(
                past_key_value,
                batch_size=batch_size,
                device=hidden_states.device,
            )

        if static_key_value and past_key_value is not None:
            key = past_key_value.key
            value = past_key_value.value
        else:
            projection_source = (hidden_states if source_states is None else source_states)
            if (projection_source.ndim != 3 or projection_source.shape[0] != batch_size or
                    projection_source.shape[-1] != self.width):
                raise ValueError("Attention source must have shape [batch, time, width].")
            key = self._split_heads(self.key(projection_source))
            value = self._split_heads(self.value(projection_source))
            if past_key_value is not None and not static_key_value:
                key = torch.cat((past_key_value.key, key), dim=-2)
                value = torch.cat((past_key_value.value, value), dim=-2)

        key_length = key.shape[-2]
        key_mask = _validate_key_mask(
            key_mask,
            batch_size=batch_size,
            key_length=key_length,
            name="key_mask",
            device=hidden_states.device,
        )

        scale = self.head_width**-0.25
        scores = (query * scale) @ (key * scale).transpose(-1, -2)
        scores = scores.float()

        if causal_offset is not None:
            if causal_offset < 0:
                raise ValueError("`causal_offset` cannot be negative.")
            query_positions = (torch.arange(query_length, device=scores.device) + causal_offset)
            key_positions = torch.arange(key_length, device=scores.device)
            blocked = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            scores = scores.masked_fill(
                blocked.unsqueeze(0).unsqueeze(0),
                -torch.inf,
            )
        if key_mask is not None:
            scores = scores.masked_fill(
                ~key_mask[:, None, None, :],
                -torch.inf,
            )

        probabilities = torch.softmax(scores, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0).to(query.dtype)
        probabilities = functional.dropout(
            probabilities,
            p=self.attention_dropout,
            training=self.training,
        )
        attended = probabilities @ value
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch_size,
            query_length,
            self.width,
        )
        output = self.out(attended)
        present = WhisperAttentionCache(key=key, value=value) if use_cache else None
        return output, present


class WhisperResidualAttentionBlock(nn.Module):
    """Pre-normalized Whisper transformer block."""

    def __init__(
        self,
        config: WhisperConfig,
        *,
        heads: int,
        ffn_width: int,
        cross_attention: bool,
    ) -> None:
        super().__init__()
        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout
        self.attn = WhisperAttention(
            config.d_model,
            heads,
            attention_dropout=config.attention_dropout,
        )
        self.attn_ln = Float32LayerNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )
        self.cross_attn = (
            WhisperAttention(
                config.d_model,
                heads,
                attention_dropout=config.attention_dropout,
            ) if cross_attention else None)
        self.cross_attn_ln = (
            Float32LayerNorm(
                config.d_model,
                eps=config.layer_norm_eps,
            ) if cross_attention else None)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, ffn_width),
            _activation(config.activation_function),
            nn.Linear(ffn_width, config.d_model),
        )
        self.mlp_ln = Float32LayerNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )

    def _residual_dropout(self, value: Tensor) -> Tensor:
        return functional.dropout(value, p=self.dropout, training=self.training)

    def forward_encoder(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
    ) -> Tensor:
        attention_output, _ = self.attn(
            self.attn_ln(hidden_states),
            key_mask=attention_mask,
        )
        hidden_states = hidden_states + self._residual_dropout(attention_output)
        mlp_hidden = self.mlp[0](self.mlp_ln(hidden_states))
        mlp_hidden = self.mlp[1](mlp_hidden)
        mlp_hidden = functional.dropout(
            mlp_hidden,
            p=self.activation_dropout,
            training=self.training,
        )
        mlp_hidden = self.mlp[2](mlp_hidden)
        return hidden_states + self._residual_dropout(mlp_hidden)

    def forward_decoder(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        *,
        self_attention_mask: Tensor | None,
        encoder_attention_mask: Tensor | None,
        causal_offset: int,
        past_key_value: WhisperLayerCache | None,
        use_cache: bool,
    ) -> tuple[Tensor, WhisperLayerCache | None]:
        self_past = (None if past_key_value is None else past_key_value.self_attention)
        self_output, self_present = self.attn(
            self.attn_ln(hidden_states),
            key_mask=self_attention_mask,
            causal_offset=causal_offset,
            past_key_value=self_past,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self._residual_dropout(self_output)

        if self.cross_attn is None or self.cross_attn_ln is None:
            raise RuntimeError("Decoder block does not define cross-attention.")
        cross_past = (None if past_key_value is None else past_key_value.cross_attention)
        if (cross_past is not None and cross_past.sequence_length != encoder_hidden_states.shape[1]):
            raise ValueError("Cross-attention cache length does not match the encoder output.")
        cross_output, cross_present = self.cross_attn(
            self.cross_attn_ln(hidden_states),
            source_states=encoder_hidden_states,
            key_mask=encoder_attention_mask,
            past_key_value=cross_past,
            static_key_value=True,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self._residual_dropout(cross_output)

        mlp_hidden = self.mlp[0](self.mlp_ln(hidden_states))
        mlp_hidden = self.mlp[1](mlp_hidden)
        mlp_hidden = functional.dropout(
            mlp_hidden,
            p=self.activation_dropout,
            training=self.training,
        )
        mlp_hidden = self.mlp[2](mlp_hidden)
        hidden_states = hidden_states + self._residual_dropout(mlp_hidden)

        if not use_cache:
            return hidden_states, None
        if self_present is None or cross_present is None:
            raise RuntimeError("Attention cache was not returned when requested.")
        return hidden_states, WhisperLayerCache(
            self_attention=self_present,
            cross_attention=cross_present,
        )


class WhisperEncoder(nn.Module):
    """Convolutional log-mel frontend followed by Transformer encoder
    blocks."""

    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            config.num_mel_bins,
            config.d_model,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.register_buffer(
            "positional_embedding",
            whisper_sinusoids(config.max_source_positions, config.d_model),
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            WhisperResidualAttentionBlock(
                config,
                heads=config.encoder_attention_heads,
                ffn_width=config.encoder_ffn_dim,
                cross_attention=False,
            ) for _ in range(config.encoder_layers))
        self.ln_post = Float32LayerNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )

    def downsample_attention_mask(
        self,
        attention_mask: Tensor | None,
        *,
        batch_size: int,
        input_frames: int,
        device: torch.device,
    ) -> Tensor | None:
        mask = _validate_key_mask(
            attention_mask,
            batch_size=batch_size,
            key_length=input_frames,
            name="attention_mask",
            device=device,
        )
        return None if mask is None else mask[:, ::2]

    def forward(
        self,
        input_features: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(input_features, Tensor):
            raise TypeError("`input_features` must be a PyTorch tensor.")
        if input_features.ndim != 3:
            raise ValueError("`input_features` must have shape [batch, mel_bins, frames].")
        if input_features.shape[1] != self.config.num_mel_bins:
            raise ValueError(
                "`input_features` has the wrong mel-bin dimension: expected "
                f"{self.config.num_mel_bins}, found {input_features.shape[1]}.")
        if input_features.shape[0] < 1 or input_features.shape[2] < 1:
            raise ValueError("Whisper input batches and frame sequences cannot be empty.")
        if not input_features.is_floating_point():
            raise TypeError("`input_features` must use a floating-point dtype.")

        batch_size, _, input_frames = input_features.shape
        if input_features.device != self.conv1.weight.device:
            raise ValueError("`input_features` must be on the model device.")
        if input_features.dtype != self.conv1.weight.dtype:
            input_features = input_features.to(dtype=self.conv1.weight.dtype)
        hidden_states = functional.gelu(self.conv1(input_features))
        hidden_states = functional.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.transpose(1, 2)
        sequence_length = hidden_states.shape[1]
        if sequence_length > self.config.max_source_positions:
            raise ValueError(
                "Whisper input is longer than the configured audio context: "
                f"{sequence_length} positions exceed "
                f"{self.config.max_source_positions}.")

        encoder_mask = self.downsample_attention_mask(
            attention_mask,
            batch_size=batch_size,
            input_frames=input_frames,
            device=hidden_states.device,
        )
        if encoder_mask is not None and encoder_mask.shape[1] != sequence_length:
            raise RuntimeError("Downsampled attention mask has an invalid length.")

        positions = self.positional_embedding[:sequence_length]
        hidden_states = hidden_states + positions.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        hidden_states = functional.dropout(
            hidden_states,
            p=self.config.dropout,
            training=self.training,
        )
        for block in self.blocks:
            drop_layer = self.training and bool(
                self.config.encoder_layerdrop and torch.rand(
                    (), device=hidden_states.device) < self.config.encoder_layerdrop)
            if drop_layer:
                continue
            hidden_states = block.forward_encoder(
                hidden_states,
                attention_mask=encoder_mask,
            )
        return self.ln_post(hidden_states)


class WhisperDecoder(nn.Module):
    """Autoregressive text decoder with reusable self/cross KV caches."""

    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_embedding = nn.Parameter(torch.empty(config.max_target_positions, config.d_model))
        self.blocks = nn.ModuleList(
            WhisperResidualAttentionBlock(
                config,
                heads=config.decoder_attention_heads,
                ffn_width=config.decoder_ffn_dim,
                cross_attention=True,
            ) for _ in range(config.decoder_layers))
        self.ln = Float32LayerNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )

    def _past_length(
        self,
        past_key_values: WhisperDecoderCache | None,
        *,
        batch_size: int,
    ) -> int:
        if past_key_values is None:
            return 0
        if len(past_key_values) != len(self.blocks):
            raise ValueError("`past_key_values` must contain one cache per decoder layer.")
        lengths = {layer.sequence_length for layer in past_key_values}
        if len(lengths) != 1:
            raise ValueError("Decoder layer caches have inconsistent lengths.")
        for layer in past_key_values:
            if layer.self_attention.key.shape[0] != batch_size:
                raise ValueError("Decoder cache batch size does not match input IDs.")
        return next(iter(lengths))

    def _decoder_mask(
        self,
        attention_mask: Tensor | None,
        *,
        batch_size: int,
        current_length: int,
        past_length: int,
        device: torch.device,
    ) -> Tensor | None:
        if attention_mask is None:
            return None
        if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
            raise ValueError("`decoder_attention_mask` must have shape [batch, tokens].")
        if attention_mask.shape[0] != batch_size:
            raise ValueError("Decoder attention mask batch size does not match.")
        if attention_mask.device != device:
            raise ValueError("Decoder attention mask must be on the input device.")
        total_length = past_length + current_length
        if attention_mask.shape[1] == current_length and past_length:
            prefix = torch.ones(
                batch_size,
                past_length,
                dtype=torch.bool,
                device=device,
            )
            attention_mask = torch.cat(
                (prefix, attention_mask.to(device=device, dtype=torch.bool)),
                dim=1,
            )
        return _validate_key_mask(
            attention_mask,
            batch_size=batch_size,
            key_length=total_length,
            name="decoder_attention_mask",
            device=device,
        )

    def forward(
        self,
        input_ids: Tensor,
        encoder_hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: WhisperDecoderCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, WhisperDecoderCache | None]:
        if not isinstance(input_ids, Tensor):
            raise TypeError("`input_ids` must be a PyTorch tensor.")
        if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("`input_ids` must have non-empty [batch, tokens] shape.")
        if (input_ids.dtype == torch.bool or input_ids.is_floating_point() or input_ids.is_complex()):
            raise TypeError("`input_ids` must use an integer dtype.")
        if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
            raise ValueError("An input token ID is outside the Whisper vocabulary.")
        if (encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != input_ids.shape[0] or
                encoder_hidden_states.shape[-1] != self.config.d_model):
            raise ValueError("`encoder_hidden_states` must have shape [batch, audio, d_model].")
        if encoder_hidden_states.device != input_ids.device:
            raise ValueError("Decoder IDs and encoder states must use one device.")
        if self.training and use_cache and self.config.decoder_layerdrop:
            raise ValueError("Decoder layerdrop cannot be combined with KV caching.")

        batch_size, current_length = input_ids.shape
        past_length = self._past_length(
            past_key_values,
            batch_size=batch_size,
        )
        if past_length + current_length > self.config.max_target_positions:
            raise ValueError("Decoder input exceeds the configured text context window.")

        decoder_mask = self._decoder_mask(
            attention_mask,
            batch_size=batch_size,
            current_length=current_length,
            past_length=past_length,
            device=input_ids.device,
        )
        encoder_attention_mask = _validate_key_mask(
            encoder_attention_mask,
            batch_size=batch_size,
            key_length=encoder_hidden_states.shape[1],
            name="encoder_attention_mask",
            device=input_ids.device,
        )

        positions = self.positional_embedding[past_length:past_length + current_length]
        hidden_states = self.token_embedding(input_ids.long())
        if self.config.scale_embedding:
            hidden_states = hidden_states * math.sqrt(self.config.d_model)
        hidden_states = hidden_states + positions.to(dtype=hidden_states.dtype)
        hidden_states = hidden_states.to(dtype=encoder_hidden_states.dtype)
        hidden_states = functional.dropout(
            hidden_states,
            p=self.config.dropout,
            training=self.training,
        )

        next_cache: list[WhisperLayerCache] = []
        for index, block in enumerate(self.blocks):
            layer_past = (None if past_key_values is None else past_key_values[index])
            drop_layer = self.training and bool(
                self.config.decoder_layerdrop and torch.rand(
                    (), device=hidden_states.device) < self.config.decoder_layerdrop)
            if drop_layer:
                continue
            hidden_states, layer_cache = block.forward_decoder(
                hidden_states,
                encoder_hidden_states,
                self_attention_mask=decoder_mask,
                encoder_attention_mask=encoder_attention_mask,
                causal_offset=past_length,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if layer_cache is not None:
                next_cache.append(layer_cache)

        hidden_states = self.ln(hidden_states)
        return hidden_states, tuple(next_cache) if use_cache else None


class WhisperModel(nn.Module):
    """Native Whisper conditional-generation model for inference and
    training."""

    def __init__(self, config: WhisperConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.config = WhisperConfig.coerce(config)
        self.encoder = WhisperEncoder(self.config)
        self.decoder = WhisperDecoder(self.config)
        self.apply(self._initialize_module)
        nn.init.normal_(
            self.decoder.positional_embedding,
            mean=0.0,
            std=self.config.init_std,
        )

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.init_std,
            )
            bias = getattr(module, "bias", None)
            if bias is not None:
                nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_embeddings(self) -> nn.Embedding:
        return self.decoder.token_embedding

    def encode(
        self,
        input_features: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        return self.encoder(input_features, attention_mask=attention_mask)

    def _encoder_mask(
        self,
        attention_mask: Tensor | None,
        *,
        input_features: Tensor | None,
        encoder_hidden_states: Tensor,
    ) -> Tensor | None:
        if attention_mask is None:
            return None
        if input_features is None:
            return _validate_key_mask(
                attention_mask,
                batch_size=encoder_hidden_states.shape[0],
                key_length=encoder_hidden_states.shape[1],
                name="attention_mask",
                device=encoder_hidden_states.device,
            )
        return self.encoder.downsample_attention_mask(
            attention_mask,
            batch_size=input_features.shape[0],
            input_frames=input_features.shape[2],
            device=encoder_hidden_states.device,
        )

    def decode(
        self,
        input_ids: Tensor,
        encoder_hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: WhisperDecoderCache | None = None,
        use_cache: bool | None = None,
    ) -> WhisperDecoderOutput:
        resolved_use_cache = (self.config.use_cache if use_cache is None else use_cache)
        if not isinstance(resolved_use_cache, bool):
            raise TypeError("`use_cache` must be a boolean.")
        hidden_states, cache = self.decoder(
            input_ids,
            encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=resolved_use_cache,
        )
        logits = functional.linear(
            hidden_states,
            self.decoder.token_embedding.weight.to(hidden_states.dtype),
        ).float()
        return WhisperDecoderOutput(
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=cache,
        )

    def _shift_labels(self, labels: Tensor) -> Tensor:
        shifted = labels.new_full(labels.shape, self.config.pad_token_id)
        shifted[:, 0] = self.config.decoder_start_token_id
        shifted[:, 1:] = labels[:, :-1]
        return shifted.masked_fill(shifted == -100, self.config.pad_token_id)

    def forward(
        self,
        input_features: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        *,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        encoder_outputs: Tensor | None = None,
        past_key_values: WhisperDecoderCache | None = None,
        use_cache: bool | None = None,
    ) -> WhisperOutput:
        if encoder_outputs is None:
            if input_features is None:
                raise ValueError("`input_features` is required when `encoder_outputs` is absent.")
            encoder_hidden_states = self.encode(
                input_features,
                attention_mask=attention_mask,
            )
        else:
            if input_features is not None:
                raise ValueError("Pass either `input_features` or `encoder_outputs`, not both.")
            if (not isinstance(encoder_outputs, Tensor) or encoder_outputs.ndim != 3 or
                    encoder_outputs.shape[-1] != self.config.d_model):
                raise ValueError("`encoder_outputs` must have shape [batch, audio, d_model].")
            encoder_hidden_states = encoder_outputs

        if labels is not None:
            if not isinstance(labels, Tensor) or labels.ndim != 2:
                raise ValueError("`labels` must have shape [batch, tokens].")
            if labels.shape[0] < 1 or labels.shape[1] < 1:
                raise ValueError("Label batches and token sequences cannot be empty.")
            if (labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex()):
                raise TypeError("`labels` must use an integer dtype.")
            if labels.device != encoder_hidden_states.device:
                raise ValueError("Labels and encoder states must use one device.")
            if decoder_input_ids is None:
                decoder_input_ids = self._shift_labels(labels)
        if decoder_input_ids is None:
            raise ValueError("Provide `decoder_input_ids` or `labels`.")
        if past_key_values is not None and labels is not None:
            raise ValueError("Training labels cannot be combined with a decoder cache.")

        encoder_mask = self._encoder_mask(
            attention_mask,
            input_features=input_features,
            encoder_hidden_states=encoder_hidden_states,
        )
        decoder_output = self.decode(
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_mask,
            past_key_values=past_key_values,
            use_cache=False if labels is not None and use_cache is None else use_cache,
        )

        loss = None
        if labels is not None:
            if labels.shape != decoder_output.logits.shape[:-1]:
                raise ValueError("`labels` must match the decoder logit time dimensions.")
            effective_labels = labels
            if decoder_attention_mask is not None:
                effective_labels = labels.masked_fill(
                    ~decoder_attention_mask.to(dtype=torch.bool),
                    -100,
                )
            try:
                from voicehub.objectives.sequence import sequence_cross_entropy
            except ImportError:
                valid_labels = effective_labels != -100
                if valid_labels.any():
                    loss = functional.cross_entropy(
                        decoder_output.logits.reshape(-1, self.config.vocab_size),
                        effective_labels.long().reshape(-1),
                        ignore_index=-100,
                    )
                else:
                    loss = decoder_output.logits.sum() * 0.0
            else:
                loss = sequence_cross_entropy(
                    decoder_output.logits,
                    effective_labels,
                    ignore_index=-100,
                )

        return WhisperOutput(
            logits=decoder_output.logits,
            encoder_last_hidden_state=encoder_hidden_states,
            decoder_last_hidden_state=decoder_output.last_hidden_state,
            loss=loss,
            past_key_values=decoder_output.past_key_values,
        )


WhisperForConditionalGeneration = WhisperModel

__all__ = [
    "Float32LayerNorm",
    "WhisperAttention",
    "WhisperAttentionCache",
    "WhisperDecoder",
    "WhisperDecoderCache",
    "WhisperDecoderOutput",
    "WhisperEncoder",
    "WhisperForConditionalGeneration",
    "WhisperLayerCache",
    "WhisperModel",
    "WhisperOutput",
    "whisper_sinusoids",
]
