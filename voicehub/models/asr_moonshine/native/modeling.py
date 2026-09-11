"""VoiceHub-owned PyTorch implementation of Moonshine ASR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from voicehub.models.asr_moonshine.native.configuration import MoonshineConfig


def _activation(name: str, value: torch.Tensor) -> torch.Tensor:
    if name == "gelu":
        return F.gelu(value)
    if name == "silu":
        return F.silu(value)
    raise ValueError(f"Unsupported Moonshine activation {name!r}.")


def _right_padded_mask(
    mask: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    name: str,
) -> torch.Tensor:
    if mask.ndim != 2 or tuple(mask.shape) != (batch_size, sequence_length):
        raise ValueError(
            f"`{name}` must have shape {(batch_size, sequence_length)}; "
            f"received {tuple(mask.shape)}.")
    if mask.dtype == torch.bool:
        normalized = mask
    elif not mask.dtype.is_floating_point and not mask.dtype.is_complex:
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"`{name}` must contain only zero and one.")
        normalized = mask.bool()
    else:
        raise TypeError(f"`{name}` must use a boolean or integer dtype.")
    if sequence_length > 1 and torch.any(normalized[:, 1:] > normalized[:, :-1]):
        raise ValueError(f"`{name}` must be right padded.")
    if torch.any(normalized.sum(dim=-1) == 0):
        raise ValueError(f"`{name}` must retain at least one value per example.")
    return normalized


def _additive_attention_mask(
    mask: torch.Tensor | None,
    *,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if mask is None:
        return None
    minimum = torch.finfo(dtype).min
    return (~mask).to(dtype=dtype)[:, None, None, :] * minimum


def _causal_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    sequence_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    minimum = torch.finfo(dtype).min
    causal = torch.full(
        (sequence_length, sequence_length),
        minimum,
        dtype=dtype,
        device=device,
    )
    causal = torch.triu(causal, diagonal=1)[None, None, :, :]
    if attention_mask is None:
        return causal
    padding = _additive_attention_mask(attention_mask, dtype=dtype)
    return causal + padding


def _rotary_embeddings(
    *,
    sequence_length: int,
    head_dim: int,
    partial_rotary_factor: float,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = int(head_dim * partial_rotary_factor)
    inverse_frequency = 1.0 / (
        theta**(torch.arange(
            0,
            rotary_dim,
            2,
            dtype=torch.float32,
            device=device,
        ) / rotary_dim))
    positions = torch.arange(
        sequence_length,
        dtype=torch.float32,
        device=device,
    )
    frequencies = torch.outer(positions, inverse_frequency)
    # Moonshine interleaves each frequency across an adjacent dimension pair.
    cosine = frequencies.cos().repeat_interleave(2, dim=-1)
    sine = frequencies.sin().repeat_interleave(2, dim=-1)
    return (
        cosine[None, :, :].to(dtype=dtype),
        sine[None, :, :].to(dtype=dtype),
    )


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    even = value[..., 0::2]
    odd = value[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine, sine = position_embeddings
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    rotary_dim = cosine.shape[-1]
    query_rotary, query_pass = (
        query[..., :rotary_dim],
        query[..., rotary_dim:],
    )
    key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    query_rotary = (query_rotary * cosine + _rotate_half(query_rotary) * sine)
    key_rotary = key_rotary * cosine + _rotate_half(key_rotary) * sine
    return (
        torch.cat((query_rotary, query_pass), dim=-1),
        torch.cat((key_rotary, key_pass), dim=-1),
    )


class MoonshineAttention(nn.Module):
    """Moonshine MHA with optional head padding and partial rotary position."""

    def __init__(
        self,
        config: MoonshineConfig,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        is_causal: bool,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = (num_attention_heads // num_key_value_heads)
        self.head_dim = config.hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = is_causal
        self.partial_rotary_factor = config.partial_rotary_factor
        self.rope_theta = config.rope_theta
        bias = config.attention_bias
        self.q_proj = nn.Linear(
            config.hidden_size,
            num_attention_heads * self.head_dim,
            bias=bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            num_key_value_heads * self.head_dim,
            bias=bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            num_key_value_heads * self.head_dim,
            bias=bias,
        )
        self.o_proj = nn.Linear(
            num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        multiple = config.pad_head_dim_to_multiple_of
        self.head_dim_padding = (0 if multiple is None else (multiple - self.head_dim % multiple) % multiple)

    @staticmethod
    def _repeat_key_values(
        value: torch.Tensor,
        repeats: int,
    ) -> torch.Tensor:
        if repeats == 1:
            return value
        batch, heads, length, dimension = value.shape
        value = value[:, :, None, :, :].expand(
            batch,
            heads,
            repeats,
            length,
            dimension,
        )
        return value.reshape(batch, heads * repeats, length, dimension)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        key_value_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, query_length, _ = hidden_states.shape
        source = (hidden_states if key_value_states is None else key_value_states)
        query = self.q_proj(hidden_states).view(
            batch_size,
            query_length,
            self.num_attention_heads,
            self.head_dim,
        )
        key = self.k_proj(source).view(
            batch_size,
            source.shape[1],
            self.num_key_value_heads,
            self.head_dim,
        )
        value = self.v_proj(source).view(
            batch_size,
            source.shape[1],
            self.num_key_value_heads,
            self.head_dim,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if key_value_states is None:
            if position_embeddings is None:
                raise ValueError("Moonshine self-attention requires rotary embeddings.")
            query, key = _apply_rotary(query, key, position_embeddings)
        key = self._repeat_key_values(key, self.num_key_value_groups)
        value = self._repeat_key_values(value, self.num_key_value_groups)

        if self.head_dim_padding:
            padding = (0, self.head_dim_padding)
            query = F.pad(query, padding)
            key = F.pad(key, padding)
            value = F.pad(value, padding)
        weights = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            weights = weights + attention_mask
        probabilities = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        probabilities = F.dropout(
            probabilities,
            p=self.attention_dropout,
            training=self.training,
        )
        attended = torch.matmul(probabilities, value)
        if self.head_dim_padding:
            attended = attended[..., :-self.head_dim_padding]
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            query_length,
            -1,
        )
        return self.o_proj(attended), (probabilities if output_attentions else None)


class MoonshineEncoderMLP(nn.Module):

    def __init__(self, config: MoonshineConfig) -> None:
        super().__init__()
        self.hidden_act = config.encoder_hidden_act
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(_activation(self.hidden_act, self.fc1(hidden_states)))


class MoonshineDecoderMLP(nn.Module):

    def __init__(self, config: MoonshineConfig) -> None:
        super().__init__()
        self.hidden_act = config.decoder_hidden_act
        self.fc1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size * 2,
        )
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values, gate = self.fc1(hidden_states).chunk(2, dim=-1)
        return self.fc2(_activation(self.hidden_act, gate) * values)


class MoonshineEncoderLayer(nn.Module):

    def __init__(self, config: MoonshineConfig) -> None:
        super().__init__()
        self.self_attn = MoonshineAttention(
            config,
            num_attention_heads=config.encoder_num_attention_heads,
            num_key_value_heads=config.encoder_num_key_value_heads,
            is_causal=False,
        )
        self.mlp = MoonshineEncoderMLP(config)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            bias=False,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        output_attentions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        attended, weights = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attended
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, weights


class MoonshineDecoderLayer(nn.Module):

    def __init__(self, config: MoonshineConfig) -> None:
        super().__init__()
        self.self_attn = MoonshineAttention(
            config,
            num_attention_heads=config.decoder_num_attention_heads,
            num_key_value_heads=config.decoder_num_key_value_heads,
            is_causal=True,
        )
        self.encoder_attn = MoonshineAttention(
            config,
            num_attention_heads=config.decoder_num_attention_heads,
            num_key_value_heads=config.decoder_num_key_value_heads,
            is_causal=False,
        )
        self.mlp = MoonshineDecoderMLP(config)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            bias=False,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            bias=False,
        )
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        output_attentions: bool,
    ) -> tuple[
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
    ]:
        residual = hidden_states
        attended, self_weights = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attended

        residual = hidden_states
        attended, cross_weights = self.encoder_attn(
            self.post_attention_layernorm(hidden_states),
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attended
        hidden_states = hidden_states + self.mlp(self.final_layernorm(hidden_states))
        return hidden_states, self_weights, cross_weights


@dataclass(frozen=True)
class MoonshineEncoderOutput:
    last_hidden_state: torch.Tensor
    attention_mask: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] = ()
    attentions: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class MoonshineModelOutput:
    last_hidden_state: torch.Tensor
    encoder_last_hidden_state: torch.Tensor
    encoder_attention_mask: torch.Tensor | None = None
    decoder_hidden_states: tuple[torch.Tensor, ...] = ()
    decoder_attentions: tuple[torch.Tensor, ...] = ()
    cross_attentions: tuple[torch.Tensor, ...] = ()
    encoder_hidden_states: tuple[torch.Tensor, ...] = ()
    encoder_attentions: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class MoonshineSeq2SeqLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    encoder_last_hidden_state: torch.Tensor | None = None
    encoder_attention_mask: torch.Tensor | None = None
    decoder_hidden_states: tuple[torch.Tensor, ...] = ()
    decoder_attentions: tuple[torch.Tensor, ...] = ()
    cross_attentions: tuple[torch.Tensor, ...] = ()
    encoder_hidden_states: tuple[torch.Tensor, ...] = ()
    encoder_attentions: tuple[torch.Tensor, ...] = ()


class MoonshineEncoder(nn.Module):

    def __init__(self, config: MoonshineConfig) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.conv1 = nn.Conv1d(
            1,
            hidden_size,
            kernel_size=127,
            stride=64,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            hidden_size,
            2 * hidden_size,
            kernel_size=7,
            stride=3,
        )
        self.conv3 = nn.Conv1d(
            2 * hidden_size,
            hidden_size,
            kernel_size=3,
            stride=2,
        )
        self.groupnorm = nn.GroupNorm(
            num_groups=1,
            num_channels=hidden_size,
            eps=1e-5,
        )
        self.layers = nn.ModuleList(
            MoonshineEncoderLayer(config) for _ in range(config.encoder_num_hidden_layers))
        self.layer_norm = nn.LayerNorm(hidden_size, bias=False)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> MoonshineEncoderOutput:
        if input_values.ndim != 2:
            raise ValueError(
                "`input_values` must have shape [batch, samples]; received "
                f"{tuple(input_values.shape)}.")
        if not input_values.dtype.is_floating_point:
            raise TypeError("`input_values` must use a floating-point dtype.")
        if input_values.shape[-1] < self.config.minimum_input_samples:
            raise ValueError(
                "Moonshine input is shorter than the convolutional frontend "
                f"minimum of {self.config.minimum_input_samples} samples.")
        batch_size, sample_count = input_values.shape
        raw_mask = None
        if attention_mask is not None:
            raw_mask = _right_padded_mask(
                attention_mask,
                batch_size=batch_size,
                sequence_length=sample_count,
                name="attention_mask",
            )

        hidden_states = torch.tanh(self.conv1(input_values.unsqueeze(1)))
        hidden_states = self.groupnorm(hidden_states)
        hidden_states = F.gelu(self.conv2(hidden_states))
        hidden_states = F.gelu(self.conv3(hidden_states))
        hidden_states = hidden_states.transpose(1, 2)

        feature_mask = None
        if raw_mask is not None:
            # This stride sampling is the published processor contract. It is
            # intentionally not replaced with a length-derived approximation.
            feature_mask = raw_mask[
                :,
                ::self.config.input_to_feature_ratio,
            ][:, :hidden_states.shape[1]]
            if feature_mask.shape[1] < hidden_states.shape[1]:
                feature_mask = F.pad(
                    feature_mask,
                    (0, hidden_states.shape[1] - feature_mask.shape[1]),
                )
        additive_mask = _additive_attention_mask(
            feature_mask,
            dtype=hidden_states.dtype,
        )
        position_embeddings = _rotary_embeddings(
            sequence_length=hidden_states.shape[1],
            head_dim=self.config.encoder_head_dim,
            partial_rotary_factor=self.config.partial_rotary_factor,
            theta=self.config.rope_theta,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        all_hidden_states: list[torch.Tensor] = []
        all_attentions: list[torch.Tensor] = []
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        for layer in self.layers:
            hidden_states, weights = layer(
                hidden_states,
                attention_mask=additive_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
            )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            if weights is not None:
                all_attentions.append(weights)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states[-1] = hidden_states
        return MoonshineEncoderOutput(
            last_hidden_state=hidden_states,
            attention_mask=feature_mask.int() if feature_mask is not None else None,
            hidden_states=tuple(all_hidden_states),
            attentions=tuple(all_attentions),
        )


class MoonshineDecoder(nn.Module):

    def __init__(self, config: MoonshineConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            MoonshineDecoderLayer(config) for _ in range(config.decoder_num_hidden_layers))
        self.norm = nn.LayerNorm(config.hidden_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[
            torch.Tensor,
            tuple[torch.Tensor, ...],
            tuple[torch.Tensor, ...],
            tuple[torch.Tensor, ...],
    ]:
        if input_ids.ndim != 2:
            raise ValueError("`decoder_input_ids` must have shape [batch, tokens].")
        if input_ids.dtype == torch.bool or input_ids.dtype.is_floating_point:
            raise TypeError("`decoder_input_ids` must use an integer dtype.")
        if input_ids.numel() and (torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size)):
            raise ValueError("`decoder_input_ids` contains an invalid token ID.")
        batch_size, sequence_length = input_ids.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch_size:
            raise ValueError("`encoder_hidden_states` must have shape "
                             "[batch, frames, hidden_size].")
        normalized_mask = None
        if attention_mask is not None:
            normalized_mask = _right_padded_mask(
                attention_mask,
                batch_size=batch_size,
                sequence_length=sequence_length,
                name="decoder_attention_mask",
            )
        normalized_encoder_mask = None
        if encoder_attention_mask is not None:
            normalized_encoder_mask = _right_padded_mask(
                encoder_attention_mask,
                batch_size=batch_size,
                sequence_length=encoder_hidden_states.shape[1],
                name="encoder_attention_mask",
            )

        hidden_states = self.embed_tokens(input_ids)
        causal_mask = _causal_attention_mask(
            normalized_mask,
            sequence_length=sequence_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        cross_mask = _additive_attention_mask(
            normalized_encoder_mask,
            dtype=hidden_states.dtype,
        )
        position_embeddings = _rotary_embeddings(
            sequence_length=sequence_length,
            head_dim=self.config.decoder_head_dim,
            partial_rotary_factor=self.config.partial_rotary_factor,
            theta=self.config.rope_theta,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        all_hidden_states: list[torch.Tensor] = []
        all_attentions: list[torch.Tensor] = []
        all_cross_attentions: list[torch.Tensor] = []
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        for layer in self.layers:
            hidden_states, weights, cross_weights = layer(
                hidden_states,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=cross_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
            )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            if weights is not None:
                all_attentions.append(weights)
            if cross_weights is not None:
                all_cross_attentions.append(cross_weights)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states[-1] = hidden_states
        return (
            hidden_states,
            tuple(all_hidden_states),
            tuple(all_attentions),
            tuple(all_cross_attentions),
        )


class MoonshineModel(nn.Module):

    def __init__(self, config: MoonshineConfig | dict[str, Any]) -> None:
        super().__init__()
        self.config = MoonshineConfig.coerce(config)
        self.encoder = MoonshineEncoder(self.config)
        self.decoder = MoonshineDecoder(self.config)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Embedding, nn.Linear)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: MoonshineEncoderOutput | None = None,
        *,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool | None = None,
        past_key_values: Any | None = None,
    ) -> MoonshineModelOutput:
        if use_cache:
            raise ValueError(
                "Native Moonshine's public graph currently uses deterministic "
                "full-prefix decoding; key/value cache inputs are unsupported.")
        if past_key_values is not None:
            raise ValueError("`past_key_values` is unsupported.")
        if decoder_input_ids is None:
            raise ValueError("`decoder_input_ids` is required.")
        if encoder_outputs is None:
            if input_values is None:
                raise ValueError("`input_values` is required when `encoder_outputs` is absent.")
            encoder_outputs = self.encoder(
                input_values,
                attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_outputs.attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return MoonshineModelOutput(
            last_hidden_state=decoder_outputs[0],
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_outputs.attention_mask,
            decoder_hidden_states=decoder_outputs[1],
            decoder_attentions=decoder_outputs[2],
            cross_attentions=decoder_outputs[3],
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def shift_tokens_right(
    labels: torch.Tensor,
    *,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> torch.Tensor:
    """Build teacher-forced decoder inputs without mutating labels."""
    if labels.ndim != 2:
        raise ValueError("`labels` must have shape [batch, tokens].")
    shifted = labels.new_full(labels.shape, pad_token_id)
    shifted[:, 0] = decoder_start_token_id
    if labels.shape[1] > 1:
        shifted[:, 1:] = labels[:, :-1]
    return shifted.masked_fill(shifted == -100, pad_token_id)


class MoonshineForConditionalGeneration(nn.Module):
    """Moonshine encoder-decoder with its tied language-model projection."""

    def __init__(self, config: MoonshineConfig | dict[str, Any]) -> None:
        super().__init__()
        self.config = MoonshineConfig.coerce(config)
        self.model = MoonshineModel(self.config)

    @property
    def proj_out(self) -> nn.Embedding:
        """Compatibility view of the tied output-projection weight."""
        return self.model.decoder.embed_tokens

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.decoder.embed_tokens

    def get_output_embeddings(self) -> nn.Embedding:
        return self.model.decoder.embed_tokens

    def optimization_compile_targets(self, mode: str):
        """Return the execution boundaries used by each runtime phase."""
        from voicehub.optimization.protocols import OptimizationCompileTarget

        if mode == "inference":
            # Generation encodes once outside ``forward`` and then calls the
            # outer module for every growing decoder prefix.
            return (
                OptimizationCompileTarget(
                    label="encoder",
                    owner=self.model.encoder,
                    attribute="forward",
                ),
                OptimizationCompileTarget(
                    label="conditional-generation",
                    owner=self,
                    attribute="forward",
                ),
            )
        if mode == "training":
            return (
                OptimizationCompileTarget(
                    label="conditional-generation",
                    owner=self,
                    attribute="forward",
                ), )
        raise ValueError("Moonshine compile mode must be 'inference' or 'training'.")

    def forward(
        self,
        input_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: MoonshineEncoderOutput | None = None,
        *,
        labels: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool | None = None,
        past_key_values: Any | None = None,
    ) -> MoonshineSeq2SeqLMOutput:
        if labels is not None:
            if labels.ndim != 2:
                raise ValueError("`labels` must have shape [batch, tokens].")
            valid = labels != -100
            if torch.any(labels[valid] < 0) or torch.any(labels[valid] >= self.config.vocab_size):
                raise ValueError("`labels` contains an invalid token ID.")
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels,
                    pad_token_id=self.config.pad_token_id,
                    decoder_start_token_id=self.config.decoder_start_token_id,
                )
        if decoder_input_ids is None:
            raise ValueError("Supply `decoder_input_ids` or `labels`.")
        outputs = self.model(
            input_values,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        logits = F.linear(
            outputs.last_hidden_state,
            self.model.decoder.embed_tokens.weight,
        )
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return MoonshineSeq2SeqLMOutput(
            logits=logits,
            loss=loss,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_attention_mask=outputs.encoder_attention_mask,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        max_new_tokens: int | None = None,
        max_length: int | None = None,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """Greedy decoding with one deterministic full-prefix
        implementation."""
        if num_beams != 1:
            raise ValueError("Native Moonshine currently supports greedy decoding only.")
        if do_sample:
            raise ValueError("Native Moonshine does not implement sampled decoding.")
        if temperature not in (None, 1, 1.0):
            raise ValueError("`temperature` is unavailable for deterministic greedy decoding.")
        if max_new_tokens is not None and max_length is not None:
            raise ValueError("Pass `max_new_tokens` or `max_length`, not both.")
        if max_new_tokens is not None:
            if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
                raise TypeError("`max_new_tokens` must be an integer or None.")
            if max_new_tokens <= 0:
                raise ValueError("`max_new_tokens` must be greater than zero.")
            maximum_length = max_new_tokens + 1
        else:
            maximum_length = (self.config.max_position_embeddings if max_length is None else max_length)
        if isinstance(maximum_length, bool) or not isinstance(maximum_length, int):
            raise TypeError("`max_length` must be an integer or None.")
        if not 2 <= maximum_length <= self.config.max_position_embeddings:
            raise ValueError("`max_length` must be between 2 and "
                             f"{self.config.max_position_embeddings}.")

        encoder_outputs = self.model.encoder(input_values, attention_mask)
        sequences = torch.full(
            (input_values.shape[0], 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=input_values.device,
        )
        finished = torch.zeros(
            input_values.shape[0],
            dtype=torch.bool,
            device=input_values.device,
        )
        for _ in range(maximum_length - 1):
            output = self(
                decoder_input_ids=sequences,
                encoder_outputs=encoder_outputs,
            )
            next_tokens = output.logits[:, -1].argmax(dim=-1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, self.config.pad_token_id),
                next_tokens,
            )
            sequences = torch.cat((sequences, next_tokens[:, None]), dim=-1)
            finished |= next_tokens == self.config.eos_token_id
            if bool(finished.all()):
                break
        return sequences


__all__ = [
    "MoonshineAttention",
    "MoonshineDecoder",
    "MoonshineDecoderLayer",
    "MoonshineEncoder",
    "MoonshineEncoderLayer",
    "MoonshineEncoderOutput",
    "MoonshineForConditionalGeneration",
    "MoonshineModel",
    "MoonshineModelOutput",
    "MoonshineSeq2SeqLMOutput",
    "shift_tokens_right",
]
