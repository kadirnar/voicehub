"""VoiceHub-native Nemotron 3.5 cache-aware FastConformer RNN-T."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from voicehub.models.asr_nemotron.native.configuration import NemotronASRArchitectureConfig, NemotronEncoderConfig
from voicehub.models.asr_nemotron.native.loss import rnnt_loss


def _activation(name: str):
    if name == "relu":
        return functional.relu
    if name == "silu":
        return functional.silu
    raise ValueError(f"Unsupported Nemotron activation {name!r}.")


@dataclass(slots=True)
class NemotronEncoderOutput:
    last_hidden_state: Tensor
    attention_mask: Tensor | None = None
    past_key_values: NemotronAttentionCache | None = None
    padding_cache: NemotronConvolutionCache | None = None
    pooler_output: Tensor | None = None


@dataclass(slots=True)
class NemotronRNNTOutput:
    logits: Tensor
    last_hidden_state: Tensor
    pooler_output: Tensor
    attention_mask: Tensor | None = None
    loss: Tensor | None = None
    encoder_past_key_values: NemotronAttentionCache | None = None
    padding_cache: NemotronConvolutionCache | None = None


@dataclass(frozen=True, slots=True)
class NemotronGenerateOutput:
    sequences: Tensor
    durations: Tensor


class NemotronAttentionCache:
    """Per-layer dynamic sliding-window K/V state."""

    def __init__(self, *, layer_count: int, sliding_window: int) -> None:
        if layer_count <= 0 or sliding_window <= 1:
            raise ValueError("Invalid Nemotron attention-cache dimensions.")
        self.sliding_window = sliding_window
        self.keys: list[Tensor | None] = [None] * layer_count
        self.values: list[Tensor | None] = [None] * layer_count
        self.cumulative_length = 0
        self._pending_length: int | None = None

    @property
    def cached_frames(self) -> int:
        return min(self.cumulative_length, self.sliding_window - 1)

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_index: int,
    ) -> tuple[Tensor, Tensor]:
        previous_keys = self.keys[layer_index]
        previous_values = self.values[layer_index]
        if previous_keys is None:
            full_keys = key_states
            full_values = value_states
        else:
            full_keys = torch.cat((previous_keys, key_states), dim=-2)
            full_values = torch.cat((previous_values, value_states), dim=-2)
        self.keys[layer_index] = full_keys[
            :,
            :,
            -self.sliding_window + 1:,
            :,
        ]
        self.values[layer_index] = full_values[
            :,
            :,
            -self.sliding_window + 1:,
            :,
        ]
        if layer_index == 0:
            self._pending_length = key_states.shape[-2]
        if layer_index == len(self.keys) - 1:
            if self._pending_length != key_states.shape[-2]:
                raise RuntimeError("Nemotron attention-cache layers advanced inconsistently.")
            self.cumulative_length += key_states.shape[-2]
            self._pending_length = None
        return full_keys, full_values


class _Conv1dCacheLayer:

    def __init__(self) -> None:
        self.cache: Tensor | None = None

    def update(self, hidden_states: Tensor, module: CausalConv1d) -> Tensor:
        if self.cache is None:
            self.cache = hidden_states.new_zeros(
                hidden_states.shape[0],
                module.in_channels,
                module.left_pad,
            )
        shortfall = max(0, module.left_pad - hidden_states.shape[-1])
        if shortfall:
            replacement = torch.cat(
                (self.cache[:, :, -shortfall:], hidden_states),
                dim=-1,
            )
        else:
            replacement = hidden_states[:, :, -module.left_pad:]
        previous = self.cache
        self.cache = replacement
        return previous


class _Conv2dCacheLayer:

    def __init__(self) -> None:
        self.cache: Tensor | None = None
        self.first_chunk = True

    def update(self, hidden_states: Tensor, module: CausalConv2d) -> Tensor:
        if self.cache is None:
            shape = list(hidden_states.shape)
            shape[2] = module.left_pad
            self.cache = hidden_states.new_zeros(shape)
        shortfall = max(0, module.left_pad - hidden_states.shape[2])
        if shortfall:
            replacement = torch.cat(
                (self.cache[:, :, -shortfall:], hidden_states),
                dim=2,
            )
        else:
            replacement = hidden_states[:, :, -module.left_pad:]
        previous = self.cache
        if self.first_chunk:
            initial_padding = module.left_pad_init - module.left_pad
            if initial_padding:
                shape = list(previous.shape)
                shape[2] = initial_padding
                previous = torch.cat(
                    (previous.new_zeros(shape), previous),
                    dim=2,
                )
        self.first_chunk = False
        self.cache = replacement
        return previous


class NemotronConvolutionCache:
    """Unified causal Conv2d/Conv1d streaming state."""

    def __init__(self) -> None:
        self.layers: dict[str, _Conv1dCacheLayer | _Conv2dCacheLayer] = {}

    def pad(
        self,
        hidden_states: Tensor,
        *,
        cache_key: str,
        module: CausalConv1d | CausalConv2d,
    ) -> Tensor:
        layer = self.layers.get(cache_key)
        if layer is None:
            layer = (_Conv2dCacheLayer() if isinstance(module, CausalConv2d) else _Conv1dCacheLayer())
            self.layers[cache_key] = layer
        previous = layer.update(hidden_states, module)
        dimension = 2 if isinstance(module, CausalConv2d) else -1
        return torch.cat((previous, hidden_states), dim=dimension)


class CausalConv1d(nn.Conv1d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        cache_key: str,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
        )
        self.cache_key = cache_key

    @cached_property
    def left_pad(self) -> int:
        return ((self.kernel_size[0] - 1) * self.dilation[0] + 1 - self.stride[0])

    def forward(
        self,
        values: Tensor,
        *,
        padding_cache: NemotronConvolutionCache | None = None,
    ) -> Tensor:
        if padding_cache is None:
            values = functional.pad(values, (self.left_pad, 0))
        else:
            values = padding_cache.pad(
                values,
                cache_key=self.cache_key,
                module=self,
            )
        return super().forward(values)


class CausalConv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        cache_key: str,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
        )
        self.cache_key = cache_key

    @property
    def left_pad(self) -> int:
        return self.kernel_size[0] - self.stride[0]

    @property
    def left_pad_init(self) -> int:
        return self.kernel_size[0] - 1

    @property
    def time_pad(self) -> tuple[int, int]:
        return self.kernel_size[0] - 1, self.stride[0] - 1

    @property
    def frequency_pad(self) -> tuple[int, int]:
        return self.kernel_size[1] - 1, self.stride[1] - 1

    def output_length(
        self,
        input_lengths: Tensor | None,
        *,
        streaming: bool,
    ) -> Tensor | None:
        if input_lengths is None:
            return None
        left, right = (self.left_pad, 0) if streaming else self.time_pad
        return (input_lengths + left + right - self.kernel_size[0]) // self.stride[0] + 1

    def forward(
        self,
        values: Tensor,
        *,
        padding_cache: NemotronConvolutionCache | None = None,
    ) -> Tensor:
        values = functional.pad(
            values,
            (self.frequency_pad[0], self.frequency_pad[1]),
        )
        if padding_cache is None:
            values = functional.pad(
                values,
                (0, 0, self.time_pad[0], self.time_pad[1]),
            )
        else:
            values = padding_cache.pad(
                values,
                cache_key=self.cache_key,
                module=self,
            )
        return super().forward(values)


class RelativePositionalEncoding(nn.Module):

    def __init__(self, config: NemotronEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.max_position_embeddings = config.max_position_embeddings
        inv_freq = 1.0 / (
            10000.0**(torch.arange(0, config.hidden_size, 2, dtype=torch.float32) / config.hidden_size))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cached_frames: int = 0,
    ) -> Tensor:
        sequence_length = hidden_states.shape[1] + cached_frames
        if sequence_length > self.max_position_embeddings:
            raise ValueError(
                f"Nemotron sequence length {sequence_length} exceeds "
                f"{self.max_position_embeddings}.")
        positions = torch.arange(
            sequence_length - 1,
            -sequence_length,
            -1,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        frequencies = torch.outer(
            positions,
            self.inv_freq.to(
                device=hidden_states.device,
                dtype=torch.float32,
            ),
        )
        sine = frequencies.sin()
        cosine = frequencies.cos()
        embeddings = torch.stack((sine, cosine), dim=-1).reshape(
            frequencies.shape[0],
            -1,
        )
        return embeddings.unsqueeze(0).expand(
            hidden_states.shape[0],
            -1,
            -1,
        ).to(dtype=hidden_states.dtype)


class ConvolutionModule(nn.Module):

    def __init__(
        self,
        config: NemotronEncoderConfig,
        *,
        layer_index: int,
    ) -> None:
        super().__init__()
        channels = config.hidden_size
        self.activation = _activation(config.hidden_act)
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            1,
            bias=config.convolution_bias,
        )
        self.depthwise_conv = CausalConv1d(
            channels,
            channels,
            config.conv_kernel_size,
            cache_key=f"conv.{layer_index}",
            groups=channels,
            bias=config.convolution_bias,
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            1,
            bias=config.convolution_bias,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        all_masked_rows: Tensor | None = None,
        padding_cache: NemotronConvolutionCache | None = None,
    ) -> Tensor:
        values = hidden_states.transpose(1, 2)
        values = functional.glu(self.pointwise_conv1(values), dim=1)
        if all_masked_rows is not None:
            values = values.masked_fill(all_masked_rows, 0.0)
        values = self.depthwise_conv(
            values,
            padding_cache=padding_cache,
        )
        values = self.norm(values.transpose(1, 2)).transpose(1, 2)
        values = self.pointwise_conv2(self.activation(values))
        return values.transpose(1, 2)


class RelativeAttention(nn.Module):

    def __init__(
        self,
        config: NemotronEncoderConfig,
        *,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.head_dim = config.head_dim
        self.num_key_value_groups = (config.num_attention_heads // config.num_key_value_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.relative_k_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.bias_u = nn.Parameter(torch.zeros(
            config.num_attention_heads,
            self.head_dim,
        ))
        self.bias_v = nn.Parameter(torch.zeros(
            config.num_attention_heads,
            self.head_dim,
        ))

    @staticmethod
    def _relative_shift(scores: Tensor) -> Tensor:
        batch, heads, queries, positions = scores.shape
        scores = functional.pad(scores, (1, 0))
        scores = scores.view(batch, heads, -1, queries)
        return scores[:, :, 1:].view(
            batch,
            heads,
            queries,
            positions,
        )

    @staticmethod
    def _repeat_key_values(values: Tensor, repeats: int) -> Tensor:
        if repeats == 1:
            return values
        batch, heads, length, dimension = values.shape
        values = values[:, :, None, :, :].expand(
            batch,
            heads,
            repeats,
            length,
            dimension,
        )
        return values.reshape(
            batch,
            heads * repeats,
            length,
            dimension,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_embeddings: Tensor,
        attention_mask: Tensor | None,
        past_key_values: NemotronAttentionCache | None,
    ) -> Tensor:
        batch, sequence_length, _ = hidden_states.shape
        shape = (batch, sequence_length, -1, self.head_dim)
        queries = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        values = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        if past_key_values is not None:
            keys, values = past_key_values.update(
                keys,
                values,
                self.layer_index,
            )
        total_key_length = keys.shape[2]
        query_u = queries + self.bias_u.view(
            1,
            self.config.num_attention_heads,
            1,
            self.head_dim,
        )
        query_v = queries + self.bias_v.view(
            1,
            self.config.num_attention_heads,
            1,
            self.head_dim,
        )
        relative_keys = self.relative_k_proj(position_embeddings).view(
            batch,
            -1,
            self.config.num_attention_heads,
            self.head_dim,
        )
        relative_scores = query_v @ relative_keys.permute(0, 2, 3, 1)
        relative_scores = self._relative_shift(relative_scores)
        relative_scores = (relative_scores[..., :total_key_length] * self.scaling)
        if attention_mask is not None:
            relative_scores = relative_scores.masked_fill(
                ~attention_mask,
                float("-inf"),
            )

        keys = self._repeat_key_values(
            keys,
            self.num_key_value_groups,
        )
        values = self._repeat_key_values(
            values,
            self.num_key_value_groups,
        )
        scores = (query_u @ keys.transpose(2, 3)) * self.scaling
        scores = scores + relative_scores
        weights = functional.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        ).to(dtype=queries.dtype)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = functional.dropout(
            weights,
            p=self.attention_dropout,
            training=self.training,
        )
        output = weights @ values
        output = output.transpose(1, 2).reshape(
            batch,
            sequence_length,
            -1,
        )
        return self.o_proj(output)


def _mask_subsampled(
    hidden_states: Tensor,
    lengths: Tensor | None,
) -> Tensor:
    if lengths is None:
        return hidden_states
    positions = torch.arange(
        hidden_states.shape[2],
        device=hidden_states.device,
    )
    mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    return hidden_states * mask[:, None, :, None]


class SubsamplingLayer(nn.Module):

    def __init__(
        self,
        config: NemotronEncoderConfig,
        *,
        layer_index: int,
    ) -> None:
        super().__init__()
        channels = config.subsampling_conv_channels
        self.depthwise_conv = CausalConv2d(
            channels,
            channels,
            config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
            groups=channels,
            cache_key=f"subsampling.{layer_index}",
        )
        self.pointwise_conv = nn.Conv2d(channels, channels, 1)

    def forward(
        self,
        hidden_states: Tensor,
        lengths: Tensor | None,
        *,
        padding_cache: NemotronConvolutionCache | None,
    ) -> tuple[Tensor, Tensor | None]:
        hidden_states = self.depthwise_conv(
            hidden_states,
            padding_cache=padding_cache,
        )
        lengths = self.depthwise_conv.output_length(
            lengths,
            streaming=padding_cache is not None,
        )
        hidden_states = self.pointwise_conv(hidden_states)
        return _mask_subsampled(hidden_states, lengths), lengths


class SubsamplingConv2d(nn.Module):

    def __init__(self, config: NemotronEncoderConfig) -> None:
        super().__init__()
        channels = config.subsampling_conv_channels
        layer_count = int(math.log2(config.subsampling_factor))
        self.conv_in = CausalConv2d(
            1,
            channels,
            config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
            cache_key="subsampling.0",
        )
        self.layers = nn.ModuleList(
            SubsamplingLayer(config, layer_index=index) for index in range(1, layer_count))
        self.linear = nn.Linear(
            config.subsampling_out_hidden_size,
            config.hidden_size,
        )

    def forward(
        self,
        input_features: Tensor,
        attention_mask: Tensor | None,
        *,
        padding_cache: NemotronConvolutionCache | None,
    ) -> Tensor:
        hidden_states = input_features.unsqueeze(1)
        lengths = (attention_mask.sum(dim=-1) if attention_mask is not None else None)
        hidden_states = self.conv_in(
            hidden_states,
            padding_cache=padding_cache,
        )
        lengths = self.conv_in.output_length(
            lengths,
            streaming=padding_cache is not None,
        )
        hidden_states = functional.relu(_mask_subsampled(hidden_states, lengths))
        for layer in self.layers:
            hidden_states, lengths = layer(
                hidden_states,
                lengths,
                padding_cache=padding_cache,
            )
            hidden_states = functional.relu(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            hidden_states.shape[0],
            hidden_states.shape[2],
            -1,
        )
        return self.linear(hidden_states)


class FeedForward(nn.Module):

    def __init__(self, config: NemotronEncoderConfig) -> None:
        super().__init__()
        self.linear1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.attention_bias,
        )
        self.linear2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.activation = _activation(config.hidden_act)
        self.activation_dropout = config.activation_dropout

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.activation(self.linear1(hidden_states))
        hidden_states = functional.dropout(
            hidden_states,
            p=self.activation_dropout,
            training=self.training,
        )
        return self.linear2(hidden_states)


class FastConformerBlock(nn.Module):

    def __init__(
        self,
        config: NemotronEncoderConfig,
        *,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.feed_forward1 = FeedForward(config)
        self.self_attn = RelativeAttention(
            config,
            layer_index=layer_index,
        )
        self.conv = ConvolutionModule(
            config,
            layer_index=layer_index,
        )
        self.feed_forward2 = FeedForward(config)
        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size)
        self.norm_self_att = nn.LayerNorm(config.hidden_size)
        self.norm_conv = nn.LayerNorm(config.hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size)
        self.norm_out = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        all_masked_rows: Tensor | None,
        position_embeddings: Tensor,
        past_key_values: NemotronAttentionCache | None = None,
        padding_cache: NemotronConvolutionCache | None = None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = residual + 0.5 * hidden_states
        attention_output = self.self_attn(
            self.norm_self_att(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
        )
        hidden_states = hidden_states + attention_output
        convolution_output = self.conv(
            self.norm_conv(hidden_states),
            all_masked_rows=all_masked_rows,
            padding_cache=padding_cache,
        )
        hidden_states = hidden_states + convolution_output
        hidden_states = hidden_states + 0.5 * self.feed_forward2(self.norm_feed_forward2(hidden_states))
        return self.norm_out(hidden_states)


class NemotronFastConformerEncoder(nn.Module):

    def __init__(
        self,
        config: NemotronEncoderConfig,
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        self.config = NemotronEncoderConfig.coerce(config)
        self.dropout = self.config.dropout
        self.dropout_positions = self.config.dropout_positions
        self.layerdrop = self.config.layerdrop
        self.input_scale = (math.sqrt(self.config.hidden_size) if self.config.scale_input else 1.0)
        self.subsampling = SubsamplingConv2d(self.config)
        self.encode_positions = RelativePositionalEncoding(self.config)
        self.layers = nn.ModuleList(
            FastConformerBlock(self.config, layer_index=index)
            for index in range(self.config.num_hidden_layers))
        self.gradient_checkpointing = False
        if initialize:
            self.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        if isinstance(module, RelativeAttention):
            nn.init.normal_(
                module.bias_u,
                mean=0.0,
                std=self.config.initializer_range,
            )
            nn.init.normal_(
                module.bias_v,
                mean=0.0,
                std=self.config.initializer_range,
            )

    def _get_subsampling_output_length(self, input_lengths: Tensor) -> Tensor:
        kernel = self.config.subsampling_conv_kernel_size
        stride = self.config.subsampling_conv_stride
        layer_count = int(math.log2(self.config.subsampling_factor))
        add_pad = (kernel - 1 + stride - 1 - kernel)
        lengths = input_lengths
        for _ in range(layer_count):
            lengths = torch.floor(torch.div(lengths.float() + add_pad, stride) + 1.0)
        return lengths.to(dtype=torch.long)

    def _output_attention_mask(
        self,
        attention_mask: Tensor,
        *,
        target_length: int,
    ) -> Tensor:
        lengths = self._get_subsampling_output_length(attention_mask.sum(dim=-1))
        positions = torch.arange(
            target_length,
            device=attention_mask.device,
        )
        return positions.unsqueeze(0) < lengths.unsqueeze(1)

    def _lookahead(self, value: int | None) -> int:
        resolved = (self.config.default_num_lookahead_tokens if value is None else value)
        if (isinstance(resolved, bool) or not isinstance(resolved, int) or
                resolved not in self.config.supported_num_lookahead_tokens):
            raise ValueError(
                f"Unsupported Nemotron lookahead {resolved!r}; expected one "
                f"of {self.config.supported_num_lookahead_tokens}.")
        return resolved

    def _attention_mask(
        self,
        *,
        batch_size: int,
        query_length: int,
        output_mask: Tensor | None,
        lookahead: int,
        cache: NemotronAttentionCache | None,
        device: torch.device,
    ) -> Tensor:
        past_seen = cache.cumulative_length if cache is not None else 0
        cached_frames = cache.cached_frames if cache is not None else 0
        query_positions = torch.arange(
            past_seen,
            past_seen + query_length,
            device=device,
        )
        key_positions = torch.arange(
            past_seen - cached_frames,
            past_seen + query_length,
            device=device,
        )
        chunk_size = lookahead + 1
        left_chunks = (self.config.sliding_window - 1) // chunk_size
        differences = (
            torch.div(
                query_positions,
                chunk_size,
                rounding_mode="floor",
            ).unsqueeze(1) - torch.div(
                key_positions,
                chunk_size,
                rounding_mode="floor",
            ).unsqueeze(0))
        allowed = (differences >= 0) & (differences <= left_chunks)
        allowed = allowed[None, None].expand(
            batch_size,
            1,
            -1,
            -1,
        )
        if output_mask is not None:
            if cache is not None:
                raise ValueError(
                    "Streaming cached encoding does not accept a padded "
                    "attention mask; pad chunks to their required size.")
            allowed = (allowed & output_mask[:, None, None, :])
        return allowed

    def forward(
        self,
        input_features: Tensor,
        attention_mask: Tensor | None = None,
        *,
        past_key_values: NemotronAttentionCache | None = None,
        padding_cache: NemotronConvolutionCache | None = None,
        use_cache: bool = False,
        num_lookahead_tokens: int | None = None,
        output_attention_mask: bool = True,
    ) -> NemotronEncoderOutput:
        if input_features.ndim != 3:
            raise ValueError("Nemotron input features must have shape "
                             "[batch, frames, mel_bins].")
        if input_features.shape[-1] != self.config.num_mel_bins:
            raise ValueError(f"Nemotron expects {self.config.num_mel_bins} mel bins.")
        if attention_mask is not None:
            if attention_mask.shape != input_features.shape[:2]:
                raise ValueError("Nemotron feature attention mask has an invalid shape.")
            attention_mask = attention_mask.to(dtype=torch.bool)
        if use_cache and self.training:
            raise ValueError("Nemotron streaming caches are inference-only.")
        if use_cache:
            if past_key_values is None:
                past_key_values = NemotronAttentionCache(
                    layer_count=len(self.layers),
                    sliding_window=self.config.sliding_window,
                )
            if padding_cache is None:
                padding_cache = NemotronConvolutionCache()
        elif past_key_values is not None or padding_cache is not None:
            raise ValueError("Nemotron caches require `use_cache=True`.")
        if self.gradient_checkpointing and use_cache:
            raise ValueError("Gradient checkpointing and streaming caches are incompatible.")

        cached_frames = (past_key_values.cached_frames if past_key_values is not None else 0)
        hidden_states = self.subsampling(
            input_features,
            attention_mask,
            padding_cache=padding_cache,
        )
        hidden_states = hidden_states * self.input_scale
        output_mask = (
            self._output_attention_mask(
                attention_mask,
                target_length=hidden_states.shape[1],
            ) if attention_mask is not None else None)
        lookahead = self._lookahead(num_lookahead_tokens)
        layer_attention_mask = self._attention_mask(
            batch_size=hidden_states.shape[0],
            query_length=hidden_states.shape[1],
            output_mask=output_mask,
            lookahead=lookahead,
            cache=past_key_values,
            device=hidden_states.device,
        )
        all_masked_rows = torch.all(
            ~layer_attention_mask,
            dim=-1,
        )
        position_embeddings = self.encode_positions(
            hidden_states,
            cached_frames=cached_frames,
        )
        hidden_states = functional.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        position_embeddings = functional.dropout(
            position_embeddings,
            p=self.dropout_positions,
            training=self.training,
        )

        for layer in self.layers:
            drop_layer = (
                self.training and self.layerdrop and torch.rand(
                    (),
                    device=hidden_states.device,
                ) < self.layerdrop)
            if drop_layer:
                continue
            if self.gradient_checkpointing and self.training:

                def layer_forward(
                    states: Tensor,
                    mask: Tensor,
                    masked_rows: Tensor,
                    positions: Tensor,
                    *,
                    current_layer: FastConformerBlock = layer,
                ) -> Tensor:
                    return current_layer(
                        states,
                        mask,
                        masked_rows,
                        positions,
                    )

                hidden_states = checkpoint(
                    layer_forward,
                    hidden_states,
                    layer_attention_mask,
                    all_masked_rows,
                    position_embeddings,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    layer_attention_mask,
                    all_masked_rows,
                    position_embeddings,
                    past_key_values,
                    padding_cache,
                )
        return NemotronEncoderOutput(
            last_hidden_state=hidden_states,
            attention_mask=(
                output_mask.to(
                    dtype=torch.long) if output_mask is not None and output_attention_mask else None),
            past_key_values=past_key_values,
            padding_cache=padding_cache,
        )


class PromptProjector(nn.Module):

    def __init__(self, config: NemotronASRArchitectureConfig) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(
            config.encoder_config.hidden_size + config.num_prompts,
            config.prompt_intermediate_size,
        )
        self.linear_2 = nn.Linear(
            config.prompt_intermediate_size,
            config.encoder_config.hidden_size,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.linear_2(functional.relu(self.linear_1(hidden_states)))


class RNNTDecoder(nn.Module):

    def __init__(self, config: NemotronASRArchitectureConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.decoder_hidden_size,
        )
        self.lstm = nn.LSTM(
            config.decoder_hidden_size,
            config.decoder_hidden_size,
            num_layers=config.num_decoder_layers,
            batch_first=True,
        )
        self.decoder_projector = nn.Linear(
            config.decoder_hidden_size,
            config.decoder_hidden_size,
        )

    def forward(
        self,
        input_ids: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("Nemotron decoder IDs must have shape [batch, labels].")
        output, state = self.lstm(self.embedding(input_ids), state)
        return self.decoder_projector(output), state


class RNNTJointNetwork(nn.Module):

    def __init__(self, config: NemotronASRArchitectureConfig) -> None:
        super().__init__()
        self.activation = _activation(config.hidden_act)
        self.head = nn.Linear(
            config.decoder_hidden_size,
            config.vocab_size,
        )

    def forward(
        self,
        decoder_hidden_states: Tensor,
        encoder_hidden_states: Tensor,
    ) -> Tensor:
        return self.head(self.activation(encoder_hidden_states + decoder_hidden_states))


class Nemotron3_5ASRForRNNT(nn.Module):
    """Exact published Nemotron tensor namespace and native execution graph."""

    def __init__(
        self,
        config: NemotronASRArchitectureConfig,
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        self.config = NemotronASRArchitectureConfig.coerce(config)
        self.encoder = NemotronFastConformerEncoder(
            self.config.encoder_config,
            initialize=False,
        )
        self.encoder_projector = nn.Linear(
            self.config.encoder_config.hidden_size,
            self.config.decoder_hidden_size,
        )
        self.decoder = RNNTDecoder(self.config)
        self.joint = RNNTJointNetwork(self.config)
        self.prompt_projector = PromptProjector(self.config)
        self.max_symbols_per_step = self.config.max_symbols_per_step
        if initialize:
            self.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.encoder_config.initializer_range,
            )
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        if isinstance(module, RelativeAttention):
            nn.init.normal_(
                module.bias_u,
                mean=0.0,
                std=self.config.encoder_config.initializer_range,
            )
            nn.init.normal_(
                module.bias_v,
                mean=0.0,
                std=self.config.encoder_config.initializer_range,
            )

    def gradient_checkpointing_enable(self) -> None:
        self.encoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.encoder.gradient_checkpointing = False

    def get_audio_features(
        self,
        input_features: Tensor,
        attention_mask: Tensor | None = None,
        *,
        prompt_ids: Tensor | None = None,
        past_key_values: NemotronAttentionCache | None = None,
        padding_cache: NemotronConvolutionCache | None = None,
        use_cache: bool = False,
        num_lookahead_tokens: int | None = None,
        output_attention_mask: bool = True,
    ) -> NemotronEncoderOutput:
        outputs = self.encoder(
            input_features,
            attention_mask,
            past_key_values=past_key_values,
            padding_cache=padding_cache,
            use_cache=use_cache,
            num_lookahead_tokens=num_lookahead_tokens,
            output_attention_mask=output_attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        if prompt_ids is None:
            prompt_ids = torch.full(
                (hidden_states.shape[0], ),
                self.config.default_prompt_id,
                dtype=torch.long,
                device=hidden_states.device,
            )
        else:
            prompt_ids = torch.as_tensor(
                prompt_ids,
                dtype=torch.long,
                device=hidden_states.device,
            )
        if prompt_ids.shape != (hidden_states.shape[0], ):
            raise ValueError("Nemotron prompt IDs must have shape [batch].")
        if torch.any(prompt_ids < 0) or torch.any(prompt_ids >= self.config.num_prompts):
            raise ValueError("Nemotron prompt ID is outside the prompt table.")
        one_hot = functional.one_hot(
            prompt_ids,
            num_classes=self.config.num_prompts,
        ).to(dtype=hidden_states.dtype)
        one_hot = one_hot[:, None, :].expand(
            -1,
            hidden_states.shape[1],
            -1,
        )
        fused = self.prompt_projector(torch.cat((hidden_states, one_hot), dim=-1))
        outputs.pooler_output = self.encoder_projector(fused)
        return outputs

    def forward(
        self,
        input_features: Tensor | None = None,
        attention_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        *,
        encoder_outputs: NemotronEncoderOutput | None = None,
        labels: Tensor | None = None,
        label_lengths: Tensor | None = None,
        prompt_ids: Tensor | None = None,
        past_key_values: NemotronAttentionCache | None = None,
        padding_cache: NemotronConvolutionCache | None = None,
        use_cache: bool = False,
        num_lookahead_tokens: int | None = None,
        output_attention_mask: bool = True,
    ) -> NemotronRNNTOutput:
        if encoder_outputs is None:
            if input_features is None:
                raise ValueError("Nemotron forward requires features or encoder outputs.")
            encoder_outputs = self.get_audio_features(
                input_features,
                attention_mask,
                prompt_ids=prompt_ids,
                past_key_values=past_key_values,
                padding_cache=padding_cache,
                use_cache=use_cache,
                num_lookahead_tokens=num_lookahead_tokens,
                output_attention_mask=output_attention_mask,
            )
        if decoder_input_ids is None:
            raise ValueError("Nemotron forward requires decoder input IDs.")
        decoder_hidden, _ = self.decoder(decoder_input_ids)
        if encoder_outputs.pooler_output is None:
            raise ValueError("Nemotron encoder outputs are missing projected features.")
        logits = self.joint(
            decoder_hidden[:, None, :, :],
            encoder_outputs.pooler_output[:, :, None, :],
        )
        if logits.shape[2] == 1:
            logits = logits.squeeze(2)
        loss = None
        if labels is not None:
            if logits.ndim != 4:
                raise ValueError("RNN-T training requires one decoder state per target "
                                 "prefix.")
            output_mask = encoder_outputs.attention_mask
            if output_mask is None:
                logit_lengths = torch.full(
                    (logits.shape[0], ),
                    logits.shape[1],
                    device=logits.device,
                    dtype=torch.long,
                )
            else:
                logit_lengths = output_mask.sum(dim=-1)
            if label_lengths is None:
                label_lengths = (labels != self.config.blank_token_id).sum(dim=-1)
            loss = rnnt_loss(
                logits[:, :int(logit_lengths.max())],
                labels,
                logit_lengths,
                label_lengths,
                self.config.blank_token_id,
            )
        return NemotronRNNTOutput(
            logits=logits,
            loss=loss,
            last_hidden_state=encoder_outputs.last_hidden_state,
            pooler_output=encoder_outputs.pooler_output,
            attention_mask=encoder_outputs.attention_mask,
            encoder_past_key_values=encoder_outputs.past_key_values,
            padding_cache=encoder_outputs.padding_cache,
        )

    def _decode_projected(
        self,
        projected: Tensor,
        valid_lengths: Tensor,
    ) -> tuple[list[list[int]], list[list[int]]]:
        sequences: list[list[int]] = []
        durations: list[list[int]] = []
        for batch_index in range(projected.shape[0]):
            token_ids = [self.config.blank_token_id]
            advances = [0]
            decoder_ids = torch.tensor(
                [[self.config.blank_token_id]],
                dtype=torch.long,
                device=projected.device,
            )
            decoder_output, decoder_state = self.decoder(decoder_ids)
            frame_index = 0
            symbols_at_frame = 0
            while frame_index < int(valid_lengths[batch_index]):
                logits = self.joint(
                    decoder_output,
                    projected[
                        batch_index:batch_index + 1,
                        frame_index:frame_index + 1,
                    ],
                )
                token_id = int(logits[0, 0].argmax(dim=-1))
                is_blank = token_id == self.config.blank_token_id
                token_ids.append(token_id)
                if is_blank:
                    advances.append(1)
                    frame_index += 1
                    symbols_at_frame = 0
                    continue
                symbols_at_frame += 1
                force_advance = (symbols_at_frame >= self.max_symbols_per_step)
                advances.append(1 if force_advance else 0)
                decoder_ids = torch.tensor(
                    [[token_id]],
                    dtype=torch.long,
                    device=projected.device,
                )
                decoder_output, decoder_state = self.decoder(
                    decoder_ids,
                    decoder_state,
                )
                if force_advance:
                    frame_index += 1
                    symbols_at_frame = 0
            sequences.append(token_ids)
            durations.append(advances)
        return sequences, durations

    @staticmethod
    def _pad_generation(
        sequences: list[list[int]],
        durations: list[list[int]],
        *,
        blank_token_id: int,
        device: torch.device,
    ) -> NemotronGenerateOutput:
        width = max(len(row) for row in sequences)
        sequence_tensor = torch.full(
            (len(sequences), width),
            blank_token_id,
            dtype=torch.long,
            device=device,
        )
        duration_tensor = torch.zeros(
            (len(sequences), width),
            dtype=torch.long,
            device=device,
        )
        for index, (tokens, advances) in enumerate(zip(sequences, durations, strict=True)):
            sequence_tensor[index, :len(tokens)] = torch.tensor(
                tokens,
                dtype=torch.long,
                device=device,
            )
            duration_tensor[index, :len(advances)] = torch.tensor(
                advances,
                dtype=torch.long,
                device=device,
            )
        return NemotronGenerateOutput(
            sequences=sequence_tensor,
            durations=duration_tensor,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_features: Tensor,
        attention_mask: Tensor | None = None,
        *,
        prompt_ids: Tensor | None = None,
        num_lookahead_tokens: int | None = None,
    ) -> NemotronGenerateOutput:
        if not isinstance(input_features, Tensor):
            raise TypeError(
                "Offline Nemotron generation requires a feature tensor; "
                "use `generate_stream` for chunk iterables.")
        outputs = self.get_audio_features(
            input_features,
            attention_mask,
            prompt_ids=prompt_ids,
            num_lookahead_tokens=num_lookahead_tokens,
        )
        valid_lengths = (
            outputs.attention_mask.sum(dim=-1) if outputs.attention_mask is not None else torch.full(
                (input_features.shape[0], ),
                outputs.pooler_output.shape[1],
                dtype=torch.long,
                device=input_features.device,
            ))
        sequences, durations = self._decode_projected(
            outputs.pooler_output,
            valid_lengths,
        )
        return self._pad_generation(
            sequences,
            durations,
            blank_token_id=self.config.blank_token_id,
            device=input_features.device,
        )

    @torch.inference_mode()
    def generate_stream(
        self,
        chunks: Iterable[Tensor],
        *,
        prompt_ids: Tensor,
        num_lookahead_tokens: int,
    ) -> NemotronGenerateOutput:
        """Cache-aware greedy decoding over exact-size log-mel chunks.

        All chunks must share a batch size and be padded to the size
        selected by ``num_lookahead_tokens``.  The first chunk uses ``1
        + subsampling_factor * right`` frames; later chunks use
        ``subsampling_factor * (right + 1)``.
        """
        lookahead = self.encoder._lookahead(num_lookahead_tokens)
        attention_cache = NemotronAttentionCache(
            layer_count=len(self.encoder.layers),
            sliding_window=self.config.encoder_config.sliding_window,
        )
        convolution_cache = NemotronConvolutionCache()
        projected_chunks: list[Tensor] = []
        batch_size: int | None = None
        for chunk_index, chunk in enumerate(chunks):
            if not isinstance(chunk, Tensor) or chunk.ndim != 3:
                raise ValueError("Nemotron stream chunks must have shape "
                                 "[batch, frames, mel_bins].")
            required = (
                1 + self.config.encoder_config.subsampling_factor * lookahead
                if chunk_index == 0 else self.config.encoder_config.subsampling_factor * (lookahead + 1))
            if chunk.shape[1] != required:
                raise ValueError(
                    f"Nemotron stream chunk {chunk_index} has "
                    f"{chunk.shape[1]} frames; expected {required}.")
            if batch_size is None:
                batch_size = chunk.shape[0]
            elif chunk.shape[0] != batch_size:
                raise ValueError("Nemotron stream chunk batch sizes must match.")
            outputs = self.get_audio_features(
                chunk,
                prompt_ids=prompt_ids,
                past_key_values=attention_cache,
                padding_cache=convolution_cache,
                use_cache=True,
                num_lookahead_tokens=lookahead,
                output_attention_mask=False,
            )
            projected_chunks.append(outputs.pooler_output)
        if not projected_chunks:
            raise ValueError("Nemotron stream did not contain a chunk.")
        projected = torch.cat(projected_chunks, dim=1)
        lengths = torch.full(
            (projected.shape[0], ),
            projected.shape[1],
            dtype=torch.long,
            device=projected.device,
        )
        sequences, durations = self._decode_projected(projected, lengths)
        return self._pad_generation(
            sequences,
            durations,
            blank_token_id=self.config.blank_token_id,
            device=projected.device,
        )


__all__ = [
    "Nemotron3_5ASRForRNNT",
    "NemotronAttentionCache",
    "NemotronConvolutionCache",
    "NemotronEncoderOutput",
    "NemotronFastConformerEncoder",
    "NemotronGenerateOutput",
    "NemotronRNNTOutput",
]
