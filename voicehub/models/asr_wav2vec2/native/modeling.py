"""PyTorch-only Wav2Vec2 CTC architecture owned by VoiceHub.

The graph and parameter namespace were independently implemented after
reviewing Hugging Face Transformers' Wav2Vec2 implementation at revision
``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``.  It imports no upstream
model runtime and keeps the convolutional frontend and encoder blocks
reusable by future HuBERT and WavLM ports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.asr_wav2vec2.native.configuration import Wav2Vec2Config
from voicehub.objectives.ctc import ctc_loss


class Float32LayerNorm(nn.LayerNorm):
    """Layer normalization with float32 accumulation for reduced precision."""

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
    """Tanh GELU used by compatible Hugging Face configurations."""

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
    if name == "selu":
        return nn.SELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported Wav2Vec2 activation {name!r}.")


def _validate_floating_input(input_values: Tensor, config: Wav2Vec2Config) -> None:
    if not isinstance(input_values, Tensor):
        raise TypeError("`input_values` must be a PyTorch tensor.")
    if input_values.ndim != 2:
        raise ValueError("`input_values` must have shape [batch, samples].")
    if not input_values.is_floating_point():
        raise TypeError("`input_values` must use a floating-point dtype.")
    if input_values.shape[0] < 1:
        raise ValueError("`input_values` must contain at least one waveform.")
    if input_values.shape[1] < config.minimum_input_samples:
        raise ValueError(
            "The waveform is too short for the configured convolutional "
            f"frontend; at least {config.minimum_input_samples} samples are "
            f"required.")
    if not torch.isfinite(input_values).all():
        raise ValueError("`input_values` cannot contain NaN or infinite values.")


def _validated_raw_attention_mask(
    attention_mask: Tensor | None,
    *,
    input_values: Tensor,
    minimum_input_samples: int,
) -> tuple[Tensor, Tensor]:
    batch_size, sample_count = input_values.shape
    if attention_mask is None:
        mask = torch.ones(
            (batch_size, sample_count),
            dtype=torch.bool,
            device=input_values.device,
        )
    else:
        if not isinstance(attention_mask, Tensor):
            raise TypeError("`attention_mask` must be a PyTorch tensor.")
        if tuple(attention_mask.shape) != (batch_size, sample_count):
            raise ValueError(
                "`attention_mask` must have the same [batch, samples] shape "
                "as `input_values`.")
        if attention_mask.device != input_values.device:
            raise ValueError("`attention_mask` must be on the same device as `input_values`.")
        if attention_mask.is_complex():
            raise TypeError("`attention_mask` cannot use a complex dtype.")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("`attention_mask` must contain only zero and one.")
        mask = attention_mask.to(dtype=torch.bool)

    if sample_count > 1 and ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ValueError("`attention_mask` must describe right-padded contiguous audio.")
    lengths = mask.sum(dim=-1, dtype=torch.long)
    if (lengths < minimum_input_samples).any():
        raise ValueError(
            "Every waveform must contain enough valid samples for the "
            "convolutional frontend.")
    return mask, lengths


def downsample_wav2vec2_lengths(
    input_lengths: Tensor,
    config: Wav2Vec2Config,
) -> Tensor:
    """Apply the exact unpadded Conv1d output-length formula."""
    if not isinstance(input_lengths, Tensor):
        raise TypeError("`input_lengths` must be a PyTorch tensor.")
    if input_lengths.ndim != 1:
        raise ValueError("`input_lengths` must have shape [batch].")
    if (input_lengths.dtype == torch.bool or input_lengths.is_floating_point() or input_lengths.is_complex()):
        raise TypeError("`input_lengths` must use an integer dtype.")
    lengths = input_lengths.to(dtype=torch.long)
    for kernel, stride in zip(config.conv_kernel, config.conv_stride):
        lengths = torch.div(
            lengths - kernel,
            stride,
            rounding_mode="floor",
        ) + 1
    if (lengths < 1).any():
        raise ValueError("A waveform produces no feature frames after downsampling.")
    return lengths


def feature_attention_mask(
    feature_length: int,
    output_lengths: Tensor,
) -> Tensor:
    """Build a right-padded boolean feature mask from exact lengths."""
    if isinstance(feature_length, bool) or not isinstance(feature_length, int):
        raise TypeError("`feature_length` must be an integer.")
    if feature_length < 1:
        raise ValueError("`feature_length` must be positive.")
    if not isinstance(output_lengths, Tensor) or output_lengths.ndim != 1:
        raise ValueError("`output_lengths` must have shape [batch].")
    if (output_lengths.dtype == torch.bool or output_lengths.is_floating_point() or
            output_lengths.is_complex()):
        raise TypeError("`output_lengths` must use an integer dtype.")
    if (output_lengths < 1).any() or (output_lengths > feature_length).any():
        raise ValueError("`output_lengths` must be within the available feature frames.")
    positions = torch.arange(
        feature_length,
        device=output_lengths.device,
    )
    return positions.unsqueeze(0) < output_lengths.unsqueeze(1)


class Wav2Vec2FeatureConvLayer(nn.Module):
    """One configurable raw-waveform convolution and normalization."""

    def __init__(
        self,
        config: Wav2Vec2Config,
        layer_index: int,
    ) -> None:
        super().__init__()
        input_channels = (1 if layer_index == 0 else config.conv_dim[layer_index - 1])
        output_channels = config.conv_dim[layer_index]
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=config.conv_kernel[layer_index],
            stride=config.conv_stride[layer_index],
            bias=config.conv_bias,
        )
        if config.feat_extract_norm == "layer":
            self.layer_norm: nn.Module | None = Float32LayerNorm(
                output_channels,
                eps=config.layer_norm_eps,
            )
            self._normalization = "layer"
        elif layer_index == 0:
            self.layer_norm = nn.GroupNorm(
                num_groups=output_channels,
                num_channels=output_channels,
                affine=True,
            )
            self._normalization = "group"
        else:
            self.layer_norm = None
            self._normalization = "none"
        self.activation = _activation(config.feat_extract_activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        if self._normalization == "layer":
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = self.layer_norm(hidden_states)
            hidden_states = hidden_states.transpose(1, 2)
        elif self._normalization == "group":
            hidden_states = self.layer_norm(hidden_states)
        return self.activation(hidden_states)


class Wav2Vec2FeatureEncoder(nn.Module):
    """Extensible convolutional frontend for raw, mono waveforms."""

    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.config = config
        self.conv_layers = nn.ModuleList(
            Wav2Vec2FeatureConvLayer(config, index) for index in range(config.num_feat_extract_layers))

    def forward(self, input_values: Tensor) -> Tensor:
        hidden_states = input_values.unsqueeze(1)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
        return hidden_states

    def freeze(self) -> None:
        """Disable gradient updates for all frontend parameters."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class Wav2Vec2FeatureProjection(nn.Module):
    """Normalize and project convolutional channels to encoder width."""

    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.layer_norm = Float32LayerNorm(
            config.conv_dim[-1],
            eps=config.layer_norm_eps,
        )
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        normalized_features = self.layer_norm(hidden_states)
        projected = self.dropout(self.projection(normalized_features))
        return projected, normalized_features


class WeightNormalizedConv1d(nn.Module):
    """Auditable Conv1d weight normalization with HF-compatible tensors.

    Hugging Face's Wav2Vec2 checkpoints store positional convolution
    tensors as ``weight_g`` and ``weight_v`` with normalization
    dimension two.  This implementation keeps that stable namespace
    without relying on PyTorch's deprecated ``nn.utils.weight_norm``
    hook.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        groups: int,
    ) -> None:
        super().__init__()
        for name, value in (
            ("channels", channels),
            ("kernel_size", kernel_size),
            ("groups", groups),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"`{name}` must be an integer.")
            if value < 1:
                raise ValueError(f"`{name}` must be positive.")
        if channels % groups:
            raise ValueError("`channels` must be divisible by `groups`.")
        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.padding = kernel_size // 2
        self.weight_g = nn.Parameter(torch.empty(1, 1, kernel_size))
        self.weight_v = nn.Parameter(torch.empty(channels, channels // groups, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        standard_deviation = 2.0 * math.sqrt(1.0 / (self.kernel_size * self.channels))
        nn.init.normal_(self.weight_v, mean=0.0, std=standard_deviation)
        with torch.no_grad():
            norm = torch.linalg.vector_norm(
                self.weight_v.float(),
                dim=(0, 1),
                keepdim=True,
            )
            self.weight_g.copy_(norm.to(dtype=self.weight_g.dtype))
            self.bias.zero_()

    def normalized_weight(self) -> Tensor:
        """Materialize the effective convolution weight."""
        working_weight = (
            self.weight_v.float() if self.weight_v.dtype in (torch.float16,
                                                             torch.bfloat16) else self.weight_v)
        norm = torch.linalg.vector_norm(
            working_weight,
            dim=(0, 1),
            keepdim=True,
        ).clamp_min(torch.finfo(working_weight.dtype).tiny)
        scale = self.weight_g.to(dtype=working_weight.dtype) / norm
        return self.weight_v * scale.to(dtype=self.weight_v.dtype)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return functional.conv1d(
            hidden_states,
            self.normalized_weight(),
            self.bias,
            padding=self.padding,
            groups=self.groups,
        )


class Wav2Vec2PositionalConvEmbedding(nn.Module):
    """Grouped convolutional positional embedding."""

    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.conv = WeightNormalizedConv1d(
            config.hidden_size,
            config.num_conv_pos_embeddings,
            config.num_conv_pos_embedding_groups,
        )
        self.remove_trailing_frame = (config.num_conv_pos_embeddings % 2 == 0)
        self.activation = _activation(config.feat_extract_activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        positions = self.conv(hidden_states.transpose(1, 2))
        if self.remove_trailing_frame:
            positions = positions[:, :, :-1]
        return self.activation(positions).transpose(1, 2)


class Wav2Vec2Attention(nn.Module):
    """Bidirectional multi-head self-attention with stable probabilities."""

    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_heads
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def _split_heads(self, value: Tensor) -> Tensor:
        batch_size, steps, _ = value.shape
        return value.reshape(
            batch_size,
            steps,
            self.num_heads,
            self.head_size,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None]:
        if hidden_states.ndim != 3:
            raise ValueError("Wav2Vec2 attention expects [batch, time, hidden].")
        batch_size, steps, width = hidden_states.shape
        if width != self.hidden_size:
            raise ValueError("Wav2Vec2 attention hidden width is incompatible.")

        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))
        working_query = (query.float() if query.dtype in (torch.float16, torch.bfloat16) else query)
        working_key = (key.float() if key.dtype in (torch.float16, torch.bfloat16) else key)
        scores = torch.matmul(
            working_query,
            working_key.transpose(-1, -2),
        ) * (self.head_size**-0.5)

        if attention_mask is not None:
            if (attention_mask.dtype != torch.bool or tuple(attention_mask.shape) != (batch_size, steps)):
                raise ValueError("Encoder attention mask must be boolean [batch, time].")
            scores = scores.masked_fill(
                ~attention_mask[:, None, None, :],
                -torch.inf,
            )

        probabilities = torch.softmax(scores, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        dropped_probabilities = functional.dropout(
            probabilities,
            p=self.dropout,
            training=self.training,
        ).to(dtype=value.dtype)
        attended = torch.matmul(dropped_probabilities, value)
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            steps,
            self.hidden_size,
        )
        output = self.out_proj(attended)
        return output, dropped_probabilities if output_attentions else None


class Wav2Vec2FeedForward(nn.Module):
    """Transformer feed-forward network shared across family variants."""

    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.intermediate_dense = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
        )
        self.intermediate_act_fn = _activation(config.hidden_act)
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.output_dense = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
        )
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        return self.output_dropout(hidden_states)


class Wav2Vec2EncoderLayer(nn.Module):
    """Post-normalized Wav2Vec2 Transformer layer."""

    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.attention = Wav2Vec2Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = Float32LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = Float32LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None]:
        residual = hidden_states
        attended, attention = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.layer_norm(residual + self.dropout(attended))
        hidden_states = self.final_layer_norm(hidden_states + self.feed_forward(hidden_states))
        return hidden_states, attention


class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    """Pre-normalized Transformer layer used by stable family variants."""

    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.attention = Wav2Vec2Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = Float32LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = Float32LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None]:
        residual = hidden_states
        attended, attention = self.attention(
            self.layer_norm(hidden_states),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + self.dropout(attended)
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
        return hidden_states, attention


@dataclass(frozen=True)
class Wav2Vec2EncoderOutput:
    """Result of the bidirectional Transformer encoder."""

    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor | None, ...] | None = None
    executed_layers: tuple[bool, ...] = ()


class Wav2Vec2Encoder(nn.Module):
    """Positional convolution followed by configurable Transformer layers."""

    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = Float32LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        layer_type = (
            Wav2Vec2EncoderLayerStableLayerNorm if config.do_stable_layer_norm else Wav2Vec2EncoderLayer)
        self.layers = nn.ModuleList(layer_type(config) for _ in range(config.num_hidden_layers))

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Wav2Vec2EncoderOutput:
        if hidden_states.ndim != 3:
            raise ValueError("Wav2Vec2 encoder input must be [batch, time, hidden].")
        if (attention_mask.dtype != torch.bool or
                tuple(attention_mask.shape) != tuple(hidden_states.shape[:2])):
            raise ValueError("Encoder attention mask must be boolean [batch, time].")

        hidden_states = hidden_states.masked_fill(
            ~attention_mask.unsqueeze(-1),
            0.0,
        )
        hidden_states = hidden_states + self.pos_conv_embed(hidden_states)
        if not self.config.do_stable_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        collected_states: list[Tensor] | None = ([] if output_hidden_states else None)
        collected_attentions: list[Tensor | None] | None = ([] if output_attentions else None)
        executed_layers: list[bool] = []
        for layer in self.layers:
            if collected_states is not None:
                collected_states.append(hidden_states)
            skip_layer = False
            if self.training and self.config.layerdrop > 0.0:
                probability = torch.rand((), device=hidden_states.device)
                skip_layer = bool(probability < self.config.layerdrop)
            if skip_layer:
                attention = None
            else:
                hidden_states, attention = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )
            executed_layers.append(not skip_layer)
            if collected_attentions is not None:
                collected_attentions.append(attention)

        if self.config.do_stable_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        if collected_states is not None:
            collected_states.append(hidden_states)
        return Wav2Vec2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=(None if collected_states is None else tuple(collected_states)),
            attentions=(None if collected_attentions is None else tuple(collected_attentions)),
            executed_layers=tuple(executed_layers),
        )


def _span_mask(
    valid_mask: Tensor,
    *,
    probability: float,
    span_length: int,
    minimum_spans: int,
    generator: torch.Generator | None,
) -> Tensor:
    """Generate bounded SpecAugment spans using only PyTorch."""
    if valid_mask.dtype != torch.bool or valid_mask.ndim != 2:
        raise ValueError("SpecAugment valid mask must be boolean [batch, axis].")
    result = torch.zeros_like(valid_mask)
    if probability <= 0.0 and minimum_spans == 0:
        return result
    sequence_length = valid_mask.shape[1]
    if span_length > sequence_length:
        raise ValueError("A SpecAugment span is longer than the available input axis.")
    stochastic_rounding = float(torch.rand(
        (),
        device=valid_mask.device,
        generator=generator,
    ).item())
    for batch_index, raw_length in enumerate(valid_mask.sum(dim=-1)):
        valid_length = int(raw_length.item())
        span_count = max(
            minimum_spans,
            int(probability * valid_length / span_length + stochastic_rounding),
        )
        if span_count * span_length > sequence_length:
            span_count = sequence_length // span_length
        possible_starts = valid_length - span_length + 1
        span_count = min(span_count, max(possible_starts, 0))
        if span_count == 0:
            continue
        starts = torch.randperm(
            possible_starts,
            device=valid_mask.device,
            generator=generator,
        )[:span_count]
        offsets = torch.arange(span_length, device=valid_mask.device)
        indices = (starts[:, None] + offsets[None, :]).reshape(-1)
        result[batch_index, indices] = True
    return result


@dataclass(frozen=True)
class Wav2Vec2ModelOutput:
    """Native base-model output with explicit frame lengths and mask."""

    last_hidden_state: Tensor
    extract_features: Tensor
    feature_attention_mask: Tensor
    input_lengths: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor | None, ...] | None = None
    executed_layers: tuple[bool, ...] = ()
    past_key_values: None = None


class Wav2Vec2Model(nn.Module):
    """Native Wav2Vec2 feature frontend and bidirectional encoder."""

    def __init__(
        self,
        config: Wav2Vec2Config | Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.config = Wav2Vec2Config.coerce(config)
        self.feature_extractor = Wav2Vec2FeatureEncoder(self.config)
        self.feature_projection = Wav2Vec2FeatureProjection(self.config)
        self.encoder = Wav2Vec2Encoder(self.config)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        config = self.config

        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    bound = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        self.feature_projection.projection.reset_parameters()
        self.encoder.pos_conv_embed.conv.reset_parameters()

    @staticmethod
    def _reject_cache(
        use_cache: bool | None,
        past_key_values: Any | None,
    ) -> None:
        if use_cache not in (None, False):
            raise ValueError(
                "Wav2Vec2 is a bidirectional encoder and does not support "
                "causal key/value caching.")
        if past_key_values is not None:
            raise ValueError("`past_key_values` is invalid for the bidirectional "
                             "Wav2Vec2 encoder.")

    def _apply_spec_augment(
        self,
        hidden_states: Tensor,
        feature_mask: Tensor,
        *,
        mask_time_indices: Tensor | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        if mask_time_indices is not None:
            if (not isinstance(mask_time_indices, Tensor) or
                    tuple(mask_time_indices.shape) != tuple(feature_mask.shape)):
                raise ValueError("`mask_time_indices` must have shape [batch, feature_time].")
            if mask_time_indices.device != hidden_states.device:
                raise ValueError("`mask_time_indices` must be on the model input device.")
            if not (mask_time_indices.dtype == torch.bool or
                    ((mask_time_indices == 0) | (mask_time_indices == 1)).all()):
                raise ValueError("`mask_time_indices` must contain only zero and one.")
            time_mask = mask_time_indices.to(dtype=torch.bool)
            if (time_mask & ~feature_mask).any():
                raise ValueError("`mask_time_indices` cannot select padded feature frames.")
        elif (self.training and self.config.apply_spec_augment and self.config.mask_time_prob > 0.0):
            time_mask = _span_mask(
                feature_mask,
                probability=self.config.mask_time_prob,
                span_length=self.config.mask_time_length,
                minimum_spans=self.config.mask_time_min_masks,
                generator=generator,
            )
        else:
            time_mask = None
        if time_mask is not None:
            hidden_states = hidden_states.masked_fill(
                time_mask.unsqueeze(-1),
                0.0,
            )

        if (self.training and self.config.apply_spec_augment and self.config.mask_feature_prob > 0.0):
            feature_valid = torch.ones(
                (hidden_states.shape[0], hidden_states.shape[2]),
                dtype=torch.bool,
                device=hidden_states.device,
            )
            masked_features = _span_mask(
                feature_valid,
                probability=self.config.mask_feature_prob,
                span_length=self.config.mask_feature_length,
                minimum_spans=self.config.mask_feature_min_masks,
                generator=generator,
            )
            hidden_states = hidden_states.masked_fill(
                masked_features.unsqueeze(1),
                0.0,
            )
        return hidden_states

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None = None,
        *,
        mask_time_indices: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool | None = None,
        past_key_values: Any | None = None,
        generator: torch.Generator | None = None,
    ) -> Wav2Vec2ModelOutput:
        _validate_floating_input(input_values, self.config)
        self._reject_cache(use_cache, past_key_values)
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("`generator` must be a PyTorch Generator or None.")
        for name, value in (
            ("output_attentions", output_attentions),
            ("output_hidden_states", output_hidden_states),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"`{name}` must be a boolean.")

        raw_mask, raw_lengths = _validated_raw_attention_mask(
            attention_mask,
            input_values=input_values,
            minimum_input_samples=self.config.minimum_input_samples,
        )
        masked_input = input_values.masked_fill(~raw_mask, 0.0)
        extract_features = self.feature_extractor(masked_input).transpose(1, 2)
        output_lengths = downsample_wav2vec2_lengths(
            raw_lengths,
            self.config,
        )
        encoded_mask = feature_attention_mask(
            extract_features.shape[1],
            output_lengths,
        )
        hidden_states, normalized_features = self.feature_projection(extract_features)
        hidden_states = self._apply_spec_augment(
            hidden_states,
            encoded_mask,
            mask_time_indices=mask_time_indices,
            generator=generator,
        )
        encoded = self.encoder(
            hidden_states,
            attention_mask=encoded_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return Wav2Vec2ModelOutput(
            last_hidden_state=encoded.last_hidden_state,
            extract_features=normalized_features,
            feature_attention_mask=encoded_mask,
            input_lengths=output_lengths,
            hidden_states=encoded.hidden_states,
            attentions=encoded.attentions,
            executed_layers=encoded.executed_layers,
        )

    def freeze_feature_encoder(self) -> None:
        """Freeze the raw-waveform convolutional frontend."""
        self.feature_extractor.freeze()


@dataclass(frozen=True)
class Wav2Vec2CTCOutput:
    """CTC logits, optional loss, and encoder diagnostics."""

    logits: Tensor
    loss: Tensor | None
    feature_attention_mask: Tensor
    input_lengths: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor | None, ...] | None = None
    executed_layers: tuple[bool, ...] = ()
    past_key_values: None = None


class Wav2Vec2ForCTC(nn.Module):
    """Wav2Vec2 encoder with a trainable native CTC projection head."""

    def __init__(
        self,
        config: Wav2Vec2Config | Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.config = Wav2Vec2Config.coerce(config)
        self.wav2vec2 = Wav2Vec2Model(self.config)
        self.dropout = nn.Dropout(self.config.final_dropout)
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
        )
        nn.init.normal_(
            self.lm_head.weight,
            mean=0.0,
            std=self.config.initializer_range,
        )
        nn.init.zeros_(self.lm_head.bias)

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None = None,
        *,
        labels: Tensor | None = None,
        mask_time_indices: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool | None = None,
        past_key_values: Any | None = None,
        generator: torch.Generator | None = None,
    ) -> Wav2Vec2CTCOutput:
        outputs = self.wav2vec2(
            input_values,
            attention_mask,
            mask_time_indices=mask_time_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            generator=generator,
        )
        logits = self.lm_head(self.dropout(outputs.last_hidden_state))

        loss = None
        if labels is not None:
            if not isinstance(labels, Tensor):
                raise TypeError("`labels` must be a PyTorch tensor.")
            if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
                raise TypeError("`labels` must use an integer dtype.")
            if labels.ndim != 2 or labels.shape[0] != input_values.shape[0]:
                raise ValueError("`labels` must have shape [batch, target_time].")
            if labels.device != logits.device:
                raise ValueError("`labels` must be on the model input device.")
            if ((labels < 0) & (labels != -100)).any():
                raise ValueError("Negative CTC labels must use the -100 ignore index.")
            label_mask = labels >= 0
            target_lengths = label_mask.sum(dim=-1, dtype=torch.long)
            targets = labels.masked_select(label_mask)
            loss = ctc_loss(
                logits,
                targets,
                outputs.input_lengths,
                target_lengths,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )

        return Wav2Vec2CTCOutput(
            logits=logits,
            loss=loss,
            feature_attention_mask=outputs.feature_attention_mask,
            input_lengths=outputs.input_lengths,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            executed_layers=outputs.executed_layers,
        )

    def freeze_feature_encoder(self) -> None:
        """Freeze only the raw-waveform convolutional frontend."""
        self.wav2vec2.freeze_feature_encoder()

    def freeze_base_model(self) -> None:
        """Freeze the frontend and Transformer while leaving the CTC head."""
        for parameter in self.wav2vec2.parameters():
            parameter.requires_grad_(False)


@dataclass(frozen=True)
class Wav2Vec2SequenceClassifierOutput:
    """Clip-level classification logits and encoder diagnostics."""

    logits: Tensor
    loss: Tensor | None
    feature_attention_mask: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor | None, ...] | None = None
    executed_layers: tuple[bool, ...] = ()


@dataclass(frozen=True)
class Wav2Vec2FrameClassifierOutput:
    """Frame-level classification logits and encoder diagnostics."""

    logits: Tensor
    loss: Tensor | None
    feature_attention_mask: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor | None, ...] | None = None
    executed_layers: tuple[bool, ...] = ()


class _Wav2Vec2ClassificationHead(nn.Module):
    """Shared hidden-state selection for official classification heads."""

    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.config = config
        if config.use_weighted_layer_sum:
            layer_count = config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.ones(layer_count) / layer_count)
        else:
            self.register_parameter("layer_weights", None)

    def _encoder(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None,
        *,
        output_attentions: bool,
        output_hidden_states: bool,
        generator: torch.Generator | None,
    ) -> tuple[Wav2Vec2ModelOutput, Tensor]:
        require_hidden_states = (output_hidden_states or self.config.use_weighted_layer_sum)
        outputs = self.wav2vec2(
            input_values,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=require_hidden_states,
            generator=generator,
        )
        if self.config.use_weighted_layer_sum:
            if outputs.hidden_states is None:
                raise RuntimeError("Weighted layer aggregation requires encoder hidden states.")
            hidden_states = torch.stack(outputs.hidden_states, dim=1)
            weights = functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * weights.view(1, -1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs.last_hidden_state
        return outputs, hidden_states

    def freeze_feature_encoder(self) -> None:
        """Freeze only the raw-waveform convolutional frontend."""
        self.wav2vec2.freeze_feature_encoder()

    def freeze_base_model(self) -> None:
        """Freeze the frontend and Transformer while leaving the task head."""
        for parameter in self.wav2vec2.parameters():
            parameter.requires_grad_(False)


class Wav2Vec2ForSequenceClassification(_Wav2Vec2ClassificationHead):
    """Official-compatible clip classification head for native Wav2Vec2."""

    def __init__(
        self,
        config: Wav2Vec2Config | Mapping[str, Any],
    ) -> None:
        resolved = Wav2Vec2Config.coerce(config)
        super().__init__(resolved)
        self.wav2vec2 = Wav2Vec2Model(resolved)
        self.projector = nn.Linear(
            resolved.hidden_size,
            resolved.classifier_proj_size,
        )
        self.classifier = nn.Linear(
            resolved.classifier_proj_size,
            resolved.num_labels,
        )
        self._initialize_head()

    def _initialize_head(self) -> None:
        for layer in (self.projector, self.classifier):
            nn.init.normal_(
                layer.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None = None,
        *,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        generator: torch.Generator | None = None,
    ) -> Wav2Vec2SequenceClassifierOutput:
        outputs, hidden_states = self._encoder(
            input_values,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generator=generator,
        )
        hidden_states = self.projector(hidden_states)
        mask = outputs.feature_attention_mask.unsqueeze(-1)
        pooled = hidden_states.masked_fill(~mask, 0.0).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp_min(1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if not isinstance(labels, Tensor):
                raise TypeError("`labels` must be a PyTorch tensor.")
            if labels.device != logits.device:
                raise ValueError("`labels` must be on the model input device.")
            problem_type = self.config.problem_type
            if problem_type is None:
                if self.config.num_labels == 1:
                    problem_type = "regression"
                elif not labels.is_floating_point():
                    problem_type = "single_label_classification"
                else:
                    problem_type = "multi_label_classification"
            if problem_type == "regression":
                loss = functional.mse_loss(
                    logits.squeeze(-1),
                    labels.to(dtype=logits.dtype).squeeze(-1),
                )
            elif problem_type == "single_label_classification":
                if labels.is_floating_point() or labels.is_complex():
                    raise TypeError("Single-label classification requires integer labels.")
                loss = functional.cross_entropy(
                    logits,
                    labels.to(dtype=torch.long).reshape(-1),
                )
            else:
                if tuple(labels.shape) != tuple(logits.shape):
                    raise ValueError("Multi-label targets must match the logits shape.")
                loss = functional.binary_cross_entropy_with_logits(
                    logits,
                    labels.to(dtype=logits.dtype),
                )
        return Wav2Vec2SequenceClassifierOutput(
            logits=logits,
            loss=loss,
            feature_attention_mask=outputs.feature_attention_mask,
            hidden_states=(outputs.hidden_states if output_hidden_states else None),
            attentions=outputs.attentions,
            executed_layers=outputs.executed_layers,
        )


class Wav2Vec2ForAudioFrameClassification(_Wav2Vec2ClassificationHead):
    """Official-compatible frame classifier for segmentation and VAD."""

    def __init__(
        self,
        config: Wav2Vec2Config | Mapping[str, Any],
    ) -> None:
        resolved = Wav2Vec2Config.coerce(config)
        super().__init__(resolved)
        self.wav2vec2 = Wav2Vec2Model(resolved)
        self.classifier = nn.Linear(
            resolved.hidden_size,
            resolved.num_labels,
        )
        nn.init.normal_(
            self.classifier.weight,
            mean=0.0,
            std=resolved.initializer_range,
        )
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None = None,
        *,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        generator: torch.Generator | None = None,
    ) -> Wav2Vec2FrameClassifierOutput:
        outputs, hidden_states = self._encoder(
            input_values,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generator=generator,
        )
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            if not isinstance(labels, Tensor):
                raise TypeError("`labels` must be a PyTorch tensor.")
            if labels.device != logits.device:
                raise ValueError("`labels` must be on the model input device.")
            if labels.ndim == 3:
                if tuple(labels.shape) != tuple(logits.shape):
                    raise ValueError("One-hot frame labels must match the logits shape.")
                targets = labels.argmax(dim=-1)
            elif labels.ndim == 2:
                if tuple(labels.shape) != tuple(logits.shape[:2]):
                    raise ValueError("Frame labels must have shape [batch, feature_time].")
                if labels.is_floating_point() or labels.is_complex():
                    raise TypeError("Class-index frame labels must be integers.")
                targets = labels.to(dtype=torch.long)
            else:
                raise ValueError("Frame labels must be class indices or one-hot targets.")
            valid_targets = targets.masked_fill(
                ~outputs.feature_attention_mask,
                -100,
            )
            loss = functional.cross_entropy(
                logits.reshape(-1, self.config.num_labels),
                valid_targets.reshape(-1),
                ignore_index=-100,
            )
        return Wav2Vec2FrameClassifierOutput(
            logits=logits,
            loss=loss,
            feature_attention_mask=outputs.feature_attention_mask,
            hidden_states=(outputs.hidden_states if output_hidden_states else None),
            attentions=outputs.attentions,
            executed_layers=outputs.executed_layers,
        )


__all__ = [
    "Float32LayerNorm",
    "Wav2Vec2Attention",
    "Wav2Vec2CTCOutput",
    "Wav2Vec2Encoder",
    "Wav2Vec2EncoderLayer",
    "Wav2Vec2EncoderLayerStableLayerNorm",
    "Wav2Vec2EncoderOutput",
    "Wav2Vec2FeatureConvLayer",
    "Wav2Vec2FeatureEncoder",
    "Wav2Vec2FeatureProjection",
    "Wav2Vec2ForCTC",
    "Wav2Vec2Model",
    "Wav2Vec2ModelOutput",
    "Wav2Vec2PositionalConvEmbedding",
    "Wav2Vec2FeedForward",
    "WeightNormalizedConv1d",
    "downsample_wav2vec2_lengths",
    "feature_attention_mask",
]
