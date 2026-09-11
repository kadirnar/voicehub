"""VoiceHub-owned XCodec2 graph used by LLaSA.

The parameter namespace follows the authors' self-contained Transformers
conversion at revision ``7f5d5d1aaca3cc3d236c80ec8cb34d06f08a5fb8``. The
implementation is intentionally PyTorch-only: Wav2Vec2-BERT semantic
encoding, BigCodec acoustic encoding, finite scalar quantization, the
Transformer/Vocos decoder, and Kaldi-compatible preprocessing all
execute inside VoiceHub.
"""

# Portions are adapted from the Hugging Face XCodec2 implementation:
# Copyright 2026 The HuggingFace Inc. team, Apache License 2.0.

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicehub.kernels.codecs import CodecSnakeBetaKernelOptimizable
from voicehub.processing.kaldi import KaldiFbankConfig, kaldi_fbank

XCODEC2_TRANSFORMERS_SOURCE_REVISION = ("7f5d5d1aaca3cc3d236c80ec8cb34d06f08a5fb8")


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return value


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"`{name}` must be finite and in [0, 1).")
    return result


@dataclass(frozen=True, slots=True)
class Wav2Vec2BertSemanticConfig:
    """Executable subset of the embedded Wav2Vec2-BERT configuration."""

    feature_projection_input_dim: int = 160
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    hidden_act: str = "swish"
    hidden_dropout: float = 0.0
    activation_dropout: float = 0.0
    attention_dropout: float = 0.0
    feat_proj_dropout: float = 0.0
    conformer_conv_dropout: float = 0.1
    conv_depthwise_kernel_size: int = 31
    layer_norm_eps: float = 1e-5
    layerdrop: float = 0.1
    position_embeddings_type: str = "relative_key"
    left_max_position_embeddings: int = 64
    right_max_position_embeddings: int = 8
    apply_spec_augment: bool = False
    mask_time_prob: float = 0.05
    mask_feature_prob: float = 0.0
    add_adapter: bool = False
    use_intermediate_ffn_before_adapter: bool = False
    output_hidden_size: int = 1024
    extra_config: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "feature_projection_input_dim",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "conv_depthwise_kernel_size",
                "left_max_position_embeddings",
                "right_max_position_embeddings",
                "output_hidden_size",
        ):
            _positive_integer(getattr(self, name), name=name)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("Semantic `hidden_size` must be divisible by "
                             "`num_attention_heads`.")
        if self.conv_depthwise_kernel_size % 2 != 1:
            raise ValueError("Semantic `conv_depthwise_kernel_size` must be odd.")
        if self.hidden_act not in {"silu", "swish"}:
            raise ValueError("Native XCodec2 semantic checkpoints require SiLU/Swish.")
        if self.position_embeddings_type != "relative_key":
            raise ValueError(
                "Native XCodec2 currently supports the published "
                "`relative_key` semantic attention only.")
        for name in (
                "hidden_dropout",
                "activation_dropout",
                "attention_dropout",
                "feat_proj_dropout",
                "conformer_conv_dropout",
                "layerdrop",
                "mask_time_prob",
                "mask_feature_prob",
        ):
            object.__setattr__(
                self,
                name,
                _probability(getattr(self, name), name=name),
            )
        if self.add_adapter or self.use_intermediate_ffn_before_adapter:
            raise ValueError("The published XCodec2 checkpoint has no Wav2Vec2-BERT "
                             "adapter stack.")

    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any] | None,
    ) -> Wav2Vec2BertSemanticConfig:
        values = dict(values or {})
        known = {item.name for item in fields(cls)} - {"extra_config"}
        selected = {name: values.pop(name) for name in tuple(values) if name in known}
        selected["extra_config"] = values
        return cls(**selected)

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        extra = output.pop("extra_config")
        output.update(extra)
        output["model_type"] = "wav2vec2-bert"
        return output


@dataclass(frozen=True, slots=True)
class XCodec2Config:
    """Validated architecture values for the released 16 kHz codec."""

    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    encoder_hidden_size: int = 48
    downsampling_ratios: tuple[int, ...] = (2, 2, 4, 4, 5)
    sampling_rate: int = 16_000
    activation_dropout: float = 0.1
    quantization_dim: int = 2048
    quantization_levels: tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4, 4)
    rope_theta: float = 10_000.0
    semantic_model_config: Wav2Vec2BertSemanticConfig = field(default_factory=Wav2Vec2BertSemanticConfig)
    extra_config: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
                "encoder_hidden_size",
                "sampling_rate",
                "quantization_dim",
        ):
            _positive_integer(getattr(self, name), name=name)
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("`num_key_value_heads` must divide `num_attention_heads`.")
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError(
                "The published XCodec2 decoder uses one key/value head per "
                "attention head. Grouped-query variants are not verified.")
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("XCodec2 requires `num_attention_heads * head_dim == "
                             "hidden_size`.")
        if self.hidden_size % 32:
            raise ValueError(
                "XCodec2 decoder hidden size must be divisible by 32 for "
                "the published GroupNorm topology.")
        if self.max_position_embeddings < self.num_attention_heads:
            raise ValueError(
                "`max_position_embeddings` must cover XCodec2's head-indexed "
                "decoder rotary positions.")
        if self.hidden_act != "silu":
            raise ValueError("XCodec2 decoder checkpoints require SiLU.")
        if self.attention_bias:
            raise ValueError(
                "The published XCodec2 checkpoint uses bias-free decoder "
                "attention projections.")
        object.__setattr__(
            self,
            "attention_dropout",
            _probability(self.attention_dropout, name="attention_dropout"),
        )
        object.__setattr__(
            self,
            "activation_dropout",
            _probability(self.activation_dropout, name="activation_dropout"),
        )
        ratios = tuple(self.downsampling_ratios)
        if not ratios:
            raise ValueError("`downsampling_ratios` cannot be empty.")
        for ratio in ratios:
            _positive_integer(ratio, name="downsampling_ratios")
        object.__setattr__(self, "downsampling_ratios", ratios)
        levels = tuple(self.quantization_levels)
        if not levels or any(isinstance(level, bool) or not isinstance(level, int) or level < 2
                             for level in levels):
            raise ValueError("`quantization_levels` must contain integers of at least two.")
        object.__setattr__(self, "quantization_levels", levels)
        if math.prod(levels) != 65_536:
            raise ValueError("LLaSA checkpoints require XCodec2's 65,536-entry codebook.")
        semantic = self.semantic_model_config
        if isinstance(semantic, dict):
            semantic = Wav2Vec2BertSemanticConfig.from_dict(semantic)
            object.__setattr__(self, "semantic_model_config", semantic)
        if not isinstance(semantic, Wav2Vec2BertSemanticConfig):
            raise TypeError("`semantic_model_config` must be a mapping or semantic config.")
        if self.quantization_dim != self.hidden_size + semantic.hidden_size:
            raise ValueError("`quantization_dim` must equal acoustic plus semantic hidden "
                             "sizes.")
        if semantic.feature_projection_input_dim != 160:
            raise ValueError(
                "The native XCodec2 frontend emits 160-dimensional paired "
                "Kaldi filter-bank frames.")
        if self.sampling_rate != self.hop_length * 50:
            raise ValueError(
                "XCodec2 acoustic and semantic frames must both run at 50 Hz; "
                "`sampling_rate` must equal 50 * `hop_length`.")
        if (isinstance(self.rope_theta, bool) or not isinstance(self.rope_theta, (int, float)) or
                not math.isfinite(self.rope_theta) or self.rope_theta <= 1.0):
            raise ValueError("`rope_theta` must be finite and greater than one.")

    @property
    def hop_length(self) -> int:
        return math.prod(self.downsampling_ratios)

    @property
    def n_fft(self) -> int:
        return self.hop_length * 4

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> XCodec2Config:
        if not isinstance(values, dict):
            raise TypeError("XCodec2 config must be a mapping.")
        values = dict(values)
        if values.get("model_type", "xcodec2") != "xcodec2":
            raise ValueError("XCodec2 config must declare `model_type=xcodec2`.")
        values.pop("model_type", None)
        rope = values.pop("rope_parameters", None)
        if rope is not None:
            if not isinstance(rope, dict):
                raise TypeError("`rope_parameters` must be a mapping.")
            if rope.get("rope_type", "default") != "default":
                raise ValueError("XCodec2 supports default RoPE only.")
            values["rope_theta"] = rope.get("rope_theta", 10_000.0)
        semantic = values.pop("semantic_model_config", None)
        known = {item.name for item in fields(cls)} - {"extra_config"}
        selected = {name: values.pop(name) for name in tuple(values) if name in known}
        if semantic is not None:
            selected["semantic_model_config"] = (Wav2Vec2BertSemanticConfig.from_dict(semantic))
        selected["extra_config"] = values
        return cls(**selected)

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        extra = output.pop("extra_config")
        output.update(extra)
        output["model_type"] = "xcodec2"
        output["downsampling_ratios"] = list(self.downsampling_ratios)
        output["quantization_levels"] = list(self.quantization_levels)
        output["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": self.rope_theta,
        }
        output.pop("rope_theta")
        output["semantic_model_config"] = self.semantic_model_config.to_dict()
        output['architectures'] = ["Xcodec2Model"]
        return output


@dataclass(frozen=True, slots=True)
class XCodec2EncoderOutput:
    audio_codes: Tensor
    latents: Tensor | None = None
    audio_codes_mask: Tensor | None = None


@dataclass(frozen=True, slots=True)
class XCodec2DecoderOutput:
    audio_values: Tensor


@dataclass(frozen=True, slots=True)
class XCodec2Output:
    audio_values: Tensor
    audio_codes: Tensor
    latents: Tensor | None = None
    audio_codes_mask: Tensor | None = None


@dataclass(frozen=True, slots=True)
class XCodec2Features:
    input_values: Tensor
    input_features: Tensor
    padding_mask: Tensor
    input_features_mask: Tensor


class Wav2Vec2BertFeatureProjection(nn.Module):

    def __init__(self, config: Wav2Vec2BertSemanticConfig) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            config.feature_projection_input_dim,
            eps=config.layer_norm_eps,
        )
        self.projection = nn.Linear(
            config.feature_projection_input_dim,
            config.hidden_size,
        )
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.layer_norm(hidden_states)
        return self.dropout(self.projection(normalized)), normalized


class Wav2Vec2BertFeedForward(nn.Module):

    def __init__(self, config: Wav2Vec2BertSemanticConfig) -> None:
        super().__init__()
        self.intermediate_dense = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
        )
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.output_dense = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
        )
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        return self.output_dropout(self.output_dense(hidden_states))


class Wav2Vec2BertConvolutionModule(nn.Module):

    def __init__(self, config: Wav2Vec2BertSemanticConfig) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            groups=config.hidden_size,
            bias=False,
        )
        self.depthwise_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.dropout = nn.Dropout(config.conformer_conv_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden_states = self.layer_norm(hidden_states)
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(
                ~attention_mask.bool().unsqueeze(-1),
                0.0,
            )
        hidden_states = self.pointwise_conv1(hidden_states.transpose(1, 2))
        hidden_states = F.glu(hidden_states, dim=1)
        hidden_states = F.pad(
            hidden_states,
            (self.depthwise_conv.kernel_size[0] - 1, 0),
        )
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        return self.dropout(hidden_states).transpose(1, 2)


class Wav2Vec2BertSelfAttention(nn.Module):

    def __init__(self, config: Wav2Vec2BertSemanticConfig) -> None:
        super().__init__()
        self.head_size = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.left_max_position_embeddings = (config.left_max_position_embeddings)
        self.right_max_position_embeddings = (config.right_max_position_embeddings)
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)
        positions = (self.left_max_position_embeddings + self.right_max_position_embeddings + 1)
        self.distance_embedding = nn.Embedding(positions, self.head_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        shape = (batch_size, sequence_length, self.num_heads, self.head_size)
        query = self.linear_q(hidden_states).view(shape).transpose(1, 2)
        key = self.linear_k(hidden_states).view(shape).transpose(1, 2)
        value = self.linear_v(hidden_states).view(shape).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_size)

        left = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=hidden_states.device,
        ).view(-1, 1)
        right = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=hidden_states.device,
        ).view(1, -1)
        distance = (right - left).clamp(
            -self.left_max_position_embeddings,
            self.right_max_position_embeddings,
        )
        positional = self.distance_embedding(distance +
                                             self.left_max_position_embeddings).to(dtype=query.dtype)
        relative_scores = torch.einsum(
            "bhld,lrd->bhlr",
            query,
            positional,
        )
        scores = scores + relative_scores / math.sqrt(self.head_size)
        if attention_mask is not None:
            scores = scores + attention_mask
        probabilities = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(probabilities, value)
        output = output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.num_heads * self.head_size,
        )
        return self.linear_out(output), probabilities


class Wav2Vec2BertEncoderLayer(nn.Module):

    def __init__(self, config: Wav2Vec2BertSemanticConfig) -> None:
        super().__init__()
        self.ffn1_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.ffn1 = Wav2Vec2BertFeedForward(config)
        self.self_attn_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.self_attn_dropout = nn.Dropout(config.attention_dropout)
        self.self_attn = Wav2Vec2BertSelfAttention(config)
        self.conv_module = Wav2Vec2BertConvolutionModule(config)
        self.ffn2_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.ffn2 = Wav2Vec2BertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        convolution_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        residual = hidden_states
        hidden_states = (self.ffn1(self.ffn1_layer_norm(hidden_states)) * 0.5 + residual)
        residual = hidden_states
        attention_output, probabilities = self.self_attn(
            self.self_attn_layer_norm(hidden_states),
            attention_mask=attention_mask,
        )
        hidden_states = (self.self_attn_dropout(attention_output) + residual)
        residual = hidden_states
        hidden_states = (self.conv_module(
            hidden_states,
            attention_mask=convolution_mask,
        ) + residual)
        residual = hidden_states
        hidden_states = (self.ffn2(self.ffn2_layer_norm(hidden_states)) * 0.5 + residual)
        return self.final_layer_norm(hidden_states), probabilities


class Wav2Vec2BertEncoder(nn.Module):

    def __init__(self, config: Wav2Vec2BertSemanticConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(Wav2Vec2BertEncoderLayer(config) for _ in range(config.num_hidden_layers))

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        convolution_mask = attention_mask
        additive_mask = None
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(
                ~attention_mask.bool().unsqueeze(-1),
                0.0,
            )
            additive_mask = (1.0 - attention_mask[:, None, None, :].to(hidden_states.dtype))
            additive_mask = additive_mask * torch.finfo(hidden_states.dtype).min
            additive_mask = additive_mask.expand(
                attention_mask.shape[0],
                1,
                attention_mask.shape[-1],
                attention_mask.shape[-1],
            )
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers:
            if self.training and torch.rand(()) < self.config.layerdrop:
                continue
            hidden_states, _ = layer(
                hidden_states,
                attention_mask=additive_mask,
                convolution_mask=convolution_mask,
            )
        return hidden_states


class Wav2Vec2BertSemanticModel(nn.Module):

    def __init__(self, config: Wav2Vec2BertSemanticConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_projection = Wav2Vec2BertFeatureProjection(config)
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.empty(config.hidden_size))
        else:
            self.register_parameter("masked_spec_embed", None)
        self.encoder = Wav2Vec2BertEncoder(config)

    def forward(
        self,
        input_features: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden_states, _ = self.feature_projection(input_features)
        return self.encoder(
            hidden_states,
            attention_mask=attention_mask,
        )


class XCodec2SnakeBeta(CodecSnakeBetaKernelOptimizable, nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self._initialize_codec_kernel_backend()

    def forward(self, hidden_states: Tensor) -> Tensor:
        alpha = self.alpha[None, :, None].exp()
        beta = self.beta[None, :, None].exp()
        return self._codec_snake_beta(hidden_states, alpha, beta)


def kaiser_sinc_filter1d(
    cutoff: float,
    half_width: float,
    kernel_size: int,
) -> Tensor:
    """Return the exact anti-aliasing kernel used by XCodec2."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    attenuation = (2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95)
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = (0.5842 * (attenuation - 21)**0.4 + 0.07886 * (attenuation - 21.0))
    else:
        beta = 0.0
    window = torch.kaiser_window(
        kernel_size,
        beta=beta,
        periodic=False,
        dtype=torch.float32,
    )
    positions = (
        torch.arange(-half_size, half_size, dtype=torch.float32) +
        0.5 if even else torch.arange(kernel_size, dtype=torch.float32) - half_size)
    if cutoff == 0:
        return torch.zeros((1, 1, kernel_size), dtype=torch.float32)
    kernel = 2 * cutoff * window * torch.sinc(2 * cutoff * positions)
    return (kernel / kernel.sum()).view(1, 1, kernel_size)


class XCodec2DownSample1d(nn.Module):

    def __init__(self, ratio: int, kernel_size: int) -> None:
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio
        self.cutoff = cutoff
        self.half_width = half_width
        self.kernel_size = kernel_size
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(cutoff, half_width, kernel_size),
            persistent=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        channels = hidden_states.shape[1]
        hidden_states = F.pad(
            hidden_states,
            (self.pad_left, self.pad_right),
            mode="replicate",
        )
        return F.conv1d(
            hidden_states,
            self.filter.to(hidden_states.dtype).expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )


class XCodec2UpSample1d(nn.Module):

    def __init__(self, ratio: int, kernel_size: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = (self.pad * ratio + (kernel_size - ratio) // 2)
        self.pad_right = (self.pad * ratio + (kernel_size - ratio + 1) // 2)
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(
                0.5 / ratio,
                0.6 / ratio,
                kernel_size,
            ),
            persistent=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        channels = hidden_states.shape[1]
        hidden_states = F.pad(
            hidden_states,
            (self.pad, self.pad),
            mode="replicate",
        )
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states,
            self.filter.to(hidden_states.dtype).expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )
        return hidden_states[..., self.pad_left:-self.pad_right]


class XCodec2AntiAliasedActivation1d(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.act = XCodec2SnakeBeta(channels)
        self.upsample = XCodec2UpSample1d(2, 12)
        self.downsample = XCodec2DownSample1d(2, 12)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.downsample(self.act(self.upsample(hidden_states)))


class XCodec2ResidualUnit(nn.Module):

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = ((7 - 1) * dilation) // 2
        self.snake1 = XCodec2AntiAliasedActivation1d(channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            dilation=dilation,
            padding=padding,
        )
        self.snake2 = XCodec2AntiAliasedActivation1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        output = self.conv1(self.snake1(hidden_states))
        output = self.conv2(self.snake2(output))
        padding = (hidden_states.shape[-1] - output.shape[-1]) // 2
        if padding > 0:
            hidden_states = hidden_states[..., padding:-padding]
        return hidden_states + output


class XCodec2EncoderBlock(nn.Module):

    def __init__(
        self,
        config: XCodec2Config,
        stride: int,
        stride_index: int,
    ) -> None:
        super().__init__()
        channels = config.encoder_hidden_size * 2**(stride_index - 1)
        self.res_unit1 = XCodec2ResidualUnit(channels, dilation=1)
        self.res_unit2 = XCodec2ResidualUnit(channels, dilation=3)
        self.res_unit3 = XCodec2ResidualUnit(channels, dilation=9)
        self.snake1 = XCodec2AntiAliasedActivation1d(channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.res_unit1(hidden_states)
        hidden_states = self.res_unit2(hidden_states)
        hidden_states = self.res_unit3(hidden_states)
        return self.conv1(self.snake1(hidden_states))


class XCodec2Encoder(nn.Module):

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            1,
            config.encoder_hidden_size,
            kernel_size=7,
            padding=3,
        )
        self.block = nn.ModuleList(
            XCodec2EncoderBlock(config, stride, index + 1)
            for index, stride in enumerate(config.downsampling_ratios))
        channels = (config.encoder_hidden_size * 2**len(config.downsampling_ratios))
        self.snake1 = XCodec2AntiAliasedActivation1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            config.hidden_size,
            kernel_size=3,
            padding=1,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv1(hidden_states)
        for block in self.block:
            hidden_states = block(hidden_states)
        return self.conv2(self.snake1(hidden_states))


class XCodec2ResNetBlock(nn.Module):

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(
            32,
            config.hidden_size,
            eps=1e-6,
        )
        self.conv1 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.norm2 = nn.GroupNorm(
            32,
            config.hidden_size,
            eps=1e-6,
        )
        self.activation_dropout = config.activation_dropout
        self.conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=3,
            padding=1,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        residual = hidden_states
        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))
        hidden_states = F.silu(self.norm2(hidden_states))
        hidden_states = F.dropout(
            hidden_states,
            p=self.activation_dropout,
            training=self.training,
        )
        hidden_states = self.conv2(hidden_states)
        return (hidden_states + residual).transpose(1, 2)


class XCodec2FiniteScalarQuantization(nn.Module):

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.quantization_levels = tuple(config.quantization_levels)
        levels, basis, codebook = self._compute_buffers()
        self.register_buffer("levels", levels, persistent=False)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("codebook", codebook, persistent=False)

    def _compute_buffers(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        levels = torch.tensor(
            self.quantization_levels,
            dtype=torch.int32,
            device=device,
        )
        basis = torch.cumprod(
            torch.tensor(
                [1, *self.quantization_levels[:-1]],
                dtype=torch.int32,
                device=device,
            ),
            dim=0,
            dtype=torch.int32,
        )
        indices = torch.arange(
            math.prod(self.quantization_levels),
            device=device,
        ).unsqueeze(-1)
        level_indices = (indices // basis) % levels
        half_width = levels // 2
        codebook = (level_indices - half_width) / half_width
        return levels, basis, codebook

    def bound(self, hidden_states: Tensor, epsilon: float = 1e-3) -> Tensor:
        half_range = (self.levels - 1) * (1 + epsilon) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_range).atanh()
        return (hidden_states + shift).tanh() * half_range - offset

    def codes_from_indices(self, indices: Tensor) -> Tensor:
        """Map flattened codebook indices to normalized scalar codes."""
        return self.codebook.to(indices.device)[indices]

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        half_width = self.levels // 2
        hidden_states = self.bound(hidden_states)
        rounded = hidden_states.round()
        codes = hidden_states + (rounded - hidden_states).detach()
        codes = codes / half_width
        indices = (((codes * half_width) + half_width) * self.basis).sum(dim=-1).to(torch.int32)
        return codes.to(original_dtype), indices


class XCodec2Quantizer(nn.Module):

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.quantizer = XCodec2FiniteScalarQuantization(config)
        dimensions = len(config.quantization_levels)
        self.project_in = nn.Linear(config.quantization_dim, dimensions)
        self.project_out = nn.Linear(dimensions, config.quantization_dim)

    def from_codes(self, indices: Tensor) -> Tensor:
        indices = indices.squeeze(-1).long()
        if indices.numel():
            maximum_index = math.prod(self.quantizer.quantization_levels)
            if bool((indices < 0).any()) or bool((indices >= maximum_index).any()):
                raise ValueError("XCodec2 audio codes are outside the codebook.")
        codes = self.quantizer.codes_from_indices(indices)
        return self.project_out(codes)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        hidden_states = self.project_in(hidden_states)
        original_dtype = hidden_states.dtype
        hidden_states = self.quantizer.bound(hidden_states)
        quantized, indices = self.quantizer(hidden_states)
        quantized = self.project_out(quantized.to(original_dtype))
        return quantized, indices.unsqueeze(-1)


class XCodec2RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = epsilon

    def forward(self, hidden_states: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(dtype)


def _rotate_half(hidden_states: Tensor) -> Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class XCodec2RotaryEmbedding(nn.Module):

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta**(torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        frequencies = torch.matmul(
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1),
            position_ids[:, None, :].float(),
        ).transpose(1, 2)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return (
            embeddings.cos().to(hidden_states.dtype),
            embeddings.sin().to(hidden_states.dtype),
        )


class XCodec2Attention(nn.Module):

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (config.num_attention_heads // config.num_key_value_heads)
        self.scale = config.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    @staticmethod
    def _repeat_key_values(hidden_states: Tensor, repeats: int) -> Tensor:
        if repeats == 1:
            return hidden_states
        batch, heads, length, dimension = hidden_states.shape
        return (
            hidden_states[:, :, None].expand(batch, heads, repeats, length,
                                             dimension).reshape(batch, heads * repeats, length, dimension))

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch,
            length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch,
            length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch,
            length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        cosine, sine = position_embeddings
        cosine = cosine.unsqueeze(2)
        sine = sine.unsqueeze(2)
        query = query * cosine + _rotate_half(query) * sine
        key = key * cosine + _rotate_half(key) * sine
        key = self._repeat_key_values(key, self.num_key_value_groups)
        value = self._repeat_key_values(value, self.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probabilities = torch.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)
        probabilities = F.dropout(
            probabilities,
            p=self.attention_dropout,
            training=self.training,
        )
        output = torch.matmul(probabilities, value)
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(output)


class XCodec2MLP(nn.Module):

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.fc2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(F.silu(self.fc1(hidden_states)))


class XCodec2DecoderLayer(nn.Module):

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.self_attn = XCodec2Attention(config)
        self.mlp = XCodec2MLP(config)
        self.input_layernorm = XCodec2RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.post_attention_layernorm = XCodec2RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class XCodec2ISTFTHead(nn.Module):

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.n_fft + 2)
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.padding = (self.n_fft - self.hop_length) // 2
        self.register_buffer(
            "window",
            torch.hann_window(config.n_fft),
            persistent=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        prediction = self.linear(hidden_states).transpose(1, 2)
        magnitude, phase = prediction.chunk(2, dim=1)
        spectrum = (magnitude.float().exp().clamp(max=1e2) * torch.exp(1j * phase.float()))
        frames = torch.fft.irfft(
            spectrum,
            self.n_fft,
            dim=1,
            norm="backward",
        )
        frames = frames * self.window[None, :, None]
        frame_count = spectrum.shape[-1]
        if frame_count == 0:
            raise ValueError("XCodec2 cannot decode an empty code sequence.")
        output_size = ((frame_count - 1) * self.hop_length + self.n_fft)
        audio = F.fold(
            frames,
            output_size=(1, output_size),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        )[:, 0, 0, self.padding:-self.padding]
        envelope = F.fold(
            self.window.square().expand(1, frame_count, -1).transpose(1, 2),
            output_size=(1, output_size),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        )[0, 0, 0, self.padding:-self.padding]
        audio = audio / envelope.clamp_min(1e-11)
        return audio.unsqueeze(1)


class XCodec2Decoder(nn.Module):

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.quantization_dim, config.hidden_size)
        self.embed = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=7,
            padding=3,
        )
        self.prior_net = nn.ModuleList((XCodec2ResNetBlock(config), XCodec2ResNetBlock(config)))
        self.num_attention_heads = config.num_attention_heads
        self.rotary_emb = XCodec2RotaryEmbedding(config)
        self.layers = nn.ModuleList(XCodec2DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.post_net = nn.ModuleList((XCodec2ResNetBlock(config), XCodec2ResNetBlock(config)))
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.head = XCodec2ISTFTHead(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc(hidden_states)
        hidden_states = self.embed(hidden_states.transpose(1, 2)).transpose(1, 2)
        for layer in self.prior_net:
            hidden_states = layer(hidden_states)
        # The released graph intentionally indexes RoPE by decoder head, not
        # by time. Preserving this unusual detail is required for parity.
        positions = torch.arange(
            self.num_attention_heads,
            device=hidden_states.device,
        ).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, positions)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )
        for layer in self.post_net:
            hidden_states = layer(hidden_states)
        return self.head(self.norm(hidden_states))


class XCodec2SemanticAdapter(nn.Module):

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        channels = config.semantic_model_config.hidden_size
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.conv3 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.conv4 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.relu(self.conv1(hidden_states))
        residual = hidden_states
        hidden_states = self.conv3(F.relu(self.conv2(hidden_states)))
        return self.conv4(hidden_states + residual)


class XCodec2FeatureExtractor(nn.Module):
    """Native batched frontend matching the authors' released processor."""

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.config = config
        self.kaldi_config = KaldiFbankConfig(
            sample_frequency=float(config.sampling_rate),
            frame_length=25.0,
            frame_shift=10.0,
            num_mel_bins=80,
            dither=0.0,
            low_frequency=20.0,
            high_frequency=float(config.sampling_rate // 2),
            preemphasis_coefficient=0.97,
            remove_dc_offset=True,
            use_log_fbank=True,
            use_energy=False,
            snip_edges=True,
            window_type="povey",
        )

    def validate_preprocessor_config(self, values: dict[str, Any]) -> None:
        """Reject frontend metadata that would change checkpoint semantics."""
        if not isinstance(values, dict):
            raise TypeError("XCodec2 preprocessor config must be a mapping.")
        expected = {
            "feature_extractor_type": "Xcodec2FeatureExtractor",
            "feature_size": 80,
            "frame_length": 400,
            "frame_shift": 160,
            "hop_length": self.config.hop_length,
            "num_mel_bins": 80,
            "padding_side": "right",
            "padding_value": 1,
            "return_attention_mask": True,
            "sampling_rate": self.config.sampling_rate,
            "stride": 2,
        }
        mismatches = {
            name: (expected_value, values[name])
            for name, expected_value in expected.items() if name in values and values[name] != expected_value
        }
        if mismatches:
            details = ", ".join(
                f"{name}={actual!r} (expected {expected_value!r})"
                for name, (expected_value, actual) in sorted(mismatches.items()))
            raise ValueError("Unsupported XCodec2 frontend metadata: " + details + ".")

    def forward(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor | None = None,
    ) -> XCodec2Features:
        if not isinstance(waveforms, Tensor):
            raise TypeError("XCodec2 waveforms must be a PyTorch tensor.")
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim == 3 and waveforms.shape[1] == 1:
            waveforms = waveforms[:, 0]
        if waveforms.ndim != 2:
            raise ValueError("XCodec2 waveforms must have shape [batch, samples].")
        if not waveforms.is_floating_point() or waveforms.is_complex():
            raise TypeError("XCodec2 waveforms must use a real floating dtype.")
        if not torch.isfinite(waveforms).all():
            raise ValueError("XCodec2 waveforms contain NaN or infinite values.")
        batch, maximum = waveforms.shape
        if waveform_lengths is None:
            lengths = torch.full(
                (batch, ),
                maximum,
                dtype=torch.long,
                device=waveforms.device,
            )
        else:
            lengths = torch.as_tensor(
                waveform_lengths,
                dtype=torch.long,
                device=waveforms.device,
            )
            if tuple(lengths.shape) != (batch, ):
                raise ValueError("`waveform_lengths` must contain one value per example.")
            if bool((lengths <= 0).any()) or bool((lengths > maximum).any()):
                raise ValueError("XCodec2 waveform lengths must lie inside the batch.")

        padded_lengths = ((lengths + 1 + self.config.hop_length - 1) // self.config.hop_length *
                          self.config.hop_length)
        padded_width = int(padded_lengths.max().item())
        input_values = waveforms.new_zeros((batch, 1, padded_width))
        padding_mask = torch.zeros(
            (batch, padded_width),
            dtype=torch.long,
            device=waveforms.device,
        )
        feature_rows = []
        feature_lengths = []
        for index, length in enumerate(lengths.tolist()):
            copied = int(length)
            input_values[index, 0, :copied] = waveforms[index, :copied]
            # The official processor appends one valid zero before padding.
            padding_mask[index, :copied + 1] = 1
            padded_length = int(padded_lengths[index].item())
            semantic_waveform = F.pad(
                input_values[index, :, :padded_length],
                (
                    self.config.hop_length // 2,
                    self.config.hop_length // 2,
                ),
            )
            features = kaldi_fbank(
                semantic_waveform * (2**15),
                self.kaldi_config,
            )
            if features.shape[0] < 2:
                raise ValueError("XCodec2 audio is too short to normalize semantic frames.")
            features = (features -
                        features.mean(dim=0)) / torch.sqrt(features.var(dim=0, unbiased=True) + 1e-7)
            feature_rows.append(features)
            feature_lengths.append(features.shape[0])

        maximum_frames = max(feature_lengths)
        if maximum_frames % 2:
            maximum_frames += 1
        padded_features = waveforms.new_full(
            (batch, maximum_frames, 80),
            1.0,
        )
        feature_mask = torch.zeros(
            (batch, maximum_frames),
            dtype=torch.long,
            device=waveforms.device,
        )
        for index, features in enumerate(feature_rows):
            padded_features[index, :features.shape[0]] = features
            feature_mask[index, :features.shape[0]] = 1
        input_features = padded_features.reshape(
            batch,
            maximum_frames // 2,
            160,
        )
        input_features_mask = feature_mask.reshape(
            batch,
            maximum_frames // 2,
            2,
        ).amin(dim=-1)
        return XCodec2Features(
            input_values=input_values,
            input_features=input_features,
            padding_mask=padding_mask,
            input_features_mask=input_features_mask,
        )


class XCodec2Model(nn.Module):
    """Complete trainable graph for the official self-contained checkpoint."""

    def __init__(
        self,
        config: XCodec2Config | dict[str, Any],
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        self.config = (config if isinstance(config, XCodec2Config) else XCodec2Config.from_dict(config))
        self.hop_length = self.config.hop_length
        self.feature_extractor = XCodec2FeatureExtractor(self.config)
        self.semantic_encoder = Wav2Vec2BertSemanticModel(self.config.semantic_model_config)
        self.semantic_adapter = XCodec2SemanticAdapter(self.config)
        self.acoustic_encoder = XCodec2Encoder(self.config)
        self.fc_encoder = nn.Linear(
            self.config.quantization_dim,
            self.config.quantization_dim,
        )
        self.quantizer = XCodec2Quantizer(self.config)
        self.acoustic_decoder = XCodec2Decoder(self.config)
        if initialize:
            self.apply(self._initialize_module)
            self._reset_derived_buffers()

    @property
    def sampling_rate(self) -> int:
        return self.config.sampling_rate

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
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
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, XCodec2RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, XCodec2SnakeBeta):
            nn.init.zeros_(module.alpha)
            nn.init.zeros_(module.beta)
        elif isinstance(module, Wav2Vec2BertSemanticModel):
            if module.masked_spec_embed is not None:
                nn.init.uniform_(module.masked_spec_embed)

    def _reset_derived_buffers(self) -> None:
        for module in self.modules():
            if isinstance(module, XCodec2ISTFTHead):
                module.window.copy_(
                    torch.hann_window(
                        module.n_fft,
                        device=module.window.device,
                        dtype=module.window.dtype,
                    ))
            elif isinstance(module, XCodec2FiniteScalarQuantization):
                levels, basis, codebook = module._compute_buffers(device=module.levels.device)
                module.levels.copy_(levels)
                module.basis.copy_(basis)
                module.codebook.copy_(codebook)
            elif isinstance(module, XCodec2DownSample1d):
                module.filter.copy_(
                    kaiser_sinc_filter1d(
                        module.cutoff,
                        module.half_width,
                        module.kernel_size,
                    ).to(module.filter.device))
            elif isinstance(module, XCodec2UpSample1d):
                module.filter.copy_(
                    kaiser_sinc_filter1d(
                        0.5 / module.ratio,
                        0.6 / module.ratio,
                        module.kernel_size,
                    ).to(module.filter.device))

    def encode(
        self,
        input_values: Tensor,
        input_features: Tensor,
        *,
        padding_mask: Tensor | None = None,
        input_features_mask: Tensor | None = None,
        output_latents: bool = False,
    ) -> XCodec2EncoderOutput:
        with torch.no_grad():
            semantic = self.semantic_encoder(
                input_features,
                attention_mask=input_features_mask,
            )
        semantic = self.semantic_adapter(semantic.transpose(1, 2))
        acoustic = self.acoustic_encoder(input_values)
        if semantic.shape[-1] != acoustic.shape[-1]:
            raise RuntimeError(
                "XCodec2 frontend produced misaligned semantic and acoustic "
                f"frames ({semantic.shape[-1]} != {acoustic.shape[-1]}).")
        hidden_states = torch.cat((semantic, acoustic), dim=1)
        hidden_states = self.fc_encoder(hidden_states.transpose(1, 2))
        latents, audio_codes = self.quantizer(hidden_states)
        latents = latents.transpose(1, 2)
        audio_codes = audio_codes.transpose(1, 2)
        code_mask = None
        if padding_mask is not None:
            lengths = padding_mask.sum(dim=-1, keepdim=True)
            token_lengths = lengths // self.hop_length
            positions = torch.arange(
                audio_codes.shape[-1],
                device=padding_mask.device,
            ).view(1, -1)
            code_mask = (positions < token_lengths).to(padding_mask.dtype)
        return XCodec2EncoderOutput(
            audio_codes=audio_codes,
            latents=latents if output_latents else None,
            audio_codes_mask=code_mask,
        )

    def encode_audio(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor | None = None,
    ) -> XCodec2EncoderOutput:
        features = self.feature_extractor(waveforms, waveform_lengths)
        return self.encode(
            features.input_values,
            features.input_features,
            padding_mask=features.padding_mask,
            input_features_mask=features.input_features_mask,
        )

    def encode_code(
        self,
        input_waveform: Tensor,
        sample_rate: int = 16_000,
    ) -> Tensor:
        if sample_rate != self.sampling_rate:
            raise ValueError(
                f"XCodec2 requires {self.sampling_rate} Hz audio; received "
                f"{sample_rate} Hz.")
        return self.encode_audio(input_waveform).audio_codes

    def decode(
        self,
        *,
        audio_codes: Tensor | None = None,
        latents: Tensor | None = None,
    ) -> XCodec2DecoderOutput:
        if (audio_codes is None) == (latents is None):
            raise ValueError("Specify exactly one of `audio_codes` or `latents`.")
        if audio_codes is not None:
            if audio_codes.ndim != 3 or audio_codes.shape[1] != 1:
                raise ValueError("XCodec2 `audio_codes` must have shape [batch, 1, frames].")
            decoded_latents = self.quantizer.from_codes(audio_codes.transpose(1, 2))
        else:
            if latents.ndim != 3:
                raise ValueError("XCodec2 `latents` must have shape [batch, channels, frames].")
            decoded_latents = latents.transpose(1, 2)
        return XCodec2DecoderOutput(audio_values=self.acoustic_decoder(decoded_latents))

    def decode_code(self, audio_codes: Tensor) -> Tensor:
        return self.decode(audio_codes=audio_codes).audio_values

    def forward(
        self,
        input_values: Tensor,
        input_features: Tensor,
        *,
        padding_mask: Tensor | None = None,
        input_features_mask: Tensor | None = None,
        output_latents: bool = False,
    ) -> XCodec2Output:
        length = input_values.shape[-1]
        encoded = self.encode(
            input_values,
            input_features,
            padding_mask=padding_mask,
            input_features_mask=input_features_mask,
            output_latents=True,
        )
        decoded = self.decode(latents=encoded.latents)
        return XCodec2Output(
            audio_values=decoded.audio_values[..., :length],
            audio_codes=encoded.audio_codes,
            latents=encoded.latents if output_latents else None,
            audio_codes_mask=encoded.audio_codes_mask,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        from voicehub.checkpointing import save_safetensors

        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "feature_extractor_type": "Xcodec2FeatureExtractor",
                    "feature_size": 80,
                    "frame_length": 400,
                    "frame_shift": 160,
                    "hop_length": self.hop_length,
                    "num_mel_bins": 80,
                    "padding_side": "right",
                    "padding_value": 1,
                    "return_attention_mask": True,
                    "sampling_rate": self.sampling_rate,
                    "stride": 2,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        save_safetensors(
            self.state_dict(),
            destination / "model.safetensors",
            metadata={
                "format": "pt",
                "architecture": "xcodec2",
                "producer": "voicehub",
            },
        )
        return destination.resolve()

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        device: str | torch.device = "cpu",
        strict: bool = True,
    ) -> XCodec2Model:
        from voicehub.checkpointing import SafeTensorReader
        from voicehub.models.llasa.checkpoint import XCodec2CheckpointAdapter

        root = Path(directory).expanduser().resolve()
        config_path = root / "config.json"
        checkpoint = root / "model.safetensors"
        if not config_path.is_file():
            raise FileNotFoundError(f"XCodec2 config was not found: {config_path}.")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"XCodec2 checkpoint was not found: {checkpoint}.")
        try:
            values = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not parse XCodec2 config: {error}.") from error
        model = cls(XCodec2Config.from_dict(values), initialize=False)
        model.to(device)
        model._reset_derived_buffers()
        with SafeTensorReader(checkpoint) as reader:
            XCodec2CheckpointAdapter.for_model(model).load_streaming(
                model,
                reader,
                values,
                strict=strict,
            )
        return model


__all__ = [
    "Wav2Vec2BertSemanticConfig",
    "XCODEC2_TRANSFORMERS_SOURCE_REVISION",
    "XCodec2Config",
    "XCodec2DecoderOutput",
    "XCodec2EncoderOutput",
    "XCodec2FeatureExtractor",
    "XCodec2Features",
    "XCodec2FiniteScalarQuantization",
    "XCodec2Model",
    "XCodec2Output",
    "kaiser_sinc_filter1d",
]
