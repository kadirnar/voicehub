"""VoiceHub-native ModifiedDAC codec used by Fish Speech S2.

The graph follows the published Fish configuration: a causal DAC
encoder/decoder, transformer refinement at the 1,024-channel bottleneck,
one 4,096-entry semantic quantizer, and nine 1,024-entry residual
quantizers. Only PyTorch and VoiceHub-owned neural building blocks are
used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from voicehub.components.audio.codecs.dac.model.base import CodecMixin
from voicehub.components.audio.codecs.dac.nn.layers import Snake1d
from voicehub.components.audio.codecs.dac.nn.quantize import ResidualVectorQuantize
from voicehub.models.fishtts.native.configuration import FishCodecConfig


def _extra_padding(
    length: int,
    *,
    kernel_size: int,
    stride: int,
    total_padding: int,
) -> int:
    frames = (length - kernel_size + total_padding) / stride + 1
    ideal = ((math.ceil(frames) - 1) * stride + kernel_size - total_padding)
    return ideal - length


class FishCausalConv1d(nn.Module):
    """Left-padded convolution with source-compatible parameter names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        normalized: bool = False,
    ) -> None:
        super().__init__()
        convolution = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.conv = weight_norm(convolution) if normalized else convolution
        self.stride = stride
        self.effective_kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.effective_kernel_size - stride

    def forward(self, values: Tensor) -> Tensor:
        extra = _extra_padding(
            values.shape[-1],
            kernel_size=self.effective_kernel_size,
            stride=self.stride,
            total_padding=self.padding,
        )
        values = F.pad(values, (self.padding, extra))
        return self.conv(values).contiguous()


class FishCausalConvTranspose1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        stride: int = 1,
        normalized: bool = False,
    ) -> None:
        super().__init__()
        convolution = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.conv = weight_norm(convolution) if normalized else convolution
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, values: Tensor) -> Tensor:
        values = self.conv(values)
        padding = self.kernel_size - self.stride
        right = math.ceil(padding)
        left = padding - right
        stop = values.shape[-1] - right
        return values[..., left:stop].contiguous()


class FishConvNeXtBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        *,
        layer_scale: float = 1e-6,
        expansion: float = 4.0,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.dwconv = FishCausalConv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
        )
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        intermediate = int(expansion * hidden_size)
        self.pwconv1 = nn.Linear(hidden_size, intermediate)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate, hidden_size)
        self.gamma = nn.Parameter(layer_scale * torch.ones(hidden_size))

    def forward(
        self,
        values: Tensor,
        *,
        apply_residual: bool = True,
    ) -> Tensor:
        residual = values
        values = self.dwconv(values).transpose(1, 2)
        values = self.pwconv2(self.act(self.pwconv1(self.norm(values))))
        values = (values * self.gamma).transpose(1, 2)
        return residual + values if apply_residual else values


@dataclass(frozen=True, slots=True)
class FishCodecTransformerConfig:
    hidden_size: int = 1_024
    intermediate_size: int = 3_072
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    max_position_embeddings: int = 8_192
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.1
    attention_dropout: float = 0.1
    window_size: int | None = 128


class FishCodecRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, values: Tensor) -> Tensor:
        result = values.float()
        result = result * torch.rsqrt(result.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return result.to(values.dtype) * self.weight


class FishCodecLayerScale(nn.Module):

    def __init__(self, hidden_size: int, initial_value: float = 1e-2) -> None:
        super().__init__()
        self.gamma = nn.Parameter(initial_value * torch.ones(hidden_size))

    def forward(self, values: Tensor) -> Tensor:
        return values * self.gamma


def _codec_rotary(
    values: Tensor,
    positions: Tensor,
    *,
    base: float,
) -> Tensor:
    dimension = values.shape[-1]
    inverse = 1.0 / (
        base**(torch.arange(
            0,
            dimension,
            2,
            dtype=torch.float32,
            device=values.device,
        ) / dimension))
    angles = positions.float().unsqueeze(-1) * inverse
    cosine = angles.cos().view(1, positions.numel(), 1, -1)
    sine = angles.sin().view(1, positions.numel(), 1, -1)
    pairs = values.float().reshape(*values.shape[:-1], -1, 2)
    output = torch.stack(
        (
            pairs[..., 0] * cosine - pairs[..., 1] * sine,
            pairs[..., 1] * cosine + pairs[..., 0] * sine,
        ),
        dim=-1,
    )
    return output.flatten(-2).to(values.dtype)


class FishCodecAttention(nn.Module):

    def __init__(self, config: FishCodecTransformerConfig) -> None:
        super().__init__()
        self.config = config
        total_width = (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim
        self.wqkv = nn.Linear(config.hidden_size, total_width, bias=False)
        self.wo = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.kv_cache = None

    def forward(
        self,
        values: Tensor,
        positions: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        batch, sequence_length, _ = values.shape
        query_width = self.config.num_attention_heads * self.config.head_dim
        key_value_width = (self.config.num_key_value_heads * self.config.head_dim)
        query, key, value = self.wqkv(values).split(
            (query_width, key_value_width, key_value_width),
            dim=-1,
        )
        query = query.view(
            batch,
            sequence_length,
            self.config.num_attention_heads,
            self.config.head_dim,
        )
        key = key.view(
            batch,
            sequence_length,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        value = value.view(
            batch,
            sequence_length,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        query = _codec_rotary(
            query,
            positions,
            base=self.config.rope_theta,
        ).transpose(1, 2)
        key = _codec_rotary(
            key,
            positions,
            base=self.config.rope_theta,
        ).transpose(1, 2)
        value = value.transpose(1, 2)
        repeat = (self.config.num_attention_heads // self.config.num_key_value_heads)
        if repeat != 1:
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=(self.config.attention_dropout if self.training else 0.0),
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch,
            sequence_length,
            query_width,
        )
        return self.wo(attended)


class FishCodecFeedForward(nn.Module):

    def __init__(self, config: FishCodecTransformerConfig) -> None:
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
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values: Tensor) -> Tensor:
        return self.w2(self.dropout(F.silu(self.w1(values)) * self.w3(values)))


class FishCodecTransformerBlock(nn.Module):

    def __init__(self, config: FishCodecTransformerConfig) -> None:
        super().__init__()
        self.attention = FishCodecAttention(config)
        self.feed_forward = FishCodecFeedForward(config)
        self.ffn_norm = FishCodecRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.attention_norm = FishCodecRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.attention_layer_scale = FishCodecLayerScale(config.hidden_size)
        self.ffn_layer_scale = FishCodecLayerScale(config.hidden_size)

    def forward(
        self,
        values: Tensor,
        positions: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        values = values + self.attention_layer_scale(
            self.attention(
                self.attention_norm(values),
                positions,
                attention_mask,
            ))
        return values + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(values)))


class FishWindowTransformer(nn.Module):

    def __init__(
        self,
        config: FishCodecTransformerConfig,
        *,
        input_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.window_size = config.window_size
        self.causal = True
        self.channels_first = True
        self.look_ahead_conv = nn.Identity()
        self.input_proj = (
            nn.Linear(input_dim, config.hidden_size) if input_dim != config.hidden_size else nn.Identity())
        self.layers = nn.ModuleList(
            FishCodecTransformerBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = FishCodecRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.output_proj = (
            nn.Linear(config.hidden_size, input_dim) if input_dim != config.hidden_size else nn.Identity())
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.use_kv_cache = False

    def _mask(self, length: int, device: torch.device) -> Tensor:
        row = torch.arange(length, device=device).view(-1, 1)
        column = torch.arange(length, device=device).view(1, -1)
        mask = column <= row
        if self.window_size is not None:
            mask &= column >= (row - self.window_size + 1).clamp_min(0)
        return mask.view(1, 1, length, length)

    def forward(
        self,
        values: Tensor,
        lengths: Tensor | None = None,
    ) -> Tensor:
        del lengths
        values = values.transpose(1, 2)
        values = self.look_ahead_conv(self.input_proj(values))
        positions = torch.arange(
            values.shape[1],
            device=values.device,
            dtype=torch.long,
        )
        mask = self._mask(values.shape[1], values.device)
        for layer in self.layers:
            values = layer(values, positions, mask)
        values = self.output_proj(self.norm(values))
        return values.transpose(1, 2)


class FishResidualUnit(nn.Module):

    def __init__(self, hidden_size: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(hidden_size),
            FishCausalConv1d(
                hidden_size,
                hidden_size,
                7,
                dilation=dilation,
                normalized=True,
            ),
            Snake1d(hidden_size),
            FishCausalConv1d(
                hidden_size,
                hidden_size,
                1,
                normalized=True,
            ),
        )
        self.causal = True

    def forward(self, values: Tensor) -> Tensor:
        result = self.block(values)
        difference = values.shape[-1] - result.shape[-1]
        if difference > 0:
            values = values[..., :-difference]
        return values + result


def _bottleneck_transformer_config(
    *,
    hidden_size: int,
    layers: int,
    window_size: int | None,
) -> FishCodecTransformerConfig:
    return FishCodecTransformerConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 3,
        num_hidden_layers=layers,
        num_attention_heads=hidden_size // 64,
        num_key_value_heads=hidden_size // 64,
        head_dim=64,
        max_position_embeddings=8_192,
        window_size=window_size,
    )


class FishEncoderBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        stride: int,
        *,
        transformer_layers: int,
    ) -> None:
        super().__init__()
        transformer: nn.Module = nn.Identity()
        if transformer_layers:
            transformer = FishWindowTransformer(
                _bottleneck_transformer_config(
                    hidden_size=hidden_size,
                    layers=transformer_layers,
                    window_size=512,
                ),
                input_dim=hidden_size,
            )
        self.block = nn.Sequential(
            FishResidualUnit(hidden_size // 2, 1),
            FishResidualUnit(hidden_size // 2, 3),
            FishResidualUnit(hidden_size // 2, 9),
            Snake1d(hidden_size // 2),
            FishCausalConv1d(
                hidden_size // 2,
                hidden_size,
                2 * stride,
                stride=stride,
                normalized=True,
            ),
            transformer,
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.block(values)


class FishEncoder(nn.Module):

    def __init__(self, config: FishCodecConfig) -> None:
        super().__init__()
        hidden_size = config.encoder_dim
        blocks: list[nn.Module] = [FishCausalConv1d(
            1,
            hidden_size,
            7,
            normalized=True,
        )]
        for stride, layers in zip(
                config.encoder_rates,
                config.encoder_transformer_layers,
        ):
            hidden_size *= 2
            blocks.append(FishEncoderBlock(
                hidden_size,
                stride,
                transformer_layers=layers,
            ))
        blocks.extend((
            Snake1d(hidden_size),
            FishCausalConv1d(
                hidden_size,
                config.latent_dim,
                3,
                normalized=True,
            ),
        ))
        self.block = nn.Sequential(*blocks)
        self.enc_dim = hidden_size

    def forward(self, values: Tensor) -> Tensor:
        return self.block(values)


class FishDecoderBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
    ) -> None:
        super().__init__()
        # The released implementation constructs but intentionally does not
        # register the configured decoder transformer.
        self.block = nn.Sequential(
            Snake1d(input_dim),
            FishCausalConvTranspose1d(
                input_dim,
                output_dim,
                2 * stride,
                stride=stride,
                normalized=True,
            ),
            FishResidualUnit(output_dim, 1),
            FishResidualUnit(output_dim, 3),
            FishResidualUnit(output_dim, 9),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.block(values)


class FishDecoder(nn.Module):

    def __init__(self, config: FishCodecConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            FishCausalConv1d(
                config.latent_dim,
                config.decoder_dim,
                7,
                normalized=True,
            )
        ]
        for index, stride in enumerate(config.decoder_rates):
            input_dim = config.decoder_dim // 2**index
            output_dim = config.decoder_dim // 2**(index + 1)
            layers.append(FishDecoderBlock(input_dim, output_dim, stride))
        layers.extend((
            Snake1d(output_dim),
            FishCausalConv1d(
                output_dim,
                1,
                7,
                normalized=True,
            ),
            nn.Tanh(),
        ))
        self.model = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        return self.model(values)


@dataclass(slots=True)
class FishVQOutput:
    quantized: Tensor
    codes: Tensor
    latents: Tensor
    commitment_loss: Tensor
    codebook_loss: Tensor


class FishDownsampleResidualVectorQuantize(nn.Module):

    def __init__(self, config: FishCodecConfig) -> None:
        super().__init__()
        hidden_size = config.latent_dim
        self.semantic_quantizer = ResidualVectorQuantize(
            input_dim=hidden_size,
            n_codebooks=1,
            codebook_size=config.semantic_codebook_size,
            codebook_dim=config.codebook_dim,
            quantizer_dropout=0.0,
        )
        self.quantizer = ResidualVectorQuantize(
            input_dim=hidden_size,
            n_codebooks=config.residual_codebooks,
            codebook_size=config.residual_codebook_size,
            codebook_dim=config.codebook_dim,
            quantizer_dropout=config.quantizer_dropout,
        )
        self.downsample_factor = config.downsample_factors
        self.downsample_dims = tuple(hidden_size for _ in config.downsample_factors)
        downsample: list[nn.Module] = []
        for factor in config.downsample_factors:
            downsample.append(
                nn.Sequential(
                    FishCausalConv1d(
                        hidden_size,
                        hidden_size,
                        factor,
                        stride=factor,
                    ),
                    FishConvNeXtBlock(hidden_size),
                ))
        self.downsample = nn.Sequential(*downsample)
        upsample: list[nn.Module] = []
        for factor in reversed(config.downsample_factors):
            upsample.append(
                nn.Sequential(
                    FishCausalConvTranspose1d(
                        hidden_size,
                        hidden_size,
                        factor,
                        stride=factor,
                    ),
                    FishConvNeXtBlock(hidden_size),
                ))
        self.upsample = nn.Sequential(*upsample)
        # The source initializes convolutional down/up-sampling modules before
        # attaching the two transformer refiners.
        self.apply(self._initialize_weights)
        transformer = FishCodecTransformerConfig(
            hidden_size=config.transformer_hidden_size,
            intermediate_size=config.transformer_intermediate_size,
            num_hidden_layers=config.transformer_layers,
            num_attention_heads=config.transformer_heads,
            num_key_value_heads=config.transformer_heads,
            head_dim=(config.transformer_hidden_size // config.transformer_heads),
            window_size=config.transformer_window_size,
        )
        self.pre_module = FishWindowTransformer(
            transformer,
            input_dim=hidden_size,
        )
        self.post_module = FishWindowTransformer(
            transformer,
            input_dim=hidden_size,
        )
        self.semantic_predictor_module = nn.Identity()

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        values: Tensor,
        n_quantizers: int | None = None,
        **unused: Any,
    ) -> FishVQOutput:
        del unused
        original_length = values.shape[-1]
        values = self.pre_module(self.downsample(values))
        (
            semantic,
            semantic_codes,
            semantic_latents,
            semantic_commitment,
            semantic_codebook,
        ) = self.semantic_quantizer(values)
        (
            residual,
            residual_codes,
            residual_latents,
            residual_commitment,
            residual_codebook,
        ) = self.quantizer(
            values - semantic,
            n_quantizers=n_quantizers,
        )
        quantized = self.upsample(self.post_module(semantic + residual))
        difference = original_length - quantized.shape[-1]
        if difference > 0:
            quantized = F.pad(quantized, (difference, 0))
        elif difference < 0:
            quantized = quantized[..., -original_length:]
        return FishVQOutput(
            quantized=quantized,
            codes=torch.cat((semantic_codes, residual_codes), dim=1),
            latents=torch.cat(
                (semantic_latents, residual_latents),
                dim=1,
            ),
            commitment_loss=(semantic_commitment + residual_commitment),
            codebook_loss=semantic_codebook + residual_codebook,
        )

    def decode(self, indices: Tensor) -> Tensor:
        indices = indices.clone()
        indices[:, 0].clamp_(max=self.semantic_quantizer.codebook_size - 1)
        indices[:, 1:].clamp_(max=self.quantizer.codebook_size - 1)
        semantic = self.semantic_quantizer.from_codes(indices[:, :1])[0]
        residual = self.quantizer.from_codes(indices[:, 1:])[0]
        return self.upsample(self.post_module(semantic + residual))


class FishModifiedDAC(nn.Module, CodecMixin):
    """Published 44.1 kHz ModifiedDAC executable graph."""

    def __init__(
        self,
        config: FishCodecConfig | None = None,
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        self.config = FishCodecConfig() if config is None else config
        if not isinstance(self.config, FishCodecConfig):
            raise TypeError("`config` must be a FishCodecConfig.")
        self.encoder_dim = self.config.encoder_dim
        self.encoder_rates = list(self.config.encoder_rates)
        self.decoder_dim = self.config.decoder_dim
        self.decoder_rates = list(self.config.decoder_rates)
        self.sample_rate = self.config.sample_rate
        self.latent_dim = self.config.latent_dim
        self.hop_length = math.prod(self.config.encoder_rates)
        self.frame_length = self.config.hop_length
        self.encoder = FishEncoder(self.config)
        self.quantizer = FishDownsampleResidualVectorQuantize(self.config)
        self.decoder = FishDecoder(self.config)
        if initialize:
            self.apply(self._initialize_convolutions)
        self.delay = self.get_delay()

    @staticmethod
    def _initialize_convolutions(module: nn.Module) -> None:
        if isinstance(module, nn.Conv1d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def preprocess(
        self,
        audio_values: Tensor,
        sample_rate: int | None,
    ) -> Tensor:
        if sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError(f"Fish codec requires {self.sample_rate} Hz audio.")
        right = (
            math.ceil(audio_values.shape[-1] / self.hop_length) * self.hop_length - audio_values.shape[-1])
        return F.pad(audio_values, (0, right))

    def encode(
        self,
        audio_values: Tensor,
        audio_lengths: Tensor | None = None,
        n_quantizers: int | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        if audio_values.ndim == 2:
            audio_values = audio_values.unsqueeze(1)
        if audio_values.ndim != 3 or audio_values.shape[1] != 1:
            raise ValueError("Fish codec audio must have shape [batch, 1, time].")
        length = audio_values.shape[-1]
        right = math.ceil(length / self.frame_length) * self.frame_length - length
        audio_values = F.pad(audio_values, (0, right))
        if audio_lengths is None:
            audio_lengths = torch.full(
                (audio_values.shape[0], ),
                length + right,
                device=audio_values.device,
                dtype=torch.long,
            )
        if (audio_lengths.ndim != 1 or audio_lengths.shape[0] != audio_values.shape[0]):
            raise ValueError("Fish codec audio lengths must have shape [batch].")
        encoded = self.quantizer(
            self.encoder(audio_values),
            n_quantizers=n_quantizers,
            **kwargs,
        )
        code_lengths = torch.ceil(audio_lengths / self.frame_length).long()
        return encoded.codes, code_lengths

    def from_indices(self, indices: Tensor) -> Tensor:
        if (indices.ndim != 3 or indices.shape[1] != self.config.num_codebooks):
            raise ValueError(
                "Fish codec indices must have shape "
                f"[batch, {self.config.num_codebooks}, time].")
        return self.decoder(self.quantizer.decode(indices.long()))

    def decode(self, quantized: Tensor) -> Tensor:
        return self.decoder(quantized)

    def forward(
        self,
        audio_values: Tensor,
        *,
        sample_rate: int | None = None,
        n_quantizers: int | None = None,
    ) -> tuple[Tensor, FishVQOutput]:
        original_length = audio_values.shape[-1]
        audio_values = self.preprocess(audio_values, sample_rate)
        if audio_values.ndim == 2:
            audio_values = audio_values.unsqueeze(1)
        encoded = self.quantizer(
            self.encoder(audio_values),
            n_quantizers=n_quantizers,
        )
        decoded = self.decoder(encoded.quantized)
        return decoded[..., :original_length], encoded

    def save_pretrained(self, directory: str | Any) -> Any:
        from voicehub.models.fishtts.native.checkpoint import save_fish_codec_pretrained

        return save_fish_codec_pretrained(self, directory)


# Source-compatible public spelling.
DAC = FishModifiedDAC

__all__ = [
    "DAC",
    "FishCodecTransformerConfig",
    "FishDownsampleResidualVectorQuantize",
    "FishModifiedDAC",
    "FishVQOutput",
    "FishWindowTransformer",
]
