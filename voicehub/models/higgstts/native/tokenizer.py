"""PyTorch-only Higgs Audio v2 semantic/acoustic tokenizer.

This module implements the released 25 Hz tokenizer graph inside
VoiceHub.  It combines a frozen HuBERT semantic encoder, a DAC acoustic
path, semantic alignment blocks, and eight residual vector quantizers.
No Transformers, torchaudio, NumPy, or repository runtime is imported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.kernels.codecs import CodecSnakeKernelOptimizable
from voicehub.models.asr_hubert.native import HubertModel
from voicehub.models.higgstts.native.tokenizer_configuration import (
    HiggsAcousticCodecConfig,
    HiggsAudioV2TokenizerConfig,
)
from voicehub.processing.waveform import resample_waveform


class Snake1d(CodecSnakeKernelOptimizable, nn.Module):
    """Periodic activation used by the released DAC graph."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, hidden_size, 1))
        self._initialize_codec_kernel_backend()

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self._codec_snake(hidden_states, self.alpha)


class DacResidualUnit(nn.Module):
    """Two-convolution residual unit with source-compatible names."""

    def __init__(self, dimension: int, dilation: int) -> None:
        super().__init__()
        padding = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension)
        self.conv1 = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=7,
            dilation=dilation,
            padding=padding,
        )
        self.snake2 = Snake1d(dimension)
        self.conv2 = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=1,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.conv1(self.snake1(hidden_states))
        hidden_states = self.conv2(self.snake2(hidden_states))
        difference = residual.shape[-1] - hidden_states.shape[-1]
        if difference < 0 or difference % 2:
            raise RuntimeError("The Higgs DAC residual convolution changed length "
                               "asymmetrically.")
        padding = difference // 2
        if padding:
            residual = residual[..., padding:-padding]
        return residual + hidden_states


class DacEncoderBlock(nn.Module):
    """Residual stack followed by one strided convolution."""

    def __init__(
        self,
        config: HiggsAcousticCodecConfig,
        stride: int,
        stride_index: int,
    ) -> None:
        super().__init__()
        dimension = config.encoder_hidden_size * 2**stride_index
        input_dimension = dimension // 2
        self.res_unit1 = DacResidualUnit(input_dimension, dilation=1)
        self.res_unit2 = DacResidualUnit(input_dimension, dilation=3)
        self.res_unit3 = DacResidualUnit(input_dimension, dilation=9)
        self.snake1 = Snake1d(input_dimension)
        self.conv1 = nn.Conv1d(
            input_dimension,
            dimension,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.res_unit1(hidden_states)
        hidden_states = self.res_unit2(hidden_states)
        hidden_states = self.res_unit3(hidden_states)
        return self.conv1(self.snake1(hidden_states))


class DacDecoderBlock(nn.Module):
    """One transposed convolution followed by a residual stack."""

    def __init__(
        self,
        config: HiggsAcousticCodecConfig,
        stride: int,
        stride_index: int,
    ) -> None:
        super().__init__()
        input_dimension = config.decoder_hidden_size // 2**stride_index
        output_dimension = (config.decoder_hidden_size // 2**(stride_index + 1))
        self.snake1 = Snake1d(input_dimension)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dimension,
            output_dimension,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
            output_padding=stride % 2,
        )
        self.res_unit1 = DacResidualUnit(output_dimension, dilation=1)
        self.res_unit2 = DacResidualUnit(output_dimension, dilation=3)
        self.res_unit3 = DacResidualUnit(output_dimension, dilation=9)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv_t1(self.snake1(hidden_states))
        hidden_states = self.res_unit1(hidden_states)
        hidden_states = self.res_unit2(hidden_states)
        return self.res_unit3(hidden_states)


class DacEncoder(nn.Module):
    """Source-compatible acoustic encoder embedded by Higgs."""

    def __init__(self, config: HiggsAcousticCodecConfig) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            1,
            config.encoder_hidden_size,
            kernel_size=7,
            padding=3,
        )
        self.block = nn.ModuleList(
            DacEncoderBlock(config, stride, index + 1)
            for index, stride in enumerate(config.downsampling_ratios))
        dimension = (config.encoder_hidden_size * 2**len(config.downsampling_ratios))
        self.snake1 = Snake1d(dimension)
        self.conv2 = nn.Conv1d(
            dimension,
            config.hidden_size,
            kernel_size=3,
            padding=1,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv1(hidden_states)
        for block in self.block:
            hidden_states = block(hidden_states)
        return self.conv2(self.snake1(hidden_states))


class DacDecoder(nn.Module):
    """Source-compatible acoustic decoder embedded by Higgs."""

    def __init__(self, config: HiggsAcousticCodecConfig) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            config.hidden_size,
            config.decoder_hidden_size,
            kernel_size=7,
            padding=3,
        )
        self.block = nn.ModuleList(
            DacDecoderBlock(config, stride, index) for index, stride in enumerate(config.upsampling_ratios))
        output_dimension = (config.decoder_hidden_size // 2**len(config.upsampling_ratios))
        self.snake1 = Snake1d(output_dimension)
        self.conv2 = nn.Conv1d(
            output_dimension,
            1,
            kernel_size=7,
            padding=3,
        )
        # The Higgs checkpoint deliberately removes DAC's final tanh.
        self.tanh = nn.Identity()

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv1(hidden_states)
        for block in self.block:
            hidden_states = block(hidden_states)
        hidden_states = self.conv2(self.snake1(hidden_states))
        return self.tanh(hidden_states)


class HiggsAudioV2TokenizerResidualUnit(nn.Module):
    """Semantic residual unit used before and after residual VQ."""

    def __init__(
        self,
        config: HiggsAudioV2TokenizerConfig,
        in_channels: int,
        out_channels: int,
        dilation: int,
    ) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("Higgs semantic residual units require equal channels.")
        padding = ((config.unit_kernel_size - 1) // 2) * dilation
        self.activation = nn.ELU()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            config.unit_kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.conv1(self.activation(hidden_states))
        hidden_states = self.conv2(self.activation(hidden_states))
        return residual + hidden_states


class HiggsAudioV2TokenizerSemanticEncoderBlock(nn.Module):
    """Semantic residual stack and optional downsampling convolution."""

    def __init__(
        self,
        config: HiggsAudioV2TokenizerConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.res_units = nn.ModuleList(
            HiggsAudioV2TokenizerResidualUnit(
                config,
                in_channels,
                in_channels,
                dilation,
            ) for dilation in config.block_dilations)
        kernel = 3 if stride == 1 else 2 * stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=(kernel - 1) // 2,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for unit in self.res_units:
            hidden_states = unit(hidden_states)
        return self.conv(hidden_states)


class SemanticEncoder(nn.Module):
    """Align averaged HuBERT states to acoustic codec frames."""

    def __init__(self, config: HiggsAudioV2TokenizerConfig) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            config.semantic_hidden_size,
            config.semantic_hidden_size,
            config.kernel_size,
            padding=config.kernel_size // 2,
            bias=False,
        )
        blocks = []
        in_channels = config.semantic_hidden_size
        for ratio, stride in zip(config.channel_ratios, config.strides):
            out_channels = int(config.semantic_hidden_size * ratio)
            blocks.append(
                HiggsAudioV2TokenizerSemanticEncoderBlock(
                    config,
                    in_channels,
                    out_channels,
                    stride,
                ))
            in_channels = out_channels
        self.conv_blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        for block in self.conv_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class SemanticDecoderBlock(nn.Module):
    """Optional semantic upsampling followed by residual refinement."""

    def __init__(
        self,
        config: HiggsAudioV2TokenizerConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()
        if stride == 1:
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )
        else:
            self.conv = nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=(stride + 1) // 2,
                output_padding=1 if stride % 2 else 0,
                bias=False,
            )
        self.res_units = nn.ModuleList(
            HiggsAudioV2TokenizerResidualUnit(
                config,
                out_channels,
                out_channels,
                dilation,
            ) for dilation in config.block_dilations)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        for unit in self.res_units:
            hidden_states = unit(hidden_states)
        return hidden_states


class SemanticDecoder(nn.Module):
    """Decode the semantic half of a quantized representation."""

    def __init__(self, config: HiggsAudioV2TokenizerConfig) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            config.semantic_hidden_size,
            int(config.semantic_hidden_size * config.channel_ratios[0]),
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
            bias=False,
        )
        blocks = []
        for index, stride in enumerate(config.strides):
            in_channels = int(config.semantic_hidden_size * config.channel_ratios[index])
            if index + 1 < len(config.channel_ratios):
                out_channels = int(config.semantic_hidden_size * config.channel_ratios[index + 1])
            else:
                out_channels = config.semantic_hidden_size
            blocks.append(SemanticDecoderBlock(
                config,
                in_channels,
                out_channels,
                stride,
            ))
        self.conv_blocks = nn.ModuleList(blocks)
        self.conv2 = nn.Conv1d(
            config.semantic_hidden_size,
            config.semantic_hidden_size,
            config.kernel_size,
            padding=config.kernel_size // 2,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv1(hidden_states)
        for block in self.conv_blocks:
            hidden_states = block(hidden_states)
        return self.conv2(hidden_states)


class HiggsAudioV2TokenizerEuclideanCodebook(nn.Module):
    """Non-trainable EMA codebook serialized by the official artifact."""

    def __init__(self, config: HiggsAudioV2TokenizerConfig) -> None:
        super().__init__()
        embeddings = torch.zeros(
            config.codebook_size,
            config.codebook_dim,
        )
        self.codebook_size = config.codebook_size
        # The upstream checkpoint intentionally serializes this flag as F32.
        self.register_buffer("inited", torch.tensor([1.0]))
        self.register_buffer(
            "cluster_size",
            torch.zeros(config.codebook_size),
        )
        self.register_buffer("embed", embeddings)
        self.register_buffer("embed_avg", embeddings.clone())

    def encode(self, hidden_states: Tensor) -> Tensor:
        shape = hidden_states.shape
        flattened = hidden_states.reshape(-1, shape[-1])
        embeddings = self.embed.t()
        distances = -(
            flattened.pow(2).sum(1, keepdim=True) - 2 * flattened @ embeddings +
            embeddings.pow(2).sum(0, keepdim=True))
        return distances.argmax(dim=-1).view(*shape[:-1])

    def decode(self, indices: Tensor) -> Tensor:
        return functional.embedding(
            indices.to(device=self.embed.device),
            self.embed,
        )


class HiggsAudioV2TokenizerVectorQuantization(nn.Module):
    """One projected residual vector quantizer."""

    def __init__(self, config: HiggsAudioV2TokenizerConfig) -> None:
        super().__init__()
        self.codebook = HiggsAudioV2TokenizerEuclideanCodebook(config)
        self.project_in = nn.Linear(
            config.hidden_size,
            config.codebook_dim,
        )
        self.project_out = nn.Linear(
            config.codebook_dim,
            config.hidden_size,
        )

    def encode(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.project_in(hidden_states.permute(0, 2, 1))
        return self.codebook.encode(hidden_states)

    def decode(self, indices: Tensor) -> Tensor:
        hidden_states = self.codebook.decode(indices)
        return self.project_out(hidden_states).permute(0, 2, 1)


class HiggsAudioV2TokenizerResidualVectorQuantization(nn.Module):
    """Bandwidth-aware residual vector quantizer."""

    def __init__(self, config: HiggsAudioV2TokenizerConfig) -> None:
        super().__init__()
        self.quantizers = nn.ModuleList(
            HiggsAudioV2TokenizerVectorQuantization(config) for _ in range(config.num_quantizers))
        self.frame_rate = config.frame_rate
        self.codebook_size = config.codebook_size
        self.num_quantizers = config.num_quantizers

    @property
    def bandwidth_per_quantizer(self) -> float:
        return math.log2(self.codebook_size) * self.frame_rate / 1_000

    def num_quantizers_for_bandwidth(
        self,
        bandwidth: float | None,
    ) -> int:
        if bandwidth is None or bandwidth <= 0.0:
            return self.num_quantizers
        return min(
            self.num_quantizers,
            max(1, math.floor(bandwidth / self.bandwidth_per_quantizer)),
        )

    def encode(
        self,
        embeddings: Tensor,
        bandwidth: float | None = None,
    ) -> Tensor:
        residual = embeddings
        indices = []
        count = self.num_quantizers_for_bandwidth(bandwidth)
        for quantizer in self.quantizers[:count]:
            current = quantizer.encode(residual)
            residual = residual - quantizer.decode(current)
            indices.append(current)
        return torch.stack(indices)

    def decode(self, codes: Tensor) -> Tensor:
        if codes.shape[0] > len(self.quantizers):
            raise ValueError("Audio codes contain more quantizers than the tokenizer.")
        quantized = None
        for index, current in enumerate(codes):
            decoded = self.quantizers[index].decode(current)
            quantized = (decoded if quantized is None else quantized + decoded)
        if quantized is None:
            raise ValueError("At least one Higgs audio quantizer is required.")
        return quantized


@dataclass(frozen=True)
class HiggsAudioV2TokenizerOutput:
    audio_codes: Tensor | None = None
    audio_values: Tensor | None = None


@dataclass(frozen=True)
class HiggsAudioV2TokenizerEncoderOutput:
    audio_codes: Tensor


@dataclass(frozen=True)
class HiggsAudioV2TokenizerDecoderOutput:
    audio_values: Tensor


class HiggsAudioV2TokenizerModel(nn.Module):
    """Complete native semantic/acoustic tokenizer."""

    def __init__(
        self,
        config: HiggsAudioV2TokenizerConfig | dict[str, Any],
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        self.config = HiggsAudioV2TokenizerConfig.coerce(config)
        self.pad = self.config.hop_length // 2
        self.acoustic_encoder = DacEncoder(self.config.acoustic_model_config)
        self.acoustic_decoder = DacDecoder(self.config.acoustic_model_config)
        self.encoder_semantic = SemanticEncoder(self.config)
        self.decoder_semantic = SemanticDecoder(self.config)
        self.semantic_model = HubertModel(self.config.semantic_model_config)
        self.fc = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc1 = nn.Linear(
            self.config.hidden_size,
            self.config.semantic_hidden_size,
        )
        self.fc2 = nn.Linear(
            self.config.hidden_size,
            self.config.acoustic_model_config.hidden_size,
        )
        self.quantizer = (HiggsAudioV2TokenizerResidualVectorQuantization(self.config))
        if initialize:
            self._initialize_weights()
        self.freeze_semantic_model()

    def _initialize_weights(self) -> None:

        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config.initializer_range,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    fan_in = (module.in_channels * module.kernel_size[0] / module.groups)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.ConvTranspose1d):
                module.reset_parameters()
            elif isinstance(module, Snake1d):
                nn.init.ones_(module.alpha)
            elif isinstance(
                    module,
                    HiggsAudioV2TokenizerEuclideanCodebook,
            ):
                module.inited.fill_(True)
                module.cluster_size.zero_()
                module.embed.zero_()
                module.embed_avg.zero_()

        self.apply(initialize)
        # The embedded DAC has a source-specific truncated-normal exception.
        for module in (
                *self.acoustic_encoder.modules(),
                *self.acoustic_decoder.modules(),
        ):
            if isinstance(module, nn.Conv1d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def freeze_semantic_model(self) -> None:
        self.semantic_model.requires_grad_(False)
        self.semantic_model.eval()

    def freeze(self) -> None:
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def _validate_waveform(input_values: Tensor) -> None:
        if not isinstance(input_values, Tensor):
            raise TypeError("`input_values` must be a PyTorch tensor.")
        if input_values.ndim != 3:
            raise ValueError("`input_values` must have shape [batch, channels, samples].")
        if input_values.shape[0] < 1 or input_values.shape[1] != 1:
            raise ValueError("Higgs audio must contain one or more mono waves.")
        if input_values.shape[-1] < 1:
            raise ValueError("Higgs audio cannot be empty.")
        if not input_values.is_floating_point():
            raise TypeError("Higgs audio must use a floating-point dtype.")
        if not torch.isfinite(input_values).all():
            raise ValueError("Higgs audio cannot contain NaN or infinity.")

    def _extract_semantic_features(
        self,
        input_values: Tensor,
    ) -> Tensor:
        if self.config.sample_rate != self.config.semantic_sample_rate:
            resampled = [
                resample_waveform(
                    waveform[0],
                    self.config.sample_rate,
                    self.config.semantic_sample_rate,
                ) for waveform in input_values
            ]
            lengths = {waveform.shape[-1] for waveform in resampled}
            if len(lengths) != 1:
                raise ValueError("Batched Higgs waveforms must have equal lengths.")
            semantic_input = torch.stack(resampled)
        else:
            semantic_input = input_values[:, 0]
        semantic_padding = self.config.downsample_factor // 2
        semantic_input = functional.pad(
            semantic_input,
            (semantic_padding, semantic_padding),
        )
        self.semantic_model.eval()
        with torch.no_grad():
            output = self.semantic_model(
                semantic_input,
                output_hidden_states=True,
            )
        if output.hidden_states is None:
            raise RuntimeError("Native HuBERT omitted requested hidden states.")
        stacked = torch.stack(
            [hidden.to(device=semantic_input.device) for hidden in output.hidden_states],
            dim=1,
        )
        hidden_states = stacked.mean(dim=1)
        factor = self.config.semantic_downsample_factor
        if factor > 1:
            hidden_states = hidden_states[:, ::factor]
        return hidden_states

    def _acoustic_output_length(self, input_length: int) -> int:
        length = input_length
        for stride in self.config.acoustic_model_config.downsampling_ratios:
            kernel = 2 * stride
            padding = math.ceil(stride / 2)
            length = ((length + 2 * padding - (kernel - 1) - 1) // stride + 1)
        return length

    def encode(
        self,
        input_values: Tensor,
        bandwidth: float | None = None,
    ) -> HiggsAudioV2TokenizerEncoderOutput:
        self._validate_waveform(input_values)
        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[-1]
        elif bandwidth not in self.config.target_bandwidths:
            raise ValueError(
                f"Unsupported Higgs bandwidth {bandwidth!r}; expected one "
                f"of {self.config.target_bandwidths!r}.")
        semantic_input = self._extract_semantic_features(input_values).detach()
        semantic = self.encoder_semantic(semantic_input.transpose(1, 2))
        if self._acoustic_output_length(input_values.shape[-1]) != (semantic.shape[-1]):
            acoustic_input = functional.pad(
                input_values,
                (self.pad, self.pad),
            )
        else:
            acoustic_input = input_values
        acoustic = self.acoustic_encoder(acoustic_input)
        if acoustic.shape[-1] != semantic.shape[-1]:
            raise RuntimeError(
                "Higgs semantic and acoustic frame counts disagree after "
                f"alignment ({semantic.shape[-1]} != {acoustic.shape[-1]}).")
        embeddings = torch.cat(
            [acoustic.to(device=semantic.device), semantic],
            dim=1,
        )
        embeddings = self.fc(embeddings.transpose(1, 2)).transpose(1, 2)
        codes = self.quantizer.encode(
            embeddings,
            bandwidth,
        ).transpose(0, 1)
        return HiggsAudioV2TokenizerEncoderOutput(codes)

    def decode(
        self,
        audio_codes: Tensor,
    ) -> HiggsAudioV2TokenizerDecoderOutput:
        if not isinstance(audio_codes, Tensor) or audio_codes.ndim != 3:
            raise ValueError("`audio_codes` must have shape "
                             "[batch, quantizers, frames].")
        if (audio_codes.dtype == torch.bool or audio_codes.is_floating_point() or audio_codes.is_complex()):
            raise TypeError("Higgs audio codes must use an integer dtype.")
        if not 1 <= audio_codes.shape[1] <= self.config.num_quantizers:
            raise ValueError("Invalid number of Higgs audio quantizers.")
        if audio_codes.numel() and (int(audio_codes.min()) < 0 or
                                    int(audio_codes.max()) >= self.config.codebook_size):
            raise ValueError("A Higgs audio code is outside the codebook.")
        quantized = self.quantizer.decode(audio_codes.transpose(0, 1))
        acoustic = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
        return HiggsAudioV2TokenizerDecoderOutput(self.acoustic_decoder(acoustic))

    def forward(
        self,
        input_values: Tensor,
        *,
        audio_codes: Tensor | None = None,
        bandwidth: float | None = None,
    ) -> HiggsAudioV2TokenizerOutput:
        self._validate_waveform(input_values)
        original_length = input_values.shape[-1]
        if audio_codes is None:
            audio_codes = self.encode(
                input_values,
                bandwidth,
            ).audio_codes
        audio_values = self.decode(audio_codes).audio_values
        return HiggsAudioV2TokenizerOutput(
            audio_codes=audio_codes,
            audio_values=audio_values[..., :original_length],
        )


__all__ = [
    "HiggsAudioV2TokenizerDecoderOutput",
    "HiggsAudioV2TokenizerEncoderOutput",
    "HiggsAudioV2TokenizerModel",
    "HiggsAudioV2TokenizerOutput",
]
