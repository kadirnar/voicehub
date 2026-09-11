"""VoiceHub-native Higgs Audio V2 tokenizer used by OmniVoice.

The graph follows the Apache-2.0 Transformers implementation at
immutable revision ``aad13b87ed59f2afcfaebc985f403301887a35fc``.  Its
HuBERT encoder is the existing VoiceHub-native implementation; all
remaining codec modules are defined here with the exact published
Safetensors namespace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.kernels.codecs import CodecSnakeKernelOptimizable
from voicehub.models.asr_hubert.native import HubertModel
from voicehub.models.omnivoice.native.configuration import HiggsAcousticConfig, HiggsAudioV2Config
from voicehub.processing.waveform import resample_waveform_kaiser


@dataclass(frozen=True, slots=True)
class HiggsAudioEncoderOutput:
    """Discrete audio codes with shape ``[batch, codebook, frame]``."""

    audio_codes: Tensor


@dataclass(frozen=True, slots=True)
class HiggsAudioDecoderOutput:
    """Decoded mono audio with shape ``[batch, channel, sample]``."""

    audio_values: Tensor


@dataclass(frozen=True, slots=True)
class HiggsAudioOutput:
    audio_codes: Tensor
    audio_values: Tensor


class Snake1d(CodecSnakeKernelOptimizable, nn.Module):
    """Periodic activation used by the published DAC encoder and decoder."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, hidden_size, 1))
        self._initialize_codec_kernel_backend()

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self._codec_snake(hidden_states, self.alpha)


class DacResidualUnit(nn.Module):
    """Snake/Conv residual unit with the official parameter names."""

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
        self.conv2 = nn.Conv1d(dimension, dimension, kernel_size=1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        output = self.conv1(self.snake1(hidden_states))
        output = self.conv2(self.snake2(output))
        padding = (hidden_states.shape[-1] - output.shape[-1]) // 2
        if padding > 0:
            hidden_states = hidden_states[..., padding:-padding]
        return hidden_states + output


class DacEncoderBlock(nn.Module):

    def __init__(
        self,
        config: HiggsAcousticConfig,
        *,
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
        hidden_states = self.snake1(self.res_unit3(hidden_states))
        return self.conv1(hidden_states)


class DacDecoderBlock(nn.Module):

    def __init__(
        self,
        config: HiggsAcousticConfig,
        *,
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

    def __init__(self, config: HiggsAcousticConfig) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            1,
            config.encoder_hidden_size,
            kernel_size=7,
            padding=3,
        )
        self.block = nn.ModuleList(
            DacEncoderBlock(
                config,
                stride=stride,
                stride_index=index + 1,
            ) for index, stride in enumerate(config.downsampling_ratios))
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

    def __init__(self, config: HiggsAcousticConfig) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            config.hidden_size,
            config.decoder_hidden_size,
            kernel_size=7,
            padding=3,
        )
        self.block = nn.ModuleList(
            DacDecoderBlock(
                config,
                stride=stride,
                stride_index=index,
            ) for index, stride in enumerate(config.upsampling_ratios))
        output_dimension = (config.decoder_hidden_size // 2**len(config.upsampling_ratios))
        self.snake1 = Snake1d(output_dimension)
        self.conv2 = nn.Conv1d(
            output_dimension,
            1,
            kernel_size=7,
            padding=3,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv1(hidden_states)
        for block in self.block:
            hidden_states = block(hidden_states)
        return self.conv2(self.snake1(hidden_states))


class HiggsResidualUnit(nn.Module):
    """Residual unit in the semantic feature encoder/decoder."""

    def __init__(
        self,
        config: HiggsAudioV2Config,
        in_channels: int,
        out_channels: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.activation = nn.ELU()
        padding = ((config.unit_kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            config.unit_kernel_size,
            stride=1,
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
        output = self.conv1(self.activation(hidden_states))
        output = self.conv2(self.activation(output))
        return hidden_states + output


class HiggsSemanticEncoderBlock(nn.Module):

    def __init__(
        self,
        config: HiggsAudioV2Config,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.res_units = nn.ModuleList(
            HiggsResidualUnit(
                config,
                in_channels,
                in_channels,
                dilation,
            ) for dilation in config.block_dilations)
        kernel_size = 3 if stride == 1 else 2 * stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=True,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for unit in self.res_units:
            hidden_states = unit(hidden_states)
        return self.conv(hidden_states)


class SemanticEncoder(nn.Module):

    def __init__(self, config: HiggsAudioV2Config) -> None:
        super().__init__()
        hidden_size = config.semantic_hidden_size
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            bias=False,
        )
        blocks: list[nn.Module] = []
        in_channels = hidden_size
        for ratio, stride in zip(config.channel_ratios, config.strides):
            out_channels = int(hidden_size * ratio)
            blocks.append(HiggsSemanticEncoderBlock(
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


class HiggsSemanticDecoderBlock(nn.Module):

    def __init__(
        self,
        config: HiggsAudioV2Config,
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
                stride=1,
                padding=1,
                bias=True,
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
            HiggsResidualUnit(
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

    def __init__(self, config: HiggsAudioV2Config) -> None:
        super().__init__()
        hidden_size = config.semantic_hidden_size
        self.conv1 = nn.Conv1d(
            hidden_size,
            int(hidden_size * config.channel_ratios[0]),
            config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            bias=False,
        )
        blocks: list[nn.Module] = []
        for index, stride in enumerate(config.strides):
            in_channels = int(hidden_size * config.channel_ratios[index])
            out_channels = (
                int(hidden_size * config.channel_ratios[index + 1]) if index +
                1 < len(config.channel_ratios) else hidden_size)
            blocks.append(HiggsSemanticDecoderBlock(
                config,
                in_channels,
                out_channels,
                stride,
            ))
        self.conv_blocks = nn.ModuleList(blocks)
        self.conv2 = nn.Conv1d(
            hidden_size,
            hidden_size,
            config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv1(hidden_states)
        for block in self.conv_blocks:
            hidden_states = block(hidden_states)
        return self.conv2(hidden_states)


class HiggsEuclideanCodebook(nn.Module):

    def __init__(self, config: HiggsAudioV2Config) -> None:
        super().__init__()
        embedding = torch.zeros(config.codebook_size, config.codebook_dim)
        # The published Higgs checkpoint serializes this sentinel as F32.
        # Keeping the dtype exact matters for strict inventory validation.
        self.register_buffer("inited", torch.ones(1))
        self.register_buffer(
            "cluster_size",
            torch.zeros(config.codebook_size),
        )
        self.register_buffer("embed", embedding)
        self.register_buffer("embed_avg", embedding.clone())

    def encode(self, hidden_states: Tensor) -> Tensor:
        shape = hidden_states.shape
        flattened = hidden_states.reshape(-1, shape[-1])
        embedding = self.embed.t()
        distance = -(
            flattened.square().sum(1, keepdim=True) - 2 * flattened @ embedding +
            embedding.square().sum(0, keepdim=True))
        return distance.argmax(dim=-1).view(*shape[:-1])

    def decode(self, indices: Tensor) -> Tensor:
        return functional.embedding(
            indices.to(self.embed.device),
            self.embed,
        )


class HiggsVectorQuantization(nn.Module):

    def __init__(self, config: HiggsAudioV2Config) -> None:
        super().__init__()
        self.codebook = HiggsEuclideanCodebook(config)
        self.project_in = nn.Linear(config.hidden_size, config.codebook_dim)
        self.project_out = nn.Linear(config.codebook_dim, config.hidden_size)

    def encode(self, hidden_states: Tensor) -> Tensor:
        values = self.project_in(hidden_states.permute(0, 2, 1))
        return self.codebook.encode(values)

    def decode(self, indices: Tensor) -> Tensor:
        values = self.project_out(self.codebook.decode(indices))
        return values.permute(0, 2, 1)


class HiggsResidualVectorQuantization(nn.Module):

    def __init__(self, config: HiggsAudioV2Config) -> None:
        super().__init__()
        self.quantizers = nn.ModuleList(HiggsVectorQuantization(config) for _ in range(config.num_quantizers))
        self.frame_rate = config.frame_rate
        self.codebook_size = config.codebook_size
        self.num_quantizers = config.num_quantizers

    def _count(self, bandwidth: float | None) -> int:
        if bandwidth is None:
            return self.num_quantizers
        bandwidth_per_quantizer = (math.log2(self.codebook_size) * self.frame_rate / 1_000)
        return max(
            1,
            min(
                self.num_quantizers,
                math.floor(bandwidth / bandwidth_per_quantizer),
            ),
        )

    def encode(
        self,
        embeddings: Tensor,
        bandwidth: float | None,
    ) -> Tensor:
        residual = embeddings
        indices = []
        for quantizer in self.quantizers[:self._count(bandwidth)]:
            code = quantizer.encode(residual)
            residual = residual - quantizer.decode(code)
            indices.append(code)
        return torch.stack(indices)

    def decode(self, codes: Tensor) -> Tensor:
        if codes.shape[0] > len(self.quantizers):
            raise ValueError("Audio codes contain more quantizers than the codec.")
        output = None
        for index, code in enumerate(codes):
            decoded = self.quantizers[index].decode(code)
            output = decoded if output is None else output + decoded
        if output is None:
            raise ValueError("Audio codes must contain at least one quantizer.")
        return output


def _replace_hubert_positional_weight_norm(model: HubertModel) -> None:
    """Expose the parametrization namespace stored by the Higgs checkpoint."""
    source = model.encoder.pos_conv_embed.conv
    convolution = nn.Conv1d(
        source.channels,
        source.channels,
        kernel_size=source.kernel_size,
        padding=source.padding,
        groups=source.groups,
        bias=True,
        device=source.weight_v.device,
        dtype=source.weight_v.dtype,
    )
    nn.utils.parametrizations.weight_norm(
        convolution,
        name="weight",
        dim=2,
    )
    model.encoder.pos_conv_embed.conv = convolution


class HiggsAudioV2Tokenizer(nn.Module):
    """Exact native codec for ``eustlb/higgs-audio-v2-tokenizer``."""

    def __init__(
        self,
        config: HiggsAudioV2Config,
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(config, HiggsAudioV2Config):
            raise TypeError("`config` must be a HiggsAudioV2Config.")
        self.config = config
        self.pad = config.hop_length // 2
        self.acoustic_encoder = DacEncoder(config.acoustic_model_config)
        self.acoustic_decoder = DacDecoder(config.acoustic_model_config)
        self.encoder_semantic = SemanticEncoder(config)
        self.decoder_semantic = SemanticDecoder(config)
        self.semantic_model = HubertModel(config.semantic_model_config)
        _replace_hubert_positional_weight_norm(self.semantic_model)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc1 = nn.Linear(
            config.hidden_size,
            config.semantic_model_config.hidden_size,
        )
        self.fc2 = nn.Linear(
            config.hidden_size,
            config.acoustic_model_config.hidden_size,
        )
        self.quantizer = HiggsResidualVectorQuantization(config)
        if initialize:
            self._initialize_weights()
        for parameter in self.semantic_model.parameters():
            parameter.requires_grad_(False)
        self.semantic_model.eval()

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def frame_rate(self) -> int:
        return self.config.frame_rate

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def train(self, mode: bool = True) -> HiggsAudioV2Tokenizer:
        super().train(mode)
        # The upstream tokenizer always extracts frozen HuBERT targets in eval
        # mode, including while raw-waveform TTS batches are prepared.
        self.semantic_model.eval()
        return self

    def _initialize_weights(self) -> None:
        initializer_range = self.config.initializer_range

        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=initializer_range,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.weight is not None:
                    nn.init.ones_(module.weight)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    bound = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.ConvTranspose1d):
                module.reset_parameters()
            elif isinstance(module, Snake1d):
                nn.init.ones_(module.alpha)
            elif isinstance(module, HiggsEuclideanCodebook):
                module.inited.fill_(True)
                module.cluster_size.zero_()
                module.embed.zero_()
                module.embed_avg.zero_()

        self.apply(initialize)

    def _semantic_features(self, input_values: Tensor) -> Tensor:
        values = input_values[:, 0, :]
        if self.config.sample_rate != self.config.semantic_sample_rate:
            values = resample_waveform_kaiser(
                values,
                self.config.sample_rate,
                self.config.semantic_sample_rate,
            )
        values = functional.pad(values, (160, 160))
        with torch.no_grad():
            outputs = self.semantic_model(
                values,
                output_hidden_states=True,
            )
        if outputs.hidden_states is None:
            raise RuntimeError("Native HuBERT did not return requested hidden states.")
        stacked = torch.stack(
            [state.to(values.device) for state in outputs.hidden_states],
            dim=1,
        )
        features = stacked.mean(dim=1)
        if self.config.semantic_downsample_factor > 1:
            features = features[
                :,
                ::self.config.semantic_downsample_factor,
                :,
            ]
        return features

    def _validate_input(self, input_values: Tensor) -> None:
        if not isinstance(input_values, Tensor):
            raise TypeError("`input_values` must be a PyTorch tensor.")
        if input_values.ndim != 3 or input_values.shape[1] != 1:
            raise ValueError("Higgs audio input must have shape [batch, 1, samples].")
        if not input_values.is_floating_point():
            raise TypeError("Higgs audio input must use a floating-point dtype.")
        if input_values.shape[0] == 0 or input_values.shape[2] == 0:
            raise ValueError("Higgs audio input cannot have an empty axis.")
        if not torch.isfinite(input_values).all():
            raise ValueError("Higgs audio input contains NaN or infinity.")

    @torch.no_grad()
    def encode(
        self,
        input_values: Tensor,
        bandwidth: float | None = None,
    ) -> HiggsAudioEncoderOutput:
        self._validate_input(input_values)
        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[-1]
        elif float(bandwidth) not in self.config.target_bandwidths:
            raise ValueError(
                f"Unsupported Higgs bandwidth {bandwidth!r}; expected one of "
                f"{self.config.target_bandwidths!r}.")

        semantic_input = self._semantic_features(input_values).detach()
        semantic = self.encoder_semantic(semantic_input.transpose(1, 2))
        acoustic = self.acoustic_encoder(input_values)
        if acoustic.shape[2] != semantic.shape[2]:
            acoustic = self.acoustic_encoder(functional.pad(input_values, (self.pad, self.pad)))
        if acoustic.shape[2] != semantic.shape[2]:
            length = min(acoustic.shape[2], semantic.shape[2])
            acoustic = acoustic[..., :length]
            semantic = semantic[..., :length]
        embeddings = torch.cat(
            [acoustic.to(semantic.device), semantic],
            dim=1,
        )
        embeddings = self.fc(embeddings.transpose(1, 2)).transpose(1, 2)
        codes = self.quantizer.encode(embeddings, float(bandwidth))
        return HiggsAudioEncoderOutput(codes.transpose(0, 1))

    @torch.no_grad()
    def decode(self, audio_codes: Tensor) -> HiggsAudioDecoderOutput:
        if not isinstance(audio_codes, Tensor):
            raise TypeError("`audio_codes` must be a PyTorch tensor.")
        if audio_codes.ndim != 3 or audio_codes.shape[1] == 0:
            raise ValueError("Higgs audio codes must have shape [batch, codebook, frame].")
        if (audio_codes.dtype == torch.bool or audio_codes.is_floating_point() or audio_codes.is_complex()):
            raise TypeError("Higgs audio codes must use an integer dtype.")
        if ((audio_codes < 0).any() or (audio_codes >= self.config.codebook_size).any()):
            raise ValueError("Higgs audio code is outside the codebook.")
        quantized = self.quantizer.decode(audio_codes.transpose(0, 1))
        acoustic = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
        return HiggsAudioDecoderOutput(self.acoustic_decoder(acoustic))

    @torch.no_grad()
    def forward(
        self,
        input_values: Tensor,
        *,
        audio_codes: Tensor | None = None,
        bandwidth: float | None = None,
    ) -> HiggsAudioOutput:
        length = input_values.shape[-1]
        if audio_codes is None:
            audio_codes = self.encode(
                input_values,
                bandwidth,
            ).audio_codes
        audio_values = self.decode(audio_codes).audio_values[..., :length]
        return HiggsAudioOutput(
            audio_codes=audio_codes,
            audio_values=audio_values,
        )


__all__ = [
    "HiggsAudioDecoderOutput",
    "HiggsAudioEncoderOutput",
    "HiggsAudioOutput",
    "HiggsAudioV2Tokenizer",
]
