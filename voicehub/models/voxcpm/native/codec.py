"""Native VoxCPM2 AudioVAE V2 codec.

The module preserves the upstream ``weight_g``/``weight_v`` namespace so
a one-time, explicitly trusted conversion of the official legacy archive
can be saved as a strict pickle-free Safetensors artifact.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils import weight_norm

from voicehub.kernels.codecs import CodecSnakeKernelOptimizable
from voicehub.models.voxcpm.native.configuration import VoxCPMAudioVAEConfig


def _weighted_conv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def _weighted_transpose_conv1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class _CausalConv1d(nn.Conv1d):

    def __init__(
        self,
        *args,
        padding: int = 0,
        output_padding: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.causal_padding = padding
        self.causal_output_padding = output_padding

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = functional.pad(
            inputs,
            (
                self.causal_padding * 2 - self.causal_output_padding,
                0,
            ),
        )
        return super().forward(inputs)


class _CausalTransposeConv1d(nn.ConvTranspose1d):

    def __init__(
        self,
        *args,
        padding: int = 0,
        output_padding: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.causal_padding = padding
        self.causal_output_padding = output_padding

    def forward(self, inputs: Tensor) -> Tensor:
        result = super().forward(inputs)
        trim = self.causal_padding * 2 - self.causal_output_padding
        return result[..., :-trim] if trim else result


def _weighted_causal_conv1d(*args, **kwargs):
    return weight_norm(_CausalConv1d(*args, **kwargs))


def _weighted_causal_transpose_conv1d(*args, **kwargs):
    return weight_norm(_CausalTransposeConv1d(*args, **kwargs))


class _Snake1d(CodecSnakeKernelOptimizable, nn.Module):

    def __init__(self, channels: int, *, device=None, dtype=None) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, device=device, dtype=dtype))
        self._initialize_codec_kernel_backend()

    def forward(self, inputs: Tensor) -> Tensor:
        return self._codec_snake(inputs, self.alpha)


class _CausalResidualUnit(nn.Module):

    def __init__(
        self,
        dimension: int,
        *,
        dilation: int,
        kernel: int = 7,
        groups: int = 1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        padding = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            _Snake1d(dimension, device=device, dtype=dtype),
            _weighted_causal_conv1d(
                dimension,
                dimension,
                kernel_size=kernel,
                dilation=dilation,
                padding=padding,
                groups=groups,
                device=device,
                dtype=dtype,
            ),
            _Snake1d(dimension, device=device, dtype=dtype),
            _weighted_causal_conv1d(
                dimension,
                dimension,
                kernel_size=1,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.block(inputs)
        if output.shape[-1] != inputs.shape[-1]:
            raise RuntimeError("AudioVAE residual unit changed its causal length.")
        return inputs + output


class _CausalEncoderBlock(nn.Module):

    def __init__(
        self,
        output_dimension: int,
        *,
        input_dimension: int,
        stride: int,
        groups: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _CausalResidualUnit(
                input_dimension,
                dilation=1,
                groups=groups,
                device=device,
                dtype=dtype,
            ),
            _CausalResidualUnit(
                input_dimension,
                dilation=3,
                groups=groups,
                device=device,
                dtype=dtype,
            ),
            _CausalResidualUnit(
                input_dimension,
                dilation=9,
                groups=groups,
                device=device,
                dtype=dtype,
            ),
            _Snake1d(input_dimension, device=device, dtype=dtype),
            _weighted_causal_conv1d(
                input_dimension,
                output_dimension,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class _CausalEncoder(nn.Module):

    def __init__(
        self,
        *,
        model_dimension: int,
        latent_dimension: int,
        strides: tuple[int, ...],
        depthwise: bool,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            _weighted_causal_conv1d(
                1,
                model_dimension,
                kernel_size=7,
                padding=3,
                device=device,
                dtype=dtype,
            )
        ]
        for stride in strides:
            input_dimension = model_dimension
            model_dimension *= 2
            groups = input_dimension if depthwise else 1
            layers.append(
                _CausalEncoderBlock(
                    model_dimension,
                    input_dimension=input_dimension,
                    stride=stride,
                    groups=groups,
                    device=device,
                    dtype=dtype,
                ))
        self.fc_mu = _weighted_causal_conv1d(
            model_dimension,
            latent_dimension,
            kernel_size=3,
            padding=1,
            device=device,
            dtype=dtype,
        )
        self.fc_logvar = _weighted_causal_conv1d(
            model_dimension,
            latent_dimension,
            kernel_size=3,
            padding=1,
            device=device,
            dtype=dtype,
        )
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        hidden = self.block(inputs)
        return {
            "hidden_state": hidden,
            "mu": self.fc_mu(hidden),
            "logvar": self.fc_logvar(hidden),
        }


class _NoiseBlock(nn.Module):

    def __init__(
        self,
        dimension: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.linear = _weighted_causal_conv1d(
            dimension,
            dimension,
            kernel_size=1,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        noise = torch.randn(
            (inputs.shape[0], 1, inputs.shape[2]),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        return inputs + noise * self.linear(inputs)


class _CausalDecoderBlock(nn.Module):

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        *,
        stride: int,
        groups: int,
        use_noise_block: bool,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            _Snake1d(input_dimension, device=device, dtype=dtype),
            _weighted_causal_transpose_conv1d(
                input_dimension,
                output_dimension,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
                device=device,
                dtype=dtype,
            ),
        ]
        if use_noise_block:
            layers.append(_NoiseBlock(
                output_dimension,
                device=device,
                dtype=dtype,
            ))
        layers.extend([
            _CausalResidualUnit(
                output_dimension,
                dilation=1,
                groups=groups,
                device=device,
                dtype=dtype,
            ),
            _CausalResidualUnit(
                output_dimension,
                dilation=3,
                groups=groups,
                device=device,
                dtype=dtype,
            ),
            _CausalResidualUnit(
                output_dimension,
                dilation=9,
                groups=groups,
                device=device,
                dtype=dtype,
            ),
        ])
        self.block = nn.Sequential(*layers)
        self.input_channels = input_dimension

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class _SampleRateCondition(nn.Module):

    def __init__(
        self,
        input_dimension: int,
        *,
        bucket_count: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.scale_embed = nn.Embedding(
            bucket_count,
            input_dimension,
            device=device,
            dtype=dtype,
        )
        self.bias_embed = nn.Embedding(
            bucket_count,
            input_dimension,
            device=device,
            dtype=dtype,
        )
        if torch.device("cpu" if device is None else device).type != "meta":
            nn.init.ones_(self.scale_embed.weight)
            nn.init.zeros_(self.bias_embed.weight)
        self.out_layer = nn.Identity()

    def forward(self, inputs: Tensor, condition: Tensor) -> Tensor:
        return (inputs * self.scale_embed(condition).unsqueeze(-1) + self.bias_embed(condition).unsqueeze(-1))


class _CausalDecoder(nn.Module):

    def __init__(
        self,
        *,
        input_channels: int,
        channels: int,
        rates: tuple[int, ...],
        depthwise: bool,
        use_noise_block: bool,
        sampling_rate_boundaries: tuple[int, ...] | None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if depthwise:
            layers: list[nn.Module] = [
                _weighted_causal_conv1d(
                    input_channels,
                    input_channels,
                    kernel_size=7,
                    padding=3,
                    groups=input_channels,
                    device=device,
                    dtype=dtype,
                ),
                _weighted_causal_conv1d(
                    input_channels,
                    channels,
                    kernel_size=1,
                    device=device,
                    dtype=dtype,
                ),
            ]
        else:
            layers = [
                _weighted_causal_conv1d(
                    input_channels,
                    channels,
                    kernel_size=7,
                    padding=3,
                    device=device,
                    dtype=dtype,
                )
            ]
        output_dimension = channels
        for index, stride in enumerate(rates):
            input_dimension = channels // (2**index)
            output_dimension = channels // (2**(index + 1))
            layers.append(
                _CausalDecoderBlock(
                    input_dimension,
                    output_dimension,
                    stride=stride,
                    groups=output_dimension if depthwise else 1,
                    use_noise_block=use_noise_block,
                    device=device,
                    dtype=dtype,
                ))
        layers.extend([
            _Snake1d(output_dimension, device=device, dtype=dtype),
            _weighted_causal_conv1d(
                output_dimension,
                1,
                kernel_size=7,
                padding=3,
                device=device,
                dtype=dtype,
            ),
            nn.Tanh(),
        ])
        if sampling_rate_boundaries is None:
            self.model = nn.Sequential(*layers)
            self.sr_bin_boundaries = None
            self.sr_cond_model = None
        else:
            self.model = nn.ModuleList(layers)
            self.register_buffer(
                "sr_bin_boundaries",
                torch.tensor(
                    sampling_rate_boundaries,
                    dtype=torch.int32,
                    device=device,
                ),
            )
            bucket_count = len(sampling_rate_boundaries) + 1
            conditions: list[nn.Module | None] = []
            for layer in self.model:
                if isinstance(layer, _CausalDecoderBlock):
                    conditions.append(
                        _SampleRateCondition(
                            layer.input_channels,
                            bucket_count=bucket_count,
                            device=device,
                            dtype=dtype,
                        ))
                else:
                    conditions.append(None)
            self.sr_cond_model = nn.ModuleList(conditions)

    def forward(
        self,
        inputs: Tensor,
        sampling_rate: Tensor | None,
    ) -> Tensor:
        if self.sr_bin_boundaries is None:
            return self.model(inputs)
        if sampling_rate is None:
            raise ValueError("Conditioned AudioVAE decoding requires a sampling rate.")
        condition = torch.bucketize(
            sampling_rate,
            self.sr_bin_boundaries,
        )
        for layer, conditioner in zip(self.model, self.sr_cond_model):
            if conditioner is not None:
                inputs = conditioner(inputs, condition)
            inputs = layer(inputs)
        return inputs


class VoxCPMAudioVAE(nn.Module):
    """Asymmetric 16 kHz encoder / 48 kHz decoder from VoxCPM2."""

    deterministic_codec_targets = ("encode", )

    def __init__(
        self,
        config: VoxCPMAudioVAEConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not isinstance(config, VoxCPMAudioVAEConfig):
            raise TypeError("`config` must be a VoxCPMAudioVAEConfig.")
        self.config = config
        self.encoder = _CausalEncoder(
            model_dimension=config.encoder_dim,
            latent_dimension=config.latent_dim,
            strides=config.encoder_rates,
            depthwise=config.depthwise,
            device=device,
            dtype=dtype,
        )
        self.decoder = _CausalDecoder(
            input_channels=config.latent_dim,
            channels=config.decoder_dim,
            rates=config.decoder_rates,
            depthwise=config.depthwise,
            use_noise_block=config.use_noise_block,
            sampling_rate_boundaries=config.sr_bin_boundaries,
            device=device,
            dtype=dtype,
        )
        self.latent_dim = config.latent_dim
        self.sample_rate = config.sample_rate
        self.out_sample_rate = config.out_sample_rate
        self.hop_length = math.prod(config.encoder_rates)
        self.chunk_size = self.hop_length
        self.decode_chunk_size = math.prod(config.decoder_rates)

    def encode(
        self,
        waveform: Tensor,
        sampling_rate: int | None = None,
    ) -> Tensor:
        if sampling_rate is not None and int(sampling_rate) != self.sample_rate:
            raise ValueError(f"VoxCPM AudioVAE expects {self.sample_rate} Hz input.")
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("AudioVAE waveform must have shape [batch, 1, samples].")
        remainder = waveform.shape[-1] % self.hop_length
        if remainder:
            waveform = functional.pad(
                waveform,
                (0, self.hop_length - remainder),
            )
        return self.encoder(waveform)["mu"]

    def decode(
        self,
        latents: Tensor,
        sampling_rate: int | Tensor | None = None,
    ) -> Tensor:
        if latents.ndim != 3 or latents.shape[1] != self.latent_dim:
            raise ValueError("AudioVAE latents must have shape [batch, latent_dim, frames].")
        condition = sampling_rate
        if self.config.sr_bin_boundaries is not None:
            if condition is None:
                condition = self.out_sample_rate
            if not isinstance(condition, Tensor):
                condition = torch.full(
                    (latents.shape[0], ),
                    int(condition),
                    device=latents.device,
                    dtype=torch.int32,
                )
            else:
                condition = condition.to(
                    device=latents.device,
                    dtype=torch.int32,
                ).reshape(-1)
                if condition.numel() == 1 and latents.shape[0] > 1:
                    condition = condition.expand(latents.shape[0])
        return self.decoder(latents, condition)


__all__ = ["VoxCPMAudioVAE"]
