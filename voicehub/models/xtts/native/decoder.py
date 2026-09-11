"""XTTS v2 HiFi-GAN decoder and speaker encoder without torchaudio."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from voicehub.models.xtts.native.audio import MelSpectrogram


def _padding(kernel: int, dilation: int) -> int:
    return (kernel * dilation - dilation) // 2


class ResBlock1(nn.Module):

    def __init__(self, channels: int, kernel_size: int, dilation) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=item,
                    padding=_padding(kernel_size, item),
                )) for item in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=1,
                    padding=_padding(kernel_size, 1),
                )) for _item in dilation
        ])

    def forward(self, value: Tensor) -> Tensor:
        for first, second in zip(self.convs1, self.convs2):
            residual = second(F.leaky_relu(first(F.leaky_relu(value, 0.1)), 0.1))
            value = value + residual
        return value

    def remove_weight_norm(self) -> None:
        for layer in (*self.convs1, *self.convs2):
            remove_parametrizations(layer, "weight")


class HifiganGenerator(nn.Module):

    def __init__(
        self,
        in_channels: int = 1_024,
        cond_channels: int = 512,
        cond_in_each_up_layer: bool = True,
    ) -> None:
        super().__init__()
        initial = 512
        factors = (8, 8, 2, 2)
        kernels = (16, 16, 4, 4)
        self.inference_padding = 0
        self.num_kernels = 3
        self.num_upsamples = len(factors)
        self.cond_in_each_up_layer = cond_in_each_up_layer
        self.conv_pre = weight_norm(nn.Conv1d(in_channels, initial, 7, padding=3))
        remove_parametrizations(self.conv_pre, "weight")
        self.ups = nn.ModuleList([
            weight_norm(
                nn.ConvTranspose1d(
                    initial // (2**index),
                    initial // (2**(index + 1)),
                    kernel,
                    factor,
                    padding=(kernel - factor) // 2,
                )) for index, (factor, kernel) in enumerate(zip(factors, kernels))
        ])
        self.resblocks = nn.ModuleList([
            ResBlock1(initial // (2**(index + 1)), kernel, (1, 3, 5)) for index in range(len(factors))
            for kernel in (3, 7, 11)
        ])
        channels = initial // (2**len(factors))
        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, padding=3, bias=False))
        remove_parametrizations(self.conv_post, "weight")
        self.cond_layer = nn.Conv1d(cond_channels, initial, 1)
        self.conds = nn.ModuleList(
            [nn.Conv1d(cond_channels, initial // (2**(index + 1)), 1) for index in range(len(factors))])

    def forward(self, value: Tensor, g: Tensor | None = None) -> Tensor:
        value = self.conv_pre(value)
        if g is not None:
            value = value + self.cond_layer(g)
        for index, upsample in enumerate(self.ups):
            value = upsample(F.leaky_relu(value, 0.1))
            if g is not None and self.cond_in_each_up_layer:
                value = value + self.conds[index](g)
            blocks = [
                self.resblocks[index * self.num_kernels + offset](value)
                for offset in range(self.num_kernels)
            ]
            value = torch.stack(blocks).mean(dim=0)
        return torch.tanh(self.conv_post(F.leaky_relu(value)))


class SELayer(nn.Module):

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, value: Tensor) -> Tensor:
        weight = self.fc(self.avg_pool(value).flatten(1))[:, :, None, None]
        return value * weight


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, value: Tensor) -> Tensor:
        residual = value if self.downsample is None else self.downsample(value)
        value = self.bn1(self.relu(self.conv1(value)))
        value = self.se(self.bn2(self.conv2(value)))
        return self.relu(value + residual)


class PreEmphasis(nn.Module):

    def __init__(self, coefficient: float = 0.97) -> None:
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer(
            "filter",
            torch.tensor([-coefficient, 1.0])[None, None],
        )

    def forward(self, value: Tensor) -> Tensor:
        value = F.pad(value[:, None], (1, 0), "reflect")
        return F.conv1d(value, self.filter).squeeze(1)


class ResNetSpeakerEncoder(nn.Module):

    def __init__(self, input_dim: int = 64, proj_dim: int = 512) -> None:
        super().__init__()
        self.encoder_type = "ASP"
        self.input_dim = input_dim
        self.log_input = True
        self.use_torch_spec = True
        self.proj_dim = proj_dim
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.inplanes = 32
        self.layer1 = self._layer(32, 3)
        self.layer2 = self._layer(64, 4, stride=(2, 2))
        self.layer3 = self._layer(128, 6, stride=(2, 2))
        self.layer4 = self._layer(256, 3, stride=(2, 2))
        self.instancenorm = nn.InstanceNorm1d(input_dim)
        self.torch_spec = nn.Sequential(
            PreEmphasis(0.97),
            MelSpectrogram(
                sample_rate=16_000,
                n_fft=512,
                win_length=400,
                hop_length=160,
                n_mels=64,
                hamming=True,
                slaney_norm=False,
            ),
        )
        output_map = input_dim // 8
        flattened = 256 * output_map
        self.attention = nn.Sequential(
            nn.Conv1d(flattened, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, flattened, 1),
            nn.Softmax(dim=2),
        )
        self.fc = nn.Linear(flattened * 2, proj_dim)

    def _layer(self, planes: int, count: int, stride=1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        blocks = [SEBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        blocks.extend(SEBasicBlock(planes, planes) for _ in range(1, count))
        return nn.Sequential(*blocks)

    def forward(self, value: Tensor, l2_norm: bool = False) -> Tensor:
        if value.ndim == 3:
            value = value.squeeze(1)
        value = self.torch_spec(value)
        value = (value + 1e-6).log()
        value = self.instancenorm(value).unsqueeze(1)
        value = self.bn1(self.relu(self.conv1(value)))
        value = self.layer4(self.layer3(self.layer2(self.layer1(value))))
        value = value.reshape(value.shape[0], -1, value.shape[-1])
        weight = self.attention(value)
        mean = torch.sum(value * weight, dim=2)
        deviation = torch.sqrt((torch.sum(value.square() * weight, dim=2) - mean.square()).clamp_min(1e-5), )
        value = self.fc(torch.cat((mean, deviation), dim=1))
        return F.normalize(value, p=2, dim=1) if l2_norm else value


class HifiDecoder(nn.Module):

    def __init__(
        self,
        *,
        input_sample_rate: int = 22_050,
        output_sample_rate: int = 24_000,
        output_hop_length: int = 256,
        ar_mel_length_compression: int = 1_024,
        decoder_input_dim: int = 1_024,
        d_vector_dim: int = 512,
        cond_d_vector_in_each_upsampling_layer: bool = True,
    ) -> None:
        super().__init__()
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.output_hop_length = output_hop_length
        self.ar_mel_length_compression = ar_mel_length_compression
        self.waveform_decoder = HifiganGenerator(
            decoder_input_dim,
            d_vector_dim,
            cond_d_vector_in_each_upsampling_layer,
        )
        self.speaker_encoder = ResNetSpeakerEncoder()

    def forward(self, latents: Tensor, g: Tensor | None = None) -> Tensor:
        value = F.interpolate(
            latents.transpose(1, 2),
            scale_factor=self.ar_mel_length_compression / self.output_hop_length,
            mode="linear",
        )
        if self.output_sample_rate != self.input_sample_rate:
            value = F.interpolate(
                value,
                scale_factor=self.output_sample_rate / self.input_sample_rate,
                mode="linear",
            )
        return self.waveform_decoder(value, g=g)


__all__ = ["HifiDecoder", "HifiganGenerator", "ResNetSpeakerEncoder"]
