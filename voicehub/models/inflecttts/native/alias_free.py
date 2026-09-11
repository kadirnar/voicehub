"""Lightweight alias-free waveform blocks derived from NVIDIA BigVGAN.

BigVGAN and alias-free-torch are MIT/Apache-2.0 licensed. The
implementation is kept local so Inflect can train without BigVGAN's
optional CUDA extension.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from voicehub.models.inflecttts.native.commons import get_padding, init_weights


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21)**0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    values = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    values /= values.sum()
    return values.view(1, 1, kernel_size)


class UpSample1d(nn.Module):

    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.kernel_size = kernel_size
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size))

    def forward(self, x):
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)
        return x[..., self.pad_left:-self.pad_right]


class DownSample1d(nn.Module):

    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size))

    def forward(self, x):
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.expand(channels, -1, -1), stride=self.ratio, groups=channels)


class SnakeBeta(nn.Module):

    def __init__(self, channels: int, logscale: bool = True):
        super().__init__()
        initial = torch.zeros(channels) if logscale else torch.ones(channels)
        self.alpha = nn.Parameter(initial.clone())
        self.beta = nn.Parameter(initial.clone())
        self.logscale = logscale

    def forward(self, x):
        alpha = self.alpha.view(1, -1, 1)
        beta = self.beta.view(1, -1, 1)
        if self.logscale:
            alpha = alpha.exp()
            beta = beta.exp()
        return x + torch.sin(x * alpha).square() / (beta + 1e-9)


class AliasFreeActivation1d(nn.Module):

    def __init__(self, activation: nn.Module):
        super().__init__()
        self.upsample = UpSample1d()
        self.act = activation
        self.downsample = DownSample1d()

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


class AliasFreeResBlock1(nn.Module):
    """Shape-compatible VITS ResBlock1 with filtered SnakeBeta activations."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), logscale=True):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in dilation
        ])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.activations = nn.ModuleList(
            [AliasFreeActivation1d(SnakeBeta(channels, logscale=logscale)) for _ in range(2 * len(dilation))])

    def forward(self, x, x_mask=None):
        first = self.activations[::2]
        second = self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, first, second):
            residual = conv2(act2(conv1(act1(x))))
            x = x + residual
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)
