"""Native continuous speech encoders used by VibeVoice ASR.

The module mirrors the public VibeVoice-ASR-HF ConvNeXt-1D namespace so
its Safetensors checkpoint can be assigned without tensor renaming.
Convolution cache objects are runtime state and are deliberately absent
from checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.vibevoice.native.configuration import VibeVoiceASRTokenizerConfig


class VibeVoiceASRConvCacheLayer:
    """Left context for one causal convolution."""

    def __init__(self) -> None:
        self.cache: Tensor | None = None
        self.left_pad: int | None = None
        self.in_channels: int | None = None

    def update(
        self,
        hidden_states: Tensor,
        convolution: VibeVoiceASRCausalConv1d,
    ) -> Tensor:
        if self.cache is None:
            self.left_pad = convolution.left_pad
            self.in_channels = convolution.in_channels
            self.cache = hidden_states.new_zeros(
                hidden_states.shape[0],
                convolution.in_channels,
                convolution.left_pad,
            )
        if (hidden_states.shape[0] != self.cache.shape[0] or hidden_states.shape[1] != self.cache.shape[1]):
            raise ValueError("VibeVoice convolution-cache batch geometry changed.")
        left_pad = int(self.left_pad or 0)
        previous = self.cache
        if left_pad:
            combined = torch.cat((previous, hidden_states), dim=-1)
            self.cache = combined[..., -left_pad:].detach()
        return previous


class VibeVoiceASRConvCache:
    """Per-layer streaming state for an ASR speech encoder."""

    def __init__(self) -> None:
        self.layers: dict[str, VibeVoiceASRConvCacheLayer] = {}

    def update(
        self,
        hidden_states: Tensor,
        key: str,
        convolution: VibeVoiceASRCausalConv1d,
    ) -> Tensor:
        layer = self.layers.setdefault(key, VibeVoiceASRConvCacheLayer())
        return torch.cat(
            (layer.update(hidden_states, convolution), hidden_states),
            dim=-1,
        )


class VibeVoiceASRRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, *, epsilon: float, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.variance_epsilon = epsilon

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return normalized.to(input_dtype) * self.weight


class VibeVoiceASRFeedForward(nn.Module):

    def __init__(
        self,
        config: VibeVoiceASRTokenizerConfig,
        hidden_size: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        expanded = config.ffn_expansion * hidden_size
        self.linear1 = nn.Linear(
            hidden_size,
            expanded,
            device=device,
            dtype=dtype,
        )
        self.linear2 = nn.Linear(
            expanded,
            hidden_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.linear2(functional.gelu(self.linear1(hidden_states)))


class VibeVoiceASRCausalConv1d(nn.Module):
    """Causal Conv1d with the exact published stride padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        cache_key: str,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            device=device,
            dtype=dtype,
        )
        self.left_pad = (kernel_size - 1) * dilation - (stride - 1)
        if self.left_pad < 0:
            raise ValueError("Causal convolution has negative left padding.")
        self.cache_key = cache_key
        self.in_channels = in_channels

    def forward(
        self,
        hidden_states: Tensor,
        padding_cache: VibeVoiceASRConvCache | None = None,
    ) -> Tensor:
        if padding_cache is None:
            hidden_states = functional.pad(hidden_states, (self.left_pad, 0))
        else:
            hidden_states = padding_cache.update(
                hidden_states,
                self.cache_key,
                self,
            )
        return self.conv(hidden_states)


class VibeVoiceASRConvNext1dLayer(nn.Module):

    def __init__(
        self,
        config: VibeVoiceASRTokenizerConfig,
        hidden_size: int,
        *,
        layer_index: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(
            torch.full(
                (hidden_size, ),
                config.layer_scale_init_value,
                device=device,
                dtype=dtype,
            ))
        self.ffn_gamma = nn.Parameter(
            torch.full(
                (hidden_size, ),
                config.layer_scale_init_value,
                device=device,
                dtype=dtype,
            ))
        self.norm = VibeVoiceASRRMSNorm(
            hidden_size,
            epsilon=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.ffn_norm = VibeVoiceASRRMSNorm(
            hidden_size,
            epsilon=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.ffn = VibeVoiceASRFeedForward(
            config,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        self.mixer = VibeVoiceASRCausalConv1d(
            hidden_size,
            hidden_size,
            config.kernel_size,
            cache_key=f"convnext_layer_{layer_index}",
            groups=hidden_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        padding_cache: VibeVoiceASRConvCache | None = None,
    ) -> Tensor:
        residual = hidden_states
        normalized = self.norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = residual + (
            self.mixer(normalized, padding_cache=padding_cache) * self.gamma.unsqueeze(-1))
        residual = hidden_states
        normalized = self.ffn_norm(hidden_states.transpose(1, 2))
        feed_forward = self.ffn(normalized).transpose(1, 2)
        return residual + feed_forward * self.ffn_gamma.unsqueeze(-1)


class VibeVoiceASREncoderStem(nn.Module):

    def __init__(
        self,
        config: VibeVoiceASRTokenizerConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.conv = VibeVoiceASRCausalConv1d(
            config.channels,
            config.num_filters,
            config.kernel_size,
            cache_key="encoder_stem",
            device=device,
            dtype=dtype,
        )
        self.stage = nn.ModuleList(
            VibeVoiceASRConvNext1dLayer(
                config,
                config.num_filters,
                layer_index=index,
                device=device,
                dtype=dtype,
            ) for index in range(1, config.depths[0] + 1))

    def forward(
        self,
        hidden_states: Tensor,
        padding_cache: VibeVoiceASRConvCache | None = None,
    ) -> Tensor:
        hidden_states = self.conv(hidden_states, padding_cache=padding_cache)
        for block in self.stage:
            hidden_states = block(hidden_states, padding_cache=padding_cache)
        return hidden_states


class VibeVoiceASREncoderLayer(nn.Module):

    def __init__(
        self,
        config: VibeVoiceASRTokenizerConfig,
        stage_index: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        depth_index = stage_index + 1
        layer_index = sum(depth + 1 for depth in config.depths[:depth_index])
        input_channels = config.num_filters * (2**stage_index)
        output_channels = config.num_filters * (2**depth_index)
        ratio = config.downsampling_ratios[stage_index]
        self.conv = VibeVoiceASRCausalConv1d(
            input_channels,
            output_channels,
            ratio * 2,
            cache_key=f"encoder_layer_{stage_index}",
            stride=ratio,
            device=device,
            dtype=dtype,
        )
        self.stage = nn.ModuleList(
            VibeVoiceASRConvNext1dLayer(
                config,
                output_channels,
                layer_index=layer_index + offset,
                device=device,
                dtype=dtype,
            ) for offset in range(1, config.depths[depth_index] + 1))

    def forward(
        self,
        hidden_states: Tensor,
        padding_cache: VibeVoiceASRConvCache | None = None,
    ) -> Tensor:
        hidden_states = self.conv(hidden_states, padding_cache=padding_cache)
        for block in self.stage:
            hidden_states = block(hidden_states, padding_cache=padding_cache)
        return hidden_states


@dataclass(frozen=True)
class VibeVoiceASREncoderOutput:
    latents: Tensor
    padding_cache: VibeVoiceASRConvCache | None = None


class VibeVoiceASRTokenizerEncoder(nn.Module):
    """Checkpoint-compatible ASR acoustic or semantic speech encoder."""

    def __init__(
        self,
        config: VibeVoiceASRTokenizerConfig | dict[str, Any],
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = VibeVoiceASRTokenizerConfig.from_dict(config)
        if not isinstance(config, VibeVoiceASRTokenizerConfig):
            raise TypeError("VibeVoice ASR encoder requires its tokenizer config.")
        self.config = config
        self.stem = VibeVoiceASREncoderStem(
            config,
            device=device,
            dtype=dtype,
        )
        self.conv_layers = nn.ModuleList(
            VibeVoiceASREncoderLayer(
                config,
                stage_index,
                device=device,
                dtype=dtype,
            ) for stage_index in range(len(config.downsampling_ratios)))
        final_channels = config.num_filters * (2**len(config.downsampling_ratios))
        self.head = VibeVoiceASRCausalConv1d(
            final_channels,
            config.hidden_size,
            config.kernel_size,
            cache_key="encoder_head",
            device=device,
            dtype=dtype,
        )
        if initialize:
            self.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, VibeVoiceASRRMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_values: Tensor,
        *,
        padding_cache: VibeVoiceASRConvCache | None = None,
        use_cache: bool = False,
    ) -> VibeVoiceASREncoderOutput:
        if not isinstance(input_values, Tensor) or input_values.ndim != 3:
            raise ValueError("ASR speech input must have shape [batch, channels, samples].")
        if input_values.shape[1] != self.config.channels:
            raise ValueError("ASR speech input has the wrong channel count.")
        if input_values.shape[0] == 0 or input_values.shape[-1] == 0:
            raise ValueError("ASR speech input cannot be empty.")
        if padding_cache is not None and not use_cache:
            raise ValueError("A convolution cache requires `use_cache=True`.")
        if use_cache and padding_cache is None:
            padding_cache = VibeVoiceASRConvCache()
        hidden_states = self.stem(
            input_values,
            padding_cache=padding_cache,
        )
        for layer in self.conv_layers:
            hidden_states = layer(
                hidden_states,
                padding_cache=padding_cache,
            )
        hidden_states = self.head(
            hidden_states,
            padding_cache=padding_cache,
        )
        return VibeVoiceASREncoderOutput(
            latents=hidden_states.transpose(1, 2),
            padding_cache=padding_cache,
        )


__all__ = [
    "VibeVoiceASRConvCache",
    "VibeVoiceASREncoderOutput",
    "VibeVoiceASRTokenizerEncoder",
]
