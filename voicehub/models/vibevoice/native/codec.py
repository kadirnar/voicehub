"""Native VibeVoice 24 kHz continuous speech codec.

This is the codec graph used by the Microsoft 1.5B and realtime TTS
checkpoints. Its nested module names intentionally match the immutable
Safetensors namespaces, including the historical convolution wrappers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.vibevoice.native.configuration import VibeVoiceLegacyTokenizerConfig


def _extra_padding(
    length: int,
    *,
    kernel_size: int,
    stride: int,
    padding_total: int,
) -> int:
    frames = (length - kernel_size + padding_total) / stride + 1
    ideal = (math.ceil(frames) - 1) * stride + kernel_size - padding_total
    return max(0, ideal - length)


class VibeVoiceCodecCache:
    """Per-convolution state used only by explicit streaming calls."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, int], Tensor] = {}

    def get(self, layer_id: str, sample_indices: Tensor) -> Tensor | None:
        values: list[Tensor] = []
        for index in sample_indices.detach().cpu().tolist():
            value = self._values.get((layer_id, int(index)))
            if value is None:
                return None
            values.append(value)
        maximum = max((value.shape[-1] for value in values), default=0)
        padded = [functional.pad(value, (maximum - value.shape[-1], 0)) for value in values]
        return torch.stack(padded)

    def set(
        self,
        layer_id: str,
        sample_indices: Tensor,
        states: Tensor,
    ) -> None:
        if states.shape[0] != sample_indices.numel():
            raise ValueError("Codec cache state and sample-index batches disagree.")
        for offset, index in enumerate(sample_indices.detach().cpu().tolist()):
            self._values[(layer_id, int(index))] = states[offset].detach()

    def clear(self) -> None:
        self._values.clear()


class VibeVoiceCodecRMSNorm(nn.Module):

    def __init__(
        self,
        dimension: int,
        *,
        epsilon: float,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.epsilon = epsilon
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dimension, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.epsilon)
        normalized = normalized.to(input_dtype)
        return normalized if self.weight is None else normalized * self.weight


class VibeVoiceConvRMSNorm(VibeVoiceCodecRMSNorm):

    def forward(self, hidden_states: Tensor) -> Tensor:
        transposed = hidden_states.transpose(1, 2)
        return super().forward(transposed).transpose(1, 2)


class VibeVoiceNormConv1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        *,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
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
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.norm = nn.Identity()

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.norm(self.conv(hidden_states))


class VibeVoiceNormConvTranspose1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        *,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.convtr = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.norm = nn.Identity()

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.norm(self.convtr(hidden_states))


class VibeVoiceSConv1d(nn.Module):
    """Historical padded Conv1d wrapper preserved for checkpoint names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = True,
        pad_mode: str = "constant",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.conv = VibeVoiceNormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.causal = causal
        self.pad_mode = pad_mode
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.context_size = (kernel_size - 1) * dilation - (stride - 1)
        self.padding_total = self.context_size
        self._layer_id: str | None = None

    @property
    def layer_id(self) -> str:
        if self._layer_id is None:
            self._layer_id = f"sconv1d_{id(self)}"
        return self._layer_id

    def _non_streaming(self, hidden_states: Tensor) -> Tensor:
        extra = _extra_padding(
            hidden_states.shape[-1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding_total=self.padding_total,
        )
        if self.causal:
            padding = (self.padding_total, extra)
        else:
            right = self.padding_total // 2
            padding = (
                self.padding_total - right,
                right + extra,
            )
        hidden_states = functional.pad(
            hidden_states,
            padding,
            mode=self.pad_mode,
        )
        return self.conv(hidden_states)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cache: VibeVoiceCodecCache | None = None,
        sample_indices: Tensor | None = None,
        use_cache: bool = False,
        is_final_chunk: bool = False,
    ) -> Tensor:
        if not use_cache:
            return self._non_streaming(hidden_states)
        if not self.causal:
            raise ValueError("Streaming codec convolution must be causal.")
        if cache is None or sample_indices is None:
            raise ValueError("Streaming codec convolution requires cache and indices.")
        previous = cache.get(self.layer_id, sample_indices)
        if previous is None:
            previous = hidden_states.new_zeros(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.context_size,
            )
        combined = torch.cat((previous, hidden_states), dim=-1)
        if is_final_chunk:
            extra = _extra_padding(
                combined.shape[-1],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding_total=self.padding_total,
            )
            combined = functional.pad(combined, (0, extra))
        if self.context_size:
            cache.set(
                self.layer_id,
                sample_indices,
                combined[..., -self.context_size:],
            )
        return self.conv(combined)


class VibeVoiceSConvTranspose1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        *,
        causal: bool = True,
        trim_right_ratio: float = 1.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.convtr = VibeVoiceNormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.causal = causal
        self.trim_right_ratio = float(trim_right_ratio)
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.padding_total = kernel_size - stride
        self.context_size = kernel_size - 1
        self._layer_id: str | None = None
        if not 0.0 <= self.trim_right_ratio <= 1.0:
            raise ValueError("Transposed-convolution trim ratio must be in [0, 1].")

    @property
    def layer_id(self) -> str:
        if self._layer_id is None:
            self._layer_id = f"sconvtr1d_{id(self)}"
        return self._layer_id

    def _trim(self, hidden_states: Tensor) -> Tensor:
        if self.causal:
            right = math.ceil(self.padding_total * self.trim_right_ratio)
        else:
            right = self.padding_total // 2
        left = self.padding_total - right
        end = hidden_states.shape[-1] - right
        return hidden_states[..., left:end] if left or right else hidden_states

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cache: VibeVoiceCodecCache | None = None,
        sample_indices: Tensor | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        if not use_cache:
            return self._trim(self.convtr(hidden_states))
        if cache is None or sample_indices is None:
            raise ValueError("Streaming transposed convolution requires cache and indices.")
        previous = cache.get(self.layer_id, sample_indices)
        if previous is None:
            previous = hidden_states[..., :0]
        combined = torch.cat((previous, hidden_states), dim=-1)
        output = self._trim(self.convtr(combined))
        if previous.shape[-1]:
            output = output[..., -(hidden_states.shape[-1] * self.stride):]
        cache.set(
            self.layer_id,
            sample_indices,
            combined[..., -self.context_size:],
        )
        return output


class VibeVoiceCodecFFN(nn.Module):

    def __init__(
        self,
        dimension: int,
        *,
        bias: bool,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(
            dimension,
            dimension * 4,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.linear2 = nn.Linear(
            dimension * 4,
            dimension,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.linear2(functional.gelu(self.linear1(hidden_states)))


class VibeVoiceCodecConvLayer(nn.Module):

    def __init__(
        self,
        dimension: int,
        config: VibeVoiceLegacyTokenizerConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        groups = dimension if config.mixer_layer == "depthwise_conv" else 1
        self.conv = VibeVoiceSConv1d(
            dimension,
            dimension,
            7,
            groups=groups,
            bias=config.conv_bias,
            causal=config.causal,
            pad_mode=config.pad_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.conv(hidden_states)


class VibeVoiceCodecBlock(nn.Module):

    def __init__(
        self,
        dimension: int,
        config: VibeVoiceLegacyTokenizerConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.norm = VibeVoiceConvRMSNorm(
            dimension,
            epsilon=config.layernorm_eps,
            elementwise_affine=config.layernorm_elementwise_affine,
            device=device,
            dtype=dtype,
        )
        self.ffn_norm = VibeVoiceConvRMSNorm(
            dimension,
            epsilon=config.layernorm_eps,
            elementwise_affine=config.layernorm_elementwise_affine,
            device=device,
            dtype=dtype,
        )
        self.mixer = VibeVoiceCodecConvLayer(
            dimension,
            config,
            device=device,
            dtype=dtype,
        )
        self.ffn = VibeVoiceCodecFFN(
            dimension,
            bias=config.conv_bias,
            device=device,
            dtype=dtype,
        )
        self.drop_path = nn.Identity()
        self.gamma = nn.Parameter(
            torch.full(
                (dimension, ),
                config.layer_scale_init_value,
                device=device,
                dtype=dtype,
            ))
        self.ffn_gamma = nn.Parameter(
            torch.full(
                (dimension, ),
                config.layer_scale_init_value,
                device=device,
                dtype=dtype,
            ))

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cache: VibeVoiceCodecCache | None = None,
        sample_indices: Tensor | None = None,
        use_cache: bool = False,
        is_final_chunk: bool = False,
    ) -> Tensor:
        residual = hidden_states
        mixed = self.mixer.conv(
            self.norm(hidden_states),
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            is_final_chunk=is_final_chunk,
        )
        hidden_states = residual + self.drop_path(mixed * self.gamma.unsqueeze(-1))
        residual = hidden_states
        feed_forward = self.ffn(self.ffn_norm(hidden_states).transpose(1, 2)).transpose(1, 2)
        return residual + self.drop_path(feed_forward * self.ffn_gamma.unsqueeze(-1))


class VibeVoiceTokenizerEncoder(nn.Module):

    def __init__(
        self,
        config: VibeVoiceLegacyTokenizerConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        ratios = tuple(reversed(config.encoder_ratios))
        depths = config.encoder_depths
        self.ratios = ratios
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                VibeVoiceSConv1d(
                    config.channels,
                    config.encoder_n_filters,
                    7,
                    causal=True,
                    pad_mode=config.pad_mode,
                    bias=config.conv_bias,
                    device=device,
                    dtype=dtype,
                )))
        for index, ratio in enumerate(ratios):
            input_channels = config.encoder_n_filters * (2**index)
            self.downsample_layers.append(
                nn.Sequential(
                    VibeVoiceSConv1d(
                        input_channels,
                        input_channels * 2,
                        ratio * 2,
                        stride=ratio,
                        causal=True,
                        pad_mode=config.pad_mode,
                        bias=config.conv_bias,
                        device=device,
                        dtype=dtype,
                    )))
        self.stages = nn.ModuleList(
            nn.Sequential(
                *(
                    VibeVoiceCodecBlock(
                        config.encoder_n_filters * (2**index),
                        config,
                        device=device,
                        dtype=dtype,
                    ) for _ in range(depth))) for index, depth in enumerate(depths))
        final_channels = config.encoder_n_filters * (2**(len(depths) - 1))
        self.norm = nn.Identity()
        self.head = VibeVoiceSConv1d(
            final_channels,
            config.vae_dim,
            7,
            causal=True,
            pad_mode=config.pad_mode,
            bias=config.conv_bias,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        input_values: Tensor,
        *,
        cache: VibeVoiceCodecCache | None = None,
        sample_indices: Tensor | None = None,
        use_cache: bool = False,
        is_final_chunk: bool = False,
    ) -> Tensor:
        hidden_states = input_values
        for index, stage in enumerate(self.stages):
            hidden_states = self.downsample_layers[index][0](
                hidden_states,
                cache=cache,
                sample_indices=sample_indices,
                use_cache=use_cache,
                is_final_chunk=is_final_chunk,
            )
            for block in stage:
                hidden_states = block(
                    hidden_states,
                    cache=cache,
                    sample_indices=sample_indices,
                    use_cache=use_cache,
                    is_final_chunk=is_final_chunk,
                )
        return self.head(
            self.norm(hidden_states),
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            is_final_chunk=is_final_chunk,
        )


class VibeVoiceTokenizerDecoder(nn.Module):

    def __init__(
        self,
        config: VibeVoiceLegacyTokenizerConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        ratios = tuple(config.decoder_ratios or ())
        depths = tuple(config.decoder_depths or ())
        self.ratios = ratios
        self.depths = depths
        highest = config.decoder_n_filters * (2**(len(depths) - 1))
        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(
            nn.Sequential(
                VibeVoiceSConv1d(
                    config.vae_dim,
                    highest,
                    7,
                    causal=True,
                    pad_mode=config.pad_mode,
                    bias=config.conv_bias,
                    device=device,
                    dtype=dtype,
                )))
        for index, ratio in enumerate(ratios):
            input_channels = config.decoder_n_filters * (2**(len(depths) - 1 - index))
            output_channels = input_channels // 2
            self.upsample_layers.append(
                nn.Sequential(
                    VibeVoiceSConvTranspose1d(
                        input_channels,
                        output_channels,
                        ratio * 2,
                        ratio,
                        causal=True,
                        trim_right_ratio=1.0,
                        bias=config.conv_bias,
                        device=device,
                        dtype=dtype,
                    )))
        self.stages = nn.ModuleList(
            nn.Sequential(
                *(
                    VibeVoiceCodecBlock(
                        config.decoder_n_filters * (2**(len(depths) - 1 - index)),
                        config,
                        device=device,
                        dtype=dtype,
                    ) for _ in range(depth))) for index, depth in enumerate(depths))
        self.norm = nn.Identity()
        self.head = VibeVoiceSConv1d(
            config.decoder_n_filters,
            config.channels,
            7,
            causal=True,
            pad_mode=config.pad_mode,
            bias=config.conv_bias,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        latents: Tensor,
        *,
        cache: VibeVoiceCodecCache | None = None,
        sample_indices: Tensor | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        hidden_states = latents
        for index, stage in enumerate(self.stages):
            upsample = self.upsample_layers[index][0]
            hidden_states = upsample(
                hidden_states,
                cache=cache,
                sample_indices=sample_indices,
                use_cache=use_cache,
            )
            for block in stage:
                hidden_states = block(
                    hidden_states,
                    cache=cache,
                    sample_indices=sample_indices,
                    use_cache=use_cache,
                )
        return self.head(
            self.norm(hidden_states),
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
        )


@dataclass(frozen=True)
class VibeVoiceCodecEncoderOutput:
    mean: Tensor
    std: float | Tensor | None = None

    def sample(
        self,
        distribution_type: str,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, float | Tensor | None]:
        if distribution_type == "none":
            return self.mean, self.std
        if distribution_type == "fix":
            noise = torch.randn(
                self.mean.shape,
                generator=generator,
                device=self.mean.device,
                dtype=self.mean.dtype,
            )
            return self.mean + float(self.std or 0.0) * noise, self.std
        if distribution_type != "gaussian":
            raise ValueError("Unsupported VibeVoice codec distribution.")
        batch = self.mean.shape[0]
        deviation = torch.randn(
            (batch, ),
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        ) * (float(self.std or 0.0) / 0.8)
        deviation = deviation.view(batch, *((1, ) * (self.mean.ndim - 1)))
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + deviation * noise, deviation


class VibeVoiceAcousticTokenizer(nn.Module):
    """Full 1.5B codec, or decoder-only realtime codec."""

    deterministic_codec_targets = ("encode", )

    def __init__(
        self,
        config: VibeVoiceLegacyTokenizerConfig | dict[str, Any],
        *,
        decoder_only: bool = False,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = VibeVoiceLegacyTokenizerConfig.from_dict(config)
        if not isinstance(config, VibeVoiceLegacyTokenizerConfig):
            raise TypeError("VibeVoice codec requires its legacy tokenizer config.")
        self.config = config
        self.std_dist_type = config.std_dist_type
        self.register_buffer(
            "fix_std",
            torch.tensor(
                config.fix_std,
                device=device,
                dtype=dtype or torch.float32,
            ),
            persistent=False,
        )
        if not decoder_only:
            self.encoder = VibeVoiceTokenizerEncoder(
                config,
                device=device,
                dtype=dtype,
            )
        self.decoder = VibeVoiceTokenizerDecoder(
            config,
            device=device,
            dtype=dtype,
        )
        if initialize:
            self.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.weight_init_value,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, VibeVoiceCodecRMSNorm) and module.weight is not None:
            nn.init.ones_(module.weight)

    @torch.no_grad()
    def encode(
        self,
        input_values: Tensor,
        *,
        cache: VibeVoiceCodecCache | None = None,
        sample_indices: Tensor | None = None,
        use_cache: bool = False,
        is_final_chunk: bool = False,
    ) -> VibeVoiceCodecEncoderOutput:
        encoder = getattr(self, "encoder", None)
        if encoder is None:
            raise RuntimeError("Realtime VibeVoice stores a decoder-only codec.")
        latents = encoder(
            input_values,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            is_final_chunk=is_final_chunk,
        )
        return VibeVoiceCodecEncoderOutput(
            mean=latents.transpose(1, 2),
            std=float(self.fix_std),
        )

    @torch.no_grad()
    def decode(
        self,
        latents: Tensor,
        *,
        cache: VibeVoiceCodecCache | None = None,
        sample_indices: Tensor | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        if latents.ndim != 3:
            raise ValueError("Codec latents must have shape [batch, time, latent].")
        if latents.shape[-1] == self.config.vae_dim:
            latents = latents.transpose(1, 2)
        elif latents.shape[1] != self.config.vae_dim:
            raise ValueError("Codec latent dimension does not match the checkpoint.")
        return self.decoder(
            latents,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
        )


class VibeVoiceSemanticTokenizer(nn.Module):

    def __init__(
        self,
        config: VibeVoiceLegacyTokenizerConfig | dict[str, Any],
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = VibeVoiceLegacyTokenizerConfig.from_dict(config)
        if not isinstance(config, VibeVoiceLegacyTokenizerConfig):
            raise TypeError("VibeVoice semantic codec requires its tokenizer config.")
        self.config = config
        self.encoder = VibeVoiceTokenizerEncoder(
            config,
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
                std=self.config.weight_init_value,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, VibeVoiceCodecRMSNorm) and module.weight is not None:
            nn.init.ones_(module.weight)

    @torch.no_grad()
    def encode(self, input_values: Tensor) -> VibeVoiceCodecEncoderOutput:
        latents = self.encoder(input_values)
        return VibeVoiceCodecEncoderOutput(
            mean=latents.transpose(1, 2),
            std=None,
        )


__all__ = [
    "VibeVoiceAcousticTokenizer",
    "VibeVoiceCodecCache",
    "VibeVoiceCodecEncoderOutput",
    "VibeVoiceSemanticTokenizer",
    "VibeVoiceTokenizerDecoder",
    "VibeVoiceTokenizerEncoder",
]
