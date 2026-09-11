"""PyTorch-only VITS/MMS-TTS architecture owned by VoiceHub.

This independent implementation follows the original MIT-licensed VITS graph
at revision ``2e561ba58618d021b5b8323d3765880f7e0ecfdb`` and the Apache-2.0
Transformers VITS checkpoint namespace at revision
``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``. It imports neither upstream
runtime and uses no NumPy, torchaudio, phonemizer, or Safetensors package.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.kernels.vits import VITSKernelOptimizable
from voicehub.models.vits.native.alignment import generate_path, maximum_path, sequence_mask
from voicehub.models.vits.native.configuration import VitsConfig
from voicehub.optimization.protocols import OptimizationCompileTarget


class VitsInputError(ValueError):
    """Raised when VITS tensor inputs violate the architecture contract."""


class VitsGenerationError(RuntimeError):
    """Raised when safe waveform generation cannot continue."""


@dataclass(frozen=True, slots=True)
class VitsSamplingConfig:
    """Request-local synthesis controls that never mutate model state."""

    speaking_rate: float = 1.0
    noise_scale: float = 0.667
    noise_scale_duration: float = 0.8
    seed: int | None = None
    max_output_frames: int = 100_000

    def __post_init__(self) -> None:
        for name in ("speaking_rate", "noise_scale", "noise_scale_duration"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"`{name}` must be a real number.")
            value = float(value)
            minimum_valid = value > 0 if name == "speaking_rate" else value >= 0
            if not math.isfinite(value) or not minimum_valid:
                comparator = "greater than zero" if name == "speaking_rate" else "non-negative"
                raise ValueError(f"`{name}` must be finite and {comparator}.")
            object.__setattr__(self, name, value)
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
                raise TypeError("`seed` must be an integer or None.")
            if not 0 <= int(self.seed) < 2**63:
                raise ValueError("`seed` must be in the interval [0, 2**63).")
            object.__setattr__(self, "seed", int(self.seed))
        if (isinstance(self.max_output_frames, bool) or not isinstance(self.max_output_frames, Integral)):
            raise TypeError("`max_output_frames` must be an integer.")
        if self.max_output_frames < 1:
            raise ValueError("`max_output_frames` must be positive.")

    @classmethod
    def from_model_config(cls, config: VitsConfig) -> VitsSamplingConfig:
        return cls(
            speaking_rate=config.speaking_rate,
            noise_scale=config.noise_scale,
            noise_scale_duration=config.noise_scale_duration,
        )


@dataclass(frozen=True, slots=True)
class VitsTextEncoderOutput:
    """Text states and diagonal-Gaussian prior statistics."""

    last_hidden_state: Tensor
    prior_means: Tensor
    prior_log_variances: Tensor
    hidden_states: tuple[Tensor, ...] = ()
    attentions: tuple[Tensor, ...] = ()


@dataclass(frozen=True, slots=True)
class VitsInferenceOutput:
    """Waveform and latent information produced by synthesis."""

    waveform: Tensor
    sequence_lengths: Tensor
    spectrogram: Tensor
    durations: Tensor
    alignment: Tensor
    hidden_states: tuple[Tensor, ...] = ()
    attentions: tuple[Tensor, ...] = ()


@dataclass(frozen=True, slots=True)
class VitsTrainingOutput:
    """Differentiable generator tensors used by the VITS training objective."""

    waveform: Tensor
    sequence_lengths: Tensor
    alignment: Tensor
    durations: Tensor
    duration_loss: Tensor
    posterior_latents: Tensor
    prior_latents: Tensor
    expanded_prior_means: Tensor
    expanded_prior_log_variances: Tensor
    posterior_means: Tensor
    posterior_log_variances: Tensor
    text_mask: Tensor
    spectrogram_mask: Tensor
    segment_start_frames: Tensor | None = None


def _activation(name: str, value: Tensor) -> Tensor:
    if name == "relu":
        return functional.relu(value)
    if name == "gelu":
        return functional.gelu(value)
    if name == "silu":
        return functional.silu(value)
    raise ValueError(f"Unsupported VITS activation {name!r}.")


def _randn(
    shape: tuple[int, ...],
    *,
    reference: Tensor,
    generator: torch.Generator | None,
) -> Tensor:
    if shape == tuple(reference.shape):
        output = torch.empty_like(
            reference,
            memory_format=torch.preserve_format,
        )
    else:
        output = torch.empty(
            shape,
            dtype=reference.dtype,
            device=reference.device,
        )
    return output.normal_(generator=generator)


def _random_segment_slices(
    inputs: Tensor,
    lengths: Tensor,
    segment_frames: int,
    *,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    """Select one source-style random latent window per batch item."""
    if isinstance(segment_frames, bool) or not isinstance(segment_frames, Integral):
        raise TypeError("`segment_frames` must be an integer.")
    segment_frames = int(segment_frames)
    if segment_frames < 1:
        raise ValueError("`segment_frames` must be positive.")
    maximum_starts = lengths - segment_frames
    if (maximum_starts < 0).any():
        shortest = int(lengths.min().item())
        raise VitsInputError(
            "Every target spectrogram must contain at least "
            f"{segment_frames} frames for windowed generator training; "
            f"the shortest item contains {shortest}.")
    random_values = torch.rand(
        tuple(lengths.shape),
        dtype=torch.float32,
        device=inputs.device,
        generator=generator,
    )
    start_frames = (random_values * (maximum_starts + 1).to(dtype=random_values.dtype)).long()
    frame_indices = (
        start_frames.unsqueeze(1) + torch.arange(segment_frames, device=inputs.device).unsqueeze(0))
    segments = inputs.gather(
        2,
        frame_indices.unsqueeze(1).expand(-1, inputs.shape[1], -1),
    )
    return segments, start_frames


def _local_generator(
    device: torch.device,
    *,
    seed: int | None,
    supplied: torch.Generator | None,
) -> torch.Generator | None:
    if seed is not None and supplied is not None:
        raise ValueError("Provide either a sampling seed or generator, not both.")
    if supplied is not None:
        return supplied
    if seed is None:
        return None
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def _unconstrained_rational_quadratic_spline(
    inputs: Tensor,
    unnormalized_widths: Tensor,
    unnormalized_heights: Tensor,
    unnormalized_derivatives: Tensor,
    *,
    reverse: bool = False,
    tail_bound: float = 5.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outputs = inputs.clone()
    log_abs_det = torch.zeros_like(inputs)
    constant = math.log(math.expm1(1.0 - min_derivative))
    derivatives = functional.pad(
        unnormalized_derivatives,
        (1, 1),
        value=constant,
    )
    if inside.any():
        transformed, transformed_log_det = _rational_quadratic_spline(
            inputs[inside],
            unnormalized_widths[inside],
            unnormalized_heights[inside],
            derivatives[inside],
            reverse=reverse,
            tail_bound=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )
        outputs[inside] = transformed
        log_abs_det[inside] = transformed_log_det
    return outputs, log_abs_det


def _rational_quadratic_spline(
    inputs: Tensor,
    unnormalized_widths: Tensor,
    unnormalized_heights: Tensor,
    unnormalized_derivatives: Tensor,
    *,
    reverse: bool,
    tail_bound: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
) -> tuple[Tensor, Tensor]:
    bin_count = unnormalized_widths.shape[-1]
    if min_bin_width * bin_count > 1.0:
        raise ValueError("Minimum spline width is too large.")
    if min_bin_height * bin_count > 1.0:
        raise ValueError("Minimum spline height is too large.")

    widths = functional.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * bin_count) * widths
    cumulative_widths = functional.pad(torch.cumsum(widths, dim=-1), (1, 0))
    cumulative_widths = 2 * tail_bound * cumulative_widths - tail_bound
    widths = cumulative_widths[..., 1:] - cumulative_widths[..., :-1]

    heights = functional.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * bin_count) * heights
    cumulative_heights = functional.pad(torch.cumsum(heights, dim=-1), (1, 0))
    cumulative_heights = 2 * tail_bound * cumulative_heights - tail_bound
    heights = cumulative_heights[..., 1:] - cumulative_heights[..., :-1]
    derivatives = min_derivative + functional.softplus(unnormalized_derivatives)

    locations = cumulative_heights if reverse else cumulative_widths
    bin_index = torch.sum(
        inputs.unsqueeze(-1) >= locations[..., :-1],
        dim=-1,
    ) - 1
    bin_index = bin_index.clamp(0, bin_count - 1).unsqueeze(-1)

    input_cumwidth = cumulative_widths.gather(-1, bin_index).squeeze(-1)
    input_width = widths.gather(-1, bin_index).squeeze(-1)
    input_cumheight = cumulative_heights.gather(-1, bin_index).squeeze(-1)
    input_height = heights.gather(-1, bin_index).squeeze(-1)
    delta = heights / widths
    input_delta = delta.gather(-1, bin_index).squeeze(-1)
    input_derivative = derivatives.gather(-1, bin_index).squeeze(-1)
    next_derivative = derivatives[..., 1:].gather(-1, bin_index).squeeze(-1)
    intermediate = input_derivative + next_derivative - 2 * input_delta

    if not reverse:
        theta = (inputs - input_cumwidth) / input_width
        theta_complement = theta * (1 - theta)
        numerator = input_height * (input_delta * theta.square() + input_derivative * theta_complement)
        denominator = input_delta + intermediate * theta_complement
        outputs = input_cumheight + numerator / denominator
        derivative_numerator = input_delta.square() * (
            next_derivative * theta.square() + 2 * input_delta * theta_complement + input_derivative *
            (1 - theta).square())
        log_abs_det = (torch.log(derivative_numerator) - 2 * torch.log(denominator))
        return outputs, log_abs_det

    offset = inputs - input_cumheight
    offset_intermediate = offset * intermediate
    a = input_height * (input_delta - input_derivative) + offset_intermediate
    b = input_height * input_derivative - offset_intermediate
    c = -input_delta * offset
    discriminant = b.square() - 4 * a * c
    if (discriminant < -1e-6).any():
        raise VitsGenerationError("Duration spline produced a negative quadratic discriminant.")
    root = (2 * c) / (-b - torch.sqrt(discriminant.clamp_min(0)))
    outputs = root * input_width + input_cumwidth
    root_complement = root * (1 - root)
    denominator = input_delta + intermediate * root_complement
    derivative_numerator = input_delta.square() * (
        next_derivative * root.square() + 2 * input_delta * root_complement + input_derivative *
        (1 - root).square())
    log_abs_det = (torch.log(derivative_numerator) - 2 * torch.log(denominator))
    return outputs, -log_abs_det


class WeightNormalizedConv1d(nn.Module):
    """Conv1d with the legacy ``weight_g``/``weight_v`` checkpoint layout."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.weight_v = nn.Parameter(torch.empty(
            output_channels,
            input_channels // groups,
            kernel_size,
        ))
        self.weight_g = nn.Parameter(torch.empty(output_channels, 1, 1))
        self.bias = nn.Parameter(torch.empty(output_channels))
        self.register_buffer(
            "_inference_weight",
            None,
            persistent=False,
        )
        self._inference_weight_signature: tuple[object, ...] | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
        with torch.no_grad():
            norm = torch.linalg.vector_norm(
                self.weight_v.float(),
                dim=(1, 2),
                keepdim=True,
            )
            self.weight_g.copy_(norm.to(dtype=self.weight_g.dtype))
            fan_in = self.weight_v.shape[1] * self.weight_v.shape[2]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _normalized_weight(self) -> Tensor:
        norm = torch.linalg.vector_norm(
            self.weight_v.float(),
            dim=(1, 2),
            keepdim=True,
        ).clamp_min(torch.finfo(torch.float32).tiny)
        scale = self.weight_g.float() / norm
        return self.weight_v * scale.to(dtype=self.weight_v.dtype)

    def cache_weight_norm_for_inference(self) -> int:
        """Cache the materialized convolution weight for eval-only inference.

        The cache is deliberately non-persistent, so canonical VITS
        checkpoint keys remain ``weight_g``/``weight_v``. Parameter
        version counters make the cached value self-invalidating after
        an in-place checkpoint load or optimizer update.
        """
        if self.training:
            raise RuntimeError("Weight-normalization caching requires eval mode.")
        with torch.no_grad():
            weight = self._normalized_weight().detach()
        self._inference_weight = weight
        self._inference_weight_signature = (
            id(self.weight_g),
            id(self.weight_v),
            self.weight_g._version,
            self.weight_v._version,
            self.weight_g.device,
            self.weight_v.device,
            self.weight_g.dtype,
            self.weight_v.dtype,
        )
        return weight.numel() * weight.element_size()

    def clear_weight_norm_cache(self) -> None:
        """Discard a previously materialized inference weight."""
        self._inference_weight = None
        self._inference_weight_signature = None

    def normalized_weight(self) -> Tensor:
        compiler = getattr(torch, "compiler", None)
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling) and is_compiling():
            # The cache's Python identity/version guard is deliberately
            # outside compiled graphs. Let Dynamo capture the original tensor
            # expression so full-graph regional compilation remains valid
            # after the public inference wrapper has materialized its cache.
            return self._normalized_weight()
        cached = self._inference_weight
        if (cached is not None and not self.training and not torch.is_grad_enabled()):
            current_signature = (
                id(self.weight_g),
                id(self.weight_v),
                self.weight_g._version,
                self.weight_v._version,
                self.weight_g.device,
                self.weight_v.device,
                self.weight_g.dtype,
                self.weight_v.dtype,
            )
            if self._inference_weight_signature == current_signature:
                return cached
            self.clear_weight_norm_cache()
        return self._normalized_weight()

    def train(self, mode: bool = True) -> WeightNormalizedConv1d:
        super().train(mode)
        if mode:
            self.clear_weight_norm_cache()
        return self

    def _apply(self, fn, recurse: bool = True):
        self.clear_weight_norm_cache()
        return super()._apply(fn, recurse=recurse)

    def forward(self, inputs: Tensor) -> Tensor:
        return functional.conv1d(
            inputs,
            self.normalized_weight(),
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class VitsWaveNet(VITSKernelOptimizable, nn.Module):
    """Gated residual WaveNet used by posterior and coupling flows."""

    def __init__(self, config: VitsConfig, num_layers: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = num_layers
        self.speaker_embedding_size = config.speaker_embedding_size
        self._initialize_vits_kernel_backend()
        self.dropout = nn.Dropout(config.wavenet_dropout)
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        if config.speaker_embedding_size:
            self.cond_layer = WeightNormalizedConv1d(
                config.speaker_embedding_size,
                2 * config.hidden_size * num_layers,
                1,
            )
        for index in range(num_layers):
            dilation = config.wavenet_dilation_rate**index
            padding = (config.wavenet_kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                WeightNormalizedConv1d(
                    config.hidden_size,
                    2 * config.hidden_size,
                    config.wavenet_kernel_size,
                    dilation=dilation,
                    padding=padding,
                ))
            output_channels = (2 * config.hidden_size if index < num_layers - 1 else config.hidden_size)
            self.res_skip_layers.append(WeightNormalizedConv1d(
                config.hidden_size,
                output_channels,
                1,
            ))

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        global_conditioning: Tensor | None = None,
    ) -> Tensor:
        outputs = torch.zeros_like(inputs)
        conditioning = (self.cond_layer(global_conditioning) if global_conditioning is not None else None)
        for index, (input_layer, output_layer) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            hidden = input_layer(inputs)
            if conditioning is None:
                condition = torch.zeros_like(hidden)
            else:
                start = index * 2 * self.hidden_size
                condition = conditioning[:, start:start + 2 * self.hidden_size]
            activations = self._vits_fused_gate(
                hidden,
                condition,
                self.hidden_size,
            )
            activations = self.dropout(activations)
            residual_skip = output_layer(activations)
            if index < self.num_layers - 1:
                inputs = (inputs + residual_skip[:, :self.hidden_size]) * padding_mask
                outputs = outputs + residual_skip[:, self.hidden_size:]
            else:
                outputs = outputs + residual_skip
        return outputs * padding_mask


class VitsPosteriorEncoder(nn.Module):
    """Encode target linear spectrograms into posterior latent variables."""

    def __init__(self, config: VitsConfig) -> None:
        super().__init__()
        self.output_channels = config.flow_size
        self.conv_pre = nn.Conv1d(
            config.spectrogram_bins,
            config.hidden_size,
            1,
        )
        self.wavenet = VitsWaveNet(
            config,
            config.posterior_encoder_num_wavenet_layers,
        )
        self.conv_proj = nn.Conv1d(
            config.hidden_size,
            config.flow_size * 2,
            1,
        )

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        global_conditioning: Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.conv_pre(inputs) * padding_mask
        hidden = self.wavenet(hidden, padding_mask, global_conditioning)
        statistics = self.conv_proj(hidden) * padding_mask
        mean, log_stddev = torch.split(
            statistics,
            self.output_channels,
            dim=1,
        )
        noise = _randn(
            tuple(mean.shape),
            reference=mean,
            generator=generator,
        )
        sampled = (mean + noise * torch.exp(log_stddev)) * padding_mask
        return sampled, mean, log_stddev


class HifiGanResidualBlock(nn.Module):
    """One HiFi-GAN residual block in the MMS checkpoint namespace."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: tuple[int, ...],
        leaky_relu_slope: float,
    ) -> None:
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.convs1 = nn.ModuleList(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=value,
                padding=(kernel_size * value - value) // 2,
            ) for value in dilation)
        self.convs2 = nn.ModuleList(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
            ) for _ in dilation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        for first, second in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = functional.leaky_relu(
                hidden_states,
                self.leaky_relu_slope,
            )
            hidden_states = first(hidden_states)
            hidden_states = functional.leaky_relu(
                hidden_states,
                self.leaky_relu_slope,
            )
            hidden_states = second(hidden_states) + residual
        return hidden_states


class VitsHifiGan(nn.Module):
    """HiFi-GAN waveform decoder used by official VITS checkpoints."""

    def __init__(self, config: VitsConfig) -> None:
        super().__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.conv_pre = nn.Conv1d(
            config.flow_size,
            config.upsample_initial_channel,
            7,
            padding=3,
        )
        self.upsampler = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        channels = config.upsample_initial_channel
        for index, (rate, kernel) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            output_channels = config.upsample_initial_channel // (2**(index + 1))
            self.upsampler.append(
                nn.ConvTranspose1d(
                    channels,
                    output_channels,
                    kernel,
                    stride=rate,
                    padding=(kernel - rate) // 2,
                ))
            for res_kernel, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(
                    HifiGanResidualBlock(
                        output_channels,
                        res_kernel,
                        dilation,
                        config.leaky_relu_slope,
                    ))
            channels = output_channels
        self.conv_post = nn.Conv1d(
            channels,
            1,
            7,
            padding=3,
            bias=False,
        )
        if config.speaker_embedding_size:
            self.cond = nn.Conv1d(
                config.speaker_embedding_size,
                config.upsample_initial_channel,
                1,
            )

    def forward(
        self,
        spectrogram: Tensor,
        global_conditioning: Tensor | None = None,
    ) -> Tensor:
        hidden = self.conv_pre(spectrogram)
        if global_conditioning is not None:
            hidden = hidden + self.cond(global_conditioning)
        for index, upsampler in enumerate(self.upsampler):
            hidden = functional.leaky_relu(
                hidden,
                self.config.leaky_relu_slope,
            )
            hidden = upsampler(hidden)
            offset = index * self.num_kernels
            residual = self.resblocks[offset](hidden)
            for kernel_index in range(1, self.num_kernels):
                residual = residual + self.resblocks[offset + kernel_index](hidden)
            hidden = residual / self.num_kernels
        hidden = functional.leaky_relu(hidden)
        return torch.tanh(self.conv_post(hidden))


class VitsResidualCouplingLayer(nn.Module):
    """Mean-only affine coupling transform used by the VITS prior."""

    def __init__(self, config: VitsConfig) -> None:
        super().__init__()
        self.half_channels = config.flow_size // 2
        self.conv_pre = nn.Conv1d(
            self.half_channels,
            config.hidden_size,
            1,
        )
        self.wavenet = VitsWaveNet(
            config,
            config.prior_encoder_num_wavenet_layers,
        )
        self.conv_post = nn.Conv1d(
            config.hidden_size,
            self.half_channels,
            1,
        )

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        global_conditioning: Tensor | None = None,
        *,
        reverse: bool = False,
    ) -> Tensor:
        first, second = torch.split(
            inputs,
            (self.half_channels, self.half_channels),
            dim=1,
        )
        hidden = self.conv_pre(first) * padding_mask
        hidden = self.wavenet(
            hidden,
            padding_mask,
            global_conditioning,
        )
        mean = self.conv_post(hidden) * padding_mask
        second = ((second - mean) if reverse else (mean + second)) * padding_mask
        return torch.cat((first, second), dim=1)


class VitsResidualCouplingBlock(nn.Module):
    """Stack of coupling transforms with channel reversal between stages."""

    def __init__(self, config: VitsConfig) -> None:
        super().__init__()
        self.flows = nn.ModuleList(
            VitsResidualCouplingLayer(config) for _ in range(config.prior_encoder_num_flows))

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        global_conditioning: Tensor | None = None,
        *,
        reverse: bool = False,
    ) -> Tensor:
        if reverse:
            for flow in reversed(self.flows):
                inputs = torch.flip(inputs, (1, ))
                inputs = flow(
                    inputs,
                    padding_mask,
                    global_conditioning,
                    reverse=True,
                )
        else:
            for flow in self.flows:
                inputs = flow(inputs, padding_mask, global_conditioning)
                inputs = torch.flip(inputs, (1, ))
        return inputs


class VitsDilatedDepthSeparableConv(nn.Module):
    """Dilated depth-separable residual stack for duration flows."""

    def __init__(
        self,
        config: VitsConfig,
        *,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        kernel = config.duration_predictor_kernel_size
        channels = config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.convs_dilated = nn.ModuleList()
        self.convs_pointwise = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for index in range(config.depth_separable_num_layers):
            dilation = kernel**index
            padding = (kernel * dilation - dilation) // 2
            self.convs_dilated.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                ))
            self.convs_pointwise.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(nn.LayerNorm(channels, eps=config.layer_norm_eps))
            self.norms_2.append(nn.LayerNorm(channels, eps=config.layer_norm_eps))

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        global_conditioning: Tensor | None = None,
    ) -> Tensor:
        if global_conditioning is not None:
            inputs = inputs + global_conditioning
        for dilated, pointwise, norm_1, norm_2 in zip(self.convs_dilated, self.convs_pointwise, self.norms_1,
                                                      self.norms_2):
            hidden = dilated(inputs * padding_mask)
            hidden = norm_1(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = functional.gelu(hidden)
            hidden = pointwise(hidden)
            hidden = norm_2(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = functional.gelu(hidden)
            inputs = inputs + self.dropout(hidden)
        return inputs * padding_mask


class VitsConvFlow(nn.Module):
    """Rational-quadratic spline coupling used for stochastic durations."""

    def __init__(self, config: VitsConfig) -> None:
        super().__init__()
        self.filter_channels = config.hidden_size
        self.half_channels = config.depth_separable_channels // 2
        self.num_bins = config.duration_predictor_flow_bins
        self.tail_bound = config.duration_predictor_tail_bound
        self.conv_pre = nn.Conv1d(
            self.half_channels,
            config.hidden_size,
            1,
        )
        self.conv_dds = VitsDilatedDepthSeparableConv(config)
        self.conv_proj = nn.Conv1d(
            config.hidden_size,
            self.half_channels * (self.num_bins * 3 - 1),
            1,
        )

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        global_conditioning: Tensor | None = None,
        *,
        reverse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        first, second = torch.split(
            inputs,
            (self.half_channels, self.half_channels),
            dim=1,
        )
        hidden = self.conv_pre(first)
        hidden = self.conv_dds(
            hidden,
            padding_mask,
            global_conditioning,
        )
        hidden = self.conv_proj(hidden) * padding_mask
        batch, channels, length = first.shape
        hidden = hidden.reshape(batch, channels, -1, length).permute(
            0,
            1,
            3,
            2,
        )
        widths = hidden[..., :self.num_bins] / math.sqrt(self.filter_channels)
        heights = hidden[..., self.num_bins:2 * self.num_bins] / math.sqrt(self.filter_channels)
        derivatives = hidden[..., 2 * self.num_bins:]
        second, log_abs_det = _unconstrained_rational_quadratic_spline(
            second,
            widths,
            heights,
            derivatives,
            reverse=reverse,
            tail_bound=self.tail_bound,
        )
        outputs = torch.cat((first, second), dim=1) * padding_mask
        log_determinant = torch.sum(
            log_abs_det * padding_mask,
            dim=(1, 2),
        )
        return outputs, log_determinant


class VitsElementwiseAffine(nn.Module):
    """Learned channel-wise affine transform for duration flows."""

    def __init__(self, config: VitsConfig) -> None:
        super().__init__()
        channels = config.depth_separable_channels
        self.translate = nn.Parameter(torch.zeros(channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(channels, 1))

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        global_conditioning: Tensor | None = None,
        *,
        reverse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del global_conditioning
        if reverse:
            outputs = (inputs - self.translate) * torch.exp(-self.log_scale)
            determinant = -torch.sum(
                self.log_scale * padding_mask,
                dim=(1, 2),
            )
        else:
            outputs = self.translate + torch.exp(self.log_scale) * inputs
            determinant = torch.sum(
                self.log_scale * padding_mask,
                dim=(1, 2),
            )
        return outputs * padding_mask, determinant


class VitsStochasticDurationPredictor(nn.Module):
    """Variational duration flow used by MMS-TTS checkpoints."""

    def __init__(self, config: VitsConfig) -> None:
        super().__init__()
        channels = config.hidden_size
        self.conv_pre = nn.Conv1d(channels, channels, 1)
        self.conv_proj = nn.Conv1d(channels, channels, 1)
        self.conv_dds = VitsDilatedDepthSeparableConv(
            config,
            dropout_rate=config.duration_predictor_dropout,
        )
        if config.speaker_embedding_size:
            self.cond = nn.Conv1d(
                config.speaker_embedding_size,
                channels,
                1,
            )
        self.flows = nn.ModuleList([VitsElementwiseAffine(config)])
        self.flows.extend(VitsConvFlow(config) for _ in range(config.duration_predictor_num_flows))

        self.post_conv_pre = nn.Conv1d(1, channels, 1)
        self.post_conv_proj = nn.Conv1d(channels, channels, 1)
        self.post_conv_dds = VitsDilatedDepthSeparableConv(
            config,
            dropout_rate=config.duration_predictor_dropout,
        )
        self.post_flows = nn.ModuleList([VitsElementwiseAffine(config)])
        self.post_flows.extend(VitsConvFlow(config) for _ in range(config.duration_predictor_num_flows))

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        global_conditioning: Tensor | None = None,
        *,
        durations: Tensor | None = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        inputs = inputs.detach()
        inputs = self.conv_pre(inputs)
        if global_conditioning is not None:
            inputs = inputs + self.cond(global_conditioning.detach())
        inputs = self.conv_dds(inputs, padding_mask)
        inputs = self.conv_proj(inputs) * padding_mask

        if reverse:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]
            latents = _randn(
                (
                    inputs.shape[0],
                    2,
                    inputs.shape[2],
                ),
                reference=inputs,
                generator=generator,
            ) * noise_scale
            for flow in flows:
                latents = torch.flip(latents, (1, ))
                latents, _ = flow(
                    latents,
                    padding_mask,
                    inputs,
                    reverse=True,
                )
            return latents[:, :1]

        if durations is None:
            raise ValueError("`durations` are required to train the stochastic predictor.")
        hidden = self.post_conv_pre(durations)
        hidden = self.post_conv_dds(hidden, padding_mask)
        hidden = self.post_conv_proj(hidden) * padding_mask
        random_posterior = _randn(
            (
                durations.shape[0],
                2,
                durations.shape[2],
            ),
            reference=inputs,
            generator=generator,
        ) * padding_mask
        posterior = random_posterior
        posterior_logdet = torch.zeros(
            durations.shape[0],
            dtype=inputs.dtype,
            device=inputs.device,
        )
        for flow in self.post_flows:
            posterior, determinant = flow(
                posterior,
                padding_mask,
                inputs + hidden,
            )
            posterior = torch.flip(posterior, (1, ))
            posterior_logdet = posterior_logdet + determinant

        first, second = torch.split(posterior, (1, 1), dim=1)
        posterior_logdet = posterior_logdet + torch.sum(
            (functional.logsigmoid(first) + functional.logsigmoid(-first)) * padding_mask,
            dim=(1, 2),
        )
        log_q = torch.sum(
            -0.5 * (math.log(2 * math.pi) + random_posterior.square()) * padding_mask,
            dim=(1, 2),
        ) - posterior_logdet

        first = (durations - torch.sigmoid(first)) * padding_mask
        first = torch.log(first.clamp_min(1e-5)) * padding_mask
        flow_logdet = torch.sum(-first, dim=(1, 2))
        latents = torch.cat((first, second), dim=1)
        for flow in self.flows:
            latents, determinant = flow(
                latents,
                padding_mask,
                inputs,
            )
            latents = torch.flip(latents, (1, ))
            flow_logdet = flow_logdet + determinant
        negative_log_likelihood = torch.sum(
            0.5 * (math.log(2 * math.pi) + latents.square()) * padding_mask,
            dim=(1, 2),
        ) - flow_logdet
        return negative_log_likelihood + log_q


class VitsDurationPredictor(nn.Module):
    """Deterministic convolutional duration predictor."""

    def __init__(self, config: VitsConfig) -> None:
        super().__init__()
        kernel = config.duration_predictor_kernel_size
        channels = config.duration_predictor_filter_channels
        self.dropout = nn.Dropout(config.duration_predictor_dropout)
        self.conv_1 = nn.Conv1d(
            config.hidden_size,
            channels,
            kernel,
            padding=kernel // 2,
        )
        self.norm_1 = nn.LayerNorm(channels, eps=config.layer_norm_eps)
        self.conv_2 = nn.Conv1d(
            channels,
            channels,
            kernel,
            padding=kernel // 2,
        )
        self.norm_2 = nn.LayerNorm(channels, eps=config.layer_norm_eps)
        self.proj = nn.Conv1d(channels, 1, 1)
        if config.speaker_embedding_size:
            self.cond = nn.Conv1d(
                config.speaker_embedding_size,
                config.hidden_size,
                1,
            )

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        global_conditioning: Tensor | None = None,
    ) -> Tensor:
        inputs = inputs.detach()
        if global_conditioning is not None:
            inputs = inputs + self.cond(global_conditioning.detach())
        inputs = self.conv_1(inputs * padding_mask)
        inputs = functional.relu(inputs)
        inputs = self.norm_1(inputs.transpose(1, 2)).transpose(1, 2)
        inputs = self.dropout(inputs)
        inputs = self.conv_2(inputs * padding_mask)
        inputs = functional.relu(inputs)
        inputs = self.norm_2(inputs.transpose(1, 2)).transpose(1, 2)
        inputs = self.dropout(inputs)
        return self.proj(inputs * padding_mask) * padding_mask


class VitsAttention(nn.Module):
    """Multi-head self-attention with VITS relative positions."""

    def __init__(self, config: VitsConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.window_size = config.window_size
        self.k_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=config.use_bias,
        )
        self.v_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=config.use_bias,
        )
        self.q_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=config.use_bias,
        )
        self.out_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=config.use_bias,
        )
        if self.window_size:
            relative_length = self.window_size * 2 + 1
            self.emb_rel_k = nn.Parameter(torch.randn(1, relative_length, self.head_dim) * self.scaling)
            self.emb_rel_v = nn.Parameter(torch.randn(1, relative_length, self.head_dim) * self.scaling)

    def _shape(self, value: Tensor, batch: int) -> Tensor:
        return value.reshape(
            batch,
            -1,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None]:
        batch, target_length, width = hidden_states.shape
        if width != self.embed_dim:
            raise ValueError("VITS attention received the wrong hidden width.")
        queries = self._shape(
            self.q_proj(hidden_states) * self.scaling,
            batch,
        ).reshape(batch * self.num_heads, target_length, self.head_dim)
        keys = self._shape(
            self.k_proj(hidden_states),
            batch,
        ).reshape(batch * self.num_heads, target_length, self.head_dim)
        values = self._shape(
            self.v_proj(hidden_states),
            batch,
        ).reshape(batch * self.num_heads, target_length, self.head_dim)
        weights = torch.bmm(queries, keys.transpose(1, 2))

        if self.window_size:
            relative_keys = self._relative_embeddings(
                self.emb_rel_k,
                target_length,
            )
            relative_logits = torch.matmul(
                queries,
                relative_keys.transpose(-2, -1),
            )
            weights = weights + self._relative_to_absolute(relative_logits)
        if attention_mask is not None:
            expected = (batch, 1, target_length, target_length)
            if tuple(attention_mask.shape) != expected:
                raise ValueError(f"VITS attention mask must have shape {expected}.")
            weights = weights.reshape(
                batch,
                self.num_heads,
                target_length,
                target_length,
            )
            weights = weights + attention_mask
            weights = weights.reshape(
                batch * self.num_heads,
                target_length,
                target_length,
            )
        weights = functional.softmax(weights.float(), dim=-1).to(dtype=queries.dtype)
        attention = (
            weights.reshape(
                batch,
                self.num_heads,
                target_length,
                target_length,
            ) if output_attentions else None)
        probabilities = functional.dropout(
            weights,
            p=self.dropout,
            training=self.training,
        )
        output = torch.bmm(probabilities, values)
        if self.window_size:
            relative_values = self._relative_embeddings(
                self.emb_rel_v,
                target_length,
            )
            relative_weights = self._absolute_to_relative(probabilities)
            output = output + torch.matmul(
                relative_weights,
                relative_values,
            )
        output = output.reshape(
            batch,
            self.num_heads,
            target_length,
            self.head_dim,
        ).transpose(1, 2).reshape(batch, target_length, self.embed_dim)
        return self.out_proj(output), attention

    def _relative_embeddings(
        self,
        embeddings: Tensor,
        length: int,
    ) -> Tensor:
        if self.window_size is None:
            raise RuntimeError("Relative positions are disabled.")
        pad = max(length - (self.window_size + 1), 0)
        if pad:
            embeddings = functional.pad(embeddings, (0, 0, pad, pad))
        start = max((self.window_size + 1) - length, 0)
        return embeddings[:, start:start + 2 * length - 1]

    @staticmethod
    def _relative_to_absolute(value: Tensor) -> Tensor:
        batch_heads, length, _ = value.shape
        value = functional.pad(value, (0, 1))
        flattened = value.reshape(batch_heads, length * 2 * length)
        flattened = functional.pad(flattened, (0, length - 1))
        return flattened.reshape(
            batch_heads,
            length + 1,
            2 * length - 1,
        )[:, :length, length - 1:]

    @staticmethod
    def _absolute_to_relative(value: Tensor) -> Tensor:
        batch_heads, length, _ = value.shape
        value = functional.pad(value, (0, length - 1))
        flattened = value.reshape(
            batch_heads,
            length * (2 * length - 1),
        )
        flattened = functional.pad(flattened, (length, 0))
        return flattened.reshape(batch_heads, length, 2 * length)[:, :, 1:]


class VitsFeedForward(nn.Module):
    """Convolutional Transformer feed-forward network."""

    def __init__(self, config: VitsConfig) -> None:
        super().__init__()
        self.activation_name = config.hidden_act
        self.conv_1 = nn.Conv1d(
            config.hidden_size,
            config.ffn_dim,
            config.ffn_kernel_size,
        )
        self.conv_2 = nn.Conv1d(
            config.ffn_dim,
            config.hidden_size,
            config.ffn_kernel_size,
        )
        self.dropout = nn.Dropout(config.activation_dropout)
        self.padding = (
            (config.ffn_kernel_size - 1) // 2,
            config.ffn_kernel_size // 2,
        )

    def forward(self, hidden_states: Tensor, padding_mask: Tensor) -> Tensor:
        hidden = hidden_states.transpose(1, 2)
        mask = padding_mask.transpose(1, 2)
        hidden = functional.pad(hidden * mask, self.padding)
        hidden = self.conv_1(hidden)
        hidden = _activation(self.activation_name, hidden)
        hidden = self.dropout(hidden)
        hidden = functional.pad(hidden * mask, self.padding)
        hidden = self.conv_2(hidden) * mask
        return hidden.transpose(1, 2)


class VitsEncoderLayer(nn.Module):
    """One pre-defined VITS Transformer encoder block."""

    def __init__(self, config: VitsConfig) -> None:
        super().__init__()
        self.attention = VitsAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.feed_forward = VitsFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        padding_mask: Tensor,
        attention_mask: Tensor,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None]:
        residual = hidden_states
        hidden_states, attention = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.layer_norm(residual + self.dropout(hidden_states))
        residual = hidden_states
        hidden_states = self.feed_forward(hidden_states, padding_mask)
        hidden_states = self.final_layer_norm(residual + self.dropout(hidden_states))
        return hidden_states, attention


class VitsEncoder(nn.Module):
    """Relative-position Transformer encoder with tensor-local LayerDrop."""

    def __init__(self, config: VitsConfig) -> None:
        super().__init__()
        self.layerdrop = config.layerdrop
        self.layers = nn.ModuleList(VitsEncoderLayer(config) for _ in range(config.num_hidden_layers))

    def forward(
        self,
        hidden_states: Tensor,
        *,
        padding_mask: Tensor,
        attention_mask: Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        key_mask = attention_mask.to(dtype=torch.bool)
        additive_mask = torch.zeros(
            (
                hidden_states.shape[0],
                1,
                hidden_states.shape[1],
                hidden_states.shape[1],
            ),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        additive_mask.masked_fill_(
            ~key_mask[:, None, None, :],
            torch.finfo(hidden_states.dtype).min,
        )
        hidden_states = hidden_states * padding_mask
        all_hidden = []
        all_attention = []
        for layer in self.layers:
            if output_hidden_states:
                all_hidden.append(hidden_states)
            skip = False
            if self.training and self.layerdrop:
                probability = torch.rand(
                    (),
                    device=hidden_states.device,
                    generator=generator,
                )
                skip = bool(probability < self.layerdrop)
            if skip:
                attention = None
            else:
                hidden_states, attention = layer(
                    hidden_states,
                    padding_mask=padding_mask,
                    attention_mask=additive_mask,
                    output_attentions=output_attentions,
                )
            if output_attentions:
                all_attention.append(
                    attention if attention is not None else torch.empty(
                        0,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ))
        hidden_states = hidden_states * padding_mask
        if output_hidden_states:
            all_hidden.append(hidden_states)
        return hidden_states, tuple(all_hidden), tuple(all_attention)


class VitsTextEncoder(nn.Module):
    """Text embedding, relative Transformer, and Gaussian prior heads."""

    def __init__(self, config: VitsConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.encoder = VitsEncoder(config)
        self.project = nn.Conv1d(
            config.hidden_size,
            config.flow_size * 2,
            1,
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        padding_mask: Tensor,
        attention_mask: Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        generator: torch.Generator | None = None,
    ) -> VitsTextEncoderOutput:
        hidden = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)
        hidden, all_hidden, all_attention = self.encoder(
            hidden,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generator=generator,
        )
        statistics = self.project(hidden.transpose(1, 2)).transpose(
            1,
            2,
        ) * padding_mask
        means, log_variances = torch.split(
            statistics,
            self.config.flow_size,
            dim=2,
        )
        return VitsTextEncoderOutput(
            last_hidden_state=hidden,
            prior_means=means,
            prior_log_variances=log_variances,
            hidden_states=all_hidden,
            attentions=all_attention,
        )


class VitsModel(nn.Module):
    """Native VITS generator with inference and supervised generator
    training."""

    def __init__(self, config: VitsConfig | Mapping[str, object]) -> None:
        super().__init__()
        self.config = VitsConfig.coerce(config)
        self.text_encoder = VitsTextEncoder(self.config)
        self.flow = VitsResidualCouplingBlock(self.config)
        self.decoder = VitsHifiGan(self.config)
        if self.config.use_stochastic_duration_prediction:
            self.duration_predictor: nn.Module = VitsStochasticDurationPredictor(self.config)
        else:
            self.duration_predictor = VitsDurationPredictor(self.config)
        if self.config.is_multispeaker:
            self.embed_speaker = nn.Embedding(
                self.config.num_speakers,
                self.config.speaker_embedding_size,
            )
        self.posterior_encoder = VitsPosteriorEncoder(self.config)
        self.apply(self._initialize_module)

    def cache_weight_norm_for_inference(self) -> int:
        """Materialize reusable WaveNet weights and return cache bytes.

        This is an opt-in deployment optimization. Call ``eval()`` first
        and run synthesis under ``torch.inference_mode()`` or
        ``torch.no_grad()``. The cache is excluded from ``state_dict``
        and is cleared by ``train()`` or device/dtype conversion.
        """
        if self.training:
            raise RuntimeError("Weight-normalization caching requires eval mode.")
        return sum(
            module.cache_weight_norm_for_inference() for module in self.modules()
            if isinstance(module, WeightNormalizedConv1d))

    def clear_weight_norm_cache(self) -> None:
        """Clear every materialized WaveNet inference weight."""
        for module in self.modules():
            if isinstance(module, WeightNormalizedConv1d):
                module.clear_weight_norm_cache()

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose only the quality-safe full-sequence training boundary."""
        if mode == "inference":
            # The regional inference graph passed tiny-graph tests but changed
            # the fixed-seed waveform for the pinned MMS-TTS checkpoint.
            return ()
        if mode == "training":
            return (OptimizationCompileTarget(
                "vits.forward",
                self,
                "forward",
            ), )
        raise ValueError(f"Unsupported optimization mode {mode!r}.")

    def _initialize_module(self, module: nn.Module) -> None:
        """Apply the published VITS initialization without framework hooks."""
        if isinstance(module, WeightNormalizedConv1d):
            nn.init.kaiming_normal_(module.weight_v)
            with torch.no_grad():
                norm = torch.linalg.vector_norm(
                    module.weight_v.float(),
                    dim=(1, 2),
                    keepdim=True,
                )
                module.weight_g.copy_(norm.to(dtype=module.weight_g.dtype))
                fan_in = module.weight_v.shape[1] * module.weight_v.shape[2]
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                bound = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Linear):
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
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        if isinstance(module, VitsAttention) and module.window_size:
            nn.init.normal_(module.emb_rel_k, std=module.head_dim**-0.5)
            nn.init.normal_(module.emb_rel_v, std=module.head_dim**-0.5)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        speaker_id: int | Tensor | None = None,
        spectrogram: Tensor | None = None,
        spectrogram_attention_mask: Tensor | None = None,
        durations: Tensor | None = None,
        segment_frames: int | None = None,
        sampling: VitsSamplingConfig | None = None,
        generator: torch.Generator | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> VitsInferenceOutput | VitsTrainingOutput:
        """Run synthesis or the supervised generator graph.

        Supplying ``spectrogram`` selects training mode and returns
        latent, alignment, duration, and waveform tensors needed by the
        complete VITS generator-side objective. Adversarial and feature-
        matching losses live in
        :mod:`voicehub.models.vits.native.losses` because discriminator
        and generator optimizers must remain separate. ``segment_frames``
        applies the original VITS windowed-generator optimization: the
        complete posterior remains available to the latent objectives,
        while only a random per-item latent window reaches the decoder.
        """
        input_ids, attention_mask, padding_mask = _validate_text_inputs(
            input_ids,
            attention_mask,
            config=self.config,
        )
        padding_mask = padding_mask.to(dtype=self.text_encoder.embed_tokens.weight.dtype)
        conditioning = self._speaker_conditioning(
            speaker_id,
            batch_size=input_ids.shape[0],
            device=input_ids.device,
        )
        if spectrogram is None:
            if (spectrogram_attention_mask is not None or durations is not None or
                    segment_frames is not None):
                raise VitsInputError(
                    "Spectrogram masks, supervised durations, and training "
                    "segments require a target spectrogram.")
            return self._synthesize(
                input_ids,
                attention_mask,
                padding_mask,
                conditioning,
                sampling=sampling,
                generator=generator,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        if sampling is not None:
            raise VitsInputError("Inference sampling options cannot be used with a target "
                                 "spectrogram.")
        return self._training_forward(
            input_ids,
            attention_mask,
            padding_mask,
            conditioning,
            spectrogram=spectrogram,
            spectrogram_attention_mask=spectrogram_attention_mask,
            durations=durations,
            segment_frames=segment_frames,
            generator=generator,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

    def synthesize(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        speaker_id: int | Tensor | None = None,
        sampling: VitsSamplingConfig | None = None,
        generator: torch.Generator | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> VitsInferenceOutput:
        """Explicit synthesis entry point with request-local sampling."""
        output = self.forward(
            input_ids,
            attention_mask,
            speaker_id=speaker_id,
            sampling=sampling,
            generator=generator,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if not isinstance(output, VitsInferenceOutput):
            raise TypeError("VITS synthesis returned a training output.")
        return output

    def _synthesize(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        padding_mask: Tensor,
        conditioning: Tensor | None,
        *,
        sampling: VitsSamplingConfig | None,
        generator: torch.Generator | None,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> VitsInferenceOutput:
        options = (VitsSamplingConfig.from_model_config(self.config) if sampling is None else sampling)
        if not isinstance(options, VitsSamplingConfig):
            raise TypeError("`sampling` must be a VitsSamplingConfig or None.")
        generator = _local_generator(
            input_ids.device,
            seed=options.seed,
            supplied=generator,
        )
        encoded = self.text_encoder(
            input_ids,
            padding_mask=padding_mask.transpose(1, 2),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generator=generator,
        )
        hidden = encoded.last_hidden_state.transpose(1, 2)
        if self.config.use_stochastic_duration_prediction:
            log_duration = self.duration_predictor(
                hidden,
                padding_mask,
                conditioning,
                reverse=True,
                noise_scale=options.noise_scale_duration,
                generator=generator,
            )
        else:
            log_duration = self.duration_predictor(
                hidden,
                padding_mask,
                conditioning,
            )
        if not torch.isfinite(log_duration).all():
            raise VitsGenerationError("Duration prediction produced NaN or infinite values.")
        durations = torch.ceil(torch.exp(log_duration.float()) * padding_mask.float() / options.speaking_rate)
        if not torch.isfinite(durations).all():
            raise VitsGenerationError(
                "Duration expansion overflowed; use a compatible checkpoint "
                "or a stricter speaking-rate/output-frame policy.")
        predicted_lengths = durations.sum(dim=(1, 2)).clamp_min(1).long()
        if (predicted_lengths > options.max_output_frames).any():
            maximum = int(predicted_lengths.max().item())
            raise VitsGenerationError(
                f"VITS predicted {maximum} frames, exceeding the request "
                f"limit of {options.max_output_frames}.")
        output_mask = sequence_mask(
            predicted_lengths,
            int(predicted_lengths.max().item()),
        ).unsqueeze(1).to(dtype=padding_mask.dtype)
        attention = generate_path(
            durations,
            padding_mask.unsqueeze(2) * output_mask.unsqueeze(-1),
        )
        prior_means = torch.matmul(
            attention.squeeze(1),
            encoded.prior_means,
        ).transpose(1, 2)
        prior_log_variances = torch.matmul(
            attention.squeeze(1),
            encoded.prior_log_variances,
        ).transpose(1, 2)
        noise = _randn(
            tuple(prior_means.shape),
            reference=prior_means,
            generator=generator,
        )
        prior_latents = (prior_means + noise * torch.exp(prior_log_variances) * options.noise_scale)
        latents = self.flow(
            prior_latents,
            output_mask,
            conditioning,
            reverse=True,
        )
        spectrogram = latents * output_mask
        waveform = self.decoder(spectrogram, conditioning).squeeze(1)
        sample_lengths = predicted_lengths * self.config.upsample_factor
        waveform = waveform * sequence_mask(
            sample_lengths,
            waveform.shape[1],
        ).to(dtype=waveform.dtype)
        return VitsInferenceOutput(
            waveform=waveform,
            sequence_lengths=sample_lengths,
            spectrogram=spectrogram,
            durations=durations,
            alignment=attention,
            hidden_states=encoded.hidden_states,
            attentions=encoded.attentions,
        )

    def _training_forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        padding_mask: Tensor,
        conditioning: Tensor | None,
        *,
        spectrogram: Tensor,
        spectrogram_attention_mask: Tensor | None,
        durations: Tensor | None,
        segment_frames: int | None,
        generator: torch.Generator | None,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> VitsTrainingOutput:
        spectrogram, spectrogram_mask, spectrogram_lengths = (
            _validate_spectrogram_inputs(
                spectrogram,
                spectrogram_attention_mask,
                config=self.config,
                batch_size=input_ids.shape[0],
                device=input_ids.device,
            ))
        model_dtype = self.text_encoder.embed_tokens.weight.dtype
        spectrogram = spectrogram.to(dtype=model_dtype)
        spectrogram_mask = spectrogram_mask.to(dtype=model_dtype)
        encoded = self.text_encoder(
            input_ids,
            padding_mask=padding_mask.transpose(1, 2),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generator=generator,
        )
        hidden = encoded.last_hidden_state.transpose(1, 2)
        posterior, posterior_means, posterior_logs = self.posterior_encoder(
            spectrogram,
            spectrogram_mask,
            conditioning,
            generator=generator,
        )
        prior_latents = self.flow(
            posterior,
            spectrogram_mask,
            conditioning,
        )
        alignment_mask = (spectrogram_mask.unsqueeze(-1) * padding_mask.unsqueeze(2))
        if durations is None:
            with torch.no_grad():
                prior_mean = encoded.prior_means.transpose(1, 2).float()
                prior_logs = encoded.prior_log_variances.transpose(
                    1,
                    2,
                ).float()
                alignment_latents = prior_latents.float()
                inverse_variance = torch.exp(-2 * prior_logs)
                score_1 = torch.sum(
                    -0.5 * math.log(2 * math.pi) - prior_logs,
                    dim=1,
                    keepdim=True,
                )
                score_2 = torch.matmul(
                    -0.5 * alignment_latents.square().transpose(1, 2),
                    inverse_variance,
                )
                score_3 = torch.matmul(
                    alignment_latents.transpose(1, 2),
                    prior_mean * inverse_variance,
                )
                score_4 = torch.sum(
                    -0.5 * prior_mean.square() * inverse_variance,
                    dim=1,
                    keepdim=True,
                )
                scores = score_1 + score_2 + score_3 + score_4
                alignment = maximum_path(
                    scores,
                    alignment_mask[:, 0],
                ).unsqueeze(1)
                durations = alignment.float().sum(dim=2)
        else:
            durations = _validate_durations(
                durations,
                padding_mask=padding_mask,
                spectrogram_lengths=spectrogram_lengths,
            )
            alignment = generate_path(durations, alignment_mask)

        if self.config.use_stochastic_duration_prediction:
            duration_values = self.duration_predictor(
                hidden,
                padding_mask,
                conditioning,
                durations=durations.to(dtype=hidden.dtype),
                generator=generator,
            )
            duration_loss = (duration_values.float().sum() / padding_mask.float().sum().clamp_min(1))
        else:
            target_log_duration = (torch.log(durations.float() + 1e-6) * padding_mask.float())
            predicted_log_duration = self.duration_predictor(
                hidden,
                padding_mask,
                conditioning,
            )
            duration_loss = ((predicted_log_duration.float() - target_log_duration).square() *
                             padding_mask.float()).sum()
            duration_loss = (duration_loss / padding_mask.float().sum().clamp_min(1))

        expanded_means = torch.matmul(
            alignment.squeeze(1),
            encoded.prior_means,
        ).transpose(1, 2)
        expanded_logs = torch.matmul(
            alignment.squeeze(1),
            encoded.prior_log_variances,
        ).transpose(1, 2)
        decoder_latents = posterior * spectrogram_mask
        if segment_frames is None:
            segment_start_frames = torch.zeros_like(spectrogram_lengths)
            sample_lengths = spectrogram_lengths * self.config.upsample_factor
        else:
            decoder_latents, segment_start_frames = _random_segment_slices(
                decoder_latents,
                spectrogram_lengths,
                segment_frames,
                generator=generator,
            )
            sample_lengths = torch.full_like(
                spectrogram_lengths,
                int(segment_frames) * self.config.upsample_factor,
            )
        waveform = self.decoder(
            decoder_latents,
            conditioning,
        ).squeeze(1)
        waveform = waveform * sequence_mask(
            sample_lengths,
            waveform.shape[1],
        ).to(dtype=waveform.dtype)
        return VitsTrainingOutput(
            waveform=waveform,
            sequence_lengths=sample_lengths,
            alignment=alignment,
            durations=durations,
            duration_loss=duration_loss,
            posterior_latents=posterior,
            prior_latents=prior_latents,
            expanded_prior_means=expanded_means,
            expanded_prior_log_variances=expanded_logs,
            posterior_means=posterior_means,
            posterior_log_variances=posterior_logs,
            text_mask=padding_mask,
            spectrogram_mask=spectrogram_mask,
            segment_start_frames=segment_start_frames,
        )

    def _speaker_conditioning(
        self,
        speaker_id: int | Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor | None:
        if not self.config.is_multispeaker:
            if speaker_id is not None:
                raise VitsInputError("`speaker_id` is invalid for a single-speaker checkpoint.")
            return None
        if speaker_id is None:
            raise VitsInputError("Multi-speaker VITS requires one `speaker_id` per item.")
        if isinstance(speaker_id, Integral) and not isinstance(speaker_id, bool):
            speaker_ids = torch.full(
                (batch_size, ),
                int(speaker_id),
                dtype=torch.long,
                device=device,
            )
        elif isinstance(speaker_id, Tensor):
            if speaker_id.ndim == 0:
                speaker_ids = speaker_id.expand(batch_size)
            elif tuple(speaker_id.shape) == (batch_size, ):
                speaker_ids = speaker_id
            else:
                raise VitsInputError("`speaker_id` tensor must be scalar or have shape [batch].")
            if speaker_ids.dtype == torch.bool or speaker_ids.is_floating_point():
                raise TypeError("`speaker_id` must use an integer dtype.")
            speaker_ids = speaker_ids.to(device=device, dtype=torch.long)
        else:
            raise TypeError("`speaker_id` must be an integer or tensor.")
        if ((speaker_ids < 0) | (speaker_ids >= self.config.num_speakers)).any():
            raise VitsInputError(f"`speaker_id` must be in [0, "
                                 f"{self.config.num_speakers - 1}].")
        return self.embed_speaker(speaker_ids).unsqueeze(-1)


def _validate_text_inputs(
    input_ids: Tensor,
    attention_mask: Tensor | None,
    *,
    config: VitsConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise VitsInputError("`input_ids` must have shape [batch, text].")
    if input_ids.dtype == torch.bool or input_ids.is_floating_point():
        raise TypeError("`input_ids` must use an integer dtype.")
    if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
        raise VitsInputError("`input_ids` cannot have empty dimensions.")
    normalized = input_ids.to(dtype=torch.long)
    if ((normalized < 0) | (normalized >= config.vocab_size)).any():
        raise VitsInputError("`input_ids` contain an out-of-vocabulary ID.")
    mask = _validated_right_padding_mask(
        attention_mask,
        batch=input_ids.shape[0],
        length=input_ids.shape[1],
        device=input_ids.device,
        name="attention_mask",
    )
    return normalized, mask, mask.unsqueeze(1).to(dtype=torch.get_default_dtype())


def _validate_spectrogram_inputs(
    spectrogram: Tensor,
    attention_mask: Tensor | None,
    *,
    config: VitsConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(spectrogram, Tensor) or spectrogram.ndim != 3:
        raise VitsInputError("`spectrogram` must have shape [batch, bins, frames].")
    expected = (batch_size, config.spectrogram_bins)
    if tuple(spectrogram.shape[:2]) != expected:
        raise VitsInputError(f"`spectrogram` must start with dimensions {expected}.")
    if spectrogram.device != device:
        raise VitsInputError("Text and spectrogram tensors must be on the same device.")
    if not spectrogram.is_floating_point():
        raise TypeError("`spectrogram` must use a floating-point dtype.")
    if spectrogram.shape[2] < 1 or not torch.isfinite(spectrogram).all():
        raise VitsInputError("`spectrogram` must contain finite acoustic frames.")
    mask = _validated_right_padding_mask(
        attention_mask,
        batch=batch_size,
        length=spectrogram.shape[2],
        device=device,
        name="spectrogram_attention_mask",
    )
    lengths = mask.sum(dim=1, dtype=torch.long)
    return (
        spectrogram,
        mask.unsqueeze(1).to(dtype=spectrogram.dtype),
        lengths,
    )


def _validated_right_padding_mask(
    value: Tensor | None,
    *,
    batch: int,
    length: int,
    device: torch.device,
    name: str,
) -> Tensor:
    if value is None:
        return torch.ones((batch, length), dtype=torch.bool, device=device)
    if not isinstance(value, Tensor) or tuple(value.shape) != (batch, length):
        raise VitsInputError(f"`{name}` must have shape [{batch}, {length}].")
    if value.device != device:
        raise VitsInputError(f"`{name}` must be on the input device.")
    if value.is_complex() or not ((value == 0) | (value == 1)).all():
        raise VitsInputError(f"`{name}` must contain only zero and one.")
    mask = value.to(dtype=torch.bool)
    if not mask.any(dim=1).all():
        raise VitsInputError(f"Every `{name}` row needs at least one token.")
    if length > 1 and ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise VitsInputError(f"`{name}` must describe contiguous right padding.")
    return mask


def _validate_durations(
    durations: Tensor,
    *,
    padding_mask: Tensor,
    spectrogram_lengths: Tensor,
) -> Tensor:
    if not isinstance(durations, Tensor):
        raise TypeError("`durations` must be a tensor.")
    expected = tuple(padding_mask.shape)
    if tuple(durations.shape) != expected:
        raise VitsInputError(f"`durations` must have shape {expected}.")
    if durations.device != padding_mask.device:
        raise VitsInputError("`durations` must be on the input device.")
    if durations.dtype == torch.bool:
        raise TypeError("`durations` cannot use a boolean dtype.")
    if durations.is_complex():
        raise TypeError("`durations` must be real-valued.")
    normalized = durations.to(dtype=torch.float32)
    if not torch.isfinite(normalized).all() or (normalized < 0).any():
        raise VitsInputError("`durations` must be finite and non-negative.")
    if not torch.equal(normalized, normalized.round()):
        raise VitsInputError("Supervised durations must contain whole frames.")
    if (normalized * (1 - padding_mask)).any():
        raise VitsInputError("Padded text tokens must have zero duration.")
    if (normalized[padding_mask.to(dtype=torch.bool)] < 1).any():
        raise VitsInputError("Every valid text token must receive at least one frame.")
    if not torch.equal(
            normalized.sum(dim=(1, 2)).long(),
            spectrogram_lengths,
    ):
        raise VitsInputError("Supervised durations must cover every valid spectrogram frame.")
    return normalized


__all__ = [
    "HifiGanResidualBlock",
    "VitsAttention",
    "VitsConvFlow",
    "VitsDilatedDepthSeparableConv",
    "VitsDurationPredictor",
    "VitsElementwiseAffine",
    "VitsEncoder",
    "VitsEncoderLayer",
    "VitsFeedForward",
    "VitsGenerationError",
    "VitsHifiGan",
    "VitsInferenceOutput",
    "VitsInputError",
    "VitsModel",
    "VitsPosteriorEncoder",
    "VitsResidualCouplingBlock",
    "VitsResidualCouplingLayer",
    "VitsSamplingConfig",
    "VitsStochasticDurationPredictor",
    "VitsTextEncoder",
    "VitsTextEncoderOutput",
    "VitsTrainingOutput",
    "VitsWaveNet",
    "WeightNormalizedConv1d",
]
