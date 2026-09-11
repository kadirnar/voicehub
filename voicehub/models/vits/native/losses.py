"""Source-faithful adversarial components for native VITS training.

The loss equations and default discriminator topology follow the
original MIT-licensed VITS implementation at revision
``2e561ba58618d021b5b8323d3765880f7e0ecfdb``.  Acoustic preprocessing is
deliberately not guessed: a training recipe must provide the
checkpoint's linear-spectrogram inputs and mel-reconstruction loss
explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.vits.native.modeling import VitsTrainingOutput, WeightNormalizedConv1d

FeatureMaps = Sequence[Sequence[Tensor]]


def _finite_float_tensor(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"`{name}` must be a PyTorch tensor.")
    if not value.is_floating_point():
        raise TypeError(f"`{name}` must use a floating-point dtype.")
    if not torch.isfinite(value).all():
        raise ValueError(f"`{name}` cannot contain NaN or infinite values.")
    return value


def _matching_sequences(
    name: str,
    first: Sequence[object],
    second: Sequence[object],
) -> tuple[tuple[object, object], ...]:
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
        raise TypeError(f"`{name}` inputs must be sequences.")
    if not isinstance(second, Sequence) or isinstance(second, (str, bytes)):
        raise TypeError(f"`{name}` inputs must be sequences.")
    if len(first) != len(second) or not first:
        raise ValueError(f"`{name}` inputs must have the same non-zero length.")
    return tuple(zip(first, second))


def feature_matching_loss(
    real_feature_maps: FeatureMaps,
    generated_feature_maps: FeatureMaps,
) -> Tensor:
    """Return the original VITS feature-matching loss.

    Real discriminator activations are detached exactly as in the
    reference implementation; gradients flow through generated
    activations only.
    """
    loss: Tensor | None = None
    for discriminator_index, (real_maps, generated_maps) in enumerate(_matching_sequences(
            "feature maps",
            real_feature_maps,
            generated_feature_maps,
    ), ):
        for layer_index, (real, generated) in enumerate(_matching_sequences(
                f"feature maps for discriminator {discriminator_index}",
                real_maps,
                generated_maps,
        ), ):
            real = _finite_float_tensor(
                f"real_feature_maps[{discriminator_index}][{layer_index}]",
                real,
            ).float().detach()
            generated = _finite_float_tensor(
                "generated_feature_maps"
                f"[{discriminator_index}][{layer_index}]",
                generated,
            ).float()
            if real.shape != generated.shape:
                raise ValueError("Real and generated feature maps must have equal shapes.")
            term = torch.mean(torch.abs(real - generated))
            loss = term if loss is None else loss + term
    if loss is None:  # pragma: no cover - guarded by non-empty validation
        raise ValueError("Feature maps cannot be empty.")
    return loss * 2.0


def discriminator_loss(
    real_outputs: Sequence[Tensor],
    generated_outputs: Sequence[Tensor],
) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Return least-squares VITS discriminator loss and its components."""
    total: Tensor | None = None
    real_losses = []
    generated_losses = []
    for index, (real, generated) in enumerate(_matching_sequences(
            "discriminator outputs",
            real_outputs,
            generated_outputs,
    ), ):
        real = _finite_float_tensor(
            f"real_outputs[{index}]",
            real,
        ).float()
        generated = _finite_float_tensor(
            f"generated_outputs[{index}]",
            generated,
        ).float()
        if real.shape != generated.shape:
            raise ValueError("Real and generated discriminator outputs must have equal "
                             "shapes.")
        real_loss = torch.mean((1.0 - real).square())
        generated_loss = torch.mean(generated.square())
        item = real_loss + generated_loss
        total = item if total is None else total + item
        real_losses.append(real_loss)
        generated_losses.append(generated_loss)
    if total is None:  # pragma: no cover - guarded by non-empty validation
        raise ValueError("Discriminator outputs cannot be empty.")
    return total, tuple(real_losses), tuple(generated_losses)


def generator_adversarial_loss(generated_outputs: Sequence[Tensor], ) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Return least-squares VITS generator adversarial loss."""
    if (not isinstance(generated_outputs, Sequence) or isinstance(generated_outputs, (str, bytes))):
        raise TypeError("`generated_outputs` must be a sequence.")
    if not generated_outputs:
        raise ValueError("`generated_outputs` cannot be empty.")
    total: Tensor | None = None
    losses = []
    for index, generated in enumerate(generated_outputs):
        generated = _finite_float_tensor(
            f"generated_outputs[{index}]",
            generated,
        ).float()
        item = torch.mean((1.0 - generated).square())
        total = item if total is None else total + item
        losses.append(item)
    if total is None:  # pragma: no cover - guarded above
        raise ValueError("`generated_outputs` cannot be empty.")
    return total, tuple(losses)


def vits_kl_loss(
    prior_latents: Tensor,
    posterior_log_variances: Tensor,
    expanded_prior_means: Tensor,
    expanded_prior_log_variances: Tensor,
    mask: Tensor,
) -> Tensor:
    """Return the exact diagonal-Gaussian KL term used by VITS."""
    tensors = (
        _finite_float_tensor("prior_latents", prior_latents).float(),
        _finite_float_tensor(
            "posterior_log_variances",
            posterior_log_variances,
        ).float(),
        _finite_float_tensor(
            "expanded_prior_means",
            expanded_prior_means,
        ).float(),
        _finite_float_tensor(
            "expanded_prior_log_variances",
            expanded_prior_log_variances,
        ).float(),
    )
    reference_shape = tensors[0].shape
    if any(value.shape != reference_shape for value in tensors[1:]):
        raise ValueError("Every VITS KL latent tensor must have equal shape.")
    mask = _finite_float_tensor("mask", mask).float()
    if mask.ndim != 3 or mask.shape[0] != reference_shape[0]:
        raise ValueError("`mask` must have shape [batch, 1, frames].")
    if mask.shape[1] not in (1, reference_shape[1]):
        raise ValueError("`mask` channels must be one or match the latents.")
    if mask.shape[2] != reference_shape[2]:
        raise ValueError("`mask` and latent frame counts must match.")
    if ((mask < 0.0) | (mask > 1.0)).any():
        raise ValueError("`mask` values must be in the interval [0, 1].")
    denominator = mask.sum()
    if denominator <= 0:
        raise ValueError("`mask` must retain at least one latent frame.")

    z_p, logs_q, means_p, logs_p = tensors
    divergence = logs_p - logs_q - 0.5
    divergence = divergence + (0.5 * (z_p - means_p).square() * torch.exp(-2.0 * logs_p))
    return torch.sum(divergence * mask) / denominator


class WeightNormalizedConv2d(nn.Module):
    """Conv2d with stable legacy ``weight_g``/``weight_v`` parameters."""

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            kernel_size: tuple[int, int],
            *,
            stride: tuple[int, int] = (1, 1),
            padding: tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight_v = nn.Parameter(torch.empty(
            output_channels,
            input_channels,
            *kernel_size,
        ), )
        self.weight_g = nn.Parameter(torch.empty(output_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.empty(output_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
        with torch.no_grad():
            norm = torch.linalg.vector_norm(
                self.weight_v.float(),
                dim=(1, 2, 3),
                keepdim=True,
            )
            self.weight_g.copy_(norm.to(dtype=self.weight_g.dtype))
            fan_in = (self.weight_v.shape[1] * self.weight_v.shape[2] * self.weight_v.shape[3])
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        norm = torch.linalg.vector_norm(
            self.weight_v.float(),
            dim=(1, 2, 3),
            keepdim=True,
        ).clamp_min(torch.finfo(torch.float32).tiny)
        weight = self.weight_v * (self.weight_g.float() / norm).to(dtype=self.weight_v.dtype)
        return functional.conv2d(
            inputs,
            weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class VitsPeriodDiscriminator(nn.Module):
    """One periodic 2-D discriminator from the original VITS objective."""

    def __init__(
            self,
            period: int,
            *,
            kernel_size: int = 5,
            stride: int = 3,
            channels: tuple[int, ...] = (32, 128, 512, 1024, 1024),
            leaky_relu_slope: float = 0.1,
    ) -> None:
        super().__init__()
        if isinstance(period, bool) or not isinstance(period, int) or period < 2:
            raise ValueError("`period` must be an integer of at least two.")
        if len(channels) != 5 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                                     for value in channels):
            raise ValueError("`channels` must contain five positive integers.")
        for name, value in (("kernel_size", kernel_size), ("stride", stride)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"`{name}` must be a positive integer.")
        self.period = period
        self.leaky_relu_slope = _activation_slope(leaky_relu_slope)
        inputs = (1, *channels[:-1])
        strides = (stride, stride, stride, stride, 1)
        padding = ((kernel_size - 1) // 2, 0)
        self.convs = nn.ModuleList(
            WeightNormalizedConv2d(
                input_channels,
                output_channels,
                (kernel_size, 1),
                stride=(layer_stride, 1),
                padding=padding,
            ) for input_channels, output_channels, layer_stride in zip(
                inputs,
                channels,
                strides,
            ))
        self.conv_post = WeightNormalizedConv2d(
            channels[-1],
            1,
            (3, 1),
            padding=(1, 0),
        )

    def forward(self, waveform: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        waveform = _validate_waveform(waveform)
        batch, channels, length = waveform.shape
        remainder = length % self.period
        if remainder:
            padding = self.period - remainder
            if padding >= length:
                raise ValueError("Waveform is too short for reflective period padding.")
            waveform = functional.pad(
                waveform,
                (0, padding),
                mode="reflect",
            )
            length += padding
        hidden = waveform.reshape(
            batch,
            channels,
            length // self.period,
            self.period,
        )
        features = []
        for convolution in self.convs:
            hidden = functional.leaky_relu(
                convolution(hidden),
                self.leaky_relu_slope,
            )
            features.append(hidden)
        hidden = self.conv_post(hidden)
        features.append(hidden)
        return hidden.flatten(1), tuple(features)


class VitsScaleDiscriminator(nn.Module):
    """One waveform-scale discriminator from the original VITS objective."""

    def __init__(
            self,
            *,
            channels: tuple[int, ...] = (16, 64, 256, 1024, 1024, 1024),
            groups: tuple[int, ...] = (1, 4, 16, 64, 256, 1),
            leaky_relu_slope: float = 0.1,
    ) -> None:
        super().__init__()
        if len(channels) != 6 or len(groups) != 6:
            raise ValueError("Scale discriminator needs six channel/group values.")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (*channels, *groups)):
            raise TypeError("Scale discriminator channels and groups must be integers.")
        input_channels = (1, *channels[:-1])
        kernel_sizes = (15, 41, 41, 41, 41, 5)
        strides = (1, 4, 4, 4, 4, 1)
        layers = []
        for index, (
                in_channels,
                out_channels,
                kernel,
                stride,
                group_count,
        ) in enumerate(zip(
                input_channels,
                channels,
                kernel_sizes,
                strides,
                groups,
        ), ):
            if (out_channels < 1 or group_count < 1 or in_channels % group_count or
                    out_channels % group_count):
                raise ValueError(f"Invalid grouped convolution at scale layer {index}.")
            layers.append(
                WeightNormalizedConv1d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride=stride,
                    padding=(kernel - 1) // 2,
                    groups=group_count,
                ), )
        self.convs = nn.ModuleList(layers)
        self.conv_post = WeightNormalizedConv1d(
            channels[-1],
            1,
            3,
            padding=1,
        )
        self.leaky_relu_slope = _activation_slope(leaky_relu_slope)

    def forward(self, waveform: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        hidden = _validate_waveform(waveform)
        features = []
        for convolution in self.convs:
            hidden = functional.leaky_relu(
                convolution(hidden),
                self.leaky_relu_slope,
            )
            features.append(hidden)
        hidden = self.conv_post(hidden)
        features.append(hidden)
        return hidden.flatten(1), tuple(features)


class VitsMultiPeriodDiscriminator(nn.Module):
    """Original scale-plus-five-period discriminator ensemble."""

    def __init__(
            self,
            *,
            periods: tuple[int, ...] = (2, 3, 5, 7, 11),
    ) -> None:
        super().__init__()
        if (not periods or any(isinstance(period, bool) or not isinstance(period, int) or period < 2
                               for period in periods) or len(set(periods)) != len(periods)):
            raise ValueError("`periods` must contain distinct values.")
        self.discriminators = nn.ModuleList([VitsScaleDiscriminator()] +
                                            [VitsPeriodDiscriminator(period) for period in periods], )

    def forward(
        self,
        real_waveform: Tensor,
        generated_waveform: Tensor,
    ) -> tuple[
            tuple[Tensor, ...],
            tuple[Tensor, ...],
            tuple[tuple[Tensor, ...], ...],
            tuple[tuple[Tensor, ...], ...],
    ]:
        real_waveform = _validate_waveform(real_waveform)
        generated_waveform = _validate_waveform(generated_waveform)
        if real_waveform.shape != generated_waveform.shape:
            raise ValueError("Real and generated waveforms must have equal shapes.")
        real_outputs = []
        generated_outputs = []
        real_features = []
        generated_features = []
        for discriminator in self.discriminators:
            real_output, real_feature = discriminator(real_waveform)
            generated_output, generated_feature = discriminator(generated_waveform, )
            real_outputs.append(real_output)
            generated_outputs.append(generated_output)
            real_features.append(real_feature)
            generated_features.append(generated_feature)
        return (
            tuple(real_outputs),
            tuple(generated_outputs),
            tuple(real_features),
            tuple(generated_features),
        )


def _validate_waveform(value: Tensor) -> Tensor:
    value = _finite_float_tensor("waveform", value)
    if value.ndim == 2:
        value = value.unsqueeze(1)
    if (value.ndim != 3 or value.shape[0] < 1 or value.shape[1] != 1 or value.shape[2] < 1):
        raise ValueError("Waveforms must have shape [batch, samples] or [batch, 1, samples].")
    return value


@dataclass(frozen=True, slots=True)
class VitsGeneratorLossOutput:
    """Named generator loss terms for logging and optimization."""

    total: Tensor
    duration: Tensor
    mel_reconstruction: Tensor
    kl_divergence: Tensor
    feature_matching: Tensor
    adversarial: Tensor


class VitsGeneratorLoss(nn.Module):
    """Combine the original VITS generator terms.

    ``mel_reconstruction_loss`` is required rather than synthesized with
    guessed STFT settings.  This keeps fine-tuning faithful to each
    checkpoint's data recipe.
    """

    def __init__(
        self,
        *,
        mel_weight: float = 45.0,
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.mel_weight = _nonnegative_weight("mel_weight", mel_weight)
        self.kl_weight = _nonnegative_weight("kl_weight", kl_weight)

    def forward(
        self,
        output: VitsTrainingOutput,
        *,
        mel_reconstruction_loss: Tensor,
        generated_discriminator_outputs: Sequence[Tensor],
        real_feature_maps: FeatureMaps,
        generated_feature_maps: FeatureMaps,
    ) -> VitsGeneratorLossOutput:
        if not isinstance(output, VitsTrainingOutput):
            raise TypeError("`output` must be a VitsTrainingOutput.")
        duration = _finite_float_tensor(
            "output.duration_loss",
            output.duration_loss,
        )
        if duration.numel() != 1:
            raise ValueError("`output.duration_loss` must be scalar.")
        mel = _finite_float_tensor(
            "mel_reconstruction_loss",
            mel_reconstruction_loss,
        )
        if mel.numel() != 1:
            raise ValueError("`mel_reconstruction_loss` must be scalar.")
        if mel.device != duration.device:
            raise ValueError("Mel reconstruction and duration losses must share a device.")
        adversarial, _ = generator_adversarial_loss(generated_discriminator_outputs, )
        feature = feature_matching_loss(
            real_feature_maps,
            generated_feature_maps,
        )
        kl = vits_kl_loss(
            output.prior_latents,
            output.posterior_log_variances,
            output.expanded_prior_means,
            output.expanded_prior_log_variances,
            output.spectrogram_mask,
        )
        total = (duration + self.mel_weight * mel + self.kl_weight * kl + feature + adversarial)
        return VitsGeneratorLossOutput(
            total=total,
            duration=duration,
            mel_reconstruction=mel,
            kl_divergence=kl,
            feature_matching=feature,
            adversarial=adversarial,
        )


def _nonnegative_weight(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return normalized


def _activation_slope(value: Real) -> float:
    normalized = _nonnegative_weight("leaky_relu_slope", value)
    if normalized > 1.0:
        raise ValueError("`leaky_relu_slope` must not exceed one.")
    return normalized


@dataclass(frozen=True, slots=True)
class VitsTrainingSupport:
    """Auditable boundary between implemented and checkpoint-specific
    pieces."""

    differentiable_generator_graph: bool = True
    posterior_encoder: bool = True
    monotonic_alignment_search: bool = True
    stochastic_duration_objective: bool = True
    discriminator_architecture: bool = True
    source_adversarial_losses: bool = True
    source_acoustic_frontend: bool = True
    adversarial_optimizer_phases: bool = True
    random_discriminator_initialization: bool = True
    checkpoint_discriminator_weights: bool = False
    checkpoint_acoustic_frontend: bool = False
    full_finetuning_ready: bool = True
    blocking_requirements: tuple[str, ...] = (
        "provide the checkpoint-specific acoustic configuration explicitly",
        "provide licensed, consented waveform/text training data",
    )


VITS_TRAINING_SUPPORT = VitsTrainingSupport()

# Source-compatible spellings used by existing VITS recipes.
feature_loss = feature_matching_loss
generator_loss = generator_adversarial_loss
kl_loss = vits_kl_loss

__all__ = [
    "FeatureMaps",
    "VITS_TRAINING_SUPPORT",
    "VitsGeneratorLoss",
    "VitsGeneratorLossOutput",
    "VitsMultiPeriodDiscriminator",
    "VitsPeriodDiscriminator",
    "VitsScaleDiscriminator",
    "VitsTrainingSupport",
    "WeightNormalizedConv2d",
    "discriminator_loss",
    "feature_loss",
    "feature_matching_loss",
    "generator_adversarial_loss",
    "generator_loss",
    "kl_loss",
    "vits_kl_loss",
]
