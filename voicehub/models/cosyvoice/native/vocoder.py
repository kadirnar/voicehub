"""PyTorch-only causal HiFT vocoder and adversarial training graph."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # pragma: no cover - old PyTorch compatibility
    from torch.nn.utils import weight_norm

from voicehub.kernels.codecs import CodecSnakeKernelOptimizable
from voicehub.models.cosyvoice.native.configuration import CosyVoiceHiFTConfig
from voicehub.optimization.protocols import OptimizationCompileTarget


class CausalConv1d(nn.Conv1d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        *,
        causal_type: str = "left",
        device=None,
        dtype=None,
    ) -> None:
        if stride != 1:
            raise ValueError("CausalConv1d supports stride one.")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            device=device,
            dtype=dtype,
        )
        self.causal_padding = (((kernel_size * dilation - dilation) // 2) * 2 + (kernel_size + 1) % 2)
        if causal_type not in {"left", "right"}:
            raise ValueError("`causal_type` must be left or right.")
        self.causal_type = causal_type

    def forward(self, values: Tensor, cache: Tensor | None = None) -> Tensor:
        if cache is None:
            cache = values.new_zeros(
                values.shape[0],
                values.shape[1],
                self.causal_padding,
            )
        if cache.shape[-1] != self.causal_padding:
            raise ValueError("Causal convolution cache length is invalid.")
        values = (
            torch.cat((cache, values), dim=-1) if self.causal_type == "left" else torch.cat(
                (values, cache), dim=-1))
        return super().forward(values)


class CausalConv1dDownSample(nn.Conv1d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        if stride == 1 or kernel_size % stride:
            raise ValueError("Downsample kernel must be a multiple of a non-unit stride.")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
        )
        self.causal_padding = stride - 1

    def forward(self, values: Tensor, cache: Tensor | None = None) -> Tensor:
        if cache is None:
            values = functional.pad(values, (self.causal_padding, 0))
        else:
            if cache.shape[-1] != self.causal_padding:
                raise ValueError("Downsample cache length is invalid.")
            values = torch.cat((cache, values), dim=-1)
        return super().forward(values)


class CausalConv1dUpsample(nn.Conv1d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
        )
        self.causal_padding = kernel_size - 1
        self.upsample = nn.Upsample(scale_factor=stride, mode="nearest")

    def forward(self, values: Tensor, cache: Tensor | None = None) -> Tensor:
        values = self.upsample(values)
        if cache is None:
            values = functional.pad(values, (self.causal_padding, 0))
        else:
            if cache.shape[-1] != self.causal_padding:
                raise ValueError("Upsample cache length is invalid.")
            values = torch.cat((cache, values), dim=-1)
        return super().forward(values)


class Snake(CodecSnakeKernelOptimizable, nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))
        self._initialize_codec_kernel_backend()

    def forward(self, values: Tensor) -> Tensor:
        alpha = self.alpha[None, :, None]
        return self._codec_snake(values, alpha)


class ResBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            weight_norm(CausalConv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
            )) for dilation in dilations)
        self.convs2 = nn.ModuleList(
            weight_norm(CausalConv1d(channels, channels, kernel_size)) for _ in dilations)
        self.activations1 = nn.ModuleList(Snake(channels) for _ in dilations)
        self.activations2 = nn.ModuleList(Snake(channels) for _ in dilations)

    def forward(self, values: Tensor) -> Tensor:
        for conv1, conv2, activation1, activation2 in zip(
                self.convs1,
                self.convs2,
                self.activations1,
                self.activations2,
        ):
            residual = conv2(activation2(conv1(activation1(values))))
            values = values + residual
        return values


class SineGenerator(nn.Module):
    """Harmonic source using only PyTorch interpolation and phase math."""

    def __init__(
        self,
        sample_rate: int,
        upsample_scale: int,
        harmonics: int,
        *,
        sine_amplitude: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 10.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.upsample_scale = upsample_scale
        self.harmonics = harmonics
        self.sine_amplitude = sine_amplitude
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    def forward(self, f0: Tensor) -> tuple[Tensor, Tensor]:
        multipliers = torch.arange(
            1,
            self.harmonics + 2,
            device=f0.device,
            dtype=f0.dtype,
        )
        frequencies = f0 * multipliers[None, None, :]
        radians = (frequencies / self.sample_rate) % 1
        radians = functional.interpolate(
            radians.transpose(1, 2),
            scale_factor=1 / self.upsample_scale,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        phases = torch.cumsum(radians, dim=1) * (2 * math.pi)
        phases = functional.interpolate(
            phases.transpose(1, 2) * self.upsample_scale,
            scale_factor=self.upsample_scale,
            mode="nearest",
        ).transpose(1, 2)
        sines = torch.sin(phases) * self.sine_amplitude
        voiced = (f0 > self.voiced_threshold).to(f0.dtype)
        noise_scale = (voiced * self.noise_std + (1 - voiced) * self.sine_amplitude / 3)
        sines = sines * voiced + noise_scale * torch.randn_like(sines)
        return sines, voiced


class SourceModuleHnNSF(nn.Module):

    def __init__(self, config: CosyVoiceHiFTConfig) -> None:
        super().__init__()
        self.sine_amplitude = 0.1
        self.l_sin_gen = SineGenerator(
            config.sample_rate,
            config.samples_per_mel_frame,
            config.harmonics,
        )
        self.l_linear = nn.Linear(config.harmonics + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0: Tensor) -> Tensor:
        with torch.no_grad():
            sines, _ = self.l_sin_gen(f0)
        return self.l_tanh(self.l_linear(sines))


class CausalConvF0Predictor(nn.Module):

    def __init__(self, config: CosyVoiceHiFTConfig) -> None:
        super().__init__()
        hidden = config.f0_hidden_size
        layers: list[nn.Module] = []
        for index in range(5):
            input_channels = config.mel_channels if index == 0 else hidden
            kernel_size = 4 if index == 0 else 3
            causal_type = "right" if index == 0 else "left"
            layers.extend((
                weight_norm(CausalConv1d(
                    input_channels,
                    hidden,
                    kernel_size,
                    causal_type=causal_type,
                )),
                nn.ELU(),
            ))
        self.condnet = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, values: Tensor) -> Tensor:
        values = self.condnet(values).transpose(1, 2)
        return self.classifier(values).squeeze(-1).abs()


class CosyVoiceHiFTGenerator(nn.Module):
    """Checkpoint-shaped causal HiFT neural source-filter generator."""

    def __init__(self, config: CosyVoiceHiFTConfig) -> None:
        super().__init__()
        if not isinstance(config, CosyVoiceHiFTConfig):
            raise TypeError("`config` must be CosyVoiceHiFTConfig.")
        self.config = config
        self.m_source = SourceModuleHnNSF(config)
        self.f0_upsamp = nn.Upsample(scale_factor=config.samples_per_mel_frame, )
        self.conv_pre = weight_norm(
            CausalConv1d(
                config.mel_channels,
                config.base_channels,
                config.conv_pre_look_right + 1,
                causal_type="right",
            ))
        self.ups = nn.ModuleList()
        for index, (rate, kernel) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    CausalConv1dUpsample(
                        config.base_channels // (2**index),
                        config.base_channels // (2**(index + 1)),
                        kernel,
                        rate,
                    )))

        reversed_rates = (1, ) + tuple(reversed(config.upsample_rates))[:-1]
        cumulative: list[int] = []
        running = 1
        for rate in reversed_rates:
            running *= rate
            cumulative.append(running)
        cumulative.reverse()
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        source_channels = config.istft_n_fft + 2
        for index, (downsample, kernel, dilations) in enumerate(zip(
                cumulative,
                config.source_resblock_kernel_sizes,
                config.source_resblock_dilations,
        )):
            channels = config.base_channels // (2**(index + 1))
            if downsample == 1:
                down = CausalConv1d(
                    source_channels,
                    channels,
                    1,
                )
            else:
                down = CausalConv1dDownSample(
                    source_channels,
                    channels,
                    downsample * 2,
                    downsample,
                )
            self.source_downs.append(down)
            self.source_resblocks.append(ResBlock(channels, kernel, dilations))

        self.resblocks = nn.ModuleList()
        for stage in range(len(config.upsample_rates)):
            channels = config.base_channels // (2**(stage + 1))
            for kernel, dilations in zip(
                    config.resblock_kernel_sizes,
                    config.resblock_dilations,
            ):
                self.resblocks.append(ResBlock(channels, kernel, dilations))
        final_channels = config.base_channels // (2**len(config.upsample_rates))
        self.conv_post = weight_norm(CausalConv1d(
            final_channels,
            config.istft_n_fft + 2,
            7,
        ))
        self.f0_predictor = CausalConvF0Predictor(config)
        self.register_buffer(
            "stft_window",
            torch.hann_window(config.istft_n_fft),
            persistent=False,
        )

    def codec_optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        if mode not in {"inference", "training"}:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (
            OptimizationCompileTarget(
                "codec.vocoder.cosyvoice_hift.forward",
                self,
                "forward",
                component="vocoder",
            ), )

    def _stft(self, source: Tensor) -> Tensor:
        complex_spectrum = torch.stft(
            source,
            self.config.istft_n_fft,
            self.config.istft_hop_length,
            self.config.istft_n_fft,
            window=self.stft_window.to(source),
            return_complex=True,
        )
        return torch.cat(
            (complex_spectrum.real, complex_spectrum.imag),
            dim=1,
        )

    def _istft(self, values: Tensor) -> Tensor:
        bins = self.config.istft_n_fft // 2 + 1
        magnitude = values[:, :bins].clamp_max(100).exp()
        phase = values[:, bins:].sin()
        spectrum = torch.polar(magnitude, phase)
        return torch.istft(
            spectrum,
            self.config.istft_n_fft,
            self.config.istft_hop_length,
            self.config.istft_n_fft,
            window=self.stft_window.to(values),
        )

    def forward(self, speech_features: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(speech_features, Tensor) or speech_features.ndim != 3:
            raise ValueError("HiFT input must have shape [batch, mel, frames].")
        if speech_features.shape[1] != self.config.mel_channels:
            raise ValueError("HiFT mel channel count is invalid.")
        f0 = self.f0_predictor(speech_features)
        source_f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        source = self.m_source(source_f0).transpose(1, 2)
        source_spectrum = self._stft(source.squeeze(1))

        values = self.conv_pre(speech_features)
        for stage, upsample in enumerate(self.ups):
            values = upsample(functional.leaky_relu(values, 0.1))
            if stage == len(self.ups) - 1:
                values = functional.pad(values, (1, 0), mode="reflect")
            source_value = self.source_downs[stage](source_spectrum)
            source_value = self.source_resblocks[stage](source_value)
            common = min(values.shape[-1], source_value.shape[-1])
            values = values[..., :common] + source_value[..., :common]
            candidates = [
                block(values)
                for block in self.resblocks[stage * len(self.config.resblock_kernel_sizes):(stage + 1) *
                                            len(self.config.resblock_kernel_sizes)]
            ]
            values = torch.stack(candidates).mean(dim=0)
        values = self.conv_post(functional.leaky_relu(values, 0.1))
        waveform = self._istft(values).clamp(
            -self.config.audio_limit,
            self.config.audio_limit,
        )
        return waveform, f0


class PeriodDiscriminator(nn.Module):

    def __init__(
        self,
        period: int,
        *,
        channel_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.period = period
        channels = [max(4, int(value * channel_multiplier)) for value in (32, 128, 512, 1_024, 1_024)]
        inputs = (1, *channels[:-1])
        strides = (3, 3, 3, 3, 1)
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(
                input_channel,
                output_channel,
                (5, 1),
                (stride, 1),
                padding=(2, 0),
            )) for input_channel, output_channel, stride in zip(
                inputs,
                channels,
                strides,
            ))
        self.conv_post = weight_norm(nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, values: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        remainder = values.shape[-1] % self.period
        if remainder:
            values = functional.pad(
                values,
                (0, self.period - remainder),
                mode="reflect",
            )
        values = values.view(
            values.shape[0],
            values.shape[1],
            -1,
            self.period,
        )
        features = []
        for layer in self.convs:
            values = functional.leaky_relu(layer(values), 0.1)
            features.append(values)
        values = self.conv_post(values)
        features.append(values)
        return values.flatten(1), tuple(features)


class MultiPeriodDiscriminator(nn.Module):

    def __init__(self, *, channel_multiplier: float = 1.0) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            PeriodDiscriminator(
                period,
                channel_multiplier=channel_multiplier,
            ) for period in (2, 3, 5, 7, 11))

    def score(self, waveform: Tensor):
        return tuple(discriminator(waveform) for discriminator in self.discriminators)

    def forward(self, real: Tensor, generated: Tensor):
        return self.score(real), self.score(generated)


class ResolutionDiscriminator(nn.Module):

    def __init__(
        self,
        n_fft: int,
        *,
        channels: int = 32,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.register_buffer(
            "window",
            torch.hann_window(n_fft),
            persistent=False,
        )
        channel_sizes = (2, channels, channels, channels, channels)
        self.convs = nn.ModuleList(
            weight_norm(
                nn.Conv2d(
                    channel_sizes[index],
                    channel_sizes[index + 1],
                    (3, 9),
                    stride=(1, 1 if index == 0 else 2),
                    padding=(1, 4),
                )) for index in range(len(channel_sizes) - 1))
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, 3, padding=1))

    def forward(self, waveform: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        waveform = waveform.squeeze(1)
        spectrum = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window=self.window.to(waveform),
            return_complex=True,
        )
        values = torch.view_as_real(spectrum).permute(0, 3, 2, 1)
        features = []
        for layer in self.convs:
            values = functional.leaky_relu(layer(values), 0.1)
            features.append(values)
        values = self.conv_post(values)
        features.append(values)
        return values.flatten(1), tuple(features)


class CosyVoiceHiFTDiscriminator(nn.Module):

    def __init__(self, *, tiny: bool = False) -> None:
        super().__init__()
        multiplier = 0.125 if tiny else 1.0
        self.mpd = MultiPeriodDiscriminator(channel_multiplier=multiplier)
        channels = 8 if tiny else 32
        resolutions = (64, 32) if tiny else (2_048, 1_024, 512)
        self.mrd = nn.ModuleList(ResolutionDiscriminator(size, channels=channels) for size in resolutions)

    def score(self, waveform: Tensor):
        period = self.mpd.score(waveform)
        resolution = tuple(model(waveform) for model in self.mrd)
        return period + resolution

    def forward(self, real: Tensor, generated: Tensor):
        return self.score(real), self.score(generated)


def _generator_adversarial_loss(scores) -> Tensor:
    return sum((1 - score).square().mean() for score, _ in scores)


def _discriminator_loss(real_scores, generated_scores) -> Tensor:
    loss = real_scores[0][0].new_zeros(())
    for (real, _), (generated, _) in zip(real_scores, generated_scores):
        loss = loss + (1 - real).square().mean() + generated.square().mean()
    return loss


def _feature_matching_loss(real_scores, generated_scores) -> Tensor:
    loss = real_scores[0][0].new_zeros(())
    for (_, real_features), (_, generated_features) in zip(
            real_scores,
            generated_scores,
    ):
        for real_feature, generated_feature in zip(
                real_features,
                generated_features,
        ):
            loss = loss + (real_feature.detach() - generated_feature).abs().mean()
    return loss


def _spectral_reconstruction_loss(real: Tensor, generated: Tensor) -> Tensor:
    losses = []
    maximum = min(real.shape[-1], generated.shape[-1])
    real = real[..., :maximum].squeeze(1)
    generated = generated[..., :maximum].squeeze(1)
    for n_fft in (64, 128, 256):
        if maximum < n_fft:
            continue
        window = torch.hann_window(n_fft, device=real.device, dtype=real.dtype)
        real_spectrum = torch.stft(
            real,
            n_fft,
            n_fft // 4,
            n_fft,
            window=window,
            return_complex=True,
        ).abs()
        generated_spectrum = torch.stft(
            generated,
            n_fft,
            n_fft // 4,
            n_fft,
            window=window,
            return_complex=True,
        ).abs()
        losses.append(
            (real_spectrum.clamp_min(1e-5).log() - generated_spectrum.clamp_min(1e-5).log()).abs().mean())
    if not losses:
        return (real - generated).abs().mean()
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class CosyVoiceHiFTTrainingOutput:
    loss: Tensor
    generated_waveform: Tensor
    losses: dict[str, Tensor]
    phase: str


class CosyVoiceHiFTTrainingModel(nn.Module):
    """Author-style alternating generator/discriminator objectives."""

    def __init__(
        self,
        generator: CosyVoiceHiFTGenerator,
        discriminator: CosyVoiceHiFTDiscriminator,
        *,
        feature_weight: float = 2.0,
        spectral_weight: float = 45.0,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.feature_weight = feature_weight
        self.spectral_weight = spectral_weight

    def forward(
        self,
        *,
        speech_features: Tensor,
        waveform: Tensor,
        pitch: Tensor,
        phase: str,
    ) -> CosyVoiceHiFTTrainingOutput:
        if phase not in {"generator", "discriminator"}:
            raise ValueError("HiFT phase must be generator or discriminator.")
        generated, predicted_pitch = self.generator(speech_features)
        if waveform.ndim == 2:
            waveform = waveform[:, None]
        generated_channel = generated[:, None]
        maximum = min(waveform.shape[-1], generated_channel.shape[-1])
        waveform = waveform[..., :maximum]
        generated_channel = generated_channel[..., :maximum]
        if phase == "discriminator":
            real_scores, generated_scores = self.discriminator(
                waveform,
                generated_channel.detach(),
            )
            discriminator_loss = _discriminator_loss(
                real_scores,
                generated_scores,
            )
            return CosyVoiceHiFTTrainingOutput(
                loss=discriminator_loss,
                generated_waveform=generated,
                losses={"discriminator_loss": discriminator_loss},
                phase=phase,
            )
        real_scores, generated_scores = self.discriminator(
            waveform,
            generated_channel,
        )
        adversarial = _generator_adversarial_loss(generated_scores)
        features = _feature_matching_loss(real_scores, generated_scores)
        spectral = _spectral_reconstruction_loss(
            waveform,
            generated_channel,
        )
        pitch_loss = functional.l1_loss(predicted_pitch, pitch)
        loss = (adversarial + self.feature_weight * features + self.spectral_weight * spectral + pitch_loss)
        return CosyVoiceHiFTTrainingOutput(
            loss=loss,
            generated_waveform=generated,
            losses={
                "adversarial_loss": adversarial,
                "feature_matching_loss": features,
                "pitch_loss": pitch_loss,
                "spectral_reconstruction_loss": spectral,
            },
            phase=phase,
        )


__all__ = [
    "CosyVoiceHiFTDiscriminator",
    "CosyVoiceHiFTGenerator",
    "CosyVoiceHiFTTrainingModel",
    "CosyVoiceHiFTTrainingOutput",
]
