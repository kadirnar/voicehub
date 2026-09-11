"""Native acoustic frontend and adversarial recipe for VITS.

The preprocessing equations, segment alignment, discriminator ordering,
and loss composition follow the original MIT-licensed VITS
implementation at revision ``2e561ba58618d021b5b8323d3765880f7e0ecfdb``.
The implementation is PyTorch-only: it reproduces librosa's default
Slaney mel bank without importing librosa or NumPy.

MMS-TTS checkpoint metadata does not publish the training FFT, window,
mel, or segment settings.  Consequently :class:`VitsAcousticConfig` is
always an explicit recipe input; this module never guesses those
checkpoint-specific values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.vits.native.losses import VitsGeneratorLoss, VitsMultiPeriodDiscriminator, discriminator_loss
from voicehub.models.vits.native.modeling import VitsModel, VitsTrainingOutput


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an integer.")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"`{name}` must be positive.")
    return normalized


def _finite_real(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"`{name}` must be finite.")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}.")
    return normalized


@dataclass(frozen=True, slots=True)
class VitsAcousticConfig:
    """Exact acoustic settings required by a VITS training checkpoint."""

    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    num_mel_channels: int
    mel_fmin: float = 0.0
    mel_fmax: float | None = None
    segment_size: int | None = None

    def __post_init__(self) -> None:
        for name in (
                "sampling_rate",
                "filter_length",
                "hop_length",
                "win_length",
                "num_mel_channels",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(name, getattr(self, name)),
            )
        if self.win_length > self.filter_length:
            raise ValueError("`win_length` cannot exceed `filter_length`.")
        if self.hop_length > self.win_length:
            raise ValueError("`hop_length` cannot exceed `win_length`.")
        if self.num_mel_channels > self.filter_length // 2 + 1:
            raise ValueError("`num_mel_channels` cannot exceed the one-sided FFT bin count.")

        object.__setattr__(
            self,
            "mel_fmin",
            _finite_real("mel_fmin", self.mel_fmin, minimum=0.0),
        )
        nyquist = self.sampling_rate / 2.0
        mel_fmax = nyquist if self.mel_fmax is None else _finite_real(
            "mel_fmax",
            self.mel_fmax,
            minimum=0.0,
        )
        if not self.mel_fmin < mel_fmax <= nyquist:
            raise ValueError(
                "`mel_fmax` must be greater than `mel_fmin` and no greater "
                "than the Nyquist frequency.")
        object.__setattr__(self, "mel_fmax", mel_fmax)

        if self.segment_size is not None:
            segment_size = _positive_integer(
                "segment_size",
                self.segment_size,
            )
            if segment_size % self.hop_length:
                raise ValueError(
                    "`segment_size` must be divisible by `hop_length` so "
                    "waveform and spectrogram slices remain aligned.")
            object.__setattr__(self, "segment_size", segment_size)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | VitsAcousticConfig,
    ) -> VitsAcousticConfig:
        """Validate a source-style VITS ``data``/``train`` mapping."""
        if isinstance(values, cls):
            return values
        if not isinstance(values, Mapping):
            raise TypeError("VITS acoustic configuration must be a mapping.")
        source = dict(values)
        aliases = {
            "n_fft": "filter_length",
            "n_mels": "num_mel_channels",
            "sample_rate": "sampling_rate",
        }
        for alias, canonical in aliases.items():
            if alias in source:
                if canonical in source:
                    raise ValueError(f"Pass `{canonical}` or its `{alias}` alias, not both.")
                source[canonical] = source.pop(alias)
        allowed = {
            "sampling_rate",
            "filter_length",
            "hop_length",
            "win_length",
            "num_mel_channels",
            "mel_fmin",
            "mel_fmax",
            "segment_size",
        }
        unexpected = tuple(sorted(set(source).difference(allowed)))
        if unexpected:
            raise ValueError("Unsupported VITS acoustic setting(s): " + ", ".join(unexpected))
        missing = tuple(
            name for name in (
                "sampling_rate",
                "filter_length",
                "hop_length",
                "win_length",
                "num_mel_channels",
            ) if name not in source)
        if missing:
            raise ValueError("VITS acoustic configuration is incomplete; missing: " + ", ".join(missing))
        return cls(**source)

    @property
    def spectrogram_bins(self) -> int:
        return self.filter_length // 2 + 1

    @property
    def segment_frames(self) -> int | None:
        if self.segment_size is None:
            return None
        return self.segment_size // self.hop_length

    def validate_model(self, model: VitsModel) -> None:
        """Reject an acoustic recipe that cannot align with ``model``."""
        if not isinstance(model, VitsModel):
            raise TypeError("VITS adversarial training requires a native VitsModel.")
        config = model.config
        if self.sampling_rate != config.sampling_rate:
            raise ValueError(
                "VITS acoustic sampling rate does not match the checkpoint: "
                f"{self.sampling_rate} != {config.sampling_rate}.")
        if self.spectrogram_bins != config.spectrogram_bins:
            raise ValueError(
                "VITS FFT size does not match the checkpoint spectrogram "
                f"width: {self.spectrogram_bins} != "
                f"{config.spectrogram_bins}.")
        if self.hop_length != config.upsample_factor:
            raise ValueError(
                "VITS acoustic hop length must equal the decoder upsample "
                f"factor: {self.hop_length} != {config.upsample_factor}.")

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "sampling_rate": self.sampling_rate,
            "filter_length": self.filter_length,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "num_mel_channels": self.num_mel_channels,
            "mel_fmin": self.mel_fmin,
            "mel_fmax": self.mel_fmax,
            "segment_size": self.segment_size,
        }


def _hz_to_mel(frequencies: Tensor) -> Tensor:
    """Match librosa's default Slaney frequency conversion."""
    frequencies = frequencies.to(dtype=torch.float64)
    linear_scale = 200.0 / 3.0
    mels = frequencies / linear_scale
    logarithmic = frequencies >= 1_000.0
    log_step = math.log(6.4) / 27.0
    logarithmic_mels = 15.0 + torch.log(
        frequencies.clamp_min(torch.finfo(torch.float64).tiny) / 1_000.0) / log_step
    return torch.where(logarithmic, logarithmic_mels, mels)


def _mel_to_hz(mels: Tensor) -> Tensor:
    """Match librosa's default Slaney inverse frequency conversion."""
    mels = mels.to(dtype=torch.float64)
    linear_scale = 200.0 / 3.0
    frequencies = linear_scale * mels
    logarithmic = mels >= 15.0
    log_step = math.log(6.4) / 27.0
    logarithmic_hz = 1_000.0 * torch.exp(log_step * (mels - 15.0))
    return torch.where(logarithmic, logarithmic_hz, frequencies)


def build_slaney_mel_filter(config: VitsAcousticConfig) -> Tensor:
    """Build the normalized triangular mel bank used by original VITS."""
    if not isinstance(config, VitsAcousticConfig):
        raise TypeError("`config` must be a VitsAcousticConfig.")
    fft_frequencies = torch.linspace(
        0.0,
        config.sampling_rate / 2.0,
        config.spectrogram_bins,
        dtype=torch.float64,
    )
    minimum_mel = _hz_to_mel(torch.tensor(config.mel_fmin))
    maximum_mel = _hz_to_mel(torch.tensor(config.mel_fmax))
    mel_frequencies = _mel_to_hz(
        torch.linspace(
            float(minimum_mel),
            float(maximum_mel),
            config.num_mel_channels + 2,
            dtype=torch.float64,
        ))
    frequency_differences = torch.diff(mel_frequencies)
    ramps = mel_frequencies.unsqueeze(1) - fft_frequencies.unsqueeze(0)
    lower = -ramps[:-2] / frequency_differences[:-1].unsqueeze(1)
    upper = ramps[2:] / frequency_differences[1:].unsqueeze(1)
    weights = torch.minimum(lower, upper).clamp_min(0.0)
    # librosa defaults to Slaney area normalization.
    weights *= (2.0 / (mel_frequencies[2:] - mel_frequencies[:-2])).unsqueeze(1)
    return weights.to(dtype=torch.float32)


class VitsAcousticFrontend(nn.Module):
    """Differentiable source-equivalent spectrogram and mel transforms."""

    def __init__(
        self,
        config: VitsAcousticConfig | Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.config = VitsAcousticConfig.from_mapping(config)
        self.register_buffer(
            "mel_filter",
            build_slaney_mel_filter(self.config),
            persistent=False,
        )
        self.register_buffer(
            "window",
            torch.hann_window(self.config.win_length),
            persistent=False,
        )

    @staticmethod
    def _waveform(value: Tensor) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError("VITS waveform values must be PyTorch tensors.")
        if value.ndim == 3 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
            raise ValueError(
                "VITS waveform values must have shape [batch, samples] or "
                "[batch, 1, samples].")
        if not value.is_floating_point():
            value = value.float()
        if not torch.isfinite(value).all():
            raise ValueError("VITS waveform values cannot contain NaN or infinity.")
        return value

    def spectrogram(self, waveform: Tensor) -> Tensor:
        """Return the original VITS magnitude spectrogram."""
        waveform = self._waveform(waveform)
        # FFT and logarithmic reconstruction remain float32 under mixed
        # precision. The cast is differentiable and avoids unsupported CPU
        # half/bfloat16 FFT kernels.
        analysis_waveform = waveform.float()
        padding = (self.config.filter_length - self.config.hop_length) // 2
        if padding >= analysis_waveform.shape[-1]:
            raise ValueError(
                "VITS waveform is too short for the configured reflective "
                f"STFT padding ({analysis_waveform.shape[-1]} <= {padding}).")
        padded = functional.pad(
            analysis_waveform.unsqueeze(1),
            (padding, padding),
            mode="reflect",
        ).squeeze(1)
        spectrum = torch.stft(
            padded,
            n_fft=self.config.filter_length,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window.to(
                device=analysis_waveform.device,
                dtype=analysis_waveform.dtype,
            ),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return torch.sqrt(spectrum.abs().square() + 1e-6)

    def spectrogram_lengths(self, audio_lengths: Tensor) -> Tensor:
        """Map unpadded waveform lengths to source STFT frame counts."""
        if not isinstance(audio_lengths, Tensor):
            raise TypeError("`audio_lengths` must be a PyTorch tensor.")
        if audio_lengths.dtype == torch.bool or audio_lengths.is_floating_point():
            raise TypeError("`audio_lengths` must use an integer dtype.")
        padding = (self.config.filter_length - self.config.hop_length) // 2
        numerator = (audio_lengths.long() + 2 * padding - self.config.filter_length)
        lengths = torch.div(
            numerator,
            self.config.hop_length,
            rounding_mode="floor",
        ) + 1
        if (lengths < 1).any():
            raise ValueError("Every VITS waveform must produce at least one STFT frame.")
        return lengths

    def spectrogram_to_mel(self, spectrogram: Tensor) -> Tensor:
        """Project magnitudes through the source Slaney bank and log scale."""
        if (not isinstance(spectrogram, Tensor) or spectrogram.ndim != 3 or
                spectrogram.shape[1] != self.config.spectrogram_bins):
            raise ValueError(
                "VITS spectrogram must have shape "
                f"[batch, {self.config.spectrogram_bins}, frames].")
        if not spectrogram.is_floating_point():
            raise TypeError("VITS spectrogram must use a floating-point dtype.")
        if not torch.isfinite(spectrogram).all():
            raise ValueError("VITS spectrogram cannot contain NaN or infinity.")
        analysis_spectrogram = spectrogram.float()
        mel_filter = self.mel_filter.to(
            device=analysis_spectrogram.device,
            dtype=analysis_spectrogram.dtype,
        )
        mel = torch.matmul(mel_filter, analysis_spectrogram)
        return torch.log(mel.clamp_min(1e-5))

    def mel_spectrogram(self, waveform: Tensor) -> Tensor:
        return self.spectrogram_to_mel(self.spectrogram(waveform))


@dataclass(frozen=True, slots=True)
class _AdversarialBatch:
    output: VitsTrainingOutput
    real_waveform: Tensor
    generated_waveform: Tensor
    target_mel: Tensor


class VitsAdversarialTrainingModel(nn.Module):
    """Own the generator, MPD, and exact two-optimizer VITS objectives."""

    def __init__(
        self,
        native_model: VitsModel,
        acoustic_config: VitsAcousticConfig | Mapping[str, Any],
        *,
        discriminator: nn.Module | None = None,
        mel_weight: float = 45.0,
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(native_model, VitsModel):
            raise TypeError("`native_model` must be a native VitsModel.")
        self.native_model = native_model
        self.acoustic_frontend = VitsAcousticFrontend(acoustic_config)
        self.acoustic_frontend.config.validate_model(native_model)
        self.discriminator = (VitsMultiPeriodDiscriminator() if discriminator is None else discriminator)
        if not isinstance(self.discriminator, nn.Module):
            raise TypeError("`discriminator` must be a PyTorch module.")
        self.generator_objective = VitsGeneratorLoss(
            mel_weight=mel_weight,
            kl_weight=kl_weight,
        )

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Default to the generator phase for direct module use."""
        return self.generator_step(*args, **kwargs)

    @staticmethod
    def _integer_lengths(
        value: Any,
        *,
        batch_size: int,
        maximum: int,
        device: torch.device,
        name: str,
    ) -> Tensor:
        if value is None:
            return torch.full(
                (batch_size, ),
                maximum,
                dtype=torch.long,
                device=device,
            )
        lengths = (value if isinstance(value, Tensor) else torch.as_tensor(value, device=device))
        lengths = lengths.to(device=device)
        if lengths.ndim == 0:
            lengths = lengths.expand(batch_size)
        if tuple(lengths.shape) != (batch_size, ):
            raise ValueError(f"`{name}` must have shape [batch].")
        if lengths.dtype == torch.bool or lengths.is_floating_point():
            raise TypeError(f"`{name}` must use an integer dtype.")
        lengths = lengths.long()
        if ((lengths < 1) | (lengths > maximum)).any():
            raise ValueError(f"`{name}` values must be in the interval [1, {maximum}].")
        return lengths

    def _training_batch(
        self,
        input_ids: Any,
        *,
        audio_values: Any,
        spectrogram: Any = None,
        attention_mask: Any = None,
        spectrogram_attention_mask: Any = None,
        durations: Any = None,
        speaker_id: Any = None,
        audio_lengths: Any = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> _AdversarialBatch:
        del kwargs
        if not isinstance(input_ids, Tensor):
            input_ids = torch.as_tensor(input_ids)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError("`input_ids` must have shape [batch, text].")
        model_device = self.native_model.text_encoder.embed_tokens.weight.device
        input_ids = input_ids.to(device=model_device, dtype=torch.long)

        real_waveform = (audio_values if isinstance(audio_values, Tensor) else torch.as_tensor(audio_values))
        real_waveform = self.acoustic_frontend._waveform(real_waveform)
        real_waveform = real_waveform.to(
            device=model_device,
            dtype=self.native_model.text_encoder.embed_tokens.weight.dtype,
        )
        if real_waveform.shape[0] != input_ids.shape[0]:
            raise ValueError("VITS waveform and text batch sizes must match.")
        audio_lengths = self._integer_lengths(
            audio_lengths,
            batch_size=real_waveform.shape[0],
            maximum=real_waveform.shape[1],
            device=model_device,
            name="audio_lengths",
        )

        if spectrogram is None:
            spectrogram = self.acoustic_frontend.spectrogram(real_waveform)
            spectrogram_lengths = self.acoustic_frontend.spectrogram_lengths(audio_lengths)
            frame_indices = torch.arange(
                spectrogram.shape[-1],
                device=model_device,
            )
            spectrogram_attention_mask = (frame_indices.unsqueeze(0) < spectrogram_lengths.unsqueeze(1))
        else:
            if not isinstance(spectrogram, Tensor):
                spectrogram = torch.as_tensor(spectrogram)
            if spectrogram.ndim == 2:
                spectrogram = spectrogram.unsqueeze(0)
            spectrogram = spectrogram.to(
                device=model_device,
                dtype=real_waveform.dtype,
            )

        segment_frames = self.acoustic_frontend.config.segment_frames
        output = self.native_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speaker_id=speaker_id,
            spectrogram=spectrogram,
            spectrogram_attention_mask=spectrogram_attention_mask,
            durations=durations,
            segment_frames=segment_frames,
            generator=generator,
        )
        if not isinstance(output, VitsTrainingOutput):
            raise RuntimeError("Native VITS did not return a training output.")
        hop_length = self.acoustic_frontend.config.hop_length
        generated_lengths = self._integer_lengths(
            output.sequence_lengths,
            batch_size=real_waveform.shape[0],
            maximum=output.waveform.shape[1],
            device=model_device,
            name="generated sequence lengths",
        )
        spectrogram_lengths = output.spectrogram_mask.to(dtype=torch.bool, ).sum(dim=(1, 2), dtype=torch.long)
        frame_lengths = torch.minimum(
            spectrogram_lengths,
            torch.div(
                audio_lengths,
                hop_length,
                rounding_mode="floor",
            ),
        )
        if (frame_lengths < 1).any():
            raise ValueError("VITS waveform/spectrogram pairs contain no aligned frames.")

        selected_frames = (int(frame_lengths.min().item()) if segment_frames is None else segment_frames)
        if (frame_lengths < selected_frames).any():
            raise ValueError(
                "Every VITS item must be at least the configured segment "
                f"size ({self.acoustic_frontend.config.segment_size} samples).")

        target_mel_full = self.acoustic_frontend.spectrogram_to_mel(spectrogram)
        sample_count = selected_frames * hop_length
        model_start_frames = output.segment_start_frames
        if segment_frames is not None:
            if (not isinstance(model_start_frames, Tensor) or
                    tuple(model_start_frames.shape) != (real_waveform.shape[0], )):
                raise RuntimeError(
                    "Windowed VITS training must return one segment start "
                    "frame per batch item.")
            if model_start_frames.dtype == torch.bool or model_start_frames.is_floating_point():
                raise RuntimeError("VITS segment start frames must use an integer dtype.")
            model_start_frames = model_start_frames.to(
                device=model_device,
                dtype=torch.long,
            )
            if ((model_start_frames < 0) | (model_start_frames + selected_frames > frame_lengths)).any():
                raise ValueError(
                    "VITS segment start frames exceed the aligned "
                    "waveform/spectrogram lengths.")
            expected_lengths = torch.full_like(
                generated_lengths,
                sample_count,
            )
            if (output.waveform.shape[1] != sample_count or
                    not torch.equal(generated_lengths, expected_lengths)):
                raise RuntimeError("Windowed VITS decoding returned an unexpected waveform "
                                   "length.")

        real_segments = []
        generated_segments = []
        mel_segments = []
        for batch_index, available_frames in enumerate(frame_lengths.tolist()):
            if segment_frames is None:
                maximum_start = int(available_frames) - selected_frames
                start_frame = (
                    0 if maximum_start == 0 else int(
                        torch.randint(
                            maximum_start + 1,
                            (1, ),
                            generator=generator,
                            device=model_device,
                        ).item()))
            else:
                start_frame = int(model_start_frames[batch_index].item())
            start_sample = start_frame * hop_length
            end_sample = start_sample + sample_count
            real_segments.append(real_waveform[batch_index, start_sample:end_sample])
            generated_start = start_sample if segment_frames is None else 0
            generated_segments.append(
                output.waveform[
                    batch_index,
                    generated_start:generated_start + sample_count,
                ])
            mel_segments.append(target_mel_full[
                batch_index,
                :,
                start_frame:start_frame + selected_frames,
            ])

        return _AdversarialBatch(
            output=output,
            real_waveform=torch.stack(real_segments),
            generated_waveform=torch.stack(generated_segments),
            target_mel=torch.stack(mel_segments),
        )

    def discriminator_step(
        self,
        input_ids: Any,
        *,
        audio_values: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute the detached-fake least-squares MPD objective."""
        batch = self._training_batch(
            input_ids,
            audio_values=audio_values,
            **kwargs,
        )
        (
            real_outputs,
            generated_outputs,
            _,
            _,
        ) = self.discriminator(
            batch.real_waveform,
            batch.generated_waveform.detach(),
        )
        loss, real_losses, generated_losses = discriminator_loss(
            real_outputs,
            generated_outputs,
        )
        return {
            "loss": loss,
            "audio_values": batch.generated_waveform.detach(),
            "losses": {
                "discriminator_loss": loss,
                "real_loss": torch.stack(real_losses).sum(),
                "generated_loss": torch.stack(generated_losses).sum(),
            },
        }

    def generator_step(
        self,
        input_ids: Any,
        *,
        audio_values: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute mel, duration, KL, feature, and adversarial losses."""
        batch = self._training_batch(
            input_ids,
            audio_values=audio_values,
            **kwargs,
        )
        (
            _,
            generated_outputs,
            real_feature_maps,
            generated_feature_maps,
        ) = self.discriminator(
            batch.real_waveform,
            batch.generated_waveform,
        )
        predicted_mel = self.acoustic_frontend.mel_spectrogram(batch.generated_waveform)
        if predicted_mel.shape != batch.target_mel.shape:
            raise RuntimeError(
                "VITS generated and target mel shapes diverged: "
                f"{tuple(predicted_mel.shape)} != "
                f"{tuple(batch.target_mel.shape)}.")
        mel_loss = functional.l1_loss(
            predicted_mel.float(),
            batch.target_mel.float(),
        )
        objective = self.generator_objective(
            batch.output,
            mel_reconstruction_loss=mel_loss,
            generated_discriminator_outputs=generated_outputs,
            real_feature_maps=real_feature_maps,
            generated_feature_maps=generated_feature_maps,
        )
        return {
            "loss": objective.total,
            "audio_values": batch.generated_waveform,
            "native_output": batch.output,
            "losses": {
                "generator_loss": objective.total,
                "adversarial_loss": objective.adversarial,
                "duration_loss": objective.duration,
                "feature_matching_loss": objective.feature_matching,
                "kl_loss": objective.kl_divergence,
                "mel_reconstruction_loss": objective.mel_reconstruction,
            },
        }


__all__ = [
    "VitsAcousticConfig",
    "VitsAcousticFrontend",
    "VitsAdversarialTrainingModel",
    "build_slaney_mel_filter",
]
