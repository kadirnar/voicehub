"""Differentiable VITS warm-start objectives for Inflect v2.

The release contains the complete deployable generator but intentionally
omits the posterior encoder and discriminators.  This module
reconstructs those published VITS components and initializes them
freshly.  It is a real fine-tuning path, but not a claim that the
unpublished author optimizer/data recipe or resumable state has been
recovered.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.inflecttts.native.configuration import InflectV2Config
from voicehub.models.inflecttts.native.modeling import MultiPeriodDiscriminator, SynthesizerTrn
from voicehub.models.vits.native.losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    vits_kl_loss,
)
from voicehub.processing.audio import mel_filter_bank


def _nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    normalized = float(value)
    if not 0.0 <= normalized < float("inf"):
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return normalized


@dataclass(frozen=True, slots=True)
class InflectLossWeights:
    """Weights matching the conventional VITS generator objective."""

    mel: float = 45.0
    kl: float = 1.0
    duration: float = 1.0
    adversarial: float = 1.0
    feature_matching: float = 1.0
    waveform: float = 0.0

    def __post_init__(self) -> None:
        for name in (
                "mel",
                "kl",
                "duration",
                "adversarial",
                "feature_matching",
                "waveform",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(name, getattr(self, name)),
            )
        if not any(getattr(self, name) > 0 for name in self.__slots__):
            raise ValueError("At least one Inflect loss weight must be positive.")


@contextmanager
def _temporarily_frozen(module: nn.Module) -> Iterator[None]:
    states = tuple((parameter, parameter.requires_grad) for parameter in module.parameters())
    try:
        for parameter, _ in states:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in states:
            parameter.requires_grad_(enabled)


class InflectV2TrainingModel(nn.Module):
    """Full preprocessed VITS objective around a warm-started generator."""

    def __init__(
        self,
        generator: SynthesizerTrn,
        config: InflectV2Config,
        *,
        discriminator: MultiPeriodDiscriminator | None = None,
        enable_discriminator: bool = True,
        loss_weights: InflectLossWeights | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(generator, SynthesizerTrn):
            raise TypeError("Inflect training requires the native SynthesizerTrn.")
        if not isinstance(config, InflectV2Config):
            raise TypeError("`config` must be an InflectV2Config.")
        if generator.inference_only or not hasattr(generator, "enc_q"):
            raise ValueError(
                "Inflect training requires a graph built from "
                "`config.for_training()` so the fresh posterior exists.")
        if not isinstance(enable_discriminator, bool):
            raise TypeError("`enable_discriminator` must be a boolean.")
        if discriminator is not None and not isinstance(
                discriminator,
                MultiPeriodDiscriminator,
        ):
            raise TypeError("`discriminator` must be a MultiPeriodDiscriminator or None.")
        if not enable_discriminator and discriminator is not None:
            raise ValueError("A discriminator cannot be supplied when it is disabled.")
        self.generator = generator
        self.config = config
        self.discriminator = (
            discriminator if discriminator is not None else MultiPeriodDiscriminator(
                use_spectral_norm=config.use_spectral_norm) if enable_discriminator else None)
        self.loss_weights = loss_weights or InflectLossWeights()
        self.register_buffer(
            "_mel_filters",
            mel_filter_bank(
                sample_rate=config.sample_rate,
                n_fft=config.filter_length,
                n_mels=config.mel_channels,
                minimum_frequency=config.mel_min_frequency,
                maximum_frequency=config.mel_max_frequency,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    @staticmethod
    def _integer_tensor(
        value: Any,
        *,
        name: str,
        device: torch.device,
    ) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if tensor.dtype == torch.bool or tensor.is_floating_point():
            raise TypeError(f"Inflect `{name}` must use an integer dtype.")
        return tensor.to(device=device, dtype=torch.long)

    @staticmethod
    def _float_tensor(
        value: Any,
        *,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if tensor.is_complex():
            raise TypeError(f"Inflect `{name}` cannot use a complex dtype.")
        tensor = tensor.to(device=device, dtype=dtype)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Inflect `{name}` contains NaN or infinity.")
        return tensor

    def _validate_batch(
        self,
        *,
        input_ids: Any,
        input_lengths: Any,
        spectrogram: Any,
        spectrogram_lengths: Any,
        audio_values: Any,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        reference = next(self.generator.parameters())
        input_ids = self._integer_tensor(
            input_ids,
            name="input_ids",
            device=reference.device,
        )
        input_lengths = self._integer_tensor(
            input_lengths,
            name="input_lengths",
            device=reference.device,
        )
        spectrogram = self._float_tensor(
            spectrogram,
            name="spectrogram",
            device=reference.device,
            dtype=reference.dtype,
        )
        spectrogram_lengths = self._integer_tensor(
            spectrogram_lengths,
            name="spectrogram_lengths",
            device=reference.device,
        )
        audio_values = self._float_tensor(
            audio_values,
            name="audio_values",
            device=reference.device,
            dtype=reference.dtype,
        )
        if input_ids.ndim != 2:
            raise ValueError("Inflect `input_ids` must have shape [batch, text].")
        batch, text_steps = input_ids.shape
        if tuple(input_lengths.shape) != (batch, ):
            raise ValueError("Inflect `input_lengths` must have shape [batch].")
        if bool(((input_ids < 0) | (input_ids >= self.config.vocabulary_size)).any()):
            raise ValueError("Inflect input IDs are outside the vocabulary.")
        if bool(((input_lengths < 1) | (input_lengths > text_steps)).any()):
            raise ValueError("Inflect input lengths are outside the padded text.")
        if (spectrogram.ndim != 3 or spectrogram.shape[0] != batch or
                spectrogram.shape[1] != self.config.spectrogram_channels):
            raise ValueError(
                "Inflect `spectrogram` must have shape "
                f"[batch, {self.config.spectrogram_channels}, frames].")
        if tuple(spectrogram_lengths.shape) != (batch, ):
            raise ValueError("Inflect `spectrogram_lengths` must have shape [batch].")
        frames = spectrogram.shape[-1]
        if bool(((spectrogram_lengths < self.config.segment_frames) | (spectrogram_lengths > frames)).any()):
            raise ValueError(
                "Every Inflect spectrogram must contain at least one complete "
                "training segment and fit inside its padded tensor.")
        if audio_values.ndim == 2:
            audio_values = audio_values.unsqueeze(1)
        if (audio_values.ndim != 3 or audio_values.shape[:2] != (batch, 1)):
            raise ValueError(
                "Inflect `audio_values` must have shape [batch, samples] or "
                "[batch, 1, samples].")
        required_samples = int(spectrogram_lengths.max().item()) * self.config.hop_length
        if audio_values.shape[-1] < required_samples:
            raise ValueError(
                "Inflect audio is shorter than the declared spectrogram "
                "length at the checkpoint hop size.")
        return (
            input_ids,
            input_lengths,
            spectrogram,
            spectrogram_lengths,
            audio_values,
        )

    @staticmethod
    def _slice_segments(
        values: Tensor,
        starts: Tensor,
        length: int,
    ) -> Tensor:
        pieces = []
        for batch_index, start in enumerate(starts):
            offset = int(start.item())
            piece = values[batch_index:batch_index + 1, :, offset:offset + length]
            if piece.shape[-1] != length:
                raise ValueError("Inflect training segment exceeds its target.")
            pieces.append(piece)
        return torch.cat(pieces, dim=0)

    def _linear_to_mel(self, spectrogram: Tensor) -> Tensor:
        filters = self._mel_filters.to(
            device=spectrogram.device,
            dtype=spectrogram.dtype,
        )
        mel = torch.matmul(filters, spectrogram)
        return torch.log(torch.clamp(mel, min=1e-5))

    def _waveform_to_mel(self, waveform: Tensor) -> Tensor:
        flattened = waveform[:, 0]
        padding = (self.config.filter_length - self.config.hop_length) // 2
        flattened = functional.pad(
            flattened.unsqueeze(1),
            (padding, padding),
            mode="reflect",
        ).squeeze(1)
        window = torch.hann_window(
            self.config.win_length,
            dtype=flattened.dtype,
            device=flattened.device,
        )
        spectrum = torch.stft(
            flattened,
            n_fft=self.config.filter_length,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            center=False,
            return_complex=True,
        ).abs()
        return self._linear_to_mel(spectrum)

    def _generator_outputs(
        self,
        *,
        input_ids: Tensor,
        input_lengths: Tensor,
        spectrogram: Tensor,
        spectrogram_lengths: Tensor,
        audio_values: Tensor,
    ) -> dict[str, Any]:
        (
            generated,
            duration_loss,
            alignment,
            frame_starts,
            text_mask,
            spectrogram_mask,
            latent_values,
        ) = self.generator(
            input_ids,
            input_lengths,
            spectrogram,
            spectrogram_lengths,
        )
        (
            posterior_latents,
            prior_latents,
            prior_means,
            prior_log_variances,
            _posterior_means,
            posterior_log_variances,
        ) = latent_values
        audio_starts = frame_starts * self.config.hop_length
        target_audio = self._slice_segments(
            audio_values,
            audio_starts,
            self.config.segment_size,
        )
        target_spectrogram = self._slice_segments(
            spectrogram,
            frame_starts,
            self.config.segment_frames,
        )
        return {
            "generated_audio":
            generated,
            "target_audio":
            target_audio,
            "target_mel":
            self._linear_to_mel(target_spectrogram),
            "generated_mel":
            self._waveform_to_mel(generated),
            "duration_loss":
            duration_loss.mean(),
            "kl_loss":
            vits_kl_loss(
                prior_latents,
                posterior_log_variances,
                prior_means,
                prior_log_variances,
                spectrogram_mask,
            ),
            "alignment":
            alignment,
            "frame_starts":
            frame_starts,
            "text_mask":
            text_mask,
            "spectrogram_mask":
            spectrogram_mask,
            "posterior_latents":
            posterior_latents,
        }

    def generator_objective(
        self,
        input_ids: Any,
        *,
        input_lengths: Any,
        spectrogram: Any,
        spectrogram_lengths: Any,
        audio_values: Any,
    ) -> dict[str, Any]:
        """Compute the generator/posterior/flow/decoder VITS objective."""
        batch = self._validate_batch(
            input_ids=input_ids,
            input_lengths=input_lengths,
            spectrogram=spectrogram,
            spectrogram_lengths=spectrogram_lengths,
            audio_values=audio_values,
        )
        outputs = self._generator_outputs(
            input_ids=batch[0],
            input_lengths=batch[1],
            spectrogram=batch[2],
            spectrogram_lengths=batch[3],
            audio_values=batch[4],
        )
        mel_loss = functional.l1_loss(
            outputs["generated_mel"].float(),
            outputs["target_mel"].float(),
        )
        maximum = min(
            outputs["generated_audio"].shape[-1],
            outputs["target_audio"].shape[-1],
        )
        waveform_loss = functional.l1_loss(
            outputs["generated_audio"][..., :maximum].float(),
            outputs["target_audio"][..., :maximum].float(),
        )
        adversarial = outputs["generated_audio"].new_zeros(())
        feature_matching = outputs["generated_audio"].new_zeros(())
        if self.discriminator is not None:
            with _temporarily_frozen(self.discriminator):
                (
                    _real_scores,
                    generated_scores,
                    real_features,
                    generated_features,
                ) = self.discriminator(
                    outputs["target_audio"],
                    outputs["generated_audio"],
                )
            adversarial, _ = generator_adversarial_loss(generated_scores)
            feature_matching = feature_matching_loss(
                real_features,
                generated_features,
            )
        weights = self.loss_weights
        total = (
            weights.mel * mel_loss + weights.kl * outputs["kl_loss"] +
            weights.duration * outputs["duration_loss"] + weights.adversarial * adversarial +
            weights.feature_matching * feature_matching + weights.waveform * waveform_loss)
        return {
            "loss": total,
            "waveform": outputs["generated_audio"],
            "mel_loss": mel_loss,
            "kl_loss": outputs["kl_loss"],
            "duration_loss": outputs["duration_loss"],
            "adversarial_loss": adversarial,
            "feature_matching_loss": feature_matching,
            "waveform_loss": waveform_loss,
            "alignment": outputs["alignment"],
            "frame_starts": outputs["frame_starts"],
        }

    def discriminator_objective(
        self,
        input_ids: Any,
        *,
        input_lengths: Any,
        spectrogram: Any,
        spectrogram_lengths: Any,
        audio_values: Any,
    ) -> dict[str, Any]:
        """Compute the fresh multi-period discriminator objective."""
        if self.discriminator is None:
            raise RuntimeError("Inflect discriminator training was explicitly disabled.")
        batch = self._validate_batch(
            input_ids=input_ids,
            input_lengths=input_lengths,
            spectrogram=spectrogram,
            spectrogram_lengths=spectrogram_lengths,
            audio_values=audio_values,
        )
        with torch.no_grad():
            outputs = self._generator_outputs(
                input_ids=batch[0],
                input_lengths=batch[1],
                spectrogram=batch[2],
                spectrogram_lengths=batch[3],
                audio_values=batch[4],
            )
        real_scores, generated_scores, _, _ = self.discriminator(
            outputs["target_audio"],
            outputs["generated_audio"].detach(),
        )
        loss, real_losses, generated_losses = discriminator_loss(
            real_scores,
            generated_scores,
        )
        return {
            "loss": loss,
            "real_losses": real_losses,
            "generated_losses": generated_losses,
            "waveform": outputs["generated_audio"],
            "frame_starts": outputs["frame_starts"],
        }

    def forward(
        self,
        input_ids: Any,
        *,
        input_lengths: Any,
        spectrogram: Any,
        spectrogram_lengths: Any,
        audio_values: Any,
        phase: str = "generator",
    ) -> dict[str, Any]:
        if phase == "generator":
            return self.generator_objective(
                input_ids,
                input_lengths=input_lengths,
                spectrogram=spectrogram,
                spectrogram_lengths=spectrogram_lengths,
                audio_values=audio_values,
            )
        if phase == "discriminator":
            return self.discriminator_objective(
                input_ids,
                input_lengths=input_lengths,
                spectrogram=spectrogram,
                spectrogram_lengths=spectrogram_lengths,
                audio_values=audio_values,
            )
        raise ValueError("`phase` must be either 'generator' or 'discriminator'.")


__all__ = [
    "InflectLossWeights",
    "InflectV2TrainingModel",
]
