"""Preprocessed, teacher-forced StyleTTS 2 fine-tuning objectives.

The upstream raw-data recipe delegates phonemization, monotonic
alignment, pitch extraction, and WavLM features to separately released
runtimes. Native VoiceHub fine-tuning therefore accepts those targets
explicitly and optimizes the complete deployable generator, including
PL-BERT, duration/prosody, style diffusion, style encoders, and HiFi-
GAN. The MPD/MSD discriminators are fresh training-only modules and are
never represented as pretrained weights.
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

from voicehub.models.styletts2.native.configuration import StyleTTS2ArchitectureConfig
from voicehub.models.styletts2.native.frontend import StyleTTS2MelSpectrogram
from voicehub.models.styletts2.source.styletts2.Modules.discriminators import (
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
)


def _nonnegative(name: str, value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real) or not 0.0 <= float(value) < float("inf")):
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return float(value)


@dataclass(frozen=True, slots=True)
class StyleTTS2LossWeights:
    """Published relative weights, excluding the unavailable WavLM term."""

    mel: float = 5.0
    f0: float = 1.0
    noise: float = 1.0
    duration: float = 1.0
    duration_ce: float = 20.0
    diffusion: float = 1.0
    adversarial: float = 1.0
    feature_matching: float = 1.0
    relativistic: float = 1.0
    waveform: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__slots__:
            object.__setattr__(
                self,
                name,
                _nonnegative(name, getattr(self, name)),
            )
        if not any(getattr(self, name) > 0 for name in self.__slots__):
            raise ValueError("At least one StyleTTS 2 loss weight must be positive.")


@contextmanager
def _frozen(module: nn.Module) -> Iterator[None]:
    states = tuple((parameter, parameter.requires_grad) for parameter in module.parameters())
    try:
        for parameter, _ in states:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in states:
            parameter.requires_grad_(enabled)


def _least_squares_generator(generated_scores: list[Tensor], ) -> Tensor:
    return sum(functional.mse_loss(score, torch.ones_like(score)) for score in generated_scores)


def _feature_matching(
    real_features: list[list[Tensor]],
    generated_features: list[list[Tensor]],
) -> Tensor:
    loss = generated_features[0][0].new_zeros(())
    for real_group, generated_group in zip(
            real_features,
            generated_features,
    ):
        for real, generated in zip(real_group, generated_group):
            loss = loss + functional.l1_loss(
                generated,
                real.detach(),
            )
    return loss * 2.0


def _least_squares_discriminator(
    real_scores: list[Tensor],
    generated_scores: list[Tensor],
) -> Tensor:
    return sum(
        functional.mse_loss(real, torch.ones_like(real)) +
        functional.mse_loss(generated, torch.zeros_like(generated))
        for real, generated in zip(real_scores, generated_scores))


def _tprls_term(left: Tensor, right: Tensor) -> Tensor:
    margin = left - right
    median = torch.median(margin)
    selected = (margin - median).square()[left < right + median]
    relative = (selected.mean() if selected.numel() else margin.new_zeros(()))
    threshold = margin.new_tensor(0.04)
    return threshold - functional.relu(threshold - relative)


def _tprls_discriminator(
    real_scores: list[Tensor],
    generated_scores: list[Tensor],
) -> Tensor:
    return sum(_tprls_term(real, generated) for real, generated in zip(real_scores, generated_scores))


def _tprls_generator(
    real_scores: list[Tensor],
    generated_scores: list[Tensor],
) -> Tensor:
    # Preserve the released generator objective's reversed TPRLS operands.
    return sum(_tprls_term(generated, real) for real, generated in zip(real_scores, generated_scores))


def _mean_losses(losses: list[Tensor], *, name: str) -> Tensor:
    if not losses:
        raise RuntimeError(f"StyleTTS 2 produced no {name} values.")
    return torch.stack(losses).mean()


class StyleTTS2TrainingModel(nn.Module):
    """Differentiable objective over explicit upstream training targets."""

    def __init__(
        self,
        model: nn.Module,
        config: StyleTTS2ArchitectureConfig,
        *,
        enable_discriminators: bool = True,
        mpd: MultiPeriodDiscriminator | None = None,
        msd: MultiResSpecDiscriminator | None = None,
        loss_weights: StyleTTS2LossWeights | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("`model` must be the native StyleTTS 2 graph.")
        if not isinstance(config, StyleTTS2ArchitectureConfig):
            raise TypeError("`config` must be a typed StyleTTS 2 config.")
        if not isinstance(enable_discriminators, bool):
            raise TypeError("`enable_discriminators` must be a boolean.")
        if not enable_discriminators and (mpd is not None or msd is not None):
            raise ValueError("Discriminators cannot be supplied when they are disabled.")
        self.model = model
        self.config = config
        self.mpd = (mpd if mpd is not None else MultiPeriodDiscriminator() if enable_discriminators else None)
        self.msd = (
            msd if msd is not None else MultiResSpecDiscriminator() if enable_discriminators else None)
        if (self.mpd is None) != (self.msd is None):
            raise ValueError("MPD and MSD must be enabled or disabled together.")
        self.loss_weights = loss_weights or StyleTTS2LossWeights()
        released_preprocessing = (
            config.sample_rate,
            config.n_fft,
            config.win_length,
            config.hop_length,
        ) == (24_000, 2_048, 1_200, 300)
        resolutions = ((
            (1_024, 600, 120),
            (2_048, 1_200, 240),
            (512, 240, 50),
        ) if released_preprocessing else ((config.n_fft, config.win_length, config.hop_length), ))
        self.mel_transforms = nn.ModuleList(
            StyleTTS2MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=config.n_mels,
            ) for n_fft, win_length, hop_length in resolutions)
        self.minimum_mel_samples = max(transform.n_fft // 2 + 1 for transform in self.mel_transforms)

    @staticmethod
    def _length_mask(lengths: Tensor, steps: int) -> Tensor:
        return (torch.arange(steps, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1))

    @staticmethod
    def _as_integer(
        value: Any,
        *,
        name: str,
        device: torch.device,
    ) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if tensor.dtype == torch.bool or tensor.is_floating_point():
            raise TypeError(f"StyleTTS 2 `{name}` must use integer dtype.")
        return tensor.to(device=device, dtype=torch.long)

    @staticmethod
    def _as_float(
        value: Any,
        *,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if tensor.is_complex():
            raise TypeError(f"StyleTTS 2 `{name}` cannot be complex.")
        tensor = tensor.to(device=device, dtype=dtype)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"StyleTTS 2 `{name}` contains NaN or infinity.")
        return tensor

    def _validate_batch(
        self,
        *,
        input_ids: Any,
        input_lengths: Any,
        alignments: Any,
        alignment_lengths: Any,
        normalized_mel: Any,
        normalized_mel_lengths: Any,
        reference_mel: Any,
        reference_mel_lengths: Any,
        f0_targets: Any,
        noise_targets: Any,
        audio_values: Any,
        audio_lengths: Any,
    ) -> tuple[Tensor, ...]:
        reference = next(self.model.parameters())
        device, dtype = reference.device, reference.dtype
        input_ids = self._as_integer(
            input_ids,
            name="input_ids",
            device=device,
        )
        input_lengths = self._as_integer(
            input_lengths,
            name="input_lengths",
            device=device,
        )
        alignments = self._as_float(
            alignments,
            name="alignments",
            device=device,
            dtype=dtype,
        )
        alignment_lengths = self._as_integer(
            alignment_lengths,
            name="alignment_lengths",
            device=device,
        )
        normalized_mel = self._as_float(
            normalized_mel,
            name="normalized_mel",
            device=device,
            dtype=dtype,
        )
        normalized_mel_lengths = self._as_integer(
            normalized_mel_lengths,
            name="normalized_mel_lengths",
            device=device,
        )
        reference_mel = self._as_float(
            reference_mel,
            name="reference_mel",
            device=device,
            dtype=dtype,
        )
        reference_mel_lengths = self._as_integer(
            reference_mel_lengths,
            name="reference_mel_lengths",
            device=device,
        )
        f0_targets = self._as_float(
            f0_targets,
            name="f0_targets",
            device=device,
            dtype=dtype,
        )
        noise_targets = self._as_float(
            noise_targets,
            name="noise_targets",
            device=device,
            dtype=dtype,
        )
        audio_values = self._as_float(
            audio_values,
            name="audio_values",
            device=device,
            dtype=dtype,
        )
        audio_lengths = self._as_integer(
            audio_lengths,
            name="audio_lengths",
            device=device,
        )
        if input_ids.ndim != 2:
            raise ValueError("`input_ids` must have shape [batch, text].")
        batch, text_steps = input_ids.shape
        if tuple(input_lengths.shape) != (batch, ):
            raise ValueError("`input_lengths` must have shape [batch].")
        if bool((input_ids < 0).any() or (input_ids >= self.config.n_token).any()):
            raise ValueError("StyleTTS 2 token IDs are outside the vocabulary.")
        if bool((input_lengths < 3).any() or (input_lengths > text_steps).any()):
            raise ValueError("StyleTTS 2 training requires BOS plus at least two "
                             "phoneme tokens.")
        if bool((input_ids[:, 0] != 0).any()):
            raise ValueError("StyleTTS 2 training IDs must begin with BOS ID 0.")
        if alignments.ndim != 3 or alignments.shape[:2] != (
                batch,
                text_steps,
        ):
            raise ValueError("`alignments` must have shape [batch, text, acoustic_frames].")
        if bool((alignments < 0).any()):
            raise ValueError("StyleTTS 2 alignments cannot be negative.")
        acoustic_frames = alignments.shape[-1]
        if tuple(alignment_lengths.shape) != (batch, ) or bool((alignment_lengths < 1).any() or
                                                               (alignment_lengths > acoustic_frames).any()):
            raise ValueError("`alignment_lengths` must describe valid acoustic frames.")
        if bool((alignment_lengths < input_lengths).any()):
            raise ValueError(
                "A monotonic StyleTTS 2 alignment needs at least one "
                "acoustic frame per text token.")
        for index, (text_length, acoustic_length) in enumerate(zip(input_lengths.tolist(),
                                                                   alignment_lengths.tolist())):
            valid = alignments[
                index,
                :text_length,
                :acoustic_length,
            ]
            frame_sums = valid.sum(dim=0)
            if not bool(torch.allclose(
                    frame_sums,
                    torch.ones_like(frame_sums),
                    atol=1e-4,
                    rtol=1e-4,
            )):
                raise ValueError(
                    "Each valid StyleTTS 2 acoustic frame must be assigned "
                    "to exactly one text token.")
            if not bool(((valid.abs() <= 1e-4) | ((valid - 1.0).abs() <= 1e-4)).all()):
                raise ValueError("StyleTTS 2 monotonic alignments must be binary paths.")
            token_path = valid.argmax(dim=0)
            advances = token_path[1:] - token_path[:-1]
            starts_at_first_token = int(token_path[0]) == 0
            ends_at_last_token = int(token_path[-1]) == text_length - 1
            advances_monotonically = not bool(((advances < 0) | (advances > 1)).any())
            if not (starts_at_first_token and ends_at_last_token and advances_monotonically):
                raise ValueError(
                    "StyleTTS 2 alignments must form a complete monotonic "
                    "path from the first to the last text token.")
            invalid_text = alignments[index, text_length:, :]
            invalid_frames = alignments[
                index,
                :text_length,
                acoustic_length:,
            ]
            if bool((invalid_text.abs() > 1e-6).any() or (invalid_frames.abs() > 1e-6).any()):
                raise ValueError("StyleTTS 2 alignment padding must contain only zeros.")
        if normalized_mel.ndim == 3:
            normalized_mel = normalized_mel.unsqueeze(1)
        if reference_mel.ndim == 3:
            reference_mel = reference_mel.unsqueeze(1)
        for name, mel in (
            ("normalized_mel", normalized_mel),
            ("reference_mel", reference_mel),
        ):
            if (mel.ndim != 4 or mel.shape[0] != batch or mel.shape[1] != 1 or
                    mel.shape[2] != self.config.n_mels):
                raise ValueError(f"`{name}` must have shape [batch, 1, "
                                 "n_mels, frames].")
        for name, lengths, mel in (
            (
                "normalized_mel_lengths",
                normalized_mel_lengths,
                normalized_mel,
            ),
            (
                "reference_mel_lengths",
                reference_mel_lengths,
                reference_mel,
            ),
        ):
            if tuple(lengths.shape) != (batch, ) or bool((lengths < 1).any() or
                                                         (lengths > mel.shape[-1]).any()):
                raise ValueError(f"`{name}` must describe valid mel frames.")
        expected_mel_lengths = alignment_lengths * 2
        if bool((normalized_mel_lengths != expected_mel_lengths).any()):
            raise ValueError("`normalized_mel_lengths` must equal "
                             "2 * `alignment_lengths`.")
        expected_prosody_frames = acoustic_frames * 2
        for name, target in (
            ("f0_targets", f0_targets),
            ("noise_targets", noise_targets),
        ):
            if tuple(target.shape) != (batch, expected_prosody_frames):
                raise ValueError(f"`{name}` must have shape [batch, "
                                 "2 * acoustic_frames].")
        if audio_values.ndim == 2:
            audio_values = audio_values.unsqueeze(1)
        if (audio_values.ndim != 3 or audio_values.shape[0] != batch or audio_values.shape[1] != 1 or
                audio_values.shape[-1] < 2):
            raise ValueError("`audio_values` must have shape [batch, 1, samples].")
        if tuple(audio_lengths.shape) != (batch, ) or bool((audio_lengths < 2).any() or
                                                           (audio_lengths > audio_values.shape[-1]).any()):
            raise ValueError("`audio_lengths` must describe valid samples.")
        expected_audio_lengths = (normalized_mel_lengths * self.config.hop_length)
        if bool((audio_lengths != expected_audio_lengths).any()):
            raise ValueError(
                "`audio_lengths` must equal `normalized_mel_lengths` times "
                f"the configured hop length ({self.config.hop_length}).")
        if bool((audio_lengths < self.minimum_mel_samples).any()):
            raise ValueError("StyleTTS 2 audio examples are too short for reflect-padded "
                             "mel extraction.")
        if self.msd is not None:
            minimum_discriminator_samples = max(
                discriminator.fft_size // 2 + 1 for discriminator in self.msd.discriminators)
            if bool((audio_lengths < minimum_discriminator_samples).any()):
                raise ValueError(
                    "StyleTTS 2 adversarial examples are too short for the "
                    "released multi-resolution discriminator.")
        return (
            input_ids,
            input_lengths,
            alignments,
            alignment_lengths,
            normalized_mel,
            normalized_mel_lengths,
            reference_mel,
            reference_mel_lengths,
            f0_targets,
            noise_targets,
            audio_values,
            audio_lengths,
        )

    @staticmethod
    def _styles(
        encoder: nn.Module,
        mel: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        return torch.cat(
            [
                encoder(mel[
                    index:index + 1,
                    :,
                    :,
                    :int(length),
                ]) for index, length in enumerate(lengths.tolist())
            ],
            dim=0,
        )

    def _generator_outputs(
        self,
        batch: tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        (
            input_ids,
            input_lengths,
            alignments,
            alignment_lengths,
            normalized_mel,
            normalized_mel_lengths,
            reference_mel,
            reference_mel_lengths,
            f0_targets,
            noise_targets,
            audio_values,
            audio_lengths,
        ) = batch
        text_mask = self._length_mask(input_lengths, input_ids.shape[1])
        acoustic_mask = self._length_mask(
            alignment_lengths,
            alignments.shape[-1],
        )
        alignments = alignments.masked_fill(
            text_mask.unsqueeze(-1) | acoustic_mask.unsqueeze(1),
            0.0,
        )
        text_encoding = self.model.text_encoder(
            input_ids,
            input_lengths,
            text_mask,
        )
        bert = self.model.bert(
            input_ids,
            attention_mask=(~text_mask).int(),
        )
        duration_encoding = self.model.bert_encoder(bert).transpose(-1, -2)
        acoustic_style = self._styles(
            self.model.style_encoder,
            normalized_mel,
            normalized_mel_lengths,
        )
        prosody_style = self._styles(
            self.model.predictor_encoder,
            normalized_mel,
            normalized_mel_lengths,
        )
        target_style = torch.cat(
            [acoustic_style, prosody_style],
            dim=-1,
        )
        with torch.no_grad():
            reference_style = torch.cat(
                [
                    self._styles(
                        self.model.style_encoder,
                        reference_mel,
                        reference_mel_lengths,
                    ),
                    self._styles(
                        self.model.predictor_encoder,
                        reference_mel,
                        reference_mel_lengths,
                    ),
                ],
                dim=-1,
            )

        duration_logits, prosody_encoding = self.model.predictor(
            duration_encoding,
            prosody_style,
            input_lengths,
            alignments,
            text_mask,
        )
        text_decoder_encoding = text_encoding @ alignments
        f0_predictions = []
        noise_predictions = []
        generated_waveforms = []
        generated_lengths = []
        maximum_prosody_frames = f0_targets.shape[-1]
        for index, acoustic_length in enumerate(alignment_lengths.tolist()):
            f0_prediction, noise_prediction = (
                self.model.predictor.F0Ntrain(
                    prosody_encoding[
                        index:index + 1,
                        :,
                        :acoustic_length,
                    ],
                    prosody_style[index:index + 1],
                ))
            expected_prosody_frames = acoustic_length * 2
            if (tuple(f0_prediction.shape) != (1, expected_prosody_frames) or
                    tuple(noise_prediction.shape) != (1, expected_prosody_frames)):
                raise RuntimeError("StyleTTS 2 prosody prediction returned an unexpected "
                                   "length.")
            waveform = self.model.decoder(
                text_decoder_encoding[
                    index:index + 1,
                    :,
                    :acoustic_length,
                ],
                f0_prediction,
                noise_prediction,
                acoustic_style[index:index + 1],
            )
            if (waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] != 1):
                raise RuntimeError("StyleTTS 2 decoder must return [batch, 1, samples].")
            if waveform.shape[-1] != int(audio_lengths[index]):
                raise RuntimeError(
                    "StyleTTS 2 decoder length does not match the waveform "
                    "target implied by the acoustic alignment.")
            generated_lengths.append(waveform.shape[-1])
            f0_predictions.append(
                functional.pad(
                    f0_prediction,
                    (
                        0,
                        maximum_prosody_frames - expected_prosody_frames,
                    ),
                ))
            noise_predictions.append(
                functional.pad(
                    noise_prediction,
                    (
                        0,
                        maximum_prosody_frames - expected_prosody_frames,
                    ),
                ))
            generated_waveforms.append(waveform)
        maximum_audio_samples = max(generated_lengths)
        f0_prediction = torch.cat(f0_predictions, dim=0)
        noise_prediction = torch.cat(noise_predictions, dim=0)
        generated = torch.cat(
            [
                functional.pad(
                    waveform,
                    (0, maximum_audio_samples - waveform.shape[-1]),
                ) for waveform in generated_waveforms
            ],
            dim=0,
        )
        diffusion_inputs = {"embedding": bert}
        if self.config.multispeaker:
            diffusion_inputs["features"] = reference_style
        diffusion_loss = self.model.diffusion(
            target_style.detach().unsqueeze(1),
            **diffusion_inputs,
        ).mean()
        duration_targets = alignments.sum(dim=-1).round().to(dtype=torch.long)
        duration_classes = torch.arange(
            duration_logits.shape[-1],
            device=duration_logits.device,
        )
        duration_binary = (duration_classes.view(1, 1, -1)
                           < duration_targets.unsqueeze(-1)).to(dtype=duration_logits.dtype)
        duration_prediction = torch.sigmoid(duration_logits).sum(dim=-1)
        duration_ce = _mean_losses(
            [
                functional.binary_cross_entropy_with_logits(
                    duration_logits[index, :length],
                    duration_binary[index, :length],
                ) for index, length in enumerate(input_lengths.tolist())
            ],
            name="duration-cross-entropy",
        )
        duration_loss = _mean_losses(
            [
                functional.l1_loss(
                    duration_prediction[index, 1:length - 1],
                    duration_targets[
                        index,
                        1:length - 1,
                    ].to(dtype=duration_prediction.dtype),
                ) for index, length in enumerate(input_lengths.tolist())
            ],
            name="duration-regression",
        )
        prosody_mask = ~self._length_mask(
            alignment_lengths * 2,
            f0_prediction.shape[-1],
        )
        f0_values = functional.smooth_l1_loss(
            f0_prediction,
            f0_targets,
            reduction="none",
        )
        noise_values = functional.smooth_l1_loss(
            noise_prediction,
            noise_targets,
            reduction="none",
        )
        f0_loss = f0_values[prosody_mask].mean() / 10.0
        noise_loss = noise_values[prosody_mask].mean()

        comparison_lengths = audio_lengths
        maximum = int(comparison_lengths.max().item())
        generated = generated[..., :maximum]
        target_audio = audio_values[..., :maximum]
        audio_mask = ~self._length_mask(comparison_lengths, maximum)
        waveform_values = (generated - target_audio).abs()
        waveform_loss = (waveform_values * audio_mask.unsqueeze(1)).sum() / audio_mask.sum().clamp_min(1)

        mel_losses = []
        for index, length in enumerate(comparison_lengths.tolist()):
            resolution_losses = []
            for transform in self.mel_transforms:
                generated_mel = transform(generated[index, 0, :length])
                target_mel = transform(target_audio[index, 0, :length])
                mel_frames = min(
                    generated_mel.shape[-1],
                    target_mel.shape[-1],
                )
                generated_mel = (torch.log(1e-5 + generated_mel[..., :mel_frames]) + 4.0) / 4.0
                target_mel = (torch.log(1e-5 + target_mel[..., :mel_frames]) + 4.0) / 4.0
                resolution_losses.append((target_mel - generated_mel).abs().sum() /
                                         target_mel.abs().sum().clamp_min(torch.finfo(target_mel.dtype).eps))
            mel_losses.append(_mean_losses(
                resolution_losses,
                name="mel-resolution-loss",
            ))
        mel_loss = _mean_losses(mel_losses, name="mel-loss")
        return {
            "waveform": generated,
            "target_audio": target_audio,
            "waveform_lengths": comparison_lengths,
            "mel_loss": mel_loss,
            "f0_loss": f0_loss,
            "noise_loss": noise_loss,
            "duration_loss": duration_loss,
            "duration_ce_loss": duration_ce,
            "diffusion_loss": diffusion_loss,
            "waveform_loss": waveform_loss,
            "duration_logits": duration_logits,
            "f0_prediction": f0_prediction,
            "noise_prediction": noise_prediction,
        }

    @staticmethod
    def _waveform_pairs(outputs: dict[str, Tensor], ) -> Iterator[tuple[Tensor, Tensor]]:
        for index, length in enumerate(outputs["waveform_lengths"].tolist()):
            yield (
                outputs["target_audio"][index:index + 1, :, :length],
                outputs["waveform"][index:index + 1, :, :length],
            )

    def generator_objective(self, **batch_values: Any) -> dict[str, Any]:
        batch = self._validate_batch(**batch_values)
        outputs = self._generator_outputs(batch)
        adversarial = outputs["waveform"].new_zeros(())
        feature_matching = outputs["waveform"].new_zeros(())
        relativistic = outputs["waveform"].new_zeros(())
        if self.mpd is not None and self.msd is not None:
            adversarial_losses = []
            feature_matching_losses = []
            relativistic_losses = []
            with _frozen(self.mpd), _frozen(self.msd):
                for target, generated in self._waveform_pairs(outputs):
                    example_adversarial = generated.new_zeros(())
                    example_features = generated.new_zeros(())
                    example_relativistic = generated.new_zeros(())
                    for discriminator in (self.mpd, self.msd):
                        (
                            real_scores,
                            generated_scores,
                            real_features,
                            generated_features,
                        ) = discriminator(target, generated)
                        example_adversarial = (
                            example_adversarial + _least_squares_generator(generated_scores))
                        example_features = (
                            example_features + _feature_matching(
                                real_features,
                                generated_features,
                            ))
                        example_relativistic = (
                            example_relativistic + _tprls_generator(
                                real_scores,
                                generated_scores,
                            ))
                    adversarial_losses.append(example_adversarial)
                    feature_matching_losses.append(example_features)
                    relativistic_losses.append(example_relativistic)
            adversarial = _mean_losses(
                adversarial_losses,
                name="generator-adversarial-loss",
            )
            feature_matching = _mean_losses(
                feature_matching_losses,
                name="feature-matching-loss",
            )
            relativistic = _mean_losses(
                relativistic_losses,
                name="generator-relativistic-loss",
            )
        weights = self.loss_weights
        total = (
            weights.mel * outputs["mel_loss"] + weights.f0 * outputs["f0_loss"] +
            weights.noise * outputs["noise_loss"] + weights.duration * outputs["duration_loss"] +
            weights.duration_ce * outputs["duration_ce_loss"] +
            weights.diffusion * outputs["diffusion_loss"] + weights.adversarial * adversarial +
            weights.feature_matching * feature_matching + weights.relativistic * relativistic +
            weights.waveform * outputs["waveform_loss"])
        return {
            "loss": total,
            **outputs,
            "adversarial_loss": adversarial,
            "feature_matching_loss": feature_matching,
            "relativistic_loss": relativistic,
        }

    def discriminator_objective(
        self,
        **batch_values: Any,
    ) -> dict[str, Any]:
        if self.mpd is None or self.msd is None:
            raise RuntimeError("StyleTTS 2 discriminators were explicitly disabled.")
        batch = self._validate_batch(**batch_values)
        with torch.no_grad():
            outputs = self._generator_outputs(batch)
        losses = []
        for target, generated in self._waveform_pairs(outputs):
            example_loss = generated.new_zeros(())
            for discriminator in (self.mpd, self.msd):
                real_scores, generated_scores, _, _ = discriminator(
                    target,
                    generated.detach(),
                )
                example_loss = example_loss + _least_squares_discriminator(
                    real_scores,
                    generated_scores,
                )
                example_loss = example_loss + _tprls_discriminator(
                    real_scores,
                    generated_scores,
                )
            losses.append(example_loss)
        loss = _mean_losses(losses, name="discriminator-loss")
        return {
            "loss": loss,
            "waveform": outputs["waveform"],
        }

    def forward(
        self,
        input_ids: Any,
        *,
        input_lengths: Any,
        alignments: Any,
        alignment_lengths: Any,
        normalized_mel: Any,
        normalized_mel_lengths: Any,
        reference_mel: Any,
        reference_mel_lengths: Any,
        f0_targets: Any,
        noise_targets: Any,
        audio_values: Any,
        audio_lengths: Any,
        phase: str = "generator",
    ) -> dict[str, Any]:
        values = {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "alignments": alignments,
            "alignment_lengths": alignment_lengths,
            "normalized_mel": normalized_mel,
            "normalized_mel_lengths": normalized_mel_lengths,
            "reference_mel": reference_mel,
            "reference_mel_lengths": reference_mel_lengths,
            "f0_targets": f0_targets,
            "noise_targets": noise_targets,
            "audio_values": audio_values,
            "audio_lengths": audio_lengths,
        }
        if phase == "generator":
            return self.generator_objective(**values)
        if phase == "discriminator":
            return self.discriminator_objective(**values)
        raise ValueError("`phase` must be 'generator' or 'discriminator'.")


__all__ = [
    "StyleTTS2LossWeights",
    "StyleTTS2TrainingModel",
]
