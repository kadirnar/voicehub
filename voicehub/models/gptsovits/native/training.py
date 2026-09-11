"""Source-faithful staged fine-tuning for GPT-SoVITS classic S2."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.gptsovits.native.configuration import GPTSoVITSS1Config, GPTSoVITSS2Config
from voicehub.models.gptsovits.native.modeling import (
    GPTSoVITSSynthesizer,
    MultiPeriodDiscriminator,
    build_s2_discriminator,
)
from voicehub.models.gptsovits.native.semantic import GPTSoVITSSemanticModel
from voicehub.models.vits.native.losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    vits_kl_loss,
)
from voicehub.processing.audio import mel_filter_bank


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


class GPTSoVITSS2TrainingModel(nn.Module):
    """Separate S2 generator and discriminator phases from ``s2_train.py``."""

    def __init__(
        self,
        generator: GPTSoVITSSynthesizer,
        *,
        discriminator: MultiPeriodDiscriminator | None = None,
        enable_discriminator: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(generator, GPTSoVITSSynthesizer):
            raise TypeError("S2 training requires the native GPTSoVITSSynthesizer.")
        if not isinstance(enable_discriminator, bool):
            raise TypeError("`enable_discriminator` must be a boolean.")
        if discriminator is not None and not enable_discriminator:
            raise ValueError("A discriminator cannot be supplied when it is disabled.")
        self.generator = generator
        self.config = generator.config
        self.discriminator = (
            discriminator if discriminator is not None else
            build_s2_discriminator(self.config) if enable_discriminator else None)
        self.register_buffer(
            "_mel_filters",
            mel_filter_bank(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.filter_length,
                n_mels=self.config.mel_channels,
                minimum_frequency=self.config.mel_min_frequency,
                maximum_frequency=self.config.mel_max_frequency,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def _batch(
        self,
        *,
        ssl_features: Any,
        spectrogram: Any,
        spectrogram_lengths: Any,
        audio_values: Any,
        phoneme_ids: Any,
        phoneme_lengths: Any,
        speaker_embedding: Any | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        reference = next(self.generator.parameters())
        device = reference.device
        dtype = reference.dtype
        ssl_features = torch.as_tensor(
            ssl_features,
            device=device,
            dtype=dtype,
        )
        spectrogram = torch.as_tensor(
            spectrogram,
            device=device,
            dtype=dtype,
        )
        spectrogram_lengths = torch.as_tensor(
            spectrogram_lengths,
            device=device,
            dtype=torch.long,
        )
        audio_values = torch.as_tensor(
            audio_values,
            device=device,
            dtype=dtype,
        )
        phoneme_ids = torch.as_tensor(
            phoneme_ids,
            device=device,
            dtype=torch.long,
        )
        phoneme_lengths = torch.as_tensor(
            phoneme_lengths,
            device=device,
            dtype=torch.long,
        )
        if ssl_features.ndim != 3 or ssl_features.shape[1] != self.config.ssl_channels:
            raise ValueError(
                "S2 SSL features must have shape "
                f"[batch, {self.config.ssl_channels}, frames].")
        prepared_speaker = None
        if self.config.requires_speaker_embedding:
            if speaker_embedding is None:
                raise ValueError(
                    f"GPT-SoVITS {self.config.version} training requires "
                    "prepared speaker embeddings.")
            prepared_speaker = torch.as_tensor(
                speaker_embedding,
                device=device,
                dtype=dtype,
            )
            expected = (ssl_features.shape[0], self.config.speaker_embedding_dim)
            if tuple(prepared_speaker.shape) != expected:
                raise ValueError(f"S2 speaker embeddings must have shape {expected}.")
        elif speaker_embedding is not None:
            raise ValueError(f"GPT-SoVITS {self.config.version} does not consume speaker embeddings.")
        batch = ssl_features.shape[0]
        if spectrogram.ndim != 3 or spectrogram.shape[:2] != (
                batch,
                self.config.spectrogram_channels,
        ):
            raise ValueError(
                "S2 spectrogram must have shape "
                f"[batch, {self.config.spectrogram_channels}, frames].")
        if tuple(spectrogram_lengths.shape) != (batch, ):
            raise ValueError("S2 spectrogram lengths must have shape [batch].")
        if phoneme_ids.ndim != 2 or phoneme_ids.shape[0] != batch:
            raise ValueError("S2 phoneme IDs must have shape [batch, time].")
        if tuple(phoneme_lengths.shape) != (batch, ):
            raise ValueError("S2 phoneme lengths must have shape [batch].")
        if bool(((phoneme_ids < 0) | (phoneme_ids >= self.config.phoneme_vocabulary_size)).any()):
            raise ValueError(f"S2 phoneme IDs are outside the {self.config.version} vocabulary.")
        if bool(((phoneme_lengths < 1) | (phoneme_lengths > phoneme_ids.shape[1])).any()):
            raise ValueError("S2 phoneme lengths are invalid.")
        if bool(((spectrogram_lengths < self.config.segment_frames) |
                 (spectrogram_lengths > spectrogram.shape[2])).any()):
            raise ValueError("S2 spectrogram lengths cannot produce a full segment.")
        if ssl_features.shape[2] < int(spectrogram_lengths.max().item()):
            raise ValueError("S2 SSL features are shorter than the declared spectrogram.")
        if audio_values.ndim == 2:
            audio_values = audio_values.unsqueeze(1)
        if audio_values.ndim != 3 or audio_values.shape[:2] != (batch, 1):
            raise ValueError("S2 audio must have shape [batch, samples] or [batch, 1, samples].")
        required = int(spectrogram_lengths.max().item()) * self.config.hop_length
        if audio_values.shape[2] < required:
            raise ValueError("S2 audio is shorter than its declared spectrogram.")
        return (
            ssl_features,
            spectrogram,
            spectrogram_lengths,
            audio_values,
            phoneme_ids,
            phoneme_lengths,
            prepared_speaker,
        )

    @staticmethod
    def _slice(inputs: Tensor, starts: Tensor, length: int) -> Tensor:
        pieces = []
        for batch_index, start in enumerate(starts):
            offset = int(start.item())
            piece = inputs[
                batch_index:batch_index + 1,
                :,
                offset:offset + length,
            ]
            if piece.shape[-1] != length:
                raise ValueError("GPT-SoVITS training segment exceeds its source.")
            pieces.append(piece)
        return torch.cat(pieces)

    def _mel(self, magnitudes: Tensor) -> Tensor:
        filters = self._mel_filters.to(
            device=magnitudes.device,
            dtype=magnitudes.dtype,
        )
        return torch.log(torch.clamp(torch.matmul(filters, magnitudes), min=1e-5))

    def _waveform_mel(self, waveform: Tensor) -> Tensor:
        padding = (self.config.filter_length - self.config.hop_length) // 2
        flattened = functional.pad(
            waveform[:, 0].unsqueeze(1),
            (padding, padding),
            mode="reflect",
        ).squeeze(1)
        window = torch.hann_window(
            self.config.window_length,
            device=flattened.device,
            dtype=flattened.dtype,
        )
        spectrum = torch.stft(
            flattened,
            n_fft=self.config.filter_length,
            hop_length=self.config.hop_length,
            win_length=self.config.window_length,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()
        spectrum = torch.sqrt(spectrum.square() + 1e-8)
        return self._mel(spectrum)

    def _generator_pass(
        self,
        batch: tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor | None,
        ],
    ) -> tuple[dict[str, Tensor], Tensor]:
        (
            ssl_features,
            spectrogram,
            spectrogram_lengths,
            audio_values,
            phoneme_ids,
            phoneme_lengths,
            speaker_embedding,
        ) = batch
        (
            waveform,
            commitment_loss,
            slice_ids,
            _,
            latent_mask,
            latents,
            _,
        ) = self.generator(
            ssl_features,
            spectrogram,
            spectrogram_lengths,
            phoneme_ids,
            phoneme_lengths,
            speaker_embedding,
        )
        (
            _,
            prior_latent,
            prior_mean,
            prior_log_scale,
            _,
            posterior_log_scale,
        ) = latents
        target_mel = self._mel(spectrogram)
        target_mel = self._slice(
            target_mel,
            slice_ids,
            self.config.segment_frames,
        )
        generated_mel = self._waveform_mel(waveform)
        target_audio = self._slice(
            audio_values,
            slice_ids * self.config.hop_length,
            self.config.segment_size,
        )
        mel_loss = functional.l1_loss(target_mel, generated_mel) * 45.0
        kl_loss = vits_kl_loss(
            prior_latent,
            posterior_log_scale,
            prior_mean,
            prior_log_scale,
            latent_mask,
        )
        outputs = {
            "waveform": waveform,
            "target_waveform": target_audio,
            "mel_loss": mel_loss,
            "kl_loss": kl_loss,
            "commitment_loss": commitment_loss.float(),
        }
        return outputs, target_audio

    def generator_objective(self, **inputs: Any) -> dict[str, Any]:
        batch = self._batch(**inputs)
        outputs, target_audio = self._generator_pass(batch)
        discriminator = self.discriminator
        if discriminator is None:
            zero = outputs["waveform"].new_zeros(())
            adversarial = zero
            feature_matching = zero
            adversarial_components: tuple[Tensor, ...] = ()
        else:
            with _frozen(discriminator):
                _, generated, real_maps, generated_maps = discriminator(
                    target_audio,
                    outputs["waveform"],
                )
            adversarial, adversarial_components = generator_adversarial_loss(generated)
            feature_matching = feature_matching_loss(real_maps, generated_maps)
        total = (
            adversarial + feature_matching + outputs["mel_loss"] + outputs["commitment_loss"] +
            outputs["kl_loss"])
        return {
            **outputs,
            "loss": total,
            "adversarial_loss": adversarial,
            "feature_matching_loss": feature_matching,
            "adversarial_components": adversarial_components,
        }

    def discriminator_objective(self, **inputs: Any) -> dict[str, Any]:
        if self.discriminator is None:
            raise RuntimeError("S2 discriminator training is disabled.")
        batch = self._batch(**inputs)
        with torch.no_grad():
            generated, target_audio = self._generator_pass(batch)
        real, fake, _, _ = self.discriminator(
            target_audio,
            generated["waveform"].detach(),
        )
        total, real_losses, generated_losses = discriminator_loss(real, fake)
        return {
            "loss": total,
            "real_losses": real_losses,
            "generated_losses": generated_losses,
        }


class GPTSoVITSStagedTrainingModel(nn.Module):
    """Explicit S1, S2-generator, and S2-discriminator training surface."""

    def __init__(
        self,
        *,
        s1: GPTSoVITSSemanticModel | None = None,
        s2_generator: GPTSoVITSSynthesizer | None = None,
        s2_discriminator: MultiPeriodDiscriminator | None = None,
        enable_s2_discriminator: bool = True,
    ) -> None:
        super().__init__()
        self.s1 = s1
        self.s2 = (
            None if s2_generator is None else GPTSoVITSS2TrainingModel(
                s2_generator,
                discriminator=s2_discriminator,
                enable_discriminator=enable_s2_discriminator,
            ))
        if self.s1 is None and self.s2 is None:
            raise ValueError("At least one GPT-SoVITS training stage is required.")

    def s1_objective(self, **inputs: Any) -> dict[str, Tensor]:
        if self.s1 is None:
            raise RuntimeError("GPT-SoVITS S1 was not loaded.")
        return self.s1(**inputs)

    def s2_generator_objective(self, **inputs: Any) -> dict[str, Any]:
        if self.s2 is None:
            raise RuntimeError("GPT-SoVITS S2 was not loaded.")
        return self.s2.generator_objective(**inputs)

    def s2_discriminator_objective(self, **inputs: Any) -> dict[str, Any]:
        if self.s2 is None:
            raise RuntimeError("GPT-SoVITS S2 was not loaded.")
        return self.s2.discriminator_objective(**inputs)

    def forward(self, *, phase: str, **inputs: Any) -> dict[str, Any]:
        methods = {
            "s1": self.s1_objective,
            "s2_generator": self.s2_generator_objective,
            "s2_discriminator": self.s2_discriminator_objective,
        }
        try:
            method = methods[phase]
        except KeyError as error:
            raise ValueError("`phase` must be s1, s2_generator, or s2_discriminator.") from error
        return method(**inputs)


def build_staged_training_model(
    *,
    include_s1: bool = True,
    include_s2: bool = True,
    enable_s2_discriminator: bool = True,
    variant: str = "v2",
) -> GPTSoVITSStagedTrainingModel:
    s1_config = GPTSoVITSS1Config.for_variant(variant)
    s2_config = GPTSoVITSS2Config.for_variant(variant)
    return GPTSoVITSStagedTrainingModel(
        s1=(GPTSoVITSSemanticModel(s1_config) if include_s1 else None),
        s2_generator=(GPTSoVITSSynthesizer(s2_config) if include_s2 else None),
        enable_s2_discriminator=enable_s2_discriminator,
    )


__all__ = [
    "GPTSoVITSS2TrainingModel",
    "GPTSoVITSStagedTrainingModel",
    "build_staged_training_model",
]
