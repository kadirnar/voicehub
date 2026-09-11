"""Source-faithful MeloTTS VITS2 fine-tuning over explicit features."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.melotts.native.configuration import MeloTTSArchitectureConfig
from voicehub.models.melotts.source.melo.models import DurationDiscriminator, MultiPeriodDiscriminator
from voicehub.processing.audio import mel_filter_bank


def _nonnegative(name: str, value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real) or not 0.0 <= float(value) < float("inf")):
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return float(value)


@dataclass(frozen=True, slots=True)
class MeloTTSLossWeights:
    """Published MeloTTS relative weights and unscaled GAN terms."""

    mel: float = 45.0
    kl: float = 1.0
    duration: float = 1.0
    adversarial: float = 1.0
    feature_matching: float = 1.0
    duration_adversarial: float = 1.0

    def __post_init__(self) -> None:
        for name in self.__slots__:
            object.__setattr__(
                self,
                name,
                _nonnegative(name, getattr(self, name)),
            )
        if not any(getattr(self, name) > 0 for name in self.__slots__):
            raise ValueError("At least one MeloTTS loss weight must be positive.")


@contextmanager
def _frozen(module: nn.Module | None) -> Iterator[None]:
    if module is None:
        yield
        return
    states = tuple((parameter, parameter.requires_grad) for parameter in module.parameters())
    try:
        for parameter, _ in states:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in states:
            parameter.requires_grad_(enabled)


def feature_matching_loss(
    real_features: list[list[Tensor]],
    generated_features: list[list[Tensor]],
) -> Tensor:
    if len(real_features) != len(generated_features):
        raise ValueError("MeloTTS discriminator feature groups must align.")
    if not generated_features or not generated_features[0]:
        raise ValueError("MeloTTS discriminator returned no feature maps.")
    loss = generated_features[0][0].new_zeros(())
    for real_group, generated_group in zip(
            real_features,
            generated_features,
    ):
        if len(real_group) != len(generated_group):
            raise ValueError("MeloTTS discriminator feature maps must align.")
        for real, generated in zip(real_group, generated_group):
            loss = loss + torch.mean(torch.abs(real.float().detach() - generated.float()))
    return loss * 2.0


def _score_groups(
    scores: Tensor | Sequence[Tensor],
    *,
    name: str,
) -> tuple[Tensor, ...]:
    """Normalize MPD lists and the duration discriminator's batch tensor.

    The released loss iterates a duration score tensor along its batch axis.
    Keeping that behavior is important: its per-item means are summed rather
    than averaged across the batch.
    """
    if isinstance(scores, Tensor):
        if scores.ndim < 1 or scores.shape[0] < 1:
            raise ValueError(f"MeloTTS {name} scores must contain a batch.")
        return tuple(scores.unbind(0))
    if isinstance(scores, (str, bytes)) or not isinstance(scores, Sequence):
        raise TypeError(f"MeloTTS {name} scores must be a tensor sequence.")
    groups = tuple(scores)
    if not groups or any(not isinstance(score, Tensor) for score in groups):
        raise ValueError(f"MeloTTS {name} scores must contain tensors.")
    return groups


def discriminator_loss(
    real_scores: Tensor | Sequence[Tensor],
    generated_scores: Tensor | Sequence[Tensor],
) -> Tensor:
    real_groups = _score_groups(real_scores, name="real discriminator")
    generated_groups = _score_groups(
        generated_scores,
        name="generated discriminator",
    )
    if len(real_groups) != len(generated_groups):
        raise ValueError("MeloTTS discriminator score groups must align.")
    return sum(
        torch.mean((1.0 - real.float()).square()) + torch.mean(generated.float().square())
        for real, generated in zip(real_groups, generated_groups))


def generator_loss(generated_scores: Tensor | Sequence[Tensor], ) -> Tensor:
    groups = _score_groups(
        generated_scores,
        name="generated discriminator",
    )
    return sum(torch.mean((1.0 - score.float()).square()) for score in groups)


def kl_loss(
    z_p: Tensor,
    logs_q: Tensor,
    m_p: Tensor,
    logs_p: Tensor,
    mask: Tensor,
) -> Tensor:
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    mask = mask.float()
    value = logs_p - logs_q - 0.5
    value = value + (0.5 * (z_p - m_p).square() * torch.exp(-2.0 * logs_p))
    denominator = mask.sum()
    if not bool(denominator > 0):
        raise ValueError("MeloTTS KL mask cannot be empty.")
    return torch.sum(value * mask) / denominator


class MeloTTSMelSpectrogram(nn.Module):
    """Torch-only equivalent of the released librosa-filtered STFT."""

    def __init__(self, config: MeloTTSArchitectureConfig) -> None:
        super().__init__()
        if not isinstance(config, MeloTTSArchitectureConfig):
            raise TypeError("`config` must be a MeloTTSArchitectureConfig.")
        data = config.data
        self.n_fft = data.n_fft
        self.hop_length = data.hop_length
        self.win_length = data.win_length
        self.register_buffer(
            "window",
            torch.hann_window(data.win_length),
            persistent=False,
        )
        self.register_buffer(
            "filters",
            mel_filter_bank(
                sample_rate=data.sample_rate,
                n_fft=data.n_fft,
                n_mels=data.n_mels,
                minimum_frequency=data.mel_fmin,
                maximum_frequency=data.mel_fmax,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def spectrogram_to_mel(self, spectrogram: Tensor) -> Tensor:
        if spectrogram.ndim != 3 or spectrogram.shape[1] != self.n_fft // 2 + 1:
            raise ValueError("MeloTTS spectrogram must have shape [batch, frequency, frames].")
        filters = self.filters.to(
            device=spectrogram.device,
            dtype=spectrogram.dtype,
        )
        mel = torch.matmul(filters, spectrogram)
        return torch.log(torch.clamp(mel, min=1e-5))

    def waveform_to_mel(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 2:
            raise ValueError("MeloTTS waveform must have shape [batch, samples].")
        source = waveform.float()
        padding = (self.n_fft - self.hop_length) // 2
        if source.shape[-1] <= padding:
            raise ValueError("MeloTTS waveform is too short for reflect padding.")
        source = functional.pad(
            source.unsqueeze(1),
            (padding, padding),
            mode="reflect",
        ).squeeze(1)
        spectrum = torch.stft(
            source,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(
                device=source.device,
                dtype=source.dtype,
            ),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = torch.sqrt(spectrum.abs().square() + 1e-6)
        return self.spectrogram_to_mel(magnitude)


class MeloTTSTrainingModel(nn.Module):
    """Differentiable generator, waveform discriminator, and duration
    phases."""

    def __init__(
        self,
        model: nn.Module,
        config: MeloTTSArchitectureConfig,
        *,
        enable_discriminators: bool = True,
        mpd: MultiPeriodDiscriminator | None = None,
        duration_discriminator: DurationDiscriminator | None = None,
        loss_weights: MeloTTSLossWeights | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("`model` must be the native MeloTTS generator.")
        if not isinstance(config, MeloTTSArchitectureConfig):
            raise TypeError("`config` must be a MeloTTSArchitectureConfig.")
        if not isinstance(enable_discriminators, bool):
            raise TypeError("`enable_discriminators` must be a boolean.")
        if not enable_discriminators and (mpd is not None or duration_discriminator is not None):
            raise ValueError("MeloTTS discriminators cannot be supplied when disabled.")
        self.model = model
        self.config = config
        self.mpd = (
            mpd if mpd is not None else
            MultiPeriodDiscriminator(config.model.use_spectral_norm) if enable_discriminators else None)
        duration_enabled = (enable_discriminators and config.model.use_duration_discriminator)
        self.duration_discriminator = (
            duration_discriminator if duration_discriminator is not None else DurationDiscriminator(
                config.model.hidden_channels,
                config.model.hidden_channels,
                3,
                0.1,
                gin_channels=(config.model.gin_channels if config.data.n_speakers > 0 else 0),
            ) if duration_enabled else None)
        if duration_discriminator is not None and not duration_enabled:
            raise ValueError("Duration discriminator is disabled by the MeloTTS config.")
        self.loss_weights = loss_weights or MeloTTSLossWeights()
        self.mel = MeloTTSMelSpectrogram(config)

    def set_step(self, step: int) -> float:
        """Apply the released monotonic-alignment noise schedule."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("MeloTTS training step must be non-negative.")
        scale = max(
            self.config.model.mas_noise_scale_initial - self.config.model.noise_scale_delta * step,
            0.0,
        )
        self.model.current_mas_noise_scale = scale
        return scale

    @staticmethod
    def _integer(
        value: Any,
        *,
        name: str,
        device: torch.device,
    ) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if tensor.dtype == torch.bool or tensor.is_floating_point():
            raise TypeError(f"MeloTTS `{name}` must use integer dtype.")
        return tensor.to(device=device, dtype=torch.long)

    @staticmethod
    def _float(
        value: Any,
        *,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if tensor.is_complex() or tensor.dtype == torch.bool:
            raise TypeError(f"MeloTTS `{name}` must contain real values.")
        tensor = tensor.to(device=device, dtype=dtype)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"MeloTTS `{name}` contains NaN or infinity.")
        return tensor

    def _validate_batch(
        self,
        *,
        input_ids: Any,
        input_lengths: Any,
        tone_ids: Any,
        language_ids: Any,
        bert_features: Any,
        ja_bert_features: Any,
        spectrogram: Any,
        spectrogram_lengths: Any,
        audio_values: Any,
        audio_lengths: Any,
        speaker_ids: Any,
    ) -> tuple[Tensor, ...]:
        reference = next(self.model.parameters())
        device, dtype = reference.device, reference.dtype
        input_ids = self._integer(
            input_ids,
            name="input_ids",
            device=device,
        )
        input_lengths = self._integer(
            input_lengths,
            name="input_lengths",
            device=device,
        )
        tone_ids = self._integer(
            tone_ids,
            name="tone_ids",
            device=device,
        )
        language_ids = self._integer(
            language_ids,
            name="language_ids",
            device=device,
        )
        speaker_ids = self._integer(
            speaker_ids,
            name="speaker_ids",
            device=device,
        )
        bert_features = self._float(
            bert_features,
            name="bert_features",
            device=device,
            dtype=dtype,
        )
        ja_bert_features = self._float(
            ja_bert_features,
            name="ja_bert_features",
            device=device,
            dtype=dtype,
        )
        spectrogram = self._float(
            spectrogram,
            name="spectrogram",
            device=device,
            dtype=dtype,
        )
        spectrogram_lengths = self._integer(
            spectrogram_lengths,
            name="spectrogram_lengths",
            device=device,
        )
        audio_values = self._float(
            audio_values,
            name="audio_values",
            device=device,
            dtype=dtype,
        )
        audio_lengths = self._integer(
            audio_lengths,
            name="audio_lengths",
            device=device,
        )
        if input_ids.ndim != 2:
            raise ValueError("MeloTTS `input_ids` must have shape [batch, text].")
        batch, text_steps = input_ids.shape
        if tuple(input_lengths.shape) != (batch, ) or bool((input_lengths < 1).any() or
                                                           (input_lengths > text_steps).any()):
            raise ValueError("MeloTTS `input_lengths` must describe valid text.")
        if tone_ids.shape != input_ids.shape:
            raise ValueError("MeloTTS tone IDs must align with input IDs.")
        if language_ids.shape != input_ids.shape:
            raise ValueError("MeloTTS language IDs must align with input IDs.")
        if bool(((input_ids < 0) | (input_ids >= self.config.vocab_size)).any()):
            raise ValueError("MeloTTS input IDs are outside the vocabulary.")
        if bool(((tone_ids < 0) | (tone_ids >= self.config.num_tones)).any()):
            raise ValueError("MeloTTS tone IDs are outside the inventory.")
        if bool(((language_ids < 0) | (language_ids >= self.config.num_languages)).any()):
            raise ValueError("MeloTTS language IDs are outside the inventory.")
        if tuple(speaker_ids.shape) != (batch, ) or bool((speaker_ids < 0).any() or
                                                         (speaker_ids >= self.config.data.n_speakers).any()):
            raise ValueError("MeloTTS speaker IDs are outside the inventory.")
        for name, features, channels in (
            ("bert_features", bert_features, 1024),
            ("ja_bert_features", ja_bert_features, 768),
        ):
            if tuple(features.shape) != (batch, channels, text_steps):
                raise ValueError(f"MeloTTS `{name}` must have shape "
                                 f"[batch, {channels}, text].")
        frequency_bins = self.config.data.n_fft // 2 + 1
        if (spectrogram.ndim != 3 or spectrogram.shape[0] != batch or spectrogram.shape[1] != frequency_bins):
            raise ValueError("MeloTTS `spectrogram` must have shape "
                             "[batch, n_fft // 2 + 1, frames].")
        frames = spectrogram.shape[-1]
        invalid_spectrogram_lengths = bool((spectrogram_lengths < self.config.segment_frames).any() or
                                           (spectrogram_lengths > frames).any())
        if (tuple(spectrogram_lengths.shape) != (batch, ) or invalid_spectrogram_lengths):
            raise ValueError("MeloTTS spectrogram lengths must contain at least one "
                             "training segment.")
        if bool((spectrogram_lengths < input_lengths).any()):
            raise ValueError(
                "MeloTTS monotonic alignment requires at least one acoustic "
                "frame per text token.")
        if audio_values.ndim == 2:
            audio_values = audio_values.unsqueeze(1)
        if (audio_values.ndim != 3 or audio_values.shape[0] != batch or audio_values.shape[1] != 1):
            raise ValueError("MeloTTS `audio_values` must have shape [batch, 1, samples].")
        if tuple(audio_lengths.shape) != (batch, ) or bool((audio_lengths < self.config.segment_size).any() or
                                                           (audio_lengths > audio_values.shape[-1]).any()):
            raise ValueError("MeloTTS audio lengths must contain at least one training segment.")
        if bool((torch.div(
                audio_lengths,
                self.config.data.hop_length,
                rounding_mode="floor",
        ) != spectrogram_lengths).any()):
            raise ValueError("MeloTTS spectrogram lengths must equal "
                             "floor(audio_lengths / hop_length).")
        return (
            input_ids,
            input_lengths,
            tone_ids,
            language_ids,
            bert_features,
            ja_bert_features,
            spectrogram,
            spectrogram_lengths,
            audio_values,
            audio_lengths,
            speaker_ids,
        )

    @staticmethod
    def _slice_segments(
        values: Tensor,
        starts: Tensor,
        segment_size: int,
    ) -> Tensor:
        segments = []
        for index in range(values.shape[0]):
            start = int(starts[index])
            stop = start + segment_size
            segment = values[index, :, start:stop]
            if segment.shape[-1] != segment_size:
                raise ValueError("MeloTTS segment exceeds a valid sequence.")
            segments.append(segment)
        return torch.stack(segments)

    def _generator_forward(
        self,
        batch: tuple[Tensor, ...],
    ) -> dict[str, Any]:
        (
            input_ids,
            input_lengths,
            tone_ids,
            language_ids,
            bert_features,
            ja_bert_features,
            spectrogram,
            spectrogram_lengths,
            audio_values,
            _audio_lengths,
            speaker_ids,
        ) = batch
        (
            generated_audio,
            duration_value,
            alignment,
            segment_starts,
            text_mask,
            spectrogram_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (hidden_text, log_durations, target_log_durations),
        ) = self.model(
            input_ids,
            input_lengths,
            spectrogram,
            spectrogram_lengths,
            speaker_ids,
            tone_ids,
            language_ids,
            bert_features,
            ja_bert_features,
        )
        del z, m_q
        target_audio = self._slice_segments(
            audio_values,
            segment_starts * self.config.data.hop_length,
            self.config.segment_size,
        )
        target_mel = self.mel.spectrogram_to_mel(spectrogram.float())
        target_mel = self._slice_segments(
            target_mel,
            segment_starts,
            self.config.segment_frames,
        )
        generated_mel = self.mel.waveform_to_mel(generated_audio.squeeze(1))
        if generated_mel.shape != target_mel.shape:
            raise RuntimeError(
                "MeloTTS generated and target mel shapes do not align: "
                f"{tuple(generated_mel.shape)} != {tuple(target_mel.shape)}.")
        duration = torch.sum(duration_value.float())
        mel = functional.l1_loss(
            generated_mel,
            target_mel.float(),
        )
        kl = kl_loss(
            z_p,
            logs_q,
            m_p,
            logs_p,
            spectrogram_mask,
        )
        return {
            "generated_audio": generated_audio,
            "target_audio": target_audio,
            "duration_loss": duration,
            "mel_loss": mel,
            "kl_loss": kl,
            "alignment": alignment,
            "hidden_text": hidden_text,
            "text_mask": text_mask,
            "log_durations": log_durations,
            "target_log_durations": target_log_durations,
        }

    def generator_objective(
        self,
        batch: tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        outputs = self._generator_forward(batch)
        generated_audio = outputs["generated_audio"]
        target_audio = outputs["target_audio"]
        zero = generated_audio.new_zeros(())
        adversarial = zero
        feature_matching = zero
        duration_adversarial = zero
        if self.mpd is not None:
            with _frozen(self.mpd):
                (
                    _real_scores,
                    generated_scores,
                    real_features,
                    generated_features,
                ) = self.mpd(target_audio, generated_audio)
            adversarial = generator_loss(generated_scores)
            feature_matching = feature_matching_loss(
                real_features,
                generated_features,
            )
        if self.duration_discriminator is not None:
            with _frozen(self.duration_discriminator):
                _real_duration, generated_duration = (
                    self.duration_discriminator(
                        outputs["hidden_text"],
                        outputs["text_mask"],
                        outputs["log_durations"],
                        outputs["target_log_durations"],
                    ))
            duration_adversarial = generator_loss(generated_duration)

        weights = self.loss_weights
        loss = (
            outputs["mel_loss"] * weights.mel + outputs["kl_loss"] * weights.kl +
            outputs["duration_loss"] * weights.duration + adversarial * weights.adversarial +
            feature_matching * weights.feature_matching + duration_adversarial * weights.duration_adversarial)
        return {
            "loss": loss,
            "mel_loss": outputs["mel_loss"],
            "kl_loss": outputs["kl_loss"],
            "duration_loss": outputs["duration_loss"],
            "adversarial_loss": adversarial,
            "feature_matching_loss": feature_matching,
            "duration_adversarial_loss": duration_adversarial,
            "generated_audio": generated_audio,
        }

    def waveform_discriminator_objective(
        self,
        batch: tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        if self.mpd is None:
            raise RuntimeError("MeloTTS waveform discriminator is disabled.")
        with torch.no_grad():
            outputs = self._generator_forward(batch)
        (
            real_scores,
            generated_scores,
            _real_features,
            _generated_features,
        ) = self.mpd(
            outputs["target_audio"].detach(),
            outputs["generated_audio"].detach(),
        )
        waveform_loss = discriminator_loss(
            real_scores,
            generated_scores,
        )
        return {
            "loss": waveform_loss,
            "discriminator_loss": waveform_loss,
        }

    def duration_discriminator_objective(
        self,
        batch: tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        if self.duration_discriminator is None:
            raise RuntimeError("MeloTTS duration discriminator is disabled.")
        with torch.no_grad():
            outputs = self._generator_forward(batch)
        real_duration, generated_duration = self.duration_discriminator(
            outputs["hidden_text"].detach(),
            outputs["text_mask"].detach(),
            outputs["log_durations"].detach(),
            outputs["target_log_durations"].detach(),
        )
        loss = discriminator_loss(
            real_duration,
            generated_duration,
        )
        return {
            "loss": loss,
            "duration_discriminator_loss": loss,
        }

    def forward(
        self,
        *,
        input_ids: Any,
        input_lengths: Any,
        tone_ids: Any,
        language_ids: Any,
        bert_features: Any,
        ja_bert_features: Any,
        spectrogram: Any,
        spectrogram_lengths: Any,
        audio_values: Any,
        audio_lengths: Any,
        speaker_ids: Any,
        phase: str = "generator",
    ) -> dict[str, Tensor]:
        batch = self._validate_batch(
            input_ids=input_ids,
            input_lengths=input_lengths,
            tone_ids=tone_ids,
            language_ids=language_ids,
            bert_features=bert_features,
            ja_bert_features=ja_bert_features,
            spectrogram=spectrogram,
            spectrogram_lengths=spectrogram_lengths,
            audio_values=audio_values,
            audio_lengths=audio_lengths,
            speaker_ids=speaker_ids,
        )
        if phase == "generator":
            return self.generator_objective(batch)
        if phase == "discriminator":
            return self.waveform_discriminator_objective(batch)
        if phase == "duration_discriminator":
            return self.duration_discriminator_objective(batch)
        raise ValueError(
            "MeloTTS training phase must be `generator`, `discriminator`, "
            "or `duration_discriminator`.")


class MeloTTSTrainingCollator:
    """Right-pad upstream-compatible linguistic and acoustic sequences."""

    REQUIRED_FIELDS = (
        "input_ids",
        "tone_ids",
        "language_ids",
        "bert_features",
        "ja_bert_features",
        "spectrogram",
        "audio_values",
        "speaker_id",
    )

    @staticmethod
    def _tensor(value: Any, *, name: str) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if tensor.is_complex():
            raise TypeError(f"MeloTTS `{name}` cannot be complex.")
        return tensor

    @staticmethod
    def _pad_last(value: Tensor, length: int) -> Tensor:
        if value.shape[-1] > length:
            raise ValueError("MeloTTS target padding length is too short.")
        return functional.pad(value, (0, length - value.shape[-1]))

    def __call__(
        self,
        features: list[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        if not features:
            return {}
        rows = [dict(feature) for feature in features]
        for index, row in enumerate(rows):
            missing = [name for name in self.REQUIRED_FIELDS if name not in row]
            if missing:
                raise ValueError(f"MeloTTS sample {index} is missing: " + ", ".join(missing) + ".")
        values = {name: [self._tensor(row[name], name=name) for row in rows] for name in self.REQUIRED_FIELDS}
        for name in ("input_ids", "tone_ids", "language_ids"):
            if any(item.ndim != 1 for item in values[name]):
                raise ValueError(f"Each MeloTTS `{name}` sample must be one-dimensional.")
            if any(item.dtype == torch.bool or item.is_floating_point() for item in values[name]):
                raise TypeError(f"MeloTTS `{name}` must use integer dtype.")
        for ids, tones, languages in zip(
                values["input_ids"],
                values["tone_ids"],
                values["language_ids"],
        ):
            if ids.numel() < 1 or tones.shape != ids.shape or languages.shape != ids.shape:
                raise ValueError("MeloTTS phone, tone, and language sequences must align.")
        max_text = max(item.shape[-1] for item in values["input_ids"])
        result: dict[str, Tensor] = {}
        for name in ("input_ids", "tone_ids", "language_ids"):
            result[name] = torch.stack([self._pad_last(item, max_text) for item in values[name]]).long()
        result["input_lengths"] = torch.tensor(
            [item.shape[-1] for item in values["input_ids"]],
            dtype=torch.long,
        )
        for name, channels in (
            ("bert_features", 1024),
            ("ja_bert_features", 768),
        ):
            if any(item.ndim != 2 or item.shape[0] != channels for item in values[name]):
                raise ValueError(f"Each MeloTTS `{name}` must have shape "
                                 f"[{channels}, text].")
            for item, ids in zip(values[name], values["input_ids"]):
                if item.shape[-1] != ids.shape[-1]:
                    raise ValueError(f"MeloTTS `{name}` must align with input IDs.")
            result[name] = torch.stack([self._pad_last(item, max_text) for item in values[name]]).float()
        spectrograms = values["spectrogram"]
        if any(item.ndim != 2 for item in spectrograms):
            raise ValueError("Each MeloTTS spectrogram must have shape [frequency, frames].")
        frequency_bins = spectrograms[0].shape[0]
        if any(item.shape[0] != frequency_bins for item in spectrograms):
            raise ValueError("MeloTTS spectrogram frequency bins must align.")
        max_frames = max(item.shape[-1] for item in spectrograms)
        result["spectrogram"] = torch.stack([self._pad_last(item, max_frames)
                                             for item in spectrograms]).float()
        result["spectrogram_lengths"] = torch.tensor(
            [item.shape[-1] for item in spectrograms],
            dtype=torch.long,
        )
        audio = [
            item.squeeze(0) if item.ndim == 2 and item.shape[0] == 1 else item
            for item in values["audio_values"]
        ]
        if any(item.ndim != 1 for item in audio):
            raise ValueError("Each MeloTTS audio sample must be one-dimensional.")
        max_samples = max(item.shape[-1] for item in audio)
        result["audio_values"] = torch.stack([self._pad_last(item, max_samples)
                                              for item in audio]).unsqueeze(1).float()
        result["audio_lengths"] = torch.tensor(
            [item.shape[-1] for item in audio],
            dtype=torch.long,
        )
        speaker_ids = []
        for item in values["speaker_id"]:
            if item.numel() != 1 or item.dtype == torch.bool or item.is_floating_point():
                raise TypeError("Each MeloTTS `speaker_id` must be one integer.")
            speaker_ids.append(int(item.reshape(-1)[0]))
        result["speaker_ids"] = torch.tensor(
            speaker_ids,
            dtype=torch.long,
        )
        return result

    def resume_fingerprint(self) -> dict[str, Any]:
        return {
            "type": "melotts-explicit-features-v1",
            "phone_padding_id": 0,
            "feature_padding": "right-zero",
        }


__all__ = [
    "MeloTTSLossWeights",
    "MeloTTSMelSpectrogram",
    "MeloTTSTrainingCollator",
    "MeloTTSTrainingModel",
    "discriminator_loss",
    "feature_matching_loss",
    "generator_loss",
    "kl_loss",
]
