"""Exact 2021 SpeechBrain Fbank and global-normalization frontend."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from voicehub.models.asr_speechbrain.native.configuration import SpeechBrainCRDNNASRConfig


def speechbrain_asr_mel_filterbank(config: SpeechBrainCRDNNASRConfig, ) -> Tensor:
    """Build SpeechBrain's legacy triangular mel matrix."""
    mel_min = 2_595.0 * math.log10(1.0 + config.f_min / 700.0)
    mel_max = 2_595.0 * math.log10(1.0 + config.f_max / 700.0)
    mel = torch.linspace(mel_min, mel_max, config.n_mels + 2)
    hz = 700.0 * (10.0**(mel / 2_595.0) - 1.0)
    bands = hz[1:] - hz[:-1]
    central = hz[1:-1]
    frequencies = torch.linspace(
        0.0,
        float(config.sampling_rate // 2),
        config.n_fft // 2 + 1,
    )
    slope = (frequencies.unsqueeze(0) - central.unsqueeze(1)) / bands[:-1].unsqueeze(1)
    filters = torch.maximum(
        torch.zeros(1, dtype=frequencies.dtype),
        torch.minimum(slope + 1.0, -slope + 1.0),
    )
    return filters.transpose(0, 1).contiguous()


class SpeechBrainGlobalNormalizer(nn.Module):
    """Stateful global CMVN matching
    ``InputNormalization(norm_type=global)``."""

    def __init__(self, config: SpeechBrainCRDNNASRConfig) -> None:
        super().__init__()
        self.config = SpeechBrainCRDNNASRConfig.coerce(config)
        self.register_buffer("glob_mean", torch.zeros(self.config.n_mels))
        self.register_buffer("glob_std", torch.ones(self.config.n_mels))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _current_statistics(
        self,
        features: Tensor,
        relative_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        means = []
        standard_deviations = []
        for index in range(features.shape[0]):
            actual_size = int(torch.round(relative_lengths[index] * features.shape[1], ).item())
            if actual_size < 2:
                raise ValueError("Global normalization needs at least two valid frames.")
            valid = features[index, :actual_size]
            means.append(valid.mean(dim=0))
            standard_deviations.append(
                valid.std(dim=0, correction=1).clamp_min(self.config.normalization_epsilon, ))
        return (
            torch.stack(means).mean(dim=0),
            torch.stack(standard_deviations).mean(dim=0),
        )

    def forward(
        self,
        features: Tensor,
        relative_lengths: Tensor,
        *,
        epoch: int = 0,
        update_statistics: bool = False,
    ) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.config.n_mels:
            raise ValueError(
                "SpeechBrain ASR features must have shape "
                f"[batch, frames, {self.config.n_mels}].")
        lengths = torch.as_tensor(
            relative_lengths,
            dtype=features.dtype,
            device=features.device,
        )
        if lengths.ndim != 1 or lengths.shape[0] != features.shape[0]:
            raise ValueError("`relative_lengths` must have shape [batch].")
        if torch.any(lengths <= 0.0) or torch.any(lengths > 1.0):
            raise ValueError("Relative waveform lengths must be in (0, 1].")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("`epoch` must be a non-negative integer.")
        if not isinstance(update_statistics, bool):
            raise TypeError("`update_statistics` must be a boolean.")
        if update_statistics:
            current_mean, current_std = self._current_statistics(
                features.detach(),
                lengths.detach(),
            )
            with torch.no_grad():
                current_mean = current_mean.to(self.glob_mean)
                current_std = current_std.to(self.glob_std)
                if int(self.count.item()) == 0:
                    self.glob_mean.copy_(current_mean)
                    self.glob_std.copy_(current_std)
                elif epoch < self.config.normalization_update_until_epoch:
                    weight = 1.0 / (int(self.count.item()) + 1)
                    self.glob_mean.lerp_(current_mean, weight)
                    self.glob_std.lerp_(current_std, weight)
                self.count.add_(1)
        return (features - self.glob_mean.to(
            device=features.device,
            dtype=features.dtype,
        )) / self.glob_std.to(
            device=features.device,
            dtype=features.dtype,
        ).clamp_min(self.config.normalization_epsilon)


class SpeechBrainASRFrontend(nn.Module):
    """Frozen Hamming STFT, power mel bank, dB clipping, and global CMVN."""

    def __init__(self, config: SpeechBrainCRDNNASRConfig) -> None:
        super().__init__()
        self.config = SpeechBrainCRDNNASRConfig.coerce(config)
        self.register_buffer(
            "window",
            torch.hamming_window(self.config.win_length),
            persistent=False,
        )
        self.register_buffer(
            "mel_filterbank",
            speechbrain_asr_mel_filterbank(self.config),
            persistent=False,
        )
        self.normalizer = SpeechBrainGlobalNormalizer(self.config)

    def _fbank(self, waveforms: Tensor) -> Tensor:
        values = waveforms.float()
        spectrum = torch.stft(
            values,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window.to(
                device=values.device,
                dtype=values.dtype,
            ),
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = (spectrum.real.square() + spectrum.imag.square()).transpose(1, 2)
        mel = power @ self.mel_filterbank.to(
            device=power.device,
            dtype=power.dtype,
        )
        decibels = 10.0 * torch.log10(mel.clamp_min(1e-10))
        return torch.maximum(
            decibels,
            decibels.max() - self.config.top_db,
        )

    def forward(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor | None = None,
        *,
        epoch: int = 0,
        update_statistics: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(waveforms, Tensor):
            raise TypeError("`waveforms` must be a torch.Tensor.")
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2 or waveforms.shape[-1] < self.config.win_length:
            raise ValueError(
                "SpeechBrain ASR waveforms must have shape [batch, samples] "
                "and contain at least one 25 ms window.")
        if waveform_lengths is None:
            lengths = torch.full(
                (waveforms.shape[0], ),
                waveforms.shape[-1],
                dtype=torch.long,
                device=waveforms.device,
            )
        else:
            lengths = torch.as_tensor(
                waveform_lengths,
                dtype=torch.long,
                device=waveforms.device,
            )
            if lengths.ndim != 1 or lengths.shape[0] != waveforms.shape[0]:
                raise ValueError("`waveform_lengths` must have shape [batch].")
            if (torch.any(lengths < self.config.win_length) or torch.any(lengths > waveforms.shape[-1])):
                raise ValueError("Every waveform length must be within the padded batch.")
        relative = lengths.to(dtype=torch.float32) / waveforms.shape[-1]
        # The pinned recipe computes features under no_grad and explicitly
        # detaches them before the CRDNN.
        with torch.no_grad():
            features = self._fbank(waveforms)
            normalized = self.normalizer(
                features,
                relative.to(features.device),
                epoch=epoch,
                update_statistics=update_statistics,
            )
        frame_lengths = torch.round(relative.to(normalized.device) * normalized.shape[1], ).long()
        return normalized.detach(), relative.to(normalized.device), frame_lengths


__all__ = [
    "SpeechBrainASRFrontend",
    "SpeechBrainGlobalNormalizer",
    "speechbrain_asr_mel_filterbank",
]
