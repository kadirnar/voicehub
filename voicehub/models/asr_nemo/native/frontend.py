"""PyTorch-only frontend matching NeMo FilterbankFeatures semantics."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.asr_nemo.native.configuration import NeMoQuartzNetCTCConfig
from voicehub.processing.audio import mel_filter_bank

_MAGNITUDE_GRADIENT_GUARD = 1e-5


def _autocast_disabled(device_type: str):
    autocast = getattr(torch, "amp", None)
    autocast_factory = getattr(autocast, "autocast", None)
    if callable(autocast_factory):
        return autocast_factory(device_type, enabled=False)
    return nullcontext()


class NeMoFilterbankFeatures(nn.Module):
    """Centered Hann STFT, Slaney power-mel, log, and per-bin normalization.

    ``window`` and ``fb`` preserve the names in the audited NeMo state
    dictionary. Loading the released checkpoint therefore replaces both
    derived buffers with their exact published values.
    """

    def __init__(self, config: NeMoQuartzNetCTCConfig) -> None:
        super().__init__()
        self.config = NeMoQuartzNetCTCConfig.coerce(config)
        self.register_buffer(
            "window",
            torch.hann_window(
                self.config.window_length,
                periodic=False,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "fb",
            mel_filter_bank(
                sample_rate=self.config.sampling_rate,
                n_fft=self.config.n_fft,
                n_mels=self.config.num_mel_bins,
                dtype=torch.float32,
            ).unsqueeze(0),
        )

    def feature_lengths(self, waveform_lengths: Tensor) -> Tensor:
        """Match NeMo's centered-STFT logical length (the final frame is
        masked)."""
        lengths = torch.as_tensor(waveform_lengths)
        return torch.div(
            lengths,
            self.config.hop_length,
            rounding_mode="floor",
        ).to(dtype=torch.long)

    @staticmethod
    def _normalize_per_feature(features: Tensor, lengths: Tensor) -> Tensor:
        frames = torch.arange(features.shape[-1], device=features.device)
        valid = frames.unsqueeze(0) < lengths.unsqueeze(1)
        counts = valid.sum(dim=1)
        if torch.any(counts < 2):
            raise ValueError(
                "Per-feature normalization requires at least two valid "
                "spectrogram frames per waveform.")
        mask = valid.unsqueeze(1)
        mean = torch.where(mask, features, 0.0).sum(dim=2)
        mean = mean / counts.unsqueeze(1)
        centered = features - mean.unsqueeze(2)
        variance = torch.where(mask, centered, 0.0).square().sum(dim=2)
        variance = variance / (counts.unsqueeze(1) - 1)
        standard_deviation = variance.sqrt() + 1e-5
        return centered / standard_deviation.unsqueeze(2)

    def _extract(
        self,
        waveforms: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        feature_lengths = self.feature_lengths(lengths)
        values = waveforms
        if self.training and self.config.dither:
            values = values + self.config.dither * torch.randn_like(values)

        sample_positions = torch.arange(values.shape[-1], device=values.device)
        valid_samples = sample_positions.unsqueeze(0) < lengths.unsqueeze(1)
        values = torch.cat(
            (
                values[:, :1],
                values[:, 1:] - self.config.preemphasis * values[:, :-1],
            ),
            dim=1,
        )
        values = values.masked_fill(~valid_samples, 0.0)

        with _autocast_disabled(values.device.type):
            spectrum = torch.stft(
                values.float(),
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.window_length,
                center=True,
                pad_mode="constant",
                window=self.window.float(),
                return_complex=True,
            )
        if self.config.frontend_gradients:
            magnitude = (torch.view_as_real(spectrum).square().sum(dim=-1) + _MAGNITUDE_GRADIENT_GUARD).sqrt()
        else:
            magnitude = spectrum.abs()
        power = magnitude.square()
        with _autocast_disabled(power.device.type):
            mel = torch.matmul(self.fb.to(dtype=power.dtype), power)
        features = torch.log(mel + self.config.log_guard)
        features = self._normalize_per_feature(features, feature_lengths)

        frame_positions = torch.arange(features.shape[-1], device=features.device)
        padding = frame_positions.unsqueeze(0) >= feature_lengths.unsqueeze(1)
        features = features.masked_fill(padding.unsqueeze(1), 0.0)
        remainder = features.shape[-1] % self.config.pad_to
        if remainder:
            features = functional.pad(
                features,
                (0, self.config.pad_to - remainder),
            )
        return features, feature_lengths

    def forward(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(waveforms, Tensor):
            raise TypeError("`waveforms` must be a PyTorch tensor.")
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2:
            raise ValueError("`waveforms` must have shape [batch, samples].")
        lengths = torch.as_tensor(
            waveform_lengths,
            dtype=torch.long,
            device=waveforms.device,
        )
        if lengths.ndim != 1 or lengths.shape[0] != waveforms.shape[0]:
            raise ValueError("`waveform_lengths` must have shape [batch].")
        if torch.any(lengths < self.config.minimum_input_samples):
            raise ValueError(
                "QuartzNet requires at least "
                f"{self.config.minimum_input_samples} samples per waveform.")
        if torch.any(lengths > waveforms.shape[-1]):
            raise ValueError("A waveform length exceeds the padded waveform size.")
        if self.config.frontend_gradients:
            return self._extract(waveforms, lengths)
        with torch.no_grad():
            return self._extract(waveforms, lengths)


class NeMoAudioPreprocessor(nn.Module):
    """Namespace-preserving wrapper for NeMo's preprocessor module."""

    def __init__(self, config: NeMoQuartzNetCTCConfig) -> None:
        super().__init__()
        self.featurizer = NeMoFilterbankFeatures(config)

    def forward(
        self,
        input_signal: Tensor,
        length: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.featurizer(input_signal, length)


class NeMoSpecCutout(nn.Module):
    """Torch-native rectangular masking matching the released recipe."""

    def __init__(self, config: NeMoQuartzNetCTCConfig) -> None:
        super().__init__()
        self.config = NeMoQuartzNetCTCConfig.coerce(config)

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        if self.config.spec_cutout_masks == 0:
            return features
        values = features.clone()
        frequency_limit = min(
            self.config.spec_cutout_frequency,
            features.shape[1],
        )
        with torch.no_grad():
            for batch_index in range(features.shape[0]):
                valid_time = min(
                    int(lengths[batch_index].item()),
                    features.shape[2],
                )
                time_limit = min(self.config.spec_cutout_time, valid_time)
                if frequency_limit == 0 or time_limit == 0:
                    continue
                for _ in range(self.config.spec_cutout_masks):
                    width_frequency = int(
                        torch.randint(
                            frequency_limit + 1,
                            (),
                            device=features.device,
                        ).item())
                    width_time = int(torch.randint(
                        time_limit + 1,
                        (),
                        device=features.device,
                    ).item())
                    start_frequency_limit = features.shape[1] - frequency_limit
                    start_time_limit = valid_time - time_limit
                    start_frequency = int(
                        torch.randint(
                            start_frequency_limit + 1,
                            (),
                            device=features.device,
                        ).item())
                    start_time = int(
                        torch.randint(
                            start_time_limit + 1,
                            (),
                            device=features.device,
                        ).item())
                    values[
                        batch_index,
                        start_frequency:start_frequency + width_frequency,
                        start_time:start_time + width_time,
                    ] = 0.0
        return values


__all__ = [
    "NeMoAudioPreprocessor",
    "NeMoFilterbankFeatures",
    "NeMoSpecCutout",
]
