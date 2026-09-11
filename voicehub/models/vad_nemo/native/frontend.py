"""PyTorch-only frontend matching NeMo's released FilterbankFeatures graph."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.vad_nemo.native.configuration import MarbleNetVADConfig
from voicehub.processing.audio import mel_filter_bank


class MarbleNetFilterbankFeatures(nn.Module):
    """Released 25 ms Hann, pre-emphasis, power-mel, log frontend.

    Buffer names intentionally match the official `.nemo` state
    dictionary so conversion is an identity mapping rather than a
    heuristic rename.
    """

    def __init__(self, config: MarbleNetVADConfig) -> None:
        super().__init__()
        self.config = MarbleNetVADConfig.coerce(config)
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
        lengths = torch.as_tensor(waveform_lengths)
        return (torch.div(lengths, self.config.hop_length, rounding_mode="floor") + 1).to(dtype=torch.long)

    def _extract(self, waveforms: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        feature_lengths = self.feature_lengths(lengths)
        values = waveforms
        if self.training and self.config.dither:
            values = values + self.config.dither * torch.randn_like(values)
        values = torch.cat(
            (
                values[:, :1],
                values[:, 1:] - self.config.preemphasis * values[:, :-1],
            ),
            dim=1,
        )
        autocast = getattr(torch, "amp", None)
        autocast_factory = getattr(autocast, "autocast", None)
        context = (
            autocast_factory(values.device.type, enabled=False)
            if callable(autocast_factory) else nullcontext())
        with context:
            spectrum = torch.stft(
                values.float(),
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.window_length,
                center=True,
                pad_mode="reflect",
                window=self.window.float(),
                return_complex=True,
            )
        magnitude = spectrum.abs()
        power = magnitude.square()
        mel = torch.matmul(self.fb.to(dtype=power.dtype), power)
        features = torch.log(mel + self.config.log_guard)

        time = torch.arange(features.shape[-1], device=features.device)
        padding_mask = time.unsqueeze(0) >= feature_lengths.unsqueeze(1)
        features = features.masked_fill(padding_mask.unsqueeze(1), 0.0)
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
        if torch.any(lengths < self.config.window_length):
            raise ValueError("MarbleNet VAD requires at least 400 samples (25 ms) per waveform.")
        if torch.any(lengths > waveforms.shape[-1]):
            raise ValueError("A waveform length exceeds the padded waveform size.")
        if self.config.frontend_gradients:
            return self._extract(waveforms, lengths)
        with torch.no_grad():
            return self._extract(waveforms, lengths)


class MarbleNetAudioPreprocessor(nn.Module):
    """Namespace-preserving wrapper for the released `preprocessor` module."""

    def __init__(self, config: MarbleNetVADConfig) -> None:
        super().__init__()
        self.featurizer = MarbleNetFilterbankFeatures(config)

    def forward(
        self,
        input_signal: Tensor,
        length: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.featurizer(input_signal, length)


class MarbleNetSpecAugment(nn.Module):
    """Vectorized source-compatible frequency and adaptive time masking."""

    def __init__(self, config: MarbleNetVADConfig) -> None:
        super().__init__()
        self.config = MarbleNetVADConfig.coerce(config)

    @staticmethod
    def _apply_masks(
        features: Tensor,
        *,
        count: int,
        lengths: Tensor,
        width: int | Tensor,
        axis: int,
    ) -> Tensor:
        if count == 0:
            return features
        batch = features.shape[0]
        axis_length = features.shape[axis]
        if isinstance(width, Tensor):
            maximum_width = torch.clamp(width, max=axis_length).unsqueeze(1)
        else:
            maximum_width = width
        mask_width = (
            torch.rand(
                batch,
                count,
                dtype=torch.float32,
                device=features.device,
            ) * maximum_width).long()
        mask_start = torch.rand(
            batch,
            count,
            dtype=torch.float32,
            device=features.device,
        )
        if axis == 2:
            mask_start = mask_start * (lengths.unsqueeze(1) - mask_width)
        else:
            mask_start = mask_start * (axis_length - mask_width)
        mask_start = mask_start.long()
        mask_end = mask_start + mask_width
        indices = torch.arange(axis_length, device=features.device)
        ranges = ((indices >= mask_start.unsqueeze(-1)) & (indices < mask_end.unsqueeze(-1))).any(dim=1)
        mask = ranges.unsqueeze(1) if axis == 2 else ranges.unsqueeze(2)
        return features.masked_fill(mask, 0.0)

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        with torch.no_grad():
            values = self._apply_masks(
                features,
                count=self.config.spec_augment_time_masks,
                lengths=lengths,
                width=self.config.spec_augment_time_width * lengths,
                axis=2,
            )
            return self._apply_masks(
                values,
                count=self.config.spec_augment_frequency_masks,
                lengths=lengths,
                width=self.config.spec_augment_frequency_width,
                axis=1,
            )


__all__ = [
    "MarbleNetAudioPreprocessor",
    "MarbleNetFilterbankFeatures",
    "MarbleNetSpecAugment",
]
