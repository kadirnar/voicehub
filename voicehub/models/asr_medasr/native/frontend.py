"""PyTorch-only log-mel frontend used by the released MedASR model."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.asr_medasr.native.configuration import MedASRConfig


def _hertz_to_kaldi_mel(frequencies: Tensor) -> Tensor:
    """Convert Hz to the Kaldi/HTK mel scale in float64."""
    return 1127.0 * torch.log1p(frequencies / 700.0)


def medasr_mel_filter_bank(
    config: MedASRConfig,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Build the exact triangular filter bank from the LASR frontend.

    The reference implementation excludes the DC bin and performs filter
    construction in float64 before the eventual feature output is cast
    to float32.
    """
    spectrogram_bins = config.feature_fft_size // 2 + 1
    linear_frequencies = torch.linspace(
        0.0,
        config.sampling_rate / 2.0,
        spectrogram_bins,
        dtype=torch.float64,
        device=device,
    )
    bin_mels = _hertz_to_kaldi_mel(linear_frequencies[1:]).unsqueeze(1)
    lower_mel = _hertz_to_kaldi_mel(
        torch.tensor(
            config.feature_lower_hertz,
            dtype=torch.float64,
            device=device,
        ))
    upper_mel = _hertz_to_kaldi_mel(
        torch.tensor(
            config.feature_upper_hertz,
            dtype=torch.float64,
            device=device,
        ))
    edges = torch.linspace(
        lower_mel,
        upper_mel,
        config.num_mel_bins + 2,
        dtype=torch.float64,
        device=device,
    )
    lower_slopes = ((bin_mels - edges[:-2].unsqueeze(0)) / (edges[1:-1] - edges[:-2]).unsqueeze(0))
    upper_slopes = ((edges[2:].unsqueeze(0) - bin_mels) / (edges[2:] - edges[1:-1]).unsqueeze(0))
    filters = torch.maximum(
        torch.zeros((), dtype=torch.float64, device=device),
        torch.minimum(lower_slopes, upper_slopes),
    )
    return functional.pad(filters, (0, 0, 1, 0))


def feature_frame_lengths(
    waveform_lengths: Tensor,
    config: MedASRConfig,
) -> Tensor:
    if not isinstance(waveform_lengths, Tensor):
        raise TypeError("`waveform_lengths` must be a PyTorch tensor.")
    if waveform_lengths.ndim != 1:
        raise ValueError("`waveform_lengths` must have shape [batch].")
    if (waveform_lengths.dtype == torch.bool or waveform_lengths.is_floating_point() or
            waveform_lengths.is_complex()):
        raise TypeError("`waveform_lengths` must use an integer dtype.")
    lengths = torch.div(
        waveform_lengths.to(dtype=torch.long) - config.feature_window_length,
        config.feature_hop_length,
        rounding_mode="floor",
    ) + 1
    if (lengths < 1).any():
        raise ValueError("Every waveform must contain at least one complete feature "
                         "window.")
    return lengths


def subsampled_lengths(
    feature_lengths: Tensor,
    config: MedASRConfig,
) -> Tensor:
    if not isinstance(feature_lengths, Tensor) or feature_lengths.ndim != 1:
        raise ValueError("`feature_lengths` must have shape [batch].")
    if (feature_lengths.dtype == torch.bool or feature_lengths.is_floating_point() or
            feature_lengths.is_complex()):
        raise TypeError("`feature_lengths` must use an integer dtype.")
    lengths = feature_lengths.to(dtype=torch.long)
    for _ in range(2):
        lengths = torch.div(
            lengths - config.subsampling_conv_kernel_size,
            config.subsampling_conv_stride,
            rounding_mode="floor",
        ) + 1
    if (lengths < 1).any():
        raise ValueError("Every utterance must produce at least one LASR encoder frame.")
    return lengths


def lengths_to_mask(lengths: Tensor, maximum: int) -> Tensor:
    if (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1):
        raise ValueError("`maximum` must be a positive integer.")
    return (torch.arange(maximum, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1))


class MedASRFeatureExtractor(nn.Module):
    """Extract unnormalized 128-bin log-mel features from mono waveforms."""

    def __init__(
        self,
        config: MedASRConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = MedASRConfig.coerce(config)
        self.register_buffer(
            "mel_filters",
            medasr_mel_filter_bank(self.config),
            persistent=False,
        )

    @staticmethod
    def _waveform(value: Any) -> Tensor:
        if isinstance(value, Tensor):
            waveform = value
        else:
            try:
                waveform = torch.as_tensor(value)
            except (TypeError, ValueError, RuntimeError) as error:
                raise TypeError("MedASR audio must contain numeric samples.") from error
        if waveform.ndim != 1:
            raise ValueError("MedASR feature extraction requires one mono waveform.")
        if waveform.dtype == torch.bool or waveform.is_complex():
            raise TypeError("MedASR waveform samples must be real numeric values.")
        waveform = waveform.to(dtype=torch.float32)
        if waveform.numel() == 0:
            raise ValueError("MedASR waveforms cannot be empty.")
        if not torch.isfinite(waveform).all():
            raise ValueError("MedASR waveforms cannot contain NaN or infinite values.")
        return waveform

    def forward(
        self,
        waveforms: Tensor | Sequence[Any],
        *,
        waveform_lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if isinstance(waveforms, Tensor):
            if waveforms.ndim == 1:
                rows = (self._waveform(waveforms), )
            elif waveforms.ndim == 2:
                rows = tuple(self._waveform(waveforms[index]) for index in range(waveforms.shape[0]))
            else:
                raise ValueError("`waveforms` must have shape [samples] or "
                                 "[batch, samples].")
        else:
            if isinstance(waveforms, (str, bytes)) or not isinstance(
                    waveforms,
                    Sequence,
            ):
                raise TypeError("`waveforms` must be a tensor or sequence of waveforms.")
            rows = tuple(self._waveform(value) for value in waveforms)
        if not rows:
            raise ValueError("A MedASR batch cannot be empty.")

        if waveform_lengths is None:
            lengths = torch.tensor(
                [row.numel() for row in rows],
                dtype=torch.long,
            )
        else:
            if (not isinstance(waveform_lengths, Tensor) or waveform_lengths.ndim != 1 or
                    waveform_lengths.shape[0] != len(rows)):
                raise ValueError("`waveform_lengths` must have one value per waveform.")
            if (waveform_lengths.dtype == torch.bool or waveform_lengths.is_floating_point() or
                    waveform_lengths.is_complex()):
                raise TypeError("`waveform_lengths` must use an integer dtype.")
            lengths = waveform_lengths.detach().to(
                device="cpu",
                dtype=torch.long,
            )
            for index, (row, length) in enumerate(zip(rows, lengths)):
                amount = int(length.item())
                if amount < 1 or amount > row.numel():
                    raise ValueError(f"Waveform length {index} is outside its sample "
                                     "buffer.")
            rows = tuple(row[:int(length.item())] for row, length in zip(rows, lengths))

        minimum = self.config.minimum_input_samples
        padded_lengths = torch.clamp(lengths, min=minimum)
        maximum = int(padded_lengths.max().item())
        padded = torch.stack([functional.pad(row, (0, maximum - row.numel())) for row in rows])
        window = torch.hann_window(
            self.config.feature_window_length,
            periodic=False,
            dtype=torch.float64,
            device=padded.device,
        )
        frames = padded.to(torch.float64).unfold(
            -1,
            self.config.feature_window_length,
            self.config.feature_hop_length,
        )
        spectrum = torch.fft.rfft(
            frames * window,
            n=self.config.feature_fft_size,
        )
        power = spectrum.abs().square()
        filters = self.mel_filters.to(
            device=power.device,
            dtype=torch.float64,
        )
        features = torch.log(torch.clamp(power @ filters, min=1e-5), ).to(dtype=torch.float32)
        valid_feature_lengths = feature_frame_lengths(
            padded_lengths.to(device=features.device),
            self.config,
        )
        attention_mask = lengths_to_mask(
            valid_feature_lengths,
            features.shape[1],
        )
        return {
            "input_features": features,
            "attention_mask": attention_mask,
        }


__all__ = [
    "MedASRFeatureExtractor",
    "feature_frame_lengths",
    "lengths_to_mask",
    "medasr_mel_filter_bank",
    "subsampled_lengths",
]
