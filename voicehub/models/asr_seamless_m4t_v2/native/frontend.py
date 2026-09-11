"""Audited PyTorch-only SeamlessM4T-v2 Kaldi-style feature frontend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from voicehub.models.asr_seamless_m4t_v2.native.configuration import SeamlessM4Tv2S2TConfig


def _hertz_to_kaldi_mel(value: Tensor) -> Tensor:
    return 1127.0 * torch.log1p(value / 700.0)


def _mel_filter_bank(config: SeamlessM4Tv2S2TConfig) -> Tensor:
    frequency_bins = config.feature_fft_size // 2 + 1
    frequencies = (
        torch.arange(frequency_bins, dtype=torch.float64) * config.sampling_rate / config.feature_fft_size)
    mel_frequencies = _hertz_to_kaldi_mel(frequencies)
    mel_min = _hertz_to_kaldi_mel(torch.tensor(20.0, dtype=torch.float64))
    mel_max = _hertz_to_kaldi_mel(torch.tensor(
        config.sampling_rate / 2,
        dtype=torch.float64,
    ))
    edges = torch.linspace(
        mel_min,
        mel_max,
        config.num_mel_bins + 2,
        dtype=torch.float64,
    )
    lower = (mel_frequencies[:, None] - edges[:-2]) / (edges[1:-1] - edges[:-2])
    upper = (edges[2:] - mel_frequencies[:, None]) / (edges[2:] - edges[1:-1])
    return torch.clamp(torch.minimum(lower, upper), min=0.0)


def _povey_window(length: int) -> Tensor:
    return torch.hann_window(
        length,
        periodic=False,
        dtype=torch.float64,
    ).pow(0.85)


@dataclass(frozen=True, slots=True)
class SeamlessM4Tv2FeatureBatch:
    input_features: Tensor
    attention_mask: Tensor

    def to(
        self,
        *,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> SeamlessM4Tv2FeatureBatch:
        features = self.input_features.to(
            device=device,
            dtype=dtype,
        )
        return SeamlessM4Tv2FeatureBatch(
            input_features=features,
            attention_mask=self.attention_mask.to(device=device),
        )


class SeamlessM4Tv2FeatureExtractor:
    """Reproduce the published 80-bin, stride-two feature contract."""

    def __init__(self, config: SeamlessM4Tv2S2TConfig) -> None:
        if not isinstance(config, SeamlessM4Tv2S2TConfig):
            raise TypeError("`config` must be SeamlessM4Tv2S2TConfig.")
        self.config = config
        self._mel_filters = _mel_filter_bank(config)
        self._window = _povey_window(config.feature_window_length)

    @property
    def minimum_samples(self) -> int:
        # At least two frames are required for the released ddof=1 CMVN.
        return (self.config.feature_window_length + self.config.feature_hop_length)

    def _validate_waveform(self, waveform: Tensor) -> Tensor:
        if not isinstance(waveform, Tensor):
            raise TypeError("Waveforms must be PyTorch tensors.")
        if waveform.ndim == 2:
            # The reference frontend explicitly takes the first channel.
            waveform = waveform[0]
        if waveform.ndim != 1:
            raise ValueError("A waveform must have shape [samples] or [channels, samples].")
        if waveform.numel() < self.minimum_samples:
            raise ValueError("SeamlessM4T-v2 requires at least "
                             f"{self.minimum_samples} waveform samples.")
        if not waveform.is_floating_point():
            waveform = waveform.to(dtype=torch.float32)
        if not torch.isfinite(waveform).all():
            raise ValueError("Waveforms cannot contain NaN or infinite values.")
        return waveform

    def _extract_one(self, waveform: Tensor) -> Tensor:
        waveform = self._validate_waveform(waveform)
        # The reference first materializes float32 waveform samples, applies
        # Kaldi's signed-16-bit scale, and then promotes the FFT graph.
        compute = (waveform.to(dtype=torch.float32) * (2**15)).to(dtype=torch.float64)
        frames = compute.unfold(
            0,
            self.config.feature_window_length,
            self.config.feature_hop_length,
        ).clone()
        frames = frames - frames.mean(dim=-1, keepdim=True)
        original = frames.clone()
        frames[:, 1:] = (original[:, 1:] - self.config.feature_preemphasis * original[:, :-1])
        frames[:, 0] = original[:, 0] * (1.0 - self.config.feature_preemphasis)
        frames = frames * self._window.to(
            device=frames.device,
            dtype=frames.dtype,
        )
        if self.config.feature_fft_size > self.config.feature_window_length:
            frames = torch.nn.functional.pad(
                frames,
                (
                    0,
                    self.config.feature_fft_size - self.config.feature_window_length,
                ),
            )
        spectrum = torch.fft.rfft(
            frames,
            n=self.config.feature_fft_size,
            dim=-1,
        )
        power = spectrum.abs().square()
        mel = torch.matmul(
            power,
            self._mel_filters.to(
                device=power.device,
                dtype=power.dtype,
            ),
        )
        mel = mel.clamp_min(self.config.feature_mel_floor).log()
        variance = mel.var(
            dim=0,
            correction=1,
            keepdim=True,
        )
        normalized = (mel - mel.mean(dim=0, keepdim=True)) / torch.sqrt(variance + 1e-7)
        return normalized.to(dtype=torch.float32)

    def __call__(
        self,
        waveforms: Tensor | Sequence[Tensor],
        *,
        sampling_rate: int,
    ) -> SeamlessM4Tv2FeatureBatch:
        if (isinstance(sampling_rate, bool) or not isinstance(sampling_rate, int)):
            raise TypeError("`sampling_rate` must be an integer.")
        if sampling_rate != self.config.sampling_rate:
            raise ValueError(
                "SeamlessM4T-v2 requires "
                f"{self.config.sampling_rate} Hz audio, found {sampling_rate}.")
        if isinstance(waveforms, Tensor):
            rows = (waveforms, )
        elif (isinstance(waveforms, Sequence) and not isinstance(waveforms, (str, bytes))):
            rows = tuple(waveforms)
        else:
            raise TypeError("`waveforms` must be a tensor or a sequence of tensors.")
        if not rows:
            raise ValueError("A feature batch cannot be empty.")
        features = tuple(self._extract_one(row) for row in rows)
        maximum = max(value.shape[0] for value in features)
        maximum = int(math.ceil(maximum / self.config.feature_stride) * self.config.feature_stride)
        padded = []
        masks = []
        for value in features:
            length = value.shape[0]
            padded.append(torch.nn.functional.pad(
                value,
                (0, 0, 0, maximum - length),
            ))
            mask = torch.arange(
                maximum,
                device=value.device,
            ) < length
            masks.append(mask)
        raw = torch.stack(padded)
        raw_mask = torch.stack(masks)
        batch = raw.shape[0]
        stacked = raw.reshape(
            batch,
            maximum // self.config.feature_stride,
            self.config.feature_projection_input_dim,
        )
        # The reference implementation selects the second frame of every
        # stack as the visibility indicator.
        stacked_mask = raw_mask[:, self.config.feature_stride - 1::self.config.feature_stride]
        return SeamlessM4Tv2FeatureBatch(
            input_features=stacked,
            attention_mask=stacked_mask.to(dtype=torch.long),
        )


__all__ = [
    "SeamlessM4Tv2FeatureBatch",
    "SeamlessM4Tv2FeatureExtractor",
]
