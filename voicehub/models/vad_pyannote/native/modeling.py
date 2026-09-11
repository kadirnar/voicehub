"""VoiceHub-owned PyTorch implementation of PyanNet and Brouhaha.

The SincNet, LSTM, feed-forward, and classifier graphs preserve the
official state-dict namespace.  The implementation replaces Asteroid,
einops, NumPy, pyannote.audio, and Lightning with direct PyTorch
operations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.vad_pyannote.native.configuration import PyanNetConfig
from voicehub.models.vad_pyannote.native.powerset import Powerset

SNR_MIN_DB = -15.0
SNR_MAX_DB = 80.0
C50_MIN_DB = -10.0
C50_MAX_DB = 60.0


def _audio_tensor(waveforms: Tensor) -> Tensor:
    if not isinstance(waveforms, Tensor):
        raise TypeError("`waveforms` must be a PyTorch tensor.")
    if waveforms.ndim == 2:
        waveforms = waveforms.unsqueeze(1)
    if waveforms.ndim != 3 or waveforms.shape[1] != 1:
        raise ValueError("`waveforms` must have shape [batch, samples] or "
                         "[batch, 1, samples].")
    if waveforms.shape[0] < 1 or waveforms.shape[-1] < 1:
        raise ValueError("`waveforms` must contain non-empty audio.")
    if not waveforms.is_floating_point():
        raise TypeError("`waveforms` must use a floating-point dtype.")
    if not torch.isfinite(waveforms).all():
        raise ValueError("`waveforms` cannot contain NaN or infinite values.")
    return waveforms


class ParametricSincFilterbank(nn.Module):
    """Asteroid ParamSincFB v0.4.0 expressed directly in PyTorch."""

    def __init__(
        self,
        *,
        n_filters: int = 80,
        kernel_size: int = 251,
        stride: int = 10,
        sample_rate: int = 16_000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ) -> None:
        super().__init__()
        if n_filters != 80 or kernel_size != 251 or sample_rate != 16_000:
            raise ValueError(
                "Published PyanNet checkpoints require 80 filters, a 251 "
                "sample kernel, and 16 kHz audio.")
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.sample_rate = float(sample_rate)
        self.min_low_hz = float(min_low_hz)
        self.min_band_hz = float(min_band_hz)
        half_kernel = kernel_size // 2

        low_hz = 30.0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        low_mel = self.to_mel(low_hz)
        high_mel = self.to_mel(high_hz)
        mel = torch.linspace(
            low_mel,
            high_mel,
            n_filters // 2 + 1,
            dtype=torch.float32,
        )
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(hz[:-1].reshape(-1, 1))
        self.band_hz_ = nn.Parameter((hz[1:] - hz[:-1]).reshape(-1, 1))
        self.register_buffer(
            "window_",
            torch.hamming_window(
                kernel_size,
                periodic=False,
                dtype=torch.float32,
            )[:half_kernel],
        )
        self.register_buffer(
            "n_",
            2 * math.pi *
            (torch.arange(
                -half_kernel,
                0,
                dtype=torch.float32,
            ).reshape(1, -1) / self.sample_rate),
        )

    @staticmethod
    def to_mel(hz: float | Tensor) -> float | Tensor:
        if isinstance(hz, Tensor):
            return 2595.0 * torch.log10(1.0 + hz / 700.0)
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def to_hz(mel: float | Tensor) -> float | Tensor:
        if isinstance(mel, Tensor):
            return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)

    def _make_filters(
        self,
        low: Tensor,
        high: Tensor,
        *,
        odd: bool,
    ) -> Tensor:
        band = (high - low)[:, 0]
        ft_low = low @ self.n_
        ft_high = high @ self.n_
        denominator = self.n_ / 2.0
        if odd:
            left = ((torch.cos(ft_low) - torch.cos(ft_high)) / denominator) * self.window_
            center = torch.zeros_like(band.reshape(-1, 1))
            right = -torch.flip(left, dims=(1, ))
        else:
            left = ((torch.sin(ft_high) - torch.sin(ft_low)) / denominator) * self.window_
            center = 2.0 * band.reshape(-1, 1)
            right = torch.flip(left, dims=(1, ))
        values = torch.cat((left, center, right), dim=1)
        values = values / (2.0 * band[:, None])
        return values.reshape(self.n_filters // 2, 1, self.kernel_size)

    def filters(self) -> Tensor:
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            min=self.min_low_hz,
            max=self.sample_rate / 2.0,
        )
        return torch.cat(
            (
                self._make_filters(low, high, odd=False),
                self._make_filters(low, high, odd=True),
            ),
            dim=0,
        )

    def forward(self, waveforms: Tensor) -> Tensor:
        return functional.conv1d(
            waveforms,
            self.filters(),
            stride=self.stride,
        )


class SincEncoder(nn.Module):
    """Namespace-compatible wrapper around the parametric filterbank."""

    def __init__(self, *, stride: int, sample_rate: int) -> None:
        super().__init__()
        self.filterbank = ParametricSincFilterbank(
            stride=stride,
            sample_rate=sample_rate,
        )

    def forward(self, waveforms: Tensor) -> Tensor:
        return self.filterbank(waveforms)


class SincNet(nn.Module):
    """The three-stage PyanNet raw-waveform frontend."""

    minimum_samples = 1_261

    def __init__(self, *, sample_rate: int = 16_000, stride: int = 10) -> None:
        super().__init__()
        self.stride = stride
        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)
        self.conv1d = nn.ModuleList((
            SincEncoder(stride=stride, sample_rate=sample_rate),
            nn.Conv1d(80, 60, 5),
            nn.Conv1d(60, 60, 5),
        ))
        self.pool1d = nn.ModuleList(nn.MaxPool1d(3, stride=3) for _ in range(3))
        self.norm1d = nn.ModuleList((
            nn.InstanceNorm1d(80, affine=True),
            nn.InstanceNorm1d(60, affine=True),
            nn.InstanceNorm1d(60, affine=True),
        ))

    def forward(self, waveforms: Tensor) -> Tensor:
        if waveforms.shape[-1] < self.minimum_samples:
            raise ValueError("PyanNet requires at least "
                             f"{self.minimum_samples} waveform samples.")
        outputs = self.wav_norm1d(waveforms)
        for index, (convolution, pooling, normalization) in enumerate(zip(self.conv1d, self.pool1d,
                                                                          self.norm1d)):
            outputs = convolution(outputs)
            if index == 0:
                outputs = torch.abs(outputs)
            outputs = functional.leaky_relu(normalization(pooling(outputs)))
        return outputs

    @staticmethod
    def num_frames(num_samples: int, *, stride: int = 10) -> int:
        if (isinstance(num_samples, bool) or not isinstance(num_samples, int) or
                num_samples < SincNet.minimum_samples):
            raise ValueError(f"`num_samples` must be at least {SincNet.minimum_samples}.")
        length = (num_samples - 251) // stride + 1
        length = (length - 3) // 3 + 1
        length = length - 5 + 1
        length = (length - 3) // 3 + 1
        length = length - 5 + 1
        return (length - 3) // 3 + 1


class ParametricSigmoid(nn.Module):
    """Map logits to a closed physical-value interval."""

    def __init__(self, start: float, end: float) -> None:
        super().__init__()
        self.start = float(start)
        self.end = float(end)

    def forward(self, values: Tensor) -> Tensor:
        return (self.end - self.start) * torch.sigmoid(values) + self.start


class BrouhahaClassifier(nn.Module):
    """Checkpoint-compatible VAD, SNR, and C50 heads."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linears = nn.ModuleDict({name: nn.Linear(in_features, 1) for name in ("vad", "snr", "c50")})

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        return {name: layer(features) for name, layer in self.linears.items()}


class BrouhahaActivation(nn.Module):
    """Output transforms from the pinned Brouhaha implementation."""

    def __init__(self) -> None:
        super().__init__()
        self.activations = nn.ModuleDict({
            "vad": nn.Sigmoid(),
            "snr": ParametricSigmoid(SNR_MAX_DB, SNR_MIN_DB),
            "c50": ParametricSigmoid(C50_MAX_DB, C50_MIN_DB),
        })

    def forward(self, values: Mapping[str, Tensor]) -> Tensor:
        return torch.cat(
            tuple(activation(values[name]) for name, activation in self.activations.items()),
            dim=-1,
        )


@dataclass(frozen=True, slots=True)
class PyanNetOutput:
    """Differentiable training output for VoiceHub's generic trainer."""

    loss: Tensor
    logits: Tensor
    probabilities: Tensor
    loss_vad: Tensor | None = None
    loss_snr: Tensor | None = None
    loss_c50: Tensor | None = None


class PyanNet(nn.Module):
    """Native PyanNet graph for segmentation, powerset, or Brouhaha."""

    def __init__(
        self,
        config: PyanNetConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = PyanNetConfig.coerce(config or {})
        self.sincnet = SincNet(
            sample_rate=self.config.sampling_rate,
            stride=self.config.sinc_stride,
        )
        self.lstm = nn.LSTM(
            60,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            bidirectional=self.config.lstm_bidirectional,
            dropout=self.config.lstm_dropout,
            batch_first=True,
        )
        input_size = self.config.lstm_hidden_size * (2 if self.config.lstm_bidirectional else 1)
        self.linear = nn.ModuleList()
        for _ in range(self.config.linear_num_layers):
            self.linear.append(nn.Linear(input_size, self.config.linear_hidden_size))
            input_size = self.config.linear_hidden_size
        if self.config.is_brouhaha:
            self.classifier = BrouhahaClassifier(input_size)
            self.activation = BrouhahaActivation()
        else:
            self.classifier = nn.Linear(input_size, self.config.output_size)
            self.activation = (nn.Softmax(dim=-1) if self.config.is_powerset else nn.Sigmoid())
        self.powerset = (
            Powerset(
                self.config.num_classes,
                self.config.max_active_classes,
            ) if self.config.is_powerset else None)

    def frame_count(self, num_samples: int) -> int:
        return self.sincnet.num_frames(
            num_samples,
            stride=self.config.sinc_stride,
        )

    def _features(self, waveforms: Tensor) -> Tensor:
        waveforms = _audio_tensor(waveforms)
        parameter = next(self.parameters())
        if waveforms.device != parameter.device:
            raise ValueError("Model parameters and waveforms must share a device.")
        if waveforms.dtype != parameter.dtype:
            waveforms = waveforms.to(dtype=parameter.dtype)
        outputs = self.sincnet(waveforms).transpose(1, 2)
        outputs, _ = self.lstm(outputs)
        for linear in self.linear:
            outputs = functional.leaky_relu(linear(outputs))
        return outputs

    def _scores(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if self.config.is_brouhaha:
            raw = self.classifier(features)
            logits = torch.cat(
                tuple(raw[name] for name in ("vad", "snr", "c50")),
                dim=-1,
            )
            return logits, self.activation(raw)
        logits = self.classifier(features)
        return logits, self.activation(logits)

    def speech_probabilities(self, probabilities: Tensor) -> Tensor:
        """Project model outputs to the score consumed by VAD hysteresis."""
        if self.powerset is not None:
            return self.powerset.to_speech(probabilities)
        if self.config.is_brouhaha:
            return probabilities[..., 0]
        return probabilities.amax(dim=-1)

    def forward(
        self,
        waveforms: Tensor | None = None,
        *,
        input_values: Tensor | None = None,
        labels: Tensor | None = None,
        frame_weights: Tensor | None = None,
        snr_loss_scale: float = 1.0,
        c50_loss_scale: float = 1.0,
    ) -> Tensor | PyanNetOutput:
        if waveforms is None:
            waveforms = input_values
        elif input_values is not None:
            raise TypeError("Pass either `waveforms` or `input_values`, not both.")
        if waveforms is None:
            raise TypeError("PyanNet requires `waveforms` or `input_values`.")
        features = self._features(waveforms)
        logits, probabilities = self._scores(features)
        if labels is None:
            return probabilities

        from voicehub.models.vad_pyannote.native.objective import pyannet_loss

        losses = pyannet_loss(
            config=self.config,
            logits=logits,
            probabilities=probabilities,
            labels=labels,
            frame_weights=frame_weights,
            snr_loss_scale=snr_loss_scale,
            c50_loss_scale=c50_loss_scale,
        )
        return PyanNetOutput(
            loss=losses["loss"],
            logits=logits,
            probabilities=probabilities,
            loss_vad=losses.get("loss_vad"),
            loss_snr=losses.get("loss_snr"),
            loss_c50=losses.get("loss_c50"),
        )


__all__ = [
    "C50_MAX_DB",
    "C50_MIN_DB",
    "SNR_MAX_DB",
    "SNR_MIN_DB",
    "BrouhahaActivation",
    "BrouhahaClassifier",
    "ParametricSigmoid",
    "ParametricSincFilterbank",
    "PyanNet",
    "PyanNetOutput",
    "SincNet",
]
