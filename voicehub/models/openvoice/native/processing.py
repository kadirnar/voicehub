"""Dependency-free waveform and spectrogram processing for OpenVoice V2."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from voicehub.models.openvoice.native.configuration import OpenVoiceConverterConfig


@dataclass(frozen=True, slots=True)
class OpenVoiceSpectrogramBatch:
    """Right-padded magnitudes and their valid frame lengths."""

    values: Tensor
    lengths: Tensor


@dataclass(frozen=True, slots=True)
class OpenVoiceWaveformBatch:
    """Right-padded mono waveforms and their valid sample lengths."""

    values: Tensor
    lengths: Tensor


class OpenVoiceAudioProcessor:
    """Create the exact released magnitude-spectrogram representation."""

    def __init__(self, config: OpenVoiceConverterConfig) -> None:
        if not isinstance(config, OpenVoiceConverterConfig):
            raise TypeError("`config` must be an OpenVoiceConverterConfig instance.")
        self.config = config

    @staticmethod
    def waveforms(value: Any) -> tuple[Tensor, ...]:
        """Normalize one waveform or a variable-length waveform batch."""
        if isinstance(value, Tensor):
            if value.ndim == 1:
                result = (value, )
            elif value.ndim == 2:
                result = tuple(value[index] for index in range(value.shape[0]))
            else:
                raise ValueError("OpenVoice waveforms must have shape [samples] or "
                                 "[batch, samples].")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                raise ValueError("OpenVoice waveform input cannot be empty.")
            if isinstance(value[0], (int, float)):
                result = (torch.as_tensor(value), )
            else:
                result = tuple(torch.as_tensor(item) for item in value)
        else:
            result = (torch.as_tensor(value), )
        normalized = []
        for waveform in result:
            if waveform.ndim != 1 or waveform.numel() == 0:
                raise ValueError("Every OpenVoice waveform must be non-empty and mono.")
            waveform = waveform.float()
            if not bool(torch.isfinite(waveform).all()):
                raise ValueError("OpenVoice waveform contains NaN or infinity.")
            normalized.append(waveform)
        return tuple(normalized)

    def waveform_batch(
        self,
        waveforms: Any,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> OpenVoiceWaveformBatch:
        """Return a finite, right-padded waveform batch."""
        values = tuple(waveform.to(device=device, dtype=dtype) for waveform in self.waveforms(waveforms))
        lengths = torch.tensor(
            [value.numel() for value in values],
            device=device,
            dtype=torch.long,
        )
        maximum = int(lengths.max().item())
        return OpenVoiceWaveformBatch(
            values=torch.stack([F.pad(value, (0, maximum - value.numel())) for value in values]),
            lengths=lengths,
        )

    def _spectrogram(self, waveform: Tensor) -> Tensor:
        padding = (self.config.n_fft - self.config.hop_length) // 2
        if waveform.numel() <= padding:
            raise ValueError("OpenVoice audio is too short for reflected STFT padding.")
        padded = F.pad(
            waveform[None, None],
            (padding, padding),
            mode="reflect",
        ).squeeze(0)
        window = torch.hann_window(
            self.config.win_length,
            device=waveform.device,
            dtype=waveform.dtype,
        )
        spectrum = torch.stft(
            padded,
            self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-6).squeeze(0)

    def spectrogram(
        self,
        waveforms: Any,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> OpenVoiceSpectrogramBatch:
        """Convert one or more mono waveforms into right-padded magnitudes."""
        values = tuple(
            self._spectrogram(waveform.to(device=device, dtype=dtype))
            for waveform in self.waveforms(waveforms))
        lengths = torch.tensor(
            [value.shape[-1] for value in values],
            device=device,
            dtype=torch.long,
        )
        maximum = int(lengths.max().item())
        padded = torch.stack([F.pad(value, (0, maximum - value.shape[-1])) for value in values])
        return OpenVoiceSpectrogramBatch(padded, lengths)

    def equal_reference_segments(
        self,
        waveform: Tensor,
        *,
        segment_seconds: float = 10.0,
    ) -> tuple[Tensor, ...]:
        """Split active reference audio like the released equal splitter.

        This method intentionally does not pretend to reproduce the
        upstream external Silero-VAD step. Callers may provide already
        trimmed speech or run a VoiceHub-native VAD before this
        processor.
        """
        waveform = self.waveforms(waveform)[0]
        if (isinstance(segment_seconds, bool) or not isinstance(segment_seconds, (int, float)) or
                not math.isfinite(float(segment_seconds)) or segment_seconds <= 0):
            raise ValueError("`segment_seconds` must be finite and positive.")
        duration = waveform.numel() / self.config.sample_rate
        number = max(1, round(duration / float(segment_seconds)))
        boundaries = [round(index * waveform.numel() / number) for index in range(number + 1)]
        return tuple(
            waveform[boundaries[index]:boundaries[index + 1]] for index in range(number)
            if boundaries[index + 1] > boundaries[index])


__all__ = [
    "OpenVoiceAudioProcessor",
    "OpenVoiceSpectrogramBatch",
    "OpenVoiceWaveformBatch",
]
