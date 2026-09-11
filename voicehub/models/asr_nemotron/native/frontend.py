"""PyTorch-only Nemotron 3.5 log-mel frontend."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn

from voicehub.models.asr_nemotron.native.configuration import NemotronFrontendConfig
from voicehub.processing.audio import mel_filter_bank

LOG_ZERO_GUARD_VALUE = 2**-24


def _autocast_disabled(device_type: str):
    amp = getattr(torch, "amp", None)
    factory = getattr(amp, "autocast", None)
    if callable(factory):
        return factory(device_type, enabled=False)
    return nullcontext()


class NemotronLogMelFrontend(nn.Module):
    """Centered Hann STFT and Slaney-normalized power-mel extraction.

    The released processor masks the final centered STFT frame.  This class
    preserves that detail: the tensor contains the frame emitted by
    :func:`torch.stft`, while the returned attention mask marks only
    ``floor(samples / hop_length)`` frames as valid.
    """

    def __init__(self, config: NemotronFrontendConfig) -> None:
        super().__init__()
        self.config = (
            config if isinstance(config, NemotronFrontendConfig) else
            NemotronFrontendConfig.from_processor_dict(config))
        self.register_buffer(
            "window",
            torch.hann_window(
                self.config.win_length,
                periodic=False,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "mel_filters",
            mel_filter_bank(
                sample_rate=self.config.sampling_rate,
                n_fft=self.config.n_fft,
                n_mels=self.config.feature_size,
                # librosa constructs its Slaney edges in float64 and casts the
                # finished bank to float32.  Mirroring that order closes the
                # otherwise visible edge-rounding gap without importing it.
                dtype=torch.float64,
            ).to(dtype=torch.float32),
            persistent=False,
        )

    def feature_lengths(
        self,
        waveform_lengths: Tensor,
        *,
        center: bool,
    ) -> Tensor:
        lengths = torch.as_tensor(waveform_lengths)
        if center:
            output = torch.div(
                lengths,
                self.config.hop_length,
                rounding_mode="floor",
            )
        else:
            output = torch.div(
                lengths - self.config.n_fft,
                self.config.hop_length,
                rounding_mode="floor",
            ) + 1
        return output.to(dtype=torch.long)

    def forward(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor | None = None,
        *,
        sampling_rate: int,
        center: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if (isinstance(sampling_rate, bool) or not isinstance(sampling_rate, int) or
                sampling_rate != self.config.sampling_rate):
            raise ValueError("Nemotron 3.5 requires 16 kHz audio; received "
                             f"{sampling_rate!r}.")
        if not isinstance(center, bool):
            raise TypeError("`center` must be a boolean.")
        values = torch.as_tensor(waveforms)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2:
            raise ValueError("`waveforms` must have shape [samples] or [batch, samples].")
        if not values.is_floating_point():
            values = values.float()
        values = values.to(dtype=torch.float32)
        if waveform_lengths is None:
            lengths = torch.full(
                (values.shape[0], ),
                values.shape[1],
                dtype=torch.long,
                device=values.device,
            )
        else:
            lengths = torch.as_tensor(
                waveform_lengths,
                dtype=torch.long,
                device=values.device,
            )
        if lengths.shape != (values.shape[0], ):
            raise ValueError("`waveform_lengths` must have shape [batch].")
        if torch.any(lengths <= 0):
            raise ValueError("Nemotron waveforms must be non-empty.")
        if torch.any(lengths > values.shape[1]):
            raise ValueError("A waveform length exceeds the padded waveform tensor.")
        if not center and torch.any(lengths < self.config.n_fft):
            raise ValueError(
                "Non-centered Nemotron chunks require at least "
                f"{self.config.n_fft} samples.")

        positions = torch.arange(values.shape[1], device=values.device)
        valid_samples = positions.unsqueeze(0) < lengths.unsqueeze(1)
        emphasized = torch.cat(
            (
                values[:, :1],
                values[:, 1:] - self.config.preemphasis * values[:, :-1],
            ),
            dim=1,
        )
        emphasized = emphasized.masked_fill(~valid_samples, 0.0)

        with _autocast_disabled(values.device.type):
            spectrum = torch.stft(
                emphasized.float(),
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                window=self.window.float(),
                return_complex=True,
                pad_mode="constant",
                center=center,
            )
            magnitude = torch.view_as_real(spectrum)
            power = magnitude.square().sum(dim=-1)
            mel = torch.matmul(
                self.mel_filters.to(
                    device=power.device,
                    dtype=power.dtype,
                ),
                power,
            )
            features = torch.log(mel + LOG_ZERO_GUARD_VALUE)

        features = features.transpose(1, 2)
        feature_lengths = self.feature_lengths(lengths, center=center)
        frame_positions = torch.arange(
            features.shape[1],
            device=features.device,
        )
        attention_mask = (frame_positions.unsqueeze(0) < feature_lengths.unsqueeze(1))
        features = features * attention_mask.unsqueeze(-1)
        return features, attention_mask


__all__ = [
    "LOG_ZERO_GUARD_VALUE",
    "NemotronLogMelFrontend",
]
