"""Pure-PyTorch frontend matching SpeechBrain's 2021 VAD feature graph."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from voicehub.models.vad_speechbrain.native.configuration import SpeechBrainCRDNNVADConfig


def speechbrain_mel_filterbank(config: SpeechBrainCRDNNVADConfig) -> Tensor:
    """Construct the legacy SpeechBrain triangular mel matrix."""
    mel_min = 2_595.0 * math.log10(1.0 + config.f_min / 700.0)
    mel_max = 2_595.0 * math.log10(1.0 + config.f_max / 700.0)
    mel = torch.linspace(mel_min, mel_max, config.n_mels + 2)
    hz = 700.0 * (10.0**(mel / 2_595.0) - 1.0)
    bands = hz[1:] - hz[:-1]
    central = hz[1:-1]
    band = bands[:-1]
    frequencies = torch.linspace(
        0.0,
        float(config.sampling_rate // 2),
        config.n_fft // 2 + 1,
    )
    slope = (frequencies.unsqueeze(0) - central.unsqueeze(1)) / band.unsqueeze(1)
    filters = torch.maximum(
        torch.zeros(1, dtype=frequencies.dtype),
        torch.minimum(slope + 1.0, -slope + 1.0),
    )
    return filters.transpose(0, 1).contiguous()


class SpeechBrainVADFrontend(nn.Module):
    """Frozen Hamming-STFT, power mel-bank, dB, and sentence CMVN."""

    def __init__(self, config: SpeechBrainCRDNNVADConfig) -> None:
        super().__init__()
        self.config = SpeechBrainCRDNNVADConfig.coerce(config)
        self.register_buffer(
            "window",
            torch.hamming_window(self.config.win_length),
            persistent=False,
        )
        self.register_buffer(
            "mel_filterbank",
            speechbrain_mel_filterbank(self.config),
            persistent=False,
        )

    def frame_count(self, sample_count: int) -> int:
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise TypeError("`sample_count` must be an integer.")
        if sample_count < 1:
            raise ValueError("`sample_count` must be positive.")
        return sample_count // self.config.hop_length + 1

    def _features(self, waveforms: Tensor) -> Tensor:
        values = waveforms.float()
        spectrum = torch.stft(
            values,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window.to(device=values.device, dtype=values.dtype),
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spectrum.real.square() + spectrum.imag.square()
        power = power.transpose(1, 2)
        mel = power @ self.mel_filterbank.to(
            device=power.device,
            dtype=power.dtype,
        )
        decibels = 10.0 * torch.log10(mel.clamp_min(1e-10))
        return torch.maximum(decibels, decibels.max() - self.config.top_db)

    def forward(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(waveforms, Tensor):
            raise TypeError("`waveforms` must be a torch.Tensor.")
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2 or waveforms.shape[-1] < self.config.hop_length:
            raise ValueError(
                "SpeechBrain VAD waveforms must have shape [batch, samples] "
                "and contain at least one 10 ms hop.")
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
            if torch.any(lengths < self.config.hop_length) or torch.any(lengths > waveforms.shape[-1]):
                raise ValueError("Every waveform length must be within the padded batch.")

        # The published SpeechBrain Fbank executes under no_grad.  Preserving
        # that boundary avoids implying trainable waveform/filter parameters.
        with torch.no_grad():
            features = self._features(waveforms)
            relative = lengths.to(dtype=features.dtype) / waveforms.shape[-1]
            frame_lengths = torch.round(relative * features.shape[1]).long()
            normalized = features.clone()
            for index, frame_length in enumerate(frame_lengths.tolist()):
                if frame_length < 2:
                    raise ValueError("Sentence normalization needs at least two valid feature frames.")
                valid = features[index, :frame_length]
                mean = valid.mean(dim=0)
                std = valid.std(dim=0, correction=1)
                std = std.clamp_min(self.config.normalization_epsilon)
                normalized[index] = (features[index] - mean) / std
        return normalized.detach(), frame_lengths


__all__ = ["SpeechBrainVADFrontend", "speechbrain_mel_filterbank"]
