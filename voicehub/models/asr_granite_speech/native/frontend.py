"""PyTorch-native Granite Speech waveform frontend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import torch
from torch import Tensor

from voicehub.processing.audio import htk_mel_filter_bank
from voicehub.processing.waveform import NativeAudio, load_native_audio

SAMPLE_RATE = 16_000


class GraniteSpeechFeatureExtractor:
    """Reproduce the released torchaudio log-mel preprocessing in PyTorch."""

    def __init__(
        self,
        *,
        sampling_rate: int = SAMPLE_RATE,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        projector_window_size: int = 15,
        projector_downsample_rate: int = 5,
    ) -> None:
        for name, value in (
            ("sampling_rate", sampling_rate),
            ("n_fft", n_fft),
            ("win_length", win_length),
            ("hop_length", hop_length),
            ("n_mels", n_mels),
            ("projector_window_size", projector_window_size),
            ("projector_downsample_rate", projector_downsample_rate),
        ):
            if (isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0):
                raise ValueError(f"`{name}` must be a positive integer.")
        if win_length > n_fft:
            raise ValueError("`win_length` cannot exceed `n_fft`.")
        if projector_window_size % projector_downsample_rate:
            raise ValueError("`projector_window_size` must be divisible by "
                             "`projector_downsample_rate`.")
        self.sampling_rate = int(sampling_rate)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.projector_window_size = int(projector_window_size)
        self.projector_downsample_rate = int(projector_downsample_rate, )

    @property
    def input_dim(self) -> int:
        return self.n_mels * 2

    def num_audio_features(self, raw_length: int) -> int:
        if (isinstance(raw_length, bool) or not isinstance(raw_length, Integral) or int(raw_length) <= 0):
            raise ValueError("Audio length must be a positive integer.")
        mel_length = int(raw_length) // self.hop_length + 1
        encoder_length = mel_length // 2
        block_count = math.ceil(encoder_length / self.projector_window_size, )
        return block_count * (self.projector_window_size // self.projector_downsample_rate)

    @staticmethod
    def _broadcast_rates(
        sampling_rates: int | None | Sequence[int | None],
        batch_size: int,
    ) -> tuple[int | None, ...]:
        if (isinstance(sampling_rates, Sequence) and not isinstance(sampling_rates, (str, bytes, bytearray))):
            rates = tuple(sampling_rates)
            if len(rates) != batch_size:
                raise ValueError("`sampling_rates` must contain one value per waveform.")
            return rates
        return (sampling_rates, ) * batch_size

    @staticmethod
    def _audio_rows(audios: Any) -> tuple[Any, ...]:
        if isinstance(audios, Tensor):
            if audios.ndim == 1:
                return (audios, )
            if audios.ndim == 2:
                return tuple(audios[index] for index in range(audios.shape[0]))
            raise ValueError("Granite Speech audio tensors must have shape [time] or "
                             "[batch, time].")
        if isinstance(audios, NativeAudio):
            return (audios, )
        if isinstance(audios, (str, bytes, bytearray)):
            return (audios, )
        if isinstance(audios, Sequence):
            values = tuple(audios)
            if not values:
                raise ValueError("Granite Speech audio cannot be empty.")
            if all(isinstance(value, (int, float)) for value in values):
                return (values, )
            return values
        return (audios, )

    def materialize(
        self,
        audios: Any,
        *,
        sampling_rates: int | None | Sequence[int | None],
    ) -> tuple[NativeAudio, ...]:
        rows = self._audio_rows(audios)
        rates = self._broadcast_rates(sampling_rates, len(rows))
        materialized = tuple(
            load_native_audio(
                audio,
                sampling_rate=rate,
                target_sampling_rate=self.sampling_rate,
            ) for audio, rate in zip(rows, rates))
        if any(item.waveform.numel() == 0 for item in materialized):
            raise ValueError("Granite Speech audio cannot be empty.")
        return materialized

    def extract(
        self,
        audios: Any,
        *,
        sampling_rates: int | None | Sequence[int | None],
        device: torch.device | str | None = None,
    ) -> dict[str, Tensor]:
        """Return stacked log-mel features and projected-feature masks."""
        materialized = self.materialize(
            audios,
            sampling_rates=sampling_rates,
        )
        lengths = tuple(int(item.waveform.numel()) for item in materialized)
        minimum_length = self.n_fft // 2 + 1
        maximum = max(max(lengths), minimum_length)
        resolved_device = (materialized[0].waveform.device if device is None else torch.device(device))
        waveforms = torch.zeros(
            len(materialized),
            maximum,
            dtype=torch.float32,
            device=resolved_device,
        )
        for index, item in enumerate(materialized):
            waveforms[index, :item.waveform.numel()] = item.waveform.to(
                device=resolved_device,
                dtype=torch.float32,
            )

        window = torch.hann_window(
            self.win_length,
            periodic=True,
            dtype=waveforms.dtype,
            device=waveforms.device,
        )
        spectrum = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spectrum.abs().square()
        filters = htk_mel_filter_bank(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            dtype=power.dtype,
            device=power.device,
        )
        mel = torch.einsum("fm,bft->bmt", filters, power)
        log_mel = mel.transpose(-1, -2).clamp_min_(1e-10).log10_()
        maximum_log_mel = log_mel.amax(
            dim=(-2, -1),
            keepdim=True,
        )
        log_mel = torch.maximum(
            log_mel,
            maximum_log_mel - 8.0,
        ).div_(4.0).add_(1.0)
        if log_mel.shape[1] % 2:
            log_mel = log_mel[:, :-1]
        input_features = log_mel.reshape(
            log_mel.shape[0],
            -1,
            self.input_dim,
        )
        audio_embed_sizes = torch.tensor(
            [self.num_audio_features(length) for length in lengths],
            dtype=torch.long,
            device=resolved_device,
        )
        mask = (
            torch.arange(
                int(audio_embed_sizes.max().item()),
                device=resolved_device,
            ).unsqueeze(0) < audio_embed_sizes.unsqueeze(1))
        return {
            "input_features": input_features,
            "input_features_mask": mask,
            "audio_embed_sizes": audio_embed_sizes,
            "audio_lengths": torch.tensor(
                lengths,
                dtype=torch.long,
                device=resolved_device,
            ),
        }


__all__ = [
    "GraniteSpeechFeatureExtractor",
    "SAMPLE_RATE",
]
