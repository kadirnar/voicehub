"""PyTorch-only frontend and augmentation for the pinned ESPnet recipe."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.asr_espnet.native.configuration import ESPnetLibriSpeechTransformerConfig
from voicehub.processing.audio import mel_filter_bank


def make_pad_mask(lengths: Tensor, maximum_length: int | None = None) -> Tensor:
    """Return ``True`` at padded positions."""
    import torch

    values = torch.as_tensor(lengths, dtype=torch.long)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("Lengths must be a non-empty one-dimensional tensor.")
    if torch.any(values < 0):
        raise ValueError("Lengths cannot be negative.")
    if maximum_length is None:
        maximum_length = int(values.max().item())
    if maximum_length < int(values.max().item()):
        raise ValueError("Maximum length is smaller than a declared sequence.")
    positions = torch.arange(maximum_length, device=values.device)
    return positions.unsqueeze(0) >= values.unsqueeze(1)


class ESPnetLogMel(nn.Module):
    """Power-spectrum to natural-log Slaney mel features."""

    def __init__(self, config: ESPnetLibriSpeechTransformerConfig) -> None:
        super().__init__()
        filters = mel_filter_bank(
            sample_rate=config.sampling_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            minimum_frequency=config.f_min,
            maximum_frequency=config.f_max,
        )
        # The source checkpoint stores librosa's transposed filter matrix as
        # ``frontend.logmel.melmat``.
        self.register_buffer("melmat", filters.transpose(0, 1).contiguous())

    def forward(
        self,
        power_spectrum: Tensor,
        frame_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        values = power_spectrum @ self.melmat.to(
            device=power_spectrum.device,
            dtype=power_spectrum.dtype,
        )
        values = values.clamp_min(1.0e-10).log()
        mask = make_pad_mask(
            frame_lengths.to(values.device),
            values.shape[1],
        )
        return values.masked_fill(mask.unsqueeze(-1), 0.0), frame_lengths


class ESPnetDefaultFrontend(nn.Module):
    """Hann STFT, power spectrum, and checkpoint-stored log-mel bank."""

    def __init__(self, config: ESPnetLibriSpeechTransformerConfig) -> None:
        import torch

        super().__init__()
        self.config = ESPnetLibriSpeechTransformerConfig.coerce(config)
        self.register_buffer(
            "_window",
            torch.hann_window(self.config.win_length),
            persistent=False,
        )
        self.logmel = ESPnetLogMel(self.config)

    def forward(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        import torch

        if not isinstance(waveforms, Tensor):
            raise TypeError("`waveforms` must be a torch.Tensor.")
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2:
            raise ValueError("ESPnet waveforms must have shape [batch, samples].")
        if waveforms.shape[-1] <= self.config.n_fft // 2:
            raise ValueError("ESPnet waveforms must be longer than the centered STFT padding.")
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
            if torch.any(lengths <= 0) or torch.any(lengths > waveforms.shape[-1]):
                raise ValueError("Waveform lengths must be within the padded batch.")
        values = waveforms.float()
        spectrum = torch.stft(
            values,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self._window.to(device=values.device, dtype=values.dtype),
            center=self.config.center,
            pad_mode=self.config.pad_mode,
            normalized=self.config.normalized_stft,
            onesided=self.config.onesided_stft,
            return_complex=True,
        ).transpose(1, 2)
        power = spectrum.real.square() + spectrum.imag.square()
        padded_lengths = (lengths + self.config.win_length if self.config.center else lengths)
        frame_lengths = ((padded_lengths - self.config.win_length) // self.config.hop_length + 1)
        return self.logmel(power, frame_lengths)


class ESPnetGlobalMVN(nn.Module):
    """Global mean/variance normalization stored inside the checkpoint."""

    def __init__(self, config: ESPnetLibriSpeechTransformerConfig) -> None:
        import torch

        super().__init__()
        self.config = ESPnetLibriSpeechTransformerConfig.coerce(config)
        self.register_buffer("mean", torch.zeros(self.config.n_mels))
        self.register_buffer("std", torch.ones(self.config.n_mels))

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.config.n_mels:
            raise ValueError("ESPnet features must have shape "
                             f"[batch, frames, {self.config.n_mels}].")
        mask = make_pad_mask(
            feature_lengths.to(features.device),
            features.shape[1],
        ).unsqueeze(-1)
        values = features - self.mean.to(
            device=features.device,
            dtype=features.dtype,
        )
        values = values.masked_fill(mask, 0.0)
        values = values / self.std.to(
            device=features.device,
            dtype=features.dtype,
        ).clamp_min(self.config.global_mvn_epsilon)
        return values, feature_lengths


def _time_warp(values: Tensor, *, window: int) -> Tensor:
    import torch

    original_shape = values.shape
    image = values.unsqueeze(1) if values.ndim == 3 else values
    time = image.shape[-2]
    if time - window <= window:
        return values
    center = int(torch.randint(
        window,
        time - window,
        (1, ),
        device=values.device,
    ).item())
    warped = int(torch.randint(
        center - window,
        center + window,
        (1, ),
        device=values.device,
    ).item()) + 1
    left = functional.interpolate(
        image[..., :center, :],
        size=(warped, image.shape[-1]),
        mode="bicubic",
        align_corners=False,
    )
    right = functional.interpolate(
        image[..., center:, :],
        size=(time - warped, image.shape[-1]),
        mode="bicubic",
        align_corners=False,
    )
    return torch.cat((left, right), dim=-2).reshape(original_shape)


def _mask_along_axis(
    values: Tensor,
    *,
    width_range: tuple[int, int],
    count: int,
    axis: int,
) -> Tensor:
    import torch

    batch = values.shape[0]
    size = values.shape[axis]
    widths = torch.randint(
        width_range[0],
        width_range[1],
        (batch, count),
        device=values.device,
    )
    maximum_start = max(1, size - int(widths.max().item()))
    starts = torch.randint(
        0,
        maximum_start,
        (batch, count),
        device=values.device,
    )
    positions = torch.arange(size, device=values.device).view(1, 1, size)
    mask = ((starts.unsqueeze(-1) <= positions) & (positions < starts.add(widths).unsqueeze(-1))).any(dim=1)
    mask = mask.unsqueeze(-1) if axis == 1 else mask.unsqueeze(1)
    return values.masked_fill(mask, 0.0)


class ESPnetSpecAugment(nn.Module):
    """Published time-warp and two-by-two frequency/time masking."""

    def __init__(self, config: ESPnetLibriSpeechTransformerConfig) -> None:
        super().__init__()
        self.config = ESPnetLibriSpeechTransformerConfig.coerce(config)

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        import torch

        if feature_lengths.ndim != 1 or feature_lengths.shape[0] != features.shape[0]:
            raise ValueError("Feature lengths must have shape [batch].")
        if torch.all(feature_lengths == feature_lengths[0]):
            values = _time_warp(
                features,
                window=self.config.time_warp_window,
            )
        else:
            rows = [
                _time_warp(
                    features[index:index + 1, :int(length.item())],
                    window=self.config.time_warp_window,
                ).squeeze(0) for index, length in enumerate(feature_lengths)
            ]
            values = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        values = _mask_along_axis(
            values,
            width_range=self.config.frequency_mask_width,
            count=self.config.frequency_masks,
            axis=2,
        )
        values = _mask_along_axis(
            values,
            width_range=self.config.time_mask_width,
            count=self.config.time_masks,
            axis=1,
        )
        return values, feature_lengths


__all__ = [
    "ESPnetDefaultFrontend",
    "ESPnetGlobalMVN",
    "ESPnetLogMel",
    "ESPnetSpecAugment",
    "make_pad_mask",
]
