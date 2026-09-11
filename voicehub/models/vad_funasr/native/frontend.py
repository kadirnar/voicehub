"""Pure-PyTorch Kaldi fbank, LFR, and CMVN frontend for FSMN VAD."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.vad_funasr.native.configuration import FSMNVADConfig


def _next_power_of_two(value: int) -> int:
    return 1 if value == 0 else 2**(value - 1).bit_length()


def _mel_scale(values: Tensor) -> Tensor:
    return 1_127.0 * (1.0 + values / 700.0).log()


def kaldi_mel_banks(
    num_bins: int,
    padded_window_size: int,
    sample_rate: int,
) -> Tensor:
    """Build Kaldi's default triangular mel bank exactly once."""
    nyquist = 0.5 * sample_rate
    low_frequency = 20.0
    high_frequency = nyquist
    mel_low = 1_127.0 * math.log(1.0 + low_frequency / 700.0)
    mel_high = 1_127.0 * math.log(1.0 + high_frequency / 700.0)
    mel_delta = (mel_high - mel_low) / (num_bins + 1)
    bins = torch.arange(num_bins, dtype=torch.float32).unsqueeze(1)
    left = mel_low + bins * mel_delta
    center = mel_low + (bins + 1.0) * mel_delta
    right = mel_low + (bins + 2.0) * mel_delta
    fft_width = sample_rate / padded_window_size
    mel = _mel_scale(fft_width * torch.arange(padded_window_size // 2, dtype=torch.float32)).unsqueeze(0)
    rising = (mel - left) / (center - left)
    falling = (right - mel) / (right - center)
    bank = torch.maximum(
        torch.zeros(1, dtype=torch.float32),
        torch.minimum(rising, falling),
    )
    return functional.pad(bank, (0, 1))


def parse_kaldi_cmvn(
    path: str | Path,
    *,
    expected_dimension: int,
) -> tuple[Tensor, Tensor]:
    """Read the fixed ``<AddShift>``/``<Rescale>`` Kaldi text transform."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Kaldi CMVN file was not found: {source}.")
    lines = source.read_text(encoding="utf-8").splitlines()

    def values_after(marker: str) -> Tensor:
        try:
            marker_index = next(index for index, line in enumerate(lines) if line.strip().startswith(marker))
        except StopIteration as error:
            raise ValueError(f"Kaldi CMVN file does not contain {marker}.") from error
        for line in lines[marker_index + 1:]:
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("<LearnRateCoef>"):
                break
            try:
                contents = stripped[stripped.index("[") + 1:stripped.rindex("]")]
            except ValueError as error:
                raise ValueError(f"Malformed {marker} values in Kaldi CMVN file.") from error
            try:
                tensor = torch.tensor(
                    [float(item) for item in contents.split()],
                    dtype=torch.float32,
                )
            except ValueError as error:
                raise ValueError(f"Non-numeric {marker} values in Kaldi CMVN file.") from error
            if tensor.numel() != expected_dimension:
                raise ValueError(
                    f"Kaldi CMVN {marker} requires {expected_dimension} "
                    f"values, found {tensor.numel()}.")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Kaldi CMVN {marker} contains non-finite values.")
            return tensor
        raise ValueError(f"Kaldi CMVN {marker} is missing its value row.")

    return values_after("<AddShift>"), values_after("<Rescale>")


class FSMNVADFrontend(nn.Module):
    """Differentiable frontend matching FunASR ``WavFrontendOnline``."""

    def __init__(
        self,
        config: FSMNVADConfig,
        *,
        cmvn_shift: Tensor | None = None,
        cmvn_scale: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = FSMNVADConfig.coerce(config)
        dimension = self.config.input_dim
        if cmvn_shift is None:
            cmvn_shift = torch.zeros(dimension, dtype=torch.float32)
        if cmvn_scale is None:
            cmvn_scale = torch.ones(dimension, dtype=torch.float32)
        for name, value in (
            ("cmvn_shift", cmvn_shift),
            ("cmvn_scale", cmvn_scale),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"`{name}` must be a PyTorch tensor.")
            if tuple(value.shape) != (dimension, ):
                raise ValueError(f"`{name}` must have shape ({dimension},).")
            if not torch.isfinite(value).all():
                raise ValueError(f"`{name}` cannot contain non-finite values.")
        self.register_buffer(
            "cmvn_shift",
            cmvn_shift.detach().to(dtype=torch.float32).clone(),
        )
        self.register_buffer(
            "cmvn_scale",
            cmvn_scale.detach().to(dtype=torch.float32).clone(),
        )
        padded = _next_power_of_two(self.config.frame_length_samples)
        self.register_buffer(
            "_window",
            torch.hamming_window(
                self.config.frame_length_samples,
                periodic=False,
                alpha=0.54,
                beta=0.46,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_mel_banks",
            kaldi_mel_banks(
                self.config.num_mel_bins,
                padded,
                self.config.sampling_rate,
            ),
            persistent=False,
        )
        self.padded_window_size = padded

    def frame_count(self, sample_count: int) -> int:
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise TypeError("`sample_count` must be an integer.")
        if sample_count < self.config.frame_length_samples:
            return 0
        return (1 + (sample_count - self.config.frame_length_samples) // self.config.frame_shift_samples)

    def fbank(self, waveforms: Tensor) -> Tensor:
        """Return unstacked 80-bin log-mel features."""
        if not isinstance(waveforms, Tensor) or waveforms.ndim != 2:
            raise ValueError("`waveforms` must have shape [batch, samples].")
        if not waveforms.is_floating_point():
            raise TypeError("`waveforms` must use a floating-point dtype.")
        if waveforms.shape[-1] < self.config.frame_length_samples:
            return waveforms.new_empty(
                waveforms.shape[0],
                0,
                self.config.num_mel_bins,
            )
        frames = waveforms.unfold(
            -1,
            self.config.frame_length_samples,
            self.config.frame_shift_samples,
        )
        # FunASR scales normalized audio to the int16 amplitude domain before
        # calling Kaldi fbank.
        frames = frames * float(1 << 15)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        previous = torch.cat(
            (frames[..., :1], frames[..., :-1]),
            dim=-1,
        )
        frames = frames - 0.97 * previous
        frames = frames * self._window.to(
            device=frames.device,
            dtype=frames.dtype,
        )
        frames = functional.pad(
            frames,
            (0, self.padded_window_size - self.config.frame_length_samples),
        )
        spectrum = torch.fft.rfft(frames).abs().pow(2.0)
        mel_banks = self._mel_banks.to(
            device=spectrum.device,
            dtype=spectrum.dtype,
        )
        mel = spectrum @ mel_banks.transpose(0, 1)
        epsilon = torch.tensor(
            torch.finfo(torch.float32).eps,
            device=mel.device,
            dtype=mel.dtype,
        )
        return mel.maximum(epsilon).log()

    def apply_lfr(
        self,
        features: Tensor,
        *,
        final: bool = True,
    ) -> Tensor:
        """Stack LFR context, retaining right-context frames while
        streaming."""
        if not isinstance(features, Tensor) or features.ndim != 3:
            raise ValueError("`features` must have shape [batch, frames, bins].")
        if features.shape[-1] != self.config.num_mel_bins:
            raise ValueError(
                f"Expected {self.config.num_mel_bins} mel bins, "
                f"found {features.shape[-1]}.")
        frame_count = features.shape[1]
        if frame_count == 0:
            return features.new_empty(
                features.shape[0],
                0,
                self.config.input_dim,
            )
        left_count = (self.config.lfr_m - 1) // 2
        right_count = self.config.lfr_m - left_count - 1
        left = features[:, :1].expand(-1, left_count, -1)
        padded = torch.cat((left, features), dim=1)
        if final:
            right = features[:, -1:].expand(-1, right_count, -1)
            padded = torch.cat((padded, right), dim=1)
            output_frames = math.ceil(frame_count / self.config.lfr_n)
        else:
            output_frames = max(
                0,
                (padded.shape[1] - self.config.lfr_m) // self.config.lfr_n + 1,
            )
        if output_frames == 0:
            return features.new_empty(
                features.shape[0],
                0,
                self.config.input_dim,
            )
        strides = (
            padded.stride(0),
            self.config.lfr_n * padded.stride(1),
            padded.stride(1),
            padded.stride(2),
        )
        windows = padded.as_strided(
            (
                padded.shape[0],
                output_frames,
                self.config.lfr_m,
                self.config.num_mel_bins,
            ),
            strides,
        )
        return windows.reshape(
            padded.shape[0],
            output_frames,
            self.config.input_dim,
        ).clone()

    def forward(
        self,
        waveforms: Tensor,
        *,
        final: bool = True,
    ) -> Tensor:
        features = self.apply_lfr(
            self.fbank(waveforms),
            final=final,
        )
        shift = self.cmvn_shift.to(
            device=features.device,
            dtype=features.dtype,
        )
        scale = self.cmvn_scale.to(
            device=features.device,
            dtype=features.dtype,
        )
        return (features + shift) * scale


__all__ = [
    "FSMNVADFrontend",
    "kaldi_mel_banks",
    "parse_kaldi_cmvn",
]
