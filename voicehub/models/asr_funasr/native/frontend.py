"""VoiceHub-owned waveform frontend for SenseVoiceSmall."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from voicehub.models.asr_funasr.native.configuration import SenseVoiceSmallConfig
from voicehub.processing.kaldi import KaldiFbank, KaldiFbankConfig
from voicehub.tokenization.assets import read_bounded_asset

_MAX_CMVN_BYTES = 16 * 1024 * 1024


def _finite_values(line: str, *, section: str) -> Tensor:
    left = line.find("[")
    right = line.rfind("]")
    if left < 0 or right <= left:
        raise ValueError(f"SenseVoice {section} CMVN vector is malformed.")
    values = []
    for index, raw_value in enumerate(line[left + 1:right].split()):
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"SenseVoice {section} CMVN value {index} is invalid.") from error
        if not math.isfinite(value):
            raise ValueError(f"SenseVoice {section} CMVN value {index} is not finite.")
        values.append(value)
    if not values:
        raise ValueError(f"SenseVoice {section} CMVN vector is empty.")
    return torch.tensor(values, dtype=torch.float32)


def load_sensevoice_cmvn(
    path: str | Path,
    *,
    expected_dimension: int = 560,
) -> tuple[Tensor, Tensor]:
    """Parse the published Kaldi nnet AddShift/Rescale transform."""
    payload = read_bounded_asset(path, max_bytes=_MAX_CMVN_BYTES)
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("SenseVoice CMVN must be UTF-8 text.") from error
    vectors: dict[str, Tensor] = {}
    markers = {
        "<AddShift>": "shift",
        "<Rescale>": "scale",
    }
    for index, line in enumerate(lines):
        marker = line.strip().split(maxsplit=1)[0] if line.strip() else ""
        section = markers.get(marker)
        if section is None:
            continue
        if section in vectors:
            raise ValueError(f"SenseVoice CMVN repeats the {section} section.")
        if index + 1 >= len(lines):
            raise ValueError(f"SenseVoice CMVN omits the {section} vector.")
        value_line = lines[index + 1].strip()
        if not value_line.startswith("<LearnRateCoef>"):
            raise ValueError(f"SenseVoice CMVN {section} must use LearnRateCoef syntax.")
        vectors[section] = _finite_values(value_line, section=section)
    if set(vectors) != {"shift", "scale"}:
        missing = sorted({"shift", "scale"} - set(vectors))
        raise ValueError(f"SenseVoice CMVN is missing {missing!r}.")
    shift = vectors["shift"]
    scale = vectors["scale"]
    if shift.numel() != expected_dimension or scale.numel() != expected_dimension:
        raise ValueError(
            "SenseVoice CMVN dimension mismatch: expected "
            f"{expected_dimension}, found {shift.numel()} and {scale.numel()}.")
    if torch.any(scale <= 0):
        raise ValueError("SenseVoice CMVN scale values must be positive.")
    return shift, scale


def low_frame_rate_stack(
    features: Tensor,
    *,
    window: int,
    stride: int,
) -> Tensor:
    """Apply FunASR's left-padded, last-frame-padded LFR splice."""
    if not isinstance(features, Tensor) or features.ndim != 2:
        raise ValueError("LFR input must have shape [frames, features].")
    if features.shape[0] < 1:
        raise ValueError("LFR input must contain at least one frame.")
    if (isinstance(window, bool) or not isinstance(window, int) or window < 1 or window % 2 == 0):
        raise ValueError("LFR `window` must be a positive odd integer.")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("LFR `stride` must be a positive integer.")
    left = (window - 1) // 2
    padded = torch.cat((features[:1].expand(left, -1), features), dim=0)
    output_frames = (features.shape[0] + stride - 1) // stride
    starts = torch.arange(
        output_frames,
        device=features.device,
        dtype=torch.long,
    ) * stride
    offsets = torch.arange(
        window,
        device=features.device,
        dtype=torch.long,
    )
    indices = (starts[:, None] + offsets[None, :]).clamp_max(padded.shape[0] - 1)
    return padded[indices].reshape(output_frames, window * features.shape[1])


class SenseVoiceFrontend(nn.Module):
    """Differentiable Kaldi fbank, LFR splice, and fixed CMVN."""

    def __init__(
        self,
        config: SenseVoiceSmallConfig | dict[str, Any] | None = None,
        *,
        cmvn_shift: Tensor,
        cmvn_scale: Tensor,
    ) -> None:
        super().__init__()
        self.config = SenseVoiceSmallConfig.coerce(config)
        for name, value in (
            ("cmvn_shift", cmvn_shift),
            ("cmvn_scale", cmvn_scale),
        ):
            if not isinstance(value, Tensor) or value.ndim != 1:
                raise ValueError(f"`{name}` must be a rank-one tensor.")
            if value.numel() != self.config.input_dimension:
                raise ValueError(f"`{name}` must contain {self.config.input_dimension} values.")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"`{name}` must contain finite floats.")
        if torch.any(cmvn_scale <= 0):
            raise ValueError("`cmvn_scale` values must be positive.")
        self.register_buffer(
            "cmvn_shift",
            cmvn_shift.detach().to(dtype=torch.float32).clone(),
        )
        self.register_buffer(
            "cmvn_scale",
            cmvn_scale.detach().to(dtype=torch.float32).clone(),
        )
        common = {
            "sample_frequency": float(self.config.sampling_rate),
            "frame_length": 25.0,
            "frame_shift": 10.0,
            "num_mel_bins": self.config.num_mel_bins,
            "energy_floor": 0.0,
            "window_type": "hamming",
            "snip_edges": True,
        }
        self.inference_fbank = KaldiFbank(
            KaldiFbankConfig(
                dither=self.config.inference_dither,
                **common,
            ),
            waveform_scale=self.config.waveform_scale,
        )
        self.training_fbank = KaldiFbank(
            KaldiFbankConfig(
                dither=self.config.training_dither,
                **common,
            ),
            waveform_scale=self.config.waveform_scale,
        )

    @classmethod
    def from_cmvn_file(
        cls,
        config: SenseVoiceSmallConfig | dict[str, Any] | None,
        path: str | Path,
    ) -> SenseVoiceFrontend:
        resolved = SenseVoiceSmallConfig.coerce(config)
        shift, scale = load_sensevoice_cmvn(
            path,
            expected_dimension=resolved.input_dimension,
        )
        return cls(
            resolved,
            cmvn_shift=shift,
            cmvn_scale=scale,
        )

    def forward(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor | None = None,
        *,
        training: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2:
            raise ValueError("SenseVoice waveforms must have shape [batch, samples].")
        minimum_samples = int(0.025 * self.config.sampling_rate)
        if waveform_lengths is None:
            lengths = torch.full(
                (waveforms.shape[0], ),
                waveforms.shape[1],
                dtype=torch.long,
                device=waveforms.device,
            )
        else:
            lengths = torch.as_tensor(
                waveform_lengths,
                dtype=torch.long,
                device=waveforms.device,
            )
        if torch.any(lengths < minimum_samples):
            raise ValueError("SenseVoice requires at least 25 ms of audio per example.")
        use_training = self.training if training is None else bool(training)
        fbank = self.training_fbank if use_training else self.inference_fbank
        frames, frame_lengths = fbank(waveforms, lengths)
        rows = []
        output_lengths = []
        for index, length in enumerate(frame_lengths.tolist()):
            row = low_frame_rate_stack(
                frames[index, :length],
                window=self.config.lfr_window,
                stride=self.config.lfr_stride,
            )
            row = (row + self.cmvn_shift.to(device=row.device, dtype=row.dtype)) * self.cmvn_scale.to(
                device=row.device, dtype=row.dtype)
            rows.append(row.to(dtype=torch.float32))
            output_lengths.append(row.shape[0])
        return (
            pad_sequence(rows, batch_first=True, padding_value=0.0),
            torch.tensor(
                output_lengths,
                dtype=torch.long,
                device=waveforms.device,
            ),
        )


__all__ = [
    "SenseVoiceFrontend",
    "load_sensevoice_cmvn",
    "low_frame_rate_stack",
]
