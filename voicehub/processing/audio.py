"""PyTorch-native audio feature operations used by speech architectures."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, log10
from typing import Any

from voicehub.processing.graph import PROCESSING_OPERATIONS, ProcessingOperation


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - package invariant
        raise RuntimeError(
            "Native audio processing requires PyTorch, VoiceHub's compute "
            "runtime.") from error
    return torch


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _hz_to_mel(frequencies):
    """Slaney mel scale used by Whisper and librosa."""
    torch = _torch()
    frequencies = torch.as_tensor(frequencies)
    linear_spacing = 200.0 / 3.0
    minimum_log_hz = 1000.0
    minimum_log_mel = minimum_log_hz / linear_spacing
    log_step = log(6.4) / 27.0
    linear = frequencies / linear_spacing
    logarithmic = minimum_log_mel + torch.log(
        torch.clamp(frequencies, min=minimum_log_hz) / minimum_log_hz) / log_step
    return torch.where(frequencies >= minimum_log_hz, logarithmic, linear)


def _mel_to_hz(mels):
    torch = _torch()
    mels = torch.as_tensor(mels)
    linear_spacing = 200.0 / 3.0
    minimum_log_hz = 1000.0
    minimum_log_mel = minimum_log_hz / linear_spacing
    log_step = log(6.4) / 27.0
    linear = mels * linear_spacing
    logarithmic = minimum_log_hz * torch.exp(log_step * (mels - minimum_log_mel))
    return torch.where(mels >= minimum_log_mel, logarithmic, linear)


def mel_filter_bank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    minimum_frequency: float = 0.0,
    maximum_frequency: float | None = None,
    dtype: Any = None,
    device: Any = None,
    match_librosa: bool = False,
):
    """Build a Slaney-normalized triangular mel bank using PyTorch only.

    With ``match_librosa=True`` the bank is evaluated in float64 on the
    CPU and rounded exactly like
    ``librosa.filters.mel(dtype=np.float32)`` (the triangles are stored
    as float32 before the float64 Slaney normalization is applied and
    rounded again). This reproduces, bit for bit, the
    ``mel_filters.npz`` asset shipped with OpenAI Whisper. The result is
    then cast to ``dtype`` and moved to ``device``.
    """
    torch = _torch()
    sample_rate = _positive_integer(sample_rate, name="sample_rate")
    n_fft = _positive_integer(n_fft, name="n_fft")
    n_mels = _positive_integer(n_mels, name="n_mels")
    maximum_frequency = (sample_rate / 2.0 if maximum_frequency is None else float(maximum_frequency))
    minimum_frequency = float(minimum_frequency)
    if not 0.0 <= minimum_frequency < maximum_frequency <= sample_rate / 2.0:
        raise ValueError("Mel frequency bounds must satisfy 0 <= min < max <= Nyquist.")
    if not isinstance(match_librosa, bool):
        raise TypeError("`match_librosa` must be a boolean.")
    output_dtype = dtype or torch.float32
    output_device = device
    if match_librosa:
        dtype, device = torch.float64, "cpu"
    else:
        dtype = output_dtype
    frequency_bins = torch.linspace(
        0.0,
        sample_rate / 2.0,
        n_fft // 2 + 1,
        dtype=dtype,
        device=device,
    )
    mel_edges = torch.linspace(
        _hz_to_mel(torch.tensor(minimum_frequency, dtype=dtype, device=device)),
        _hz_to_mel(torch.tensor(maximum_frequency, dtype=dtype, device=device)),
        n_mels + 2,
        dtype=dtype,
        device=device,
    )
    frequency_edges = _mel_to_hz(mel_edges)
    ramps = frequency_edges.unsqueeze(1) - frequency_bins.unsqueeze(0)
    edge_widths = frequency_edges[1:] - frequency_edges[:-1]
    lower = -ramps[:-2] / edge_widths[:-1].unsqueeze(1)
    upper = ramps[2:] / edge_widths[1:].unsqueeze(1)
    filters = torch.clamp(torch.minimum(lower, upper), min=0.0)
    if match_librosa:
        filters = filters.float().double()
    filters *= (2.0 / (frequency_edges[2:] - frequency_edges[:-2])).unsqueeze(1)
    if match_librosa:
        filters = filters.float().to(dtype=output_dtype, device=output_device)
    return filters


def htk_mel_filter_bank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    minimum_frequency: float = 0.0,
    maximum_frequency: float | None = None,
    dtype: Any = None,
    device: Any = None,
):
    """Build an unnormalised HTK mel bank using PyTorch only.

    The returned layout is ``[frequency_bin, mel_bin]``.  This matches
    ``torchaudio.functional.melscale_fbanks(..., mel_scale="htk",
    norm=None)`` and, importantly, the persistent ``mel_scale.fb``
    tensor in released Vocos checkpoints.
    """
    torch = _torch()
    sample_rate = _positive_integer(sample_rate, name="sample_rate")
    n_fft = _positive_integer(n_fft, name="n_fft")
    n_mels = _positive_integer(n_mels, name="n_mels")
    maximum_frequency = (sample_rate / 2.0 if maximum_frequency is None else float(maximum_frequency))
    minimum_frequency = float(minimum_frequency)
    if not 0.0 <= minimum_frequency < maximum_frequency <= sample_rate / 2.0:
        raise ValueError("Mel frequency bounds must satisfy 0 <= min < max <= Nyquist.")

    dtype = dtype or torch.float32
    minimum_mel = 2_595.0 * log10(1.0 + minimum_frequency / 700.0)
    maximum_mel = 2_595.0 * log10(1.0 + maximum_frequency / 700.0)
    mel_edges = torch.linspace(
        minimum_mel,
        maximum_mel,
        n_mels + 2,
        dtype=dtype,
        device=device,
    )
    frequency_edges = 700.0 * (
        torch.pow(
            torch.tensor(10.0, dtype=dtype, device=device),
            mel_edges / 2_595.0,
        ) - 1.0)
    frequency_bins = torch.linspace(
        0.0,
        sample_rate / 2.0,
        n_fft // 2 + 1,
        dtype=dtype,
        device=device,
    )
    slopes = frequency_edges.unsqueeze(0) - frequency_bins.unsqueeze(1)
    edge_widths = frequency_edges[1:] - frequency_edges[:-1]
    descending = -slopes[:, :-2] / edge_widths[:-1]
    ascending = slopes[:, 2:] / edge_widths[1:]
    return torch.maximum(
        torch.zeros(1, dtype=dtype, device=device),
        torch.minimum(descending, ascending),
    )


@dataclass(frozen=True)
class PadOrTrimAudio(ProcessingOperation):
    """Right-pad or truncate waveforms to an exact number of samples."""

    operation_id = "audio.pad-or-trim"
    operation_version = "1"

    length: int
    input_key: str = "waveform"
    output_key: str = "padded_waveform"
    length_key: str = "waveform_length"

    def __post_init__(self) -> None:
        _positive_integer(self.length, name="length")
        for name in ("input_key", "output_key", "length_key"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"`{name}` must be a non-empty string.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_key, )

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_key, self.length_key

    def process(self, values):
        torch = _torch()
        waveform = torch.as_tensor(values[self.input_key])
        if waveform.ndim not in (1, 2):
            raise ValueError("Audio waveform must have shape [time] or [batch, time].")
        original_length = min(waveform.shape[-1], self.length)
        if waveform.shape[-1] < self.length:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, self.length - waveform.shape[-1]),
            )
        else:
            waveform = waveform[..., :self.length]
        lengths = torch.full(
            (waveform.shape[0], ) if waveform.ndim == 2 else (),
            original_length,
            dtype=torch.long,
            device=waveform.device,
        )
        return {
            self.output_key: waveform,
            self.length_key: lengths,
        }

    def to_config(self):
        return {
            "length": self.length,
            "input_key": self.input_key,
            "output_key": self.output_key,
            "length_key": self.length_key,
        }


@dataclass(frozen=True)
class LogMelSpectrogram(ProcessingOperation):
    """Compute a stable Slaney log-mel spectrogram in float32."""

    operation_id = "audio.log-mel-spectrogram"
    operation_version = "1"

    sample_rate: int = 16_000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    dynamic_range: float = 8.0
    whisper_scaling: bool = False
    input_key: str = "waveform"
    output_key: str = "input_features"
    librosa_filters: bool = False

    def __post_init__(self) -> None:
        for name in ("sample_rate", "n_fft", "hop_length", "n_mels"):
            _positive_integer(getattr(self, name), name=name)
        if self.hop_length > self.n_fft:
            raise ValueError("`hop_length` cannot exceed `n_fft`.")
        if (isinstance(self.dynamic_range, bool) or not isinstance(self.dynamic_range, (int, float)) or
                self.dynamic_range <= 0):
            raise ValueError("`dynamic_range` must be positive.")
        if not isinstance(self.whisper_scaling, bool):
            raise TypeError("`whisper_scaling` must be a boolean.")
        if not isinstance(self.librosa_filters, bool):
            raise TypeError("`librosa_filters` must be a boolean.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_key, )

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_key, )

    def process(self, values):
        torch = _torch()
        waveform = torch.as_tensor(values[self.input_key])
        if waveform.ndim not in (1, 2):
            raise ValueError("Audio waveform must have shape [time] or [batch, time].")
        if not waveform.is_floating_point():
            waveform = waveform.float()
        waveform = waveform.float()
        window = torch.hann_window(
            self.n_fft,
            dtype=waveform.dtype,
            device=waveform.device,
        )
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        power = stft[..., :-1].abs().square()
        filters = mel_filter_bank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            dtype=power.dtype,
            device=power.device,
            match_librosa=self.librosa_filters,
        )
        mel = torch.matmul(filters, power)
        log_mel = torch.clamp(mel, min=1e-10).log10()
        maximum = log_mel.amax(dim=(-2, -1), keepdim=True)
        log_mel = torch.maximum(log_mel, maximum - self.dynamic_range)
        if self.whisper_scaling:
            log_mel = (log_mel + 4.0) / 4.0
        return {self.output_key: log_mel}

    def to_config(self):
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "dynamic_range": self.dynamic_range,
            "whisper_scaling": self.whisper_scaling,
            "input_key": self.input_key,
            "output_key": self.output_key,
            "librosa_filters": self.librosa_filters,
        }


PROCESSING_OPERATIONS.register(PadOrTrimAudio, exist_ok=True)
PROCESSING_OPERATIONS.register(LogMelSpectrogram, exist_ok=True)

__all__ = [
    "LogMelSpectrogram",
    "PadOrTrimAudio",
    "htk_mel_filter_bank",
    "mel_filter_bank",
]
