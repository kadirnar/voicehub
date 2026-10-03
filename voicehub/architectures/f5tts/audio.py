"""Native audio preprocessing used by F5-TTS and its Vocos decoder."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from voicehub.processing.audio import htk_mel_filter_bank as _htk_mel_filter_bank


def htk_mel_filter_bank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return torchaudio-compatible, unnormalised HTK mel filters.

    The returned orientation is ``[frequency, mel]``, matching the
    buffer in ``charactr/vocos-mel-24khz``.
    """
    return _htk_mel_filter_bank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        minimum_frequency=f_min,
        maximum_frequency=f_max,
        device=device,
    )


class F5MelSpectrogram(nn.Module):
    """Magnitude log-mel frontend matching F5-TTS' Vocos configuration."""

    def __init__(
        self,
        *,
        sample_rate: int = 24_000,
        n_fft: int = 1_024,
        hop_length: int = 256,
        win_length: int = 1_024,
        n_mels: int = 100,
        clamp_min: float = 1e-5,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.clamp_min = clamp_min
        # These buffers are not stored in checkpoints. Build them on the CPU
        # (as torchaudio does) even inside a CUDA device context, so the
        # conditioning mel matches the released frontend bit for bit.
        self.register_buffer(
            "window",
            torch.hann_window(win_length, device="cpu"),
            persistent=False,
        )
        self.register_buffer(
            "filter_bank",
            htk_mel_filter_bank(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                device="cpu",
            ),
            persistent=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0]
        if waveform.ndim != 2:
            raise ValueError("F5-TTS mel extraction expects `[batch, samples]` audio.")
        # cuFFT does not implement FP16/BF16 transforms, and running the
        # logarithmic acoustic frontend in FP32 is also materially more
        # stable near ``clamp_min``. Preserve the model-facing dtype at the
        # boundary so mixed-precision DiT inference remains end-to-end
        # compatible.
        output_dtype = waveform.dtype
        fft_dtype = (waveform.dtype if waveform.dtype in {torch.float32, torch.float64} else torch.float32)
        fft_waveform = waveform.to(dtype=fft_dtype)
        window = self.window.to(device=waveform.device, dtype=fft_dtype)
        spectrum = torch.stft(
            fft_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()
        filter_bank = self.filter_bank.to(
            device=waveform.device,
            dtype=spectrum.dtype,
        )
        mel = torch.matmul(spectrum.transpose(-1, -2), filter_bank)
        return (mel.transpose(-1, -2).clamp_min(self.clamp_min).log().to(dtype=output_dtype))


# Released reference preprocessing (``preprocess_ref_audio_text``) is written
# with pydub: millisecond slicing, integer-sample RMS, and a 50 ms silence
# generated at 11025 Hz and rate-converted with ``audioop.ratecv``.
_PYDUB_SILENCE_RATE = 11_025
_PYDUB_SILENCE_FRAMES = int(_PYDUB_SILENCE_RATE * (50 / 1000.0))


class _PydubAudio:
    """Mono waveform with pydub's millisecond slicing and RMS semantics.

    pydub measures loudness on integer samples of the decoded file's sample
    width (``audioop.rms``, truncated to an integer). ``sample_width`` is that
    width in bytes: 2 for 16-bit PCM, 4 for 24/32-bit PCM and for float WAVE
    files, which pydub converts to 32-bit PCM through FFmpeg.
    """

    def __init__(self, waveform: torch.Tensor, sample_rate: int, sample_width: int = 4) -> None:
        if waveform.ndim != 2:
            raise ValueError("pydub emulation expects channel-first `[channels, samples]` audio.")
        if sample_width not in (1, 2, 4):
            raise ValueError("pydub sample width must be 1, 2, or 4 bytes.")
        self.waveform = waveform
        self.sample_rate = int(sample_rate)
        self.sample_width = sample_width
        self.full_scale = float(2**(8 * sample_width - 1))
        values = waveform.detach().to(device="cpu", dtype=torch.float64)
        integers = (values * self.full_scale).round().clamp(-self.full_scale, self.full_scale - 1)
        if sample_width < 4:
            # Exact integer sums, like audioop's double accumulator at this width.
            squares = integers.to(dtype=torch.int64).square()
        else:
            squares = (integers / self.full_scale).square()
        # audioop.rms averages over interleaved samples of every channel.
        self._cumulative = F.pad(torch.cumsum(squares.sum(dim=0), dim=0), (1, 0))

    def __len__(self) -> int:
        return round(1000 * (self.waveform.shape[-1] / self.sample_rate))

    def _frame(self, milliseconds: int) -> int:
        return int(milliseconds * (self.sample_rate / 1000.0))

    def _bounds(self, start: int, end: int) -> tuple[int, int]:
        length = len(self)
        return self._frame(min(start, length)), self._frame(min(end, length))

    def slice(self, start: int, end: int | None = None) -> torch.Tensor:
        """Return ``segment[start:end]``, zero-filling pydub's rounding gap."""
        first, last = self._bounds(start, len(self) if end is None else end)
        data = self.waveform[:, first:last]
        missing = last - first - data.shape[-1]
        if missing > 0:
            data = F.pad(data, (0, missing))
        return data

    def rms(self, start: int, end: int) -> float:
        """Return ``segment[start:end].rms`` in pydub's sample units."""
        first, last = self._bounds(start, end)
        if last <= first:
            return 0.0
        available = self.waveform.shape[-1]
        total = float(
            (self._cumulative[min(last, available)] - self._cumulative[min(first, available)]).item())
        if self.sample_width == 4:
            total *= self.full_scale * self.full_scale
        return float(math.floor(math.sqrt(total / ((last - first) * self.waveform.shape[0]))))

    def dbfs(self, start: int, end: int) -> float:
        rms = self.rms(start, end)
        if not rms:
            return -math.inf
        return 20 * math.log(rms / self.full_scale, 10)

    def leading_silence(self, threshold: float, chunk_size: int = 10) -> int:
        """Port of ``pydub.silence.detect_leading_silence``."""
        trimmed = 0
        while self.dbfs(trimmed, trimmed + chunk_size) < threshold and trimmed < len(self):
            trimmed += chunk_size
        return min(trimmed, len(self))

    def silent_ranges(
        self,
        minimum_silence: int,
        threshold: float,
        seek_step: int,
    ) -> list[list[int]]:
        """Port of ``pydub.silence.detect_silence``."""
        length = len(self)
        if length < minimum_silence:
            return []
        limit = 10**(threshold / 20) * self.full_scale
        last_start = length - minimum_silence
        starts = list(range(0, last_start + 1, seek_step))
        if last_start % seek_step:
            starts.append(last_start)
        silent = [start for start in starts if self.rms(start, start + minimum_silence) <= limit]
        if not silent:
            return []
        ranges = []
        previous = silent.pop(0)
        range_start = previous
        for start in silent:
            continuous = start == previous + seek_step
            has_gap = start > previous + minimum_silence
            if not continuous and has_gap:
                ranges.append([range_start, previous + minimum_silence])
                range_start = start
            previous = start
        ranges.append([range_start, previous + minimum_silence])
        return ranges

    def split_on_silence(
        self,
        *,
        minimum_silence: int,
        threshold: float,
        keep_silence: int,
        seek_step: int,
    ) -> list[torch.Tensor]:
        """Port of ``pydub.silence.split_on_silence``."""
        length = len(self)
        silent = self.silent_ranges(minimum_silence, threshold, seek_step)
        if not silent:
            voiced = [[0, length]]
        elif silent[0][0] == 0 and silent[0][1] == length:
            voiced = []
        else:
            voiced = []
            previous_end = 0
            for start, end in silent:
                voiced.append([previous_end, start])
                previous_end = end
            if silent[-1][1] != length:
                voiced.append([previous_end, length])
            if voiced[0] == [0, 0]:
                voiced.pop(0)
        ranges = [[start - keep_silence, end + keep_silence] for start, end in voiced]
        for current, following in zip(ranges, ranges[1:]):
            if following[0] < current[1]:
                current[1] = (current[1] + following[0]) // 2
                following[0] = current[1]
        return [self.slice(max(start, 0), min(end, length)) for start, end in ranges]


def _milliseconds(samples: int, sample_rate: int) -> int:
    return round(1000 * (samples / sample_rate))


def _ratecv_frames(frames: int, source_rate: int, target_rate: int) -> int:
    """Frame count produced by ``audioop.ratecv`` from a fresh state."""
    divisor = math.gcd(source_rate, target_rate)
    source_rate //= divisor
    target_rate //= divisor
    phase = -target_rate
    produced = 0
    while True:
        while phase < 0:
            if frames == 0:
                return produced
            frames -= 1
            phase += target_rate
        while phase >= 0:
            produced += 1
            phase -= source_rate


def _join_until_twelve_seconds(
        segments: list[torch.Tensor], sample_rate: int, empty: torch.Tensor) -> torch.Tensor:
    joined = empty
    for segment in segments:
        if (_milliseconds(joined.shape[-1], sample_rate) > 6_000 and
                _milliseconds(joined.shape[-1] + segment.shape[-1], sample_rate) > 12_000):
            break
        joined = torch.cat((joined, segment), dim=-1)
    return joined


def pydub_sample_width(path: str | Path) -> int:
    """Return the integer sample width pydub measures a WAVE file in.

    16-bit (and 8-bit) PCM is read as is; 24/32-bit PCM is widened to 32 bits,
    and float WAVE files are converted to 32-bit PCM through FFmpeg.
    """
    with open(path, "rb") as stream:
        header = stream.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return 4
        while True:
            chunk = stream.read(8)
            if len(chunk) < 8:
                return 4
            size = int.from_bytes(chunk[4:8], "little")
            if chunk[:4] != b"fmt ":
                stream.seek(size + (size & 1), 1)
                continue
            fmt = stream.read(size)
            if len(fmt) < 16:
                return 4
            tag = int.from_bytes(fmt[0:2], "little")
            if tag == 0xFFFE and len(fmt) >= 26:
                tag = int.from_bytes(fmt[24:26], "little")
            bits = int.from_bytes(fmt[14:16], "little")
            if tag == 1 and bits in (8, 16):
                return bits // 8
            return 4


def preprocess_reference_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    sample_width: int = 4,
) -> torch.Tensor:
    """Clip and trim a mono reference like F5-TTS' released recipe.

    This ports ``preprocess_ref_audio_text`` at its native sampling rate:
    keep whole non-silent phrases up to about 12 seconds, hard-clip at 12
    seconds otherwise, trim leading/trailing audio quieter than -42 dBFS
    in 10 ms steps, and append 50 ms of silence. ``sample_width`` is the
    integer width pydub measures loudness in (see :func:`pydub_sample_width`).
    The released recipe runs pydub at 11025 Hz or above; lower rates are
    first upsampled by pydub and are therefore not reproduced bit for bit.
    """
    sample_rate = int(sample_rate)
    if waveform.ndim not in (1, 2):
        raise ValueError("Reference audio must be `[samples]` or `[channels, samples]`.")
    mono = waveform.ndim == 1
    if mono:
        waveform = waveform.unsqueeze(0)
    audio = _PydubAudio(waveform, sample_rate, sample_width)
    empty = waveform[:, :0]
    clipped = _join_until_twelve_seconds(
        audio.split_on_silence(minimum_silence=1_000, threshold=-50, keep_silence=1_000, seek_step=10),
        sample_rate,
        empty,
    )
    if _milliseconds(clipped.shape[-1], sample_rate) > 12_000:
        clipped = _join_until_twelve_seconds(
            audio.split_on_silence(minimum_silence=100, threshold=-40, keep_silence=1_000, seek_step=10),
            sample_rate,
            empty,
        )
    clipped_audio = _PydubAudio(clipped, sample_rate, sample_width)
    if len(clipped_audio) > 12_000:
        clipped = clipped_audio.slice(0, 12_000)
        clipped_audio = _PydubAudio(clipped, sample_rate, sample_width)
    leading = clipped_audio.leading_silence(-42)
    trimmed = clipped_audio.slice(leading)
    trimmed_audio = _PydubAudio(trimmed, sample_rate, sample_width)
    trailing = _PydubAudio(trimmed.flip(-1), sample_rate, sample_width).leading_silence(-42)
    if trailing > 0:
        trimmed = trimmed_audio.slice(0, len(trimmed_audio) - trailing)
    silence_frames = (
        _PYDUB_SILENCE_FRAMES if sample_rate == _PYDUB_SILENCE_RATE else _ratecv_frames(
            _PYDUB_SILENCE_FRAMES,
            _PYDUB_SILENCE_RATE,
            sample_rate,
        ))
    prepared = F.pad(trimmed, (0, silence_frames))
    return prepared.squeeze(0) if mono else prepared


def normalize_reference_rms(
    waveform: torch.Tensor,
    *,
    target_rms: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """Raise a quiet reference to the source recipe's target RMS.

    The arithmetic mirrors the released ``infer_batch_process`` exactly
    (``audio * target_rms / rms`` on a ``[1, samples]`` tensor).
    """
    rms_tensor = torch.sqrt(torch.mean(torch.square(waveform.float().unsqueeze(0))))
    rms = float(rms_tensor.item())
    if not math.isfinite(rms):
        raise ValueError("Reference audio contains non-finite samples.")
    if rms <= 0:
        raise ValueError("Reference audio is silent.")
    if rms < target_rms:
        return waveform * target_rms / rms_tensor, rms
    return waveform, rms


def trim_silence(
    waveform: torch.Tensor,
    *,
    threshold: float = 1e-3,
    padding: int = 0,
) -> torch.Tensor:
    """Trim leading/trailing low-energy samples without a DSP dependency."""
    if waveform.ndim != 1:
        raise ValueError("Silence trimming expects one mono waveform.")
    active = torch.where(waveform.abs() >= threshold)[0]
    if active.numel() == 0:
        return waveform[:0]
    start = max(0, int(active[0].item()) - padding)
    end = min(waveform.numel(), int(active[-1].item()) + padding + 1)
    return waveform[start:end]


def cross_fade(
    first: torch.Tensor,
    second: torch.Tensor,
    overlap_samples: int,
) -> torch.Tensor:
    """Concatenate mono waveforms using a linear cross-fade.

    The released recipe mixes with float64 NumPy ramps; the overlap is mixed
    in float64 the same way and returned in the inputs' dtype.
    """
    overlap = min(overlap_samples, first.numel(), second.numel())
    if overlap <= 0:
        return torch.cat((first, second))
    fade_out = torch.linspace(1.0, 0.0, overlap, device=first.device, dtype=torch.float64)
    fade_in = torch.linspace(0.0, 1.0, overlap, device=first.device, dtype=torch.float64)
    mixed = (first[-overlap:].double() * fade_out + second[:overlap].double() * fade_in).to(first.dtype)
    return torch.cat((first[:-overlap], mixed, second[overlap:]))


def remove_generated_silence(
    waveform: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    """Drop pauses longer than one second like the released ``remove_silence``.

    Ports ``remove_silence_for_generated_wav``: split on at least 1 s below
    -50 dBFS, keep 500 ms around each phrase, and join the phrases. The
    released recipe applies this to a 16-bit WAVE file it has just written,
    so loudness is measured at 16 bits here too; its output file is
    additionally quantized, while this returns float samples.
    """
    segments = _PydubAudio(waveform.unsqueeze(0), sample_rate, 2).split_on_silence(
        minimum_silence=1_000,
        threshold=-50,
        keep_silence=500,
        seek_step=10,
    )
    return torch.cat(segments, dim=-1).squeeze(0) if segments else waveform[:0]


def pad_mel(
    mel: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    if target_length < mel.shape[1]:
        raise ValueError("Target mel length cannot be shorter than the input.")
    return F.pad(mel, (0, 0, 0, target_length - mel.shape[1]))


__all__ = [
    "F5MelSpectrogram",
    "cross_fade",
    "htk_mel_filter_bank",
    "normalize_reference_rms",
    "pad_mel",
    "preprocess_reference_audio",
    "pydub_sample_width",
    "remove_generated_silence",
    "trim_silence",
]
