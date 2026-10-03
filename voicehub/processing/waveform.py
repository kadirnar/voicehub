"""PyTorch-native waveform loading, normalization, and resampling.

Native VoiceHub architectures use this module instead of delegating
their input boundary to NumPy, SoundFile, librosa, or torchaudio. Tensor
and Python sequence inputs are accepted directly.  File input
intentionally starts with the portable PCM WAVE format; additional
codecs can be introduced as explicit VoiceHub decoders without changing
the processor contract.
"""

from __future__ import annotations

import sys
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from math import ceil, gcd, isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def _positive_rate(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return int(value)


def _waveform_from_mapping(value: Mapping[str, Any]) -> tuple[Any, Any]:
    for name in ("array", "waveform", "audio", "input_values"):
        if name in value:
            return value[name], value.get(
                "sampling_rate",
                value.get("sample_rate"),
            )
    raise ValueError("Audio mappings must contain one of: array, waveform, audio, "
                     "input_values.")


def _decode_pcm(payload: bytes, *, sample_width: int) -> Tensor:
    """Decode little-endian PCM frames to normalized float32 samples."""
    if sample_width == 1:
        values = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        return (values.float() - 128.0) / 128.0
    if sample_width == 2:
        values = torch.frombuffer(bytearray(payload), dtype=torch.int16)
        return values.float() / 32768.0
    if sample_width == 3:
        octets = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        if octets.numel() % 3:
            raise ValueError("24-bit WAVE payload is not aligned to samples.")
        octets = octets.reshape(-1, 3).to(dtype=torch.int32)
        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        values = torch.where(
            values >= 2**23,
            values - 2**24,
            values,
        )
        return values.float() / float(2**23)
    if sample_width == 4:
        values = torch.frombuffer(bytearray(payload), dtype=torch.int32)
        return values.float() / float(2**31)
    raise ValueError(
        "VoiceHub's native WAVE decoder supports 8-, 16-, 24-, and 32-bit "
        f"PCM; received {sample_width * 8}-bit samples.")


_WAVE_FORMAT_IEEE_FLOAT = 0x0003
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def _read_float_wave(source) -> tuple[Tensor, int, int, int] | None:
    """Decode an IEEE-float WAVE container, or return ``None`` for others.

    The standard-library ``wave`` module only reads integer PCM, while
    32-bit float WAVE is the default output of common writers such as
    SoundFile. Returns ``(samples, channels, sample_rate, frames)``.
    """
    stream = open(source, "rb") if isinstance(source, str) else source
    try:
        start = stream.tell()
        header = stream.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return None
        layout = None
        while True:
            chunk = stream.read(8)
            if len(chunk) < 8:
                return None
            name = chunk[:4]
            size = int.from_bytes(chunk[4:8], "little")
            if name == b"fmt ":
                fmt = stream.read(size)
                if len(fmt) < 16:
                    return None
                tag = int.from_bytes(fmt[0:2], "little")
                if tag == _WAVE_FORMAT_EXTENSIBLE and len(fmt) >= 26:
                    tag = int.from_bytes(fmt[24:26], "little")
                if tag != _WAVE_FORMAT_IEEE_FLOAT:
                    return None
                layout = (
                    int.from_bytes(fmt[2:4], "little"),
                    int.from_bytes(fmt[4:8], "little"),
                    int.from_bytes(fmt[14:16], "little"),
                )
            elif name == b"data":
                if layout is None:
                    return None
                channels, sample_rate, bits = layout
                if bits not in (32, 64):
                    raise ValueError(
                        f"IEEE-float WAVE input must use 32- or 64-bit samples; received {bits}-bit.")
                if not 1 <= channels <= 8:
                    raise ValueError("WAVE input must contain between one and eight channels.")
                payload = stream.read(size)
                frame_bytes = channels * bits // 8
                payload = payload[:len(payload) - len(payload) % frame_bytes]
                if sys.byteorder != "little":  # pragma: no cover - uncommon platform
                    raise RuntimeError("Native float WAVE decoding requires a little-endian host.")
                dtype = torch.float32 if bits == 32 else torch.float64
                values = torch.frombuffer(bytearray(payload), dtype=dtype).float()
                return values, channels, sample_rate, len(payload) // frame_bytes
            else:
                stream.seek(size + (size & 1), 1)
    finally:
        if stream is not source:
            stream.close()
        else:
            stream.seek(start)


def _read_pcm_wave(
    source,
    *,
    preserve_channels: bool,
    source_label: str,
) -> tuple[Tensor, int]:
    floating = _read_float_wave(source)
    if floating is not None:
        values, channels, sample_rate, frame_count = floating
        if channels > 1:
            values = values.reshape(frame_count, channels).transpose(0, 1)
            if not preserve_channels:
                values = values.mean(dim=0)
        elif preserve_channels:
            values = values.unsqueeze(0)
        return values.contiguous(), _positive_rate(sample_rate, name="sample_rate")
    try:
        with wave.open(source, "rb") as stream:
            if stream.getcomptype() != "NONE":
                raise ValueError("Compressed WAVE input is not supported by the native "
                                 "PCM decoder.")
            channels = stream.getnchannels()
            if not 1 <= channels <= 8:
                raise ValueError("WAVE input must contain between one and eight channels.")
            sample_rate = stream.getframerate()
            sample_width = stream.getsampwidth()
            frame_count = stream.getnframes()
            payload = stream.readframes(frame_count)
    except wave.Error as error:
        raise ValueError(f"Invalid PCM WAVE {source_label}: {error}.") from error

    values = _decode_pcm(payload, sample_width=sample_width)
    expected_samples = frame_count * channels
    if values.numel() != expected_samples:
        raise ValueError(f"WAVE payload contains {values.numel()} samples; expected "
                         f"{expected_samples}.")
    if channels > 1:
        values = values.reshape(frame_count, channels).transpose(0, 1)
        if not preserve_channels:
            values = values.mean(dim=0)
    elif preserve_channels:
        values = values.unsqueeze(0)
    return values.contiguous(), _positive_rate(sample_rate, name="sample_rate")


def load_pcm_wave(
    path: str | Path,
    *,
    preserve_channels: bool = False,
) -> tuple[Tensor, int]:
    """Decode an uncompressed PCM WAVE file with the standard library.

    Args:
        path: File to decode.
        preserve_channels: Return channel-first audio when ``True``. The
            default returns a mono waveform and averages multi-channel input.

    Returns:
        A floating-point waveform and its sampling rate.

    This is the file-level counterpart to :func:`load_native_audio`. Codec
    implementations use the channel-preserving form while ASR and VAD
    processors intentionally consume mono audio.
    """
    if not isinstance(preserve_channels, bool):
        raise TypeError("`preserve_channels` must be a boolean.")
    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Audio file was not found: {source_path}.")
    return _read_pcm_wave(
        str(source_path),
        preserve_channels=preserve_channels,
        source_label=f"file {source_path}",
    )


def decode_pcm_wave(
    payload: bytes | bytearray | memoryview,
    *,
    preserve_channels: bool = False,
    max_bytes: int = 512 * 1024 * 1024,
) -> tuple[Tensor, int]:
    """Decode an in-memory PCM WAVE container without NumPy or SoundFile."""
    if not isinstance(preserve_channels, bool):
        raise TypeError("`preserve_channels` must be a boolean.")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("`max_bytes` must be a positive integer.")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("PCM WAVE `payload` must be bytes-like.")
    encoded = bytes(payload)
    if not encoded:
        raise ValueError("PCM WAVE `payload` cannot be empty.")
    if len(encoded) > max_bytes:
        raise ValueError(f"PCM WAVE payload is {len(encoded)} bytes; the limit is "
                         f"{max_bytes}.")
    return _read_pcm_wave(
        BytesIO(encoded),
        preserve_channels=preserve_channels,
        source_label="payload",
    )


def normalize_waveform(value: Any) -> Tensor:
    """Return one finite mono float32 waveform without copying
    unnecessarily."""
    if isinstance(value, Tensor):
        waveform = value.detach()
    else:
        if isinstance(value, (str, bytes, bytearray, Mapping)):
            raise TypeError("Audio samples must be a real numeric sequence.")
        try:
            waveform = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as error:
            raise TypeError("Audio input must expose a finite real numeric array.") from error

    if waveform.numel() == 0:
        raise ValueError("Audio input cannot be empty.")
    if waveform.dtype == torch.bool or waveform.is_complex():
        raise TypeError("Audio samples must be real numeric values.")
    if waveform.is_floating_point():
        waveform = waveform.float()
    elif waveform.dtype == torch.uint8:
        waveform = (waveform.float() - 128.0) / 128.0
    else:
        limits = torch.iinfo(waveform.dtype)
        scale = float(max(abs(limits.min), limits.max))
        waveform = waveform.float() / scale
    if waveform.ndim == 0:
        waveform = waveform.reshape(1)
    while waveform.ndim > 1 and 1 in waveform.shape:
        waveform = waveform.squeeze()
    if waveform.ndim == 2:
        first_is_channels = waveform.shape[0] <= 8
        last_is_channels = waveform.shape[1] <= 8
        if first_is_channels and (not last_is_channels or waveform.shape[0] <= waveform.shape[1]):
            waveform = waveform.float().mean(dim=0)
        elif last_is_channels:
            waveform = waveform.float().mean(dim=1)
        else:
            raise ValueError("Two-dimensional audio must have a channel dimension of at "
                             "most eight.")
    if waveform.ndim != 1:
        raise ValueError(
            "Audio input must resolve to one mono waveform; received shape "
            f"{tuple(waveform.shape)}.")

    if not torch.isfinite(waveform).all():
        raise ValueError("Audio input contains NaN or infinite samples.")
    return waveform.contiguous()


def resample_waveform(
    waveform: Tensor,
    source_rate: int,
    target_rate: int,
    *,
    filter_width: int = 16,
    chunk_size: int = 8_192,
) -> Tensor:
    """Band-limit and resample a mono waveform with a windowed-sinc kernel.

    The operation is differentiable with respect to ``waveform``.  Work is
    chunked so long recordings do not allocate a full
    ``output_samples × kernel_width`` matrix.
    """
    source_rate = _positive_rate(source_rate, name="source_rate")
    target_rate = _positive_rate(target_rate, name="target_rate")
    if not isinstance(waveform, Tensor) or waveform.ndim != 1:
        raise ValueError("`waveform` must be a rank-one PyTorch tensor.")
    if not waveform.is_floating_point():
        raise TypeError("`waveform` must use a floating-point dtype.")
    if waveform.numel() == 0:
        raise ValueError("`waveform` cannot be empty.")
    if (isinstance(filter_width, bool) or not isinstance(filter_width, int) or filter_width < 2):
        raise ValueError("`filter_width` must be an integer of at least two.")
    if (isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError("`chunk_size` must be a positive integer.")
    if source_rate == target_rate:
        return waveform
    if waveform.numel() == 1:
        output_length = max(
            1,
            round(waveform.numel() * target_rate / source_rate),
        )
        return waveform.expand(output_length).clone()

    output_length = max(
        1,
        round(waveform.numel() * target_rate / source_rate),
    )
    computation_dtype = (torch.float64 if waveform.dtype == torch.float64 else torch.float32)
    source = waveform.to(dtype=computation_dtype)
    ratio = source_rate / target_rate
    cutoff = min(1.0, target_rate / source_rate)
    offsets = torch.arange(
        -filter_width + 1,
        filter_width + 1,
        dtype=computation_dtype,
        device=waveform.device,
    )
    chunks: list[Tensor] = []

    for start in range(0, output_length, chunk_size):
        stop = min(start + chunk_size, output_length)
        positions = (torch.arange(
            start,
            stop,
            dtype=computation_dtype,
            device=waveform.device,
        ) * ratio)
        left = positions.floor().to(dtype=torch.long)
        indices = left[:, None] + offsets.to(dtype=torch.long)[None, :]
        distances = positions[:, None] - indices.to(dtype=computation_dtype)
        scaled = distances * cutoff
        support = scaled.abs() / filter_width
        window = torch.where(
            support <= 1.0,
            0.5 * (1.0 + torch.cos(torch.pi * support)),
            torch.zeros_like(support),
        )
        weights = cutoff * torch.sinc(scaled) * window
        weights /= weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(computation_dtype).eps)
        samples = source[indices.clamp(min=0, max=source.shape[0] - 1)]
        chunks.append((samples * weights).sum(dim=-1))

    return torch.cat(chunks).to(dtype=waveform.dtype)


def resample_waveform_kaiser(
    waveform: Tensor,
    source_rate: int,
    target_rate: int,
    *,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    beta: float = 14.769656459379492,
) -> Tensor:
    """Resample ``[..., time]`` audio with a Kaiser-windowed sinc kernel.

    This is the native counterpart of the well-known polyphase algorithm
    used by torchaudio.  Rates are reduced by their greatest common
    divisor, one phase kernel is constructed per target-rate step, and
    all leading dimensions are treated as independent waveforms.  The
    operation remains differentiable with respect to its input.
    """
    source_rate = _positive_rate(source_rate, name="source_rate")
    target_rate = _positive_rate(target_rate, name="target_rate")
    if not isinstance(waveform, Tensor) or waveform.ndim < 1:
        raise ValueError("`waveform` must be a PyTorch tensor with a time axis.")
    if not waveform.is_floating_point():
        raise TypeError("`waveform` must use a floating-point dtype.")
    if waveform.shape[-1] == 0:
        raise ValueError("`waveform` cannot be empty.")
    if (isinstance(lowpass_filter_width, bool) or not isinstance(lowpass_filter_width, int) or
            lowpass_filter_width <= 0):
        raise ValueError("`lowpass_filter_width` must be a positive integer.")
    if (isinstance(rolloff, bool) or not isinstance(rolloff, Real) or not isfinite(float(rolloff)) or
            not 0.0 < float(rolloff) <= 1.0):
        raise ValueError("`rolloff` must be finite and in the interval (0, 1].")
    if (isinstance(beta, bool) or not isinstance(beta, Real) or not isfinite(float(beta)) or
            float(beta) <= 0.0):
        raise ValueError("`beta` must be a finite positive number.")
    if source_rate == target_rate:
        return waveform

    divisor = gcd(source_rate, target_rate)
    original_frequency = source_rate // divisor
    target_frequency = target_rate // divisor
    base_frequency = min(
        original_frequency,
        target_frequency,
    ) * float(rolloff)
    width = ceil(lowpass_filter_width * original_frequency / base_frequency)
    computation_dtype = (torch.float64 if waveform.dtype == torch.float64 else torch.float32)
    source = waveform.to(dtype=computation_dtype)
    indices = (
        torch.arange(
            -width,
            width + original_frequency,
            dtype=computation_dtype,
            device=waveform.device,
        )[None, None] / original_frequency)
    phases = (
        torch.arange(
            0,
            -target_frequency,
            -1,
            dtype=computation_dtype,
            device=waveform.device,
        )[:, None, None] / target_frequency)
    positions = (phases + indices) * base_frequency
    positions = positions.clamp(
        min=-lowpass_filter_width,
        max=lowpass_filter_width,
    )
    normalized = positions / lowpass_filter_width
    beta_tensor = torch.tensor(
        float(beta),
        dtype=computation_dtype,
        device=waveform.device,
    )
    window = torch.i0(beta_tensor * torch.sqrt(
        (1.0 - normalized.square()).clamp_min(0.0))) / torch.i0(beta_tensor)
    radians = positions * torch.pi
    sinc = torch.where(
        radians == 0,
        torch.ones_like(radians),
        radians.sin() / radians,
    )
    kernel = sinc * window * (base_frequency / original_frequency)

    leading_shape = source.shape[:-1]
    source_length = source.shape[-1]
    flattened = source.reshape(-1, 1, source_length)
    padded = torch.nn.functional.pad(
        flattened,
        (width, width + original_frequency - 1),
    )
    resampled = torch.nn.functional.conv1d(
        padded,
        kernel,
        stride=original_frequency,
    )
    output_length = ceil(target_frequency * source_length / original_frequency)
    resampled = (resampled.transpose(1, 2).reshape(*leading_shape, -1)[..., :output_length].contiguous())
    return resampled.to(dtype=waveform.dtype)


_HANN_MATCH_MODES = ("functional", "transform")


def resample_waveform_hann(
    waveform: Tensor,
    source_rate: int,
    target_rate: int,
    *,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    match: str = "functional",
) -> Tensor:
    """Resample ``[..., time]`` audio with a Hann-windowed sinc kernel.

    This is torchaudio's ``sinc_interp_hann`` polyphase resampler,
    reproduced bit for bit (same kernel arithmetic, dtypes, padding and
    output-length rounding) without importing torchaudio. Use it where a
    model's reference frontend resampled with torchaudio, since
    codec tokens and speaker embeddings are sensitive to the filter's
    transition band; :func:`resample_waveform` is not a substitute there.

    ``match`` selects which torchaudio entry point is reproduced:

    * ``"functional"`` (default): ``torchaudio.functional.resample``. The
      kernel is evaluated in the waveform's own dtype and on its device.
    * ``"transform"``: ``torchaudio.transforms.Resample(source_rate,
      target_rate)`` moved to the waveform's device and dtype. The kernel
      is evaluated on the CPU in float64 (output phases in the default
      dtype), rounded to float32 and then cast to the waveform's dtype.

    All leading dimensions are independent waveforms and the operation is
    differentiable with respect to ``waveform``.
    """
    source_rate = _positive_rate(source_rate, name="source_rate")
    target_rate = _positive_rate(target_rate, name="target_rate")
    if not isinstance(waveform, Tensor) or waveform.ndim < 1:
        raise ValueError("`waveform` must be a PyTorch tensor with a time axis.")
    if not waveform.is_floating_point():
        raise TypeError("`waveform` must use a floating-point dtype.")
    if waveform.shape[-1] == 0:
        raise ValueError("`waveform` cannot be empty.")
    if (isinstance(lowpass_filter_width, bool) or not isinstance(lowpass_filter_width, int) or
            lowpass_filter_width <= 0):
        raise ValueError("`lowpass_filter_width` must be a positive integer.")
    if (isinstance(rolloff, bool) or not isinstance(rolloff, Real) or not isfinite(float(rolloff)) or
            not 0.0 < float(rolloff) <= 1.0):
        raise ValueError("`rolloff` must be finite and in the interval (0, 1].")
    if match not in _HANN_MATCH_MODES:
        raise ValueError("`match` must be 'functional' or 'transform'.")
    if source_rate == target_rate:
        return waveform

    divisor = gcd(source_rate, target_rate)
    original_frequency = source_rate // divisor
    target_frequency = target_rate // divisor
    if match == "functional":
        kernel, width = _hann_sinc_kernel(
            original_frequency,
            target_frequency,
            lowpass_filter_width,
            float(rolloff),
            dtype=waveform.dtype,
            device=waveform.device,
        )
    else:
        kernel, width = _hann_sinc_kernel(
            original_frequency,
            target_frequency,
            lowpass_filter_width,
            float(rolloff),
            dtype=None,
            device=torch.device("cpu"),
        )
        kernel = kernel.to(device=waveform.device, dtype=waveform.dtype)

    shape = waveform.shape
    flattened = waveform.reshape(-1, shape[-1])
    length = flattened.shape[-1]
    padded = torch.nn.functional.pad(flattened, (width, width + original_frequency))
    resampled = torch.nn.functional.conv1d(padded[:, None], kernel, stride=original_frequency)
    resampled = resampled.transpose(1, 2).reshape(flattened.shape[0], -1)
    # torchaudio rounds the output length through a default-dtype (float32)
    # tensor, which drops a final partial sample for some long inputs (for
    # example 47,561 samples at 16 kHz -> 22.05 kHz); keep its length.
    output_length = int(torch.ceil(torch.as_tensor(target_frequency * length / original_frequency)))
    return resampled[..., :output_length].reshape(*shape[:-1], -1)


def _hann_sinc_kernel(
    original_frequency: int,
    target_frequency: int,
    lowpass_filter_width: int,
    rolloff: float,
    *,
    dtype: torch.dtype | None,
    device: torch.device,
) -> tuple[Tensor, int]:
    """Return torchaudio's ``sinc_interp_hann`` kernel and its half width.

    Mirrors ``torchaudio.functional._get_sinc_resample_kernel`` operation
    for operation: ``dtype=None`` is the ``transforms.Resample`` kernel
    (float64 taps, default-dtype phases, float32 result), otherwise the
    ``functional.resample`` kernel evaluated in ``dtype`` on ``device``.
    """
    base_frequency = min(original_frequency, target_frequency) * rolloff
    width = ceil(lowpass_filter_width * original_frequency / base_frequency)
    index_dtype = torch.float64 if dtype is None else dtype
    indices = torch.arange(
        -width,
        width + original_frequency,
        dtype=index_dtype,
        device=device,
    )[None, None] / original_frequency
    positions = torch.arange(
        0,
        -target_frequency,
        -1,
        dtype=dtype,
        device=device,
    )[:, None, None] / target_frequency + indices
    positions *= base_frequency
    positions = positions.clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(positions * torch.pi / lowpass_filter_width / 2)**2
    positions *= torch.pi
    scale = base_frequency / original_frequency
    kernel = torch.where(positions == 0, torch.tensor(1.0).to(positions), positions.sin() / positions)
    kernel *= window * scale
    if dtype is None:
        kernel = kernel.to(dtype=torch.float32)
    return kernel, width


# resampy 0.4 ``kaiser_best``: 50 zero crossings, 2**13 table entries per
# crossing, Kaiser taper. The published table is ``sinc_window`` with these
# parameters (maximum deviation 2.3e-12).
_KAISER_BEST_ZEROS = 50
_KAISER_BEST_PRECISION = 2**13
_KAISER_BEST_ROLLOFF = 0.9173473712608761
_KAISER_BEST_BETA = 12.984585251213232
_kaiser_best_table: Tensor | None = None


def _resampy_kaiser_best_table() -> Tensor:
    global _kaiser_best_table
    if _kaiser_best_table is None:
        count = _KAISER_BEST_PRECISION * _KAISER_BEST_ZEROS
        positions = torch.linspace(0, _KAISER_BEST_ZEROS, count + 1, dtype=torch.float64)
        sinc = _KAISER_BEST_ROLLOFF * torch.sinc(_KAISER_BEST_ROLLOFF * positions)
        taper = torch.kaiser_window(
            2 * count + 1,
            periodic=False,
            beta=_KAISER_BEST_BETA,
            dtype=torch.float64,
        )[count:]
        _kaiser_best_table = taper * sinc
    return _kaiser_best_table


def resample_waveform_kaiser_best(
    waveform: Tensor,
    source_rate: int,
    target_rate: int,
) -> Tensor:
    """Resample mono audio like ``librosa.load(path, sr=target_rate)``.

    This reproduces librosa's default ``res_type="kaiser_best"`` path:
    resampy's interpolated windowed-sinc filter (float64 weights, samples
    accumulated in the input dtype in resampy's order), followed by
    librosa's ``fix_length`` to ``ceil(samples * target / source)``.
    Upstream recipes that load audio with librosa defaults need it to see
    the same samples; the generic :func:`resample_waveform` uses a
    different Hann-windowed kernel.
    """
    source_rate = _positive_rate(source_rate, name="source_rate")
    target_rate = _positive_rate(target_rate, name="target_rate")
    if not isinstance(waveform, Tensor) or waveform.ndim != 1:
        raise ValueError("`waveform` must be a rank-one PyTorch tensor.")
    if not waveform.is_floating_point():
        raise TypeError("`waveform` must use a floating-point dtype.")
    if waveform.numel() == 0:
        raise ValueError("`waveform` cannot be empty.")
    if source_rate == target_rate:
        return waveform
    ratio = float(target_rate) / source_rate
    resampled_length = int(waveform.numel() * float(target_rate) / float(source_rate))
    if resampled_length < 1:
        raise ValueError("`waveform` is too short to resample.")
    device = waveform.device
    table = _resampy_kaiser_best_table().to(device=device)
    if ratio < 1:
        table = ratio * table
    delta = torch.diff(table, append=table[-1:])
    scale = min(1.0, ratio)
    index_step = int(scale * _KAISER_BEST_PRECISION)
    times = torch.arange(resampled_length, dtype=torch.float64, device=device) * (1.0 / ratio)
    whole = times.to(dtype=torch.long)
    fraction = scale * (times - whole.to(dtype=torch.float64))
    rows = -(-table.numel() // index_step)
    # Row ``k`` views the filter entries ``k * index_step + offset`` for
    # ``0 <= offset <= index_step``: one contiguous row per tap keeps the
    # per-sample lookups cache-local.
    padding = (0, (rows + 1) * index_step + 1 - table.numel())
    weight_rows = torch.nn.functional.pad(table, padding).unfold(0, index_step + 1, index_step)
    delta_rows = torch.nn.functional.pad(delta, padding).unfold(0, index_step + 1, index_step)
    # Zero padding replaces resampy's per-sample edge bounds: a zero
    # sample adds an exact zero to the float32 accumulator.
    padded = torch.nn.functional.pad(waveform.to(dtype=torch.float64), (rows, rows))
    output = torch.zeros(resampled_length, dtype=torch.float64, device=device)
    wings = (
        (fraction, -1, whole + rows),
        (scale - fraction, 1, whole + rows + 1),
    )
    for wing_fraction, direction, origin in wings:
        position = wing_fraction * _KAISER_BEST_PRECISION
        offset = position.to(dtype=torch.long)
        eta = position - offset.to(dtype=torch.float64)
        limits = (table.numel() - offset) // index_step
        minimum = int(limits.min().item())
        for tap in range(int(limits.max().item())):
            weights = torch.addcmul(
                weight_rows[tap].index_select(0, offset),
                eta,
                delta_rows[tap].index_select(0, offset),
            )
            if tap >= minimum:
                weights = weights.masked_fill(limits <= tap, 0.0)
            products = weights * padded.index_select(0, origin + direction * tap)
            # resampy adds each float64 product to a float32 output in
            # tap order; reproduce that rounding sequence exactly.
            output = (output + products).to(dtype=waveform.dtype).to(dtype=torch.float64)
    output = output.to(dtype=waveform.dtype)
    output_length = ceil(waveform.numel() * ratio)
    return _fix_length(output, output_length)


def _fix_length(waveform: Tensor, length: int) -> Tensor:
    """Right-pad with zeros or trim to ``length`` (librosa ``fix_length``)."""
    if waveform.numel() >= length:
        return waveform[:length].contiguous()
    return torch.nn.functional.pad(waveform, (0, length - waveform.numel()))


@dataclass(frozen=True, slots=True)
class NativeAudio:
    """A normalized mono waveform with an explicit sampling rate."""

    waveform: Tensor
    sampling_rate: int
    path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.waveform, Tensor) or self.waveform.ndim != 1:
            raise ValueError("`waveform` must be a rank-one PyTorch tensor.")
        if not self.waveform.is_floating_point():
            raise TypeError("`waveform` must use a floating-point dtype.")
        if self.waveform.numel() == 0 or not torch.isfinite(self.waveform).all():
            raise ValueError("`waveform` must contain finite audio samples.")
        object.__setattr__(
            self,
            "sampling_rate",
            _positive_rate(self.sampling_rate, name="sampling_rate"),
        )

    @property
    def duration(self) -> float:
        """Waveform duration in seconds."""
        return self.waveform.shape[-1] / self.sampling_rate


def load_native_audio(
    audio: NativeAudio | Mapping[str, Any] | str | Path | Any,
    *,
    sampling_rate: int | None = None,
    target_sampling_rate: int | None = None,
    num_samples: int | None = None,
) -> NativeAudio:
    """Materialize audio using only the standard library and PyTorch.

    ``num_samples`` trims valid samples in the source-rate domain before
    optional resampling. This is the safe boundary for padded training
    batches and file-backed rows carrying an explicit valid length.
    """
    source_path: Path | None = None
    if isinstance(audio, NativeAudio):
        if (sampling_rate is not None and int(sampling_rate) != audio.sampling_rate):
            raise ValueError("`sampling_rate` conflicts with the rate stored in "
                             "NativeAudio.")
        waveform = audio.waveform
        source_rate = audio.sampling_rate
        source_path = audio.path
    elif isinstance(audio, Mapping):
        waveform, mapped_rate = _waveform_from_mapping(audio)
        if (sampling_rate is not None and mapped_rate is not None and int(sampling_rate) != int(mapped_rate)):
            raise ValueError("`sampling_rate` conflicts with the rate stored in the audio "
                             "mapping.")
        source_rate = sampling_rate if sampling_rate is not None else mapped_rate
    elif isinstance(audio, (str, Path)):
        source_path = Path(audio).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(f"Audio file was not found: {source_path}.")
        if source_path.suffix.lower() not in {".wav", ".wave"}:
            raise ValueError(
                "Native VoiceHub file decoding currently accepts PCM WAVE "
                "input. Pass other formats as a decoded tensor.")
        waveform, file_rate = load_pcm_wave(source_path)
        if sampling_rate is not None and int(sampling_rate) != file_rate:
            raise ValueError("`sampling_rate` does not match the WAVE file's sampling rate.")
        source_rate = file_rate
    else:
        waveform = audio
        source_rate = sampling_rate

    source_rate = _positive_rate(source_rate, name="sampling_rate")
    resolved_target = (
        source_rate if target_sampling_rate is None else _positive_rate(
            target_sampling_rate,
            name="target_sampling_rate",
        ))
    normalized = normalize_waveform(waveform)
    if num_samples is not None:
        if (isinstance(num_samples, bool) or not isinstance(num_samples, Integral)):
            raise TypeError("`num_samples` must be an integer.")
        valid_samples = int(num_samples)
        if valid_samples <= 0:
            raise ValueError("`num_samples` must be positive.")
        if valid_samples > normalized.numel():
            raise ValueError("`num_samples` exceeds the waveform's source sample count.")
        normalized = normalized[:valid_samples]
    normalized = resample_waveform(
        normalized,
        source_rate,
        resolved_target,
    )
    return NativeAudio(
        waveform=normalized,
        sampling_rate=resolved_target,
        path=source_path,
    )


def save_pcm_wave(
    path: str | Path,
    waveform: Tensor,
    sampling_rate: int,
) -> Path:
    """Write finite mono or channel-first audio as 16-bit PCM WAVE.

    The writer deliberately has one portable output contract. Richer
    codecs belong behind explicit export strategies; model examples and
    native runtimes should not need NumPy, SoundFile, or torchaudio just
    to emit a playable waveform.
    """
    rate = _positive_rate(sampling_rate, name="sampling_rate")
    if not isinstance(waveform, Tensor):
        raise TypeError("`waveform` must be a PyTorch tensor.")
    if waveform.ndim == 1:
        channels = 1
        samples = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        if not 1 <= waveform.shape[0] <= 8:
            raise ValueError("Channel-first WAVE output must contain one to eight channels.")
        channels = waveform.shape[0]
        samples = waveform
    else:
        raise ValueError("`waveform` must have shape [time] or [channels, time].")
    if samples.shape[-1] == 0:
        raise ValueError("`waveform` cannot be empty.")
    if samples.dtype == torch.bool or samples.is_complex():
        raise TypeError("WAVE samples must be real numeric values.")
    materialized = samples.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(materialized).all():
        raise ValueError("WAVE samples must be finite.")
    interleaved = (
        materialized.clamp(-1.0, 1.0).transpose(0, 1).mul(32767.0).round().to(dtype=torch.int16).contiguous())
    if sys.byteorder != "little":  # pragma: no cover - uncommon platform
        raise RuntimeError("Native WAVE writing currently requires a little-endian host.")
    payload = bytes(interleaved.untyped_storage())
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(payload)
    return output_path


__all__ = [
    "NativeAudio",
    "load_pcm_wave",
    "load_native_audio",
    "normalize_waveform",
    "resample_waveform",
    "resample_waveform_hann",
    "resample_waveform_kaiser",
    "resample_waveform_kaiser_best",
    "save_pcm_wave",
]
