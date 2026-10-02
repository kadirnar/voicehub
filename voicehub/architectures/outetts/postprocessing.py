"""OuteTTS V3 text chunking and waveform post-processing.

Ports the author pipeline around generation in ``edwko/OuteTTS``:

* ``outetts/utils/chunking.py``: ``chunk_text`` groups sentences into
  10-30 word chunks for the default chunked generation mode;
* ``outetts/dac/interface.py``: ``DacInterface.decode`` decodes the
  concatenated codes of every chunk in 2048-frame windows, applies a 15 ms
  fade to each window, and passes the result through
  ``process_audio_tensor`` (``pyloudnorm`` BS.1770-4 integrated loudness
  normalized to -18 LUFS, then peak-limited to -1 dBFS).

Only PyTorch and the standard library are used. MeCab-based word counting
for Chinese/Japanese text is not ported; such text keeps sentence chunks.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

TARGET_LOUDNESS_LUFS = -18.0
PEAK_LIMIT_DBFS = -1.0
LOUDNESS_BLOCK_SECONDS = 0.4
FADE_SECONDS = 0.015
DECODE_CHUNK_FRAMES = 2_048

_SENTENCE_END = re.compile(r"([.!?。！？︕︖]+\s*)")


def has_cjk(text: str | Sequence[str]) -> bool:
    """Upstream ``check_language``: Hiragana, Katakana, or CJK ideographs.

    Like upstream, a token list is checked token by token.
    """
    return any("぀" <= item <= "ゟ" or "゠" <= item <= "ヿ" or "一" <= item <= "鿿"
               for item in text)


def _join(tokens: Sequence[str], *, cjk: bool) -> str:
    return "".join(tokens) if cjk else " ".join(tokens)


def chunk_text(text: str, *, min_words: int = 10, max_words: int = 30) -> list[str]:
    """Port of ``outetts.utils.chunking.chunk_text`` for whitespace-delimited
    text."""
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    if not text:
        return []
    parts = _SENTENCE_END.split(text)
    sentences = []
    for index in range(0, len(parts), 2):
        sentence = parts[index] + (parts[index + 1] if index + 1 < len(parts) else "")
        if sentence.strip():
            sentences.append(sentence.strip())

    chunks: list[str] = []
    current = ""
    current_count = 0
    for sentence in sentences:
        tokens = sentence.split()
        if len(tokens) > max_words:
            if current:
                chunks.append(current)
                current = ""
                current_count = 0
            cjk = has_cjk(sentence)
            for start in range(0, len(tokens), max_words):
                chunks.append(_join(tokens[start:start + max_words], cjk=cjk))
            continue
        if current_count + len(tokens) <= max_words:
            current = f"{current} {sentence}" if current else sentence
            current_count += len(tokens)
        elif current_count >= min_words:
            chunks.append(current)
            current = sentence
            current_count = len(tokens)
        else:
            space_left = max_words - current_count
            head, rest = tokens[:space_left], tokens[space_left:]
            if current:
                chunks.append(current + ("".join(head) if has_cjk(head) else " " + " ".join(head)))
            else:
                chunks.append(_join(head, cjk=has_cjk(head)))
            current = _join(rest, cjk=has_cjk(rest))
            current_count = len(rest)
    if current:
        chunks.append(current)
    return chunks


def _biquad(
    kind: str,
    gain_db: float,
    q: float,
    frequency: float,
    sample_rate: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Normalized RBJ coefficients, spelled as in ``pyloudnorm.IIRfilter``."""
    amplitude = 10**(gain_db / 40.0)
    omega = 2.0 * math.pi * (frequency / sample_rate)
    alpha = math.sin(omega) / (2.0 * q)
    cosine = math.cos(omega)
    if kind == "high_shelf":
        shelf = 2 * math.sqrt(amplitude) * alpha
        numerator = (
            amplitude * ((amplitude + 1) + (amplitude - 1) * cosine + shelf),
            -2 * amplitude * ((amplitude - 1) + (amplitude + 1) * cosine),
            amplitude * ((amplitude + 1) + (amplitude - 1) * cosine - shelf),
        )
        denominator = (
            (amplitude + 1) - (amplitude - 1) * cosine + shelf,
            2 * ((amplitude - 1) - (amplitude + 1) * cosine),
            (amplitude + 1) - (amplitude - 1) * cosine - shelf,
        )
    elif kind == "high_pass":
        numerator = ((1 + cosine) / 2, -(1 + cosine), (1 + cosine) / 2)
        denominator = (1 + alpha, -2 * cosine, 1 - alpha)
    else:  # pragma: no cover - private call sites are exhaustive.
        raise ValueError(f"Unknown OuteTTS loudness filter {kind!r}.")
    scale = denominator[0]
    return (
        tuple(value / scale for value in numerator),
        tuple(value / scale for value in denominator),
    )


def _second_order_responses(a1: float, a2: float, length: int) -> tuple[list[float], ...]:
    """Impulse and unit-initial-state responses of ``1 / (1 + a1 z^-1 + a2 z^-2)``."""
    impulse = [1.0, -a1]
    previous = [-a1, a1 * a1 - a2]  # y[-1] = 1
    before_previous = [-a2, a1 * a2]  # y[-2] = 1
    for sequence in (impulse, previous, before_previous):
        while len(sequence) < length:
            sequence.append(-a1 * sequence[-1] - a2 * sequence[-2])
    return impulse[:length], previous[:length], before_previous[:length]


def _lfilter(
    signal: Tensor,
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    block: int = 256,
) -> Tensor:
    """Zero-initial-state biquad filter of a one-dimensional float64 tensor.

    Equivalent to ``scipy.signal.lfilter``: each block's zero-state response
    is an exact Toeplitz product and only the two-sample recursion state is
    carried between blocks, so no impulse response is truncated.
    """
    length = signal.shape[-1]
    padded = torch.nn.functional.pad(signal, (2, 0))
    excitation = (numerator[0] * padded[2:] + numerator[1] * padded[1:-1] + numerator[2] * padded[:-2])
    impulse, previous, before_previous = _second_order_responses(denominator[1], denominator[2], block)
    options = {"dtype": signal.dtype, "device": signal.device}
    lag = torch.arange(block, device=signal.device)
    lag = lag[:, None] - lag[None, :]
    toeplitz = torch.tensor(impulse, **options)[lag.clamp_min(0)].masked_fill(lag < 0, 0.0)
    blocks = math.ceil(length / block)
    excitation = torch.nn.functional.pad(excitation, (0, blocks * block - length)).view(blocks, block)
    zero_state = excitation @ toeplitz.T
    states = []
    last, penultimate = 0.0, 0.0
    for block_penultimate, block_last in zero_state[:, -2:].tolist():
        states.append((last, penultimate))
        last, penultimate = (
            block_last + previous[-1] * last + before_previous[-1] * penultimate,
            block_penultimate + previous[-2] * last + before_previous[-2] * penultimate,
        )
    state = torch.tensor(states, **options)
    output = (zero_state + state[:, :1] * torch.tensor(previous, **options) +
              state[:, 1:] * torch.tensor(before_previous, **options))
    return output.reshape(-1)[:length]


def integrated_loudness(
    audio: Tensor,
    sample_rate: int,
    *,
    block_size: float = LOUDNESS_BLOCK_SECONDS,
) -> float:
    """Mono ITU-R BS.1770-4 gated loudness with ``pyloudnorm.Meter``
    semantics."""
    signal = audio.detach().reshape(-1).to(dtype=torch.float64)
    if signal.shape[0] < block_size * sample_rate:
        raise ValueError("Audio must be at least one loudness block long.")
    for kind, gain_db, q, frequency in (
        ("high_shelf", 4.0, 1 / math.sqrt(2), 1500.0),
        ("high_pass", 0.0, 0.5, 38.0),
    ):
        signal = _lfilter(signal, *_biquad(kind, gain_db, q, frequency, sample_rate))
    step = 1.0 - 0.75
    duration = signal.shape[0] / sample_rate
    block_count = int(round((duration - block_size) / (block_size * step))) + 1
    cumulative = torch.nn.functional.pad(signal.square().cumsum(0), (1, 0))
    lower = [int(block_size * (index * step) * sample_rate) for index in range(block_count)]
    upper = [
        min(int(block_size * (index * step + 1) * sample_rate), signal.shape[0]) for index in range(block_count)
    ]
    energy = ((cumulative[upper] - cumulative[lower]) * (1.0 / (block_size * sample_rate))).clamp_min(0.0)
    block_loudness = -0.691 + 10.0 * torch.log10(energy)
    gated = block_loudness >= -70.0
    if not bool(gated.any()):
        return float("-inf")
    relative = -0.691 + 10.0 * math.log10(float(energy[gated].mean())) - 10.0
    gated = (block_loudness > relative) & (block_loudness > -70.0)
    if not bool(gated.any()):
        return float("-inf")
    return -0.691 + 10.0 * math.log10(float(energy[gated].mean()))


def normalize_loudness(
    audio: Tensor,
    sample_rate: int,
    *,
    target_loudness: float = TARGET_LOUDNESS_LUFS,
    peak_limit: float = PEAK_LIMIT_DBFS,
    block_size: float = LOUDNESS_BLOCK_SECONDS,
) -> Tensor:
    """Upstream ``process_audio_tensor`` for one mono waveform.

    Audio shorter than one block is measured zero-padded, as upstream does.
    Unlike upstream (which divides by an infinite gain and returns NaN), a
    silent waveform is returned unchanged.
    """
    flat = audio.reshape(-1)
    minimum = int(block_size * sample_rate)
    measured = flat
    if flat.shape[0] < minimum:
        measured = torch.nn.functional.pad(flat, (0, minimum - flat.shape[0]))
    loudness = integrated_loudness(measured, sample_rate, block_size=block_size)
    if not math.isfinite(loudness):
        return audio
    normalized = flat.to(dtype=torch.float64) * 10.0**((target_loudness - loudness) / 20.0)
    peak = float(normalized.abs().max())
    threshold = 10**(peak_limit / 20)
    if peak > threshold:
        normalized = normalized * (10.0**(peak_limit / 20.0) / peak)
    return normalized.to(dtype=audio.dtype).reshape(audio.shape)


def apply_fade(audio: Tensor, sample_rate: int, *, seconds: float = FADE_SECONDS) -> Tensor:
    """Linear fade-in/out on the last axis, as in ``DacInterface.apply_fade``."""
    length = min(int(sample_rate * seconds), audio.shape[-1] // 2)
    if not length:
        return audio
    audio = audio.clone()
    options = {"device": audio.device, "dtype": audio.dtype}
    audio[..., :length] *= torch.linspace(0.0, 1.0, length, **options)
    audio[..., audio.shape[-1] - length:] *= torch.linspace(1.0, 0.0, length, **options)
    return audio


def decode_audio_codes(
    decode: Callable[[Tensor], Tensor],
    codes: Tensor,
    sample_rate: int,
    *,
    chunk_frames: int = DECODE_CHUNK_FRAMES,
) -> Tensor:
    """Windowed DAC decode, fades, and loudness normalization
    (``DacInterface.decode``)."""
    pieces = [
        apply_fade(decode(codes[..., start:start + chunk_frames]), sample_rate)
        for start in range(0, codes.shape[-1], chunk_frames)
    ]
    return normalize_loudness(torch.cat(pieces, dim=-1), sample_rate)


__all__ = [
    "DECODE_CHUNK_FRAMES",
    "apply_fade",
    "chunk_text",
    "decode_audio_codes",
    "has_cjk",
    "integrated_loudness",
    "normalize_loudness",
]
