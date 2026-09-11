"""Six-band fixed-point feature extractor from the WebRTC VAD."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from voicehub.models.vad_webrtc.native.fixed_point import int16, int32, norm_int32, norm_uint32, uint32

_LOG_CONSTANT = 24660
_LOG_ENERGY_INTEGER_PART = 14336
_HIGH_PASS_ZERO_COEFFICIENTS = (6631, -13262, 6631)
_HIGH_PASS_POLE_COEFFICIENTS = (16384, -7756, 5620)
_ALL_PASS_COEFFICIENTS_Q15 = (20972, 5571)
_ENERGY_OFFSETS = (368, 368, 272, 176, 176, 176)
_MINIMUM_ENERGY = 10


@dataclass
class FilterbankState:
    """Persistent filter memories for one sequential VAD stream."""

    upper: list[int] = field(default_factory=lambda: [0] * 5)
    lower: list[int] = field(default_factory=lambda: [0] * 5)
    high_pass: list[int] = field(default_factory=lambda: [0] * 4)


def _all_pass_filter(
    samples: Sequence[int],
    coefficient: int,
    state: int,
) -> tuple[list[int], int]:
    state32 = int32(state * (1 << 16))
    output: list[int] = []
    for sample in samples:
        temporary32 = int32(state32 + coefficient * sample)
        temporary16 = int16(temporary32 >> 16)
        output.append(temporary16)
        state32 = int32((sample * (1 << 14)) - coefficient * temporary16)
        state32 = int32(state32 * 2)
    return output, int16(state32 >> 16)


def _split_filter(
    samples: Sequence[int],
    upper_state: int,
    lower_state: int,
) -> tuple[list[int], list[int], int, int]:
    half_length = len(samples) >> 1
    upper, upper_state = _all_pass_filter(
        samples[0:half_length * 2:2],
        _ALL_PASS_COEFFICIENTS_Q15[0],
        upper_state,
    )
    lower, lower_state = _all_pass_filter(
        samples[1:half_length * 2:2],
        _ALL_PASS_COEFFICIENTS_Q15[1],
        lower_state,
    )
    high: list[int] = []
    low: list[int] = []
    for upper_value, lower_value in zip(upper, lower, strict=True):
        high.append(int16(upper_value - lower_value))
        low.append(int16(lower_value + upper_value))
    return high, low, upper_state, lower_state


def _high_pass_filter(samples: Sequence[int], state: list[int]) -> list[int]:
    output: list[int] = []
    for sample in samples:
        temporary = int32(_HIGH_PASS_ZERO_COEFFICIENTS[0] * sample)
        temporary = int32(temporary + _HIGH_PASS_ZERO_COEFFICIENTS[1] * state[0])
        temporary = int32(temporary + _HIGH_PASS_ZERO_COEFFICIENTS[2] * state[1])
        state[1] = state[0]
        state[0] = sample
        temporary = int32(temporary - _HIGH_PASS_POLE_COEFFICIENTS[1] * state[2])
        temporary = int32(temporary - _HIGH_PASS_POLE_COEFFICIENTS[2] * state[3])
        state[3] = state[2]
        state[2] = int16(temporary >> 14)
        output.append(state[2])
    return output


def _scaling_for_square(samples: Sequence[int]) -> int:
    number_of_bits = len(samples).bit_length()
    maximum = -1
    for sample in samples:
        absolute = int16(sample if sample > 0 else -sample)
        if absolute > maximum:
            maximum = absolute
    normalization = norm_int32(maximum * maximum)
    if maximum == 0:
        return 0
    return 0 if normalization > number_of_bits else number_of_bits - normalization


def _energy(samples: Sequence[int]) -> tuple[int, int]:
    scaling = _scaling_for_square(samples)
    value = 0
    for sample in samples:
        value = int32(value + ((sample * sample) >> scaling))
    return value, scaling


def _log_energy(
    samples: Sequence[int],
    offset: int,
    total_energy: int,
) -> tuple[int, int]:
    energy, right_shifts = _energy(samples)
    energy = uint32(energy)
    if energy == 0:
        return offset, total_energy

    normalizing_shifts = 17 - norm_uint32(energy)
    right_shifts += normalizing_shifts
    if normalizing_shifts < 0:
        energy = uint32(energy << -normalizing_shifts)
    else:
        energy >>= normalizing_shifts

    log2_energy = int16(_LOG_ENERGY_INTEGER_PART + ((energy & 0x00003FFF) >> 4), )
    log_energy = int16(((_LOG_CONSTANT * log2_energy) >> 19) + ((right_shifts * _LOG_CONSTANT) >> 9), )
    if log_energy < 0:
        log_energy = 0
    log_energy = int16(log_energy + offset)

    if total_energy <= _MINIMUM_ENERGY:
        if right_shifts >= 0:
            total_energy = int16(total_energy + _MINIMUM_ENERGY + 1)
        else:
            total_energy = int16(total_energy + (energy >> -right_shifts))
    return log_energy, total_energy


def calculate_features(
    samples: Sequence[int],
    state: FilterbankState,
) -> tuple[list[int], int]:
    """Extract the reference 80–4000 Hz log-energy feature vector."""
    if len(samples) not in (80, 160, 240):
        raise ValueError("WebRTC VAD expects 80, 160, or 240 samples at 8 kHz.")

    features = [0] * 6
    total_energy = 0

    high_120, low_120, state.upper[0], state.lower[0] = _split_filter(
        samples,
        state.upper[0],
        state.lower[0],
    )
    high_60, low_60, state.upper[1], state.lower[1] = _split_filter(
        high_120,
        state.upper[1],
        state.lower[1],
    )
    features[5], total_energy = _log_energy(
        high_60,
        _ENERGY_OFFSETS[5],
        total_energy,
    )
    features[4], total_energy = _log_energy(
        low_60,
        _ENERGY_OFFSETS[4],
        total_energy,
    )

    high_60, low_60, state.upper[2], state.lower[2] = _split_filter(
        low_120,
        state.upper[2],
        state.lower[2],
    )
    features[3], total_energy = _log_energy(
        high_60,
        _ENERGY_OFFSETS[3],
        total_energy,
    )

    high_120, low_120, state.upper[3], state.lower[3] = _split_filter(
        low_60,
        state.upper[3],
        state.lower[3],
    )
    features[2], total_energy = _log_energy(
        high_120,
        _ENERGY_OFFSETS[2],
        total_energy,
    )

    high_60, low_60, state.upper[4], state.lower[4] = _split_filter(
        low_120,
        state.upper[4],
        state.lower[4],
    )
    features[1], total_energy = _log_energy(
        high_60,
        _ENERGY_OFFSETS[1],
        total_energy,
    )
    lowest_band = _high_pass_filter(low_60, state.high_pass)
    features[0], total_energy = _log_energy(
        lowest_band,
        _ENERGY_OFFSETS[0],
        total_energy,
    )
    return features, total_energy


__all__ = ["FilterbankState", "calculate_features"]
