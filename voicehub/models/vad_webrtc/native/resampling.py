"""Stateful fixed-point resamplers used by the WebRTC VAD."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from voicehub.models.vad_webrtc.native.fixed_point import int16, int32, saturate_int16

_VAD_ALL_PASS_Q13 = (5243, 1392)
_RESAMPLE_ALL_PASS = (
    (821, 6110, 12382),
    (3050, 9368, 15063),
)
_COEFFICIENTS_48_TO_32 = (
    (778, -2050, 1087, 23285, 12903, -3783, 441, 222),
    (222, 441, -3783, 12903, 23285, 1087, -2050, 778),
)


@dataclass
class ResamplerState:
    """All persistent memories required by the supported input rates."""

    downsample_by_2: list[int] = field(default_factory=lambda: [0] * 4)
    from_48_to_24: list[int] = field(default_factory=lambda: [0] * 8)
    at_24: list[int] = field(default_factory=lambda: [0] * 16)
    from_24_to_16: list[int] = field(default_factory=lambda: [0] * 8)
    from_16_to_8: list[int] = field(default_factory=lambda: [0] * 8)


def _truncate_all_pass_shift(value: int) -> int:
    value = int32(value) >> 14
    return value + 1 if value < 0 else value


def downsample_by_2(
    samples: Sequence[int],
    state: list[int],
    *,
    state_offset: int = 0,
) -> list[int]:
    """Apply the two-branch VAD decimator used for 16 and 32 kHz."""
    if len(samples) % 2:
        raise ValueError("The WebRTC by-two decimator requires an even frame.")
    first_state = int32(state[state_offset])
    second_state = int32(state[state_offset + 1])
    output: list[int] = []
    for index in range(0, len(samples), 2):
        first = int16((first_state >> 1) + ((_VAD_ALL_PASS_Q13[0] * samples[index]) >> 14), )
        first_state = int32(samples[index] - ((_VAD_ALL_PASS_Q13[0] * first) >> 12), )
        second = int16((second_state >> 1) + ((_VAD_ALL_PASS_Q13[1] * samples[index + 1]) >> 14), )
        second_state = int32(samples[index + 1] - ((_VAD_ALL_PASS_Q13[1] * second) >> 12), )
        output.append(int16(first + second))
    state[state_offset] = first_state
    state[state_offset + 1] = second_state
    return output


def _down_by_2_short_to_int(
    samples: Sequence[int],
    state: list[int],
) -> list[int]:
    half_length = len(samples) >> 1
    output = [0] * half_length

    for index in range(half_length):
        temporary0 = int32((samples[index << 1] << 15) + (1 << 14))
        difference = int32(temporary0 - state[1])
        difference = int32((int32(difference + (1 << 13))) >> 14)
        temporary1 = int32(state[0] + difference * _RESAMPLE_ALL_PASS[1][0], )
        state[0] = temporary0
        difference = _truncate_all_pass_shift(int32(temporary1 - state[2]), )
        temporary0 = int32(state[1] + difference * _RESAMPLE_ALL_PASS[1][1], )
        state[1] = temporary1
        difference = _truncate_all_pass_shift(int32(temporary0 - state[3]), )
        state[3] = int32(state[2] + difference * _RESAMPLE_ALL_PASS[1][2], )
        state[2] = temporary0
        output[index] = int32(state[3] >> 1)

    for index in range(half_length):
        temporary0 = int32((samples[(index << 1) + 1] << 15) + (1 << 14))
        difference = int32(temporary0 - state[5])
        difference = int32((int32(difference + (1 << 13))) >> 14)
        temporary1 = int32(state[4] + difference * _RESAMPLE_ALL_PASS[0][0], )
        state[4] = temporary0
        difference = _truncate_all_pass_shift(int32(temporary1 - state[6]), )
        temporary0 = int32(state[5] + difference * _RESAMPLE_ALL_PASS[0][1], )
        state[5] = temporary1
        difference = _truncate_all_pass_shift(int32(temporary0 - state[7]), )
        state[7] = int32(state[6] + difference * _RESAMPLE_ALL_PASS[0][2], )
        state[6] = temporary0
        output[index] = int32(output[index] + (state[7] >> 1))
    return output


def _down_by_2_int_to_short(
    samples: Sequence[int],
    state: list[int],
) -> list[int]:
    working = [int32(value) for value in samples]
    half_length = len(working) >> 1

    for index in range(half_length):
        source_index = index << 1
        temporary0 = working[source_index]
        difference = int32(temporary0 - state[1])
        difference = int32((int32(difference + (1 << 13))) >> 14)
        temporary1 = int32(state[0] + difference * _RESAMPLE_ALL_PASS[1][0], )
        state[0] = temporary0
        difference = _truncate_all_pass_shift(int32(temporary1 - state[2]), )
        temporary0 = int32(state[1] + difference * _RESAMPLE_ALL_PASS[1][1], )
        state[1] = temporary1
        difference = _truncate_all_pass_shift(int32(temporary0 - state[3]), )
        state[3] = int32(state[2] + difference * _RESAMPLE_ALL_PASS[1][2], )
        state[2] = temporary0
        working[source_index] = int32(state[3] >> 1)

    for index in range(half_length):
        source_index = (index << 1) + 1
        temporary0 = working[source_index]
        difference = int32(temporary0 - state[5])
        difference = int32((int32(difference + (1 << 13))) >> 14)
        temporary1 = int32(state[4] + difference * _RESAMPLE_ALL_PASS[0][0], )
        state[4] = temporary0
        difference = _truncate_all_pass_shift(int32(temporary1 - state[6]), )
        temporary0 = int32(state[5] + difference * _RESAMPLE_ALL_PASS[0][1], )
        state[5] = temporary1
        difference = _truncate_all_pass_shift(int32(temporary0 - state[7]), )
        state[7] = int32(state[6] + difference * _RESAMPLE_ALL_PASS[0][2], )
        state[6] = temporary0
        working[source_index] = int32(state[7] >> 1)

    output: list[int] = []
    for index in range(0, half_length, 2):
        first = int32(working[index << 1] + working[(index << 1) + 1], ) >> 15
        second = int32(working[(index << 1) + 2] + working[(index << 1) + 3], ) >> 15
        output.extend((saturate_int16(first), saturate_int16(second)))
    return output


def _all_pass_stage(
    sample: int,
    state: list[int],
    *,
    base: int,
    coefficients: tuple[int, int, int],
) -> tuple[int, int]:
    difference = int32(sample - state[base + 1])
    difference = int32((int32(difference + (1 << 13))) >> 14)
    temporary1 = int32(state[base] + difference * coefficients[0])
    state[base] = sample
    difference = _truncate_all_pass_shift(int32(temporary1 - state[base + 2]), )
    temporary0 = int32(state[base + 1] + difference * coefficients[1], )
    state[base + 1] = temporary1
    difference = _truncate_all_pass_shift(int32(temporary0 - state[base + 3]), )
    state[base + 3] = int32(state[base + 2] + difference * coefficients[2], )
    state[base + 2] = temporary0
    return temporary0, state[base + 3]


def _low_pass_by_2_int(
    samples: Sequence[int],
    state: list[int],
) -> list[int]:
    half_length = len(samples) >> 1
    output = [0] * len(samples)

    delayed = state[12]
    for index in range(half_length):
        _, filtered = _all_pass_stage(
            delayed,
            state,
            base=0,
            coefficients=_RESAMPLE_ALL_PASS[1],
        )
        output[index << 1] = int32(filtered >> 1)
        delayed = int32(samples[(index << 1) + 1])

    for index in range(half_length):
        _, filtered = _all_pass_stage(
            int32(samples[index << 1]),
            state,
            base=4,
            coefficients=_RESAMPLE_ALL_PASS[0],
        )
        output[index << 1] = int32(int32(output[index << 1] + (filtered >> 1)) >> 15, )

    for index in range(half_length):
        _, filtered = _all_pass_stage(
            int32(samples[index << 1]),
            state,
            base=8,
            coefficients=_RESAMPLE_ALL_PASS[1],
        )
        output[(index << 1) + 1] = int32(filtered >> 1)

    for index in range(half_length):
        _, filtered = _all_pass_stage(
            int32(samples[(index << 1) + 1]),
            state,
            base=12,
            coefficients=_RESAMPLE_ALL_PASS[0],
        )
        output[(index << 1) + 1] = int32(int32(output[(index << 1) + 1] + (filtered >> 1)) >> 15, )
    return output


def _resample_48_to_32(samples: Sequence[int], blocks: int) -> list[int]:
    output: list[int] = []
    for block in range(blocks):
        start = block * 3
        first = 1 << 14
        for coefficient, sample in zip(
                _COEFFICIENTS_48_TO_32[0],
                samples[start:start + 8],
                strict=True,
        ):
            first = int32(first + coefficient * sample)
        second = 1 << 14
        for coefficient, sample in zip(
                _COEFFICIENTS_48_TO_32[1],
                samples[start + 1:start + 9],
                strict=True,
        ):
            second = int32(second + coefficient * sample)
        output.extend((first, second))
    return output


def _resample_48_to_8(
    samples: Sequence[int],
    state: ResamplerState,
) -> list[int]:
    if len(samples) % 480:
        raise ValueError("48 kHz WebRTC frames must contain complete 10 ms blocks.")
    output: list[int] = []
    # The pinned WebRTC implementation invokes its 10 ms resampler once per
    # sub-frame but intentionally retains the original input pointer.  Keeping
    # this historical behavior is required for decision compatibility on
    # 20/30 ms frames.
    block = samples[:480]
    for _ in range(len(samples) // 480):
        at_24 = _down_by_2_short_to_int(block, state.from_48_to_24)
        filtered_24 = _low_pass_by_2_int(at_24, state.at_24)
        resampling_input = [
            *state.from_24_to_16,
            *filtered_24,
        ]
        state.from_24_to_16[:] = filtered_24[-8:]
        at_16 = _resample_48_to_32(resampling_input, 80)
        output.extend(_down_by_2_int_to_short(at_16, state.from_16_to_8), )
    return output


def resample_to_8khz(
    samples: Sequence[int],
    sample_rate: int,
    state: ResamplerState,
) -> list[int]:
    """Resample one valid WebRTC frame to its 8 kHz analysis rate."""
    normalized = [int16(value) for value in samples]
    if sample_rate == 8000:
        return normalized
    if sample_rate == 16000:
        return downsample_by_2(normalized, state.downsample_by_2)
    if sample_rate == 32000:
        at_16 = downsample_by_2(
            normalized,
            state.downsample_by_2,
            state_offset=2,
        )
        return downsample_by_2(at_16, state.downsample_by_2)
    if sample_rate == 48000:
        return _resample_48_to_8(normalized, state)
    raise ValueError("WebRTC VAD supports 8, 16, 32, or 48 kHz audio.")


__all__ = ["ResamplerState", "downsample_by_2", "resample_to_8khz"]
