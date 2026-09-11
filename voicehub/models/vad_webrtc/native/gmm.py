"""Adaptive two-component GMM decision rule from the WebRTC VAD."""

from __future__ import annotations

from dataclasses import dataclass, field

from voicehub.models.vad_webrtc.native.fixed_point import divide_int32_by_int16, int16, int32, norm_int32

_CHANNELS = 6
_GAUSSIANS = 2
_TABLE_SIZE = _CHANNELS * _GAUSSIANS
_MINIMUM_ENERGY = 10

_SPECTRUM_WEIGHT = (6, 8, 10, 12, 14, 16)
_NOISE_UPDATE = 655
_SPEECH_UPDATE = 6554
_BACK_ETA = 154
_MINIMUM_DIFFERENCE = (544, 544, 576, 576, 576, 576)
_MAXIMUM_SPEECH = (11392, 11392, 11520, 11520, 11520, 11520)
_MINIMUM_MEAN = (640, 768)
_MAXIMUM_NOISE = (9216, 9088, 8960, 8832, 8704, 8576)
_NOISE_WEIGHTS = (34, 62, 72, 66, 53, 25, 94, 66, 56, 62, 75, 103)
_SPEECH_WEIGHTS = (48, 82, 45, 87, 50, 47, 80, 46, 83, 41, 78, 81)
_INITIAL_NOISE_MEANS = (
    6738,
    4892,
    7065,
    6715,
    6771,
    3369,
    7646,
    3863,
    7820,
    7266,
    5020,
    4362,
)
_INITIAL_SPEECH_MEANS = (
    8306,
    10085,
    10078,
    11823,
    11843,
    6309,
    9473,
    9571,
    10879,
    7581,
    8180,
    7483,
)
_INITIAL_NOISE_STDS = (378, 1064, 493, 582, 688, 593, 474, 697, 475, 688, 421, 455)
_INITIAL_SPEECH_STDS = (555, 505, 567, 524, 585, 1231, 509, 828, 492, 1540, 1079, 850)
_MAXIMUM_SPEECH_FRAMES = 6
_MINIMUM_STD = 384

_MODE_PARAMETERS = {
    0: ((8, 4, 3), (14, 7, 5), (24, 21, 24), (57, 48, 57)),
    1: ((8, 4, 3), (14, 7, 5), (37, 32, 37), (100, 80, 100)),
    2: ((6, 3, 2), (9, 5, 3), (82, 78, 82), (285, 260, 285)),
    3: ((6, 3, 2), (9, 5, 3), (94, 94, 94), (1100, 1050, 1100)),
}

_GAUSSIAN_COMPARISON_VARIANCE = 22005
_LOG2_EXPONENT = 5909
_SMOOTHING_DOWN = 6553
_SMOOTHING_UP = 32439


@dataclass
class GMMState:
    """Mutable adaptive model state for one sequential stream."""

    noise_means: list[int] = field(default_factory=lambda: list(_INITIAL_NOISE_MEANS))
    speech_means: list[int] = field(default_factory=lambda: list(_INITIAL_SPEECH_MEANS))
    noise_stds: list[int] = field(default_factory=lambda: list(_INITIAL_NOISE_STDS))
    speech_stds: list[int] = field(default_factory=lambda: list(_INITIAL_SPEECH_STDS))
    frame_counter: int = 0
    over_hang: int = 0
    speech_frames: int = 0
    ages: list[int] = field(default_factory=lambda: [0] * (16 * _CHANNELS))
    low_values: list[int] = field(default_factory=lambda: [10000] * (16 * _CHANNELS))
    mean_values: list[int] = field(default_factory=lambda: [1600] * _CHANNELS)
    over_hang_max_1: tuple[int, int, int] = _MODE_PARAMETERS[0][0]
    over_hang_max_2: tuple[int, int, int] = _MODE_PARAMETERS[0][1]
    individual_thresholds: tuple[int, int, int] = _MODE_PARAMETERS[0][2]
    total_thresholds: tuple[int, int, int] = _MODE_PARAMETERS[0][3]

    def set_mode(self, mode: int) -> None:
        """Set the official aggressiveness thresholds."""
        try:
            (
                self.over_hang_max_1,
                self.over_hang_max_2,
                self.individual_thresholds,
                self.total_thresholds,
            ) = _MODE_PARAMETERS[mode]
        except KeyError as error:
            raise ValueError("WebRTC VAD mode must be 0, 1, 2, or 3.") from error


def _gaussian_probability(
    input_value: int,
    mean: int,
    standard_deviation: int,
) -> tuple[int, int]:
    temporary32 = int32(131072 + (standard_deviation >> 1))
    inverse_std = int16(divide_int32_by_int16(temporary32, standard_deviation), )
    temporary16 = int16(inverse_std >> 2)
    inverse_variance = int16((temporary16 * temporary16) >> 2)

    temporary16 = int16(input_value << 3)
    temporary16 = int16(temporary16 - mean)
    delta = int16((inverse_variance * temporary16) >> 10)
    exponent = int32((delta * temporary16) >> 9)

    exponential = 0
    if exponent < _GAUSSIAN_COMPARISON_VARIANCE:
        temporary16 = int16((_LOG2_EXPONENT * exponent) >> 12)
        temporary16 = int16(-temporary16)
        exponential = int16(0x0400 | (temporary16 & 0x03FF))
        temporary16 = int16(temporary16 ^ 0xFFFF)
        temporary16 = int16(temporary16 >> 10)
        temporary16 = int16(temporary16 + 1)
        exponential = int16(exponential >> temporary16)
    return int32(inverse_std * exponential), delta


def _weighted_average(
    data: list[int],
    channel: int,
    offset: int,
    weights: tuple[int, ...],
) -> int:
    average = 0
    for gaussian in range(_GAUSSIANS):
        index = channel + gaussian * _CHANNELS
        data[index] = int16(data[index] + offset)
        average = int32(average + data[index] * weights[index])
    return average


def _find_minimum(state: GMMState, feature: int, channel: int) -> int:
    offset = channel << 4
    ages = state.ages
    values = state.low_values

    for index in range(16):
        absolute_index = offset + index
        if ages[absolute_index] != 100:
            ages[absolute_index] = int16(ages[absolute_index] + 1)
        else:
            for successor in range(index, 15):
                values[offset + successor] = values[offset + successor + 1]
                ages[offset + successor] = ages[offset + successor + 1]
            ages[offset + 15] = 101
            values[offset + 15] = 10000

    position = -1
    for index in range(16):
        if feature < values[offset + index]:
            position = index
            break
    if position >= 0:
        for index in range(15, position, -1):
            values[offset + index] = values[offset + index - 1]
            ages[offset + index] = ages[offset + index - 1]
        values[offset + position] = feature
        ages[offset + position] = 1

    current_median = 1600
    if state.frame_counter > 2:
        current_median = values[offset + 2]
    elif state.frame_counter > 0:
        current_median = values[offset]

    alpha = 0
    if state.frame_counter > 0:
        alpha = (_SMOOTHING_DOWN if current_median < state.mean_values[channel] else _SMOOTHING_UP)
    temporary = int32((alpha + 1) * state.mean_values[channel])
    temporary = int32(temporary + (32767 - alpha) * current_median)
    temporary = int32(temporary + 16384)
    state.mean_values[channel] = int16(temporary >> 15)
    return state.mean_values[channel]


def classify(
    features: list[int],
    total_power: int,
    frame_length: int,
    state: GMMState,
) -> int:
    """Classify and adapt one 8 kHz frame, returning the raw VAD flag."""
    frame_index = {80: 0, 160: 1}.get(frame_length, 2)
    overhead_1 = state.over_hang_max_1[frame_index]
    overhead_2 = state.over_hang_max_2[frame_index]
    individual_threshold = state.individual_thresholds[frame_index]
    total_threshold = state.total_thresholds[frame_index]

    vad_flag = 0
    noise_deltas = [0] * _TABLE_SIZE
    speech_deltas = [0] * _TABLE_SIZE
    noise_conditionals = [0] * _TABLE_SIZE
    speech_conditionals = [0] * _TABLE_SIZE

    if total_power > _MINIMUM_ENERGY:
        likelihood_sum = 0
        for channel in range(_CHANNELS):
            noise_test = 0
            speech_test = 0
            noise_probability = [0] * _GAUSSIANS
            speech_probability = [0] * _GAUSSIANS
            for gaussian_index in range(_GAUSSIANS):
                gaussian = channel + gaussian_index * _CHANNELS
                probability, noise_deltas[gaussian] = _gaussian_probability(
                    features[channel],
                    state.noise_means[gaussian],
                    state.noise_stds[gaussian],
                )
                noise_probability[gaussian_index] = int32(_NOISE_WEIGHTS[gaussian] * probability, )
                noise_test = int32(noise_test + noise_probability[gaussian_index], )

                probability, speech_deltas[gaussian] = _gaussian_probability(
                    features[channel],
                    state.speech_means[gaussian],
                    state.speech_stds[gaussian],
                )
                speech_probability[gaussian_index] = int32(_SPEECH_WEIGHTS[gaussian] * probability, )
                speech_test = int32(speech_test + speech_probability[gaussian_index], )

            noise_shifts = 31 if noise_test == 0 else norm_int32(noise_test)
            speech_shifts = 31 if speech_test == 0 else norm_int32(speech_test)
            likelihood_ratio = int16(noise_shifts - speech_shifts)
            likelihood_sum = int32(likelihood_sum + likelihood_ratio * _SPECTRUM_WEIGHT[channel], )
            if likelihood_ratio * 4 > individual_threshold:
                vad_flag = 1

            noise_total_q15 = int16(noise_test >> 12)
            if noise_total_q15 > 0:
                numerator = int32((noise_probability[0] & 0xFFFFF000) << 2, )
                noise_conditionals[channel] = int16(divide_int32_by_int16(numerator, noise_total_q15), )
                noise_conditionals[channel + _CHANNELS] = int16(16384 - noise_conditionals[channel], )
            else:
                noise_conditionals[channel] = 16384

            speech_total_q15 = int16(speech_test >> 12)
            if speech_total_q15 > 0:
                numerator = int32((speech_probability[0] & 0xFFFFF000) << 2, )
                speech_conditionals[channel] = int16(divide_int32_by_int16(numerator, speech_total_q15), )
                speech_conditionals[channel + _CHANNELS] = int16(16384 - speech_conditionals[channel], )

        if likelihood_sum >= total_threshold:
            vad_flag |= 1

        maximum_speech = 12800
        for channel in range(_CHANNELS):
            feature_minimum = _find_minimum(
                state,
                features[channel],
                channel,
            )
            noise_global_mean = _weighted_average(
                state.noise_means,
                channel,
                0,
                _NOISE_WEIGHTS,
            )
            mean_q8 = int16(noise_global_mean >> 6)

            for gaussian_index in range(_GAUSSIANS):
                gaussian = channel + gaussian_index * _CHANNELS
                noise_mean = state.noise_means[gaussian]
                speech_mean = state.speech_means[gaussian]
                noise_std = state.noise_stds[gaussian]
                speech_std = state.speech_stds[gaussian]

                updated_noise_mean = noise_mean
                if not vad_flag:
                    delta = int16((noise_conditionals[gaussian] * noise_deltas[gaussian]) >> 11, )
                    updated_noise_mean = int16(noise_mean + int16((delta * _NOISE_UPDATE) >> 22), )

                noise_correction = int16((feature_minimum << 4) - mean_q8, )
                corrected_noise_mean = int16(
                    updated_noise_mean + int16((noise_correction * _BACK_ETA) >> 9), )
                lower_mean = int16((gaussian_index + 5) << 7)
                upper_mean = int16((72 + gaussian_index - channel) << 7, )
                corrected_noise_mean = max(
                    lower_mean,
                    min(upper_mean, corrected_noise_mean),
                )
                state.noise_means[gaussian] = corrected_noise_mean

                if vad_flag:
                    delta = int16((speech_conditionals[gaussian] * speech_deltas[gaussian]) >> 11, )
                    update_q8 = int16((delta * _SPEECH_UPDATE) >> 21)
                    updated_speech_mean = int16(speech_mean + ((update_q8 + 1) >> 1), )
                    maximum_mean = int16(maximum_speech + 640)
                    updated_speech_mean = max(
                        _MINIMUM_MEAN[gaussian_index],
                        min(maximum_mean, updated_speech_mean),
                    )
                    state.speech_means[gaussian] = updated_speech_mean

                    residual = int16((speech_mean + 4) >> 3)
                    residual = int16(features[channel] - residual)
                    temporary32 = int32((speech_deltas[gaussian] * residual) >> 3, )
                    temporary32 = int32(temporary32 - 4096)
                    conditional_q12 = int16(speech_conditionals[gaussian] >> 2, )
                    temporary32 = int32(conditional_q12 * temporary32)
                    temporary32 = int32(temporary32 >> 4)
                    if temporary32 > 0:
                        standard_deviation_update = int16(
                            divide_int32_by_int16(
                                temporary32,
                                int16(speech_std * 10),
                            ), )
                    else:
                        standard_deviation_update = int16(
                            divide_int32_by_int16(
                                -temporary32,
                                int16(speech_std * 10),
                            ), )
                        standard_deviation_update = int16(-standard_deviation_update, )
                    standard_deviation_update = int16(standard_deviation_update + 128, )
                    speech_std = int16(speech_std + (standard_deviation_update >> 8), )
                    state.speech_stds[gaussian] = max(
                        _MINIMUM_STD,
                        speech_std,
                    )
                else:
                    residual = int16(features[channel] - (noise_mean >> 3), )
                    temporary32 = int32((noise_deltas[gaussian] * residual) >> 3, )
                    temporary32 = int32(temporary32 - 4096)
                    conditional_q12 = int16((noise_conditionals[gaussian] + 2) >> 2, )
                    temporary32 = int32(conditional_q12 * temporary32)
                    temporary32 = int32(temporary32 >> 14)
                    if temporary32 > 0:
                        standard_deviation_update = int16(divide_int32_by_int16(
                            temporary32,
                            noise_std,
                        ), )
                    else:
                        standard_deviation_update = int16(divide_int32_by_int16(
                            -temporary32,
                            noise_std,
                        ), )
                        standard_deviation_update = int16(-standard_deviation_update, )
                    standard_deviation_update = int16(standard_deviation_update + 32, )
                    noise_std = int16(noise_std + (standard_deviation_update >> 6), )
                    state.noise_stds[gaussian] = max(
                        _MINIMUM_STD,
                        noise_std,
                    )

            noise_global_mean = _weighted_average(
                state.noise_means,
                channel,
                0,
                _NOISE_WEIGHTS,
            )
            speech_global_mean = _weighted_average(
                state.speech_means,
                channel,
                0,
                _SPEECH_WEIGHTS,
            )
            difference = int16(int16(speech_global_mean >> 9) - int16(noise_global_mean >> 9), )
            if difference < _MINIMUM_DIFFERENCE[channel]:
                correction = int16(_MINIMUM_DIFFERENCE[channel] - difference, )
                speech_offset = int16((13 * correction) >> 2)
                noise_offset = int16((3 * correction) >> 2)
                speech_global_mean = _weighted_average(
                    state.speech_means,
                    channel,
                    speech_offset,
                    _SPEECH_WEIGHTS,
                )
                noise_global_mean = _weighted_average(
                    state.noise_means,
                    channel,
                    -noise_offset,
                    _NOISE_WEIGHTS,
                )

            maximum_speech = _MAXIMUM_SPEECH[channel]
            global_mean_q7 = int16(speech_global_mean >> 7)
            if global_mean_q7 > maximum_speech:
                mean_offset = int16(global_mean_q7 - maximum_speech)
                for gaussian_index in range(_GAUSSIANS):
                    gaussian = channel + gaussian_index * _CHANNELS
                    state.speech_means[gaussian] = int16(state.speech_means[gaussian] - mean_offset, )

            global_mean_q7 = int16(noise_global_mean >> 7)
            if global_mean_q7 > _MAXIMUM_NOISE[channel]:
                mean_offset = int16(global_mean_q7 - _MAXIMUM_NOISE[channel], )
                for gaussian_index in range(_GAUSSIANS):
                    gaussian = channel + gaussian_index * _CHANNELS
                    state.noise_means[gaussian] = int16(state.noise_means[gaussian] - mean_offset, )
        state.frame_counter = int32(state.frame_counter + 1)

    if not vad_flag:
        if state.over_hang > 0:
            vad_flag = 2 + state.over_hang
            state.over_hang = int16(state.over_hang - 1)
        state.speech_frames = 0
    else:
        state.speech_frames = int16(state.speech_frames + 1)
        if state.speech_frames > _MAXIMUM_SPEECH_FRAMES:
            state.speech_frames = _MAXIMUM_SPEECH_FRAMES
            state.over_hang = overhead_2
        else:
            state.over_hang = overhead_1
    return int16(vad_flag)


__all__ = ["GMMState", "classify"]
