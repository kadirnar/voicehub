"""Dependency-free WebRTC fixed-point voice activity detector."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from voicehub.models.vad_webrtc.native.filterbank import FilterbankState, calculate_features
from voicehub.models.vad_webrtc.native.gmm import GMMState, classify
from voicehub.models.vad_webrtc.native.resampling import ResamplerState, resample_to_8khz

_VALID_SAMPLE_RATES = (8000, 16000, 32000, 48000)
_VALID_FRAME_DURATIONS_MS = (10, 20, 30)


@dataclass
class WebRTCVADState:
    """Complete mutable state for one ordered audio stream."""

    filterbank: FilterbankState = field(default_factory=FilterbankState)
    gmm: GMMState = field(default_factory=GMMState)
    resampler: ResamplerState = field(default_factory=ResamplerState)


class NativeWebRTCVAD:
    """Bit-exact Python port of the pinned WebRTC fixed-point detector.

    Instances are stateful because the reference algorithm adapts its
    GMM and maintains filters and hangover counters between consecutive
    frames. Create one instance per independent stream.
    """

    def __init__(self, aggressiveness: int = 0) -> None:
        self._aggressiveness = 0
        self.state = WebRTCVADState()
        self.set_mode(aggressiveness)

    @property
    def aggressiveness(self) -> int:
        return self._aggressiveness

    def set_mode(self, aggressiveness: int) -> None:
        """Change the official threshold set without resetting stream state."""
        self.state.gmm.set_mode(aggressiveness)
        self._aggressiveness = aggressiveness

    def reset(self) -> None:
        """Reset adaptation, filters, and resamplers while retaining the
        mode."""
        mode = self._aggressiveness
        self.state = WebRTCVADState()
        self.set_mode(mode)

    @staticmethod
    def valid_rate_and_frame_length(
        sample_rate: int,
        frame_length: int,
    ) -> bool:
        if sample_rate not in _VALID_SAMPLE_RATES:
            return False
        return any(frame_length == sample_rate // 1000 * duration for duration in _VALID_FRAME_DURATIONS_MS)

    def is_speech(
        self,
        frame: Sequence[int],
        sample_rate: int,
    ) -> bool:
        """Return the binary decision for one 10, 20, or 30 ms PCM frame."""
        if not self.valid_rate_and_frame_length(sample_rate, len(frame)):
            expected = ", ".join(
                str(sample_rate // 1000 * duration) for duration in _VALID_FRAME_DURATIONS_MS)
            raise ValueError(
                "Invalid WebRTC frame length "
                f"{len(frame)} for {sample_rate} Hz; expected {expected}.", )
        analysis_frame = resample_to_8khz(
            frame,
            sample_rate,
            self.state.resampler,
        )
        features, total_power = calculate_features(
            analysis_frame,
            self.state.filterbank,
        )
        decision = classify(
            features,
            total_power,
            len(analysis_frame),
            self.state.gmm,
        )
        return decision > 0


__all__ = ["NativeWebRTCVAD", "WebRTCVADState"]
