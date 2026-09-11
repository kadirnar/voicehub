import json
import unittest
from pathlib import Path

from voicehub.models.vad_webrtc.native import NativeWebRTCVAD
from voicehub.runtime import ARCHITECTURE_REGISTRY

_REFERENCE_DECISIONS = {
    (8000, 10, 0): "000000000001111111111111000000111111111111110000111111111111",
    (8000, 30, 3): "000000001111111111000000000000111111110000000000001111111100",
    (16000, 20, 2): "000000000001111111100000000011111111111110000000111111111111",
    (32000, 10, 2): "000000000111111111111100000001111111111111111000011111111111",
    (32000, 30, 3): "000000000001111111000000000000011111110000000000000111111100",
    (48000, 10, 0): "000000001111111111111111100000111111111111111111111111111111",
    (48000, 20, 3): "000000000001111111100000000000011111111000000000000111111110",
    (48000, 30, 2): "000000000001111111000000000000111111110000000000001111111100",
}


def _reference_frames(sample_rate, duration_ms, mode, count=60):
    frame_length = sample_rate * duration_ms // 1000
    random_state = (sample_rate << 8) + (duration_ms << 4) + mode
    for frame_index in range(count):
        position = frame_index % 20
        amplitude = (
            0 if position < 8 or position >= 15 else (32, 128, 512, 3000, 12000, 20000, 800)[position - 8])
        frame = []
        for sample_index in range(frame_length):
            random_state = (1103515245 * random_state + 12345) & 0x7FFFFFFF
            if position == 14:
                period = max(2, sample_rate // 180)
                value = (amplitude if sample_index % period < period // 2 else -amplitude)
            else:
                value = (((random_state >> 7) % (2 * amplitude + 1)) - amplitude if amplitude else 0)
            frame.append(value)
        yield frame


class NativeWebRTCVADTests(unittest.TestCase):

    def test_pinned_reference_decisions_cover_every_resampler(self):
        for key, expected in _REFERENCE_DECISIONS.items():
            sample_rate, duration_ms, mode = key
            detector = NativeWebRTCVAD(mode)
            actual = "".join(
                "1" if detector.is_speech(frame, sample_rate) else "0" for frame in _reference_frames(*key))
            with self.subTest(
                    sample_rate=sample_rate,
                    duration_ms=duration_ms,
                    mode=mode,
            ):
                self.assertEqual(actual, expected)

    def test_reset_retains_mode_and_restores_stream_state(self):
        frames = tuple(_reference_frames(16000, 20, 3, count=24))
        detector = NativeWebRTCVAD(3)
        first = tuple(detector.is_speech(frame, 16000) for frame in frames)

        self.assertGreater(detector.state.gmm.frame_counter, 0)
        detector.reset()
        second = tuple(detector.is_speech(frame, 16000) for frame in frames)

        self.assertEqual(detector.aggressiveness, 3)
        self.assertEqual(first, second)

    def test_frame_validation_matches_the_public_contract(self):
        detector = NativeWebRTCVAD()

        for sample_rate in (8000, 16000, 32000, 48000):
            for duration_ms in (10, 20, 30):
                with self.subTest(
                        sample_rate=sample_rate,
                        duration_ms=duration_ms,
                ):
                    self.assertTrue(
                        detector.valid_rate_and_frame_length(
                            sample_rate,
                            sample_rate * duration_ms // 1000,
                        ), )
        self.assertFalse(detector.valid_rate_and_frame_length(44100, 441))
        with self.assertRaisesRegex(ValueError, "Invalid WebRTC frame length"):
            detector.is_speech([0] * 159, 16000)

    def test_catalog_and_provenance_describe_the_algorithmic_boundary(self):
        spec = ARCHITECTURE_REGISTRY.get("webrtc-vad")

        self.assertFalse(spec.capabilities.training)
        self.assertTrue(spec.capabilities.streaming)
        self.assertEqual(spec.capabilities.checkpoint_formats, ("none", ))
        self.assertEqual(spec.metadata["implementation"], "voicehub-native")

        source_path = (Path(__file__).resolve().parents[1] / 'voicehub/models/vad_webrtc/native/SOURCE.json')
        source = json.loads(source_path.read_text(encoding="utf-8"))
        self.assertEqual(
            source["upstream"]["revision"],
            "e283ca41df3a84b0e87fb1f5cb9b21580a286b09",
        )
        self.assertFalse(source["training"]["supported"])


if __name__ == "__main__":
    unittest.main()
