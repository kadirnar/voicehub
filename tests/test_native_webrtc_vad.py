import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from voicehub.architectures import ARCHITECTURE_REGISTRY
from voicehub.architectures.webrtc_vad import NativeWebRTCVAD

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

    def test_compiler_unavailability_preserves_the_python_fallback(self):
        from voicehub.architectures.webrtc_vad.acceleration import get_accelerator

        get_accelerator.cache_clear()
        try:
            with patch("voicehub.architectures.webrtc_vad.acceleration.shutil.which", return_value=None):
                accelerator, status = get_accelerator()
            self.assertIsNone(accelerator)
            self.assertTrue(status.startswith("python:"))
        finally:
            get_accelerator.cache_clear()

    @unittest.skipUnless(
        sys.platform in {"linux", "darwin"} and shutil.which("cc") and shutil.which("c++"),
        "Native WebRTC compilation requires POSIX C/C++ compilers")
    def test_compiled_batches_match_stateful_python_decisions_and_tail_padding(self):
        import torch

        from voicehub.architectures.webrtc_vad.acceleration import get_accelerator

        accelerator, status = get_accelerator()
        self.assertIsNotNone(accelerator, status)
        for rate in (8000, 16000, 32000, 48000):
            for duration in (10, 20, 30):
                for mode in range(4):
                    frames = list(_reference_frames(rate, duration, mode, count=40))
                    tail = frames[-1][:len(frames[-1]) // 3]
                    values = [value for frame in frames for value in frame] + tail
                    detector = NativeWebRTCVAD(mode)
                    expected = [int(detector.is_speech(frame, rate)) for frame in frames]
                    expected.append(int(detector.is_speech(tail + [0] * (len(frames[0]) - len(tail)), rate)))
                    pcm = torch.tensor(values, dtype=torch.int16)
                    with self.subTest(rate=rate, duration=duration, mode=mode):
                        actual = accelerator(pcm, rate, len(frames[0]), mode)
                        self.assertEqual(actual, expected)
                        # Separate calls must start fresh rather than retain
                        # GMM adaptation from the previous request.
                        self.assertEqual(accelerator(pcm, rate, len(frames[0]), mode), expected)

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

        source_path = (
            Path(__file__).resolve().parents[1] / "voicehub" / "architectures" / "webrtc_vad" / "SOURCE.json")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        self.assertEqual(
            source["upstream"]["revision"],
            "e283ca41df3a84b0e87fb1f5cb9b21580a286b09",
        )
        self.assertFalse(source["training"]["supported"])


if __name__ == "__main__":
    unittest.main()
