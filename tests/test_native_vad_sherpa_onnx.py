"""Regression tests for the Sherpa-compatible native VAD segmentation.

Expected values follow the pinned sherpa-onnx C++ Silero model and
VoiceActivityDetector sources at commit
``d1fedc1daac9304cd8f85350a38aed2e5e120f02``.
"""

import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError:  # pragma: no cover - torch is a runtime dependency
    torch = None


def _silero_model():
    from voicehub.architectures.silero_vad.configuration import SileroVADConfig
    from voicehub.architectures.silero_vad.modeling import SileroVADModel
    from voicehub.models.vad_sherpa_onnx import SherpaONNXVADConfig, SherpaONNXVADForVoiceActivityDetection
    from voicehub.models.vad_sherpa_onnx.modeling_vad_sherpa_onnx import _NativeVADRuntime

    class RandomSileroVAD(SherpaONNXVADForVoiceActivityDetection):
        """Randomly initialized native graph; no checkpoint download."""

        def _load_pretrained_model(self):
            torch.manual_seed(11)
            self.native_config = SileroVADConfig(sampling_rate=16_000)
            self.model = SileroVADModel(self.native_config).eval()
            self.runtime = _NativeVADRuntime(
                architecture="silero-vad",
                checkpoint=Path("random-init"),
                checkpoint_format="random",
                checkpoint_adapter="random",
                revision=None,
            )

    return RandomSileroVAD(SherpaONNXVADConfig(), device="cpu")


def _scripted_probabilities(values):
    """Patch the Silero scorer to return ``values`` window by window."""
    iterator = iter(values)
    return mock.patch(
        "voicehub.models.vad_sherpa_onnx.streaming.NativeSileroScorer.compute",
        autospec=True,
        side_effect=lambda self, samples: next(iterator),
    )


@unittest.skipUnless(torch is not None, "Native Sherpa-compatible VAD uses PyTorch")
class SherpaDecisionArithmeticTests(unittest.TestCase):

    def test_durations_use_sherpa_float32_truncation(self):
        from voicehub.models.vad_sherpa_onnx.streaming import _duration_samples, _SpeechDecision

        # int32_t(16000 * 0.251f) == 4015 in C++; double arithmetic gives 4016.
        self.assertEqual(_duration_samples(16_000, 0.251), 4015)
        self.assertEqual(_duration_samples(16_000, 1.001), 16016)
        self.assertEqual(_duration_samples(16_000, 16.3), 260799)
        self.assertEqual(_duration_samples(16_000, 0.1), 1600)
        decision = _SpeechDecision(
            family="silero",
            sample_rate=16_000,
            window_shift=512,
            threshold=0.5,
            negative_threshold=None,
            min_speech_duration=0.251,
            min_silence_duration=0.253,
        )
        self.assertEqual(decision.min_speech_samples, 4015)
        self.assertEqual(decision.min_silence_samples, 4047)

    def test_thresholds_compare_in_float32(self):
        from voicehub.models.vad_sherpa_onnx.streaming import _float32, _SpeechDecision

        decision = _SpeechDecision(
            family="silero",
            sample_rate=16_000,
            window_shift=512,
            threshold=0.3,
            negative_threshold=None,
            min_speech_duration=0.0,
            min_silence_duration=0.1,
        )
        # A probability equal to float32(0.3) is not `> 0.3f` upstream.
        self.assertFalse(decision.update(_float32(0.3)))
        self.assertEqual(decision.temp_start, 0)
        self.assertFalse(decision.update(0.31))
        self.assertTrue(decision.update(0.31))


@unittest.skipUnless(torch is not None, "Native Sherpa-compatible VAD uses PyTorch")
class SherpaSessionParityTests(unittest.TestCase):

    def test_partial_tail_is_not_zero_padded_into_an_extra_window(self):
        model = _silero_model()
        waveform = torch.randn(3 * 512 + 100, generator=torch.Generator().manual_seed(3))
        output = model.detect(waveform, sampling_rate=16_000, return_frames=True, speech_pad_ms=0)
        # Sherpa scores [0:576] and [512:1088]; the remaining 64 + 100
        # samples never form a complete 576-sample window.
        self.assertEqual(len(output.probabilities), 2)
        scorer_model = model.model
        state = scorer_model.initial_state(1)
        expected = []
        with torch.inference_mode():
            recurrent = (state.hidden, state.cell)
            for start in (0, 512):
                probability, _, recurrent = scorer_model.forward_with_context(
                    waveform[start:start + 576].unsqueeze(0), recurrent)
                expected.append(float(probability.item()))
        self.assertEqual(list(output.probabilities), expected)

    def test_max_speech_duration_only_raises_the_threshold(self):
        model = _silero_model()
        waveform = torch.zeros(80_000)
        with _scripted_probabilities([0.99] * 200):
            output = model.detect(
                waveform,
                sampling_rate=16_000,
                speech_pad_ms=0,
                max_speech_duration_s=1.0,
                return_frames=True,
            )
        # 156 complete shifts are fed, 155 windows scored. Sherpa never
        # splits a segment; at 0.99 > 0.9 the speech continues to Flush(),
        # which ends at the buffer tail (155 * 512 samples).
        self.assertEqual(len(output.probabilities), 155)
        self.assertEqual([(s.start, s.end) for s in output.segments], [(0.0, 155 * 512 / 16_000)])

    def test_touching_sherpa_segments_are_not_merged_without_padding(self):
        model = _silero_model()
        waveform = torch.zeros(20 * 512)
        probabilities = [0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9] + [0.1] * 20
        options = dict(
            sampling_rate=16_000,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            max_speech_duration_s=20.0,
        )
        with _scripted_probabilities(probabilities):
            output = model.detect(waveform, speech_pad_ms=0, **options)
        # Upstream: the first segment ends at the first silent window
        # (buffer tail 2048) and the buffer is popped to 2048. Speech
        # retriggers at window 5 (tail 3072): start = max(3072 - 2 * 576,
        # head) = 2048, so the segments touch; the second ends at the next
        # silent window (tail 4608).
        self.assertEqual(
            [(s.metadata["start_sample"], s.metadata["num_samples"]) for s in output.segments],
            [(0, 2048), (2048, 2560)],
        )
        with _scripted_probabilities(probabilities):
            padded = model.detect(waveform, speech_pad_ms=10, **options)
        self.assertEqual(len(padded.segments), 1)


if __name__ == "__main__":
    unittest.main()
