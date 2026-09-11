import json
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from voicehub.models.vad_silero import SileroVADConfig, SileroVADForVoiceActivityDetection
from voicehub.models.vad_transformers import TransformersVADConfig, TransformersVADForVoiceActivityDetection
from voicehub.models.vad_webrtc import WebRTCVADConfig, WebRTCVADForVoiceActivityDetection
from voicehub.models.vad_webrtc.modeling import _pcm16_samples

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _temporary_modules(modules):
    missing = object()
    originals = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class VADProviderContractTests(unittest.TestCase):

    def test_importing_wrappers_keeps_all_runtime_backends_lazy(self):
        code = """
import json
import sys
from voicehub.models.vad_silero import SileroVADForVoiceActivityDetection
from voicehub.models.vad_transformers import TransformersVADForVoiceActivityDetection
from voicehub.models.vad_webrtc import WebRTCVADForVoiceActivityDetection
optional = ("silero_vad", "torch", "transformers", "webrtcvad")
print(json.dumps({name: name in sys.modules for name in optional}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(result.stdout.strip().splitlines()[-1]),
            {
                "silero_vad": False,
                "torch": False,
                "transformers": False,
                "webrtcvad": False,
            },
        )

    def test_provider_configurations_reject_unsupported_audio_contracts(self):
        invalid = (
            lambda: SileroVADConfig(sample_rate=44_100),
            lambda: SileroVADConfig(use_onnx=1),
            lambda: WebRTCVADConfig(sample_rate=44_100),
            lambda: WebRTCVADConfig(aggressiveness=4),
            lambda: WebRTCVADConfig(frame_duration_ms=15),
            lambda: TransformersVADConfig(architecture_family="token-classification"),
            lambda: TransformersVADConfig(speech_labels=()),
            lambda: TransformersVADConfig(speech_labels="speech"),
            lambda: TransformersVADConfig(window_duration_s=True),
            lambda: TransformersVADConfig(hop_duration_s=float("nan")),
            lambda: TransformersVADConfig(window_duration_s=0.5, hop_duration_s=0.6),
            lambda: TransformersVADConfig(
                processor_kwargs={
                    "nested": {
                        "use_auth_token": "must-not-be-persisted",
                    },
                }),
        )
        for factory in invalid:
            with self.subTest(factory=factory), self.assertRaises((TypeError, ValueError)):
                factory()

    def test_native_vad_device_contracts_are_provider_specific(self):
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True), )
        with _temporary_modules({"torch": fake_torch}):
            self.assertEqual(
                SileroVADForVoiceActivityDetection._resolve_device("auto"),
                "cpu",
            )
            self.assertEqual(
                WebRTCVADForVoiceActivityDetection._resolve_device("auto"),
                "cpu",
            )
        with self.assertRaisesRegex(ValueError, "CPU and CUDA"):
            SileroVADForVoiceActivityDetection._resolve_device("mps")
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            WebRTCVADForVoiceActivityDetection._resolve_device("mps")

    def test_fine_tuning_boundaries_match_backend_capabilities(self):
        silero = SileroVADForVoiceActivityDetection(SileroVADConfig())
        self.assertIsNone(silero._validate_training_runtime())
        self.assertEqual(
            type(silero.get_training_adapter()).__name__,
            "NativeSileroVADTrainingAdapter",
        )

        webrtc = WebRTCVADForVoiceActivityDetection(WebRTCVADConfig())
        with self.assertRaisesRegex(ValueError, "not applicable"):
            webrtc._validate_training_runtime()

        trainable = TransformersVADForVoiceActivityDetection(TransformersVADConfig())
        self.assertIsNone(trainable._validate_training_runtime())

    def test_native_silero_rejects_removed_external_runtime_modes(self):
        with self.assertRaisesRegex(ValueError, "does not execute ONNX"):
            SileroVADConfig(use_onnx=True)
        with self.assertRaisesRegex(ValueError, "immutable, verified"):
            SileroVADConfig(force_reload=True)


class SileroVADInferenceTests(unittest.TestCase):

    def test_native_silero_resolves_auto_device_before_model_placement(self):
        captured = {}

        class Runtime:

            def __init__(self, config):
                captured["config"] = config

            def to(self, *, device):
                captured["placed_on"] = device
                return self

        model = SileroVADForVoiceActivityDetection(
            SileroVADConfig(),
            device="auto",
        )
        artifact = SimpleNamespace(checkpoint_format="safetensors")
        with (
                patch.object(
                    SileroVADForVoiceActivityDetection,
                    "_resolve_device",
                    return_value="cpu",
                ) as resolve_device,
                patch(
                    "voicehub.models.vad_silero.artifacts."
                    "resolve_silero_vad_artifact",
                    return_value=artifact,
                ),
                patch(
                    "voicehub.models.vad_silero.artifacts."
                    "load_silero_vad_checkpoint",
                    return_value=("safetensors", "voicehub-native-silero-vad"),
                ),
                patch(
                    "voicehub.models.vad_silero.native.modeling.SileroVADModel",
                    Runtime,
                ),
        ):
            model._load_pretrained_model()

        resolve_device.assert_called_once_with("auto")
        self.assertEqual(model.device, "cpu")
        self.assertEqual(captured["placed_on"], "cpu")
        self.assertIs(model.artifact, artifact)

    def test_native_silero_maps_controls_and_returns_frame_scores(self):
        import torch

        from voicehub.models.vad_silero.native.configuration import SileroVADConfig as NativeSileroVADConfig

        captured = {}

        class Runtime:

            def frame_probabilities(self, waveform, *, pad_final_frame):
                captured["waveform"] = waveform
                captured["pad_final_frame"] = pad_final_frame
                return SimpleNamespace(
                    probabilities=torch.tensor([[0.1, 0.9, 0.9]]),
                    valid_samples=waveform.shape[-1],
                )

        model = SileroVADForVoiceActivityDetection(
            SileroVADConfig(),
            device="cpu",
        )
        model.native_config = NativeSileroVADConfig()
        model.model = Runtime()
        model.checkpoint_format = "safetensors"
        output = model._detect(
            np.zeros(1_300, dtype=np.float32),
            sampling_rate=16_000,
            onset=0.65,
            offset=0.35,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
            max_speech_duration_s=4.0,
            window_size_samples=512,
            return_frames=True,
        )

        self.assertEqual(captured["waveform"].shape, (1, 1_300))
        self.assertTrue(captured["pad_final_frame"])
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(512 / 16_000, 1_300 / 16_000)],
        )
        self.assertEqual(output.duration, 1_300 / 16_000)
        np.testing.assert_allclose(output.probabilities, (0.1, 0.9, 0.9))
        self.assertTrue(output.metadata["frame_scores_available"])
        self.assertEqual(output.metadata["backend"], "voicehub-native")

    def test_native_silero_frame_scores_are_direct_model_probabilities(self):
        import torch

        from voicehub.models.vad_silero.native.configuration import SileroVADConfig as NativeSileroVADConfig

        model = SileroVADForVoiceActivityDetection(
            SileroVADConfig(),
            device="cpu",
        )
        model.native_config = NativeSileroVADConfig()
        segmentation = model._segmentation_config(
            threshold=0.5,
            onset=None,
            offset=None,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
            max_speech_duration_s=None,
            window_size_samples=512,
        )
        output = model._probabilities_to_output(
            torch.tensor([0.2, 0.8]),
            valid_samples=1_024,
            segmentation_config=segmentation,
            return_frames=True,
            streaming=False,
        )

        np.testing.assert_allclose(output.probabilities, (0.2, 0.8))
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(512 / 16_000, 1_024 / 16_000)],
        )


class WebRTCVADInferenceTests(unittest.TestCase):

    def test_webrtc_vectorized_pcm_conversion_matches_scalar_reference(self):
        values = np.array(
            [
                -np.inf,
                -2.0,
                -1.0,
                -2.5 / 32767,
                -1.5 / 32767,
                -0.5 / 32767,
                0.0,
                0.5 / 32767,
                1.5 / 32767,
                2.5 / 32767,
                1.0,
                2.0,
                np.inf,
                np.nan,
            ],
            dtype=np.float64,
        )
        expected = []
        for value in values:
            normalized = float(value)
            if not np.isfinite(normalized):
                normalized = 0.0
            normalized = max(-1.0, min(1.0, normalized))
            expected.append(round(normalized * 32767.0))

        self.assertEqual(
            _pcm16_samples(torch.from_numpy(values)),
            expected,
        )

    def test_webrtc_frames_pcm_and_uses_request_local_native_runtime(self):
        instances = []
        decisions = (False, True, True, False)

        class NativeRuntime:

            def __init__(self, aggressiveness):
                self.aggressiveness = aggressiveness
                self.calls = []
                instances.append(self)

            def is_speech(self, frame, sample_rate):
                self.calls.append((frame, sample_rate))
                return decisions[len(self.calls) - 1]

        with patch(
                "voicehub.models.vad_webrtc.modeling."
                "NativeWebRTCVAD",
                NativeRuntime,
        ):
            model = WebRTCVADForVoiceActivityDetection(
                WebRTCVADConfig(
                    aggressiveness=3,
                    frame_duration_ms=10,
                ),
                device="cpu",
            )
            output = model.detect(
                np.linspace(-2.0, 2.0, 640, dtype=np.float32),
                sampling_rate=16_000,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
            )

        self.assertEqual(len(instances), 2)
        self.assertIs(model.model, instances[0])
        self.assertIsNot(instances[0], instances[1])
        self.assertEqual(instances[1].aggressiveness, 3)
        self.assertEqual(len(instances[1].calls), 4)
        self.assertTrue(all(len(frame) == 160 for frame, _ in instances[1].calls))
        self.assertTrue(all(rate == 16_000 for _, rate in instances[1].calls))
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.01, 0.03)],
        )
        self.assertIsNone(output.segments[0].score)
        self.assertIsNone(output.probabilities)
        self.assertFalse(output.metadata["frame_scores_available"])

    def test_webrtc_rejects_controls_its_binary_runtime_cannot_honor(self):
        model = WebRTCVADForVoiceActivityDetection(
            WebRTCVADConfig(),
            device="cpu",
        )

        for option in (
            {"threshold": 0.75},
            {"onset": 0.6},
            {"offset": 0.4},
            {"return_frames": True},
        ):
            with (
                    self.subTest(option=option),
                    self.assertRaisesRegex(ValueError, next(iter(option))),
            ):
                model._detect(
                    np.zeros(160, dtype=np.float32),
                    sampling_rate=16_000,
                    **option,
                )

    def test_webrtc_rejects_non_native_window_size(self):
        model = WebRTCVADForVoiceActivityDetection(
            WebRTCVADConfig(frame_duration_ms=20),
            device="cpu",
        )

        model._load_pretrained_model()
        with self.assertRaisesRegex(ValueError, "expected 320"):
            model._detect(
                np.zeros(320, dtype=np.float32),
                sampling_rate=16_000,
                window_size_samples=160,
            )


if __name__ == "__main__":
    unittest.main()
