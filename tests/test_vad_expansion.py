import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from voicehub.models.audio import PreTrainedVADModel
from voicehub.models.vad_auditok import AuditokVADConfig, AuditokVADForVoiceActivityDetection
from voicehub.models.vad_pyannote_brouhaha import (
    PyannoteBrouhahaVADConfig,
    PyannoteBrouhahaVADForVoiceActivityDetection,
)
from voicehub.models.vad_pyannote_segmentation import (
    PyannoteSegmentationVADConfig,
    PyannoteSegmentationVADForVoiceActivityDetection,
)
from voicehub.models.vad_sherpa_onnx import (
    SherpaONNXVADConfig,
    SherpaONNXVADForVoiceActivityDetection,
    SherpaONNXVADSession,
)


class VADExpansionConfigTests(unittest.TestCase):

    def test_wrappers_are_lazy_and_do_not_import_provider_runtimes(self):
        code = """
import json
import sys
from voicehub.models.vad_auditok import AuditokVADForVoiceActivityDetection
from voicehub.models.vad_pyannote_brouhaha import PyannoteBrouhahaVADForVoiceActivityDetection
from voicehub.models.vad_pyannote_segmentation import PyannoteSegmentationVADForVoiceActivityDetection
from voicehub.models.vad_sherpa_onnx import SherpaONNXVADForVoiceActivityDetection
print(json.dumps({
    "auditok": "auditok" in sys.modules,
    "pyannote": "pyannote.audio" in sys.modules,
    "sherpa": "sherpa_onnx" in sys.modules,
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(
            json.loads(completed.stdout.strip().splitlines()[-1]),
            {
                "auditok": False,
                "pyannote": False,
                "sherpa": False,
            },
        )

    def test_public_wrappers_follow_the_vad_lifecycle(self):
        cases = (
            (
                AuditokVADForVoiceActivityDetection,
                AuditokVADConfig(),
                "inference-only",
                False,
            ),
            (
                SherpaONNXVADForVoiceActivityDetection,
                SherpaONNXVADConfig(),
                "native",
                True,
            ),
            (
                PyannoteSegmentationVADForVoiceActivityDetection,
                PyannoteSegmentationVADConfig(),
                "native",
                True,
            ),
            (
                PyannoteBrouhahaVADForVoiceActivityDetection,
                PyannoteBrouhahaVADConfig(),
                "native",
                True,
            ),
        )
        for model_class, config, support, generic_finetuning in cases:
            with self.subTest(model=model_class.__name__):
                model = model_class(config)
                self.assertIsInstance(model, PreTrainedVADModel)
                self.assertIsNone(model.model)
                self.assertEqual(model.training_support, support)
                self.assertEqual(
                    model.supports_generic_finetuning,
                    generic_finetuning,
                )
                if generic_finetuning:
                    model._validate_training_runtime()
                else:
                    with self.assertRaises(ValueError):
                        model._validate_training_runtime()

    def test_auditok_calibration_config_is_validated_and_serializable(self):
        config = AuditokVADConfig(
            threshold_method="p20",
            energy_threshold_db=47.5,
            analysis_window_s=0.02,
            calibration_duration_s=2.5,
            minimum_energy_threshold_db=35,
            strict_min_duration=True,
        )
        restored = AuditokVADConfig.from_dict(config.to_dict())
        self.assertEqual(restored.threshold_method, "p20")
        self.assertEqual(restored.analysis_window_s, 0.02)
        self.assertTrue(restored.strict_min_duration)
        for kwargs in (
            {"threshold_method": "p0"},
            {"threshold_method": "median"},
            {"analysis_window_s": 0.2},
            {"calibration_duration_s": 0},
            {"inference_config": {"token": "secret"}},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    AuditokVADConfig(**kwargs)

    def test_sherpa_config_rejects_unsafe_assets_and_credentials(self):
        config = SherpaONNXVADConfig(
            model_family="ten",
            subfolder="onnx/vad",
            local_files_only=True,
            provider="cpu",
        )
        self.assertEqual(config.model_filename, "ten-vad.onnx")
        self.assertEqual(config.window_size_samples, 256)
        self.assertEqual(config.subfolder, "onnx/vad")
        self.assertEqual(
            SherpaONNXVADForVoiceActivityDetection(config).device,
            "cpu",
        )
        for kwargs in (
            {"model_filename": "../model.onnx"},
            {"model_filename": "/tmp/model.onnx"},
            {"model_filename": "model.pt"},
            {"subfolder": "../weights"},
            {"model_family": "unknown"},
            {"provider": "coreml"},
            {"provider": "tensorrt"},
            {"kwargs": {"token": "secret"}},
        ):
            with self.subTest(kwargs=kwargs):
                if "kwargs" in kwargs:
                    kwargs = kwargs["kwargs"]
                with self.assertRaises((TypeError, ValueError)):
                    SherpaONNXVADConfig(**kwargs)

    def test_runtime_tokens_are_not_persisted(self):
        model = SherpaONNXVADForVoiceActivityDetection(
            SherpaONNXVADConfig(),
            token="runtime-only",
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = model.config.save_pretrained(directory)
            serialized = Path(config_path).read_text(encoding="utf-8")
        self.assertNotIn("runtime-only", serialized)
        self.assertNotIn("token", json.loads(serialized))


class AuditokVADRuntimeTests(unittest.TestCase):

    def test_energy_detection_is_native_and_normalizes_seconds(self):
        waveform = np.zeros(16_000, dtype=np.float32)
        waveform[1_600:6_400] = 0.1
        waveform[11_200:] = 0.1
        model = AuditokVADForVoiceActivityDetection(
            AuditokVADConfig(
                threshold_method="otsu",
                analysis_window_s=0.02,
                calibration_duration_s=2,
                minimum_energy_threshold_db=38,
            ))
        output = model.detect(
            waveform,
            sampling_rate=16_000,
            min_speech_duration_ms=0,
            min_silence_duration_ms=120,
            speech_pad_ms=20,
        )

        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.08, 0.42), (0.68, 1.0)],
        )
        self.assertEqual(output.metadata["backend"], "voicehub-native")
        self.assertEqual(output.metadata["threshold_method"], "otsu")
        self.assertGreater(
            output.metadata["resolved_energy_threshold_db"],
            60,
        )
        self.assertIsNone(output.probabilities)

    def test_probability_options_are_rejected_instead_of_misrepresented(self):
        model = AuditokVADForVoiceActivityDetection(AuditokVADConfig())
        model.model = SimpleNamespace(split=lambda *args, **kwargs: ())
        for kwargs in (
            {"threshold": 0.8},
            {"onset": 0.6},
            {"offset": 0.4},
            {"return_frames": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "calibrated speech probabilities"):
                    model._detect(
                        np.zeros(160, dtype=np.float32),
                        sampling_rate=16_000,
                        **kwargs,
                    )


def _native_ten_artifact(directory: str | Path) -> Path:
    import torch

    from voicehub.checkpointing import save_safetensors
    from voicehub.models.vad_ten.native.checkpoint import NATIVE_TEN_VAD_FILENAME, NATIVE_TEN_VAD_FORMAT
    from voicehub.models.vad_ten.native.configuration import TENVADConfig
    from voicehub.models.vad_ten.native.modeling import TENVADModel

    destination = Path(directory)
    config = TENVADConfig()
    native = TENVADModel(config)
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.zero_()
        native.output.bias.fill_(20)
    save_safetensors(
        native.state_dict(),
        destination / NATIVE_TEN_VAD_FILENAME,
        metadata={
            "format": NATIVE_TEN_VAD_FORMAT,
            "architecture": "ten-vad",
        },
    )
    (destination / "config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    return destination


class SherpaONNXVADRuntimeTests(unittest.TestCase):

    def test_silero_scorer_preserves_native_frame_and_state_contract(self):
        import torch

        from voicehub.models.vad_sherpa_onnx.streaming import NativeSileroScorer
        from voicehub.models.vad_silero.native.configuration import SileroVADConfig
        from voicehub.models.vad_silero.native.modeling import SileroVADModel

        torch.manual_seed(29)
        native = SileroVADModel(SileroVADConfig()).eval()
        waveform = torch.randn(native.config.frame_size * 3)
        expected = native.frame_probabilities(
            waveform.unsqueeze(0),
            pad_final_frame=False,
        ).probabilities[0]

        scorer = NativeSileroScorer(native)
        actual = torch.tensor([scorer.compute(frame) for frame in waveform.split(native.config.frame_size)])

        self.assertEqual(scorer.window_size, native.config.frame_size)
        torch.testing.assert_close(actual, expected)

    @staticmethod
    def _model(directory: str | Path):
        return SherpaONNXVADForVoiceActivityDetection(
            SherpaONNXVADConfig(
                name_or_path=_native_ten_artifact(directory),
                model_family="ten",
                model_filename="model.safetensors",
            ),
            lazy_load=False,
        )

    def test_streaming_session_is_incremental_idempotent_and_resettable(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self._model(directory)
            session = model.stream(
                sampling_rate=16_000,
                speech_pad_ms=0,
                min_speech_duration_ms=0,
                return_frames=True,
            )
            self.assertIsInstance(session, SherpaONNXVADSession)
            self.assertEqual(session.push(np.zeros(1_024, dtype=np.float32)), ())
            first = session.flush()
            second = session.flush()
            self.assertIs(first, second)
            self.assertAlmostEqual(first.duration, 0.064)
            self.assertEqual(len(first.probabilities), 4)
            self.assertGreater(len(first.segments), 0)
            self.assertTrue(all(segment.end <= first.duration for segment in first.segments))

            session.reset()
            session.push(np.zeros(512, dtype=np.float32))
            self.assertGreater(len(session.flush().segments), 0)
            session.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                session.push(np.zeros(256, dtype=np.float32))

    def test_ten_vad_rejects_an_unsupported_offset_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self._model(directory)
            with self.assertRaisesRegex(ValueError, "TEN VAD"):
                model._detect(
                    np.zeros(512, dtype=np.float32),
                    sampling_rate=16_000,
                    offset=0.3,
                )

    def test_direct_local_onnx_requires_explicit_review(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "custom-vad.onnx"
            model_path.touch()
            model = SherpaONNXVADForVoiceActivityDetection(
                SherpaONNXVADConfig(
                    name_or_path=model_path,
                    model_family="ten",
                ), )
            with self.assertRaisesRegex(ValueError, "Review the TEN artifact"):
                model._load_pretrained_model()

    def test_streaming_validates_common_inference_values_and_closed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self._model(directory)
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                model.stream(sampling_rate=16_000, threshold=2)
            session = model.stream(sampling_rate=16_000)
            session.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                session.flush()


class PyannotePresetRuntimeTests(unittest.TestCase):

    def test_presets_protect_their_managed_runtime_options(self):
        with self.assertRaisesRegex(ValueError, "managed segmentation"):
            PyannoteSegmentationVADConfig(pipeline_kwargs={"segmentation": "other/model"})
        with self.assertRaisesRegex(ValueError, "not supported"):
            PyannoteBrouhahaVADConfig(pipeline_kwargs={"batch_size": 1})

    def test_official_presets_require_the_explicit_conversion_boundary(self):
        cases = (
            (
                PyannoteSegmentationVADForVoiceActivityDetection,
                PyannoteSegmentationVADConfig(),
            ),
            (
                PyannoteBrouhahaVADForVoiceActivityDetection,
                PyannoteBrouhahaVADConfig(),
            ),
        )
        for model_class, config in cases:
            with self.subTest(model=model_class.__name__):
                model = model_class(config, device="cpu")
                with self.assertRaisesRegex(
                        ValueError,
                        "Lightning pickle checkpoint",
                ):
                    model._load_pretrained_model()


if __name__ == "__main__":
    unittest.main()
