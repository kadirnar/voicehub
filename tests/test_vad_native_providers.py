import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from voicehub.models.audio import PreTrainedVADModel
from voicehub.models.vad_nemo import NeMoVADConfig, NeMoVADForVoiceActivityDetection
from voicehub.models.vad_pyannote import PyannoteVADConfig, PyannoteVADForVoiceActivityDetection
from voicehub.models.vad_speechbrain import SpeechBrainVADConfig, SpeechBrainVADForVoiceActivityDetection


class _FakeTensor:

    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    def unsqueeze(self, dimension):
        return _FakeTensor(np.expand_dims(self.values, axis=dimension))

    def to(self, device):
        del device
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()


def _fake_torch():
    module = ModuleType("torch")
    module.as_tensor = lambda values: _FakeTensor(values)
    return module


class NativeVADProviderTests(unittest.TestCase):

    def test_importing_wrappers_does_not_import_native_providers(self):
        code = """
import json
import sys
from voicehub.models.vad_nemo import NeMoVADForVoiceActivityDetection
from voicehub.models.vad_pyannote import PyannoteVADForVoiceActivityDetection
from voicehub.models.vad_speechbrain import SpeechBrainVADForVoiceActivityDetection
print(json.dumps({
    "nemo": "nemo" in sys.modules,
    "pyannote": "pyannote.audio" in sys.modules,
    "speechbrain": "speechbrain" in sys.modules,
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
                "nemo": False,
                "pyannote": False,
                "speechbrain": False,
            },
        )

    def test_public_classes_are_lazy_task_specific_wrappers(self):
        cases = (
            (
                PyannoteVADConfig,
                PyannoteVADForVoiceActivityDetection,
                "native",
                True,
            ),
            (
                SpeechBrainVADConfig,
                SpeechBrainVADForVoiceActivityDetection,
                "native",
                True,
            ),
            (
                NeMoVADConfig,
                NeMoVADForVoiceActivityDetection,
                "native",
                True,
            ),
        )
        for config_class, model_class, support, generic_finetuning in cases:
            with self.subTest(model=model_class.__name__):
                model = model_class(config_class())
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
                    with self.assertRaisesRegex(
                            ValueError,
                            "upstream-custom",
                    ):
                        model._validate_training_runtime()

    def test_native_vad_runtimes_never_auto_select_mps(self):
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True), )
        model_classes = (
            PyannoteVADForVoiceActivityDetection,
            SpeechBrainVADForVoiceActivityDetection,
            NeMoVADForVoiceActivityDetection,
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            for model_class in model_classes:
                with self.subTest(model=model_class.__name__):
                    self.assertEqual(model_class._resolve_device("auto"), "cpu")
                    with self.assertRaisesRegex(ValueError, "CPU and CUDA"):
                        model_class._resolve_device("mps")

    def test_provider_configs_refuse_to_serialize_credentials(self):
        cases = (
            (PyannoteVADConfig, {
                "pipeline_kwargs": {
                    "token": "secret"
                }
            }),
            (SpeechBrainVADConfig, {
                "loader_kwargs": {
                    "fetch_config": object()
                }
            }),
            (NeMoVADConfig, {
                "model_kwargs": {
                    "hub_kwargs": {
                        "use_auth_token": "secret"
                    }
                }
            }),
        )
        for config_class, kwargs in cases:
            with self.subTest(config=config_class.__name__):
                with self.assertRaisesRegex(ValueError, "token|auth"):
                    config_class(**kwargs)

        model = PyannoteVADForVoiceActivityDetection(
            PyannoteVADConfig(),
            token="runtime-secret",
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = model.config.save_pretrained(directory)
            serialized = Path(config_path).read_text(encoding="utf-8")
        self.assertNotIn("runtime-secret", serialized)
        self.assertNotIn("token", json.loads(serialized))

    def test_segment_splitters_do_not_emit_floating_point_slivers(self):
        from voicehub.models.vad_pyannote.modeling import _finalize_segments as finalize_pyannote
        from voicehub.models.vad_speechbrain.modeling import _finalize_segments as finalize_speechbrain

        values = ({"start": 0.0, "end": 0.30000000000000004}, )
        for finalize in (finalize_pyannote, finalize_speechbrain):
            with self.subTest(finalize=finalize.__module__):
                segments = finalize(
                    values,
                    duration=1.0,
                    sample_rate=16_000,
                    speech_pad_ms=0,
                    max_speech_duration_s=0.1,
                )
                self.assertEqual(
                    [(segment.start, segment.end) for segment in segments],
                    [(0.0, 0.1), (0.1, 0.2), (0.2, 0.30000000000000004)],
                )

    def test_pyannote_official_pickle_requires_explicit_conversion(self):
        model = PyannoteVADForVoiceActivityDetection(
            PyannoteVADConfig(),
            device="cpu",
        )
        with self.assertRaisesRegex(
                ValueError,
                "Lightning pickle checkpoint",
        ):
            model._load_pretrained_model()

    def test_speechbrain_official_pickle_requires_explicit_conversion(self):
        model = SpeechBrainVADForVoiceActivityDetection(
            SpeechBrainVADConfig(),
            device="cpu",
        )
        with self.assertRaisesRegex(
                ValueError,
                "trust_pickle_checkpoint",
        ):
            model._load_pretrained_model()

    def test_nemo_accepts_only_the_verified_multilingual_frame_graph(self):
        with self.assertRaisesRegex(ValueError, "Frame-VAD"):
            NeMoVADConfig(architecture_family="window")
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            NeMoVADConfig(sample_rate=10)


if __name__ == "__main__":
    unittest.main()
