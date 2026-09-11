import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from voicehub.models.asr_native._shared import normalize_asr_result
from voicehub.models.asr_native.configuration import (
    ESPnetASRConfig,
    FasterWhisperConfig,
    FunASRConfig,
    NeMoASRConfig,
    OpenAIWhisperConfig,
    SpeechBrainASRConfig,
    WeNetASRConfig,
    WhisperXConfig,
)
from voicehub.models.asr_native.espnet import ESPnetASRForSpeechRecognition
from voicehub.models.asr_native.faster_whisper import FasterWhisperForSpeechRecognition
from voicehub.models.asr_native.funasr import FunASRForSpeechRecognition
from voicehub.models.asr_native.nemo import NeMoASRForSpeechRecognition
from voicehub.models.asr_native.openai_whisper import OpenAIWhisperForSpeechRecognition
from voicehub.models.asr_native.speechbrain import SpeechBrainASRForSpeechRecognition
from voicehub.models.asr_native.wenet import WeNetASRForSpeechRecognition
from voicehub.models.asr_native.whisperx import WhisperXForSpeechRecognition
from voicehub.models.asr_whisper_native import NativeWhisperTrainingAdapter, WhisperForSpeechRecognition
from voicehub.outputs import ASROutput
from voicehub.registry import get_model_spec

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


class _InferenceRuntime:

    def __init__(self):
        self.training = True
        self.eval_calls = 0

    def eval(self):
        self.training = False
        self.eval_calls += 1
        return self

    def train(self):
        self.training = True
        return self


class NativeASRProviderContractTests(unittest.TestCase):

    def test_importing_wrappers_does_not_import_provider_packages(self):
        code = """
import json
import sys
from voicehub.models.asr_native.espnet import ESPnetASRForSpeechRecognition
from voicehub.models.asr_native.faster_whisper import FasterWhisperForSpeechRecognition
from voicehub.models.asr_native.funasr import FunASRForSpeechRecognition
from voicehub.models.asr_native.nemo import NeMoASRForSpeechRecognition
from voicehub.models.asr_native.openai_whisper import OpenAIWhisperForSpeechRecognition
from voicehub.models.asr_native.speechbrain import SpeechBrainASRForSpeechRecognition
from voicehub.models.asr_native.wenet import WeNetASRForSpeechRecognition
from voicehub.models.asr_native.whisperx import WhisperXForSpeechRecognition
providers = (
    "espnet2",
    "faster_whisper",
    "funasr",
    "nemo",
    "speechbrain",
    "wenet",
    "whisper",
    "whisperx",
)
print(json.dumps({name: name in sys.modules for name in providers}))
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
                "espnet2": False,
                "faster_whisper": False,
                "funasr": False,
                "nemo": False,
                "speechbrain": False,
                "wenet": False,
                "whisper": False,
                "whisperx": False,
            },
        )

    def test_normalizer_preserves_segment_word_and_provider_metadata(self):
        result = {
            "text":
            " hello world ",
            "language":
            "en",
            "segments": [
                {
                    "id":
                    7,
                    "start":
                    100,
                    "end":
                    700,
                    "text":
                    "hello world",
                    "confidence":
                    0.8,
                    "words": [
                        {
                            "word": "hello",
                            "start": 100,
                            "end": 300,
                            "probability": 0.9,
                            "speaker": "SPEAKER_00",
                        },
                        {
                            "word": "world",
                            "start": 400,
                            "end": 700,
                            "score": 0.75,
                        },
                    ],
                },
            ],
        }

        output = normalize_asr_result(
            result,
            backend="test-provider",
            duration=1.0,
            timestamp_scale=0.001,
        )

        self.assertEqual(output.text, "hello world")
        self.assertEqual(output.metadata["backend"], "test-provider")
        self.assertEqual(output.segments[0].start, 0.1)
        self.assertEqual(output.segments[0].end, 0.7)
        self.assertEqual(output.segments[0].metadata["id"], 7)
        self.assertEqual(output.segments[0].words[0].speaker, "SPEAKER_00")
        self.assertEqual(output.segments[0].words[1].confidence, 0.75)

    def test_inference_only_and_upstream_training_boundaries_are_explicit(self):
        funasr = FunASRForSpeechRecognition(FunASRConfig())
        self.assertIsNone(funasr._validate_training_runtime())
        self.assertEqual(funasr.training_support, "native")
        self.assertTrue(funasr.supports_generic_finetuning)
        self.assertTrue(funasr.get_training_adapter().spec.native_training)

        espnet = ESPnetASRForSpeechRecognition(ESPnetASRConfig())
        self.assertIsNone(espnet._validate_training_runtime())
        self.assertEqual(espnet.training_support, "native")
        self.assertTrue(espnet.supports_generic_finetuning)
        self.assertTrue(espnet.get_training_adapter().spec.native_training)
        self.assertIsNone(NeMoASRForSpeechRecognition(NeMoASRConfig(), )._validate_training_runtime())
        wenet = WeNetASRForSpeechRecognition(WeNetASRConfig())
        self.assertIsNone(wenet._validate_training_runtime())
        self.assertEqual(wenet.training_support, "native")
        self.assertTrue(wenet.supports_generic_finetuning)
        speechbrain = SpeechBrainASRForSpeechRecognition(SpeechBrainASRConfig(), )
        self.assertIsNone(speechbrain._validate_training_runtime())
        self.assertEqual(speechbrain.training_support, "native")
        self.assertTrue(speechbrain.supports_generic_finetuning)

    def test_runtime_tokens_are_not_serialized(self):
        models = (
            WhisperXForSpeechRecognition(
                WhisperXConfig(),
                token="whisperx-secret",
            ),
            NeMoASRForSpeechRecognition(
                NeMoASRConfig(),
                token="nemo-secret",
            ),
            SpeechBrainASRForSpeechRecognition(
                SpeechBrainASRConfig(),
                token="speechbrain-secret",
            ),
        )

        for model in models:
            with self.subTest(model=model.config.model_type):
                serialized = json.dumps(model.config.to_dict())
                self.assertNotIn("secret", serialized)
                self.assertNotIn('"token":', serialized)
                self.assertNotIn('"auth_token":', serialized)
                self.assertNotIn('"use_auth_token":', serialized)

    def test_configs_reject_nested_credentials_and_managed_loader_options(self):
        with self.assertRaisesRegex(ValueError, "credentials"):
            FasterWhisperConfig(
                inference_config={
                    "provider": {
                        "api_key": "secret",
                    },
                }, )
        with self.assertRaisesRegex(ValueError, "managed"):
            NeMoASRConfig(
                model_kwargs={
                    "map_location": "cuda",
                }, )
        with self.assertRaisesRegex(ValueError, "quantization pass"):
            FasterWhisperConfig(compute_type="int8")
        with self.assertRaisesRegex(ValueError, "cannot be applied silently"):
            FasterWhisperConfig(cpu_threads=4)

    def test_funasr_wrapper_and_registry_use_the_same_default_checkpoint(self):
        self.assertEqual(
            FunASRForSpeechRecognition.default_model_name_or_path,
            get_model_spec("asr_funasr").default_model_path,
        )

    def test_cpu_cuda_native_runtimes_never_auto_select_mps(self):
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True), )
        model_classes = (
            NeMoASRForSpeechRecognition,
            SpeechBrainASRForSpeechRecognition,
            FunASRForSpeechRecognition,
            ESPnetASRForSpeechRecognition,
            WeNetASRForSpeechRecognition,
        )

        with _temporary_modules({"torch": fake_torch}):
            for model_class in model_classes:
                with self.subTest(model=model_class.__name__):
                    self.assertEqual(model_class._resolve_device("auto"), "cpu")
                    with self.assertRaisesRegex(ValueError, "CPU and CUDA"):
                        model_class._resolve_device("mps")

    def test_whisperx_rejects_half_precision_on_cpu(self):
        model = WhisperXForSpeechRecognition(
            WhisperXConfig(compute_type="float16"),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "float16 execution on CPU"):
            model._model_dtype()


class WhisperProviderInferenceTests(unittest.TestCase):

    def test_faster_whisper_alias_uses_the_native_trainable_graph(self):
        model = FasterWhisperForSpeechRecognition(
            FasterWhisperConfig(
                name_or_path="small.en",
                compute_type="float32",
            ),
            device="cpu",
        )
        adapter = model.get_training_adapter()

        self.assertIsInstance(model, WhisperForSpeechRecognition)
        self.assertEqual(
            model.config.name_or_path,
            "openai/whisper-small.en",
        )
        self.assertEqual(model.config.torch_dtype, "float32")
        self.assertIsInstance(adapter, NativeWhisperTrainingAdapter)
        self.assertTrue(adapter.spec.native_training)

    def test_openai_whisper_alias_uses_the_native_trainable_graph(self):
        model = OpenAIWhisperForSpeechRecognition(
            OpenAIWhisperConfig(name_or_path="medium"),
            device="cpu",
        )
        adapter = model.get_training_adapter()

        self.assertIsInstance(model, WhisperForSpeechRecognition)
        self.assertEqual(
            model.config.name_or_path,
            "openai/whisper-medium",
        )
        self.assertIsInstance(adapter, NativeWhisperTrainingAdapter)
        self.assertTrue(adapter.spec.native_training)
        self.assertEqual(
            adapter.spec.source_entrypoints,
            ("voicehub.models.asr_whisper_native.native.WhisperModel", ),
        )

    def test_whisperx_alias_uses_the_native_trainable_graph(self):
        model = WhisperXForSpeechRecognition(
            WhisperXConfig(
                name_or_path="large-v3",
                compute_type="float32",
            ),
            device="cpu",
        )
        adapter = model.get_training_adapter()

        self.assertIsInstance(model, WhisperForSpeechRecognition)
        self.assertEqual(
            model.config.name_or_path,
            "openai/whisper-large-v3",
        )
        self.assertIsInstance(adapter, NativeWhisperTrainingAdapter)
        self.assertTrue(adapter.spec.native_training)


class NativeToolkitInferenceTests(unittest.TestCase):

    def test_unsupported_non_default_options_fail_instead_of_disappearing(self):
        cases = (
            (
                FasterWhisperForSpeechRecognition(FasterWhisperConfig()),
                {
                    "stride_length_s": 1.0
                },
                "stride_length_s",
            ),
            (
                OpenAIWhisperForSpeechRecognition(OpenAIWhisperConfig()),
                {
                    "num_beams": 2
                },
                "num_beams",
            ),
            (
                WhisperXForSpeechRecognition(WhisperXConfig()),
                {
                    "hotwords": ["VoiceHub"]
                },
                "hotwords",
            ),
            (
                NeMoASRForSpeechRecognition(NeMoASRConfig()),
                {
                    "task": "translate"
                },
                "task",
            ),
            (
                SpeechBrainASRForSpeechRecognition(SpeechBrainASRConfig()),
                {
                    "hotwords": "VoiceHub"
                },
                "hotwords",
            ),
            (
                FunASRForSpeechRecognition(FunASRConfig()),
                {
                    "batch_size": 12
                },
                "batch_size",
            ),
            (
                ESPnetASRForSpeechRecognition(ESPnetASRConfig()),
                {
                    "batch_size": 2
                },
                "batch_size",
            ),
            (
                WeNetASRForSpeechRecognition(WeNetASRConfig()),
                {
                    "hotwords": "VoiceHub"
                },
                "hotwords",
            ),
        )

        for model, options, option_name in cases:
            with (
                    self.subTest(model=model.config.model_type),
                    self.assertRaisesRegex(ValueError, option_name),
            ):
                model._transcribe(
                    np.zeros(160, dtype=np.float32),
                    sampling_rate=16_000,
                    **options,
                )

    def test_nemo_rejects_unverified_graph_families_before_import(self):
        model = NeMoASRForSpeechRecognition(
            NeMoASRConfig(name_or_path="nvidia/parakeet-test"),
            device="cpu",
        )

        with self.assertRaisesRegex(ValueError, "not the verified QuartzNet15x5"):
            model.load()
        self.assertNotIn("nemo", sys.modules)

    def test_speechbrain_cache_path_round_trip_stays_portable(self):
        config = SpeechBrainASRConfig(savedir=Path("cache") / "speechbrain", )

        serialized = config.to_dict()
        restored = SpeechBrainASRConfig.from_dict(serialized)

        self.assertEqual(config.cache_dir, "cache/speechbrain")
        self.assertEqual(config.savedir, "cache/speechbrain")
        self.assertEqual(restored.cache_dir, "cache/speechbrain")
        self.assertEqual(restored.savedir, "cache/speechbrain")
        self.assertEqual(restored.to_dict(), serialized)

    def test_speechbrain_uses_native_beam_search_and_training_boundaries(self):
        model = SpeechBrainASRForSpeechRecognition(
            SpeechBrainASRConfig(savedir="cache/speechbrain"),
            device="cpu",
            token="private",
        )

        self.assertEqual(model.config.cache_dir, "cache/speechbrain")
        self.assertEqual(model.architecture_family, "speech-seq2seq")
        self.assertEqual(
            model._validate_inference_request(
                language="english",
                task="transcribe",
                return_timestamps=False,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=1,
                num_beams=4,
                max_new_tokens=None,
                hotwords=None,
            ),
            ("en", 4),
        )
        self.assertIsNone(model._validate_training_runtime())
        with self.assertRaisesRegex(ValueError, "timestamps"):
            model._validate_inference_request(
                language=None,
                task="transcribe",
                return_timestamps=True,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                num_beams=None,
                max_new_tokens=None,
                hotwords=None,
            )

    def test_funasr_compatibility_is_scope_limited_to_native_sensevoice(self):
        model = FunASRForSpeechRecognition(
            FunASRConfig(vad_model="fsmn-vad"),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "separate architectures"):
            model._validate_composed_options()
        with self.assertRaisesRegex(ValueError, "SenseVoiceSmall artifact"):
            model._validate_architecture({
                "model_type": "paraformer",
                'architectures': ["Paraformer"],
            })
        with self.assertRaisesRegex(ValueError, "hotwords"):
            model._validate_request(
                language="zh",
                task="transcribe",
                return_timestamps=True,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=1,
                num_beams=1,
                max_new_tokens=None,
                hotwords=("VoiceHub", ),
            )

    def test_espnet_native_request_contract_is_narrow_and_explicit(self):
        language, beam_size = (
            ESPnetASRForSpeechRecognition._validate_inference_request(
                language="english",
                task="transcribe",
                return_timestamps=False,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=1,
                num_beams=4,
                max_new_tokens=None,
                hotwords=None,
            ))

        self.assertEqual(language, "en")
        self.assertEqual(beam_size, 4)
        with self.assertRaisesRegex(ValueError, "timestamps"):
            ESPnetASRForSpeechRecognition._validate_inference_request(
                language="en",
                task="transcribe",
                return_timestamps=True,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=1,
                num_beams=None,
                max_new_tokens=None,
                hotwords=None,
            )

    def test_wenet_is_native_and_validates_english_requests(self):
        model = WeNetASRForSpeechRecognition(
            WeNetASRConfig(
                name_or_path="wenet/gigaspeech-u2pp-conformer",
                language="en",
                decoding_strategy="attention_rescoring",
            ),
            device="cpu",
        )

        language, timestamps = model._validate_request(
            language=None,
            task="transcribe",
            return_timestamps="word",
            chunk_length_s=None,
            stride_length_s=None,
            batch_size=None,
            max_new_tokens=None,
            hotwords=None,
        )

        self.assertEqual(language, "en")
        self.assertTrue(timestamps)
        self.assertEqual(model.training_support, "native")
        with self.assertRaisesRegex(ValueError, "English-only"):
            model._validate_request(
                language="tr",
                task="transcribe",
                return_timestamps=False,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                max_new_tokens=None,
                hotwords=None,
            )


if __name__ == "__main__":
    unittest.main()
