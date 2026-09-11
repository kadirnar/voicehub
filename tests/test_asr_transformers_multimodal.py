import importlib
import subprocess
import sys
import unittest
from pathlib import Path

from voicehub.models.asr_qwen3 import Qwen3ASRConfig as NativeQwen3ASRConfig
from voicehub.models.asr_qwen3 import Qwen3ASRForSpeechRecognition as NativeQwen3ASRForSpeechRecognition
from voicehub.models.asr_transformers_multimodal import (
    MultimodalTransformersASRConfig,
    MultimodalTransformersASRForSpeechRecognition,
    Qwen3ASRConfig,
    Qwen3ASRForSpeechRecognition,
    VibeVoiceASRConfig,
    VibeVoiceASRForSpeechRecognition,
)
from voicehub.models.asr_vibevoice import VibeVoiceASRConfig as NativeVibeVoiceASRConfig
from voicehub.models.asr_vibevoice import VibeVoiceForSpeechRecognition as NativeVibeVoiceForSpeechRecognition

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NativeMultimodalCompatibilityTests(unittest.TestCase):

    def test_package_import_is_lazy_and_dependency_free(self):
        command = (
            "import sys; "
            "import voicehub.models.asr_transformers_multimodal; "
            "print('torch' in sys.modules, 'transformers' in sys.modules, "
            "'voicehub.models.asr_transformers_multimodal."
            "modeling' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False False False")

    def test_family_configs_are_canonical_native_aliases(self):
        self.assertIs(Qwen3ASRConfig, NativeQwen3ASRConfig)
        self.assertIs(VibeVoiceASRConfig, NativeVibeVoiceASRConfig)
        configuration_module = importlib.import_module(
            "voicehub.models.asr_transformers_multimodal."
            "configuration")
        self.assertIs(configuration_module.Qwen3ASRConfig, NativeQwen3ASRConfig)
        self.assertIs(
            configuration_module.VibeVoiceASRConfig,
            NativeVibeVoiceASRConfig,
        )

    def test_family_models_are_canonical_native_aliases(self):
        self.assertIs(
            Qwen3ASRForSpeechRecognition,
            NativeQwen3ASRForSpeechRecognition,
        )
        self.assertIs(
            VibeVoiceASRForSpeechRecognition,
            NativeVibeVoiceForSpeechRecognition,
        )
        modeling_module = importlib.import_module("voicehub.models.asr_transformers_multimodal."
                                                  "modeling")
        self.assertIs(
            modeling_module.Qwen3ASRForSpeechRecognition,
            NativeQwen3ASRForSpeechRecognition,
        )
        self.assertIs(
            modeling_module.VibeVoiceASRForSpeechRecognition,
            NativeVibeVoiceForSpeechRecognition,
        )

    def test_generic_config_factory_requires_explicit_architecture_evidence(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            MultimodalTransformersASRConfig()
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            MultimodalTransformersASRConfig(provider="unknown")

    def test_generic_config_factory_dispatches_to_native_families(self):
        qwen = MultimodalTransformersASRConfig(provider="qwen3")
        vibe = MultimodalTransformersASRConfig(model_type="asr_vibevoice", )
        inferred = MultimodalTransformersASRConfig(name_or_path="Qwen/Qwen3-ASR-0.6B", )

        self.assertIsInstance(qwen, NativeQwen3ASRConfig)
        self.assertIsInstance(vibe, NativeVibeVoiceASRConfig)
        self.assertIsInstance(inferred, NativeQwen3ASRConfig)
        self.assertEqual(qwen.architecture_family, "speech-seq2seq")
        self.assertEqual(vibe.architecture_family, "causal-multimodal-lm")

    def test_generic_model_factory_dispatches_without_loading_weights(self):
        qwen = MultimodalTransformersASRForSpeechRecognition(
            NativeQwen3ASRConfig(),
            device="cpu",
        )
        vibe = MultimodalTransformersASRForSpeechRecognition(
            provider="vibevoice",
            device="cpu",
        )

        self.assertIsInstance(qwen, NativeQwen3ASRForSpeechRecognition)
        self.assertIsInstance(vibe, NativeVibeVoiceForSpeechRecognition)
        self.assertFalse(qwen.is_loaded)
        self.assertFalse(vibe.is_loaded)

    def test_conflicting_generic_provider_hints_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            MultimodalTransformersASRConfig(
                provider="qwen3",
                model_type="asr_vibevoice",
            )

    def test_compatibility_imports_never_import_external_runtime(self):
        command = (
            "import builtins, sys\n"
            "_original = builtins.__import__\n"
            "def _guard(name, *args, **kwargs):\n"
            "    if name == 'transformers' or name.startswith('transformers.'):\n"
            "        raise AssertionError(name)\n"
            "    return _original(name, *args, **kwargs)\n"
            "builtins.__import__ = _guard\n"
            "from voicehub.models.asr_transformers_multimodal import "
            "Qwen3ASRConfig, Qwen3ASRForSpeechRecognition, "
            "VibeVoiceASRConfig, VibeVoiceASRForSpeechRecognition\n"
            "print('transformers' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
