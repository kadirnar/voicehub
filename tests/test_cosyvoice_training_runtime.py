from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from voicehub.models.cosyvoice import CosyVoiceConfig
from voicehub.models.cosyvoice import CosyVoiceConfig as NativeCosyVoiceConfig
from voicehub.models.cosyvoice import CosyVoiceForTextToSpeech
from voicehub.models.cosyvoice import CosyVoiceForTextToSpeech as NativeCosyVoiceForTextToSpeech
from voicehub.models.cosyvoice import CosyVoiceTrainingAdapter, CosyVoiceTrainingCollator, CosyVoiceTTS
from voicehub.registry import get_model_spec
from voicehub.training.specs import get_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CosyVoicePublicRuntimeTests(unittest.TestCase):

    def test_public_api_routes_to_the_native_runtime(self):
        self.assertIs(CosyVoiceConfig, NativeCosyVoiceConfig)
        self.assertIs(
            CosyVoiceForTextToSpeech,
            NativeCosyVoiceForTextToSpeech,
        )
        self.assertIs(CosyVoiceTTS, CosyVoiceForTextToSpeech)
        self.assertEqual(
            CosyVoiceTrainingAdapter.__module__,
            "voicehub.models.cosyvoice.training_cosyvoice",
        )
        self.assertEqual(
            CosyVoiceTrainingCollator.__module__,
            "voicehub.models.cosyvoice.training_cosyvoice",
        )

    def test_public_package_is_dependency_lazy(self):
        code = """
import json
import sys
import voicehub.models.cosyvoice as cosyvoice
print(json.dumps({
    "exports": sorted(cosyvoice.__all__),
    "torch": "torch" in sys.modules,
    "transformers": "transformers" in sys.modules,
    "modelscope": "modelscope" in sys.modules,
    "hyperpyyaml": "hyperpyyaml" in sys.modules,
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            (
                '{"exports": ["CosyVoiceConfig", '
                '"CosyVoiceForTextToSpeech", "CosyVoiceTTS", '
                '"CosyVoiceTrainingAdapter", "CosyVoiceTrainingCollator"], '
                '"torch": false, "transformers": false, '
                '"modelscope": false, "hyperpyyaml": false}'),
        )

    def test_registry_describes_only_executable_native_capabilities(self):
        spec = get_model_spec("cosyvoice")
        self.assertTrue(spec.is_voicehub_native)
        self.assertEqual(spec.architecture, "cosyvoice-native")
        self.assertNotIn("streaming", spec.capabilities)
        self.assertIn("safetensors", spec.capabilities)
        self.assertIn("fine-tuning", spec.capabilities)
        architecture = spec.native_architecture
        self.assertFalse(architecture.capabilities.streaming)
        self.assertTrue(architecture.capabilities.training)
        self.assertEqual(
            architecture.metadata["executable_checkpoint_compatibility"],
            "cosyvoice3-only",
        )

    def test_training_profile_routes_each_native_objective(self):
        spec = get_training_spec("cosyvoice")
        self.assertEqual(
            tuple(spec.phase_map),
            (
                "llm",
                "flow",
                "hifigan_generator",
                "hifigan_discriminator",
            ),
        )
        self.assertEqual(spec.default_phase, "llm")
        self.assertEqual(
            spec.phase_map["llm"].component_paths,
            ("model.llm", ),
        )
        self.assertEqual(
            spec.phase_map["flow"].component_paths,
            ("model.flow", ),
        )
        self.assertEqual(
            spec.phase_map["hifigan_discriminator"].component_paths,
            ("model.hifigan.discriminator", ),
        )
        self.assertTrue(spec.native_training)

    def test_adapter_plans_only_the_configured_component_job(self):

        class Wrapper:
            config = type(
                "Config",
                (),
                {
                    "training_component": "flow",
                },
            )()

        adapter = CosyVoiceTrainingAdapter(
            Wrapper(),
            get_training_spec("cosyvoice"),
        )
        self.assertEqual(
            tuple(phase.name for phase in adapter.plan_training_phases(12)),
            ("flow", ),
        )
        with self.assertRaisesRegex(ValueError, "separate job"):
            adapter.select_training_phase("llm")

    def test_collator_preserves_records_for_runtime_tokenization(self):
        records = [
            {
                "text": "Merhaba",
                "speech_tokens": [1, 2, 3],
            },
            {
                "text_tokens": [4, 5],
                "speech_tokens": [6],
            },
        ]
        batch = CosyVoiceTrainingCollator()(records)
        self.assertEqual(batch, {"records": records})
        self.assertIsNot(batch["records"][0], records[0])


if __name__ == "__main__":
    unittest.main()
