from __future__ import annotations

import json
import subprocess
import sys
import unittest

from voicehub.models.speecht5.native.metadata import (
    SPEECHT5_HIFIGAN_REFERENCE_INVENTORY,
    SPEECHT5_REFERENCE_INVENTORY,
    SPEECHT5_SOURCE_REVISION,
)
from voicehub.models.speecht5.native.registration import (
    DEFAULT_SPEECHT5_ALIASES,
    create_speecht5_architecture_spec,
    register_speecht5_architecture,
)
from voicehub.runtime.registry import ArchitectureRegistry
from voicehub.tasks import SpeechTask


class NativeSpeechT5RegistrationTests(unittest.TestCase):

    def test_registration_is_lazy_and_dependency_free(self):
        code = """
import json
import sys
import voicehub.models.speecht5.native.registration
print(json.dumps({
    name: name in sys.modules
    for name in ("torch", "transformers", "numpy", "safetensors")
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(result.stdout),
            {
                "torch": False,
                "transformers": False,
                "numpy": False,
                "safetensors": False,
            },
        )

    def test_spec_points_only_to_native_speecht5_components(self):
        spec = create_speecht5_architecture_spec()

        self.assertEqual(spec.architecture_id, "speecht5")
        self.assertEqual(spec.upstream_revision, SPEECHT5_SOURCE_REVISION)
        self.assertEqual(spec.license_id, "Apache-2.0")
        self.assertTrue(spec.supports_task(SpeechTask.TEXT_TO_SPEECH))
        self.assertTrue(spec.capabilities.training)
        self.assertTrue(spec.capabilities.batched_inference)
        self.assertTrue(spec.capabilities.distributed_training)
        self.assertFalse(spec.capabilities.streaming)
        self.assertEqual(
            spec.capabilities.checkpoint_formats,
            ("safetensors", "bin", "pytorch"),
        )
        for reference in spec.component_references.values():
            self.assertTrue(
                reference.module.startswith("voicehub.models.speecht5."),
                reference.path,
            )

    def test_spec_encodes_exact_checkpoint_inventories(self):
        spec = create_speecht5_architecture_spec()

        acoustic = spec.metadata["reference_checkpoint"]
        vocoder = spec.metadata["vocoder_reference_checkpoint"]
        self.assertEqual(acoustic["tensor_count"], 398)
        self.assertEqual(acoustic["state_values"], 146_335_465)
        self.assertEqual(
            acoustic["tensor_inventory_fingerprint"],
            SPEECHT5_REFERENCE_INVENTORY["tensor_inventory_fingerprint"],
        )
        self.assertEqual(vocoder["tensor_count"], 158)
        self.assertEqual(vocoder["state_values"], 12_656_417)
        self.assertEqual(
            vocoder["tensor_inventory_fingerprint"],
            SPEECHT5_HIFIGAN_REFERENCE_INVENTORY["tensor_inventory_fingerprint"],
        )
        self.assertEqual(acoustic["license"], "MIT")
        self.assertEqual(vocoder["license"], "MIT")
        self.assertFalse(spec.metadata["official_safetensors_published"])
        self.assertTrue(spec.metadata["full_finetuning_ready"])
        self.assertEqual(
            spec.metadata["always_frozen_components"],
            ("vocoder", ),
        )

    def test_every_lazy_component_resolves(self):
        spec = create_speecht5_architecture_spec()

        for name, reference in spec.component_references.items():
            resolved = reference.resolve()
            self.assertTrue(callable(resolved), name)
            self.assertTrue(
                resolved.__module__.startswith("voicehub.models.speecht5."),
                name,
            )

    def test_registrar_registers_canonical_id_and_aliases(self):
        registry = ArchitectureRegistry()

        spec = register_speecht5_architecture(registry=registry)

        self.assertIs(registry.get("speecht5"), spec)
        for alias in DEFAULT_SPEECHT5_ALIASES:
            self.assertIs(registry.get(alias), spec)
        self.assertIs(
            register_speecht5_architecture(
                registry=registry,
                exist_ok=True,
            ),
            registry.get("speecht5"),
        )
        with self.assertRaisesRegex(TypeError, "ArchitectureRegistry"):
            register_speecht5_architecture(registry=object())


if __name__ == "__main__":
    unittest.main()
