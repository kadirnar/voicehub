from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from voicehub.models.vad_sherpa_onnx.native.registration import (
    create_vad_dispatch_architecture_spec,
    register_vad_dispatch_architecture,
)
from voicehub.runtime.registry import ArchitectureRegistry


class NativeVADDispatchTests(unittest.TestCase):

    def test_spec_declares_closed_silero_and_ten_bundle(self):
        spec = create_vad_dispatch_architecture_spec()

        self.assertEqual(spec.architecture_id, "native-vad-dispatch")
        self.assertEqual(
            tuple(spec.components),
            ("silero-vad", "ten-vad"),
        )
        self.assertEqual(
            spec.metadata["families"],
            ("silero-vad", "ten-vad"),
        )
        self.assertTrue(spec.capabilities.training)
        self.assertTrue(spec.capabilities.streaming)
        self.assertTrue(spec.capabilities.supports_task("voice-activity-detection"), )
        self.assertEqual(
            spec.model_builder.path,
            ("voicehub.models.vad_sherpa_onnx.modeling:"
             "SherpaONNXVADForVoiceActivityDetection"),
        )

    def test_registration_supports_isolated_registry_and_aliases(self):
        registry = ArchitectureRegistry()

        spec = register_vad_dispatch_architecture(registry=registry)

        self.assertIs(registry.get("native-vad-dispatch"), spec)
        self.assertIs(registry.get("native-vad"), spec)
        self.assertIs(registry.get("generic-native-vad"), spec)

    def test_registration_does_not_import_family_graphs(self):
        script = """
import json
import sys
from voicehub.runtime.registry import ArchitectureRegistry
from voicehub.models.vad_sherpa_onnx.native.registration import (
    register_vad_dispatch_architecture,
)

register_vad_dispatch_architecture(registry=ArchitectureRegistry())
print(json.dumps({
    "silero": "voicehub.models.vad_silero.native.modeling" in sys.modules,
    "ten": "voicehub.models.vad_ten.native.modeling" in sys.modules,
    "wrapper": (
        "voicehub.models.vad_sherpa_onnx.modeling"
        in sys.modules
    ),
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )

        self.assertEqual(
            result.stdout.strip(),
            '{"silero": false, "ten": false, "wrapper": false}',
        )


if __name__ == "__main__":
    unittest.main()
