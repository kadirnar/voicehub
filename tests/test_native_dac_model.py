from __future__ import annotations

import unittest

import torch

from voicehub.checkpointing.transforms import CopyTensor
from voicehub.models.dac.native.checkpoint import (
    DESCRIPT_DAC_44KHZ_HEADER_FINGERPRINT,
    HuggingFaceDacCheckpointAdapter,
    WeightNormalizedTensor,
    dac_tensor_inventory_fingerprint,
    huggingface_dac_tensor_names,
    huggingface_dac_tensor_shapes,
)
from voicehub.models.dac.native.configuration import DacConfig
from voicehub.models.dac.native.modeling import DacModel
from voicehub.runtime import ArchitectureRegistry, register_builtin_architectures


class NativeDACModelTests(unittest.TestCase):

    @staticmethod
    def tiny_config() -> DacConfig:
        return DacConfig(
            encoder_hidden_size=4,
            downsampling_ratios=(2, 2),
            decoder_hidden_size=16,
            n_codebooks=2,
            codebook_size=8,
            codebook_dim=2,
            sampling_rate=16_000,
        )

    def test_published_configuration_derivations_are_validated(self):
        values = {
            'architectures': ["DacModel"],
            "codebook_dim": 8,
            "codebook_size": 1_024,
            "decoder_hidden_size": 1_536,
            "downsampling_ratios": [2, 4, 8, 8],
            "encoder_hidden_size": 64,
            "hidden_size": 1_024,
            "hop_length": 512,
            "model_type": "dac",
            "n_codebooks": 9,
            "quantizer_dropout": 0.0,
            "sampling_rate": 44_100,
            "upsampling_ratios": [8, 8, 4, 2],
        }

        config = DacConfig.from_dict(values)

        self.assertEqual(config.hidden_size, 1_024)
        self.assertEqual(config.hop_length, 512)
        self.assertEqual(config.frame_rate, 87)
        self.assertEqual(config.to_dict()["model_type"], "dac")
        invalid = dict(values, hop_length=320)
        with self.assertRaisesRegex(ValueError, "hop_length"):
            DacConfig.from_dict(invalid)

    def test_default_checkpoint_inventory_matches_published_header_count(self):
        config = DacConfig()

        source_names = huggingface_dac_tensor_names(config)
        with torch.device("meta"):
            model = DacModel(config)

        self.assertEqual(len(source_names), 223)
        self.assertEqual(len(model.state_dict()), 301)
        self.assertEqual(
            dac_tensor_inventory_fingerprint(huggingface_dac_tensor_shapes(config)),
            DESCRIPT_DAC_44KHZ_HEADER_FINGERPRINT,
        )

    def test_checkpoint_adapter_strictly_assigns_weight_normalization(self):
        config = self.tiny_config()
        source_model = DacModel(config)
        source_state = source_model.state_dict()
        adapter = HuggingFaceDacCheckpointAdapter()
        plan = adapter.tensor_plan(config.to_dict())
        checkpoint = {}
        for rule in plan.rules:
            if isinstance(rule, CopyTensor):
                checkpoint[rule.source] = source_state[rule.target].clone()
            elif isinstance(rule, WeightNormalizedTensor):
                checkpoint[rule.source] = source_state[rule.weight_target].clone()
            else:  # pragma: no cover - conversion plan is intentionally closed.
                self.fail(f"Unexpected DAC tensor rule {type(rule).__name__}")

        with torch.device("meta"):
            restored = DacModel(config)
        report = adapter.load_assign(
            restored,
            checkpoint,
            config.to_dict(),
            strict=True,
        )

        self.assertTrue(report.is_compatible)
        restored_state = restored.state_dict()
        for name, expected in source_state.items():
            torch.testing.assert_close(restored_state[name], expected)

    def test_forward_and_typed_codec_methods_are_differentiable(self):
        model = DacModel(self.tiny_config())
        waveform = torch.randn(1, 1, 64, requires_grad=True)

        output = model(waveform, sample_rate=16_000)
        encoded = model.encode_output(model.preprocess(waveform, sample_rate=16_000))
        decoded = model.decode_output(encoded.quantized_representation)
        loss = output["audio"].square().mean() + model.quantizer_loss(encoded)
        loss.backward()

        self.assertEqual(tuple(output["audio"].shape), (1, 1, 64))
        self.assertEqual(encoded.audio_codes.shape[1], 2)
        self.assertEqual(decoded.audio_values.ndim, 3)
        self.assertIsNotNone(waveform.grad)
        self.assertTrue(torch.isfinite(waveform.grad).all())

    def test_native_catalog_declares_dac_training_components(self):
        registry = ArchitectureRegistry()
        register_builtin_architectures(registry=registry)

        spec = registry.get("native-dac")

        self.assertEqual(spec.architecture_id, "dac")
        self.assertTrue(spec.capabilities.training)
        self.assertTrue(spec.capabilities.has_feature("audio-codec"))
        self.assertIs(
            spec.resolve_component("model-builder"),
            DacModel,
        )


if __name__ == "__main__":
    unittest.main()
