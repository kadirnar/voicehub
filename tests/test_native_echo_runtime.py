from __future__ import annotations

import unittest

import torch

from voicehub.models.echo.model import EchoDiT
from voicehub.models.echo.sampling import _assign_validated_state, _discard_blockwise_only_modules
from voicehub.registry import get_model_spec
from voicehub.runtime import get_architecture_spec


def _tiny_echo() -> EchoDiT:
    return EchoDiT(
        latent_size=4,
        model_size=8,
        num_layers=1,
        num_heads=2,
        intermediate_size=16,
        norm_eps=1e-5,
        text_vocab_size=32,
        text_model_size=8,
        text_num_layers=1,
        text_num_heads=2,
        text_intermediate_size=16,
        speaker_patch_size=2,
        speaker_model_size=8,
        speaker_num_layers=1,
        speaker_num_heads=2,
        speaker_intermediate_size=16,
        timestep_embed_size=8,
        adaln_rank=4,
    )


class NativeEchoRuntimeTests(unittest.TestCase):

    def test_registry_resolves_the_lazy_native_echo_architecture(self):
        model_spec = get_model_spec("echo")
        architecture = get_architecture_spec("echo-tts")

        self.assertTrue(model_spec.is_voicehub_native)
        self.assertIs(model_spec.native_architecture, architecture)
        self.assertEqual(architecture.architecture_id, "echo-dit")
        self.assertEqual(
            architecture.upstream_revision,
            "2ed95fce62d33bf7b56f835fd9ec0f0b6fb9155e",
        )
        self.assertEqual(
            set(architecture.components),
            {"blockwise-decoder", "fish-s1-dac", "pca-loader"},
        )

    def test_safe_state_assignment_requires_an_exact_inventory(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        state = {name: value.detach().clone() for name, value in source.state_dict().items()}

        _assign_validated_state(target, state)

        torch.testing.assert_close(target.weight, source.weight)
        torch.testing.assert_close(target.bias, source.bias)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            _assign_validated_state(
                torch.nn.Linear(3, 2),
                {"weight": state["weight"]},
            )
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            _assign_validated_state(
                torch.nn.Linear(3, 2),
                {
                    **state,
                    "unknown": torch.zeros(1),
                },
            )

    def test_non_blockwise_load_removes_every_intentionally_omitted_module(self):
        model = _tiny_echo()
        _discard_blockwise_only_modules(model)

        self.assertFalse(model.blockwise_generation_available)
        self.assertFalse(
            any(
                name.startswith(("latent_encoder.",
                                 "latent_norm")) or ".wk_latent" in name or ".wv_latent" in name
                for name in model.state_dict()))
        with self.assertRaisesRegex(RuntimeError, "without blockwise"):
            model.get_kv_cache_latent(torch.zeros(1, 2, 4))


if __name__ == "__main__":
    unittest.main()
