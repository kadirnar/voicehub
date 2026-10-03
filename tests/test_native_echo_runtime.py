from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch

from voicehub.architectures import get_architecture_spec
from voicehub.models.echo.autoencoder import DecoderBlock, Snake1d
from voicehub.models.echo.model import EchoDiT, LowRankAdaLN
from voicehub.checkpointing import save_safetensors
from voicehub.models.echo import sampling as echo_sampling
from voicehub.models.echo.sampling import _assign_validated_state, _discard_blockwise_only_modules, load_audio
from voicehub.processing import save_pcm_wave
from voicehub.registry import get_model_spec


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

    def test_low_precision_adaln_preserves_upstream_gain_rounding(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.random.fork_rng():
                torch.manual_seed(42)
                layer = LowRankAdaLN(8, 4, 1e-5).to(dtype)
                inputs = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
                condition = torch.randn(2, 1, 24, dtype=dtype, requires_grad=True)
                shift, scale, gate = condition.chunk(3, dim=-1)
                shift = layer.shift_up(layer.shift_down(torch.nn.functional.silu(shift))) + shift
                scale = layer.scale_up(layer.scale_down(torch.nn.functional.silu(scale))) + scale
                gate = layer.gate_up(layer.gate_down(torch.nn.functional.silu(gate))) + gate
                converted = inputs.float()
                normalized = converted * torch.rsqrt(converted.square().mean(-1, keepdim=True) + layer.eps)
                expected = (normalized * (scale + 1) + shift).to(dtype)
                actual, actual_gate = layer(inputs, condition)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_gate, gate.tanh(), rtol=0, atol=0)
                expected_grad = torch.autograd.grad(expected.sum(), inputs, retain_graph=True)[0]
                actual_grad = torch.autograd.grad(actual.sum(), inputs)[0]
                torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_decoder_loads_published_block_numbering_even_with_transformer_option(self):
        # The released codec's decoder block.0 is Snake, not a transformer
        # or Identity. Inserting either shifts every published tensor key.
        reference = DecoderBlock(input_dim=8, output_dim=4, stride=2, causal=True)
        target = DecoderBlock(input_dim=8, output_dim=4, stride=2, causal=True, n_t_layer=4)
        self.assertIsInstance(target.block[0], Snake1d)
        state = reference.state_dict()
        self.assertIn("block.0.alpha", state)
        self.assertIn("block.1.conv.bias", state)
        _assign_validated_state(target, state)
        inputs = torch.randn(1, 8, 12)
        torch.testing.assert_close(target(inputs), reference(inputs), rtol=0, atol=0)

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

    def test_legacy_weight_norm_checkpoint_preserves_values_and_rejects_alias_collisions(self):
        source = torch.nn.Sequential(torch.nn.utils.weight_norm(torch.nn.Conv1d(3, 2, 1)))
        target = torch.nn.Sequential(torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(3, 2, 1)))
        legacy = source.state_dict()
        _assign_validated_state(target, legacy)
        inputs = torch.randn(1, 3, 5)
        torch.testing.assert_close(target(inputs), source(inputs), rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "duplicate weight aliases"):
            _assign_validated_state(target, {**legacy, **target.state_dict()})

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

    def test_reference_audio_is_resampled_like_upstream_torchaudio_functional(self):
        # Upstream ``load_audio`` decodes at most ``max_duration`` seconds,
        # averages channels and calls ``torchaudio.functional.resample``.
        try:
            import torchaudio.functional as torchaudio_functional
        except ImportError:  # pragma: no cover - torchaudio is a test extra
            self.skipTest("torchaudio is not installed")
        generator = torch.Generator().manual_seed(0)
        stereo = (torch.rand(2, 16_000 * 2 + 37, generator=generator) - 0.5) * 1.6
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speaker.wav"
            save_pcm_wave(path, stereo, 16_000)
            stored = torch.round(stereo.clamp(-1, 1) * 32767) / 32768
            mono = stored.mean(dim=0, keepdim=True)
            expected = torchaudio_functional.resample(mono, 16_000, 44_100)
            expected = expected / torch.maximum(expected.abs().max(), torch.tensor(1.0))
            torch.testing.assert_close(load_audio(str(path)), expected, rtol=0, atol=0)

            limited = torchaudio_functional.resample(mono[:, :16_000], 16_000, 44_100)
            limited = limited / torch.maximum(limited.abs().max(), torch.tensor(1.0))
            torch.testing.assert_close(load_audio(str(path), max_duration=1), limited, rtol=0, atol=0)

    def test_float32_codec_load_keeps_stored_buffer_dtypes_like_upstream(self):
        # Upstream only casts the Fish codec state for a non-float32 dtype; the
        # default must not inflate boolean causal masks or bf16 RoPE tables.
        class TinyCodec(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(2, 2)
                self.register_buffer("freqs_cis", torch.ones(3, 2, dtype=torch.bfloat16))
                self.register_buffer("causal_mask", torch.tril(torch.ones(3, 3, dtype=torch.bool)))

        source = TinyCodec()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pytorch_model.safetensors"
            save_safetensors(source.state_dict(), path)
            with mock.patch.object(echo_sampling, "build_ae", TinyCodec):
                default = echo_sampling.load_fish_ae_from_hf(directory, device="cpu")
                half = echo_sampling.load_fish_ae_from_hf(directory, device="cpu", dtype=torch.bfloat16)
        self.assertEqual(default.causal_mask.dtype, torch.bool)
        self.assertEqual(default.freqs_cis.dtype, torch.bfloat16)
        self.assertEqual(default.proj.weight.dtype, torch.float32)
        torch.testing.assert_close(default.proj.weight, source.proj.weight, rtol=0, atol=0)
        self.assertEqual(half.proj.weight.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
