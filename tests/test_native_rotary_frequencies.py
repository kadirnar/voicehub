"""RoPE inverse frequencies must be rebuilt exactly as they are constructed."""

from __future__ import annotations

import dataclasses
import unittest

import torch
from torch import nn

from voicehub.architectures.cosyvoice_native.checkpoint import _materialize_runtime_buffers as cosyvoice_materialize
from voicehub.architectures.qwen3_asr.modeling import materialize_qwen3_asr_nonpersistent_buffers
from voicehub.architectures.qwen3_tts.modeling import materialize_qwen3_tts_buffers
from voicehub.architectures.vibevoice.checkpoint import _materialize_runtime_buffers as vibevoice_materialize
from voicehub.neural.rotary import RotaryEmbedding

_LLAMA3_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8_192,
}

_LOADERS = {
    "cosyvoice": lambda model, device: cosyvoice_materialize(model, device=device),
    "qwen3_asr": lambda model, device: materialize_qwen3_asr_nonpersistent_buffers(model, device=device),
    "qwen3_tts": lambda model, device: materialize_qwen3_tts_buffers(model, device=device),
    "vibevoice": lambda model, device: vibevoice_materialize(model, device=device),
}


def _transformers_inverse_frequency(base: float, dimension: int) -> torch.Tensor:
    # Transformers' default RoPE init, evaluated on the CPU at construction.
    return 1.0 / (base**(torch.arange(0, dimension, 2, dtype=torch.int64).float() / dimension))


class RotaryInverseFrequencyTests(unittest.TestCase):

    def test_construction_matches_transformers_cpu_init(self):
        for base, dimension in ((10_000.0, 64), (500_000.0, 128), (1_000_000.0, 128), (10_000.0, 4)):
            with self.subTest(base=base, dimension=dimension):
                self.assertTrue(
                    torch.equal(
                        RotaryEmbedding(dimension, base=base).inverse_frequency,
                        _transformers_inverse_frequency(base, dimension),
                    ))

    def test_reset_rebuilds_scaled_frequencies_after_meta_construction(self):
        for scaling in (None, _LLAMA3_SCALING):
            with self.subTest(scaling=scaling):
                expected = RotaryEmbedding(128, base=500_000.0, scaling=scaling).inverse_frequency
                rotary = RotaryEmbedding(128, base=500_000.0, scaling=scaling, device="meta")
                rotary.reset_inverse_frequency("cpu")
                self.assertEqual(rotary.inverse_frequency.device.type, "cpu")
                self.assertTrue(torch.equal(rotary.inverse_frequency, expected))
        self.assertFalse(
            torch.equal(
                RotaryEmbedding(128, base=500_000.0, scaling=_LLAMA3_SCALING).inverse_frequency,
                _transformers_inverse_frequency(500_000.0, 128),
            ))

    def test_checkpoint_loaders_keep_rope_scaling(self):
        # Loaders used to recompute the unscaled formula after a meta load,
        # silently dropping e.g. Llama-3 RoPE scaling.
        expected = RotaryEmbedding(128, base=500_000.0, scaling=_LLAMA3_SCALING).inverse_frequency
        for name, materialize in _LOADERS.items():
            with self.subTest(loader=name):
                rotary = RotaryEmbedding(128, base=500_000.0, scaling=_LLAMA3_SCALING, device="meta")
                materialize(nn.ModuleList([rotary]), "cpu")
                self.assertEqual(rotary.inverse_frequency.device.type, "cpu")
                self.assertTrue(torch.equal(rotary.inverse_frequency, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA pow differs from the CPU reference only on GPU.")
    def test_checkpoint_loaders_on_cuda_match_transformers_cpu_init(self):
        # CUDA float32 pow differs from the CPU result in the last bit for
        # some frequencies (e.g. indices 18 and 50 for base 1e6, dim 128).
        from tests.test_asr_granite_speech import _tiny_config
        from voicehub.architectures.granite_speech.modeling import (
            GraniteSpeechForConditionalGeneration,
            materialize_granite_speech_nonpersistent_buffers,
        )

        loaders = dict(_LOADERS)
        loaders["granite_speech"] = None
        for name, materialize in loaders.items():
            with self.subTest(loader=name):
                if materialize is None:
                    config = _tiny_config()
                    text_config = dataclasses.replace(
                        config.text_config,
                        hidden_size=128,
                        num_attention_heads=1,
                        num_key_value_heads=1,
                        head_dim=128,
                    )
                    with torch.device("meta"):
                        model = GraniteSpeechForConditionalGeneration(
                            dataclasses.replace(config, text_config=text_config))
                    materialize_granite_speech_nonpersistent_buffers(model, device="cuda")
                else:
                    model = nn.ModuleList([RotaryEmbedding(128, base=1_000_000.0, device="meta")])
                    materialize(model, "cuda")
                modules = [module for module in model.modules() if isinstance(module, RotaryEmbedding)]
                self.assertTrue(modules)
                for module in modules:
                    self.assertEqual(module.inverse_frequency.device.type, "cuda")
                    self.assertTrue(
                        torch.equal(
                            module.inverse_frequency.cpu(),
                            _transformers_inverse_frequency(module.base, module.dimension),
                        ))


if __name__ == "__main__":
    unittest.main()
