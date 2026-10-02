"""Regressions found by the edwko/OuteTTS upstream parity audit.

Reference values were produced with the author package (outetts 0.4.4 at
f5eac6e70d792844c6a6959d900a47af2c061a5b, pyloudnorm 0.1.x, NumPy 2.4).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.architectures.causal_lm.configuration import LlamaConfig
from voicehub.architectures.outetts.artifacts import resolve_outetts_artifacts
from voicehub.architectures.outetts.checkpoint import load_outetts_language_model
from voicehub.architectures.outetts.modeling import OuteTTSForCausalLM


def _tiny_llama3_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
        rope_theta=500_000.0,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8,
            "rope_type": "llama3",
        },
    )


class OuteTTSCheckpointLoadingTests(unittest.TestCase):

    def _export(self, root: Path) -> OuteTTSForCausalLM:
        torch.manual_seed(3)
        original = OuteTTSForCausalLM(_tiny_llama3_config()).eval()
        original.save_pretrained(root)
        (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        return original

    def test_loaded_language_model_runs_with_materialized_rotary_frequencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self._export(root)
            artifacts = resolve_outetts_artifacts(root)
            restored, _ = load_outetts_language_model(artifacts, device="cpu", dtype=torch.float32)

        self.assertFalse([name for name, value in restored.named_buffers() if value.device.type == "meta"])
        input_ids = torch.tensor([[1, 5, 6, 7, 9, 11]], dtype=torch.long)
        with torch.no_grad():
            expected = original(input_ids).logits
            observed = restored.eval()(input_ids).logits
        self.assertTrue(torch.equal(observed, expected))

    def test_hugging_face_snapshot_symlinks_keep_safetensors_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            export = Path(temporary) / "export"
            export.mkdir()
            self._export(export)
            blobs = Path(temporary) / "blobs"
            snapshot = Path(temporary) / "snapshots" / "0123"
            blobs.mkdir()
            snapshot.mkdir(parents=True)
            for index, path in enumerate(sorted(export.iterdir())):
                blob = blobs / f"blob{index}"
                path.rename(blob)
                os.symlink(blob, snapshot / path.name)

            artifacts = resolve_outetts_artifacts(snapshot)
            self.assertEqual(artifacts.checkpoint.name, "model.safetensors")
            model, config = load_outetts_language_model(artifacts, device="cpu", dtype=torch.float32)

        self.assertEqual(config.model_type, "llama")
        self.assertIsInstance(model, OuteTTSForCausalLM)


if __name__ == "__main__":
    unittest.main()
