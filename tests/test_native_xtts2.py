from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from voicehub.models.xtts.native.checkpoint import (
    convert_trusted_legacy_xtts2_checkpoint,
    inspect_xtts2_checkpoint,
    load_xtts2_checkpoint,
    save_xtts2_checkpoint,
)
from voicehub.models.xtts.native.configuration import XTTS2Config
from voicehub.models.xtts.native.gpt import XTTS2GPT
from voicehub.models.xtts.native.tokenizer import XTTS2Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = (
    PROJECT_ROOT / 'voicehub/models/xtts/native',
    PROJECT_ROOT / 'voicehub/models/xtts',
)
FORBIDDEN = {
    "TTS",
    "coqpit",
    "einops",
    "librosa",
    "numpy",
    "torchaudio",
    "transformers",
}


def _tiny_gpt() -> XTTS2GPT:
    return XTTS2GPT(
        start_text_token=30,
        stop_text_token=0,
        layers=2,
        model_dim=32,
        heads=4,
        max_text_tokens=20,
        max_mel_tokens=20,
        max_prompt_tokens=4,
        number_text_tokens=32,
        num_audio_tokens=18,
        start_audio_token=16,
        stop_audio_token=17,
    )


class NativeXTTS2Tests(unittest.TestCase):

    def test_public_namespaces_are_lazy(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import voicehub.models.xtts.native; "
                    "import voicehub.models.xtts; "
                    "print('torch' in sys.modules)"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_runtime_has_no_disallowed_import_boundary(self):
        violations = []
        for root in RUNTIME_ROOTS:
            for path in root.glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports = [item.name for item in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        imports = [node.module or ""]
                    else:
                        continue
                    for imported in imports:
                        if imported.partition(".")[0] in FORBIDDEN:
                            violations.append((path.name, imported))
        self.assertEqual(violations, [])

    def test_published_configuration_subset_is_immutable_and_exact(self):
        config = XTTS2Config.from_mapping({
            "audio": {
                "sample_rate": 22_050,
                "output_sample_rate": 24_000,
            },
            "model_args": {
                "gpt_layers": 30,
                "gpt_n_model_channels": 1_024,
                "gpt_n_heads": 16,
                "gpt_number_text_tokens": 6_681,
                "gpt_num_audio_tokens": 1_026,
                "gpt_start_audio_token": 1_024,
                "gpt_stop_audio_token": 1_025,
                "gpt_use_perceiver_resampler": True,
            },
        })
        self.assertEqual(config.audio.output_sample_rate, 24_000)
        self.assertEqual(config.model_args.gpt_num_audio_tokens, 1_026)
        with self.assertRaises((AttributeError, TypeError)):
            config.audio.sample_rate = 16_000

    def test_forward_preserves_source_cross_entropy_objectives(self):
        model = _tiny_gpt().train()
        text_loss, mel_loss, logits = model(
            torch.tensor([[3, 4]]),
            torch.tensor([2]),
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([1_024]),
            cond_latents=torch.randn(1, 3, 32),
        )
        self.assertEqual(logits.shape, (1, 18, 6))
        self.assertTrue(torch.isfinite(text_loss))
        self.assertTrue(torch.isfinite(mel_loss))
        (0.01 * text_loss + mel_loss).backward()
        self.assertIsNotNone(model.gpt.h[0].attn.c_attn.weight.grad)

    def test_tokenizer_preserves_space_and_last_duplicate_merge_semantics(self):
        vocabulary = {
            "[UNK]": 0,
            "[START]": 1,
            "[STOP]": 2,
            "[SPACE]": 3,
            "[en]": 4,
            "[zh-cn]": 5,
            "m": 6,
            "e": 7,
            "r": 8,
            "me": 9,
            "er": 10,
        }
        payload = {
            "normalizer": None,
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "decoder": None,
            "model": {
                "type": "BPE",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "vocab": vocabulary,
                # The final duplicate wins in tokenizers' BPE model. This
                # makes ``e r`` rank before the final ``m e`` record.
                "merges": ["m e", "e r", "m e"],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            tokenizer = XTTS2Tokenizer.from_file(path)

        self.assertEqual(
            tokenizer.encode(
                "mer mer",
                language="en",
                preprocessed=True,
            ),
            [4, 6, 10, 3, 6, 10],
        )
        with self.assertRaisesRegex(ValueError, "numeric"):
            tokenizer.encode("model 2", language="en")
        with self.assertRaisesRegex(ValueError, "transliteration"):
            tokenizer.encode("你好", language="zh")
        self.assertEqual(
            tokenizer.encode(
                "mer",
                language="zh",
                preprocessed=True,
            ),
            [5, 6, 10],
        )

    def test_native_generation_uses_signed_repetition_penalty(self):
        model = _tiny_gpt().eval()

        def logits(_prefix, generated):
            value = torch.full((generated.shape[0], 18), -4.0)
            value[:, 0] = -0.9
            value[:, model.start_audio_token] = -1.0
            return value

        model.autoregressive_step = logits
        generated = model.generate(
            torch.randn(1, 2, 32),
            torch.tensor([[2, 3]]),
            max_new_tokens=1,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            repetition_penalty=2.0,
        )
        self.assertEqual(generated.tolist(), [[0]])

    def test_safetensors_inventory_and_strict_namespace(self):
        source = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
        target = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
        with tempfile.TemporaryDirectory() as directory:
            path = save_xtts2_checkpoint(
                source,
                Path(directory) / "model.safetensors",
            )
            inventory = inspect_xtts2_checkpoint(path)
            self.assertEqual(inventory.tensor_count, len(source.state_dict()))
            self.assertEqual(len(inventory.header_fingerprint), 64)
            load_xtts2_checkpoint(target, path)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[name])

    def test_safetensors_loader_materializes_a_meta_graph(self):
        source = nn.Linear(3, 4)
        with torch.device("meta"):
            target = nn.Linear(3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = save_xtts2_checkpoint(
                source,
                Path(directory) / "model.safetensors",
            )
            load_xtts2_checkpoint(target, path)
        self.assertFalse(any(value.is_meta for value in target.state_dict().values()))
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[name])

    def test_configuration_rejects_invalid_generation_and_conditioning_values(self):
        with self.assertRaisesRegex(ValueError, "top_p"):
            XTTS2Config.from_mapping({"top_p": 0.0})
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            XTTS2Config.from_mapping({
                "gpt_cond_len": 2,
                "gpt_cond_chunk_len": 3,
            })
        with self.assertRaisesRegex(ValueError, "start- and stop-audio"):
            XTTS2Config.from_mapping({
                "model_args": {
                    "gpt_start_audio_token": 1,
                    "gpt_stop_audio_token": 1,
                },
            })

    def test_legacy_conversion_is_explicitly_trusted(self):
        with self.assertRaises(PermissionError):
            convert_trusted_legacy_xtts2_checkpoint(
                "model.pth",
                "model.safetensors",
            )

    def test_provenance_distinguishes_code_and_weight_licenses(self):
        root = RUNTIME_ROOTS[0]
        source = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
        self.assertEqual(source["implementation_license"], "MPL-2.0")
        self.assertEqual(source["model_license"], "Coqui Public Model License")
        self.assertTrue((root / "THIRD_PARTY_LICENSE").is_file())


if __name__ == "__main__":
    unittest.main()
