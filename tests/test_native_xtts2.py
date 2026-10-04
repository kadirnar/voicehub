from __future__ import annotations

import ast
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from voicehub.architectures.xtts2.checkpoint import (
    convert_trusted_legacy_xtts2_checkpoint,
    inspect_xtts2_checkpoint,
    load_xtts2_checkpoint,
    save_xtts2_checkpoint,
)
from voicehub.architectures.xtts2.configuration import XTTS2Config
from voicehub.architectures.xtts2.gpt import XTTS2GPT
from voicehub.architectures.xtts2.tokenizer import XTTS2Tokenizer
from voicehub.models.xtts import XTTSForTextToSpeech

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = (
    PROJECT_ROOT / "voicehub" / "architectures" / "xtts2",
    PROJECT_ROOT / "voicehub" / "models" / "xtts_native",
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
                    "import voicehub.architectures.xtts2; "
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

    def test_tokenizer_keeps_normalized_text_without_unicode_folding(self):
        vocabulary = {
            "[UNK]": 0,
            "[START]": 1,
            "[STOP]": 2,
            "[SPACE]": 3,
            "[en]": 4,
            "m": 5,
            "e": 6,
            "r": 7,
            "er": 8,
            "\uff4d": 9,  # FULLWIDTH LATIN SMALL LETTER M (NFKC -> "m")
            "\u00b2": 10,  # SUPERSCRIPT TWO (NFKC -> "2")
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
                "merges": ["e r"],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            tokenizer = XTTS2Tokenizer.from_file(path)

        # Like the source tokenizer (no normalizer), pre-normalized text is
        # tokenized as given; compatibility characters are not NFKC-folded.
        self.assertEqual(
            tokenizer.encode(
                " \uff4der  \u00b2 ",
                language="en",
                preprocessed=True,
            ),
            [4, 9, 8, 3, 10],
        )

    def test_native_generation_uses_signed_repetition_penalty(self):
        model = _tiny_gpt().eval()

        class Logits(nn.Module):

            def forward(self, hidden):
                value = torch.full((*hidden.shape[:-1], 18), -4.0)
                value[..., 0] = -0.9
                value[..., model.start_audio_token] = -1.0
                return value

        model.mel_head = Logits()
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

    def test_native_generation_penalizes_source_placeholder_code(self):
        # Coqui feeds placeholder id 1 for every prefix position to Hugging
        # Face generate, so the repetition penalty also covers code 1.
        model = _tiny_gpt().eval()

        class Logits(nn.Module):

            def forward(self, hidden):
                value = torch.full((*hidden.shape[:-1], 18), -4.0)
                value[..., 1] = 1.0
                value[..., 0] = 0.6
                return value

        model.mel_head = Logits()
        generated = model.generate(
            torch.randn(1, 2, 32),
            torch.tensor([[2, 3]]),
            max_new_tokens=1,
            do_sample=False,
            repetition_penalty=2.0,
        )
        self.assertEqual(generated.tolist(), [[0]])

    def test_cached_generation_matches_full_recomputation(self):
        torch.manual_seed(0)
        model = _tiny_gpt().eval()
        conditioning = torch.randn(1, 3, 32)
        text = torch.tensor([[2, 3, 4]])
        generated = model.generate(
            conditioning,
            text,
            max_new_tokens=8,
            do_sample=False,
            repetition_penalty=1.0,
        )
        padded = F.pad(text, (1, 1), value=model.stop_text_token)
        padded[:, 0] = model.start_text_token
        prefix = torch.cat((conditioning, model.text_embedding(padded) + model.text_pos_embedding(padded)),
                           dim=1)
        expected = torch.full((1, 1), model.start_audio_token)
        with torch.no_grad():
            for _ in range(generated.shape[1]):
                token = model.autoregressive_step(prefix, expected).argmax(dim=-1, keepdim=True)
                expected = torch.cat((expected, token), dim=1)
        self.assertEqual(generated.tolist(), expected[:, 1:].tolist())

    def test_sampling_filters_follow_hugging_face_warpers(self):
        try:
            from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
        except ImportError:  # pragma: no cover - transformers is a test extra
            self.skipTest("transformers is not installed")
        from voicehub.architectures.xtts2.gpt import _filter_logits

        generator = torch.Generator().manual_seed(0)
        for _ in range(20):
            logits = torch.randn(3, 1026, generator=generator) * 4
            logits[:, :8] = logits[:, :1]  # ties at the boundary
            expected = TopPLogitsWarper(0.85)(None, TopKLogitsWarper(50)(None, logits.clone()))
            torch.testing.assert_close(_filter_logits(logits, top_k=50, top_p=0.85), expected, rtol=0, atol=0)

    def test_synthesis_decodes_one_latent_per_generated_code(self):
        from types import SimpleNamespace

        from voicehub.architectures.xtts2.modeling import XTTS2Model

        torch.manual_seed(0)
        gpt = _tiny_gpt().eval()
        codes = torch.tensor([[3, 4, 5, 6, gpt.stop_audio_token]])
        gpt.generate = lambda *args, **kwargs: codes
        runtime = SimpleNamespace(gpt=gpt, hifigan_decoder=lambda latents, g: latents)
        latents = XTTS2Model.synthesize_tokens(
            runtime,
            torch.tensor([[2, 3]]),
            torch.randn(1, 3, 32),
            torch.randn(1, 8, 1),
        )
        self.assertEqual(latents.shape[1], codes.shape[1])
        # Speeds are clamped at 0.05 like the source (no division by zero).
        slow = XTTS2Model.synthesize_tokens(
            runtime,
            torch.tensor([[2, 3]]),
            torch.randn(1, 3, 32),
            torch.randn(1, 8, 1),
            speed=0.0,
        )
        self.assertEqual(slow.shape[1], codes.shape[1] * 20)

    def test_tokenizer_matches_source_multilingual_cleaners(self):
        cases = (
            (
                "en", 'Mr. Smith & Dr. "Jones" met at St. Paul\'s.',
                "mister smith and doctor jones met at saint paul's."),
            # The source lowercases before its Turkish capital replacements.
            ("tr", "İstanbul'da Dr. Öz", "i\u0307stanbul'da doktor öz"),
            ("hi", 'नमस्ते "दोस्त"  & आप', 'नमस्ते "दोस्त" & आप'),
            ("ru", "Г-н Петров и д-р Иванов", "господин петров и доктор иванов"),
        )
        for language, text, expected in cases:
            with self.subTest(language=language):
                self.assertEqual(
                    XTTS2Tokenizer._preprocess_text(text, language=language, preprocessed=False),
                    expected,
                )

    def test_tokenizer_pretokenizes_combining_marks_like_onig_word_class(self):
        from voicehub.architectures.xtts2.tokenizer import _pretokenize

        self.assertEqual(_pretokenize("नमस्ते,"), ["नमस्ते", ","])
        self.assertEqual(_pretokenize("مَرْحَبًا"), ["مَرْحَبًا"])
        self.assertEqual(_pretokenize("i\u0307stanbul'da"), ["i\u0307stanbul", "'", "da"])
        self.assertEqual(_pretokenize("a_b...c"), ["a_b", "...", "c"])

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

    def test_legacy_conversion_reads_published_coqui_payload(self):
        # The published model.pth pickles Coqui config objects next to the
        # weights and stores BatchNorm step counters as int64 tensors.
        import types

        module_name = "TTS.tts.configs.xtts_config"
        fake_module = types.ModuleType(module_name)
        fake_config = type("XttsConfig", (), {"__module__": module_name})
        fake_module.XttsConfig = fake_config
        source = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2))
        payload = {
            "config": fake_config(),
            "model": {
                "xtts." + name: value
                for name, value in source.state_dict().items()
            },
        }
        payload["config"].temperature = 0.75
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "model.pth"
            previous = {
                name: sys.modules.get(name)
                for name in ("TTS", "TTS.tts", "TTS.tts.configs", module_name)
            }
            try:
                for name in previous:
                    sys.modules[name] = fake_module if name == module_name else types.ModuleType(name)
                torch.save(payload, legacy)
            finally:
                for name, value in previous.items():
                    if value is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = value
            converted = convert_trusted_legacy_xtts2_checkpoint(
                legacy,
                Path(directory) / "model.safetensors",
                trust_legacy_pickle=True,
            )
            target = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2))
            load_xtts2_checkpoint(target, converted, dtype=torch.float16)
            exported = save_xtts2_checkpoint(target, Path(directory) / "export.safetensors")
            self.assertEqual(inspect_xtts2_checkpoint(exported).tensor_count, len(source.state_dict()))
        counter = target.state_dict()["1.num_batches_tracked"]
        self.assertEqual(counter.dtype, torch.int64)
        self.assertEqual(target.state_dict()["0.weight"].dtype, torch.float16)
        self.assertNotIn("TTS", sys.modules)

    def test_provenance_distinguishes_code_and_weight_licenses(self):
        root = RUNTIME_ROOTS[0]
        source = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
        self.assertEqual(source["implementation_license"], "MPL-2.0")
        self.assertEqual(source["model_license"], "Coqui Public Model License")
        self.assertTrue((root / "THIRD_PARTY_LICENSE").is_file())


class XTTSLegacyCheckpointLoadingTests(unittest.TestCase):
    """The published repository ships only a legacy ``model.pth``."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.voicehub_cache = self.root / "voicehub-cache"
        environment = patch.dict(os.environ, {"VOICEHUB_CACHE": str(self.voicehub_cache)})
        environment.start()
        self.addCleanup(environment.stop)
        self.state = {"gpt.text_head.weight": torch.randn(3, 2)}
        self.repository = self.root / "repository"
        self.repository.mkdir()
        (self.repository / "config.json").write_text("{}", encoding="utf-8")
        (self.repository / "vocab.json").write_text("{}", encoding="utf-8")
        torch.save(
            {"model": {
                "xtts." + name: value
                for name, value in self.state.items()
            }},
            self.repository / "model.pth",
        )
        (self.repository / "dvae.pth").write_bytes(b"unused by inference")

    def _mock_hub(self):
        """Serve ``self.repository`` as a Hub repository, recording fetches."""
        fetched = []

        def download_file(repo_id, filename, *, subfolder="", **kwargs):
            del repo_id, kwargs
            path = self.repository / subfolder / filename
            if not path.is_file():
                raise FileNotFoundError(f"Could not find the requested Hub file: {filename}.")
            fetched.append(f"{subfolder}/{filename}" if subfolder else filename)
            return path

        def download_snapshot(repo_id, **kwargs):
            del repo_id, kwargs
            fetched.extend(path.name for path in self.repository.iterdir())
            return self.repository

        file_patch = patch("voicehub.hub.download_hugging_face_file", side_effect=download_file)
        snapshot_patch = patch(
            "voicehub.models._shared.download_hugging_face_snapshot",
            side_effect=download_snapshot,
        )
        file_patch.start()
        snapshot_patch.start()
        self.addCleanup(file_patch.stop)
        self.addCleanup(snapshot_patch.stop)
        return fetched

    def _make_read_only(self):
        paths = [self.repository, *self.repository.iterdir()]
        for path in paths:
            path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        self.addCleanup(lambda: [path.chmod(path.stat().st_mode | stat.S_IWUSR) for path in paths])

    def test_untrusted_hub_checkpoint_fails_before_downloading_pickles(self):
        fetched = self._mock_hub()
        model = XTTSForTextToSpeech(model_path="acme/xtts", device="cpu")

        with self.assertRaises(PermissionError) as raised:
            model._load_pretrained_model()

        message = str(raised.exception)
        self.assertIn("convert_trusted_legacy_xtts2_checkpoint", message)
        self.assertIn("trust_pickle_checkpoint=True", message)
        self.assertNotIn("model.pth", fetched)
        self.assertNotIn("dvae.pth", fetched)

    def test_untrusted_local_checkpoint_names_the_conversion_options(self):
        model = XTTSForTextToSpeech(model_path=self.repository, device="cpu")

        with self.assertRaisesRegex(PermissionError, "convert_trusted_legacy_xtts2_checkpoint"):
            model._load_pretrained_model()
        self.assertFalse(self.voicehub_cache.exists())

    def test_trusted_hub_checkpoint_converts_outside_the_snapshot(self):
        fetched = self._mock_hub()
        self._make_read_only()
        before = sorted(path.name for path in self.repository.iterdir())
        model = XTTSForTextToSpeech(
            model_path="acme/xtts",
            device="cpu",
            trust_pickle_checkpoint=True,
        )

        directory = model._resolve_artifact_directory()
        converted = model._convert_legacy_checkpoint(directory / "model.pth")

        self.assertEqual(directory, self.repository)
        self.assertIn("model.pth", fetched)
        self.assertNotIn("dvae.pth", fetched)
        self.assertEqual(sorted(path.name for path in self.repository.iterdir()), before)
        self.assertTrue(converted.resolve().is_relative_to(self.voicehub_cache.resolve()))
        self.assertEqual(converted, model._convert_legacy_checkpoint(directory / "model.pth"))
        target = nn.Linear(2, 3, bias=False)
        target.weight.data.zero_()
        load_xtts2_checkpoint(nn.ModuleDict({"gpt": nn.ModuleDict({"text_head": target})}), converted)
        torch.testing.assert_close(target.weight, self.state["gpt.text_head.weight"])

    def test_trusted_hub_checkpoint_uses_one_complete_cache_snapshot(self):
        # Offline, config.json may come from VoiceHub's cache while the
        # other files come from a huggingface_hub snapshot.
        partial = self.root / "partial"
        partial.mkdir()
        (partial / "config.json").write_text("{}", encoding="utf-8")
        self._mock_hub()

        def download_file(repo_id, filename, *, subfolder="", **kwargs):
            if filename == "config.json":
                return partial / filename
            path = self.repository / subfolder / filename
            if not path.is_file():
                raise FileNotFoundError(f"Could not find the requested Hub file: {filename}.")
            return path

        model = XTTSForTextToSpeech(
            model_path="acme/xtts",
            device="cpu",
            trust_pickle_checkpoint=True,
        )
        with patch("voicehub.hub.download_hugging_face_file", side_effect=download_file):
            self.assertEqual(model._resolve_artifact_directory(), self.repository)
            (self.repository / "config.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "different cache snapshots"):
                model._resolve_artifact_directory()

    def test_configuration_rejects_non_boolean_trust(self):
        with self.assertRaisesRegex(TypeError, "trust_pickle_checkpoint"):
            XTTSForTextToSpeech(device="cpu", trust_pickle_checkpoint="yes")


if __name__ == "__main__":
    unittest.main()
