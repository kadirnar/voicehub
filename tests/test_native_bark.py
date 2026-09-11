import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(TORCH_AVAILABLE, "Native Bark requires PyTorch")
class NativeBarkTests(unittest.TestCase):

    @staticmethod
    def _tiny_config():
        from voicehub.components.audio.codecs.encodec import EncodecConfig
        from voicehub.models.bark.native.configuration import (
            BarkArchitectureConfig,
            BarkCoarseConfig,
            BarkCoarseGenerationConfig,
            BarkFineConfig,
            BarkFineGenerationConfig,
            BarkGenerationConfig,
            BarkSemanticConfig,
            BarkSemanticGenerationConfig,
        )

        architecture = BarkArchitectureConfig(
            semantic=BarkSemanticConfig(
                block_size=16,
                input_vocab_size=32,
                output_vocab_size=12,
                num_layers=1,
                num_heads=2,
                hidden_size=8,
                bias=False,
            ),
            coarse=BarkCoarseConfig(
                block_size=16,
                input_vocab_size=32,
                output_vocab_size=32,
                num_layers=1,
                num_heads=2,
                hidden_size=8,
                bias=False,
            ),
            fine=BarkFineConfig(
                block_size=16,
                input_vocab_size=12,
                output_vocab_size=12,
                num_layers=1,
                num_heads=2,
                hidden_size=8,
                bias=False,
                n_codes_total=4,
                n_codes_given=1,
            ),
            codec=EncodecConfig(
                target_bandwidths=(1.0, ),
                sample_rate=8_000,
                channels=1,
                dimension=4,
                n_filters=2,
                n_residual_layers=0,
                ratios=(2, ),
                kernel_size=1,
                last_kernel_size=1,
                residual_kernel_size=1,
                compress=1,
                lstm=0,
                bins=8,
                n_q=4,
                name="bark_test_codec",
            ),
        )
        generation = BarkGenerationConfig(
            sample_rate=8_000,
            codebook_size=8,
            semantic=BarkSemanticGenerationConfig(
                max_input_semantic_length=4,
                max_new_tokens=4,
                semantic_infer_token=31,
                semantic_pad_token=10,
                semantic_rate_hz=50.0,
                semantic_vocab_size=10,
                text_encoding_offset=12,
                text_pad_token=30,
                eos_token_id=10,
            ),
            coarse=BarkCoarseGenerationConfig(
                coarse_infer_token=31,
                coarse_rate_hz=75.0,
                coarse_semantic_pad_token=30,
                max_coarse_history=4,
                max_coarse_input_length=4,
                n_coarse_codebooks=2,
                sliding_window_len=4,
            ),
            fine=BarkFineGenerationConfig(
                max_fine_history_length=4,
                max_fine_input_length=16,
                n_fine_codebooks=4,
                temperature=None,
            ),
        )
        return architecture, generation

    def test_pinned_graph_matches_exact_remote_archive_inventory(self):
        import torch

        from voicehub.models.bark.native.checkpoint import (
            provider_state_dict,
            tensor_inventory_fingerprint,
            verify_native_graph_contract,
        )
        from voicehub.models.bark.native.configuration import BarkArchitectureConfig
        from voicehub.models.bark.native.metadata import (
            BARK_INVENTORY_FINGERPRINT,
            BARK_STATE_VALUES,
            BARK_TENSOR_COUNT,
        )
        from voicehub.models.bark.native.modeling import BarkModel

        with torch.device("meta"):
            model = BarkModel(BarkArchitectureConfig())
        state = provider_state_dict(model)

        self.assertEqual(len(state), BARK_TENSOR_COUNT)
        self.assertEqual(
            sum(tensor.numel() for tensor in state.values()),
            BARK_STATE_VALUES,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(state),
            BARK_INVENTORY_FINGERPRINT,
        )
        self.assertIn(
            "codec_model.encoder.layers.0.conv.weight_g",
            state,
        )
        verify_native_graph_contract(model)

    def test_causal_and_fine_objectives_are_differentiable(self):
        import torch

        from voicehub.models.bark.native.modeling import BarkModel
        from voicehub.models.bark.native.training import BarkTrainingModel

        architecture, generation = self._tiny_config()
        model = BarkModel(architecture, generation_config=generation)
        training = BarkTrainingModel.from_model(model)
        causal_ids = torch.tensor([[1, 2, 3, 4]])
        causal_labels = torch.tensor([[1, 2, 3, 4]])
        fine_ids = torch.randint(0, 8, (1, 4, 4))
        fine_labels = torch.tensor([[2, 3, 4, 5]])

        semantic = training.semantic(causal_ids, labels=causal_labels)
        fine = training.fine(
            fine_ids,
            labels=fine_labels,
            codebook_idx=2,
        )
        loss = semantic["loss"] + fine["loss"]
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.semantic.lm_head.weight.grad)
        self.assertIsNotNone(model.fine_acoustics.input_embeds_layers[2].weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.codec_model.parameters()))

    def test_causal_cache_matches_full_prefix_and_coarse_ranges_alternate(self):
        import torch

        from voicehub.models.bark.native.modeling import BarkModel

        architecture, generation = self._tiny_config()
        model = BarkModel(architecture, generation_config=generation).eval()
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            full = model.semantic(tokens, use_cache=False).logits[:, -1]
            prefix = model.semantic(tokens[:, :3], use_cache=True)
            incremental = model.semantic(
                tokens[:, 3:],
                past_key_values=prefix.past_key_values,
                use_cache=True,
            ).logits[:, -1]
        torch.testing.assert_close(incremental, full)

        with torch.no_grad():
            for parameter in model.coarse_acoustics.parameters():
                parameter.zero_()
            generated = model.coarse_acoustics._autoregressive_generate(
                torch.tensor([[1, 2]]),
                max_new_tokens=4,
                do_sample=False,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                alternating_ranges=((8, 12), (12, 16)),
            )
        self.assertEqual(generated[0, -4:].tolist(), [8, 12, 8, 12])

    def test_safe_export_reconstructs_config_and_exact_state(self):
        import torch

        from voicehub.models.bark.native.checkpoint import (
            load_bark_model_from_safetensors,
            provider_state_dict,
            save_bark_safetensors,
        )
        from voicehub.models.bark.native.modeling import BarkModel

        torch.manual_seed(9)
        architecture, generation = self._tiny_config()
        model = BarkModel(architecture, generation_config=generation).eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_bark_safetensors(
                model,
                Path(directory) / "model.safetensors",
            )
            restored = load_bark_model_from_safetensors(checkpoint).eval()

        self.assertEqual(
            restored.config.to_dict(),
            model.config.to_dict(),
        )
        self.assertEqual(
            restored.generation_config.to_dict(),
            model.generation_config.to_dict(),
        )
        expected = provider_state_dict(model)
        actual = provider_state_dict(restored)
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])

    def test_tiny_end_to_end_generation_decodes_with_native_encodec(self):
        import torch

        from voicehub.models.bark.native.modeling import BarkModel

        architecture, generation = self._tiny_config()
        model = BarkModel(
            architecture,
            generation_config=generation,
        ).eval()
        with torch.no_grad():
            audio, lengths = model.generate(
                torch.tensor([[1, 2, 0, 0]]),
                attention_mask=torch.tensor([[1, 1, 0, 0]]),
                semantic_do_sample=False,
                coarse_do_sample=False,
                fine_temperature=None,
                return_output_lengths=True,
            )

        self.assertEqual(audio.ndim, 2)
        self.assertEqual(audio.shape[0], 1)
        self.assertEqual(len(lengths), 1)
        self.assertEqual(lengths[0], audio.shape[1])
        self.assertTrue(torch.isfinite(audio).all())

    def test_safe_loader_rejects_incomplete_namespace(self):
        import torch

        from voicehub.checkpointing import save_safetensors
        from voicehub.models.bark.native.checkpoint import load_bark_safetensors, provider_state_dict
        from voicehub.models.bark.native.modeling import BarkModel

        architecture, generation = self._tiny_config()
        model = BarkModel(architecture, generation_config=generation)
        state = provider_state_dict(model)
        state.pop(next(iter(state)))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_safetensors(
                state,
                Path(directory) / "broken.safetensors",
            )
            with self.assertRaisesRegex(ValueError, "namespace mismatch"):
                load_bark_safetensors(model, checkpoint)

    def test_wordpiece_and_numpy_prompt_loading_need_no_provider(self):
        import torch

        from voicehub.models.bark.native.processing import BarkProcessor, BarkWordPieceTokenizer, _read_npy_integer

        tokenizer = BarkWordPieceTokenizer([
            "[PAD]",
            "[UNK]",
            "Hello",
            ",",
            "world",
            "##s",
            "你",
        ])
        self.assertEqual(
            tokenizer.tokenize("Hello, worlds 你"),
            ["Hello", ",", "world", "##s", "你"],
        )
        ids, mask = tokenizer.encode("Hello", max_length=4)
        self.assertEqual(ids, [2, 0, 0, 0])
        self.assertEqual(mask, [1, 0, 0, 0])

        values = [1, 2, 3, 4, 5, 6]
        header = (b"{'descr': '<i8', 'fortran_order': False, "
                  b"'shape': (2, 3), }")
        padding = 16 - ((10 + len(header) + 1) % 16)
        header += b" " * padding + b"\n"
        payload = (
            b"\x93NUMPY" + bytes(
                (1, 0)) + struct.pack("<H", len(header)) + header + struct.pack("<6q", *values))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.npy"
            path.write_bytes(payload)
            tensor = _read_npy_integer(path)
        torch.testing.assert_close(
            tensor,
            torch.tensor(values).reshape(2, 3),
        )

        processor = BarkProcessor(
            tokenizer,
            speaker_embeddings={},
        )
        encoded = processor(text="Hello", max_length=4)
        self.assertEqual(tuple(encoded["input_ids"].shape), (1, 4))

    def test_voice_preset_rejects_path_traversal(self):
        from voicehub.models.bark.native.processing import BarkProcessor, BarkWordPieceTokenizer

        processor = BarkProcessor(
            BarkWordPieceTokenizer(["[PAD]", "[UNK]", "hello"]),
            speaker_embeddings={
                "bad": {
                    "semantic_prompt": "../semantic.npy",
                    "coarse_prompt": "coarse.npy",
                    "fine_prompt": "fine.npy",
                },
            },
            speaker_source="some/repository",
        )
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            processor.load_voice_preset("bad")

    def test_legacy_conversion_requires_explicit_trust_before_reading(self):
        from voicehub.models.bark.native.checkpoint import convert_official_bark_checkpoint

        architecture, generation = self._tiny_config()
        with self.assertRaisesRegex(PermissionError, "trust_official_pickle"):
            convert_official_bark_checkpoint(
                "does-not-exist.bin",
                "unused.safetensors",
                config=architecture,
                generation_config=generation,
            )

    def test_architecture_spec_is_native_and_truthful_about_training(self):
        from voicehub.models.bark.native.registration import create_bark_architecture_spec
        from voicehub.registry import get_model_spec
        from voicehub.training.contracts import TrainingSupport
        from voicehub.training.specs import get_training_spec

        spec = create_bark_architecture_spec()
        model_spec = get_model_spec("bark")
        training_spec = get_training_spec("bark")
        self.assertEqual(spec.architecture_id, "bark")
        self.assertTrue(spec.capabilities.training)
        self.assertIn("safetensors", spec.capabilities.checkpoint_formats)
        self.assertFalse(spec.metadata["official_safetensors_published"])
        self.assertFalse(spec.metadata["raw_audio_finetuning_ready"])
        self.assertEqual(
            spec.metadata["training_scope"],
            "pretokenized-stage-specific",
        )
        self.assertTrue(model_spec.is_voicehub_native)
        self.assertEqual(model_spec.architecture, "bark")
        self.assertTrue(training_spec.native_training)
        self.assertIs(training_spec.support, TrainingSupport.PREPROCESSED)
        self.assertFalse(
            any(entrypoint.startswith("transformers") for entrypoint in training_spec.source_entrypoints))


class NativeBarkImportTests(unittest.TestCase):

    def test_public_import_is_lazy_and_never_imports_transformers(self):
        script = (
            "import sys; import voicehub.models.bark; "
            "print('torch' in sys.modules, 'transformers' in sys.modules)")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "False False")

    def test_native_runtime_has_no_external_model_runtime_imports(self):
        files = [
            PROJECT_ROOT / "voicehub/models/bark/native" / name for name in (
                "artifacts.py",
                "checkpoint.py",
                "configuration.py",
                "modeling.py",
                "processing.py",
                "training.py",
            )
        ]
        source = "\n".join(path.read_text(encoding="utf-8") for path in files)
        self.assertNotIn("import transformers", source)
        self.assertNotIn("from transformers", source)
        self.assertNotIn("import numpy", source)
        self.assertNotIn("from encodec", source)

    def test_source_manifest_is_pinned(self):
        source = json.loads(
            (PROJECT_ROOT / "voicehub/models/bark/native/SOURCE.json").read_text(encoding="utf-8"))
        self.assertEqual(
            source["checkpoint"]["revision"],
            "1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd",
        )
        self.assertEqual(source["checkpoint"]["license"], "MIT")
        self.assertIn(
            "no Safetensors",
            " ".join(source["notes"]),
        )


if __name__ == "__main__":
    unittest.main()
