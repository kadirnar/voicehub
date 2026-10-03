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
        from voicehub.architectures.bark.configuration import (
            BarkArchitectureConfig,
            BarkCoarseConfig,
            BarkCoarseGenerationConfig,
            BarkFineConfig,
            BarkFineGenerationConfig,
            BarkGenerationConfig,
            BarkSemanticConfig,
            BarkSemanticGenerationConfig,
        )
        from voicehub.components.audio.codecs.encodec import EncodecConfig

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

        from voicehub.architectures.bark.checkpoint import (
            provider_state_dict,
            tensor_inventory_fingerprint,
            verify_native_graph_contract,
        )
        from voicehub.architectures.bark.configuration import BarkArchitectureConfig
        from voicehub.architectures.bark.metadata import (
            BARK_INVENTORY_FINGERPRINT,
            BARK_STATE_VALUES,
            BARK_TENSOR_COUNT,
        )
        from voicehub.architectures.bark.modeling import BarkModel

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

        from voicehub.architectures.bark.modeling import BarkModel
        from voicehub.architectures.bark.training import BarkTrainingModel

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

        from voicehub.architectures.bark.modeling import BarkModel

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

    def test_attention_uses_fused_sdpa_like_upstream_and_matches_eager(self):
        from unittest import mock

        import torch

        from voicehub.architectures.bark import modeling

        architecture, generation = self._tiny_config()
        torch.manual_seed(0)
        model = modeling.BarkModel(architecture, generation_config=generation).eval()
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        fine_tokens = torch.randint(0, 12, (1, 6, 4))
        sdpa = torch.nn.functional.scaled_dot_product_attention
        calls = []

        def record(query, key, value, **kwargs):
            calls.append((
                query.shape[-2],
                key.shape[-2],
                kwargs["attn_mask"] is not None,
                kwargs["is_causal"],
            ))
            return sdpa(query, key, value, **kwargs)

        with torch.no_grad():
            # Requesting attention weights keeps the explicit softmax path.
            eager = model.semantic(tokens, output_attentions=True).logits
            eager_fine = model.fine_acoustics(
                fine_tokens,
                codebook_idx=2,
                output_attentions=True,
            ).logits
            self.assertEqual(calls, [])
            with mock.patch.object(
                    modeling.F,
                    "scaled_dot_product_attention",
                    side_effect=record,
            ):
                full = model.semantic(tokens, use_cache=False).logits
                prefix = model.semantic(tokens[:, :2], use_cache=True)
                chunk = model.semantic(
                    tokens[:, 2:5],
                    past_key_values=prefix.past_key_values,
                    use_cache=True,
                )
                step = model.semantic(
                    tokens[:, 5:],
                    past_key_values=chunk.past_key_values,
                    use_cache=True,
                )
                fine = model.fine_acoustics(fine_tokens, codebook_idx=2).logits
        self.assertEqual(
            calls,
            [
                (6, 6, False, True),  # uncached prefill: top-left causal
                (2, 2, False, True),
                (3, 5, True, False),  # cached chunk: explicit causal slice
                (1, 6, False, False),  # one-token decode: every cached key
                (6, 6, False, False),  # fine stage is bidirectional
            ],
        )
        torch.testing.assert_close(full, eager)
        torch.testing.assert_close(
            torch.cat((prefix.logits, chunk.logits, step.logits), dim=1),
            eager,
        )
        torch.testing.assert_close(fine, eager_fine)

    def test_sampling_follows_upstream_filter_order_and_eos_rule(self):
        import torch

        from voicehub.architectures.bark.modeling import _filter_logits, _split_options

        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
        # Upstream applies top-p to the unscaled distribution (p = .64, .24,
        # .09, .03): top_p=0.8 keeps two tokens at any temperature.
        filtered = _filter_logits(logits, temperature=0.1, top_k=0, top_p=0.8)
        self.assertEqual(torch.isfinite(filtered).sum().item(), 2)
        torch.testing.assert_close(filtered[0, :2], logits[0, :2] / 0.1)

        semantic, coarse, fine = _split_options({
            "temperature": 0.9,
            "min_eos_p": 0.3,
            "fine_temperature": 0.4,
        })
        self.assertEqual(semantic, {"temperature": 0.9, "min_eos_p": 0.3})
        self.assertEqual(coarse, {"temperature": 0.9})
        # The fine stage keeps its own temperature, as in generate_audio().
        self.assertEqual(fine, {"temperature": 0.4})
        self.assertEqual(_split_options({"temperature": 0.9})[2], {})

    def test_min_eos_p_uses_the_temperature_scaled_distribution(self):
        import torch

        from voicehub.architectures.bark.modeling import BarkModel

        architecture, generation = self._tiny_config()
        model = BarkModel(architecture, generation_config=generation).eval()
        semantic = model.semantic
        # EOS (id 10) has probability ~0.27 at temperature 1 but ~0.12 at
        # temperature 0.5: upstream keeps sampling at 0.5, whereas a check
        # on the unscaled logits would stop immediately.
        bias = torch.full((12, ), -30.0)
        bias[3] = 1.0
        bias[10] = 0.0
        with torch.no_grad():
            for parameter in semantic.parameters():
                parameter.zero_()
            semantic.lm_head = torch.nn.Linear(8, 12, bias=True)
            semantic.lm_head.weight.zero_()
            semantic.lm_head.bias.copy_(bias)

            def run(temperature):
                return semantic._autoregressive_generate(
                    torch.tensor([[1]]),
                    max_new_tokens=3,
                    do_sample=False,
                    temperature=temperature,
                    top_k=0,
                    top_p=1.0,
                    eos_token_id=10,
                    min_eos_p=0.2,
                    allowed_token_range=(0, 11),
                )[0, 1:].tolist()

            self.assertEqual(run(0.5), [3, 3, 3])
            self.assertEqual(run(1.0), [10])

    def test_sampling_draws_within_the_permitted_slice_like_upstream(self):
        import torch

        from voicehub.architectures.bark.modeling import BarkModel

        architecture, generation = self._tiny_config()
        torch.manual_seed(5)
        model = BarkModel(architecture, generation_config=generation).eval()
        coarse = model.coarse_acoustics
        prefix = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            logits = coarse(prefix).logits[:, -1, 12:16].float()
            torch.manual_seed(11)
            expected = torch.multinomial(torch.softmax(logits / 0.7, dim=-1), 1)[0, 0] + 12
            torch.manual_seed(11)
            generated = coarse._autoregressive_generate(
                prefix,
                max_new_tokens=1,
                do_sample=True,
                temperature=0.7,
                top_k=0,
                top_p=1.0,
                alternating_ranges=((12, 16), (16, 20)),
            )
        # Upstream samples from the codebook's logit slice; drawing from the
        # masked full vocabulary would consume different random numbers.
        self.assertEqual(generated[0, -1].item(), expected.item())

    def test_fine_temperature_one_samples_instead_of_argmax(self):
        import torch

        from voicehub.architectures.bark.modeling import BarkModel

        architecture, generation = self._tiny_config()
        torch.manual_seed(0)
        model = BarkModel(architecture, generation_config=generation).eval()
        coarse = torch.tensor([[10, 18, 11, 19, 12, 20, 13, 21]])
        calls = []
        original = torch.multinomial

        def record(*args, **kwargs):
            calls.append(args[0].shape)
            return original(*args, **kwargs)

        torch.multinomial = record
        try:
            model.fine_acoustics.generate(
                coarse,
                semantic_config=generation.semantic,
                coarse_config=generation.coarse,
                generation_config=generation.fine,
                codebook_size=generation.codebook_size,
                temperature=1.0,
            )
        finally:
            torch.multinomial = original
        self.assertTrue(calls)

    def test_transformers_generation_defaults_do_not_override_upstream_sampling(self):
        from voicehub.architectures.bark.configuration import BarkGenerationConfig

        published = {
            "sample_rate": 24_000,
            "codebook_size": 1024,
            "semantic_config": {
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 1.0,
                "transformers_version": "4.31.0.dev0",
            },
            "coarse_acoustics_config": {
                "temperature": 0.7,
                "top_k": 50,
                "transformers_version": "4.31.0.dev0",
            },
            "fine_acoustics_config": {
                "temperature": 0.5,
                "transformers_version": "4.31.0.dev0",
            },
        }
        config = BarkGenerationConfig.from_dict(published)
        self.assertEqual(config.semantic.top_k, 0)
        self.assertEqual(config.semantic.min_eos_p, 0.2)
        self.assertEqual(config.coarse.top_k, 0)
        self.assertEqual(config.fine.temperature, 0.5)
        # VoiceHub's own serialized settings round-trip unchanged.
        config.semantic.top_k = 7
        self.assertEqual(BarkGenerationConfig.from_dict(config.to_dict()).semantic.top_k, 7)

    def test_safe_export_reconstructs_config_and_exact_state(self):
        import torch

        from voicehub.architectures.bark.checkpoint import (
            load_bark_model_from_safetensors,
            provider_state_dict,
            save_bark_safetensors,
        )
        from voicehub.architectures.bark.modeling import BarkModel

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

        from voicehub.architectures.bark.modeling import BarkModel

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

    def _write_tiny_runtime(self, directory):
        from voicehub.architectures.bark.checkpoint import save_bark_safetensors
        from voicehub.architectures.bark.modeling import BarkModel

        architecture, generation = self._tiny_config()
        model = BarkModel(architecture, generation_config=generation).eval()
        root = Path(directory)
        save_bark_safetensors(model, root / "model.safetensors")
        (root / "config.json").write_text(json.dumps(architecture.to_dict()))
        (root / "generation_config.json").write_text(json.dumps(generation.to_dict()))
        (root / "tokenizer.json").write_text("{}")
        (root / "tokenizer_config.json").write_text("{}")
        (root / "vocab.txt").write_text("[PAD]\n[UNK]\nhello\nworld\n")
        (root / "speaker_embeddings_path.json").write_text("{}")
        return root

    def test_public_generate_can_be_called_repeatedly_after_load(self):
        import torch

        from voicehub.models.bark.inference import BarkForTextToSpeech

        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_tiny_runtime(directory)
            model = BarkForTextToSpeech(
                model_path=root,
                device="cpu",
                verify_official_integrity=False,
            )
            model.load()
            options = {
                "semantic_do_sample": False,
                "coarse_do_sample": False,
                "fine_temperature": 1.0,
            }
            first = model.generate("hello world", seed=1, **options)
            second = model.generate("hello world", seed=1, **options)

        # Loading must not replace the wrapper's text processor, which the
        # shared generate() contract calls positionally on every request.
        self.assertNotIsInstance(model.processor, type(model.transformers_processor))
        torch.testing.assert_close(first.audio, second.audio)

    def test_safe_loader_rejects_incomplete_namespace(self):
        import torch

        from voicehub.architectures.bark.checkpoint import load_bark_safetensors, provider_state_dict
        from voicehub.architectures.bark.modeling import BarkModel
        from voicehub.checkpointing import save_safetensors

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

        from voicehub.architectures.bark.processing import BarkProcessor, BarkWordPieceTokenizer, _read_npy_integer

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

    def test_wordpiece_cleaning_matches_bert_basic_tokenizer(self):
        from voicehub.architectures.bark.processing import BarkWordPieceTokenizer

        tokenizer = BarkWordPieceTokenizer(["[PAD]", "[UNK]", "Hello", ",", "world", "caf", "##é"])
        # Newlines/tabs separate words (they are whitespace, not dropped
        # control characters), as in upstream Bark's BertTokenizer.
        self.assertEqual(
            tokenizer.tokenize("Hello,\nworld\tHello\r\nworld\x00"),
            ["Hello", ",", "world", "Hello", "world"],
        )
        # Decomposed accents are NFC-composed before WordPiece lookup.
        self.assertEqual(tokenizer.tokenize("cafe\u0301"), ["caf", "##é"])
        self.assertEqual(tokenizer.tokenize("caf\u00e9"), ["caf", "##é"])

        special = BarkWordPieceTokenizer(["[PAD]", "[UNK]", "[MASK]", "[", "]", "a", "##b", "MASK"])
        self.assertEqual(special.tokenize("ab[MASK]a"), ["a", "##b", "[MASK]", "a"])
        self.assertEqual(special.tokenize("[ MASK ]"), ["[", "MASK", "]"])

    def test_numpy_prompt_loading_accepts_fortran_order(self):
        import torch

        from voicehub.architectures.bark.processing import _read_npy_integer

        values = torch.arange(6).reshape(2, 3)
        header = b"{'descr': '<i8', 'fortran_order': True, 'shape': (2, 3), }"
        padding = 16 - ((10 + len(header) + 1) % 16)
        header += b" " * padding + b"\n"
        column_major = values.T.contiguous().reshape(-1).tolist()
        prefix = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
        payload = prefix + struct.pack("<6q", *column_major)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coarse_prompt.npy"
            path.write_bytes(payload)
            tensor = _read_npy_integer(path)
        torch.testing.assert_close(tensor, values)

    def test_numpy_prompt_loading_accepts_uint16(self):
        import torch

        from voicehub.architectures.bark.processing import _read_npy_integer, _validate_preset

        header = b"{'descr': '<u2', 'fortran_order': False, 'shape': (3,), }"
        padding = 16 - ((10 + len(header) + 1) % 16)
        header += b" " * padding + b"\n"
        prefix = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
        payload = prefix + struct.pack("<3H", 7, 9999, 512)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantic_prompt.npy"
            path.write_bytes(payload)
            semantic = _read_npy_integer(path)
        preset = _validate_preset({
            "semantic_prompt": semantic,
            "coarse_prompt": torch.zeros(2, 2, dtype=torch.uint16),
            "fine_prompt": torch.zeros(8, 2, dtype=torch.uint16),
        })
        self.assertEqual(preset["semantic_prompt"].tolist(), [7, 9999, 512])
        self.assertEqual(preset["semantic_prompt"].dtype, torch.long)

    def test_voice_preset_rejects_path_traversal(self):
        from voicehub.architectures.bark.processing import BarkProcessor, BarkWordPieceTokenizer

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
        from voicehub.architectures.bark.checkpoint import convert_official_bark_checkpoint

        architecture, generation = self._tiny_config()
        with self.assertRaisesRegex(PermissionError, "trust_official_pickle"):
            convert_official_bark_checkpoint(
                "does-not-exist.bin",
                "unused.safetensors",
                config=architecture,
                generation_config=generation,
            )

    def test_architecture_spec_is_native_and_truthful_about_training(self):
        from voicehub.architectures.bark.registration import create_bark_architecture_spec
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
            PROJECT_ROOT / "voicehub/architectures/bark" / name for name in (
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
            (PROJECT_ROOT / "voicehub/architectures/bark/SOURCE.json").read_text(encoding="utf-8"))
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
