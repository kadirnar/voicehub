from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tests._moonshine_test_utils import tiny_moonshine_config, tiny_tokenizer_document
from voicehub.models.asr_moonshine.native import (
    USEFULSENSORS_MOONSHINE_BASE_HEADER_FINGERPRINT,
    USEFULSENSORS_MOONSHINE_BASE_REVISION,
    USEFULSENSORS_MOONSHINE_TINY_HEADER_FINGERPRINT,
    USEFULSENSORS_MOONSHINE_TINY_REVISION,
    HuggingFaceMoonshineCheckpointAdapter,
    MoonshineConfig,
    MoonshineForConditionalGeneration,
    create_moonshine_architecture_spec,
    huggingface_moonshine_tensor_shapes,
    native_moonshine_tensor_shapes,
    safetensors_header_fingerprint,
)
from voicehub.tokenization import SentencePieceBPETokenizer, TokenizerAssetError


class MoonshineConfigurationAndCheckpointTests(unittest.TestCase):

    def test_official_tiny_and_base_header_inventories_are_exact(self):
        tiny = MoonshineConfig()
        base = MoonshineConfig.from_dict({
            "model_type": "moonshine",
            "vocab_size": 32_768,
            "hidden_size": 416,
            "intermediate_size": 1_664,
            "encoder_num_hidden_layers": 8,
            "decoder_num_hidden_layers": 8,
            "encoder_num_attention_heads": 8,
            "decoder_num_attention_heads": 8,
            "encoder_num_key_value_heads": 8,
            "decoder_num_key_value_heads": 8,
            "partial_rotary_factor": 0.62,
            "max_position_embeddings": 194,
            "pad_head_dim_to_multiple_of": 8,
        })

        tiny_shapes = huggingface_moonshine_tensor_shapes(tiny)
        base_shapes = huggingface_moonshine_tensor_shapes(base)
        self.assertEqual(len(tiny_shapes), 160)
        self.assertEqual(len(base_shapes), 210)
        self.assertEqual(
            safetensors_header_fingerprint(tiny_shapes),
            USEFULSENSORS_MOONSHINE_TINY_HEADER_FINGERPRINT,
        )
        self.assertEqual(
            safetensors_header_fingerprint(base_shapes),
            USEFULSENSORS_MOONSHINE_BASE_HEADER_FINGERPRINT,
        )
        self.assertEqual(
            USEFULSENSORS_MOONSHINE_TINY_REVISION,
            "390624ed33d594443aa4aa221f5b9f283b545b5a",
        )
        self.assertEqual(
            USEFULSENSORS_MOONSHINE_BASE_REVISION,
            "7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf",
        )

    def test_native_state_namespace_matches_official_checkpoint(self):
        config = tiny_moonshine_config()
        model = MoonshineForConditionalGeneration(config)
        state_shapes = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}

        self.assertEqual(
            state_shapes,
            native_moonshine_tensor_shapes(config),
        )
        self.assertIs(
            model.proj_out.weight,
            model.model.decoder.embed_tokens.weight,
        )
        adapter = HuggingFaceMoonshineCheckpointAdapter()
        report, _ = adapter.inspect(
            model,
            model.state_dict(),
            config.to_dict(),
        )
        self.assertTrue(report.is_compatible, report.summary())

    def test_configuration_rejects_graph_changing_variants(self):
        fully_rotary = tiny_moonshine_config(partial_rotary_factor=1.0)
        self.assertEqual(fully_rotary.partial_rotary_factor, 1.0)
        with self.assertRaisesRegex(ValueError, "grouped-query"):
            MoonshineConfig(encoder_num_key_value_heads=4, )
        with self.assertRaisesRegex(ValueError, "default RoPE"):
            MoonshineConfig(rope_scaling={"rope_type": "dynamic", "factor": 2.0}, )
        with self.assertRaisesRegex(ValueError, "is_encoder_decoder"):
            MoonshineConfig.from_dict({"is_encoder_decoder": False})

    def test_architecture_declaration_is_training_capable_and_lazy(self):
        spec = create_moonshine_architecture_spec()

        self.assertEqual(spec.architecture_id, "moonshine")
        self.assertTrue(spec.capabilities.training)
        self.assertEqual(spec.capabilities.checkpoint_formats, ("safetensors", ))
        self.assertEqual(
            spec.checkpoint_adapter.path,
            "voicehub.models.asr_moonshine.native.checkpoint:"
            "HuggingFaceMoonshineCheckpointAdapter",
        )
        self.assertTrue(spec.capabilities.supports_optimization("compile"))


class MoonshineModelTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(911)
        self.config = tiny_moonshine_config()
        self.model = MoonshineForConditionalGeneration(self.config)
        self.waveforms = torch.randn(2, 1_000)
        self.attention_mask = torch.tensor([
            [1] * 900 + [0] * 100,
            [1] * 1_000,
        ])
        self.labels = torch.tensor([
            [1, 7, 8, 2],
            [1, 7, 2, -100],
        ])

    def test_forward_loss_backward_and_introspection(self):
        output = self.model(
            self.waveforms,
            self.attention_mask,
            labels=self.labels,
            output_attentions=True,
            output_hidden_states=True,
        )

        self.assertEqual(tuple(output.logits.shape), (2, 4, 13))
        self.assertEqual(
            tuple(output.encoder_last_hidden_state.shape),
            (2, 1, 8),
        )
        self.assertEqual(len(output.encoder_attentions), 1)
        self.assertEqual(len(output.decoder_attentions), 1)
        self.assertEqual(len(output.cross_attentions), 1)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for parameter in (
                self.model.model.encoder.conv1.weight,
                self.model.model.encoder.layers[0].self_attn.q_proj.weight,
                self.model.model.decoder.embed_tokens.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_greedy_generation_is_deterministic_and_bounded(self):
        self.model.eval()
        with torch.no_grad():
            first = self.model.generate(
                self.waveforms[:1],
                self.attention_mask[:1],
                max_new_tokens=3,
            )
            second = self.model.generate(
                self.waveforms[:1],
                self.attention_mask[:1],
                max_new_tokens=3,
            )
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(tuple(first.shape), (1, 4))
        with self.assertRaisesRegex(ValueError, "greedy"):
            self.model.generate(
                self.waveforms[:1],
                self.attention_mask[:1],
                num_beams=2,
            )
        with self.assertRaisesRegex(ValueError, "sampled"):
            self.model.generate(
                self.waveforms[:1],
                self.attention_mask[:1],
                do_sample=True,
            )

    def test_compile_targets_match_generation_execution_boundaries(self):
        inference_targets = self.model.optimization_compile_targets("inference", )
        training_targets = self.model.optimization_compile_targets("training", )

        self.assertEqual(
            tuple(target.label for target in inference_targets),
            ("encoder", "conditional-generation"),
        )
        self.assertIs(inference_targets[0].owner, self.model.model.encoder)
        self.assertIs(inference_targets[1].owner, self.model)
        self.assertEqual(
            tuple(target.label for target in training_targets),
            ("conditional-generation", ),
        )
        with self.assertRaisesRegex(ValueError, "inference.*training"):
            self.model.optimization_compile_targets("export")

    def test_invalid_masks_and_cache_modes_are_rejected(self):
        invalid_mask = self.attention_mask.clone()
        invalid_mask[0, 200] = 0
        invalid_mask[0, 201] = 1
        with self.assertRaisesRegex(ValueError, "right padded"):
            self.model(
                self.waveforms,
                invalid_mask,
                labels=self.labels,
            )
        with self.assertRaisesRegex(ValueError, "cache"):
            self.model(
                self.waveforms,
                self.attention_mask,
                labels=self.labels,
                use_cache=True,
            )


class MoonshineTokenizerTests(unittest.TestCase):

    def _tokenizer(self, root: Path) -> SentencePieceBPETokenizer:
        path = root / "tokenizer.json"
        path.write_text(
            json.dumps(tiny_tokenizer_document(), ensure_ascii=False),
            encoding="utf-8",
        )
        return SentencePieceBPETokenizer.from_tokenizer_json(
            path,
            pad_token_id=2,
            bos_token_id=1,
            eos_token_id=2,
        )

    def test_bos_merges_byte_fallback_and_decode_match_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = self._tokenizer(Path(directory))

            self.assertEqual(tokenizer.encode("hi").input_ids, (1, 7))
            self.assertEqual(
                tokenizer.encode("🙂").input_ids,
                (1, 3, 9, 10, 11, 12),
            )
            self.assertEqual(
                tokenizer.decode((1, 3, 9, 10, 11, 12, 2)),
                "🙂",
            )
            self.assertEqual(tokenizer.encode("").input_ids, (1, ))
            batch = tokenizer.encode_batch(("hi", "hi!"), pad=True)
            self.assertTrue(batch.is_padded)
            self.assertEqual(batch.attention_mask[0][-1], 0)

    def test_unsupported_or_ambiguous_tokenizer_graphs_are_rejected(self):
        document = tiny_tokenizer_document()
        document["model"]["dropout"] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                    TokenizerAssetError,
                    "dropout",
            ):
                SentencePieceBPETokenizer.from_tokenizer_json(
                    path,
                    pad_token_id=2,
                    bos_token_id=1,
                    eos_token_id=2,
                )

    def test_tokenizer_json_rejects_ambiguous_json_before_graph_interpretation(self):
        valid = json.dumps(tiny_tokenizer_document(), ensure_ascii=False)
        cases = (
            (
                "duplicate",
                valid.replace('"model": {', '"model": "discarded-secret", "model": {', 1),
                r"Duplicate JSON object key 'model'",
            ),
            (
                "constant",
                valid[:-1] + ', "metadata": {"score": NaN}}',
                r"non-finite JSON constant 'NaN'",
            ),
            (
                "overflow",
                valid[:-1] + ', "metadata": {"score": 1e400}}',
                r"\$\.metadata\.score.*non-finite",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            for name, document, diagnostic in cases:
                with self.subTest(name=name):
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaises(TokenizerAssetError) as captured:
                        SentencePieceBPETokenizer.from_tokenizer_json(
                            path,
                            pad_token_id=2,
                            bos_token_id=1,
                            eos_token_id=2,
                        )
                    rendered = str(captured.exception)
                    self.assertIn(str(path), rendered)
                    self.assertRegex(rendered, diagnostic)
                    self.assertNotIn("discarded-secret", rendered)


if __name__ == "__main__":
    unittest.main()
