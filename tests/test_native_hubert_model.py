from __future__ import annotations

import ast
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from voicehub.models.asr_hubert.native import (
        FACEBOOK_HUBERT_LARGE_LS960_FT_HEADER_FINGERPRINT,
        FACEBOOK_HUBERT_LARGE_LS960_FT_REVISION,
        FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_REVISION,
        FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_SHA256,
        TRANSFORMERS_HUBERT_REVISION,
        HubertConfig,
        HubertForCTC,
        HuggingFaceHubertCheckpointAdapter,
        huggingface_hubert_tensor_mapping,
        huggingface_hubert_tensor_shapes,
        native_hubert_tensor_shapes,
        safetensors_header_fingerprint,
    )
    from voicehub.models.asr_wav2vec2.native.modeling import Wav2Vec2EncoderLayerStableLayerNorm


def _tiny_config(**overrides):
    values = {
        "vocab_size": 8,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "hidden_dropout": 0.0,
        "activation_dropout": 0.0,
        "attention_dropout": 0.0,
        "feat_proj_dropout": 0.0,
        "final_dropout": 0.0,
        "layerdrop": 0.0,
        "conv_dim": (4, 8),
        "conv_stride": (2, 2),
        "conv_kernel": (4, 2),
        "conv_bias": True,
        "num_conv_pos_embeddings": 4,
        "num_conv_pos_embedding_groups": 2,
        "do_stable_layer_norm": True,
        "apply_spec_augment": True,
        "mask_time_prob": 0.5,
        "mask_time_length": 2,
        "mask_time_min_masks": 1,
        "mask_feature_prob": 0.0,
        "mask_feature_min_masks": 0,
        "ctc_loss_reduction": "sum",
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    values.update(overrides)
    return HubertConfig(**values)


@unittest.skipUnless(torch is not None, "Native HuBERT uses PyTorch")
class HubertConfigurationTests(unittest.TestCase):

    def test_official_large_configuration_preserves_stable_layer_norm(self):
        config = HubertConfig.from_dict({
            "model_type": "hubert",
            'architectures': ["HubertForCTC"],
            "vocab_size": 32,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "conv_bias": True,
            "feat_extract_norm": "layer",
            "feat_proj_layer_norm": True,
            "feat_proj_dropout": 0.1,
            "do_stable_layer_norm": True,
            "num_feat_extract_layers": 7,
            "gradient_checkpointing": False,
            "feat_extract_dropout": 0.0,
        })

        self.assertTrue(config.do_stable_layer_norm)
        self.assertTrue(config.feat_proj_layer_norm)
        self.assertEqual(config.inputs_to_logits_ratio, 320)
        self.assertEqual(config.minimum_input_samples, 400)
        self.assertEqual(config.feature_output_length(16_000), 49)
        self.assertEqual(config.to_dict()["model_type"], "hubert")
        self.assertEqual(
            config.to_dict()['architectures'],
            ["HubertForCTC"],
        )

    def test_unsupported_graph_variants_are_rejected(self):
        cases = (
            ({
                "add_adapter": True
            }, "language-adapter"),
            ({
                "adapter_attn_dim": 64
            }, "attention adapters"),
            ({
                "conv_pos_batch_norm": True
            }, "conv_pos_batch_norm"),
            ({
                "gradient_checkpointing": True
            }, "gradient checkpointing"),
            ({
                "feat_extract_dropout": 0.1
            }, "feat_extract_dropout"),
            ({
                "use_weighted_layer_sum": True
            }, "sequence-classification"),
        )
        for values, message in cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    HubertConfig.from_dict(values)


@unittest.skipUnless(torch is not None, "Native HuBERT uses PyTorch")
class HubertModelTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(101)
        self.config = _tiny_config()
        self.waveforms = torch.randn(2, 30)
        self.attention_mask = torch.tensor([
            [1] * 24 + [0] * 6,
            [1] * 30,
        ])

    def test_forward_loss_backward_and_stable_encoder_are_native(self):
        model = HubertForCTC(self.config).train()
        labels = torch.tensor([
            [1, 2, 3, -100],
            [2, 3, 4, 5],
        ])
        output = model(
            self.waveforms,
            self.attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            generator=torch.Generator().manual_seed(11),
        )

        self.assertEqual(tuple(output.logits.shape), (2, 7, 8))
        torch.testing.assert_close(
            output.input_lengths,
            torch.tensor([5, 7]),
        )
        self.assertEqual(len(output.hidden_states), 3)
        self.assertEqual(len(output.attentions), 2)
        self.assertIsInstance(
            model.hubert.encoder.layers[0],
            Wav2Vec2EncoderLayerStableLayerNorm,
        )
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for parameter in (
                model.hubert.masked_spec_embed,
                model.hubert.feature_extractor.conv_layers[0].conv.weight,
                model.hubert.encoder.layers[0].attention.q_proj.weight,
                model.lm_head.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_explicit_time_mask_uses_learned_hubert_embedding(self):
        model = HubertForCTC(self.config)
        with torch.no_grad():
            model.hubert.masked_spec_embed.fill_(3.25)
        hidden = torch.zeros(1, 4, self.config.hidden_size)
        valid = torch.ones(1, 4, dtype=torch.bool)
        selected = torch.tensor([[False, True, False, True]])

        masked = model.hubert._apply_spec_augment(
            hidden,
            valid,
            mask_time_indices=selected,
            generator=None,
        )

        torch.testing.assert_close(
            masked[0, selected[0]],
            torch.full((2, self.config.hidden_size), 3.25),
        )
        torch.testing.assert_close(
            masked[0, ~selected[0]],
            torch.zeros(2, self.config.hidden_size),
        )

    def test_state_dict_keeps_exact_hubert_namespace(self):
        model = HubertForCTC(self.config)
        actual = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}

        self.assertEqual(actual, native_hubert_tensor_shapes(self.config))
        self.assertIn("hubert.masked_spec_embed", actual)
        self.assertFalse(any(name.startswith("wav2vec2.") for name in actual))

    def test_feature_projection_layer_norm_switch_changes_the_graph(self):
        config = _tiny_config(
            feat_proj_layer_norm=False,
            apply_spec_augment=False,
            mask_time_prob=0.0,
        )
        model = HubertForCTC(config)

        self.assertIsNone(model.hubert.feature_projection.layer_norm)
        self.assertNotIn(
            "hubert.feature_projection.layer_norm.weight",
            model.state_dict(),
        )
        self.assertEqual(
            {
                name: tuple(tensor.shape)
                for name, tensor in model.state_dict().items()
            },
            native_hubert_tensor_shapes(config),
        )


@unittest.skipUnless(torch is not None, "Native HuBERT uses PyTorch")
class HubertCheckpointTests(unittest.TestCase):

    def test_official_safe_conversion_header_inventory_is_frozen(self):
        config = HubertConfig.from_dict({
            "vocab_size": 32,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "conv_bias": True,
            "feat_extract_norm": "layer",
            "feat_proj_layer_norm": True,
            "feat_proj_dropout": 0.1,
            "do_stable_layer_norm": True,
        })
        shapes = huggingface_hubert_tensor_shapes(config)

        self.assertEqual(len(shapes), 424)
        self.assertEqual(
            safetensors_header_fingerprint(shapes),
            FACEBOOK_HUBERT_LARGE_LS960_FT_HEADER_FINGERPRINT,
        )
        self.assertEqual(
            FACEBOOK_HUBERT_LARGE_LS960_FT_REVISION,
            "ece5fabbf034c1073acae96d5401b25be96709d8",
        )
        self.assertEqual(
            FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_REVISION,
            "ba42e7f7a888fd65f7af7849c452e3e7d5216aad",
        )
        self.assertEqual(
            FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_SHA256,
            "1fefcd85b08c83451afd1df872bc11b92333dc4b5393506def29a20baa69c4ed",
        )
        self.assertEqual(
            TRANSFORMERS_HUBERT_REVISION,
            "ebea912f0bb6f9e28ad2df04acd9b4df035933a9",
        )

    def test_checkpoint_mapping_strictly_loads_every_tensor(self):
        config = _tiny_config()
        torch.manual_seed(17)
        reference = HubertForCTC(config)
        reference_state = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}
        source = {
            source_name: reference_state[target_name]
            for source_name, target_name in huggingface_hubert_tensor_mapping(config)
        }

        torch.manual_seed(99)
        restored = HubertForCTC(config)
        report = HuggingFaceHubertCheckpointAdapter().load(
            restored,
            source,
            config.to_dict(),
            strict=True,
        )

        self.assertTrue(report.is_compatible, report.summary())
        self.assertEqual(set(report.loaded), set(reference_state))
        self.assertEqual(report.ignored_sources, ())
        for name, expected in reference_state.items():
            torch.testing.assert_close(
                restored.state_dict()[name],
                expected,
                rtol=0,
                atol=0,
            )

    def test_native_files_import_no_external_architecture_runtime(self):
        package = (Path(__file__).resolve().parents[1] / 'voicehub/models/asr_hubert/native')
        forbidden = {"numpy", "safetensors", "tokenizers", "transformers"}
        violations = []
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = {alias.name.partition(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots = {node.module.partition(".")[0]}
                else:
                    continue
                if roots & forbidden:
                    violations.append((path.name, node.lineno, roots))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
