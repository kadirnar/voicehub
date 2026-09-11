from __future__ import annotations

import ast
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from voicehub.models.asr_wavlm.native import (
        MICROSOFT_WAVLM_SOURCE_REVISION,
        TRANSFORMERS_WAVLM_REVISION,
        WAVLM_BASE_PLUS_CTC_HEADER_FINGERPRINT,
        WAVLM_BASE_PLUS_CTC_REVISION,
        WAVLM_BASE_PLUS_CTC_SAFETENSORS_REVISION,
        WAVLM_BASE_PLUS_CTC_SAFETENSORS_SHA256,
        HuggingFaceWavLMCheckpointAdapter,
        WavLMAttention,
        WavLMConfig,
        WavLMForCTC,
        huggingface_wavlm_tensor_mapping,
        huggingface_wavlm_tensor_shapes,
        native_wavlm_tensor_shapes,
        safetensors_header_fingerprint,
    )


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
        "num_buckets": 8,
        "max_bucket_distance": 16,
        "do_stable_layer_norm": False,
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
    return WavLMConfig(**values)


@unittest.skipUnless(torch is not None, "Native WavLM uses PyTorch")
class WavLMConfigurationTests(unittest.TestCase):

    def test_official_ctc_configuration_preserves_relative_position_fields(self):
        config = WavLMConfig.from_dict({
            "model_type": "wavlm",
            'architectures': ["WavLMForCTC"],
            "vocab_size": 31,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "num_buckets": 320,
            "max_bucket_distance": 800,
            "pad_token_id": 28,
            "bos_token_id": 1,
            "eos_token_id": 2,
        })

        self.assertEqual(config.num_buckets, 320)
        self.assertEqual(config.max_bucket_distance, 800)
        self.assertEqual(config.inputs_to_logits_ratio, 320)
        self.assertEqual(config.to_dict()["model_type"], "wavlm")
        self.assertEqual(config.to_dict()['architectures'], ["WavLMForCTC"])

    def test_unsupported_graph_variants_and_invalid_buckets_are_rejected(self):
        cases = (
            ({
                "add_adapter": True
            }, "language-adapter"),
            ({
                "adapter_attn_dim": 64
            }, "attention-adapter"),
            ({
                "gradient_checkpointing": True
            }, "gradient checkpointing"),
            ({
                "use_weighted_layer_sum": True
            }, "classification"),
            ({
                "mask_time_selection": "uniform"
            }, "static"),
            ({
                "no_mask_time_overlap": True
            }, "approximate"),
            ({
                "mask_channel_prob": 0.2
            }, "canonical"),
            ({
                "feat_extract_dropout": 0.1
            }, "feat_extract_dropout"),
            ({
                "num_buckets": 7
            }, "even"),
            ({
                "num_buckets": 8,
                "max_bucket_distance": 2
            }, "greater"),
        )
        for values, message in cases:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                WavLMConfig.from_dict(values)


@unittest.skipUnless(torch is not None, "Native WavLM uses PyTorch")
class WavLMModelTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(133)
        self.config = _tiny_config()
        self.waveforms = torch.randn(2, 30)
        self.attention_mask = torch.tensor([
            [1] * 24 + [0] * 6,
            [1] * 30,
        ])

    def test_forward_ctc_backward_reaches_relative_position_parameters(self):
        model = WavLMForCTC(self.config).train()
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
            generator=torch.Generator().manual_seed(17),
        )

        self.assertEqual(tuple(output.logits.shape), (2, 7, 8))
        torch.testing.assert_close(output.input_lengths, torch.tensor([5, 7]))
        self.assertEqual(len(output.hidden_states), 3)
        self.assertEqual(len(output.attentions), 2)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for parameter in (
                model.wavlm.masked_spec_embed,
                model.wavlm.encoder.layers[0].attention.rel_attn_embed.weight,
                model.wavlm.encoder.layers[1].attention.gru_rel_pos_linear.weight,
                model.lm_head.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_gated_attention_matches_torch_reference_equations(self):
        attention = WavLMAttention(
            self.config,
            has_relative_position_bias=True,
        ).eval()
        hidden_states = torch.randn(2, 5, self.config.hidden_size)
        attention_mask = torch.tensor([
            [True, True, True, True, False],
            [True, True, True, True, True],
        ])

        actual, _, position_bias = attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=None,
            output_attentions=True,
        )
        gated_bias = attention._gated_position_bias(
            hidden_states,
            position_bias,
        ).reshape(
            hidden_states.shape[0] * self.config.num_attention_heads,
            hidden_states.shape[1],
            hidden_states.shape[1],
        )
        key_padding_mask = torch.zeros(
            attention_mask.shape,
            dtype=hidden_states.dtype,
        ).masked_fill(~attention_mask, -torch.inf)
        time_major = hidden_states.transpose(0, 1)
        expected, _ = torch.nn.functional.multi_head_attention_forward(
            time_major,
            time_major,
            time_major,
            self.config.hidden_size,
            self.config.num_attention_heads,
            torch.empty(0),
            torch.cat((
                attention.q_proj.bias,
                attention.k_proj.bias,
                attention.v_proj.bias,
            )),
            None,
            None,
            False,
            0.0,
            attention.out_proj.weight,
            attention.out_proj.bias,
            False,
            key_padding_mask,
            False,
            gated_bias,
            use_separate_proj_weight=True,
            q_proj_weight=attention.q_proj.weight,
            k_proj_weight=attention.k_proj.weight,
            v_proj_weight=attention.v_proj.weight,
        )
        torch.testing.assert_close(
            actual,
            expected.transpose(0, 1),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_relative_position_buckets_and_first_layer_execution_are_stable(self):
        attention = WavLMAttention(
            self.config,
            has_relative_position_bias=True,
        )
        positions = torch.tensor([[-20, -2, -1, 0, 1, 2, 20]])
        buckets = attention.relative_position_buckets(positions)
        self.assertEqual(
            buckets.tolist(),
            [[3, 2, 1, 0, 5, 6, 7]],
        )

        config = _tiny_config(
            num_hidden_layers=3,
            layerdrop=1.0,
            apply_spec_augment=False,
        )
        model = WavLMForCTC(config).train()
        output = model(
            torch.randn(1, 30),
            output_hidden_states=True,
        )
        self.assertEqual(output.executed_layers, (True, False, False))

    def test_request_generator_scopes_spec_augment_and_layerdrop(self):
        config = _tiny_config(
            num_hidden_layers=4,
            layerdrop=0.5,
        )
        model = WavLMForCTC(config).train()
        waveform = torch.randn(1, 30)

        first = model(
            waveform,
            generator=torch.Generator().manual_seed(515),
        )
        second = model(
            waveform,
            generator=torch.Generator().manual_seed(515),
        )

        self.assertEqual(first.executed_layers, second.executed_layers)
        torch.testing.assert_close(first.logits, second.logits)

    def test_explicit_time_mask_uses_the_learned_wavlm_embedding(self):
        model = WavLMForCTC(self.config)
        with torch.no_grad():
            model.wavlm.masked_spec_embed.fill_(2.75)
        hidden = torch.zeros(1, 4, self.config.hidden_size)
        valid = torch.ones(1, 4, dtype=torch.bool)
        selected = torch.tensor([[False, True, False, True]])

        masked = model.wavlm._apply_spec_augment(
            hidden,
            valid,
            mask_time_indices=selected,
            generator=None,
        )

        torch.testing.assert_close(
            masked[0, selected[0]],
            torch.full((2, self.config.hidden_size), 2.75),
        )
        torch.testing.assert_close(
            masked[0, ~selected[0]],
            torch.zeros(2, self.config.hidden_size),
        )

    def test_state_dict_keeps_exact_wavlm_namespace(self):
        model = WavLMForCTC(self.config)
        actual = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}

        self.assertEqual(actual, native_wavlm_tensor_shapes(self.config))
        self.assertIn(
            "wavlm.encoder.layers.0.attention.rel_attn_embed.weight",
            actual,
        )
        self.assertNotIn(
            "wavlm.encoder.layers.1.attention.rel_attn_embed.weight",
            actual,
        )
        self.assertFalse(any(name.startswith("wav2vec2.") for name in actual))


@unittest.skipUnless(torch is not None, "Native WavLM uses PyTorch")
class WavLMCheckpointTests(unittest.TestCase):

    def test_safe_conversion_header_inventory_and_revisions_are_frozen(self):
        config = WavLMConfig.from_dict({
            "vocab_size": 31,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_dropout": 0.0,
            "activation_dropout": 0.0,
            "attention_dropout": 0.0,
            "feat_proj_dropout": 0.0,
            "final_dropout": 0.0,
            "layerdrop": 0.0,
            "num_buckets": 320,
            "max_bucket_distance": 800,
            "ctc_loss_reduction": "mean",
            "pad_token_id": 28,
        })
        shapes = huggingface_wavlm_tensor_shapes(config)

        self.assertEqual(len(shapes), 250)
        self.assertEqual(
            safetensors_header_fingerprint(shapes),
            WAVLM_BASE_PLUS_CTC_HEADER_FINGERPRINT,
        )
        self.assertEqual(
            WAVLM_BASE_PLUS_CTC_REVISION,
            "02c289c4471cd1ba4b0ff3e7c304afe395c5026a",
        )
        self.assertEqual(
            WAVLM_BASE_PLUS_CTC_SAFETENSORS_REVISION,
            "561f43a6081f379876b6633a38526aabe140ba3b",
        )
        self.assertEqual(
            WAVLM_BASE_PLUS_CTC_SAFETENSORS_SHA256,
            "cc6b213ad14d4589568ad844841f6d3b3c58d12f2326cce14804530c02ff2dd1",
        )
        self.assertEqual(
            MICROSOFT_WAVLM_SOURCE_REVISION,
            "833df7e7832e5064a281131ee64a481afa8e5b95",
        )
        self.assertEqual(
            TRANSFORMERS_WAVLM_REVISION,
            "ebea912f0bb6f9e28ad2df04acd9b4df035933a9",
        )

    def test_checkpoint_mapping_strictly_loads_every_tensor(self):
        config = _tiny_config()
        torch.manual_seed(19)
        reference = WavLMForCTC(config)
        reference_state = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}
        source = {
            source_name: reference_state[target_name]
            for source_name, target_name in huggingface_wavlm_tensor_mapping(config)
        }

        restored = WavLMForCTC(config)
        report = HuggingFaceWavLMCheckpointAdapter().load(
            restored,
            source,
            config.to_dict(),
            strict=True,
        )

        self.assertTrue(report.is_compatible, report.summary())
        self.assertEqual(set(report.loaded), set(reference_state))
        for name, expected in reference_state.items():
            torch.testing.assert_close(
                restored.state_dict()[name],
                expected,
                rtol=0,
                atol=0,
            )

    def test_native_files_import_no_external_architecture_runtime(self):
        package = (Path(__file__).resolve().parents[1] / 'voicehub/models/asr_wavlm/native')
        forbidden = {
            "huggingface_hub",
            "numpy",
            "safetensors",
            "tokenizers",
            "torchaudio",
            "transformers",
        }
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
