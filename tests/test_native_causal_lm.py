import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch

from voicehub.architectures import ArchitectureRegistry
from voicehub.architectures.causal_lm import (
    REFERENCE_CAUSAL_LM_CHECKPOINTS,
    TRANSFORMERS_CAUSAL_LM_REVISION,
    CausalLMConfig,
    GraniteConfig,
    GraniteForCausalLM,
    HFCausalLMCheckpointAdapter,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    create_causal_lm_architecture_spec,
    native_causal_lm_tensor_shapes,
    register_causal_lm_architecture,
)
from voicehub.checkpointing import CheckpointCompatibilityError
from voicehub.generation import GenerationConfig


def _tiny_config(config_type, **overrides):
    values = {
        "vocab_size": 41,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 32,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    values.update(overrides)
    return config_type(**values)


class CausalLMConfigurationTests(unittest.TestCase):

    def test_huggingface_dict_dispatches_to_the_exact_family(self):
        for model_type, expected in (
            ("granite", GraniteConfig),
            ("llama", LlamaConfig),
            ("qwen2", Qwen2Config),
            ("qwen3", Qwen3Config),
        ):
            with self.subTest(model_type=model_type):
                source = _tiny_config(expected).to_dict()
                source["custom_metadata"] = {"source": "test"}
                config = CausalLMConfig.from_dict(source)

                self.assertIsInstance(config, expected)
                self.assertEqual(config.model_type, model_type)
                self.assertEqual(
                    config.extra_config["custom_metadata"],
                    {"source": "test"},
                )
                self.assertEqual(config.to_dict()["model_type"], model_type)

    def test_family_traits_match_the_pinned_official_sources(self):
        llama = _tiny_config(LlamaConfig)
        granite = _tiny_config(GraniteConfig)
        qwen2 = _tiny_config(Qwen2Config)
        qwen3 = _tiny_config(Qwen3Config)

        self.assertFalse(granite.qkv_bias)
        self.assertFalse(granite.attention_output_bias)
        self.assertFalse(granite.uses_qk_norm)
        self.assertFalse(llama.qkv_bias)
        self.assertFalse(llama.attention_output_bias)
        self.assertFalse(llama.uses_qk_norm)
        self.assertTrue(qwen2.qkv_bias)
        self.assertFalse(qwen2.attention_output_bias)
        self.assertFalse(qwen2.uses_qk_norm)
        self.assertFalse(qwen3.qkv_bias)
        self.assertFalse(qwen3.attention_output_bias)
        self.assertTrue(qwen3.uses_qk_norm)

    def test_qwen_defaults_match_the_pinned_configuration_classes(self):
        qwen2 = Qwen2Config()
        qwen3 = Qwen3Config()

        self.assertEqual(qwen2.intermediate_size, 22_016)
        self.assertEqual(qwen2.num_key_value_heads, 32)
        self.assertEqual(qwen2.max_position_embeddings, 32_768)
        self.assertEqual(qwen2.rope_theta, 10_000.0)
        self.assertIsNone(qwen2.eos_token_id)
        self.assertEqual(qwen3.intermediate_size, 22_016)
        self.assertEqual(qwen3.num_key_value_heads, 32)
        self.assertEqual(qwen3.head_dim, 128)
        self.assertEqual(qwen3.max_position_embeddings, 32_768)
        self.assertEqual(qwen3.rope_theta, 10_000.0)
        self.assertIsNone(qwen3.eos_token_id)

    def test_mathematically_unsupported_variants_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Sliding-window"):
            _tiny_config(Qwen2Config, use_sliding_window=True)
        with self.assertRaisesRegex(ValueError, "default RoPE"):
            _tiny_config(
                LlamaConfig,
                rope_scaling={
                    "rope_type": "linear",
                    "factor": 2.0
                },
            )
        with self.assertRaisesRegex(ValueError, "partial_rotary_factor"):
            _tiny_config(
                LlamaConfig,
                rope_scaling={
                    "rope_type": "default",
                    "partial_rotary_factor": 0.5,
                },
            )
        with self.assertRaisesRegex(ValueError, "mixture-of-experts"):
            CausalLMConfig.from_dict({"model_type": "qwen3_moe"})
        with self.assertRaisesRegex(ValueError, "divide"):
            _tiny_config(
                LlamaConfig,
                num_attention_heads=6,
                num_key_value_heads=4,
            )
        with self.assertRaisesRegex(ValueError, "must be even"):
            _tiny_config(LlamaConfig, head_dim=3)
        with self.assertRaisesRegex(ValueError, "bias-free SwiGLU"):
            _tiny_config(Qwen3Config, mlp_bias=True)
        with self.assertRaisesRegex(ValueError, "cannot parse"):
            Qwen2Config.from_dict({"model_type": "llama"})

    def test_config_is_detached_and_json_serializable(self):
        extras = {"nested": {"values": [1]}}
        config = _tiny_config(LlamaConfig, extra_config=extras)
        extras["nested"]["values"].append(2)

        self.assertEqual(config.extra_config["nested"]["values"], [1])
        json.dumps(config.to_dict())

    def test_new_rope_parameters_accept_a_null_legacy_field(self):
        values = _tiny_config(LlamaConfig).to_dict()
        values["rope_scaling"] = None
        values["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        }

        config = CausalLMConfig.from_dict(values)

        self.assertEqual(config.rope_theta, 10_000.0)
        self.assertEqual(config.rope_scaling["rope_type"], "default")


class CausalLMGraphTests(unittest.TestCase):

    def _families(self):
        return (
            (GraniteConfig, GraniteForCausalLM),
            (LlamaConfig, LlamaForCausalLM),
            (Qwen2Config, Qwen2ForCausalLM),
            (Qwen3Config, Qwen3ForCausalLM),
        )

    def test_state_dict_matches_the_strict_family_inventory(self):
        for config_type, model_type in self._families():
            with self.subTest(family=config_type.__name__):
                config = _tiny_config(config_type)
                model = model_type(config)
                expected = native_causal_lm_tensor_shapes(config)
                actual = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
                self.assertEqual(actual, expected)

        qwen2_names = set(Qwen2ForCausalLM(_tiny_config(Qwen2Config)).state_dict())
        self.assertIn(
            "model.layers.0.self_attn.q_proj.bias",
            qwen2_names,
        )
        self.assertNotIn(
            "model.layers.0.self_attn.o_proj.bias",
            qwen2_names,
        )
        qwen3_names = set(Qwen3ForCausalLM(_tiny_config(Qwen3Config)).state_dict())
        self.assertIn(
            "model.layers.0.self_attn.q_norm.weight",
            qwen3_names,
        )
        self.assertIn(
            "model.layers.0.self_attn.k_norm.weight",
            qwen3_names,
        )

    def test_granite_applies_all_published_architecture_multipliers(self):
        config = _tiny_config(
            GraniteConfig,
            embedding_multiplier=3.0,
            logits_scaling=4.0,
            residual_multiplier=0.25,
            attention_multiplier=0.125,
        )
        model = GraniteForCausalLM(config)

        self.assertEqual(model.config.embedding_multiplier, 3.0)
        self.assertEqual(model.config.logits_scaling, 4.0)
        self.assertEqual(
            model.model.layers[0].self_attn.scaling,
            0.125,
        )
        self.assertEqual(
            model.model.layers[0].config.residual_multiplier,
            0.25,
        )

    def test_causal_loss_backpropagates_through_every_family(self):
        token_ids = torch.tensor(
            [[1, 7, 8, 2], [1, 9, 10, 2]],
            dtype=torch.long,
        )
        for config_type, model_type in self._families():
            with self.subTest(family=config_type.__name__):
                torch.manual_seed(7)
                model = model_type(_tiny_config(config_type))
                output = model(
                    token_ids,
                    labels=token_ids,
                )
                self.assertEqual(
                    tuple(output.logits.shape),
                    (2, 4, 41),
                )
                self.assertIsNone(output.past_key_values)
                self.assertTrue(torch.isfinite(output.loss))
                output.loss.backward()
                self.assertTrue(
                    all(
                        parameter.grad is not None for parameter in model.parameters()
                        if parameter.requires_grad))

    def test_incremental_cache_matches_full_sequence_logits(self):
        token_ids = torch.tensor(
            [[1, 5, 6, 7, 2], [1, 8, 9, 10, 2]],
            dtype=torch.long,
        )
        for config_type, model_type in self._families():
            with self.subTest(family=config_type.__name__):
                torch.manual_seed(11)
                model = model_type(_tiny_config(config_type)).eval()
                with torch.no_grad():
                    full = model(token_ids, use_cache=False).logits
                    cache = None
                    pieces = []
                    for index in range(token_ids.shape[1]):
                        output = model(
                            token_ids[:, index:index + 1],
                            past_key_values=cache,
                            use_cache=True,
                        )
                        cache = output.past_key_values
                        pieces.append(output.logits)
                    incremental = torch.cat(pieces, dim=1)

                torch.testing.assert_close(
                    incremental,
                    full,
                    atol=1e-6,
                    rtol=1e-5,
                )
                self.assertEqual(
                    cache.sequence_length(),
                    token_ids.shape[1],
                )

    def test_positions_beyond_the_trained_context_are_extrapolated(self):
        # Transformers does not cap RoPE positions at max_position_embeddings
        # (Higgs Audio v2 publishes 2048 yet decodes longer sequences).
        torch.manual_seed(13)
        model = LlamaForCausalLM(_tiny_config(LlamaConfig, max_position_embeddings=4)).eval()
        token_ids = torch.tensor([[1, 5, 6, 7, 8, 9, 10]])
        # `warnings.catch_warnings` instead of `assertWarns`, which touches
        # every loaded module and trips lazy third-party imports.
        with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            full = model(token_ids, use_cache=False).logits
            explicit = model(
                token_ids,
                position_ids=torch.arange(token_ids.shape[1]),
                use_cache=False,
            ).logits

        messages = [str(warning.message) for warning in caught]
        self.assertEqual(sum("max_position_embeddings" in message for message in messages), 2)
        self.assertTrue(torch.isfinite(full).all())
        torch.testing.assert_close(explicit, full, atol=0.0, rtol=0.0)
        with torch.no_grad():
            within = model(token_ids[:, :4], use_cache=False).logits
        torch.testing.assert_close(within, full[:, :4], atol=1e-6, rtol=1e-5)

    def test_chunked_cache_uses_bottom_right_causal_alignment(self):
        token_ids = torch.tensor(
            [[1, 5, 6, 7, 8, 2], [1, 9, 10, 11, 12, 2]],
            dtype=torch.long,
        )
        for key_value_heads in (2, 4):
            with self.subTest(key_value_heads=key_value_heads):
                torch.manual_seed(13)
                model = Qwen3ForCausalLM(_tiny_config(
                    Qwen3Config,
                    num_key_value_heads=key_value_heads,
                )).eval()
                with torch.no_grad():
                    full = model(token_ids, use_cache=False).logits
                    prefix = model(
                        token_ids[:, :2],
                        use_cache=True,
                    )
                    chunk = model(
                        token_ids[:, 2:],
                        past_key_values=prefix.past_key_values,
                        use_cache=True,
                    )

                torch.testing.assert_close(
                    chunk.logits,
                    full[:, 2:],
                    atol=1e-6,
                    rtol=1e-5,
                )

    def test_masked_chunk_prefill_never_requests_grouped_sdpa(self):
        # SDPA has no fused kernel for grouped K/V plus an explicit mask, so
        # a cached multi-token chunk must expand heads (Transformers'
        # repeat_kv) instead of silently falling back to the math kernel.
        torch.manual_seed(19)
        model = Qwen2ForCausalLM(_tiny_config(Qwen2Config)).eval()
        token_ids = torch.tensor([[1, 5, 6, 7, 8, 2]], dtype=torch.long)
        calls = []
        original = torch.nn.functional.scaled_dot_product_attention

        def record(query, key, value, **kwargs):
            calls.append((
                query.shape[1],
                key.shape[1],
                kwargs.get("attn_mask") is not None,
                kwargs.get("enable_gqa", False),
            ))
            return original(query, key, value, **kwargs)

        with torch.no_grad():
            full = model(token_ids, use_cache=False).logits
            prefix = model(token_ids[:, :2], use_cache=True)
            with mock.patch.object(
                    torch.nn.functional,
                    "scaled_dot_product_attention",
                    side_effect=record,
            ):
                chunk = model(
                    token_ids[:, 2:],
                    past_key_values=prefix.past_key_values,
                    use_cache=True,
                )
                step = model(
                    token_ids[:, -1:],
                    past_key_values=chunk.past_key_values,
                    use_cache=True,
                )
        self.assertEqual(calls[:2], [(4, 4, True, False)] * 2)
        self.assertEqual(calls[2:], [(4, 2, False, True)] * 2)
        self.assertEqual(prefix.past_key_values.get(0).key.shape[1], 2)
        torch.testing.assert_close(chunk.logits, full[:, 2:], atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.isfinite(step.logits).all())

    def test_output_attentions_preserves_explicit_attention_path(self):
        torch.manual_seed(17)
        model = Qwen3ForCausalLM(_tiny_config(Qwen3Config)).eval()
        token_ids = torch.tensor([[1, 5, 6, 2]], dtype=torch.long)

        with torch.no_grad():
            fused = model(token_ids, use_cache=False)
            diagnostic = model(
                token_ids,
                use_cache=False,
                output_attentions=True,
            )

        self.assertEqual(len(diagnostic.attentions), 2)
        self.assertEqual(
            tuple(diagnostic.attentions[0].shape),
            (1, 4, 4, 4),
        )
        torch.testing.assert_close(
            diagnostic.logits,
            fused.logits,
            atol=1e-6,
            rtol=1e-5,
        )

    def test_left_padding_is_finite_and_cache_equivalent(self):
        token_ids = torch.tensor(
            [[0, 0, 1, 7, 2], [0, 1, 8, 9, 2]],
            dtype=torch.long,
        )
        attention_mask = token_ids.ne(0)
        model = Qwen3ForCausalLM(_tiny_config(Qwen3Config), ).eval()
        with torch.no_grad():
            full = model(
                token_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            cache = None
            pieces = []
            for index in range(token_ids.shape[1]):
                output = model(
                    token_ids[:, index:index + 1],
                    attention_mask=attention_mask[:, :index + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
                pieces.append(output.logits)
            incremental = torch.cat(pieces, dim=1)

        self.assertTrue(torch.isfinite(full).all())
        torch.testing.assert_close(
            incremental[attention_mask],
            full[attention_mask],
            atol=1e-6,
            rtol=1e-5,
        )

    def test_gradient_checkpointing_supports_training(self):
        model = LlamaForCausalLM(_tiny_config(LlamaConfig))
        model.train()
        model.gradient_checkpointing_enable()
        token_ids = torch.tensor([[1, 5, 6, 2]])

        with self.assertRaisesRegex(ValueError, "KV-cache"):
            model(token_ids, use_cache=True)
        output = model(
            token_ids,
            labels=token_ids,
            use_cache=False,
        )
        output.loss.backward()
        self.assertIsNotNone(model.model.layers[0].self_attn.q_proj.weight.grad)
        self.assertIsNotNone(model.model.layers[1].self_attn.q_proj.weight.grad)

    def test_generation_uses_the_shared_cache_aware_engine(self):
        model = LlamaForCausalLM(_tiny_config(LlamaConfig)).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        prompt = torch.tensor([[1, 5, 6]])
        output = model.generate(
            prompt,
            generation_config=GenerationConfig(
                max_new_tokens=3,
                eos_token_id=2,
                pad_token_id=0,
                use_cache=True,
            ),
        )

        self.assertEqual(tuple(output.sequences.shape), (1, 6))
        self.assertEqual(output.sequences[0, -3:].tolist(), [0, 0, 0])
        self.assertEqual(output.cache.sequence_length(), 5)

    def test_generation_collapses_only_dense_prompt_masks(self):
        model = LlamaForCausalLM(_tiny_config(LlamaConfig)).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        config = GenerationConfig(
            max_new_tokens=2,
            pad_token_id=0,
            use_cache=True,
        )
        prompt = torch.tensor([[1, 5, 6]])

        with mock.patch(
                "torch.nn.functional.scaled_dot_product_attention",
                wraps=torch.nn.functional.scaled_dot_product_attention,
        ) as fused_attention:
            model.generate(
                prompt,
                attention_mask=torch.ones_like(prompt, dtype=torch.bool),
                generation_config=config,
            )
        self.assertGreater(fused_attention.call_count, 0)

        padded_prompt = torch.tensor([[0, 1, 5]])
        with mock.patch(
                "torch.nn.functional.scaled_dot_product_attention",
                wraps=torch.nn.functional.scaled_dot_product_attention,
        ) as fused_attention:
            model.generate(
                padded_prompt,
                attention_mask=padded_prompt.ne(0),
                generation_config=config,
            )
        self.assertEqual(fused_attention.call_count, 0)

    def test_rotary_table_is_computed_once_per_forward(self):
        model = Qwen3ForCausalLM(_tiny_config(Qwen3Config, num_hidden_layers=3)).eval()
        calls = []
        for layer in model.model.layers:
            layer.self_attn.rotary.register_forward_hook(lambda *_: calls.append(None))
        token_ids = torch.tensor([[1, 5, 6, 7]])

        with torch.no_grad():
            output = model(token_ids, use_cache=True)
            self.assertEqual(len(calls), 1)
            model(token_ids[:, -1:], past_key_values=output.past_key_values, use_cache=True)
            self.assertEqual(len(calls), 2)

            # The shared table reproduces the per-layer computation bit for bit.
            hidden_states = model.model.embed_tokens(token_ids)
            position_ids = torch.arange(token_ids.shape[1]).unsqueeze(0)
            for layer in model.model.layers:
                hidden_states, _, _ = layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    cache=None,
                    use_cache=False,
                    output_attentions=False,
                )
            expected = model.lm_head(model.model.norm(hidden_states)).float()
        self.assertTrue(torch.equal(output.logits, expected))

    def test_logits_to_keep_projects_only_trailing_positions(self):
        model = LlamaForCausalLM(_tiny_config(LlamaConfig)).eval()
        token_ids = torch.tensor([[1, 5, 6, 7, 8]])
        with torch.no_grad():
            full = model(token_ids).logits
            self.assertTrue(torch.equal(model(token_ids, logits_to_keep=0).logits, full))
            trailing = model(token_ids, logits_to_keep=2).logits

        self.assertEqual(tuple(trailing.shape), (1, 2, 41))
        self.assertEqual(trailing.dtype, torch.float32)
        torch.testing.assert_close(trailing, full[:, -2:], atol=1e-6, rtol=1e-5)
        for invalid in (-1, True, 1.0):
            with self.subTest(logits_to_keep=invalid), self.assertRaisesRegex(ValueError, "logits_to_keep"):
                model(token_ids, logits_to_keep=invalid)
        with self.assertRaisesRegex(ValueError, "complete sequence"):
            model(token_ids, labels=token_ids, logits_to_keep=1)

    def test_generation_projects_only_the_last_prompt_position(self):
        model = LlamaForCausalLM(_tiny_config(LlamaConfig)).eval()
        projected_lengths = []
        model.lm_head.register_forward_hook(
            lambda _, inputs, __: projected_lengths.append(inputs[0].shape[1]))
        prompt = torch.tensor([[1, 5, 6, 7]])

        with torch.no_grad():
            output = model.generate(
                prompt,
                generation_config=GenerationConfig(
                    max_new_tokens=3,
                    pad_token_id=0,
                    use_cache=True,
                ),
            )
            self.assertEqual(projected_lengths, [1, 1, 1])

            # Greedy tokens match a full-sequence, full-vocabulary recompute.
            expected = prompt
            for _ in range(3):
                next_token = model(expected, use_cache=False).logits[:, -1].argmax(dim=-1, keepdim=True)
                expected = torch.cat((expected, next_token), dim=-1)
        self.assertTrue(torch.equal(output.sequences, expected))


class CausalLMCheckpointTests(unittest.TestCase):

    def test_audited_tiny_checkpoint_headers_match_native_shapes(self):
        representative_configs = {
            "llama":
            LlamaConfig(
                vocab_size=32_000,
                hidden_size=16,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=2_048,
            ),
            "qwen2":
            Qwen2Config(
                vocab_size=151_665,
                hidden_size=8,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32_768,
            ),
            "qwen3":
            Qwen3Config(
                vocab_size=151_669,
                hidden_size=8,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=128,
                max_position_embeddings=32_768,
            ),
        }
        shapes = {
            family: native_causal_lm_tensor_shapes(config)
            for family, config in representative_configs.items()
        }

        for family, inventory in shapes.items():
            self.assertEqual(
                len(inventory),
                REFERENCE_CAUSAL_LM_CHECKPOINTS[family]["tensor_count"],
            )
        self.assertEqual(
            shapes["llama"]["model.layers.0.self_attn.q_proj.weight"],
            (16, 16),
        )
        self.assertEqual(
            shapes["qwen2"]["model.layers.0.self_attn.k_proj.weight"],
            (4, 8),
        )
        self.assertEqual(
            shapes["qwen3"]["model.layers.0.self_attn.q_proj.weight"],
            (512, 8),
        )
        self.assertEqual(
            shapes["qwen3"]["model.layers.0.self_attn.q_norm.weight"],
            (128, ),
        )

    def test_identity_mapping_strictly_loads_all_three_namespaces(self):
        for config_type, model_type in (
            (LlamaConfig, LlamaForCausalLM),
            (Qwen2Config, Qwen2ForCausalLM),
            (Qwen3Config, Qwen3ForCausalLM),
        ):
            with self.subTest(family=config_type.__name__):
                torch.manual_seed(17)
                config = _tiny_config(config_type)
                source_model = model_type(config)
                target_model = model_type(config, initialize=False)
                report = HFCausalLMCheckpointAdapter().load(
                    target_model,
                    source_model.state_dict(),
                    config.to_dict(),
                    strict=True,
                )

                self.assertTrue(report.is_compatible)
                self.assertEqual(
                    set(report.loaded),
                    set(source_model.state_dict()),
                )
                for name, tensor in source_model.state_dict().items():
                    torch.testing.assert_close(
                        target_model.state_dict()[name],
                        tensor,
                    )

    def test_strict_mapping_reports_unexpected_checkpoint_tensors(self):
        config = _tiny_config(LlamaConfig)
        model = LlamaForCausalLM(config)
        source = dict(model.state_dict())
        source["unexpected.weight"] = torch.zeros(1)

        with self.assertRaises(CheckpointCompatibilityError):
            HFCausalLMCheckpointAdapter().load(
                model,
                source,
                config.to_dict(),
                strict=True,
            )

    def test_safetensors_roundtrip_preserves_qwen3_exactly(self):
        torch.manual_seed(23)
        config = _tiny_config(Qwen3Config)
        source = Qwen3ForCausalLM(config)
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            loaded = Qwen3ForCausalLM.from_pretrained(directory)

            for name, tensor in source.state_dict().items():
                torch.testing.assert_close(
                    loaded.state_dict()[name],
                    tensor,
                )

    def test_explicit_sharded_snapshot_index_keeps_logical_shard_directory(self):
        from voicehub.architectures.causal_lm.checkpoint import open_causal_lm_tensor_source
        from voicehub.checkpointing import save_safetensors

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blobs = root / "blobs"
            snapshot = root / "snapshots" / "revision"
            blobs.mkdir()
            snapshot.mkdir(parents=True)
            shard_blob = blobs / ("c" * 64)
            index_blob = blobs / ("d" * 64)
            save_safetensors({"weight": torch.tensor([5.0])}, shard_blob)
            index_blob.write_text(
                json.dumps({
                    "weight_map": {
                        "weight": "model-00001-of-00001.safetensors",
                    },
                }),
                encoding="utf-8",
            )
            index_path = snapshot / "model.safetensors.index.json"
            shard_path = snapshot / "model-00001-of-00001.safetensors"
            index_path.symlink_to(Path("../../blobs") / index_blob.name)
            shard_path.symlink_to(Path("../../blobs") / shard_blob.name)

            with open_causal_lm_tensor_source(index_path) as reader:
                torch.testing.assert_close(
                    reader.get_tensor("weight"),
                    torch.tensor([5.0]),
                )

    def test_tied_embedding_roundtrip_uses_one_checkpoint_tensor(self):
        config = _tiny_config(
            LlamaConfig,
            tie_word_embeddings=True,
        )
        source = LlamaForCausalLM(config)
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            from voicehub.checkpointing import SafeTensorReader

            with SafeTensorReader(Path(directory) / "model.safetensors") as reader:
                self.assertIn("model.embed_tokens.weight", reader)
                self.assertNotIn("lm_head.weight", reader)
            loaded = LlamaForCausalLM.from_pretrained(directory)

        self.assertIs(
            loaded.lm_head.weight,
            loaded.model.embed_tokens.weight,
        )
        torch.testing.assert_close(
            loaded.lm_head.weight,
            source.lm_head.weight,
        )


class CausalLMRegistrationTests(unittest.TestCase):

    def test_spec_records_the_immutable_official_revision(self):
        spec = create_causal_lm_architecture_spec()

        self.assertEqual(spec.architecture_id, "causal-lm")
        self.assertEqual(
            spec.upstream_revision,
            TRANSFORMERS_CAUSAL_LM_REVISION,
        )
        self.assertTrue(spec.capabilities.training)
        self.assertTrue(spec.capabilities.has_feature("qwen3"))
        self.assertIn(
            TRANSFORMERS_CAUSAL_LM_REVISION,
            spec.metadata["transformers_sources"]["llama"],
        )

    def test_registration_aliases_share_one_lazy_family_spec(self):
        registry = ArchitectureRegistry()
        spec = register_causal_lm_architecture(registry=registry)

        self.assertIs(registry.get("llama"), spec)
        self.assertIs(registry.get("qwen2"), spec)
        self.assertIs(registry.get("qwen3"), spec)

    def test_catalog_discovery_does_not_import_the_model_graph(self):
        code = """
import json
import sys
import voicehub.architectures
print(json.dumps({
    "registered": "causal-lm" in voicehub.architectures.ARCHITECTURES,
    "modeling": "voicehub.architectures.causal_lm.modeling" in sys.modules,
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload["registered"])
        self.assertFalse(payload["modeling"])


if __name__ == "__main__":
    unittest.main()
