from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from voicehub.models.asr_wav2vec2.native import (
        FACEBOOK_WAV2VEC2_BASE_960H_HEADER_FINGERPRINT,
        FACEBOOK_WAV2VEC2_BASE_960H_REVISION,
        TRANSFORMERS_WAV2VEC2_REVISION,
        HuggingFaceWav2Vec2CheckpointAdapter,
        HuggingFaceWav2Vec2ClassificationCheckpointAdapter,
        Wav2Vec2Config,
        Wav2Vec2ForAudioFrameClassification,
        Wav2Vec2ForCTC,
        Wav2Vec2ForSequenceClassification,
        create_wav2vec2_architecture_spec,
        huggingface_wav2vec2_tensor_mapping,
        huggingface_wav2vec2_tensor_shapes,
        native_wav2vec2_frame_classification_tensor_shapes,
        native_wav2vec2_sequence_classification_tensor_shapes,
        native_wav2vec2_tensor_shapes,
        register_wav2vec2_architecture,
        safetensors_header_fingerprint,
    )
    from voicehub.runtime.registry import ArchitectureRegistry


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
        "num_conv_pos_embeddings": 4,
        "num_conv_pos_embedding_groups": 2,
        "apply_spec_augment": False,
        "mask_time_prob": 0.0,
        "mask_time_min_masks": 0,
        "mask_feature_prob": 0.0,
        "mask_feature_min_masks": 0,
        "ctc_loss_reduction": "sum",
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    values.update(overrides)
    return Wav2Vec2Config(**values)


@unittest.skipUnless(torch is not None, "Native Wav2Vec2 uses PyTorch")
class Wav2Vec2ConfigurationTests(unittest.TestCase):

    def test_official_base_dimensions_and_length_formula(self):
        config = Wav2Vec2Config.from_dict({
            "model_type": "wav2vec2",
            'architectures': ["Wav2Vec2ForCTC"],
            "hidden_dropout": 0.1,
            "hidden_dropout_prob": 0.1,
            "num_feat_extract_layers": 7,
        })

        self.assertEqual(config.inputs_to_logits_ratio, 320)
        self.assertEqual(config.minimum_input_samples, 400)
        self.assertEqual(config.feature_output_length(16_000), 49)
        self.assertEqual(config.extra_config["model_type"], "wav2vec2")
        self.assertEqual(
            Wav2Vec2Config.from_dict(config.to_dict()).conv_kernel,
            config.conv_kernel,
        )

    def test_invalid_dimensions_and_alias_conflicts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            Wav2Vec2Config(
                conv_dim=(8, 8),
                conv_stride=(2, ),
                conv_kernel=(3, 3),
            )
        with self.assertRaisesRegex(ValueError, "divisible"):
            Wav2Vec2Config(hidden_size=10, num_attention_heads=3)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            Wav2Vec2Config.from_dict({
                "hidden_dropout": 0.1,
                "hidden_dropout_prob": 0.2,
            })
        with self.assertRaisesRegex(ValueError, "num_feat_extract_layers"):
            Wav2Vec2Config.from_dict({
                "conv_dim": [8, 8],
                "num_feat_extract_layers": 3,
            })
        with self.assertRaisesRegex(ValueError, "num_feat_extract_layers"):
            Wav2Vec2Config.from_dict({"num_feat_extract_layers": 6})
        with self.assertRaisesRegex(ValueError, "language-adapter"):
            Wav2Vec2Config.from_dict({"add_adapter": True})


@unittest.skipUnless(torch is not None, "Native Wav2Vec2 uses PyTorch")
class Wav2Vec2ModelTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1337)
        self.config = _tiny_config()
        self.waveforms = torch.randn(2, 30)
        self.attention_mask = torch.tensor([
            [1] * 24 + [0] * 6,
            [1] * 30,
        ])

    def test_forward_ctc_loss_and_backward_are_finite(self):
        model = Wav2Vec2ForCTC(self.config)
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
        )

        self.assertEqual(tuple(output.logits.shape), (2, 7, 8))
        torch.testing.assert_close(
            output.input_lengths,
            torch.tensor([5, 7]),
        )
        torch.testing.assert_close(
            output.feature_attention_mask.sum(dim=-1),
            output.input_lengths,
        )
        self.assertEqual(len(output.hidden_states), 3)
        self.assertEqual(len(output.attentions), 2)
        self.assertEqual(output.executed_layers, (True, True))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.isfinite(output.loss))

        output.loss.backward()
        for parameter in (
                model.wav2vec2.feature_extractor.conv_layers[0].conv.weight,
                model.wav2vec2.encoder.layers[0].attention.q_proj.weight,
                model.lm_head.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_padding_values_cannot_change_valid_logits(self):
        model = Wav2Vec2ForCTC(self.config).eval()
        first = self.waveforms[:1].clone()
        second = first.clone()
        second[:, 24:] = torch.randn_like(second[:, 24:]) * 100.0
        mask = self.attention_mask[:1]

        with torch.no_grad():
            first_output = model(first, mask)
            second_output = model(second, mask)

        valid_frames = int(first_output.input_lengths.item())
        torch.testing.assert_close(
            first_output.logits[:, :valid_frames],
            second_output.logits[:, :valid_frames],
            rtol=0,
            atol=0,
        )

    def test_short_padded_audio_remains_valid_with_spec_augment(self):
        config = _tiny_config(
            apply_spec_augment=True,
            mask_time_prob=0.5,
            mask_time_length=4,
            mask_time_min_masks=2,
        )
        model = Wav2Vec2ForCTC(config).train()
        waveforms = torch.randn(2, 30)
        attention_mask = torch.tensor([
            [1] * 10 + [0] * 20,
            [1] * 30,
        ])

        output = model(
            waveforms,
            attention_mask,
            generator=torch.Generator().manual_seed(9),
        )

        torch.testing.assert_close(
            output.input_lengths,
            torch.tensor([2, 7]),
        )
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_attention_mask_must_be_right_padded(self):
        invalid_mask = self.attention_mask.clone()
        invalid_mask[0, 10] = 0
        invalid_mask[0, 11] = 1

        with self.assertRaisesRegex(ValueError, "right-padded"):
            Wav2Vec2ForCTC(self.config)(
                self.waveforms,
                invalid_mask,
            )

    def test_public_length_helpers_reject_non_integer_lengths(self):
        from voicehub.models.asr_wav2vec2.native import downsample_wav2vec2_lengths, feature_attention_mask

        with self.assertRaisesRegex(TypeError, "integer dtype"):
            downsample_wav2vec2_lengths(
                torch.tensor([30.0]),
                self.config,
            )
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            feature_attention_mask(
                7,
                torch.tensor([7.0]),
            )

    def test_bidirectional_encoder_explicitly_rejects_kv_cache(self):
        model = Wav2Vec2ForCTC(self.config).eval()
        with self.assertRaisesRegex(ValueError, "bidirectional"):
            model(
                self.waveforms,
                self.attention_mask,
                use_cache=True,
            )
        with self.assertRaisesRegex(ValueError, "past_key_values"):
            model(
                self.waveforms,
                self.attention_mask,
                past_key_values=(),
            )

    def test_layerdrop_is_training_only_and_reports_execution(self):
        config = _tiny_config(layerdrop=1.0)
        model = Wav2Vec2ForCTC(config)

        training_output = model(
            self.waveforms,
            self.attention_mask,
        )
        self.assertEqual(training_output.executed_layers, (False, False))

        evaluation_output = model.eval()(
            self.waveforms,
            self.attention_mask,
        )
        self.assertEqual(evaluation_output.executed_layers, (True, True))
        self.assertTrue(torch.isfinite(evaluation_output.logits).all())

    def test_stable_layer_norm_variant_preserves_public_shapes(self):
        config = _tiny_config(do_stable_layer_norm=True)
        model = Wav2Vec2ForCTC(config).eval()

        with torch.no_grad():
            output = model(
                self.waveforms,
                self.attention_mask,
            )

        self.assertEqual(tuple(output.logits.shape), (2, 7, 8))
        self.assertEqual(
            set(model.state_dict()),
            set(native_wav2vec2_tensor_shapes(config)),
        )

    def test_freezing_scopes_leave_ctc_head_trainable(self):
        model = Wav2Vec2ForCTC(self.config)
        model.freeze_base_model()

        self.assertFalse(any(parameter.requires_grad for parameter in model.wav2vec2.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.lm_head.parameters()))

    def test_sequence_classifier_masks_padding_and_backpropagates(self):
        config = _tiny_config(
            num_labels=3,
            classifier_proj_size=5,
            use_weighted_layer_sum=True,
        )
        model = Wav2Vec2ForSequenceClassification(config)
        output = model(
            self.waveforms,
            self.attention_mask,
            labels=torch.tensor([1, 2]),
            output_hidden_states=True,
        )

        self.assertEqual(tuple(output.logits.shape), (2, 3))
        self.assertEqual(len(output.hidden_states), 3)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.layer_weights.grad)
        self.assertIsNotNone(model.projector.weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)

    def test_frame_classifier_masks_padded_labels_and_backpropagates(self):
        config = _tiny_config(num_labels=2)
        model = Wav2Vec2ForAudioFrameClassification(config)
        labels = torch.zeros((2, 7), dtype=torch.long)
        labels[1, 2:4] = 1
        output = model(
            self.waveforms,
            self.attention_mask,
            labels=labels,
        )

        self.assertEqual(tuple(output.logits.shape), (2, 7, 2))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.classifier.weight.grad)

    def test_classifier_state_names_match_the_checkpoint_contract(self):
        sequence_config = _tiny_config(
            num_labels=3,
            classifier_proj_size=5,
            use_weighted_layer_sum=True,
        )
        frame_config = _tiny_config(num_labels=2)

        sequence = Wav2Vec2ForSequenceClassification(sequence_config)
        frame = Wav2Vec2ForAudioFrameClassification(frame_config)

        self.assertEqual(
            {
                name: tuple(value.shape)
                for name, value in sequence.state_dict().items()
            },
            native_wav2vec2_sequence_classification_tensor_shapes(sequence_config),
        )
        self.assertEqual(
            {
                name: tuple(value.shape)
                for name, value in frame.state_dict().items()
            },
            native_wav2vec2_frame_classification_tensor_shapes(frame_config),
        )


@unittest.skipUnless(torch is not None, "Native Wav2Vec2 uses PyTorch")
class Wav2Vec2CheckpointTests(unittest.TestCase):

    def test_official_safetensors_header_inventory_is_frozen(self):
        shapes = huggingface_wav2vec2_tensor_shapes(Wav2Vec2Config())

        self.assertEqual(len(shapes), 212)
        self.assertEqual(
            safetensors_header_fingerprint(shapes),
            FACEBOOK_WAV2VEC2_BASE_960H_HEADER_FINGERPRINT,
        )
        self.assertEqual(
            FACEBOOK_WAV2VEC2_BASE_960H_REVISION,
            "22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
        )

    def test_huggingface_mapping_strictly_loads_every_native_tensor(self):
        config = _tiny_config()
        torch.manual_seed(17)
        reference = Wav2Vec2ForCTC(config)
        reference_state = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}
        mapping = huggingface_wav2vec2_tensor_mapping(config)
        source = {source_name: reference_state[target_name] for source_name, target_name in mapping}
        source["wav2vec2.masked_spec_embed"] = torch.zeros(config.hidden_size)

        torch.manual_seed(99)
        restored = Wav2Vec2ForCTC(config)
        report = HuggingFaceWav2Vec2CheckpointAdapter().load(
            restored,
            source,
            config.to_dict(),
            strict=True,
        )

        self.assertTrue(report.is_compatible, report.summary())
        self.assertEqual(
            report.ignored_sources,
            ("wav2vec2.masked_spec_embed", ),
        )
        self.assertEqual(set(report.loaded), set(reference_state))
        for name, expected in reference_state.items():
            torch.testing.assert_close(
                restored.state_dict()[name],
                expected,
                rtol=0,
                atol=0,
            )

    def test_native_state_shapes_match_checkpoint_contract(self):
        config = _tiny_config(feat_extract_norm="layer", conv_bias=True)
        model = Wav2Vec2ForCTC(config)

        actual = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
        self.assertEqual(actual, native_wav2vec2_tensor_shapes(config))

    def test_probe_requires_wav2vec2_safetensors(self):
        config = Wav2Vec2Config().to_dict()
        adapter = HuggingFaceWav2Vec2CheckpointAdapter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(adapter.probe(
                (root / "model.safetensors", ),
                config,
            ))
            self.assertFalse(adapter.probe(
                (root / "pytorch_model.bin", ),
                config,
            ))

    def test_classification_adapter_strictly_loads_frame_head(self):
        config = _tiny_config(num_labels=2)
        config_values = {
            **config.to_dict(),
            "model_type": "wav2vec2",
            'architectures': ["Wav2Vec2ForAudioFrameClassification"],
        }
        reference = Wav2Vec2ForAudioFrameClassification(config)
        source = {name: value.detach().clone() for name, value in reference.state_dict().items()}
        restored = Wav2Vec2ForAudioFrameClassification(config)

        report = (
            HuggingFaceWav2Vec2ClassificationCheckpointAdapter().load(
                restored,
                source,
                config_values,
                strict=True,
            ))

        self.assertTrue(report.is_compatible, report.summary())
        for name, expected in source.items():
            torch.testing.assert_close(
                restored.state_dict()[name],
                expected,
                rtol=0,
                atol=0,
            )


@unittest.skipUnless(torch is not None, "Native Wav2Vec2 uses PyTorch")
class Wav2Vec2ArchitectureSpecTests(unittest.TestCase):

    def test_spec_is_lazy_pinned_and_resolvable(self):
        spec = create_wav2vec2_architecture_spec()

        self.assertEqual(spec.architecture_id, "wav2vec2")
        self.assertEqual(
            spec.upstream_revision,
            TRANSFORMERS_WAV2VEC2_REVISION,
        )
        self.assertTrue(spec.capabilities.training)
        self.assertTrue(spec.capabilities.has_feature("no-kv-cache"))
        self.assertIs(
            spec.resolve_component("model"),
            Wav2Vec2ForCTC,
        )
        self.assertIs(
            spec.resolve_component("config"),
            Wav2Vec2Config,
        )

    def test_registration_supports_family_aliases(self):
        registry = ArchitectureRegistry()
        spec = register_wav2vec2_architecture(registry=registry)

        self.assertIs(registry.get("wav2vec2-ctc"), spec)
        self.assertIs(registry.get("native-wav2vec2"), spec)

    def test_native_modules_do_not_import_external_architecture_runtimes(self):
        package = (Path(__file__).parents[1] / 'voicehub/models/asr_wav2vec2/native')
        forbidden = {
            "huggingface_hub",
            "numpy",
            "safetensors",
            "torchaudio",
            "transformers",
        }
        imported_roots = set()
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.add(node.module.partition(".")[0])

        self.assertTrue(forbidden.isdisjoint(imported_roots))


if __name__ == "__main__":
    unittest.main()
