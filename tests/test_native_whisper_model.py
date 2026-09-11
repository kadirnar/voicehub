from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from voicehub.models.asr_whisper_native.native import (
        HuggingFaceWhisperCheckpointAdapter,
        OpenAIWhisperCheckpointAdapter,
        WhisperConfig,
        WhisperModel,
        huggingface_whisper_tensor_mapping,
        native_whisper_tensor_names,
        openai_whisper_tensor_mapping,
    )


def _tiny_config():
    return WhisperConfig(
        vocab_size=32,
        num_mel_bins=4,
        d_model=8,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=24,
        decoder_layers=2,
        decoder_attention_heads=2,
        decoder_ffn_dim=24,
        max_source_positions=4,
        max_target_positions=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        use_cache=True,
    )


@unittest.skipUnless(torch is not None, "Native Whisper uses PyTorch")
class WhisperConfigurationTests(unittest.TestCase):

    def test_openai_dimensions_and_huggingface_fields_round_trip(self):
        dimensions = {
            "n_mels": 4,
            "n_audio_ctx": 6,
            "n_audio_state": 8,
            "n_audio_head": 2,
            "n_audio_layer": 2,
            "n_vocab": 32,
            "n_text_ctx": 10,
            "n_text_state": 8,
            "n_text_head": 2,
            "n_text_layer": 3,
            "decoder_start_token_id": 5,
            "model_type": "whisper",
        }

        config = WhisperConfig.from_dict(dimensions)

        self.assertEqual(config.d_model, 8)
        self.assertEqual(config.encoder_ffn_dim, 32)
        self.assertEqual(config.decoder_ffn_dim, 32)
        self.assertEqual(config.decoder_layers, 3)
        self.assertEqual(config.extra_config["model_type"], "whisper")
        self.assertEqual(
            config.to_openai_dimensions(),
            {
                name: value
                for name, value in dimensions.items() if name.startswith("n_")
            },
        )
        restored = WhisperConfig.from_dict(config.to_dict())
        self.assertEqual(restored.to_openai_dimensions(), config.to_openai_dimensions())

    def test_conflicting_or_invalid_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            WhisperConfig.from_dict({"d_model": 8, "n_audio_state": 16})
        with self.assertRaisesRegex(ValueError, "Conflicting|n_audio_state"):
            WhisperConfig.from_dict({
                "n_audio_state": 8,
                "n_text_state": 16,
            })
        with self.assertRaisesRegex(ValueError, "divisible"):
            WhisperConfig(d_model=10, encoder_attention_heads=3)


@unittest.skipUnless(torch is not None, "Native Whisper uses PyTorch")
class WhisperModelTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)
        self.config = _tiny_config()
        self.features = torch.randn(2, 4, 8)
        self.tokens = torch.tensor([
            [1, 7, 8, 9],
            [1, 4, 3, 2],
        ])

    def test_forward_loss_and_backward_are_finite(self):
        model = WhisperModel(self.config)
        labels = torch.tensor([
            [7, 8, 9, -100],
            [4, 3, 2, -100],
        ])

        output = model(
            input_features=self.features,
            labels=labels,
            use_cache=False,
        )

        self.assertEqual(tuple(output.logits.shape), (2, 4, 32))
        self.assertEqual(tuple(output.encoder_last_hidden_state.shape), (2, 4, 8))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for parameter in (
                model.encoder.conv1.weight,
                model.decoder.token_embedding.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

        expected_logits = torch.nn.functional.linear(
            output.decoder_last_hidden_state,
            model.decoder.token_embedding.weight,
        ).float()
        torch.testing.assert_close(output.logits, expected_logits)
        self.assertNotIn("proj_out.weight", model.state_dict())

    def test_incremental_cache_matches_full_decoder(self):
        model = WhisperModel(self.config).eval()
        with torch.no_grad():
            encoded = model.encode(self.features)
            full = model.decode(
                self.tokens,
                encoded,
                use_cache=False,
            )
            prefix = model.decode(
                self.tokens[:, :3],
                encoded,
                use_cache=True,
            )
            incremental = model.decode(
                self.tokens[:, 3:],
                encoded,
                past_key_values=prefix.past_key_values,
                use_cache=True,
            )

        self.assertIsNotNone(prefix.past_key_values)
        self.assertIsNotNone(incremental.past_key_values)
        self.assertEqual(len(incremental.past_key_values), self.config.decoder_layers)
        for layer in incremental.past_key_values:
            self.assertEqual(layer.self_attention.sequence_length, 4)
            self.assertEqual(
                layer.cross_attention.sequence_length,
                self.config.max_source_positions,
            )
        torch.testing.assert_close(
            incremental.logits,
            full.logits[:, -1:],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_variable_length_audio_mask_is_supported(self):
        model = WhisperModel(self.config).eval()
        features = self.features[:, :, :7]
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ])

        output = model(
            features,
            self.tokens,
            attention_mask=mask,
            use_cache=False,
        )

        self.assertEqual(tuple(output.encoder_last_hidden_state.shape), (2, 4, 8))
        self.assertTrue(torch.isfinite(output.logits).all())


@unittest.skipUnless(torch is not None, "Native Whisper uses PyTorch")
class WhisperCheckpointTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(11)
        self.config = _tiny_config()
        self.reference = WhisperModel(self.config)
        self.reference_state = {
            name: tensor.detach().clone()
            for name, tensor in self.reference.state_dict().items()
        }

    def _assert_loaded(self, adapter, source, config):
        torch.manual_seed(99)
        loaded = WhisperModel(self.config)
        report = adapter.load(loaded, source, config, strict=True)

        self.assertTrue(report.is_compatible, report.summary())
        self.assertEqual(set(report.loaded), set(self.reference_state))
        for name, expected in self.reference_state.items():
            torch.testing.assert_close(
                loaded.state_dict()[name],
                expected,
                rtol=0,
                atol=0,
            )

    def test_openai_mapping_has_complete_tensor_coverage(self):
        mapping = openai_whisper_tensor_mapping(self.config)
        self.assertEqual(
            {target
             for _, target in mapping},
            set(self.reference_state),
        )
        self.assertEqual(
            set(native_whisper_tensor_names(self.config)),
            set(self.reference_state),
        )
        source = {source_name: self.reference_state[target_name] for source_name, target_name in mapping}
        self._assert_loaded(
            OpenAIWhisperCheckpointAdapter(),
            source,
            self.config.to_openai_dimensions(),
        )

    def test_huggingface_safetensors_mapping_has_complete_coverage(self):
        config = self.config.to_dict()
        config['architectures'] = ["WhisperForConditionalGeneration"]
        mapping = huggingface_whisper_tensor_mapping(config)
        self.assertEqual(
            {target
             for _, target in mapping},
            set(self.reference_state),
        )
        self.assertEqual(
            len({source
                 for source, _ in mapping}),
            len(self.reference_state),
        )
        source = {source_name: self.reference_state[target_name] for source_name, target_name in mapping}
        self._assert_loaded(
            HuggingFaceWhisperCheckpointAdapter(),
            source,
            config,
        )

    def test_bare_huggingface_model_uses_an_unprefixed_namespace(self):
        config = self.config.to_dict()
        config['architectures'] = ["WhisperModel"]
        mapping = huggingface_whisper_tensor_mapping(
            config,
            source_prefix="",
        )
        source = {source_name: self.reference_state[target_name] for source_name, target_name in mapping}

        self._assert_loaded(
            HuggingFaceWhisperCheckpointAdapter(),
            source,
            config,
        )

    def test_checkpoint_probes_do_not_import_upstream_runtimes(self):
        openai_config = self.config.to_openai_dimensions()
        hf_config = self.config.to_dict()
        hf_config["model_type"] = "whisper"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(OpenAIWhisperCheckpointAdapter().probe(
                (root / "tiny.pt", ),
                openai_config,
            ))
            self.assertTrue(
                HuggingFaceWhisperCheckpointAdapter().probe(
                    (root / "model.safetensors", ),
                    hf_config,
                ))


if __name__ == "__main__":
    unittest.main()
