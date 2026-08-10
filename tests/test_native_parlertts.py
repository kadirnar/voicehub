from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from voicehub.architectures.dac.configuration import DacConfig
from voicehub.architectures.parlertts.artifacts import resolve_parlertts_artifacts
from voicehub.architectures.parlertts.checkpoint import (
    export_parlertts_checkpoint,
    load_parlertts_checkpoint,
    tensor_inventory_fingerprint,
)
from voicehub.architectures.parlertts.configuration import (
    ParlerDecoderConfig,
    ParlerTTSArchitectureConfig,
    T5EncoderConfig,
)
from voicehub.architectures.parlertts.metadata import (
    PARLER_TTS_CHECKPOINT_LICENSE,
    PARLER_TTS_CHECKPOINT_REVISION,
    PARLER_TTS_HEADER_FINGERPRINT,
    PARLER_TTS_PARAMETER_COUNT,
    PARLER_TTS_SOURCE_REVISION,
    PARLER_TTS_TENSOR_COUNT,
)
from voicehub.architectures.parlertts.modeling import (
    ParlerTTSForCausalLM,
    ParlerTTSForConditionalGeneration,
    apply_delay_pattern_mask,
    build_delay_pattern_mask,
    prepare_audio_code_labels,
)
from voicehub.architectures.parlertts.processing import ParlerTextTokenizer
from voicehub.architectures.parlertts.registration import create_parlertts_architecture_spec
from voicehub.architectures.parlertts.t5 import NativeT5EncoderModel
from voicehub.models.parlertts.inference import ParlerTTSForTextToSpeech
from voicehub.models.parlertts.training import ParlerTTSTrainingAdapter
from voicehub.training.specs import get_training_spec

TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


def _tiny_decoder_config() -> ParlerDecoderConfig:
    return ParlerDecoderConfig(
        vocab_size=10,
        max_position_embeddings=64,
        num_hidden_layers=2,
        ffn_dim=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_cross_attention_key_value_heads=4,
        hidden_size=16,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        num_codebooks=2,
        pad_token_id=8,
        bos_token_id=9,
        eos_token_id=8,
    )


def _tiny_config() -> ParlerTTSArchitectureConfig:
    return ParlerTTSArchitectureConfig(
        text_encoder=T5EncoderConfig(
            vocab_size=32,
            d_model=16,
            d_kv=4,
            d_ff=24,
            num_layers=2,
            num_heads=4,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=16,
            dropout_rate=0.0,
        ),
        audio_encoder=DacConfig(
            encoder_hidden_size=4,
            downsampling_ratios=(2, 2),
            decoder_hidden_size=16,
            n_codebooks=2,
            codebook_size=8,
            codebook_dim=2,
            sampling_rate=8_000,
        ),
        decoder=_tiny_decoder_config(),
        vocab_size=32,
        decoder_start_token_id=9,
        pad_token_id=8,
        sampling_rate=8_000,
    )


class NativeParlerTTSGraphTests(unittest.TestCase):

    def test_released_graph_matches_the_remote_safetensors_header(self):
        with torch.device("meta"):
            model = ParlerTTSForConditionalGeneration(ParlerTTSArchitectureConfig())
        state = model.state_dict()
        self.assertEqual(len(state), PARLER_TTS_TENSOR_COUNT)
        self.assertEqual(
            sum(tensor.numel() for tensor in state.values()),
            PARLER_TTS_PARAMETER_COUNT,
        )
        inventory = {name: ("F32", tuple(tensor.shape)) for name, tensor in state.items()}
        self.assertEqual(
            tensor_inventory_fingerprint(inventory),
            PARLER_TTS_HEADER_FINGERPRINT,
        )
        self.assertEqual(
            tuple(state["text_encoder.shared.weight"].shape),
            (32_128, 1_024),
        )
        self.assertEqual(
            tuple(state["decoder.model.decoder.layers.23.encoder_attn.q_proj.weight"].shape),
            (1_024, 1_024),
        )
        self.assertEqual(
            tuple(state["audio_encoder.model.quantizer.quantizers.8."
                        "codebook.weight"].shape),
            (1_024, 8),
        )

    def test_provenance_distinguishes_source_and_weight_artifacts(self):
        spec = create_parlertts_architecture_spec()
        self.assertEqual(len(PARLER_TTS_SOURCE_REVISION), 40)
        self.assertEqual(len(PARLER_TTS_CHECKPOINT_REVISION), 40)
        self.assertEqual(PARLER_TTS_CHECKPOINT_LICENSE, "Apache-2.0")
        self.assertTrue(spec.capabilities.training)
        self.assertTrue(spec.metadata["full_finetuning_ready"])
        self.assertEqual(
            spec.metadata["reference_header_fingerprint"],
            PARLER_TTS_HEADER_FINGERPRINT,
        )
        self.assertEqual(
            spec.metadata["always_frozen_components"],
            ("audio_encoder", ),
        )

    def test_delay_pattern_round_trip_uses_every_codebook(self):
        initial = torch.tensor([
            [9, 1, 2],
            [9, 3, 4],
        ])
        delayed, mask = build_delay_pattern_mask(
            initial,
            bos_token_id=9,
            pad_token_id=8,
            max_length=8,
            num_codebooks=2,
        )
        completed = torch.full_like(mask, 7)
        completed[:, :delayed.shape[1]] = delayed
        constrained = apply_delay_pattern_mask(completed, mask)
        self.assertEqual(tuple(mask.shape), (2, 8))
        self.assertTrue((constrained[0, :1] == 9).all())
        self.assertTrue((constrained[1, :2] == 9).all())
        self.assertTrue((constrained[0, -1:] == 8).all())

    def test_delayed_codebook_objective_is_differentiable(self):
        model = ParlerTTSForConditionalGeneration(_tiny_config())
        description = torch.tensor([[3, 4, 1], [5, 1, 0]])
        description_mask = (description != 0).long()
        prompt = torch.tensor([[6, 7, 1], [3, 1, 0]])
        prompt_mask = (prompt != 0).long()
        labels = torch.tensor([
            [[1, 2], [2, 3], [8, 8]],
            [[2, 1], [3, 2], [8, 8]],
        ])
        output = model(
            description,
            attention_mask=description_mask,
            prompt_input_ids=prompt,
            prompt_attention_mask=prompt_mask,
            labels=labels,
        )
        self.assertEqual(output.loss.ndim, 0)
        self.assertEqual(len(output.per_codebook_losses), 2)
        output.loss.backward()
        self.assertIsNotNone(model.decoder.lm_heads[0].weight.grad)
        self.assertIsNotNone(model.embed_prompts.weight.grad)
        self.assertTrue(torch.isfinite(model.decoder.lm_heads[0].weight.grad).all())

    def test_sdpa_option_routes_decoder_attention_through_pytorch(self):
        eager = ParlerTTSForCausalLM(_tiny_decoder_config()).eval()
        model = ParlerTTSForCausalLM(
            _tiny_decoder_config(),
            attention_implementation="sdpa",
        ).eval()
        model.load_state_dict(eager.state_dict())
        input_ids = torch.tensor([[1, 2], [3, 4]])
        encoder_hidden = torch.randn(1, 3, 16)
        implementation = torch.nn.functional.scaled_dot_product_attention

        with torch.no_grad():
            expected = eager(
                input_ids,
                encoder_hidden_states=encoder_hidden,
            ).logits
        with patch(
                "torch.nn.functional.scaled_dot_product_attention",
                wraps=implementation,
        ) as sdpa, torch.no_grad():
            output = model(
                input_ids,
                encoder_hidden_states=encoder_hidden,
            )

        self.assertEqual(tuple(output.logits.shape), (2, 2, 10))
        self.assertGreaterEqual(sdpa.call_count, 4)
        torch.testing.assert_close(
            output.logits,
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_sdpa_option_routes_t5_attention_through_pytorch(self):
        config = _tiny_config().text_encoder
        eager = NativeT5EncoderModel(config).eval()
        model = NativeT5EncoderModel(
            config,
            attention_implementation="sdpa",
        ).eval()
        model.load_state_dict(eager.state_dict())
        input_ids = torch.tensor([[3, 4, 1]])
        implementation = torch.nn.functional.scaled_dot_product_attention

        with torch.no_grad():
            expected = eager(input_ids).last_hidden_state
        with patch(
                "torch.nn.functional.scaled_dot_product_attention",
                wraps=implementation,
        ) as sdpa, torch.no_grad():
            output = model(input_ids)

        self.assertEqual(tuple(output.last_hidden_state.shape), (1, 3, 16))
        self.assertEqual(sdpa.call_count, config.num_layers)
        torch.testing.assert_close(
            output.last_hidden_state,
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_raw_dac_codes_are_converted_to_delayed_training_labels(self):
        codes = torch.tensor([[[1, 2], [3, 4]]])
        labels = prepare_audio_code_labels(
            codes,
            bos_token_id=9,
            eos_token_id=8,
        )
        self.assertEqual(
            labels.tolist(),
            [[[1, 9], [2, 3], [8, 4], [8, 8]]],
        )

    def test_variable_dac_code_lengths_use_minus_100_batch_padding(self):
        codes = torch.tensor([
            [[1, 2, 3], [4, 5, 6]],
            [[2, 7, 7], [3, 7, 7]],
        ])
        labels = prepare_audio_code_labels(
            codes,
            bos_token_id=9,
            eos_token_id=8,
            audio_code_lengths=torch.tensor([3, 1]),
        )

        self.assertEqual(tuple(labels.shape), (2, 5, 2))
        self.assertEqual(labels[1, :3].tolist(), [[2, 9], [8, 3], [8, 8]])
        self.assertTrue((labels[1, 3:] == -100).all())

    def test_raw_waveform_lengths_trim_padded_dac_training_codes(self):
        config = _tiny_config()
        codes = torch.tensor([
            [[1, 2, 3], [4, 5, 6]],
            [[2, 7, 7], [3, 7, 7]],
        ])

        class AudioEncoder:

            def eval(self):
                return self

            @staticmethod
            def encode(audio):
                return codes.to(audio.device)

        wrapper = SimpleNamespace(
            model=SimpleNamespace(audio_encoder=AudioEncoder()),
            architecture_config=config,
        )
        prepared = ParlerTTSForTextToSpeech.prepare_training_inputs(
            wrapper,
            {
                "input_ids": torch.tensor([[1], [2]]),
                "prompt_input_ids": torch.tensor([[1], [2]]),
                "audio_values": torch.zeros(2, 12),
                "audio_lengths": torch.tensor([12, 4]),
            },
            phase="default",
        )

        self.assertEqual(tuple(prepared["labels"].shape), (2, 5, 2))
        self.assertTrue((prepared["labels"][1, 3:] == -100).all())
        self.assertNotIn("audio_values", prepared)
        self.assertNotIn("audio_lengths", prepared)

    def test_generation_honors_minimum_length_and_stops_without_extra_step(self):
        model = ParlerTTSForConditionalGeneration(_tiny_config()).eval()
        decoder_calls = 0

        def encode_text(input_ids, attention_mask=None):
            del attention_mask
            return torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                model.config.decoder.hidden_size,
            )

        def decoder_forward(input_ids, **kwargs):
            nonlocal decoder_calls
            del kwargs
            decoder_calls += 1
            logits = torch.full(
                (
                    input_ids.shape[0],
                    input_ids.shape[1],
                    model.config.decoder.vocab_size,
                ),
                -100.0,
            )
            logits[..., 1] = 0.0
            logits[..., model.config.decoder.eos_token_id] = 100.0
            return SimpleNamespace(logits=logits)

        def decode_codes(audio_codes):
            return torch.zeros(audio_codes.shape[0], 1, 4)

        model.encode_text = encode_text
        model.decoder.forward = decoder_forward
        model.audio_encoder.decode = decode_codes
        output = model.generate(
            torch.tensor([[3, 1]]),
            prompt_input_ids=torch.tensor([[4, 1]]),
            max_new_tokens=8,
            min_new_tokens=2,
            do_sample=False,
            top_k=None,
        )

        self.assertEqual(decoder_calls, 4)
        self.assertEqual(tuple(output.audio_values.shape), (1, 4))

    def test_training_adapter_freezes_dac_and_uses_native_loss(self):
        runtime = ParlerTTSForConditionalGeneration(_tiny_config())

        class Wrapper:
            model = runtime

            @staticmethod
            def load_for_training():
                return None

        adapter = ParlerTTSTrainingAdapter(
            Wrapper(),
            get_training_spec("parlertts"),
        ).setup()
        self.assertFalse(any(parameter.requires_grad for parameter in runtime.audio_encoder.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in runtime.text_encoder.parameters()))
        self.assertTrue(runtime.decoder.lm_heads[0].weight.requires_grad)
        batch = {
            "input_ids": torch.tensor([[3, 4, 1]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "prompt_input_ids": torch.tensor([[5, 6, 1]]),
            "prompt_attention_mask": torch.ones(1, 3, dtype=torch.long),
            "labels": torch.tensor([[[1, 2], [2, 3], [8, 8]]]),
        }
        output = adapter.execute_training_phase(adapter.create_training_context(batch))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(runtime.decoder.lm_heads[0].weight.grad)
        self.assertIsNotNone(runtime.text_encoder.shared.weight.grad)

    def test_training_adapter_supports_the_upstream_text_encoder_freeze(self):
        runtime = ParlerTTSForConditionalGeneration(_tiny_config())

        class Wrapper:
            model = runtime
            config = SimpleNamespace(freeze_text_encoder=True)

            @staticmethod
            def load_for_training():
                return None

        ParlerTTSTrainingAdapter(
            Wrapper(),
            get_training_spec("parlertts"),
        ).setup()

        self.assertFalse(any(parameter.requires_grad for parameter in runtime.text_encoder.parameters()))

    def test_safe_export_reloads_into_a_fresh_graph(self):
        source = ParlerTTSForConditionalGeneration(_tiny_config())
        target = ParlerTTSForConditionalGeneration(_tiny_config())
        with torch.no_grad():
            for index, parameter in enumerate(source.parameters(), start=1):
                parameter.fill_((index % 5) / 10)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            export_parlertts_checkpoint(source, checkpoint)
            report = load_parlertts_checkpoint(target, checkpoint)
        self.assertEqual(report.tensor_count, len(source.state_dict()))
        for name, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], expected)

    def test_training_export_is_a_complete_fresh_inference_snapshot(self):
        config = _tiny_config()
        runtime = ParlerTTSForConditionalGeneration(config)

        class SentencePiece:

            @staticmethod
            def save_pretrained(directory, *, filename):
                path = Path(directory) / filename
                path.write_bytes(b"native-tokenizer-fixture")
                return path

        class Tokenizer:
            sentencepiece = SentencePiece()

        class Wrapper:
            model = runtime
            architecture_config = config
            tokenizer = Tokenizer()
            artifacts = None

            @staticmethod
            def load_for_training():
                return None

        adapter = ParlerTTSTrainingAdapter(
            Wrapper(),
            get_training_spec("parlertts"),
        )
        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            root = Path(directory)
            self.assertEqual(
                {path.name
                 for path in root.iterdir()},
                {
                    "config.json",
                    "generation_config.json",
                    "model.safetensors",
                    "spiece.model",
                },
            )
            restored_config = ParlerTTSArchitectureConfig.from_dict(
                json.loads((root / "config.json").read_text(encoding="utf-8")))
            restored = ParlerTTSForConditionalGeneration(restored_config)
            load_parlertts_checkpoint(
                restored,
                root / "model.safetensors",
            )
            artifacts = resolve_parlertts_artifacts(
                root,
                verify_integrity=True,
                verify_checkpoint_integrity=True,
            )

        self.assertFalse(artifacts.official_snapshot)
        for name, expected in runtime.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], expected)

    def test_waveform_extraction_preserves_public_wrapper_contract(self):

        class Output:
            audio_values = torch.tensor([[[0.25, -0.5]]])

        waveform = ParlerTTSForTextToSpeech._extract_waveform(Output())
        self.assertEqual(waveform.tolist(), [0.25, -0.5])

    @unittest.skipUnless(
        TRANSFORMERS_AVAILABLE,
        "Upstream parity audit requires Transformers only in tests.",
    )
    def test_t5_encoder_is_bit_exact_to_upstream_on_a_tiny_graph(self):
        from transformers import T5Config, T5EncoderModel

        native_config = _tiny_config().text_encoder
        upstream_config = T5Config(
            **native_config.to_dict(),
            feed_forward_proj="gated-gelu",
        )
        # The released graph is pinned to eager T5 attention. Transformers
        # 5.15 selects SDPA by default, so keep the audit backend explicit.
        upstream_config._attn_implementation = "eager"
        upstream = T5EncoderModel(upstream_config).eval()
        native = NativeT5EncoderModel(native_config).eval()
        upstream_state = upstream.state_dict()
        native.load_state_dict(
            {
                name: value
                for name, value in upstream_state.items() if name in native.state_dict()
            },
            strict=True,
        )
        input_ids = torch.tensor([[3, 4, 1, 0], [8, 2, 0, 0]])
        attention_mask = (input_ids != 0).long()
        with torch.no_grad():
            expected = upstream(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
            actual = native(input_ids, attention_mask).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    @unittest.skipUnless(
        TRANSFORMERS_AVAILABLE,
        "Upstream parity audit requires the vendored reference runtime.",
    )
    def test_decoder_is_bit_exact_to_pinned_upstream_on_a_tiny_graph(self):
        from voicehub.models.parlertts.source.parler_tts import configuration_parler_tts, modeling_parler_tts

        native_config = _tiny_decoder_config()
        values = native_config.to_dict()
        values.pop("model_type")
        upstream_config = configuration_parler_tts.ParlerTTSDecoderConfig(
            **values,
            attn_implementation="eager",
        )
        upstream_config._attn_implementation = "eager"
        upstream = modeling_parler_tts.ParlerTTSForCausalLM(upstream_config).eval()
        native = ParlerTTSForCausalLM(native_config).eval()
        native.load_state_dict(upstream.state_dict(), strict=True)
        decoder_ids = torch.tensor(
            [[1, 2, 3], [3, 2, 1]],
            dtype=torch.long,
        )
        encoder_hidden = torch.randn(1, 4, 16)
        encoder_mask = torch.tensor([[1, 1, 1, 0]])
        prompt_hidden = torch.randn(1, 2, 16)
        prompt_mask = torch.ones(1, 2, dtype=torch.long)
        with torch.no_grad():
            expected = upstream(
                input_ids=decoder_ids,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_mask,
                prompt_hidden_states=prompt_hidden,
                prompt_attention_mask=prompt_mask,
                use_cache=False,
                return_dict=True,
            ).logits
            actual = native(
                decoder_ids,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_mask,
                prompt_hidden_states=prompt_hidden,
                prompt_attention_mask=prompt_mask,
            ).logits
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class ParlerTokenizerTests(unittest.TestCase):

    def test_frontend_appends_eos_and_pads_batches(self):

        class FakeSentencePiece:
            vocabulary_size = 16

            @staticmethod
            def encode_as_ids(text):
                return [len(text)] if len(text) == 1 else [len(text), 3]

        # The frontend intentionally requires the audited native tokenizer,
        # so exercise tensor collation through a constructor bypass while
        # keeping this test independent from a binary fixture.
        tokenizer = object.__new__(ParlerTextTokenizer)
        tokenizer.sentencepiece = FakeSentencePiece()
        tokenizer.eos_token_id = 1
        tokenizer.pad_token_id = 0
        tokenizer.model_vocabulary_size = 16
        batch = tokenizer(("a", "abcd"))
        self.assertEqual(batch.input_ids.tolist(), [[1, 1, 0], [4, 3, 1]])
        self.assertEqual(batch.attention_mask.tolist(), [[1, 1, 0], [1, 1, 1]])


if __name__ == "__main__":
    unittest.main()
