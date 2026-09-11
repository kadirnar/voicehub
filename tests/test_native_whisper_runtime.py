from __future__ import annotations

import unittest
from types import SimpleNamespace

from voicehub.tokenization import Encoding

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from voicehub.generation import GenerationConfig
    from voicehub.models.asr_whisper_native.native import (
        WhisperDecodingConfig,
        WhisperGenerationAdapter,
        WhisperTokenSet,
        apply_whisper_suppression,
        apply_whisper_timestamp_rules,
        create_whisper_architecture_spec,
        register_whisper_architecture,
    )
    from voicehub.runtime.registry import ArchitectureRegistry


def _token_set(*, multilingual=True):
    return WhisperTokenSet(
        eot=15,
        sot=1,
        translate=2,
        transcribe=3,
        sot_lm=4,
        sot_prev=5,
        no_speech=6,
        no_timestamps=19,
        timestamp_begin=20,
        language_tokens={
            "en": 7,
            "fr": 8
        } if multilingual else {},
        non_speech_tokens=(9, ),
        blank_token_ids=(10, ),
    )


class _Tokenizer:

    def __init__(self):
        self.inputs = []

    def encode(self, text):
        self.inputs.append(text)
        if text == " ":
            return [10]
        return [12, 12]


class _EncodingTokenizer(_Tokenizer):

    def encode(self, text):
        self.inputs.append(text)
        token_ids = (12, 13) if text == " hello" else (14, )
        return Encoding(input_ids=token_ids)


class _Encoder:

    @staticmethod
    def downsample_attention_mask(
        attention_mask,
        *,
        batch_size,
        input_frames,
        device,
    ):
        if tuple(attention_mask.shape) != (batch_size, input_frames):
            raise ValueError("invalid fake mask")
        return attention_mask.to(device=device, dtype=torch.bool)[:, ::2]


class _ScriptedWhisper:

    def __init__(self, *, detected_language=8):
        self.config = SimpleNamespace(
            vocab_size=32,
            max_target_positions=16,
        )
        self.encoder = _Encoder()
        self.detected_language = detected_language
        self.calls = []

    def encode(self, input_features, *, attention_mask=None):
        del attention_mask
        return torch.zeros(
            input_features.shape[0],
            4,
            8,
            dtype=input_features.dtype,
            device=input_features.device,
        )

    def decode(
        self,
        input_ids,
        encoder_hidden_states,
        *,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
    ):
        del encoder_hidden_states, encoder_attention_mask
        self.calls.append({
            "tokens": input_ids.detach().clone(),
            "past": past_key_values,
            "use_cache": use_cache,
        })
        logits = torch.full(
            (*input_ids.shape, self.config.vocab_size),
            -20.0,
            device=input_ids.device,
        )
        if not use_cache:
            logits[:, -1, self.detected_language] = 30.0
            cache = None
        else:
            step = 0 if past_key_values is None else int(past_key_values)
            if step == 0:
                logits[:, -1, 9] = 30.0
                logits[:, -1, 11] = 20.0
            else:
                logits[:, -1, 15] = 30.0
            cache = step + 1
        return SimpleNamespace(logits=logits, past_key_values=cache)


@unittest.skipUnless(torch is not None, "Native Whisper decoding uses PyTorch")
class WhisperTokenPolicyTests(unittest.TestCase):

    def test_huggingface_generation_metadata_is_normalized(self):
        tokens = WhisperTokenSet.from_huggingface_config({
            "eos_token_id": 15,
            "decoder_start_token_id": 1,
            "no_timestamps_token_id": 19,
            "task_to_id": {
                "translate": 2,
                "transcribe": 3,
            },
            "lang_to_id": {
                "<|en|>": 7,
                "<|fr|>": 8,
            },
            "suppress_tokens": [9],
            "begin_suppress_tokens": [10, 15],
        })

        self.assertEqual(tokens.sot_lm, 16)
        self.assertEqual(tokens.sot_prev, 17)
        self.assertEqual(tokens.no_speech, 18)
        self.assertEqual(tokens.timestamp_begin, 20)
        self.assertEqual(tokens.language_token("<|fr|>"), 8)
        self.assertEqual(tokens.blank_token_ids, (10, 15))

    def test_english_huggingface_metadata_uses_standard_control_layout(self):
        tokens = WhisperTokenSet.from_huggingface_config({
            "eos_token_id": 15,
            "decoder_start_token_id": 16,
            "no_timestamps_token_id": 22,
            "is_multilingual": False,
            "task_to_id": None,
            "lang_to_id": None,
        })

        self.assertFalse(tokens.is_multilingual)
        self.assertEqual(tokens.translate, 17)
        self.assertEqual(tokens.transcribe, 18)
        self.assertEqual(tokens.sot_lm, 19)
        self.assertEqual(tokens.sot_prev, 20)
        self.assertEqual(tokens.no_speech, 21)
        self.assertEqual(tokens.timestamp_begin, 23)

    def test_permanent_and_first_step_suppression_are_non_mutating(self):
        logits = torch.zeros(2, 32)
        original = logits.clone()
        initial_history = torch.tensor([[1, 7, 3], [1, 8, 3]])

        processed = apply_whisper_suppression(
            logits,
            initial_history,
            suppress_tokens=(9, ),
            begin_suppress_tokens=(10, 15),
            sample_begin=3,
        )

        torch.testing.assert_close(logits, original)
        self.assertTrue(torch.isneginf(processed[:, 9]).all())
        self.assertTrue(torch.isneginf(processed[:, 10]).all())
        self.assertTrue(torch.isneginf(processed[:, 15]).all())

        later = apply_whisper_suppression(
            logits,
            torch.tensor([[1, 7, 3, 11], [1, 8, 3, 11]]),
            suppress_tokens=(9, ),
            begin_suppress_tokens=(10, 15),
            sample_begin=3,
        )
        self.assertTrue(torch.isfinite(later[:, 10]).all())
        self.assertTrue(torch.isfinite(later[:, 15]).all())

    def test_timestamp_rules_limit_initial_time_and_enforce_pairs(self):
        tokens = _token_set()
        logits = torch.zeros(1, 32)
        prompt = torch.tensor([[1, 7, 3]])

        initial = apply_whisper_timestamp_rules(
            logits,
            prompt,
            token_set=tokens,
            sample_begin=3,
            max_initial_timestamp_index=2,
        )

        self.assertTrue(torch.isneginf(initial[:, :20]).all())
        self.assertTrue(torch.isfinite(initial[:, 20:23]).all())
        self.assertTrue(torch.isneginf(initial[:, 23:]).all())

        after_single_timestamp = apply_whisper_timestamp_rules(
            logits,
            torch.tensor([[1, 7, 3, 11, 22]]),
            token_set=tokens,
            sample_begin=3,
            max_initial_timestamp_index=None,
        )
        self.assertTrue(torch.isneginf(after_single_timestamp[:, :15]).all())
        self.assertTrue(torch.isneginf(after_single_timestamp[:, 20:22]).all())
        self.assertTrue(torch.isfinite(after_single_timestamp[:, 22:]).all())

        after_timestamp_pair = apply_whisper_timestamp_rules(
            logits,
            torch.tensor([[1, 7, 3, 22, 23]]),
            token_set=tokens,
            sample_begin=3,
            max_initial_timestamp_index=None,
        )
        self.assertTrue(torch.isfinite(after_timestamp_pair[:, 11]).all())
        self.assertTrue(torch.isneginf(after_timestamp_pair[:, 20:]).all())

    def test_timestamp_probability_mass_can_force_a_timestamp(self):
        logits = torch.full((1, 32), -10.0)
        logits[:, 11] = 0.5
        logits[:, 20:] = 0.0

        processed = apply_whisper_timestamp_rules(
            logits,
            torch.tensor([[1, 7, 3, 11]]),
            token_set=_token_set(),
            sample_begin=3,
            max_initial_timestamp_index=None,
        )

        self.assertTrue(torch.isneginf(processed[:, :20]).all())
        self.assertTrue(torch.isfinite(processed[:, 20:]).all())


@unittest.skipUnless(torch is not None, "Native Whisper decoding uses PyTorch")
class WhisperGenerationAdapterTests(unittest.TestCase):

    def test_prompt_suppression_and_cache_are_integrated(self):
        model = _ScriptedWhisper()
        tokenizer = _Tokenizer()
        adapter = WhisperGenerationAdapter(
            model,
            _token_set(),
            tokenizer=tokenizer,
        )
        features = torch.randn(2, 4, 8)
        config = WhisperDecodingConfig(
            generation=GenerationConfig(
                max_new_tokens=2,
                use_cache=True,
            ),
            language="en",
            task="transcribe",
            prompt=(12, 12),
            prefix=(14, ),
        )

        output = adapter.generate(features, config=config)

        expected_prompt = torch.tensor([
            [5, 12, 12, 1, 7, 3, 19, 14],
            [5, 12, 12, 1, 7, 3, 19, 14],
        ])
        torch.testing.assert_close(output.sequences[:, :8], expected_prompt)
        torch.testing.assert_close(
            output.generated_sequences,
            torch.tensor([[11, 15], [11, 15]]),
        )
        self.assertTrue(output.generation.finished.all())
        self.assertEqual([call["tokens"].shape[1] for call in model.calls], [8, 1])
        self.assertIsNone(model.calls[0]["past"])
        self.assertEqual(model.calls[1]["past"], 1)
        self.assertEqual(output.cache, 2)

    def test_language_detection_and_string_prompt_use_public_protocols(self):
        model = _ScriptedWhisper(detected_language=8)
        tokenizer = _Tokenizer()
        adapter = WhisperGenerationAdapter(
            model,
            _token_set(),
            tokenizer=tokenizer,
        )
        config = WhisperDecodingConfig(
            generation=GenerationConfig(max_new_tokens=2),
            language=None,
            prompt="hello",
        )

        output = adapter.generate(torch.randn(1, 4, 8), config=config)

        self.assertEqual(output.language_token_ids.tolist(), [8])
        self.assertEqual(output.sequences[0, 0].item(), 5)
        self.assertIn(" hello", tokenizer.inputs)
        self.assertEqual(model.calls[0]["tokens"].tolist(), [[1]])
        self.assertFalse(model.calls[0]["use_cache"])
        self.assertTrue(model.calls[1]["use_cache"])

    def test_string_prompt_and_prefix_accept_native_encoding_results(self):
        tokenizer = _EncodingTokenizer()
        adapter = WhisperGenerationAdapter(
            _ScriptedWhisper(),
            _token_set(),
            tokenizer=tokenizer,
        )
        config = WhisperDecodingConfig(
            generation=GenerationConfig(max_new_tokens=2),
            language="en",
            prompt="hello",
            prefix="world",
        )

        output = adapter.generate(torch.randn(1, 4, 8), config=config)

        self.assertEqual(tokenizer.inputs, [" world", " hello", " "])
        self.assertEqual(
            output.sequences[0, :8].tolist(),
            [5, 12, 13, 1, 7, 3, 19, 14],
        )

    def test_language_task_and_context_validation_are_explicit(self):
        features = torch.randn(1, 4, 8)
        multilingual = WhisperGenerationAdapter(
            _ScriptedWhisper(),
            _token_set(),
        )
        with self.assertRaisesRegex(ValueError, "not supported"):
            multilingual.generate(
                features,
                config=WhisperDecodingConfig(
                    generation=GenerationConfig(max_new_tokens=2),
                    language="de",
                ),
            )

        english_only = WhisperGenerationAdapter(
            _ScriptedWhisper(),
            _token_set(multilingual=False),
        )
        with self.assertRaisesRegex(ValueError, "Translation"):
            english_only.generate(
                features,
                config=WhisperDecodingConfig(
                    generation=GenerationConfig(max_new_tokens=2),
                    task="translate",
                ),
            )
        with self.assertRaisesRegex(ValueError, "no room"):
            multilingual.generate(
                features,
                config=WhisperDecodingConfig(
                    generation=GenerationConfig(max_new_tokens=16),
                    language="en",
                ),
            )


@unittest.skipUnless(torch is not None, "Native Whisper discovery uses PyTorch tests")
class WhisperArchitectureRegistrationTests(unittest.TestCase):

    def test_specification_keeps_every_runtime_component_lazy(self):
        spec = create_whisper_architecture_spec()

        self.assertEqual(spec.architecture_id, "whisper")
        self.assertEqual(
            spec.model_builder.path,
            "voicehub.models.asr_whisper_native.native.modeling:WhisperModel",
        )
        self.assertEqual(
            spec.decoder.path,
            "voicehub.models.asr_whisper_native.native.decoding:WhisperGenerationAdapter",
        )
        self.assertTrue(spec.capabilities.training)
        self.assertTrue(spec.capabilities.supports_task("automatic-speech-recognition"))

    def test_registration_supports_isolated_registries_and_aliases(self):
        registry = ArchitectureRegistry()

        spec = register_whisper_architecture(registry=registry)

        self.assertIs(registry.get("whisper"), spec)
        self.assertIs(registry.get("native-whisper"), spec)
        self.assertIs(registry.get("openai-whisper"), spec)
        self.assertIs(registry.get("hf-whisper"), spec)


if __name__ == "__main__":
    unittest.main()
