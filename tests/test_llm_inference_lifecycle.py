import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from voicehub.models.csm.modeling import CSMForTextToSpeech
from voicehub.models.llasa.modeling import LlasaForTextToSpeech
from voicehub.models.neutts.modeling import NeuTTSForTextToSpeech
from voicehub.models.orpheustts.modeling import OrpheusTTSForTextToSpeech
from voicehub.models.outetts.inference import OuteTTSForTextToSpeech
from voicehub.models.qwen3tts.modeling import Qwen3TTSForTextToSpeech


class _EvalModule:

    def __init__(self, *, config=None):
        self.config = config
        self.training = True
        self.eval_calls = 0

    def eval(self):
        self.training = False
        self.eval_calls += 1
        return self


class LLMInferenceLifecycleTests(unittest.TestCase):

    def _assert_transition(
        self,
        wrapper,
        *,
        modules,
        cache_configs,
    ):
        runtime = wrapper.model
        wrapper._training_ready = True
        wrapper._inference_ready = False

        wrapper.load()

        self.assertIs(wrapper.model, runtime)
        self.assertTrue(wrapper._inference_ready)
        self.assertFalse(wrapper._training_ready)
        for module in modules:
            self.assertFalse(module.training)
            self.assertEqual(module.eval_calls, 1)
        for config in cache_configs:
            self.assertTrue(config.use_cache)

    def test_orpheus_transition_preserves_model_and_codec(self):
        model_config = SimpleNamespace(use_cache=False)
        language_model = _EvalModule(config=model_config)
        codec = _EvalModule()
        wrapper = OrpheusTTSForTextToSpeech(device="cpu")
        wrapper.model = language_model
        wrapper.codec = codec

        self._assert_transition(
            wrapper,
            modules=(language_model, codec),
            cache_configs=(model_config, ),
        )

    def test_csm_transition_evaluates_vendored_runtime_components(self):
        model_config = SimpleNamespace(use_cache=False)
        language_model = _EvalModule(config=model_config)
        audio_tokenizer = _EvalModule()
        watermarker = _EvalModule()
        runtime = SimpleNamespace(
            _model=language_model,
            _audio_tokenizer=audio_tokenizer,
            _watermarker=watermarker,
            config=model_config,
        )
        wrapper = CSMForTextToSpeech(device="cpu")
        wrapper.model = runtime

        self._assert_transition(
            wrapper,
            modules=(
                language_model,
                audio_tokenizer,
                watermarker,
            ),
            cache_configs=(model_config, ),
        )

    def test_llasa_transition_preserves_model_and_codec(self):
        model_config = SimpleNamespace(use_cache=False)
        language_model = _EvalModule(config=model_config)
        codec = _EvalModule()
        wrapper = LlasaForTextToSpeech(device="cpu")
        wrapper.model = language_model
        wrapper.codec = codec

        self._assert_transition(
            wrapper,
            modules=(language_model, codec),
            cache_configs=(model_config, ),
        )

    def test_oute_transition_preserves_nested_language_model(self):
        model_config = SimpleNamespace(use_cache=False)
        language_model = _EvalModule(config=model_config)
        runtime = SimpleNamespace(model=SimpleNamespace(model=language_model), )
        wrapper = OuteTTSForTextToSpeech(device="cpu")
        wrapper.model = runtime

        self._assert_transition(
            wrapper,
            modules=(language_model, ),
            cache_configs=(model_config, ),
        )

    def test_neutts_transition_preserves_backbone_and_codec(self):
        model_config = SimpleNamespace(use_cache=False)
        backbone = _EvalModule(config=model_config)
        codec = _EvalModule()
        runtime = SimpleNamespace(
            backbone=backbone,
            codec=codec,
        )
        wrapper = NeuTTSForTextToSpeech(device="cpu")
        wrapper.model = runtime

        self._assert_transition(
            wrapper,
            modules=(backbone, codec),
            cache_configs=(model_config, ),
        )

    def test_qwen_transition_restores_nested_cache_configs(self):
        code_predictor_config = SimpleNamespace(use_cache=False)
        talker_config = SimpleNamespace(
            use_cache=False,
            code_predictor_config=code_predictor_config,
        )
        model_config = SimpleNamespace(
            use_cache=False,
            talker_config=talker_config,
        )
        language_model = _EvalModule(config=model_config)
        runtime = SimpleNamespace(model=language_model)
        wrapper = Qwen3TTSForTextToSpeech(device="cpu")
        wrapper.model = runtime

        self._assert_transition(
            wrapper,
            modules=(language_model, ),
            cache_configs=(
                model_config,
                talker_config,
                code_predictor_config,
            ),
        )


class LLMInferenceRegressionTests(unittest.TestCase):

    def test_qwen_unknown_loaded_role_is_actionable(self):
        wrapper = Qwen3TTSForTextToSpeech(device="cpu")
        wrapper.model = SimpleNamespace(model=SimpleNamespace(tts_model_type="future_role"), )

        with self.assertRaisesRegex(
                ValueError,
                "tts_model_type.*future_role.*base, custom_voice, voice_design",
        ):
            wrapper._resolve_generation_mode(
                "auto",
                speaker_audio_path=None,
            )

    def test_oute_suffixless_output_defaults_to_wav(self):
        self.assertEqual(
            OuteTTSForTextToSpeech._normalize_output_file("speech"),
            "speech.wav",
        )
        self.assertEqual(
            OuteTTSForTextToSpeech._normalize_output_file("speech.flac"),
            "speech.flac",
        )
        self.assertIsNone(OuteTTSForTextToSpeech._normalize_output_file(None), )

    def test_csm_audio_length_must_be_finite_and_positive(self):
        for value in (float("nan"), float("inf"), float("-inf"), 0):
            with self.subTest(value=value):
                wrapper = CSMForTextToSpeech(device="cpu")
                with self.assertRaisesRegex(
                        ValueError,
                        "finite and greater than zero",
                ):
                    wrapper.generate(
                        "hello",
                        max_audio_length_ms=value,
                    )
                self.assertFalse(wrapper.is_loaded)

    def test_csm_zero_temperature_uses_greedy_transformers_generation(self):
        processor = Mock()
        processor.apply_chat_template.return_value = {}
        backend_model = Mock()
        backend_model.generate.return_value = SimpleNamespace(audio=[[0.1, -0.1]], )
        wrapper = CSMForTextToSpeech(device="cpu")
        wrapper.model = backend_model
        wrapper._training_backend = SimpleNamespace(processor=processor)

        audio, context_segments = wrapper._generate_transformers(
            "hello",
            speaker=0,
            speaker_audio_path=None,
            reference_text=None,
            max_audio_length_ms=1_000,
            temperature=0,
            top_k=50,
            generation_options={
                "do_sample": True,
                "temperature": 0.5,
                "depth_decoder_do_sample": True,
                "depth_decoder_temperature": 0.5,
            },
        )

        self.assertEqual(audio, [0.1, -0.1])
        self.assertEqual(context_segments, 0)
        generation_kwargs = backend_model.generate.call_args.kwargs
        self.assertFalse(generation_kwargs["do_sample"])
        self.assertFalse(generation_kwargs["depth_decoder_do_sample"])
        for option in (
                "temperature",
                "top_k",
                "depth_decoder_temperature",
                "depth_decoder_top_k",
        ):
            self.assertNotIn(option, generation_kwargs)


if __name__ == "__main__":
    unittest.main()
