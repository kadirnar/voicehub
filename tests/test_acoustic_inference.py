import importlib
import importlib.util
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from inspect import signature
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from voicehub.models.chatterbox.modeling import ChatterboxForTextToSpeech
from voicehub.models.f5tts.modeling import F5TTSForTextToSpeech
from voicehub.models.gptsovits.modeling import GPTSoVITSForTextToSpeech
from voicehub.models.inflecttts.modeling import InflectTTSForTextToSpeech
from voicehub.models.kokoro.modeling import KokoroForTextToSpeech
from voicehub.models.melotts.modeling import MeloTTSForTextToSpeech
from voicehub.models.neutts.modeling import NeuTTSForTextToSpeech
from voicehub.models.openvoice.modeling import OpenVoiceForTextToSpeech
from voicehub.models.parlertts.modeling import ParlerTTSForTextToSpeech
from voicehub.models.styletts2.modeling import StyleTTS2ForTextToSpeech
from voicehub.models.supertonic.modeling import SupertonicForTextToSpeech
from voicehub.models.vibevoice.modeling import VibeVoiceForTextToSpeech
from voicehub.models.vui.modeling import VuiForTextToSpeech
from voicehub.models.xtts.modeling import XTTSForTextToSpeech
from voicehub.models.zonos2.modeling import Zonos2Config, Zonos2ForTextToSpeech
from voicehub.models.zonos.modeling import ZonosForTextToSpeech

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@contextmanager
def _temporary_modules(modules):
    """Override selected imports without discarding modules loaded in scope."""
    missing = object()
    originals = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class _EvalModule:

    def __init__(self):
        self.training = True
        self.eval_calls = 0

    def eval(self):
        self.training = False
        self.eval_calls += 1
        return self

    def train(self, mode=True):
        self.training = mode
        return self


class PreloadValidationTests(unittest.TestCase):

    def assert_rejected_without_loading(
        self,
        model,
        expected_exception,
        expected_message,
        **generation_options,
    ):
        with self.assertRaisesRegex(expected_exception, expected_message):
            model.generate("Hello", **generation_options)
        self.assertFalse(model.is_loaded)

    def test_required_reference_files_are_checked_before_loading(self):
        cases = (
            (
                F5TTSForTextToSpeech(),
                {
                    "speaker_audio_path": "/missing/reference.wav"
                },
                "reference audio",
            ),
            (
                GPTSoVITSForTextToSpeech(),
                {
                    "text_language": "en",
                    "prompt_language": "en",
                    "speaker_audio_path": "/missing/reference.wav",
                },
                "reference audio",
            ),
            (
                OpenVoiceForTextToSpeech(),
                {
                    "speaker_audio_path": "/missing/reference.wav"
                },
                "reference audio",
            ),
            (
                XTTSForTextToSpeech(),
                {
                    "speaker_audio_path": "/missing/reference.wav"
                },
                "reference audio",
            ),
            (
                VibeVoiceForTextToSpeech(),
                {
                    "voice_prompt_path": "/missing/prompt.pt"
                },
                "voice prompt",
            ),
        )
        for model, options, message in cases:
            with self.subTest(model=model.config.model_type):
                self.assert_rejected_without_loading(
                    model,
                    FileNotFoundError,
                    message,
                    **options,
                )

    def test_backend_numeric_ranges_are_checked_before_loading(self):
        cases = (
            (ChatterboxForTextToSpeech(), {
                "max_new_tokens": 0
            }, "max_new_tokens"),
            (KokoroForTextToSpeech(), {
                "split_pattern": "["
            }, "split_pattern"),
            (StyleTTS2ForTextToSpeech(), {
                "alpha": 1.1
            }, "alpha"),
            (InflectTTSForTextToSpeech(), {
                "speed": 0.25
            }, "speed"),
            (SupertonicForTextToSpeech(), {
                "total_steps": 0
            }, "total_steps"),
            (VuiForTextToSpeech(), {
                "top_p": 0
            }, "top_p"),
            (VuiForTextToSpeech(), {
                "max_chunk_retries": 0
            }, "max_chunk_retries"),
            (
                ZonosForTextToSpeech(),
                {
                    "emotion": [0.0] * 8
                },
                "emotion",
            ),
        )
        for model, options, message in cases:
            with self.subTest(model=model.config.model_type):
                self.assert_rejected_without_loading(
                    model,
                    ValueError,
                    message,
                    **options,
                )

    def test_supported_zero_sampling_boundaries_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.wav"
            reference.touch()
            GPTSoVITSForTextToSpeech()._validate_generation_inputs({
                "text_language": "en",
                "prompt_language": "en",
                "speaker_audio_path": reference,
                "top_p": 0,
                "temperature": 0,
            })

        Zonos2ForTextToSpeech()._validate_generation_inputs({
            "temperature": 0,
            "top_k": 0,
        })
        ZonosForTextToSpeech()._validate_generation_inputs({
            "speaking_rate": 0,
            "pitch_std": 0,
            "cfg_scale": 0,
        })

    def test_finite_acoustic_options_are_rejected_before_loading(self):
        for model in (
                InflectTTSForTextToSpeech(),
                SupertonicForTextToSpeech(),
                VuiForTextToSpeech(),
        ):
            with self.subTest(model=model.config.model_type):
                self.assert_rejected_without_loading(
                    model,
                    ValueError,
                    "Unsupported generation option",
                    misspelled_option=True,
                )

    def test_styletts_runtime_lifecycle_reaches_nested_modules(self):
        from voicehub.models.styletts2.runtime import StyleTTS2Runtime

        components = {
            "bert": _EvalModule(),
            "diffusion": _EvalModule(),
        }
        runtime = StyleTTS2Runtime.__new__(StyleTTS2Runtime)
        runtime.model = components
        model = StyleTTS2ForTextToSpeech(device="cpu")
        model.model = runtime
        model._training_ready = True

        model.load()

        self.assertTrue(model._inference_ready)
        self.assertFalse(model._training_ready)
        for component in components.values():
            self.assertFalse(component.training)
            self.assertEqual(component.eval_calls, 1)

    def test_zonos_rejects_cfg_scale_one_before_loading(self):
        self.assert_rejected_without_loading(
            ZonosForTextToSpeech(),
            ValueError,
            "cfg_scale",
            cfg_scale=1,
        )

    def test_styletts_rejects_non_finite_controls_before_loading(self):
        for option in ("alpha", "beta", "embedding_scale"):
            with self.subTest(option=option):
                self.assert_rejected_without_loading(
                    StyleTTS2ForTextToSpeech(),
                    ValueError,
                    option,
                    **{option: float("nan")},
                )

    def test_model_specific_non_finite_controls_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.wav"
            reference.touch()
            cases = (
                (
                    F5TTSForTextToSpeech(),
                    {
                        "speaker_audio_path": reference,
                        "sway_sampling_coef": float("nan"),
                    },
                    "sway_sampling_coef",
                ),
                (
                    MeloTTSForTextToSpeech(),
                    {
                        "noise_scale": float("nan")
                    },
                    "noise_scale",
                ),
                (
                    OpenVoiceForTextToSpeech(),
                    {
                        "speaker_audio_path": reference,
                        "speed": float("nan"),
                    },
                    "speed",
                ),
                (
                    KokoroForTextToSpeech(),
                    {
                        "speed": float("nan")
                    },
                    "speed",
                ),
                (
                    VuiForTextToSpeech(),
                    {
                        "top_p": float("nan")
                    },
                    "top_p",
                ),
            )
            for model, options, message in cases:
                with self.subTest(model=model.config.model_type):
                    self.assert_rejected_without_loading(
                        model,
                        ValueError,
                        message,
                        **options,
                    )

    def test_zonos2_rejects_token_only_configuration_before_loading(self):
        model = Zonos2ForTextToSpeech(Zonos2Config(decode_audio=False))

        self.assert_rejected_without_loading(
            model,
            ValueError,
            "decode_audio",
        )

    def test_supertonic_rejects_unsupported_language_before_loading(self):
        self.assert_rejected_without_loading(
            SupertonicForTextToSpeech(),
            ValueError,
            "Unsupported Supertonic language",
            language="not-a-language",
        )


class InferenceHelperTests(unittest.TestCase):

    def test_chatterbox_forwards_configurable_generation_limit(self):
        runtime = SimpleNamespace(generate=Mock(return_value=np.ones(32, dtype=np.float32)), )
        model = ChatterboxForTextToSpeech(device="cpu")
        model.model = runtime
        model.config.sample_rate = 24_000

        output = model._generate(
            "hello",
            max_new_tokens=1234,
            seed=7,
        )

        self.assertEqual(output.sample_rate, 24_000)
        runtime.generate.assert_called_once_with(
            "hello",
            audio_prompt_path=None,
            max_new_tokens=1234,
        )

    def test_stochastic_wrappers_expose_the_common_seed_option(self):
        wrappers = (
            GPTSoVITSForTextToSpeech,
            InflectTTSForTextToSpeech,
            KokoroForTextToSpeech,
            MeloTTSForTextToSpeech,
            NeuTTSForTextToSpeech,
            OpenVoiceForTextToSpeech,
            ParlerTTSForTextToSpeech,
            StyleTTS2ForTextToSpeech,
            SupertonicForTextToSpeech,
            VibeVoiceForTextToSpeech,
            VuiForTextToSpeech,
            XTTSForTextToSpeech,
            Zonos2ForTextToSpeech,
        )

        for wrapper in wrappers:
            with self.subTest(wrapper=wrapper.__name__):
                self.assertIn("seed", signature(wrapper._generate).parameters)

    def test_gptsovits_inference_transition_evaluates_nested_modules(self):
        components = {
            name: _EvalModule()
            for name in (
                "t2s_model",
                "vits_model",
                "cnhuhbert_model",
                "bert_model",
                "vocoder",
            )
        }
        runtime = SimpleNamespace(**components)
        model = GPTSoVITSForTextToSpeech(device="cpu")
        model.model = runtime
        model._training_ready = True

        model.load()

        self.assertIs(model.model, runtime)
        self.assertTrue(model._inference_ready)
        self.assertFalse(model._training_ready)
        for component in components.values():
            self.assertFalse(component.training)
            self.assertEqual(component.eval_calls, 1)

    def test_gptsovits_normalizes_flat_config_and_resets_cpu_half(self):
        captured = {}

        class FakeRuntimeConfig:

            def __init__(self, source):
                captured["source"] = source
                self.device = "cuda"
                self.is_half = True

            def update_configs(self):
                captured["device"] = self.device
                captured["is_half"] = self.is_half

        fake_runtime = SimpleNamespace(
            TTS_Config=FakeRuntimeConfig,
            TTS=lambda config: SimpleNamespace(config=config),
        )
        model = GPTSoVITSForTextToSpeech(
            runtime_config={
                "t2s_weights_path": "semantic.ckpt",
                "vits_weights_path": "acoustic.pth",
            },
            device="cpu",
        )
        model.device = "cpu"

        with patch(
                "voicehub.models.gptsovits.modeling.import_optional",
                return_value=fake_runtime,
        ):
            model._load_pretrained_model()

        self.assertEqual(
            captured["source"],
            {
                "custom": {
                    "t2s_weights_path": "semantic.ckpt",
                    "vits_weights_path": "acoustic.pth",
                    "version": "v2",
                },
            },
        )
        self.assertEqual(captured["device"], "cpu")
        self.assertFalse(captured["is_half"])

    def test_gptsovits_random_seed_is_resolved_and_forwarded(self):
        request = {}
        model = GPTSoVITSForTextToSpeech(device="cpu")
        model.device = "cpu"
        model.model = SimpleNamespace(
            run=lambda value: (request.update(value) or [(24_000, np.asarray([0.1], dtype=np.float32))]), )

        with (
                patch(
                    "voicehub.models.gptsovits.modeling.secrets.randbelow",
                    return_value=1234,
                ),
                patch(
                    "voicehub.models.gptsovits.modeling.seeded_inference",
                    return_value=nullcontext(1234),
                ),
        ):
            output = model._generate(
                "Hello",
                text_language="en",
                speaker_audio_path="reference.wav",
                prompt_language="en",
                seed=-1,
            )

        self.assertEqual(request["seed"], 1234)
        self.assertEqual(output.metadata["seed"], 1234)

    def test_kokoro_uses_fixed_sample_rate_and_auto_mps_fallback(self):
        model = KokoroForTextToSpeech(
            sample_rate=8_000,
            device="auto",
        )
        model.device = "mps"

        with patch.dict(os.environ, {}, clear=True):
            runtime_device = model._runtime_device()

        self.assertEqual(runtime_device, "cpu")
        self.assertEqual(model.sample_rate, 24_000)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional Kokoro extra")
    def test_kokoro_resolves_model_assets_from_local_snapshot(self):
        fake_loguru = ModuleType("loguru")
        fake_loguru.logger = SimpleNamespace(debug=Mock())
        fake_transformers = ModuleType("transformers")
        fake_transformers.AlbertConfig = object
        fake_istftnet = ModuleType("voicehub.models.kokoro.istftnet")
        fake_istftnet.Decoder = object
        fake_kokoro_modules = ModuleType("voicehub.models.kokoro.modules")
        fake_kokoro_modules.CustomAlbert = object
        fake_kokoro_modules.ProsodyPredictor = object
        fake_kokoro_modules.TextEncoder = object
        modules = {
            "loguru": fake_loguru,
            "transformers": fake_transformers,
            "voicehub.models.kokoro.istftnet": fake_istftnet,
            "voicehub.models.kokoro.modules": fake_kokoro_modules,
        }
        with _temporary_modules(modules):
            from voicehub.models.kokoro.model import KModel

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            checkpoint = root / "kokoro-v1_0.pth"
            config.touch()
            checkpoint.touch()

            self.assertEqual(
                KModel._local_snapshot_file(str(root), "config.json"),
                str(config.resolve()),
            )
            self.assertEqual(
                KModel._model_file(str(root)),
                str(checkpoint.resolve()),
            )

    def test_melotts_speaker_resolution_validates_names_and_ids(self):
        speakers = {
            "speaker-a": 2,
            "speaker-b": 7,
        }

        self.assertEqual(
            MeloTTSForTextToSpeech._resolve_speaker_id(speakers, None),
            2,
        )
        self.assertEqual(
            MeloTTSForTextToSpeech._resolve_speaker_id(speakers, "speaker-b"),
            7,
        )
        with self.assertRaisesRegex(ValueError, "Available IDs"):
            MeloTTSForTextToSpeech._resolve_speaker_id(speakers, 3)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is the tensor substrate")
    def test_openvoice_embedding_shape_is_validated_explicitly(self):
        import torch

        valid = OpenVoiceForTextToSpeech._load_embedding(
            torch.zeros(256, 1),
            name="target_embedding",
            device="cpu",
        )
        self.assertEqual(tuple(valid.shape), (1, 256, 1))
        with self.assertRaisesRegex(ValueError, r"shape \[batch, 256, 1\]"):
            OpenVoiceForTextToSpeech._load_embedding(
                torch.zeros(255, 1),
                name="target_embedding",
                device="cpu",
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is the tensor substrate")
    def test_openvoice_explicit_base_audio_uses_native_runtime(self):
        import torch

        model = OpenVoiceForTextToSpeech(device="cpu")
        model.device = "cpu"
        model.runtime = SimpleNamespace(convert=Mock(return_value=torch.tensor([0.1, -0.1])), )
        source_embedding = torch.zeros(1, 256, 1)
        target_embedding = torch.ones(1, 256, 1)
        with patch(
                "voicehub.models.openvoice.modeling.seeded_inference",
                return_value=nullcontext(31),
        ) as seeded:
            output = model._generate(
                "hello",
                base_audio=torch.zeros(2_048),
                base_audio_sampling_rate=22_050,
                source_embedding=source_embedding,
                target_embedding=target_embedding,
                vad=False,
                seed=31,
            )

        seeded.assert_called_once_with(
            31,
            device="cpu",
            model_type="openvoice",
        )
        self.assertEqual(output.metadata["seed"], 31)
        self.assertEqual(output.metadata["base_source"], "provided-audio")
        model.runtime.convert.assert_called_once()

    def test_supertonic_trims_runtime_padding_using_reported_duration(self):
        model = SupertonicForTextToSpeech()
        model.config.sample_rate = 10
        padded = np.arange(10, dtype=np.float32)[None, :]

        waveform = model._trim_waveform(
            padded,
            np.asarray([0.5], dtype=np.float32),
        )

        np.testing.assert_array_equal(waveform, padded[0, :5])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is the tensor substrate")
    def test_supertonic_trims_each_chunk_before_concatenation(self):
        import torch

        from voicehub.models.supertonic.native.runtime import NativeSupertonicRuntime

        runtime = SimpleNamespace(
            sample_rate=10,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        runtime.infer_batch = Mock(
            side_effect=[
                (
                    torch.tensor([[1, 2, 3, 4, 5, 90, 91, 92]]),
                    torch.tensor([0.5]),
                ),
                (
                    torch.tensor([[6, 7, 8, 9, 93, 94, 95, 96]]),
                    torch.tensor([0.4]),
                ),
            ], )

        with patch(
                "voicehub.models.supertonic.native.runtime.chunk_text",
                return_value=("first", "second"),
        ):
            waveform, duration = NativeSupertonicRuntime.synthesize(
                runtime,
                "long input",
                "en",
                object(),
                total_steps=2,
                silence_duration=0.2,
            )

        torch.testing.assert_close(
            waveform,
            torch.tensor(
                [[1, 2, 3, 4, 5, 0, 0, 6, 7, 8, 9]],
                dtype=torch.float32,
            ),
        )
        self.assertAlmostEqual(float(duration[0]), 1.1, places=6)

    def test_parler_extracts_audio_from_model_output(self):

        class FakeTensor:

            def __init__(self, values):
                self.values = values

            def numel(self):
                return self.values.size

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def squeeze(self):
                self.values = np.squeeze(self.values)
                return self

            def contiguous(self):
                return self

            def __array__(self, dtype=None):
                return np.asarray(self.values, dtype=dtype)

        class FakeOutput:
            audio_values = FakeTensor(np.asarray([[0.1, -0.1]], dtype=np.float32))

        waveform = ParlerTTSForTextToSpeech._extract_waveform(FakeOutput())

        np.testing.assert_array_equal(
            waveform,
            np.asarray([0.1, -0.1], dtype=np.float32),
        )

    def test_parler_extracts_return_dict_sequences(self):

        class FakeTensor:

            def __init__(self, values):
                self.values = values

            def numel(self):
                return self.values.size

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def squeeze(self):
                self.values = np.squeeze(self.values)
                return self

            def contiguous(self):
                return self

            def __array__(self, dtype=None):
                return np.asarray(self.values, dtype=dtype)

        output = SimpleNamespace(sequences=FakeTensor(np.asarray([[0.25, -0.25]], dtype=np.float32)), )

        waveform = ParlerTTSForTextToSpeech._extract_waveform(output)

        np.testing.assert_array_equal(
            waveform,
            np.asarray([0.25, -0.25], dtype=np.float32),
        )

    def test_styletts_asset_resolution_prefers_existing_search_root(self):
        from voicehub.models.styletts2.runtime import StyleTTS2Runtime

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            expected = second / "asset.bin"
            expected.touch()

            resolved = StyleTTS2Runtime._resolve_asset(
                "asset.bin",
                (first, second),
            )

        self.assertEqual(resolved, expected.resolve())

    def test_vui_has_a_working_default_checkpoint_filename(self):
        model = VuiForTextToSpeech()

        self.assertEqual(model.config.name_or_path, "vui-abraham-100m.pt")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional Vui extra")
    def test_vui_cpu_precision_uses_the_model_dtype(self):
        import torch

        fake_inflect = ModuleType("inflect")
        fake_inflect.engine = Mock()
        fake_torchaudio = ModuleType("torchaudio")
        fake_model = ModuleType("voicehub.models.vui.model")
        fake_model.Vui = object
        fake_sampling = ModuleType("voicehub.models.vui.sampling")
        fake_sampling.multinomial = Mock()
        fake_sampling.sample_top_k = Mock()
        fake_sampling.sample_top_p = Mock()
        fake_sampling.sample_top_p_top_k = Mock()
        fake_vad = ModuleType("voicehub.models.vui.vad")
        fake_vad.detect_voice_activity = Mock()
        modules = {
            "inflect": fake_inflect,
            "torchaudio": fake_torchaudio,
            "voicehub.models.vui.model": fake_model,
            "voicehub.models.vui.sampling": fake_sampling,
            "voicehub.models.vui.vad": fake_vad,
        }
        with _temporary_modules(modules):
            vui_tts = importlib.import_module("voicehub.models.vui.tts")

        context, cache_dtype = vui_tts._inference_precision(
            SimpleNamespace(
                device=SimpleNamespace(type="cpu"),
                dtype=torch.float32,
            ))

        self.assertIsInstance(context, type(nullcontext()))
        self.assertEqual(cache_dtype, torch.float32)
        self.assertNotIn("?.", vui_tts.simple_clean("Really?"))

        prepared = vui_tts._prepare_prompt_codes(
            torch.ones((4, 3), dtype=torch.float32),
            batch_size=1,
            n_quantizers=4,
            max_gen_len=10,
            device=torch.device("cpu"),
        )
        self.assertEqual(prepared.shape, (1, 4, 3))
        self.assertEqual(prepared.dtype, torch.int64)
        with self.assertRaisesRegex(ValueError, "shape"):
            vui_tts._prepare_prompt_codes(
                torch.ones((3, )),
                batch_size=1,
                n_quantizers=4,
                max_gen_len=10,
                device=torch.device("cpu"),
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional Vui extra")
    def test_vui_render_cleans_once_and_uses_codec_rate_and_final_vad_endpoint(self):
        import torch

        fake_model_module = ModuleType("voicehub.models.vui.model")
        fake_model_module.Vui = object
        fake_sampling = ModuleType("voicehub.models.vui.sampling")
        fake_sampling.multinomial = Mock()
        fake_sampling.sample_top_k = Mock()
        fake_sampling.sample_top_p = Mock()
        fake_sampling.sample_top_p_top_k = Mock()
        fake_vad = ModuleType("voicehub.models.vui.vad")
        fake_vad.detect_voice_activity = Mock()
        modules = {
            "voicehub.models.vui.model": fake_model_module,
            "voicehub.models.vui.sampling": fake_sampling,
            "voicehub.models.vui.vad": fake_vad,
        }
        with _temporary_modules(modules):
            vui_tts = importlib.import_module("voicehub.models.vui.tts")
        resample = Mock(side_effect=lambda audio, _source, _target: audio)
        fake_codec = SimpleNamespace(
            config=SimpleNamespace(sample_rate=100),
            hz=10,
            from_indices=lambda _codes: torch.arange(
                100,
                dtype=torch.float32,
            ).unsqueeze(0),
        )
        fake_model = SimpleNamespace(
            codec=fake_codec,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        def fake_generate(_model, chunk, *args, **kwargs):
            vui_tts.simple_clean(chunk)
            return torch.zeros((1, 4, 30), dtype=torch.int64)

        with (
                patch.object(
                    vui_tts,
                    "simple_clean",
                    wraps=vui_tts.simple_clean,
                ) as clean,
                patch.object(vui_tts, "generate", side_effect=fake_generate),
                patch.object(vui_tts, "resample_waveform", resample),
                patch.object(
                    vui_tts,
                    "vad",
                    return_value=[(0.0, 0.1), (0.2, 0.3)],
                ),
        ):
            waveform = vui_tts.render(fake_model, "x" * 1000)

        self.assertEqual(clean.call_count, 1)
        self.assertEqual(resample.call_args.args[1:], (100, 16_000))
        self.assertEqual(waveform.shape[-1], 50)

    def test_vibevoice_cached_prompt_requires_negative_branches(self):
        model = VibeVoiceForTextToSpeech()
        model._safe_globals = ()
        model._torch = SimpleNamespace(
            serialization=SimpleNamespace(safe_globals=lambda _: nullcontext(), ),
            load=lambda *args, **kwargs: {
                "lm": object(),
                "tts_lm": object(),
            },
        )

        with self.assertRaisesRegex(ValueError, "neg_lm"):
            model._load_cached_prompt("prompt.pt")

    def test_vibevoice_uses_checkpoint_processor_sample_rate(self):
        processor = SimpleNamespace(audio_processor=SimpleNamespace(sampling_rate=48_000), )

        self.assertEqual(
            VibeVoiceForTextToSpeech._checkpoint_sample_rate(processor),
            48_000,
        )

    def test_xtts_uses_checkpoint_output_sample_rate(self):
        checkpoint_config = SimpleNamespace(audio=SimpleNamespace(output_sample_rate=48_000), )

        sample_rate = XTTSForTextToSpeech._checkpoint_sample_rate(checkpoint_config)

        self.assertEqual(sample_rate, 48_000)

    def test_xtts_scopes_generation_seed_and_reports_effective_value(self):
        import torch

        model = XTTSForTextToSpeech(device="cpu")
        model.device = "cpu"
        model._runtime_config = SimpleNamespace(
            languages=("en", ),
            max_ref_len=30,
            gpt_cond_len=30,
            gpt_cond_chunk_len=4,
            sound_norm_refs=False,
            temperature=0.75,
            top_k=50,
            top_p=0.85,
            repetition_penalty=5.0,
        )
        model._tokenizer = SimpleNamespace(encode=Mock(return_value=(1, 2)), )
        model.model = SimpleNamespace(
            gpt=SimpleNamespace(max_text_tokens=404),
            conditioning_latents=Mock(return_value=(
                torch.ones(1, 2, 4),
                torch.ones(1, 4, 1),
            ), ),
            synthesize_tokens=Mock(return_value=torch.tensor([[[0.1]]]), ),
        )

        with patch(
                "voicehub.models.xtts.modeling.seeded_inference",
                return_value=nullcontext(37),
        ) as seeded:
            output = model._generate(
                "hello",
                speaker_audio_path="reference.wav",
                language="en",
                seed=37,
            )

        seeded.assert_called_once_with(
            37,
            device="cpu",
            model_type="xtts",
        )
        self.assertEqual(output.metadata["seed"], 37)
        self.assertEqual(output.metadata["requested_seed"], 37)

    def test_zonos2_forwards_the_selected_device_to_native_runtime(self):
        runtime = SimpleNamespace(
            model=_EvalModule(),
            artifacts=object(),
            config=object(),
        )
        model = Zonos2ForTextToSpeech(device="cuda:1")
        model.device = "cuda:1"
        with patch(
                "voicehub.models.zonos2.modeling."
                "NativeZonos2Runtime.from_pretrained",
                return_value=runtime,
        ) as load_runtime:
            model._load_pretrained_model()

        self.assertEqual(load_runtime.call_args.kwargs["device"], "cuda:1")
        self.assertIs(model.model, runtime.model)

    def test_zonos2_materializes_a_request_local_seed(self):
        runtime = SimpleNamespace(
            generate=Mock(
                return_value=SimpleNamespace(
                    audio=np.asarray([0.1], dtype=np.float32),
                    sample_rate=44_100,
                    eos_frame=3,
                    text_frontend="raw-utf8",
                ), ))
        model = Zonos2ForTextToSpeech(device="cpu")
        model.device = "cpu"
        model._runtime = runtime
        model.artifacts = SimpleNamespace(safe_conversion=False)

        with patch(
                "voicehub.models.zonos2.modeling.seeded_inference",
                return_value=nullcontext(41),
        ) as seeded:
            output = model._generate("hello")

        seeded.assert_called_once_with(
            None,
            device="cpu",
            model_type="zonos2",
        )
        options = runtime.generate.call_args.kwargs["options"]
        self.assertEqual(options.seed, 41)
        self.assertEqual(output.metadata["seed"], 41)
        self.assertIsNone(output.metadata["requested_seed"])

    def test_zonos_falls_back_from_unsupported_mps_runtime(self):
        model = ZonosForTextToSpeech(device="mps")
        model.device = "mps"

        self.assertEqual(model._runtime_device(), "cpu")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
    def test_zonos_speaker_audio_uses_the_native_pcm_boundary(self):
        import torch

        from voicehub.processing.waveform import save_pcm_wave

        make_embedding = Mock(return_value=torch.ones(1, 4))
        model = ZonosForTextToSpeech(device="cpu")
        model.model = SimpleNamespace(make_speaker_embedding=make_embedding)
        with tempfile.TemporaryDirectory() as directory:
            path = save_pcm_wave(
                Path(directory) / "speaker.wav",
                torch.zeros(800),
                16_000,
            )
            embedding = model._speaker_embedding(str(path))

        waveform, sample_rate = make_embedding.call_args.args
        self.assertEqual(waveform.shape, (1, 800))
        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(embedding.shape, (1, 4))

    def test_zonos2_supports_native_cpu_runtime(self):
        runtime = SimpleNamespace(
            model=_EvalModule(),
            artifacts=object(),
            config=object(),
        )
        model = Zonos2ForTextToSpeech(device="cpu")
        model.device = "cpu"

        with patch(
                "voicehub.models.zonos2.modeling."
                "NativeZonos2Runtime.from_pretrained",
                return_value=runtime,
        ) as load_runtime:
            model._load_pretrained_model()

        self.assertEqual(load_runtime.call_args.kwargs["device"], "cpu")
        self.assertIs(model.model, runtime.model)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional Zonos2 extra")
    def test_zonos2_parallel_context_is_idempotent(self):
        from voicehub.models.zonos2.source.zonos2.distributed import info

        with patch.object(info, "_TP_INFO", None):
            info.set_tp_info(0, 1)
            info.set_tp_info(0, 1)
            with self.assertRaisesRegex(RuntimeError, "already configured"):
                info.set_tp_info(1, 2)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional Zonos2 extra")
    def test_zonos2_global_state_can_be_reused_by_a_sequential_engine(self):
        from voicehub.models.zonos2.source.zonos2 import core
        from voicehub.models.zonos2.source.zonos2.distributed import info

        first_owner = object()
        second_owner = object()
        first_context = object()
        second_context = object()
        with (
                patch.object(core, "_GLOBAL_CTX", None),
                patch.object(core, "_GLOBAL_CTX_OWNER", None),
                patch.object(info, "_TP_INFO", None),
        ):
            core.set_global_ctx(first_context, owner=first_owner)
            self.assertTrue(info.set_tp_info(0, 1))
            core.reset_global_ctx(owner=first_owner)
            info.reset_tp_info(info.DistributedInfo(0, 1))

            core.set_global_ctx(second_context, owner=second_owner)
            self.assertTrue(info.set_tp_info(0, 1))
            self.assertIs(core.get_global_ctx(), second_context)

    def test_styletts_checkpoint_loader_requires_real_parameter_matches(self):
        from voicehub.models.styletts2.runtime import StyleTTS2Runtime

        mismatch = Mock()
        mismatch.named_parameters.return_value = [("weight", object())]
        mismatch.state_dict.return_value = {"weight": object()}
        mismatch.load_state_dict.side_effect = RuntimeError("strict mismatch")
        with self.assertRaisesRegex(RuntimeError, "no parameter keys matching"):
            StyleTTS2Runtime._load_module_checkpoint(
                "decoder",
                mismatch,
                {"module.unrelated": object()},
            )

        compatible = Mock()
        compatible.named_parameters.return_value = [("weight", object())]
        compatible.state_dict.return_value = {"weight": object()}
        compatible.load_state_dict.side_effect = [
            RuntimeError("prefix mismatch"),
            SimpleNamespace(unexpected_keys=()),
        ]
        matched = StyleTTS2Runtime._load_module_checkpoint(
            "decoder",
            compatible,
            {"module.weight": object()},
        )
        self.assertEqual(matched, {"weight"})
        self.assertIn(
            "decoder",
            StyleTTS2Runtime._CRITICAL_CHECKPOINT_MODULES,
        )

    def test_acoustic_wrappers_scope_generation_seeds(self):
        inflect = InflectTTSForTextToSpeech(device="cpu")
        inflect.device = "cpu"
        inflect.model = SimpleNamespace(
            synthesize=Mock(return_value=(
                22_050,
                np.asarray([0.1], dtype=np.float32),
            )), )
        with patch(
                "voicehub.models.inflecttts.modeling.seeded_inference",
                return_value=nullcontext(7),
        ) as seeded:
            inflect._generate("hello", seed=7)
        seeded.assert_called_once_with(
            7,
            device="cpu",
            model_type="inflecttts",
        )

        styletts = StyleTTS2ForTextToSpeech(device="cpu")
        styletts.device = "cpu"
        styletts.model = SimpleNamespace(generate=Mock(return_value=np.asarray([0.1], dtype=np.float32)), )
        with patch(
                "voicehub.models.styletts2.modeling.seeded_inference",
                return_value=nullcontext(11),
        ):
            output = styletts._generate("hello", seed=11)
        self.assertEqual(
            styletts.model.generate.call_args.kwargs["seed"],
            11,
        )
        self.assertEqual(output.metadata["seed"], 11)

        neutts = NeuTTSForTextToSpeech(device="cpu", seed=13)
        neutts.device = "cpu"
        neutts.model = SimpleNamespace(
            encode_reference=Mock(return_value=np.asarray([1])),
            infer=Mock(return_value=np.asarray([0.1], dtype=np.float32)),
            last_seed=13,
        )
        with patch(
                "voicehub.models.neutts.modeling.seeded_inference",
                return_value=nullcontext(13),
        ):
            output = neutts._generate(
                "hello",
                speaker_audio_path="reference.wav",
                reference_text="hello",
            )
        self.assertEqual(output.metadata["seed"], 13)


if __name__ == "__main__":
    unittest.main()
