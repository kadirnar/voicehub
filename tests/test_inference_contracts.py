import importlib.util
import os
import random
import tempfile
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import torch

from voicehub import AutoConfig, PreTrainedTTSModel, TTSGenerationConfig, TTSOutput
from voicehub.configuration import VoiceHubConfig
from voicehub.inference_strategy import InferenceStrategy
from voicehub.models._shared import resolve_model_directory, resolve_torch_dtype, seeded_inference
from voicehub.models.melotts.modeling import MeloTTSConfig, MeloTTSForTextToSpeech

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class InferenceConfig(VoiceHubConfig):
    model_type = "inference-contract"


class LifecycleModel(PreTrainedTTSModel):
    config_class = InferenceConfig

    def __init__(self, config=None, **kwargs):
        self.load_count = 0
        self.inference_prepare_count = 0
        self.training_prepare_count = 0
        super().__init__(
            self._coerce_config(config),
            **kwargs,
        )

    def _load_pretrained_model(self) -> None:
        self.load_count += 1
        self.model = object()

    def _prepare_for_inference(self) -> None:
        self.inference_prepare_count += 1
        super()._prepare_for_inference()

    def _prepare_for_training(self) -> None:
        self.training_prepare_count += 1

    def _generate(self, text: str, **kwargs) -> TTSOutput:
        return TTSOutput(
            audio=[0.0],
            sample_rate=self.sample_rate,
            metadata={
                "text": text,
                **kwargs
            },
        )


class PortableStateRuntime:

    def __init__(self):
        self.loaded_states = []
        self.evaluated = False

    def load_state_dict(self, state):
        self.loaded_states.append(state)

    def eval(self):
        self.evaluated = True
        return self


class PortableStateLifecycleModel(LifecycleModel):

    def _load_pretrained_model(self) -> None:
        self.load_count += 1
        self.model = PortableStateRuntime()


class RequiredReferenceModel(LifecycleModel):

    def _validate_generation_inputs(self, model_inputs):
        if not model_inputs.get("speaker_audio_path"):
            raise ValueError("A speaker reference is required.")


class InvalidOutputModel(LifecycleModel):

    def _generate(self, text: str, **kwargs):
        del text, kwargs
        return [0.0]


class FinitePassthroughModel(LifecycleModel):
    passthrough_generation_options = frozenset({"backend_option"})


class FlakyLoadModel(LifecycleModel):

    def _load_pretrained_model(self) -> None:
        self.load_count += 1
        self.model = object()
        if self.load_count == 1:
            raise RuntimeError("transient load failure")


class FlakyPrepareModel(LifecycleModel):

    def _prepare_for_inference(self) -> None:
        self.inference_prepare_count += 1
        if self.inference_prepare_count == 1:
            raise RuntimeError("transient preparation failure")
        PreTrainedTTSModel._prepare_for_inference(self)


class FailedTrainingTransitionModel(LifecycleModel):

    def __init__(self, config=None, **kwargs):
        self.runtime_is_usable = False
        self.fail_training_transition = True
        super().__init__(config, **kwargs)

    def _prepare_for_inference(self) -> None:
        super()._prepare_for_inference()
        self.runtime_is_usable = True

    def _prepare_for_training(self) -> None:
        super()._prepare_for_training()
        self.runtime_is_usable = False
        if self.fail_training_transition:
            self.fail_training_transition = False
            raise RuntimeError("training transition failed")

    def _generate(self, text: str, **kwargs) -> TTSOutput:
        if not self.runtime_is_usable:
            raise RuntimeError("inference runtime was not restored")
        return super()._generate(text, **kwargs)


class BlockingLoadModel(LifecycleModel):

    def __init__(self, config=None, **kwargs):
        self.load_started = Event()
        self.release_load = Event()
        super().__init__(config, **kwargs)

    def _load_pretrained_model(self) -> None:
        self.load_count += 1
        self.load_started.set()
        if not self.release_load.wait(timeout=5):
            raise TimeoutError("Test did not release the model load.")
        self.model = object()


class ReentrantRestoreAdapter:

    recipe_id = "inference-contract"

    def __init__(self, model):
        self.model = model

    def setup(self):
        self.model.load_for_training()
        return self

    def load_state_dict(self, state, **kwargs):
        del state, kwargs

    def eval(self):
        return self


class WrappingInferenceStrategy(InferenceStrategy):

    name = "wrapping"

    def __init__(self):
        self.events = []

    def validate(self, wrapper):
        self.events.append(("validate", wrapper.model))

    def prepare(self, model, *, wrapper):
        self.events.append(("prepare", model))
        return {
            "runtime": model,
        }

    def restore_for_training(self, model, *, wrapper):
        self.events.append(("restore", model))
        return model["runtime"]


class InferenceLifecycleTests(unittest.TestCase):

    def test_forward_owns_lazy_loading_and_preparation(self):
        model = LifecycleModel()

        first = model.generate("first")
        second = model.generate("second")

        self.assertEqual(first.metadata["text"], "first")
        self.assertEqual(second.metadata["text"], "second")
        self.assertEqual(model.load_count, 1)
        self.assertEqual(model.inference_prepare_count, 1)

    def test_training_to_inference_transition_is_prepared_once(self):
        model = LifecycleModel()
        model.load_for_training()

        model.generate("after training")
        model.generate("still serving")

        self.assertEqual(model.load_count, 1)
        self.assertEqual(model.training_prepare_count, 1)
        self.assertEqual(model.inference_prepare_count, 1)

    def test_inference_strategy_wraps_once_and_restores_before_training(self):
        strategy = WrappingInferenceStrategy()
        model = LifecycleModel().set_inference_strategy(strategy)

        model.generate("first")
        wrapped_runtime = model.model
        model.generate("second")
        model.load_for_training()

        self.assertEqual(strategy.events[0], ("validate", None))
        self.assertEqual(
            [event[0] for event in strategy.events],
            ["validate", "prepare", "restore"],
        )
        self.assertIs(model.model, wrapped_runtime["runtime"])
        self.assertEqual(model.training_prepare_count, 1)

    def test_inference_strategy_reapplies_after_training_transition(self):
        strategy = WrappingInferenceStrategy()
        model = LifecycleModel().set_inference_strategy(strategy)

        model.generate("first")
        model.load_for_training()
        model.generate("second")

        self.assertEqual(
            [event[0] for event in strategy.events],
            ["validate", "prepare", "restore", "prepare"],
        )
        self.assertEqual(model.inference_prepare_count, 2)

    def test_inference_strategy_cannot_change_while_serving(self):
        model = LifecycleModel()
        model.generate("first")

        with self.assertRaisesRegex(RuntimeError, "active serving runtime"):
            model.set_inference_strategy(WrappingInferenceStrategy())

        model.load_for_training()
        replacement = WrappingInferenceStrategy()
        self.assertIs(model.set_inference_strategy(replacement), model)
        self.assertIs(model.inference_strategy, replacement)

    def test_eager_factory_load_applies_selected_inference_strategy(self):
        strategy = WrappingInferenceStrategy()

        model = LifecycleModel.from_pretrained(
            lazy_load=False,
            inference_strategy=strategy,
        )

        self.assertEqual(model.load_count, 1)
        self.assertEqual(
            [event[0] for event in strategy.events],
            ["validate", "prepare"],
        )
        self.assertIn("runtime", model.model)

    def test_default_inference_transition_evaluates_module_like_runtimes(self):
        model = LifecycleModel()
        model.load_for_training()
        runtime = SimpleNamespace(eval=lambda: setattr(runtime, "evaluated", True))
        runtime.evaluated = False
        model.model = runtime

        model.generate("after training")

        self.assertTrue(runtime.evaluated)

    def test_training_preparation_is_exactly_once_during_portable_restore(self):
        model = LifecycleModel(device="cpu")
        model._pending_model_state_path = Path("unused-model-state.pt")
        adapter = ReentrantRestoreAdapter(model)
        model.get_training_adapter = lambda: adapter
        fake_torch = SimpleNamespace(
            load=lambda *args, **kwargs: {
                "__voicehub_training_adapter__": "inference-contract",
            },
        )

        with patch(
                "voicehub.model.import_module",
                return_value=fake_torch,
        ):
            model.load_for_training()
            model.load_for_training()

        self.assertEqual(model.training_prepare_count, 1)

    def test_tts_portable_state_rejects_credentials_before_runtime_mutation(self):
        credential = "must-not-appear-in-errors"
        unsafe_state = {
            "__voicehub_training_adapter__": "inference-contract",
            "metadata": {
                "api_key": credential,
            },
        }
        safe_state = {
            "weight": "safe",
        }

        with tempfile.TemporaryDirectory() as directory:
            InferenceConfig().save_pretrained(directory)
            state_path = Path(directory) / "model_state.pt"
            state_path.write_bytes(b"patched-test-state")
            model = PortableStateLifecycleModel.from_pretrained(
                directory,
                device="cpu",
            )
            adapter_events = []
            model.get_training_adapter = lambda: SimpleNamespace(
                setup=lambda: adapter_events.append("setup"),
                load_state_dict=lambda *args, **kwargs: adapter_events.append("load"),
                eval=lambda: adapter_events.append("eval"),
            )
            fake_torch = SimpleNamespace(load=lambda *args, **kwargs: unsafe_state)

            with patch(
                    "voicehub.model.import_module",
                    return_value=fake_torch,
            ), self.assertRaisesRegex(
                    ValueError,
                    r"VoiceHub portable model state.*metadata\.api_key",
            ) as captured:
                model.load()

            self.assertNotIn(credential, str(captured.exception))
            self.assertEqual(adapter_events, [])
            self.assertEqual(model.model.loaded_states, [])
            self.assertEqual(model._pending_model_state_path.resolve(), state_path.resolve())

            fake_torch.load = lambda *args, **kwargs: safe_state
            with patch(
                    "voicehub.model.import_module",
                    return_value=fake_torch,
            ):
                model.load()

        self.assertEqual(model.model.loaded_states, [safe_state])
        self.assertTrue(model.model.evaluated)
        self.assertIsNone(model._pending_model_state_path)

    def test_backend_preflight_runs_before_model_allocation(self):
        model = RequiredReferenceModel()

        with self.assertRaisesRegex(ValueError, "speaker reference"):
            model.generate("hello")

        self.assertEqual(model.load_count, 0)

    def test_generation_options_are_validated_before_model_allocation(self):
        model = LifecycleModel()

        with self.assertRaisesRegex(ValueError, "temperature"):
            model.generate("hello", temperature=-0.1)

        self.assertEqual(model.load_count, 0)

    def test_forward_enforces_the_output_contract(self):
        model = InvalidOutputModel()

        with self.assertRaisesRegex(TypeError, "must return a TTSOutput"):
            model.generate("hello")

    def test_failed_partial_load_can_be_retried(self):
        model = FlakyLoadModel()

        with self.assertRaisesRegex(RuntimeError, "transient load failure"):
            model.generate("first attempt")
        output = model.generate("second attempt")

        self.assertEqual(output.metadata["text"], "second attempt")
        self.assertEqual(model.load_count, 2)
        self.assertTrue(model.is_loaded)

    def test_failed_inference_preparation_discards_runtime_for_retry(self):
        model = FlakyPrepareModel()

        with self.assertRaisesRegex(
                RuntimeError,
                "transient preparation failure",
        ):
            model.generate("first attempt")
        self.assertIsNone(model.model)
        self.assertFalse(model.is_loaded)

        output = model.generate("second attempt")

        self.assertEqual(output.metadata["text"], "second attempt")
        self.assertEqual(model.load_count, 2)
        self.assertEqual(model.inference_prepare_count, 2)
        self.assertTrue(model.is_loaded)

    def test_failed_training_transition_invalidates_inference_state(self):
        model = FailedTrainingTransitionModel()
        model.generate("before transition")

        with self.assertRaisesRegex(RuntimeError, "transition failed"):
            model.load_for_training()
        output = model.generate("after failed transition")
        model.load_for_training()
        model.load_for_training()

        self.assertEqual(output.metadata["text"], "after failed transition")
        self.assertEqual(model.inference_prepare_count, 2)
        self.assertEqual(model.training_prepare_count, 2)

    def test_concurrent_first_use_loads_the_runtime_once(self):
        model = BlockingLoadModel()
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(model.generate, "first")
            self.assertTrue(model.load_started.wait(timeout=2))
            second = executor.submit(model.generate, "second")
            model.release_load.set()
            outputs = (first.result(timeout=5), second.result(timeout=5))

        self.assertEqual([item.metadata["text"] for item in outputs], [
            "first",
            "second",
        ])
        self.assertEqual(model.load_count, 1)
        self.assertEqual(model.inference_prepare_count, 1)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional inference extra")
    def test_model_loading_restores_process_random_state(self):
        import numpy
        import torch

        random.seed(101)
        numpy.random.seed(202)
        torch.manual_seed(303)
        python_state = random.getstate()
        numpy_state = numpy.random.get_state()
        torch_state = torch.random.get_rng_state()
        benchmark = torch.backends.cudnn.benchmark
        model = LifecycleModel(device="cpu")

        def load_with_global_side_effects():
            random.random()
            numpy.random.random()
            torch.rand(1)
            torch.backends.cudnn.benchmark = not benchmark
            model.model = object()

        with patch.object(
                model,
                "_load_pretrained_model",
                side_effect=load_with_global_side_effects,
        ):
            model.load()

        self.assertEqual(random.getstate(), python_state)
        restored_numpy_state = numpy.random.get_state()
        self.assertEqual(restored_numpy_state[0], numpy_state[0])
        numpy.testing.assert_array_equal(
            restored_numpy_state[1],
            numpy_state[1],
        )
        self.assertEqual(restored_numpy_state[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        self.assertEqual(torch.backends.cudnn.benchmark, benchmark)


class SharedInferenceHelperTests(unittest.TestCase):

    def test_model_constructor_accepts_path_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)

            model = LifecycleModel(model_path)

        self.assertEqual(model.config.name_or_path, str(model_path.resolve()))

    def test_model_constructor_rejects_missing_path_checkpoint(self):
        with self.assertRaisesRegex(FileNotFoundError, "was not found"):
            LifecycleModel(Path("missing-local-checkpoint"))

    def test_explicit_string_paths_are_normalized_or_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.mkdir()
            home_environment = {
                "HOME": directory,
                "USERPROFILE": directory,
            }
            with patch.dict(os.environ, home_environment):
                model = LifecycleModel("~/model")

            missing = Path(directory) / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "was not found"):
                LifecycleModel(str(missing))

        self.assertEqual(model.config.name_or_path, str(model_path.resolve()))
        hub_model = LifecycleModel("organization/model")
        self.assertEqual(
            hub_model.config.name_or_path,
            "organization/model",
        )

    def test_from_pretrained_accepts_path_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            InferenceConfig(name_or_path="upstream-checkpoint").save_pretrained(model_path, )

            model = LifecycleModel.from_pretrained(model_path)

        self.assertEqual(model.config.name_or_path, str(model_path.resolve()))

    def test_from_pretrained_routes_weight_files_without_json_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "fine-tuned.safetensors"
            model_path.write_bytes(b"not a JSON configuration")

            model = LifecycleModel.from_pretrained(model_path)

        self.assertEqual(model.config.name_or_path, str(model_path.resolve()))
        self.assertFalse(model.is_loaded)

    def test_auto_config_requires_model_type_for_raw_weight_files(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "fine-tuned.safetensors"
            model_path.touch()

            config = AutoConfig.from_pretrained(
                model_path,
                model_type="echo",
            )
            with self.assertRaisesRegex(ValueError, "model_type"):
                AutoConfig.from_pretrained(model_path)

        self.assertEqual(config.name_or_path, str(model_path.resolve()))
        self.assertEqual(config.model_type, "echo")

    def test_model_config_serializes_nested_paths(self):
        config = InferenceConfig(
            name_or_path=Path("model.safetensors"),
            backend_paths={
                "vocoder": Path("vocoder/model.safetensors"),
                "references": [Path("speaker.wav")],
            },
        )

        serialized = config.to_dict()

        self.assertEqual(serialized["name_or_path"], "model.safetensors")
        self.assertEqual(
            serialized["backend_paths"],
            {
                "vocoder": str(Path("vocoder/model.safetensors")),
                "references": [str(Path("speaker.wav"))],
            },
        )

    def test_melotts_local_directory_preserves_language_and_resolves_files(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            config_path = model_directory / "config.json"
            checkpoint_path = model_directory / "checkpoint.pth"
            config_path.touch()
            checkpoint_path.touch()
            model = MeloTTSForTextToSpeech(
                MeloTTSConfig(language="JP"),
                model_path=model_directory,
                device="cpu",
            )

            resolved_paths = model._resolve_checkpoint_paths()

        self.assertEqual(model.config.language, "JP")
        self.assertEqual(
            resolved_paths,
            (str(config_path.resolve()), str(checkpoint_path.resolve())),
        )

    def test_melotts_string_checkpoint_code_remains_a_language_alias(self):
        model = MeloTTSForTextToSpeech("JP", device="cpu")

        self.assertEqual(model.config.language, "JP")
        self.assertEqual(model._resolve_checkpoint_paths(), (None, None))

    def test_melotts_accepts_noncanonical_checkpoint_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            config_path = model_directory / "config.json"
            checkpoint_path = model_directory / "G_100.pth"
            config_path.touch()
            checkpoint_path.touch()
            model = MeloTTSForTextToSpeech(
                MeloTTSConfig(language="JP"),
                model_path=checkpoint_path,
                device="cpu",
            )

            resolved_paths = model._resolve_checkpoint_paths()

        self.assertEqual(
            resolved_paths,
            (str(config_path.resolve()), str(checkpoint_path.resolve())),
        )

    def test_generation_config_update_recognizes_unset_common_fields(self):
        config = TTSGenerationConfig()

        unused = config.update(speed=1.25, backend_option=True)

        self.assertEqual(config.speed, 1.25)
        self.assertEqual(unused, {"backend_option": True})

    def test_zero_temperature_remains_available_for_greedy_backends(self):
        config = TTSGenerationConfig(temperature=0)

        self.assertEqual(config.temperature, 0)

    def test_generation_config_rejects_non_finite_values(self):
        for option in ("speed", "temperature", "top_p"):
            for value in (float("nan"), float("inf")):
                with self.subTest(option=option, value=value):
                    with self.assertRaises(ValueError):
                        TTSGenerationConfig(**{option: value})

    def test_generation_config_rejects_seeds_outside_torch_range(self):
        for seed in (-(2**63) - 1, 2**64):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "supported range"):
                    TTSGenerationConfig(seed=seed)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional inference extra")
    def test_seeded_inference_restores_all_cpu_random_states(self):
        import numpy
        import torch

        random.seed(101)
        numpy.random.seed(202)
        torch.manual_seed(303)
        python_state = random.getstate()
        numpy_state = numpy.random.get_state()
        torch_state = torch.random.get_rng_state()

        with patch.dict(os.environ, {"PYTHONHASHSEED": "caller-seed"}):
            with seeded_inference(
                    404,
                    device="cpu",
                    model_type="test",
            ) as effective_seed:
                os.environ["PYTHONHASHSEED"] = "upstream-seed"
                first_values = (
                    random.random(),
                    numpy.random.random(),
                    torch.rand(1).item(),
                )
            self.assertEqual(os.environ["PYTHONHASHSEED"], "caller-seed")
        with seeded_inference(
                404,
                device="cpu",
                model_type="test",
        ):
            second_values = (
                random.random(),
                numpy.random.random(),
                torch.rand(1).item(),
            )

        self.assertEqual(effective_seed, 404)
        self.assertEqual(first_values, second_values)
        self.assertEqual(random.getstate(), python_state)
        restored_numpy_state = numpy.random.get_state()
        self.assertEqual(restored_numpy_state[0], numpy_state[0])
        numpy.testing.assert_array_equal(
            restored_numpy_state[1],
            numpy_state[1],
        )
        self.assertEqual(restored_numpy_state[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_generation_config_serializes_path_outputs(self):
        config = TTSGenerationConfig(
            output_file=Path("speech.wav"),
            backend_paths={
                "speaker": Path("speaker.wav"),
                "references": [
                    Path("first.wav"),
                    (Path("second.wav"), ),
                ],
            },
        )

        serialized = config.to_dict()

        self.assertEqual(serialized["output_file"], "speech.wav")
        self.assertEqual(
            serialized["backend_paths"],
            {
                "speaker": "speaker.wav",
                "references": [
                    "first.wav",
                    ("second.wav", ),
                ],
            },
        )

    def test_generation_config_saves_nested_backend_paths(self):
        config = TTSGenerationConfig(
            backend_paths={
                "speaker": Path("speaker.wav"),
                "references": [Path("reference.wav")],
            }, )

        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            restored = TTSGenerationConfig.from_pretrained(directory)

        self.assertEqual(
            restored.backend_paths,
            {
                "speaker": "speaker.wav",
                "references": ["reference.wav"],
            },
        )

    def test_output_directory_is_rejected_before_model_loading(self):
        model = LifecycleModel(device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(IsADirectoryError, "directory"):
                model.generate(
                    "hello",
                    output_file=Path(directory),
                )

        self.assertEqual(model.load_count, 0)
        self.assertFalse(model.is_loaded)

    def test_finite_passthrough_contract_rejects_typos_before_loading(self):
        model = FinitePassthroughModel(device="cpu")

        with self.assertRaisesRegex(ValueError, "temprature"):
            model.generate("hello", temprature=0.5)

        self.assertEqual(model.load_count, 0)
        model.generate("hello", backend_option=True)
        self.assertEqual(model.load_count, 1)

    def test_tts_output_rejects_invalid_sample_rates(self):
        for sample_rate in (0, -1, 24_000.0, True):
            with self.subTest(sample_rate=sample_rate):
                with self.assertRaises(ValueError):
                    TTSOutput(audio=[0.0], sample_rate=sample_rate)

    def test_tts_output_rejects_invalid_audio_without_saving(self):
        invalid_audio = (
            ([], ValueError),
            (["not-a-sample"], TypeError),
            ([float("nan")], ValueError),
            ([float("inf")], ValueError),
        )
        for audio, error_type in invalid_audio:
            with self.subTest(audio=audio):
                with self.assertRaises(error_type):
                    TTSOutput(audio=audio, sample_rate=24_000)

    def test_tts_output_validates_python_audio_without_numpy(self):
        with patch(
                "voicehub.models.base.import_module",
                side_effect=AssertionError("Plain Python audio should use the standard-library path"),
        ):
            output = TTSOutput(audio=[0.0, 0.25], sample_rate=24_000)

        self.assertEqual(output.audio, [0.0, 0.25])

    def test_tts_output_validates_array_protocol_without_tensor_conversion(self):

        class FloatDType:
            kind = "f"

        class ArrayLike:
            size = 2
            flat = (0.0, 0.25)
            dtype = FloatDType()

        with patch(
                "voicehub.models.base.import_module",
                side_effect=AssertionError("Array-protocol audio should not require tensor conversion"),
        ):
            output = TTSOutput(audio=ArrayLike(), sample_rate=24_000)

        self.assertEqual(tuple(output.audio.flat), (0.0, 0.25))

    def test_tts_output_uses_the_native_pcm_wave_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = TTSOutput(
                audio=torch.tensor([-1.0, 0.0, 1.0]),
                sample_rate=24_000,
            )
            path = Path(output.save(Path(directory) / "speech.wav"))
            with wave.open(str(path), "rb") as stream:
                self.assertEqual(stream.getnchannels(), 1)
                self.assertEqual(stream.getsampwidth(), 2)
                self.assertEqual(stream.getframerate(), 24_000)
                self.assertEqual(stream.getnframes(), 3)

        self.assertEqual(output.file_path, str(path))

    def test_tts_output_rejects_an_unsupported_native_container(self):
        with tempfile.TemporaryDirectory() as directory:
            output = TTSOutput(audio=[0.0], sample_rate=24_000)
            with self.assertRaisesRegex(ValueError, "PCM WAVE"):
                output.save(Path(directory) / "speech.flac")

    def test_resolve_model_directory_rejects_a_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "model.safetensors"
            model_file.touch()

            with self.assertRaisesRegex(NotADirectoryError, "model directory"):
                resolve_model_directory(
                    str(model_file),
                    model_type="test",
                )

    def test_resolve_model_directory_accepts_path(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)

            resolved = resolve_model_directory(
                model_directory,
                model_type="test",
            )

        self.assertEqual(resolved, model_directory.resolve())

    def test_resolve_model_directory_does_not_treat_missing_path_as_hub_id(self):
        missing_directory = Path("missing-local-checkpoint")

        with self.assertRaisesRegex(FileNotFoundError, "was not found"):
            resolve_model_directory(
                missing_directory,
                model_type="test",
            )

    def test_dtype_resolver_supports_aliases_and_cpu_safety(self):
        float16 = object()
        bfloat16 = object()
        float32 = object()
        fake_torch = SimpleNamespace(
            float16=float16,
            bfloat16=bfloat16,
            float32=float32,
        )

        self.assertIs(
            resolve_torch_dtype(fake_torch, "fp16", "cpu:0"),
            float32,
        )
        self.assertIs(
            resolve_torch_dtype(fake_torch, "torch.bfloat16", "cuda"),
            bfloat16,
        )

    def test_dtype_resolver_rejects_non_dtype_attributes(self):

        class FakeDType:
            pass

        fake_torch = SimpleNamespace(
            dtype=FakeDType,
            float16=FakeDType(),
            bfloat16=FakeDType(),
            float32=FakeDType(),
            seed=lambda: 1,
        )

        with self.assertRaisesRegex(ValueError, "Unknown torch dtype"):
            resolve_torch_dtype(fake_torch, "seed", "cpu")

    def test_dtype_resolver_rejects_non_floating_dtypes(self):

        class FakeDType:

            def __init__(self, *, is_floating_point):
                self.is_floating_point = is_floating_point

        fake_torch = SimpleNamespace(
            dtype=FakeDType,
            float16=FakeDType(is_floating_point=True),
            bfloat16=FakeDType(is_floating_point=True),
            float32=FakeDType(is_floating_point=True),
            int8=FakeDType(is_floating_point=False),
        )

        with self.assertRaisesRegex(ValueError, "floating-point"):
            resolve_torch_dtype(fake_torch, "int8", "cuda")


if __name__ == "__main__":
    unittest.main()
