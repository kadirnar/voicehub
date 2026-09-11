import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from voicehub import AutoModelForTextToSpeech
from voicehub.checkpointing import save_safetensors
from voicehub.models.bark.modeling import BarkConfig, BarkForTextToSpeech, _build_bark_training_model
from voicehub.models.speecht5.modeling import SpeechT5Config, SpeechT5ForTextToSpeech
from voicehub.models.vits.modeling import VitsConfig, VitsForTextToSpeech, _build_vits_training_model
from voicehub.models.vits.training import NativeVitsGeneratorTrainingAdapter
from voicehub.registry import ModelSpec, get_model_spec
from voicehub.training import AutoTrainingAdapter
from voicehub.training.arguments import TrainingArguments
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import TrainingFamily, get_training_spec
from voicehub.training.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class _FakeTorch:

    float32 = object()

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def as_tensor(value, **kwargs):
        del kwargs
        import torch
        return torch.as_tensor(value, dtype=torch.float32, device="cpu")

    @staticmethod
    def isfinite(value):
        import torch
        return torch.isfinite(value)


@contextmanager
def _fixed_seed(*args, **kwargs):
    del args, kwargs
    yield 123


class TransformersTTSConfigurationTests(unittest.TestCase):

    def test_configs_validate_and_serialize_family_controls(self):
        bark = BarkConfig(
            name_or_path="suno/bark-small",
            use_safetensors=True,
            torch_dtype="float32",
        )
        speech = SpeechT5Config(
            name_or_path="microsoft/speecht5_tts",
            vocoder_name_or_path="microsoft/speecht5_hifigan",
        )
        vits = VitsConfig(
            name_or_path="facebook/mms-tts-eng",
            speaking_rate=1.2,
            training_spectral_loss_weight=0.25,
        )

        self.assertTrue(bark.to_dict()["use_safetensors"])
        self.assertEqual(
            speech.to_dict()["vocoder_name_or_path"],
            "microsoft/speecht5_hifigan",
        )
        self.assertEqual(vits.to_dict()["speaking_rate"], 1.2)
        self.assertEqual(vits.to_dict()["training_spectral_loss_weight"], 0.25)
        self.assertFalse(vits.to_dict()["enable_experimental_reconstruction_training"])

    def test_configs_reject_provider_owned_or_secret_loader_options(self):
        with self.assertRaisesRegex(ValueError, "provider-owned"):
            BarkConfig(model_kwargs={"state_dict": {}})
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            SpeechT5Config(processor_kwargs={"token": "secret"})
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            VitsConfig(speaking_rate=0)
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            VitsConfig(enable_experimental_reconstruction_training=1)

    def test_hub_tokens_are_runtime_only_for_every_wrapper(self):
        secret = "private-hub-token"
        models = (
            BarkForTextToSpeech(token=f" {secret} "),
            SpeechT5ForTextToSpeech(token=secret),
            VitsForTextToSpeech(token=secret),
        )

        with tempfile.TemporaryDirectory() as directory:
            for index, model in enumerate(models):
                with self.subTest(model=model.config.model_type):
                    serialized = json.dumps(
                        model.config.to_dict(),
                        sort_keys=True,
                    )
                    self.assertNotIn(secret, serialized)
                    self.assertNotIn("token", model.config.to_dict())
                    self.assertEqual(model._hub_kwargs()["token"], secret)
                    config_path = model.config.save_pretrained(Path(directory) / str(index))
                    self.assertNotIn(
                        secret,
                        Path(config_path).read_text(encoding="utf-8"),
                    )

        for model_class in (
                BarkForTextToSpeech,
                SpeechT5ForTextToSpeech,
                VitsForTextToSpeech,
        ):
            with self.subTest(model=model_class.__name__):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    model_class(token=" ")

    def test_vits_generator_training_requires_explicit_opt_in(self):
        model = VitsForTextToSpeech(lazy_load=True)
        adapter = AutoTrainingAdapter.from_model(model)

        self.assertIsInstance(
            adapter,
            NativeVitsGeneratorTrainingAdapter,
        )
        self.assertTrue(adapter.supports_custom_recipe)
        self.assertFalse(adapter.experimental_reconstruction_enabled)
        with self.assertRaisesRegex(
                ValueError,
                "generator-only warm-start",
        ):
            adapter.validate_support()
        self.assertFalse(model.is_loaded)

    def test_raw_safetensors_source_uses_sibling_config_and_processor(self):
        if not TORCH_AVAILABLE:
            self.skipTest("Native Safetensors checkpoints require PyTorch")
        import torch

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            expected = torch.tensor([1.0])
            save_safetensors({"weight": expected}, checkpoint)
            model = BarkForTextToSpeech(
                checkpoint,
                config_name_or_path=directory,
                processor_name_or_path=directory,
            )
            state_dict = model._direct_state_dict()

            torch.testing.assert_close(state_dict["weight"], expected)
            self.assertEqual(
                model._model_source(),
                str(Path(directory).resolve()),
            )
            self.assertEqual(model._config_source(), directory)

    def test_modules_remain_lazy_without_torch_or_transformers(self):
        script = (
            "import sys; "
            "import voicehub.models.bark; "
            "import voicehub.models.speecht5; "
            "import voicehub.models.vits; "
            "print('torch' in sys.modules, 'transformers' in sys.modules)")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "False False")


class TransformersTTSRegistryTests(unittest.TestCase):

    EXPECTED = {
        "bark": (
            "suno/bark-small",
            TrainingFamily.COMPOSITE,
            TrainingSupport.PREPROCESSED,
            True,
            True,
        ),
        "speecht5": (
            "microsoft/speecht5_tts",
            TrainingFamily.SEQ2SEQ,
            TrainingSupport.NATIVE,
            True,
            True,
        ),
        "vits": (
            "facebook/mms-tts-eng",
            TrainingFamily.VITS,
            TrainingSupport.PREPROCESSED,
            True,
            True,
        ),
    }

    def test_registry_exposes_real_checkpoints_and_training_profiles(self):
        for model_type, (
                checkpoint,
                family,
                support,
                native_training,
                advertises_fine_tuning,
        ) in self.EXPECTED.items():
            with self.subTest(model_type=model_type):
                inference = get_model_spec(model_type)
                training = get_training_spec(model_type)
                self.assertEqual(inference.default_model_path, checkpoint)
                self.assertIsNone(inference.install_extra)
                self.assertIn("safetensors", inference.capabilities)
                self.assertEqual(
                    "fine-tuning" in inference.capabilities,
                    advertises_fine_tuning,
                )
                self.assertEqual(training.family, family)
                self.assertEqual(training.support, support)
                self.assertEqual(training.install_extra, "training")
                self.assertEqual(training.native_training, native_training)

    def test_aliases_and_auto_factories_resolve_concrete_wrappers(self):
        cases = {
            "bark-tts": BarkForTextToSpeech,
            "speech-t5": SpeechT5ForTextToSpeech,
            "mms-tts": VitsForTextToSpeech,
        }
        for alias, expected_class in cases.items():
            with self.subTest(alias=alias):
                model = AutoModelForTextToSpeech.from_pretrained(
                    "",
                    model_type=alias,
                )
                legacy_model = AutoModelForTextToSpeech.from_pretrained(model_type=alias)
                self.assertIsInstance(model, expected_class)
                self.assertIsInstance(legacy_model, expected_class)
                self.assertFalse(model.is_loaded)

    def test_model_spec_accepts_none_or_a_nonempty_extension_extra(self):
        base = {
            "model_type": "future-transformers-tts",
            "module": "future.model",
            "class_name": "FutureForTextToSpeech",
            "default_model_path": "org/model",
        }
        self.assertIsNone(ModelSpec(**base).install_extra)
        self.assertEqual(
            ModelSpec(**base, install_extra=" future ").install_extra,
            "future",
        )
        with self.assertRaisesRegex(ValueError, "non-empty string or None"):
            ModelSpec(**base, install_extra=" ")


class TransformersTTSInferenceTests(unittest.TestCase):

    def test_bark_normalizes_and_trims_native_waveform(self):

        class Processor:

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": "ids"}

        class Model:

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return np.array([[0.1, 0.2, 0.3, 9.0]], dtype=np.float32), [3]

        wrapper = BarkForTextToSpeech(device="cpu")
        wrapper.model = Model()
        wrapper.transformers_processor = Processor()
        wrapper._torch = _FakeTorch()

        with patch(
                "voicehub.models.bark.modeling.seeded_inference",
                _fixed_seed,
        ):
            output = wrapper._generate(
                "hello",
                voice_preset="v2/en_speaker_6",
                max_new_tokens=20,
            )

        np.testing.assert_allclose(output.audio, [0.1, 0.2, 0.3])
        self.assertEqual(output.sample_rate, 24_000)
        self.assertEqual(
            wrapper.model.kwargs["semantic_max_new_tokens"],
            20,
        )
        self.assertTrue(wrapper.model.kwargs["return_output_lengths"])
        self.assertEqual(output.metadata["voice_preset"], "v2/en_speaker_6")

    def test_speecht5_passes_vocoder_and_speaker_conditioning(self):

        class Processor:

            def __call__(self, **kwargs):
                return {"input_ids": "ids", "attention_mask": "mask"}

        class Model:

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return np.array([[0.4, 0.5, 7.0]], dtype=np.float32), [2]

        wrapper = SpeechT5ForTextToSpeech(device="cpu")
        wrapper.model = Model()
        wrapper.vocoder = object()
        wrapper.transformers_processor = Processor()
        wrapper._torch = _FakeTorch()
        wrapper._coerce_speaker_embeddings = Mock(return_value="speaker")

        with patch(
                "voicehub.models.speecht5.modeling.seeded_inference",
                _fixed_seed,
        ):
            output = wrapper._generate(
                "hello",
                speaker_embeddings=[0.0] * 512,
            )

        np.testing.assert_allclose(output.audio, [0.4, 0.5])
        self.assertIs(wrapper.model.kwargs["vocoder"], wrapper.vocoder)
        self.assertEqual(
            wrapper.model.kwargs["speaker_embeddings"],
            "speaker",
        )
        self.assertTrue(wrapper.model.kwargs["return_output_lengths"])

    def test_speecht5_training_materializes_manifest_audio_paths(self):
        from voicehub.audio import AudioInput

        wrapper = SpeechT5ForTextToSpeech(device="cpu")
        loaded = AudioInput(
            waveform=np.array([0.1, 0.2], dtype=np.float32),
            sampling_rate=16_000,
        )
        with patch(
                "voicehub.models.speecht5.modeling.load_audio",
                return_value=loaded,
        ) as loader:
            waveform, sampling_rate, batch_size = wrapper._training_audio_batch(
                "/dataset/wavs/sample.wav",
                None,
            )

        loader.assert_called_once_with(
            "/dataset/wavs/sample.wav",
            target_sampling_rate=16_000,
        )
        np.testing.assert_allclose(waveform, loaded.waveform)
        self.assertEqual(sampling_rate, 16_000)
        self.assertEqual(batch_size, 1)

    def test_speecht5_training_resamples_materialized_audio_inputs(self):
        from voicehub.audio import AudioInput

        wrapper = SpeechT5ForTextToSpeech(device="cpu")
        source = AudioInput(
            waveform=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sampling_rate=22_050,
        )
        loaded = AudioInput(
            waveform=np.array([0.1, 0.25], dtype=np.float32),
            sampling_rate=16_000,
        )
        with patch(
                "voicehub.models.speecht5.modeling.load_audio",
                return_value=loaded,
        ) as loader:
            waveform, sampling_rate, batch_size = wrapper._training_audio_batch(
                source,
                None,
            )

        loader.assert_called_once_with(
            source,
            target_sampling_rate=16_000,
        )
        np.testing.assert_allclose(waveform, loaded.waveform)
        self.assertEqual(sampling_rate, 16_000)
        self.assertEqual(batch_size, 1)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for batched audio")
    def test_speecht5_training_resamples_collated_waveform_mappings(self):
        import torch

        from voicehub.audio import AudioInput

        wrapper = SpeechT5ForTextToSpeech(device="cpu")
        batch = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])
        loaded = [
            AudioInput(
                waveform=np.array([0.1, 0.25], dtype=np.float32),
                sampling_rate=16_000,
            ),
            AudioInput(
                waveform=np.array([0.4, 0.55], dtype=np.float32),
                sampling_rate=16_000,
            ),
        ]
        with patch(
                "voicehub.models.speecht5.modeling.load_audio",
                side_effect=loaded,
        ) as loader:
            waveforms, sampling_rate, batch_size = wrapper._training_audio_batch(
                {
                    "array": batch,
                    "sampling_rate": torch.tensor([22_050, 22_050]),
                },
                None,
            )

        self.assertEqual(loader.call_count, 2)
        for index, call in enumerate(loader.call_args_list):
            torch.testing.assert_close(call.args[0], batch[index])
            self.assertEqual(call.kwargs["sampling_rate"], 22_050)
            self.assertEqual(
                call.kwargs["target_sampling_rate"],
                16_000,
            )
        self.assertEqual(sampling_rate, 16_000)
        np.testing.assert_allclose(waveforms[0], loaded[0].waveform)
        np.testing.assert_allclose(waveforms[1], loaded[1].waveform)
        self.assertEqual(batch_size, 2)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for native VITS")
    def test_vits_uses_request_local_sampling_and_sequence_length(self):
        import torch

        class Model:

            def __init__(self):
                self.sampling = None

            def synthesize(self, **kwargs):
                self.kwargs = kwargs
                self.sampling = kwargs["sampling"]
                return SimpleNamespace(
                    waveform=torch.tensor([[0.2, 0.3, 0.4, 8.0]]),
                    sequence_lengths=torch.tensor([3]),
                )

        wrapper = VitsForTextToSpeech(device="cpu")
        wrapper.model = Model()
        wrapper.native_config = SimpleNamespace(
            num_speakers=3,
            noise_scale=0.6,
            noise_scale_duration=0.7,
        )
        wrapper.tokenizer = SimpleNamespace(config=SimpleNamespace(language="eng"), )
        wrapper._torch = torch

        with patch.object(
                wrapper,
                "_tokenize",
                return_value={
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.tensor([[True, True]]),
                },
        ):
            output = wrapper._generate(
                "hello",
                speaker_id=2,
                speed=1.25,
                noise_scale=0.2,
                noise_scale_duration=0.3,
                seed=123,
            )

        np.testing.assert_allclose(output.audio, [0.2, 0.3, 0.4])
        self.assertEqual(wrapper.model.sampling.noise_scale, 0.2)
        self.assertEqual(wrapper.model.sampling.noise_scale_duration, 0.3)
        self.assertEqual(wrapper.model.sampling.speaking_rate, 1.25)
        self.assertEqual(wrapper.model.sampling.seed, 123)
        self.assertNotIn("noise_scale", wrapper.model.__dict__)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for objective tests")
class TransformersTTSTrainingObjectiveTests(unittest.TestCase):

    @staticmethod
    def _training_vits_wrapper(torch, *, enabled: bool):

        class NativeVits(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, input_ids, *, spectrogram, **kwargs):
                del kwargs
                waveform = input_ids.float().repeat_interleave(128, dim=-1)
                batch_size = input_ids.shape[0]
                frames = spectrogram.shape[-1]
                latent = spectrogram[:, :1] * self.scale
                mask = torch.ones(
                    batch_size,
                    1,
                    frames,
                    device=spectrogram.device,
                )
                return SimpleNamespace(
                    waveform=waveform * self.scale,
                    sequence_lengths=torch.full(
                        (batch_size, ),
                        waveform.shape[-1],
                        dtype=torch.long,
                    ),
                    duration_loss=self.scale.square(),
                    prior_latents=latent,
                    posterior_log_variances=torch.zeros_like(latent),
                    expanded_prior_means=torch.zeros_like(latent),
                    expanded_prior_log_variances=torch.zeros_like(latent),
                    spectrogram_mask=mask,
                )

        wrapper = VitsForTextToSpeech(
            device="cpu",
            enable_native_generator_training=enabled,
        )
        wrapper.model = NativeVits()
        wrapper._torch = torch
        return wrapper

    def test_vits_shared_trainer_reaches_explicit_reconstruction_recipe(self):
        import torch

        wrapper = self._training_vits_wrapper(torch, enabled=True)
        initial_scale = wrapper.model.scale.detach().clone()
        adapter = AutoTrainingAdapter.from_model(wrapper)
        dataset = adapter.create_dataset([{
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "spectrogram": torch.ones(1, 4),
            "audio_values": torch.ones(512),
        }])

        self.assertIsInstance(
            adapter,
            NativeVitsGeneratorTrainingAdapter,
        )
        self.assertTrue(adapter.experimental_reconstruction_enabled)
        self.assertTrue(adapter.supports_custom_recipe)
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=wrapper,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    max_grad_norm=0,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=dataset,
            )
            result = trainer.train()

        self.assertIsInstance(
            trainer.training_adapter,
            NativeVitsGeneratorTrainingAdapter,
        )
        self.assertEqual(result.global_step, 1)
        self.assertFalse(torch.equal(wrapper.model.scale, initial_scale))
        output = trainer.training_adapter(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            spectrogram=torch.ones(1, 1, 4),
            audio_values=torch.ones(1, 512),
        )
        self.assertFalse(output.metadata["full_vits_fine_tuning"])
        self.assertEqual(
            output.metadata["objective"],
            "preprocessed-generator-warm-start",
        )

    def test_vits_shared_trainer_fails_closed_without_opt_in(self):
        import torch

        wrapper = self._training_vits_wrapper(torch, enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=wrapper,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=[{
                    "input_ids": torch.tensor([1, 2, 3, 4]),
                    "spectrogram": torch.ones(1, 4),
                    "audio_values": torch.ones(512),
                }],
            )
            with self.assertRaisesRegex(
                    ValueError,
                    "generator-only warm-start",
            ):
                trainer.train()

        self.assertFalse(wrapper._training_ready)

    def test_bark_facade_computes_differentiable_stage_losses(self):
        import torch

        class TokenModel(torch.nn.Module):

            def __init__(self, *, fine=False):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 8)
                self.head = torch.nn.Linear(8, 16)
                self.fine = fine

            def forward(self, input_ids, **kwargs):
                del kwargs
                if input_ids.ndim == 3:
                    input_ids = input_ids[..., 0]
                return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

        native = SimpleNamespace(
            semantic=TokenModel(),
            coarse_acoustics=TokenModel(),
            fine_acoustics=TokenModel(fine=True),
        )
        facade = _build_bark_training_model(torch, native)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[1, 2, 3, 4]])

        semantic = facade.semantic(input_ids, labels=labels)
        fine = facade.fine(
            input_ids.unsqueeze(-1).repeat(1, 1, 2),
            labels=labels,
            codebook_idx=1,
        )
        total = semantic["loss"] + fine["loss"]
        total.backward()

        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(native.semantic.head.weight.grad)
        self.assertIsNotNone(native.fine_acoustics.head.weight.grad)

    def test_vits_facade_computes_waveform_and_spectral_loss(self):
        import torch

        class NativeVits(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, input_ids, *, spectrogram, **kwargs):
                del kwargs
                waveform = input_ids.float().repeat_interleave(128, dim=-1)
                latent = spectrogram[:, :1] * self.scale
                return SimpleNamespace(
                    waveform=waveform * self.scale,
                    sequence_lengths=torch.tensor([waveform.shape[-1]]),
                    duration_loss=self.scale.square(),
                    prior_latents=latent,
                    posterior_log_variances=torch.zeros_like(latent),
                    expanded_prior_means=torch.zeros_like(latent),
                    expanded_prior_log_variances=torch.zeros_like(latent),
                    spectrogram_mask=torch.ones_like(latent),
                )

        native = NativeVits()
        facade = _build_vits_training_model(
            torch,
            native,
            spectral_loss_weight=0.1,
        )
        output = facade(
            torch.tensor([[1, 2, 3, 4]]),
            spectrogram=torch.ones(1, 1, 4),
            audio_values=torch.ones(1, 512),
        )
        output["loss"].backward()

        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(output["losses"]["spectral_loss"]))
        self.assertIsNotNone(native.scale.grad)


if __name__ == "__main__":
    unittest.main()
