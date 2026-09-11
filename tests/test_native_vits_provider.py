from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from voicehub.models.vits.configuration import VitsConfig
from voicehub.models.vits.modeling import VitsForTextToSpeech
from voicehub.models.vits.training import NativeVitsGeneratorTrainingAdapter
from voicehub.registry import get_model_spec
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.specs import get_training_spec

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _native_config() -> object:
    from voicehub.models.vits.native.configuration import VitsConfig

    return VitsConfig(
        vocab_size=8,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        window_size=2,
        ffn_dim=16,
        ffn_kernel_size=3,
        flow_size=8,
        spectrogram_bins=5,
        layerdrop=0.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        use_stochastic_duration_prediction=False,
        upsample_initial_channel=16,
        upsample_rates=(2, ),
        upsample_kernel_sizes=(4, ),
        resblock_kernel_sizes=(3, ),
        resblock_dilation_sizes=((1, ), ),
        depth_separable_channels=2,
        depth_separable_num_layers=1,
        duration_predictor_flow_bins=4,
        duration_predictor_tail_bound=2.0,
        duration_predictor_kernel_size=3,
        duration_predictor_dropout=0.0,
        duration_predictor_num_flows=2,
        duration_predictor_filter_channels=8,
        prior_encoder_num_flows=2,
        prior_encoder_num_wavenet_layers=2,
        posterior_encoder_num_wavenet_layers=2,
        wavenet_kernel_size=3,
        wavenet_dilation_rate=2,
        wavenet_dropout=0.0,
        pad_token_id=0,
    )


def _write_checkpoint(directory: Path) -> None:
    from voicehub.checkpointing import save_safetensors
    from voicehub.hub import write_json_file
    from voicehub.models.vits.native.modeling import VitsModel

    native_config = _native_config()
    model = VitsModel(native_config)
    save_safetensors(
        model.state_dict(),
        directory / "model.safetensors",
    )
    write_json_file(directory / "config.json", native_config.to_dict())
    write_json_file(
        directory / "vocab.json",
        {
            "<pad>": 0,
            "<unk>": 1,
            " ": 2,
            "a": 3,
            "b": 4,
            "c": 5,
            "d": 6,
            "-": 7,
        },
    )
    write_json_file(
        directory / "tokenizer_config.json",
        {
            "add_blank": True,
            "language": "eng",
            "normalize": True,
            "pad_token": "<pad>",
            "phonemize": False,
            "unk_token": "<unk>",
        },
    )


class NativeVitsProviderDeclarationTests(unittest.TestCase):

    def test_public_package_does_not_import_torch_or_transformers(self):
        script = (
            "import sys; import voicehub.models.vits; "
            "print('torch' in sys.modules, 'transformers' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False False")

    def test_external_runtime_controls_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "coherent artifact root"):
            VitsConfig(config_name_or_path="other/config")
        with self.assertRaisesRegex(ValueError, "never executes repository code"):
            VitsConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "does not delegate"):
            VitsConfig(model_kwargs={"device_map": "auto"})
        with self.assertRaisesRegex(ValueError, "Safetensors only"):
            VitsConfig(use_safetensors=False)

    def test_training_profile_supports_legacy_and_adversarial_opt_ins(self):
        model_spec = get_model_spec("vits")
        self.assertTrue(model_spec.is_voicehub_native)
        self.assertEqual(model_spec.architecture, "vits")
        self.assertIn("fine-tuning", model_spec.capabilities)
        self.assertIn("adversarial-training", model_spec.capabilities)
        self.assertIn(
            "explicit-acoustic-training-config",
            model_spec.capabilities,
        )

        shared_spec = get_training_spec("vits")
        self.assertTrue(shared_spec.separate_optimizers)
        self.assertEqual(
            tuple(phase.name for phase in shared_spec.phases),
            ("discriminator", "generator"),
        )

        wrapper = VitsForTextToSpeech()
        adapter = AutoTrainingAdapter.from_model(wrapper)
        self.assertIsInstance(adapter, NativeVitsGeneratorTrainingAdapter)
        self.assertFalse(adapter.spec.separate_optimizers)
        self.assertEqual(
            tuple(phase.name for phase in adapter.spec.phases),
            ("generator", ),
        )
        with self.assertRaisesRegex(ValueError, "training is opt-in"):
            adapter.validate_support()

        enabled = VitsForTextToSpeech(enable_native_generator_training=True)
        enabled_adapter = AutoTrainingAdapter.from_model(enabled)
        enabled_adapter.validate_support()
        configuration = enabled_adapter.recipe_resume_configuration()
        self.assertFalse(configuration["full_vits_fine_tuning"])
        self.assertIn(
            "enable_native_adversarial_training",
            configuration["blocking_requirements"],
        )

        with self.assertRaisesRegex(ValueError, "training_acoustic_config"):
            AutoTrainingAdapter.from_model(VitsForTextToSpeech(
                enable_native_adversarial_training=True, ), ).validate_support()

        full = VitsForTextToSpeech(
            enable_native_adversarial_training=True,
            training_acoustic_config={
                "sampling_rate": 16_000,
                "filter_length": 1_024,
                "hop_length": 256,
                "win_length": 1_024,
                "num_mel_channels": 80,
                "mel_fmin": 0.0,
                "mel_fmax": 8_000.0,
                "segment_size": 8_192,
            },
        )
        full_adapter = AutoTrainingAdapter.from_model(full)
        full_adapter.validate_support()
        self.assertTrue(full_adapter.spec.separate_optimizers)
        self.assertEqual(
            tuple(phase.name for phase in full_adapter.spec.phases),
            ("discriminator", "generator"),
        )
        self.assertEqual(
            full_adapter.optimizer_plan(),
            {
                "discriminator": ("training_model.discriminator", ),
                "generator": ("training_model.native_model", ),
            },
        )
        full_configuration = full_adapter.recipe_resume_configuration()
        self.assertTrue(full_configuration["full_vits_fine_tuning"])
        self.assertFalse(full_configuration["mms_checkpoint_acoustic_metadata_inferred"])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for native VITS")
class NativeVitsProviderRuntimeTests(unittest.TestCase):

    def test_full_recipe_accepts_raw_audio_without_guessing_a_spectrogram(self):
        import torch

        model = VitsForTextToSpeech(
            enable_native_adversarial_training=True,
            training_acoustic_config={
                "sampling_rate": 16_000,
                "filter_length": 1_024,
                "hop_length": 256,
                "win_length": 1_024,
                "num_mel_channels": 80,
            },
        )
        model._torch = torch
        prepared = model.prepare_training_inputs(
            {
                "input_ids": [1, 2, 3],
                "audio_values": torch.randn(4_096),
            },
            phase="discriminator",
        )
        self.assertEqual(prepared["input_ids"].shape, (1, 3))
        self.assertEqual(prepared["audio_values"].shape, (1, 4_096))
        self.assertNotIn("spectrogram", prepared)

        legacy = VitsForTextToSpeech(enable_native_generator_training=True, )
        legacy._torch = torch
        with self.assertRaisesRegex(ValueError, "precomputed"):
            legacy.prepare_training_inputs(
                {
                    "input_ids": [1, 2, 3],
                    "audio_values": torch.randn(4_096),
                },
                phase="generator",
            )

    def test_local_checkpoint_inference_and_native_export_round_trip(self):
        import torch

        from voicehub.models.vits.native.modeling import WeightNormalizedConv1d

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            export = root / "export"
            source.mkdir()
            _write_checkpoint(source)

            model = VitsForTextToSpeech(
                source,
                device="cpu",
                lazy_load=False,
            )
            cached_layers = [
                module for module in model.model.modules() if isinstance(module, WeightNormalizedConv1d)
            ]
            self.assertTrue(cached_layers)
            self.assertTrue(all(layer._inference_weight is not None for layer in cached_layers))
            first = model(
                "ab",
                seed=17,
                max_output_frames=128,
            )
            self.assertEqual(first.metadata["backend"], "voicehub-native")
            self.assertEqual(first.metadata["seed"], 17)
            self.assertEqual(first.sample_rate, 16_000)
            self.assertEqual(first.audio.ndim, 1)
            self.assertTrue(torch.isfinite(first.audio).all())

            model.export_native_pretrained(export)
            restored = VitsForTextToSpeech(
                export,
                device="cpu",
                lazy_load=False,
            )
            second = restored(
                "ab",
                seed=17,
                max_output_frames=128,
            )
            torch.testing.assert_close(first.audio, second.audio)
            self.assertEqual(
                restored.artifacts.checkpoint.name,
                "model.safetensors",
            )

    def test_missing_checkpoint_frontend_fails_with_exact_asset(self):
        from voicehub.hub import write_json_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json_file(root / "config.json", _native_config().to_dict())
            write_json_file(root / "vocab.json", {"<pad>": 0})
            (root / "model.safetensors").touch()
            with self.assertRaisesRegex(
                    FileNotFoundError,
                    "tokenizer_config.json",
            ):
                VitsForTextToSpeech(
                    root,
                    device="cpu",
                    lazy_load=False,
                )


if __name__ == "__main__":
    unittest.main()
