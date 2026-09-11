from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from voicehub.models.inflecttts.modeling import InflectTTSForTextToSpeech
from voicehub.models.inflecttts.native.checkpoint import (
    INFLECT_MICRO_V2_INVENTORY_FINGERPRINT,
    INFLECT_NANO_V2_INVENTORY_FINGERPRINT,
    export_inflect_checkpoint,
    load_inflect_checkpoint,
    resolve_inflect_artifacts,
    tensor_inventory_fingerprint,
)
from voicehub.models.inflecttts.native.configuration import (
    INFLECT_MICRO_V2_CONFIG,
    INFLECT_NANO_V2_CONFIG,
    InflectV2Config,
)
from voicehub.models.inflecttts.native.frontend import InflectFrontendError, phonemes_to_ids
from voicehub.models.inflecttts.native.modeling import build_inflect_model
from voicehub.models.inflecttts.native.registration import create_inflect_architecture_spec
from voicehub.models.inflecttts.native.runtime import InflectV2Runtime
from voicehub.models.inflecttts.native.training import InflectV2TrainingModel


def _tiny_config(*, training: bool = False) -> InflectV2Config:
    return InflectV2Config(
        vocabulary_size=178,
        segment_size=16,
        sample_rate=24_000,
        filter_length=16,
        hop_length=4,
        win_length=16,
        mel_channels=4,
        mel_max_frequency=12_000.0,
        inter_channels=8,
        hidden_channels=8,
        filter_channels=16,
        attention_heads=2,
        attention_layers=1,
        kernel_size=3,
        dropout=0.0,
        resblock_kernel_sizes=(3, ),
        resblock_dilation_sizes=((1, 3, 5), ),
        upsample_rates=(2, 2),
        upsample_initial_channel=32,
        upsample_kernel_sizes=(4, 4),
        inference_only=not training,
    )


def _build(config: InflectV2Config):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        return build_inflect_model(config)


class NativeInflectArchitectureTests(unittest.TestCase):

    def test_configuration_import_does_not_load_the_inflect_model_graph(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import voicehub.models.inflecttts."
                    "configuration; "
                    "print(*(int(name in sys.modules) for name in ("
                    "'voicehub.models.inflecttts.modeling', "
                    "'voicehub.models.inflecttts.native.modeling', "
                    "'voicehub.models.inflecttts.native.training')))"),
            ],
            cwd=Path(__file__).parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "0 0 0")

    def test_release_graphs_match_audited_exact_inventories(self):
        cases = (
            (
                INFLECT_MICRO_V2_CONFIG,
                9_356_513,
                INFLECT_MICRO_V2_INVENTORY_FINGERPRINT,
            ),
            (
                INFLECT_NANO_V2_CONFIG,
                3_966_721,
                INFLECT_NANO_V2_INVENTORY_FINGERPRINT,
            ),
        )
        for config, parameter_count, fingerprint in cases:
            with self.subTest(hidden_channels=config.hidden_channels):
                model = _build(config)
                state = model.state_dict()
                self.assertEqual(len(state), 410)
                self.assertEqual(
                    sum(tensor.numel() for tensor in state.values()),
                    parameter_count,
                )
                self.assertEqual(
                    tensor_inventory_fingerprint(state),
                    fingerprint,
                )

    def test_frontend_requires_explicit_checkpoint_compatible_phonemes(self):
        model = _build(_tiny_config())
        runtime = InflectV2Runtime(model.eval(), _tiny_config())
        self.assertIsNotNone(runtime._weight_norm_cache)
        self.assertGreater(runtime._weight_norm_cache.module_count, 0)
        runtime.train()
        self.assertIsNone(runtime._weight_norm_cache)
        runtime.eval()
        self.assertIsNotNone(runtime._weight_norm_cache)
        with self.assertRaisesRegex(
                InflectFrontendError,
                "requires checkpoint-compatible",
        ):
            runtime.synthesize("raw English text")
        self.assertEqual(phonemes_to_ids("a", add_blank=True), [0, 43, 0])
        with self.assertRaisesRegex(
                InflectFrontendError,
                "outside the published",
        ):
            phonemes_to_ids("🙂")

    def test_safe_export_strict_load_and_fresh_runtime_reload(self):
        config = _tiny_config()
        source_model = _build(config)
        with tempfile.TemporaryDirectory() as directory:
            export_inflect_checkpoint(source_model, config, directory)
            artifacts = resolve_inflect_artifacts(directory)
            loaded_model = _build(artifacts.config)
            report = load_inflect_checkpoint(loaded_model, artifacts)

            self.assertEqual(report.tensor_count, len(source_model.state_dict()))
            self.assertFalse(report.missing_training_tensors)
            for name, expected in source_model.state_dict().items():
                self.assertTrue(
                    torch.equal(expected,
                                loaded_model.state_dict()[name]),
                    name,
                )

    def test_legacy_pickle_is_explicitly_trust_gated(self):
        config = _tiny_config()
        model = _build(config)
        state = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                __import__("json").dumps(config.to_dict()),
                encoding="utf-8",
            )
            torch.save(
                {
                    "format": "inflect_vits_inference_checkpoint_v1",
                    "model": state,
                    "iteration": 0,
                    "learning_rate": 0.0,
                    "deployable_parameters": sum(value.numel() for value in state.values()),
                },
                root / "model.pth",
            )
            artifacts = resolve_inflect_artifacts(root)
            with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint"):
                load_inflect_checkpoint(_build(config), artifacts)
            report = load_inflect_checkpoint(
                _build(config),
                artifacts,
                trust_pickle_checkpoint=True,
            )
            self.assertEqual(report.tensor_count, len(state))

    def test_preprocessed_full_vits_objective_is_differentiable(self):
        config = _tiny_config(training=True)
        generator = _build(config)
        objective = InflectV2TrainingModel(
            generator,
            config,
            enable_discriminator=False,
        )
        output = objective.generator_objective(
            torch.tensor([[1, 2]], dtype=torch.long),
            input_lengths=torch.tensor([2]),
            spectrogram=torch.rand(1, config.spectrogram_channels, 6),
            spectrogram_lengths=torch.tensor([6]),
            audio_values=torch.rand(1, 24) * 0.2 - 0.1,
        )
        self.assertEqual(output["loss"].ndim, 0)
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertTrue(
            all(
                parameter.grad is not None for parameter in generator.parameters()
                if parameter.requires_grad))

    def test_wrapper_training_expands_public_graph_and_exports_fresh_reload(self):
        config = _tiny_config()
        with tempfile.TemporaryDirectory() as source_directory:
            export_inflect_checkpoint(
                _build(config),
                config,
                source_directory,
            )
            wrapper = InflectTTSForTextToSpeech(
                model_path=source_directory,
                device="cpu",
                enable_native_finetuning=True,
                training_enable_discriminator=False,
            )
            wrapper.load_for_training()
            self.assertIsInstance(
                wrapper.training_model,
                InflectV2TrainingModel,
            )
            adapter = wrapper.get_training_adapter()
            adapter.setup()
            self.assertEqual(
                tuple(phase.name for phase in adapter.plan_training_phases(0)),
                ("generator", ),
            )
            with self.assertRaisesRegex(
                    ValueError,
                    "discriminator training is disabled",
            ):
                adapter.select_training_phase("discriminator")
            self.assertFalse(wrapper.model.generator.inference_only)
            self.assertEqual(
                len([name for name in wrapper.model.generator.state_dict() if name.startswith("enc_q.")]),
                100,
            )
            with tempfile.TemporaryDirectory() as export_directory:
                wrapper.export_native_pretrained(export_directory)
                fresh = InflectTTSForTextToSpeech(
                    model_path=export_directory,
                    device="cpu",
                )
                fresh.load()
                self.assertFalse(fresh.model.generator.inference_only)
                self.assertEqual(
                    set(fresh.model.generator.state_dict()),
                    set(wrapper.model.generator.state_dict()),
                )

    def test_architecture_spec_is_truthful_about_release_boundaries(self):
        spec = create_inflect_architecture_spec()
        from voicehub.registry import get_model_spec
        from voicehub.training.contracts import TrainingSupport
        from voicehub.training.specs import get_training_spec

        model_spec = get_model_spec("inflecttts")
        training_spec = get_training_spec("inflecttts")
        self.assertEqual(spec.architecture_id, "inflecttts")
        self.assertTrue(spec.capabilities.training)
        self.assertIn(
            "trusted-pickle-conversion",
            spec.capabilities.features,
        )
        self.assertFalse(spec.metadata["official_safetensors_published"])
        self.assertFalse(spec.metadata["author_recipe_recovered"])
        self.assertTrue(spec.metadata["full_finetuning_ready"])
        self.assertTrue(model_spec.is_voicehub_native)
        self.assertEqual(model_spec.architecture, "inflecttts")
        self.assertTrue(training_spec.native_training)
        self.assertIs(training_spec.support, TrainingSupport.PREPROCESSED)
        self.assertEqual(
            tuple(phase.name for phase in training_spec.phases),
            ("generator", "discriminator"),
        )

    def test_executable_boundary_imports_only_torch_and_voicehub(self):
        package = (Path(__file__).parents[1] / 'voicehub/models/inflecttts/native')
        forbidden = {
            "librosa",
            "matplotlib",
            "num2words",
            "numpy",
            "phonemizer",
            "scipy",
            "soundfile",
            "transformers",
            "unidecode",
        }
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".", 1)[0])
            self.assertFalse(
                imports & forbidden,
                f"{path.name}: {sorted(imports & forbidden)}",
            )


if __name__ == "__main__":
    unittest.main()
