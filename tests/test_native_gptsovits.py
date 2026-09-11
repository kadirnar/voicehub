from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from voicehub.models.gptsovits.native.checkpoint import (
    S1_FILENAME,
    S2_GENERATOR_FILENAME,
    export_gptsovits_checkpoint,
    load_gptsovits_checkpoints,
    resolve_gptsovits_artifacts,
    tensor_inventory_fingerprint,
)
from voicehub.models.gptsovits.native.configuration import (
    SUPPORTED_GPT_SOVITS_VARIANTS,
    GPTSoVITSS1Config,
    GPTSoVITSS2Config,
)
from voicehub.models.gptsovits.native.frontend import (
    GPTSoVITSFrontendError,
    reject_raw_text,
    validate_prepared_inference,
)
from voicehub.models.gptsovits.native.metadata import GPT_SOVITS_VARIANTS
from voicehub.models.gptsovits.native.modeling import build_s2_discriminator, build_s2_generator
from voicehub.models.gptsovits.native.registration import create_gptsovits_architecture_spec
from voicehub.models.gptsovits.native.runtime import GPTSoVITSRuntime, TTS_Config
from voicehub.models.gptsovits.native.semantic import GPTSoVITSSemanticModel
from voicehub.models.gptsovits.native.training import GPTSoVITSS2TrainingModel, build_staged_training_model
from voicehub.models.gptsovits.training import GPTSoVITSTrainingAdapter
from voicehub.training.contracts import TrainingPhaseSpec, TrainingSupport
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _s1_upstream_config() -> dict:
    return {
        "model": {
            "vocab_size": 1_025,
            "phoneme_vocab_size": 732,
            "embedding_dim": 512,
            "hidden_dim": 512,
            "head": 16,
            "n_layer": 24,
            "dropout": 0,
            "EOS": 1_024,
        },
    }


def _s2_upstream_config() -> dict:
    return {
        "train": {
            "segment_size": 20_480,
            "c_mel": 45,
            "c_kl": 1.0,
        },
        "data": {
            "sampling_rate": 32_000,
            "filter_length": 2_048,
            "hop_length": 640,
            "win_length": 2_048,
            "n_mel_channels": 128,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "n_speakers": 300,
        },
        "model": {
            "version": "v2",
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [
                [1, 3, 5],
                [1, 3, 5],
                [1, 3, 5],
            ],
            "upsample_initial_channel": 512,
            "upsample_rates": [10, 8, 2, 2, 2],
            "upsample_kernel_sizes": [16, 16, 8, 2, 2],
            "gin_channels": 512,
            "semantic_frame_rate": "25hz",
            "freeze_quantizer": True,
            "use_spectral_norm": False,
        },
    }


def _s2_batch() -> dict[str, torch.Tensor]:
    return {
        "ssl_features": torch.randn(1, 768, 32),
        "spectrogram": torch.rand(1, 1_025, 32),
        "spectrogram_lengths": torch.tensor([32]),
        "audio_values": torch.rand(1, 20_480) * 0.1,
        "phoneme_ids": torch.tensor([[1, 2, 3, 4]]),
        "phoneme_lengths": torch.tensor([4]),
    }


class NativeGPTSoVITSTests(unittest.TestCase):

    def test_release_graphs_match_every_supported_variant_inventory(self):
        for variant, release in GPT_SOVITS_VARIANTS.items():
            s1_config = GPTSoVITSS1Config.for_variant(variant)
            s2_config = GPTSoVITSS2Config.for_variant(variant)
            cases = (
                (
                    lambda: GPTSoVITSSemanticModel(s1_config),
                    release.s1,
                    "s1",
                ),
                (
                    lambda: build_s2_generator(s2_config),
                    release.s2_generator,
                    "s2_generator",
                ),
                (
                    lambda: build_s2_discriminator(s2_config),
                    release.s2_discriminator,
                    "s2_discriminator",
                ),
            )
            for factory, checkpoint, component in cases:
                with self.subTest(variant=variant, component=component):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        with torch.device("meta"):
                            model = factory().half()
                    state = model.state_dict()
                    self.assertEqual(len(state), checkpoint.tensor_count)
                    self.assertEqual(
                        sum(tensor.numel() for tensor in state.values()),
                        checkpoint.parameter_count,
                    )
                    self.assertEqual(
                        tensor_inventory_fingerprint(state),
                        checkpoint.inventory_fingerprint,
                    )

    def test_configuration_and_runtime_fail_closed_on_other_graph_families(self):
        with self.assertRaisesRegex(ValueError, "flow-matching/vocoder"):
            GPTSoVITSS2Config(version="v3")
        with self.assertRaisesRegex(ValueError, "topology mismatch"):
            GPTSoVITSS1Config.from_upstream({
                "model": {
                    **_s1_upstream_config()["model"],
                    "n_layer": 12,
                },
            })
        selected = TTS_Config({"v2": {"artifact_root": "/tmp/gptsovits"}})
        self.assertEqual(selected.artifact_root, "/tmp/gptsovits")
        self.assertEqual(selected.variant, "v2")
        for variant in SUPPORTED_GPT_SOVITS_VARIANTS:
            selected = TTS_Config({
                variant: {
                    "artifact_root": "/tmp/gptsovits",
                },
            })
            self.assertEqual(selected.variant, variant)
        with self.assertRaisesRegex(ValueError, "flow-matching"):
            TTS_Config({"v4": {"artifact_root": "/tmp/gptsovits"}})
        with self.assertRaisesRegex(ValueError, "LoRA"):
            TTS_Config({"v3lora": {"artifact_root": "/tmp/gptsovits"}})

    def test_pinned_source_configs_dispatch_to_exact_classic_variants(self):
        v1_s1 = _s1_upstream_config()
        v1_s1["model"]["phoneme_vocab_size"] = 512
        self.assertEqual(
            GPTSoVITSS1Config.from_upstream(
                v1_s1,
                variant="v1",
            ).version,
            "v1",
        )
        v1_s2 = _s2_upstream_config()
        v1_s2["model"].pop("version")
        self.assertEqual(
            GPTSoVITSS2Config.from_upstream(
                v1_s2,
                variant="v1",
            ).version,
            "v1",
        )
        source_configs = (
            ("v2Pro", "s2v2Pro.json"),
            ("v2ProPlus", "s2v2ProPlus.json"),
        )
        config_root = (
            PROJECT_ROOT / "voicehub" / "models" / "gptsovits" / "source" / "GPT_SoVITS" / "configs")
        for variant, filename in source_configs:
            with self.subTest(variant=variant):
                payload = json.loads((config_root / filename).read_text(encoding="utf-8"))
                self.assertEqual(
                    GPTSoVITSS2Config.from_upstream(
                        payload,
                        variant=variant,
                    ).version,
                    variant,
                )

    def test_frontend_accepts_exact_prepared_tensors_and_rejects_raw_text(self):
        with self.assertRaisesRegex(GPTSoVITSFrontendError, "does not guess"):
            reject_raw_text("raw text")
        prepared = validate_prepared_inference(
            s1_phoneme_ids=[1, 2],
            s1_bert_features=torch.zeros(1, 1_024, 2),
            s2_phoneme_ids=[3, 4],
            prompt_semantic_ids=[5, 6],
            reference_spectrogram=torch.zeros(1_025, 3),
            semantic_codes=[7, 8],
        )
        self.assertEqual(
            tuple(prepared["semantic_codes"].shape),
            (1, 1, 2),
        )
        with self.assertRaisesRegex(
                GPTSoVITSFrontendError,
                "BERT features",
        ):
            validate_prepared_inference(
                s1_phoneme_ids=[1, 2],
                s1_bert_features=torch.zeros(1, 768, 2),
                s2_phoneme_ids=[3, 4],
                prompt_semantic_ids=None,
                reference_spectrogram=torch.zeros(1_025, 3),
            )

    def test_pro_frontend_requires_exact_prepared_speaker_embedding(self):
        s1_config = GPTSoVITSS1Config.for_variant("v2Pro")
        s2_config = GPTSoVITSS2Config.for_variant("v2Pro")
        common = {
            "s1_phoneme_ids": [1, 2],
            "s1_bert_features": torch.zeros(1, 1_024, 2),
            "s2_phoneme_ids": [3, 4],
            "prompt_semantic_ids": None,
            "reference_spectrogram": torch.zeros(1_025, 3),
            "semantic_codes": [5, 6],
            "s1_config": s1_config,
            "s2_config": s2_config,
        }
        with self.assertRaisesRegex(
                GPTSoVITSFrontendError,
                "speaker embedding",
        ):
            validate_prepared_inference(**common)
        prepared = validate_prepared_inference(
            **common,
            speaker_embedding=torch.zeros(20_480),
        )
        self.assertEqual(
            tuple(prepared["speaker_embedding"].shape),
            (1, 20_480),
        )
        with self.assertRaisesRegex(
                GPTSoVITSFrontendError,
                "does not consume speaker embeddings",
        ):
            validate_prepared_inference(
                **{
                    **common,
                    "s1_config": GPTSoVITSS1Config.for_variant("v1"),
                    "s2_config": GPTSoVITSS2Config.for_variant("v1"),
                    "s1_phoneme_ids": [1, 2],
                    "s2_phoneme_ids": [3, 4],
                },
                speaker_embedding=torch.zeros(20_480),
            )

    def test_s1_source_sum_cross_entropy_is_differentiable(self):
        model = GPTSoVITSSemanticModel()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.model.ar_predict_layer.weight.requires_grad_(True)
        output = model(
            phoneme_ids=torch.tensor([[1, 2]]),
            phoneme_lengths=torch.tensor([2]),
            semantic_ids=torch.tensor([[3, 4]]),
            semantic_lengths=torch.tensor([2]),
            bert_features=torch.randn(1, 1_024, 2),
        )
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(output["loss"].ndim, 0)
        output["loss"].backward()
        self.assertIsNotNone(model.model.ar_predict_layer.weight.grad)

    def test_pro_staged_training_builds_matching_s1_generator_and_discriminator(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with torch.device("meta"):
                model = build_staged_training_model(variant="v2Pro")
        self.assertEqual(model.s1.config.version, "v2")
        self.assertEqual(model.s2.config.version, "v2Pro")
        self.assertEqual(model.s2.generator.sv_emb.in_features, 20_480)
        self.assertEqual(len(model.s2.discriminator.discriminators), 8)

    def test_pro_speaker_conditioning_is_differentiable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            generator = build_s2_generator(GPTSoVITSS2Config.for_variant("v2Pro"))
        for parameter in generator.parameters():
            parameter.requires_grad_(False)
        generator.sv_emb.weight.requires_grad_(True)
        style, text_style, _ = generator._style(
            torch.randn(1, 1_025, 3),
            torch.tensor([3]),
            torch.randn(1, 20_480),
        )
        (style.sum() + text_style.sum()).backward()
        self.assertIsNotNone(generator.sv_emb.weight.grad)
        self.assertTrue(torch.isfinite(generator.sv_emb.weight.grad).all())

    def test_s2_vits_objective_is_differentiable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            generator = build_s2_generator()
        for parameter in generator.parameters():
            parameter.requires_grad_(False)
        generator.dec.conv_post.weight.requires_grad_(True)
        generator.quantizer.vq.layers[0]._codebook.inited.fill_(1)
        objective = GPTSoVITSS2TrainingModel(
            generator,
            enable_discriminator=False,
        )
        output = objective.generator_objective(**_s2_batch())
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(output["mel_loss"]))
        self.assertTrue(torch.isfinite(output["kl_loss"]))
        output["loss"].backward()
        self.assertIsNotNone(generator.dec.conv_post.weight.grad)

    def test_s2_discriminator_has_its_own_differentiable_phase(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            generator = build_s2_generator()
            discriminator = build_s2_discriminator()
        for parameter in generator.parameters():
            parameter.requires_grad_(False)
        generator.quantizer.vq.layers[0]._codebook.inited.fill_(1)
        objective = GPTSoVITSS2TrainingModel(
            generator,
            discriminator=discriminator,
        )
        output = objective.discriminator_objective(**_s2_batch())
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertTrue(all(parameter.grad is not None for parameter in discriminator.parameters()), )
        self.assertTrue(all(parameter.grad is None for parameter in generator.parameters()), )

    def test_precomputed_semantics_run_through_fresh_native_runtime(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            generator = build_s2_generator().eval()
        generator.quantizer.vq.layers[0]._codebook.inited.fill_(1)
        runtime = GPTSoVITSRuntime(
            GPTSoVITSSemanticModel().eval(),
            generator,
        )
        self.assertIsNotNone(runtime._weight_norm_cache)
        self.assertGreater(runtime._weight_norm_cache.module_count, 0)
        runtime.train()
        self.assertIsNone(runtime._weight_norm_cache)
        runtime.eval()
        self.assertIsNotNone(runtime._weight_norm_cache)
        sample_rate, waveform = runtime.synthesize_prepared(
            s1_phoneme_ids=[1, 2],
            s1_bert_features=torch.zeros(1, 1_024, 2),
            s2_phoneme_ids=[3, 4, 5],
            prompt_semantic_ids=None,
            reference_spectrogram=torch.rand(1, 1_025, 4),
            semantic_codes=[6, 7],
            noise_scale=0.0,
        )
        self.assertEqual(sample_rate, 32_000)
        self.assertEqual(waveform.ndim, 1)
        self.assertGreater(waveform.numel(), 0)
        self.assertTrue(torch.isfinite(waveform).all())

    def test_safe_export_has_integrity_and_strict_fresh_reload(self):
        s1 = nn.Linear(2, 3)
        generator = nn.Linear(3, 4)
        discriminator = nn.Linear(4, 5)
        with tempfile.TemporaryDirectory() as directory:
            export_gptsovits_checkpoint(
                directory,
                s1=s1,
                s2_generator=generator,
                s2_discriminator=discriminator,
                s1_config=GPTSoVITSS1Config(),
                s2_config=GPTSoVITSS2Config(),
            )
            artifacts = resolve_gptsovits_artifacts(
                directory,
                require_discriminator=True,
            )
            restored_s1 = nn.Linear(2, 3)
            restored_generator = nn.Linear(3, 4)
            restored_discriminator = nn.Linear(4, 5)
            reports = load_gptsovits_checkpoints(
                s1=restored_s1,
                s2_generator=restored_generator,
                s2_discriminator=restored_discriminator,
                artifacts=artifacts,
            )
            self.assertEqual(
                set(reports),
                {"s1", "s2_generator", "s2_discriminator"},
            )
            for source, restored in (
                (s1, restored_s1),
                (generator, restored_generator),
                (discriminator, restored_discriminator),
            ):
                for name, expected in source.state_dict().items():
                    torch.testing.assert_close(
                        restored.state_dict()[name],
                        expected,
                    )
            with self.assertRaisesRegex(ValueError, "inventory is incompatible"):
                load_gptsovits_checkpoints(
                    s1=nn.Linear(2, 4),
                    s2_generator=nn.Linear(3, 4),
                    artifacts=artifacts,
                )

            artifacts.s1_path.write_bytes(artifacts.s1_path.read_bytes() + b"tampered", )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                resolve_gptsovits_artifacts(directory)

    def test_native_export_round_trips_variant_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            export_gptsovits_checkpoint(
                directory,
                s1=nn.Linear(2, 3),
                s2_generator=nn.Linear(3, 4),
                s1_config=GPTSoVITSS1Config.for_variant("v1"),
                s2_config=GPTSoVITSS2Config.for_variant("v1"),
            )
            inferred = resolve_gptsovits_artifacts(directory)
            self.assertEqual(inferred.s1_config.version, "v1")
            self.assertEqual(inferred.s2_config.version, "v1")
            with self.assertRaisesRegex(ValueError, "does not match requested"):
                resolve_gptsovits_artifacts(directory, variant="v2")

    def test_legacy_pickle_requires_explicit_trust(self):
        s1 = nn.Linear(2, 3)
        generator = nn.Linear(3, 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(
                {
                    "weight": s1.state_dict(),
                    "config": _s1_upstream_config(),
                },
                root / S1_FILENAME,
            )
            torch.save(
                {
                    "weight": generator.state_dict(),
                    "config": _s2_upstream_config(),
                },
                root / S2_GENERATOR_FILENAME,
            )
            artifacts = resolve_gptsovits_artifacts(root)
            with self.assertRaisesRegex(
                    ValueError,
                    "trust_pickle_checkpoint",
            ):
                load_gptsovits_checkpoints(
                    s1=nn.Linear(2, 3),
                    s2_generator=nn.Linear(3, 4),
                    artifacts=artifacts,
                )
            reports = load_gptsovits_checkpoints(
                s1=nn.Linear(2, 3),
                s2_generator=nn.Linear(3, 4),
                artifacts=artifacts,
                trust_pickle_checkpoint=True,
            )
            self.assertEqual(set(reports), {"s1", "s2_generator"})

    def test_architecture_spec_declares_truthful_staged_boundaries(self):
        spec = create_gptsovits_architecture_spec()
        from voicehub.registry import get_model_spec
        from voicehub.training.specs import get_training_spec

        model_spec = get_model_spec("gptsovits")
        training_spec = get_training_spec("gptsovits")
        self.assertEqual(spec.qualified_id, "gptsovits@2")
        self.assertTrue(spec.capabilities.training)
        self.assertIn(
            "separate-s1-s2-generator-s2-discriminator-phases",
            spec.capabilities.features,
        )
        self.assertFalse(spec.metadata["official_safetensors_published"])
        self.assertFalse(spec.metadata["raw_text_frontend_available"])
        self.assertEqual(
            spec.metadata["supported_variants"],
            ("v1", "v2", "v2Pro", "v2ProPlus"),
        )
        self.assertTrue(spec.metadata["full_finetuning_ready"])
        self.assertTrue(model_spec.is_voicehub_native)
        self.assertEqual(model_spec.architecture, "gptsovits")
        self.assertTrue(training_spec.native_training)
        self.assertIs(training_spec.support, TrainingSupport.PREPROCESSED)
        self.assertEqual(
            tuple(phase.name for phase in training_spec.phases),
            ("s1", "s2_generator", "s2_discriminator"),
        )

    def test_adapter_omits_disabled_discriminator_phase_truthfully(self):
        spec = ModelTrainingSpec(
            model_type="gptsovits",
            family=TrainingFamily.COMPOSITE,
            module_paths=("training_model", ),
            support=TrainingSupport.CUSTOM,
            phases=(
                TrainingPhaseSpec(name="s1"),
                TrainingPhaseSpec(name="s2_generator"),
                TrainingPhaseSpec(name="s2_discriminator"),
            ),
            default_phase="s1",
        )
        wrapper = SimpleNamespace(
            config=SimpleNamespace(
                model_type="gptsovits",
                training_enable_s2_discriminator=False,
            ), )
        adapter = GPTSoVITSTrainingAdapter(wrapper, spec)
        self.assertEqual(
            tuple(phase.name for phase in adapter.plan_training_phases(0)),
            ("s1", "s2_generator"),
        )
        with self.assertRaisesRegex(
                ValueError,
                "discriminator training is disabled",
        ):
            adapter.select_training_phase("s2_discriminator")
        self.assertEqual(
            adapter.artifact_manifest()["training_scope"],
            "preprocessed-s1-and-non-adversarial-s2",
        )

    def test_configuration_import_is_framework_lazy(self):
        script = (
            "import sys; "
            "from voicehub.models.gptsovits.configuration "
            "import GPTSoVITSConfig; "
            "print('torch' in sys.modules, 'transformers' in sys.modules, "
            "GPTSoVITSConfig.model_type)")
        output = subprocess.check_output(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
        self.assertEqual(output, "False False gptsovits")

    def test_active_native_boundary_has_no_provider_or_transformers_imports(self):
        paths = list((PROJECT_ROOT / 'voicehub/models/gptsovits/native').glob("*.py", ), )
        paths.extend([
            PROJECT_ROOT / "voicehub" / "models" / "gptsovits" / "__init__.py",
            PROJECT_ROOT / "voicehub" / "models" / "gptsovits" / "configuration.py",
            PROJECT_ROOT / 'voicehub/models/gptsovits/modeling.py',
            PROJECT_ROOT / "voicehub" / "models" / "gptsovits" / "training.py",
        ])
        forbidden = {
            "einops",
            "librosa",
            "numpy",
            "pytorch_lightning",
            "scipy",
            "soundfile",
            "torchmetrics",
            "transformers",
        }
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = set()
            provider_imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                else:
                    continue
                imports.update(module.split(".", 1)[0] for module in modules)
                provider_imports.extend(
                    module for module in modules if module.startswith("voicehub.models.gptsovits.source"))
            self.assertFalse(
                imports & forbidden,
                f"{path.name}: {sorted(imports & forbidden)}",
            )
            self.assertFalse(provider_imports, f"{path.name}: {provider_imports}")


if __name__ == "__main__":
    unittest.main()
