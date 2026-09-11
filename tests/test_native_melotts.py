"""Focused contracts for VoiceHub's native MeloTTS implementation."""

from __future__ import annotations

import ast
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from voicehub.models.melotts.configuration import MeloTTSConfig
from voicehub.models.melotts.modeling import MeloTTSForTextToSpeech
from voicehub.models.melotts.native.checkpoint import (
    inspect_melotts_checkpoint,
    read_legacy_melotts_checkpoint,
    save_melotts_pretrained,
)
from voicehub.models.melotts.native.configuration import (
    MeloTTSArchitectureConfig,
    MeloTTSDataConfig,
    MeloTTSModelConfig,
)
from voicehub.models.melotts.native.frontend import NativeMeloTTSFrontend
from voicehub.models.melotts.native.metadata import (
    MELOTTS_EN_NEWEST_INVENTORY_FINGERPRINT,
    MELOTTS_EN_NEWEST_PARAMETER_COUNT,
    MELOTTS_EN_NEWEST_TENSOR_COUNT,
    MELOTTS_GENERATOR_COMPONENTS,
    MELOTTS_RELEASES,
    MELOTTS_SOURCE_REVISION,
)
from voicehub.models.melotts.native.modeling import DEPLOYABLE_MELOTTS_COMPONENTS, build_melotts_model
from voicehub.models.melotts.native.registration import create_melotts_architecture_spec
from voicehub.models.melotts.native.runtime import MeloTTSRuntime
from voicehub.models.melotts.native.training import (
    MeloTTSTrainingCollator,
    MeloTTSTrainingModel,
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
)
from voicehub.models.melotts.source.melo.models import TextEncoder
from voicehub.models.melotts.training import MeloTTSTrainingAdapter
from voicehub.registry import get_model_spec
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import get_training_spec


def _tiny_config(
    *,
    use_duration_discriminator: bool = False,
) -> MeloTTSArchitectureConfig:
    return MeloTTSArchitectureConfig(
        symbols=("_", "a", "b", "SP"),
        num_tones=3,
        num_languages=2,
        segment_size=16,
        data=MeloTTSDataConfig(
            sample_rate=16_000,
            n_fft=16,
            hop_length=4,
            win_length=16,
            n_mels=4,
            n_speakers=2,
            speaker_ids=(("speaker-a", 0), ("speaker-b", 1)),
        ),
        model=MeloTTSModelConfig(
            inter_channels=8,
            hidden_channels=8,
            filter_channels=16,
            n_heads=2,
            n_layers=3,
            n_layers_trans_flow=3,
            n_flow_layer=1,
            kernel_size=3,
            p_dropout=0.0,
            resblock="1",
            resblock_kernel_sizes=(3, ),
            resblock_dilation_sizes=((1, 3, 5), ),
            upsample_rates=(2, 2),
            upsample_initial_channel=16,
            upsample_kernel_sizes=(4, 4),
            gin_channels=8,
            use_noise_scaled_mas=False,
            use_duration_discriminator=use_duration_discriminator,
            mas_noise_scale_initial=0.01,
            noise_scale_delta=0.002,
        ),
    )


def _training_batch() -> dict[str, torch.Tensor | str]:
    torch.manual_seed(17)
    return {
        "input_ids": torch.tensor([[1, 2, 1], [2, 1, 0]]),
        "input_lengths": torch.tensor([3, 2]),
        "tone_ids": torch.tensor([[0, 1, 0], [1, 0, 0]]),
        "language_ids": torch.tensor([[0, 0, 0], [1, 1, 0]]),
        "bert_features": torch.randn(2, 1024, 3),
        "ja_bert_features": torch.randn(2, 768, 3),
        "spectrogram": torch.rand(2, 9, 5),
        "spectrogram_lengths": torch.tensor([5, 4]),
        "audio_values": torch.rand(2, 1, 20) * 2.0 - 1.0,
        "audio_lengths": torch.tensor([20, 19]),
        "speaker_ids": torch.tensor([0, 1]),
        "phase": "generator",
    }


class _TinyWaveformDiscriminator(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, real, generated):
        real_feature = real * self.scale
        generated_feature = generated * self.scale
        return (
            [real_feature.mean(dim=-1)],
            [generated_feature.mean(dim=-1)],
            [[real_feature]],
            [[generated_feature]],
        )


class _TinyDurationDiscriminator(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, _hidden, mask, real, generated):
        return [
            torch.sigmoid(real.transpose(1, 2) * mask.transpose(1, 2) * self.scale),
            torch.sigmoid(generated.transpose(1, 2) * mask.transpose(1, 2) * self.scale),
        ]


class NativeMeloTTSTests(unittest.TestCase):

    def test_text_encoder_requires_explicit_native_language_and_tone_counts(self):
        arguments = {
            "n_vocab": 4,
            "out_channels": 4,
            "hidden_channels": 4,
            "filter_channels": 8,
            "n_heads": 1,
            "n_layers": 1,
            "kernel_size": 3,
            "p_dropout": 0.0,
        }
        for field_name in ("num_languages", "num_tones"):
            values = {
                "num_languages": 2,
                "num_tones": 3,
            }
            values[field_name] = None
            with self.subTest(field=field_name), self.assertRaisesRegex(
                    ValueError,
                    f"`{field_name}` must be an explicit positive integer",
            ):
                TextEncoder(**arguments, **values)

        encoder = TextEncoder(
            **arguments,
            num_languages=2,
            num_tones=3,
        )
        self.assertEqual(encoder.language_emb.num_embeddings, 2)
        self.assertEqual(encoder.tone_emb.num_embeddings, 3)

    def setUp(self) -> None:
        warnings.filterwarnings(
            "ignore",
            message="`torch.nn.utils.weight_norm` is deprecated",
            category=FutureWarning,
        )

    def test_source_and_release_pins_are_explicit(self):
        self.assertEqual(
            MELOTTS_SOURCE_REVISION,
            "209145371cff8fc3bd60d7be902ea69cbdb7965a",
        )
        self.assertEqual(
            set(MELOTTS_RELEASES),
            {"EN", "EN_V2", "EN_NEWEST", "FR", "JP", "ES", "ZH", "KR"},
        )
        for repository, revision, config_sha256, checkpoint_sha256 in (MELOTTS_RELEASES.values()):
            self.assertTrue(repository.startswith("myshell-ai/MeloTTS-"))
            self.assertEqual(len(revision), 40)
            self.assertEqual(len(config_sha256), 64)
            self.assertEqual(len(checkpoint_sha256), 64)
        self.assertEqual(MELOTTS_EN_NEWEST_TENSOR_COUNT, 1_051)
        self.assertEqual(MELOTTS_EN_NEWEST_PARAMETER_COUNT, 51_808_433)
        self.assertEqual(
            MELOTTS_EN_NEWEST_INVENTORY_FINGERPRINT,
            "c505248490ac8de6668aa818388cfa5ca4bcf2ce75a7aacfa9a35dfe6b15816d",
        )
        self.assertEqual(
            MELOTTS_GENERATOR_COMPONENTS,
            DEPLOYABLE_MELOTTS_COMPONENTS,
        )

    def test_architecture_spec_declares_the_feature_boundary(self):
        spec = create_melotts_architecture_spec()

        self.assertEqual(spec.architecture_id, "melotts")
        self.assertTrue(spec.capabilities.training)
        self.assertEqual(
            spec.capabilities.checkpoint_formats,
            ("safetensors", ),
        )
        self.assertIn(
            "explicit-bert-feature-input",
            spec.capabilities.features,
        )
        self.assertEqual(
            spec.metadata["raw_text_frontend"],
            "unsupported-without-exact-upstream-features",
        )

    def test_shared_registry_and_trainer_use_the_native_recipe(self):
        model_spec = get_model_spec("melotts")
        training_spec = get_training_spec("melotts")
        required_inputs = (
            "input_ids",
            "input_lengths",
            "tone_ids",
            "language_ids",
            "bert_features",
            "ja_bert_features",
            "spectrogram",
            "spectrogram_lengths",
            "audio_values",
            "audio_lengths",
            "speaker_ids",
        )

        self.assertEqual(model_spec.architecture, "melotts")
        self.assertTrue(model_spec.is_voicehub_native)
        self.assertIn("preprocessed-training", model_spec.capabilities)
        self.assertEqual(training_spec.support, TrainingSupport.PREPROCESSED)
        self.assertTrue(training_spec.native_training)
        self.assertEqual(
            tuple(phase.name for phase in training_spec.phases),
            ("generator", "discriminator", "duration_discriminator"),
        )
        for phase in training_spec.phases:
            self.assertEqual(phase.forward_component, "training_model")
            self.assertEqual(phase.loss_keys, ("loss", ))
            self.assertEqual(phase.required_inputs, required_inputs)
            self.assertIsNone(phase.fallback_objective)
            self.assertEqual(phase.detach_inputs, ())

        wrapper = MeloTTSForTextToSpeech(MeloTTSConfig(enable_native_finetuning=True), )
        self.assertIsInstance(
            wrapper.get_training_adapter(),
            MeloTTSTrainingAdapter,
        )

    def test_shared_adapter_resolves_every_optimizer_component(self):
        config = _tiny_config(use_duration_discriminator=True)
        with tempfile.TemporaryDirectory() as directory:
            artifact = save_melotts_pretrained(
                build_melotts_model(config),
                config,
                directory,
            )
            wrapper = MeloTTSForTextToSpeech(
                model_path=artifact,
                device="cpu",
                enable_native_finetuning=True,
                training_enable_discriminators=True,
            )
            adapter = wrapper.get_training_adapter().setup()
            groups = adapter.named_parameter_groups()

        self.assertIs(adapter.primary_model, wrapper.training_model)
        self.assertEqual(
            tuple(name for name, _parameters in groups),
            ("generator", "discriminator", "duration_discriminator"),
        )
        self.assertTrue(all(parameters for _name, parameters in groups))
        self.assertEqual(
            tuple(phase.name for phase in adapter.plan_training_phases(0)),
            ("generator", "discriminator", "duration_discriminator"),
        )

    def test_active_graph_uses_only_stdlib_torch_and_voicehub(self):
        root = Path(__file__).parents[1]
        files = tuple((root / "voicehub/models/melotts/native").glob("*.py")) + (
            root / "voicehub/models/melotts/configuration.py",
            root / "voicehub/models/melotts/modeling.py",
            root / "voicehub/models/melotts/modeling.py",
            root / "voicehub/models/melotts/training.py",
            root / "voicehub/models/melotts/source/melo/models.py",
            root / "voicehub/models/melotts/source/melo/modules.py",
            root / "voicehub/models/melotts/source/melo/attentions.py",
            root / "voicehub/models/melotts/source/melo/commons.py",
            root / "voicehub/models/melotts/source/melo/transforms.py",
            root / "voicehub/models/melotts/source/melo/monotonic_align/__init__.py",
            root / "voicehub/models/melotts/source/melo/monotonic_align/core.py",
        )
        forbidden = {
            "transformers",
            "torchaudio",
            "numpy",
            "librosa",
            "phonemizer",
            "numba",
            "scipy",
            "soundfile",
            "melo",
        }
        violations = []
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = {alias.name.split(".", 1)[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom):
                    roots = {(node.module or "").split(".", 1)[0]}
                else:
                    continue
                blocked = sorted(roots & forbidden)
                if blocked:
                    violations.append((path.name, node.lineno, blocked))
        self.assertEqual(violations, [])

    def test_config_round_trips_upstream_and_native_layouts(self):
        config = _tiny_config()
        upstream = {
            "train": {
                "segment_size": config.segment_size
            },
            "data": {
                "sampling_rate": config.data.sample_rate,
                "filter_length": config.data.n_fft,
                "hop_length": config.data.hop_length,
                "win_length": config.data.win_length,
                "n_mel_channels": config.data.n_mels,
                "n_speakers": config.data.n_speakers,
                "spk2id": config.data.speakers,
                "add_blank": True,
            },
            "model": {
                **{
                    name: getattr(config.model, name)
                    for name in config.model.__slots__
                },
                "n_layers_q": 3,
                "use_mel_posterior_encoder": False,
            },
            "symbols": list(config.symbols),
            "num_tones": config.num_tones,
            "num_languages": config.num_languages,
        }

        parsed = MeloTTSArchitectureConfig.from_dict(upstream)
        reloaded = MeloTTSArchitectureConfig.from_dict(parsed.to_dict())

        self.assertEqual(parsed, config)
        self.assertEqual(reloaded, config)
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            MeloTTSDataConfig(
                n_speakers=2,
                speaker_ids=(("duplicate", 0), ("duplicate", 1)),
            )

    def test_frontend_requires_all_checkpoint_compatible_features(self):
        frontend = NativeMeloTTSFrontend(_tiny_config())
        with self.assertRaisesRegex(ValueError, "bert_features"):
            frontend.prepare(
                input_ids=[1, 2],
                tone_ids=[0, 1],
                language_ids=[0, 0],
                bert_features=torch.zeros(1024, 1),
                ja_bert_features=torch.zeros(768, 2),
                speaker="speaker-a",
                device="cpu",
                dtype=torch.float32,
            )
        model = MeloTTSForTextToSpeech()
        with self.assertRaisesRegex(ValueError, "precomputed"):
            model.generate("raw text")
        self.assertFalse(model.is_loaded)

    def test_model_inventory_matches_the_released_namespace(self):
        model = build_melotts_model(_tiny_config())
        prefixes = {name.split(".", 1)[0] for name in model.state_dict()}

        self.assertEqual(prefixes, set(DEPLOYABLE_MELOTTS_COMPONENTS))
        self.assertEqual(len(model.state_dict()), 574)

    def test_legacy_import_is_trust_gated_and_strict(self):
        model = build_melotts_model(_tiny_config())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            torch.save({"model": model.state_dict()}, checkpoint)
            with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint"):
                read_legacy_melotts_checkpoint(
                    model,
                    checkpoint,
                    trust_pickle_checkpoint=False,
                )
            state = read_legacy_melotts_checkpoint(
                model,
                checkpoint,
                trust_pickle_checkpoint=True,
            )

        self.assertEqual(tuple(state), tuple(sorted(model.state_dict())))

    def test_safetensors_export_reloads_fresh_inference_identity(self):
        config = _tiny_config()
        original = build_melotts_model(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            artifact = save_melotts_pretrained(
                original,
                config,
                directory,
            )
            report = inspect_melotts_checkpoint(artifact / "model.safetensors")
            runtime = MeloTTSRuntime(artifact, device="cpu")
            self.assertIsNotNone(runtime._weight_norm_cache)
            self.assertGreater(runtime._weight_norm_cache.module_count, 0)
            runtime.train()
            self.assertIsNone(runtime._weight_norm_cache)
            runtime.eval()
            self.assertIsNotNone(runtime._weight_norm_cache)
            kwargs = {
                "input_ids": [1, 2, 1],
                "tone_ids": [0, 1, 0],
                "language_ids": [0, 0, 0],
                "bert_features": torch.zeros(1024, 3),
                "ja_bert_features": torch.zeros(768, 3),
                "speaker": "speaker-a",
                "noise_scale": 0.0,
                "noise_scale_w": 0.0,
                "sdp_ratio": 0.0,
                "max_frames": 20,
            }
            torch.manual_seed(91)
            direct = original.infer(
                torch.tensor([[1, 2, 1]]),
                torch.tensor([3]),
                torch.tensor([0]),
                torch.tensor([[0, 1, 0]]),
                torch.tensor([[0, 0, 0]]),
                torch.zeros(1, 1024, 3),
                torch.zeros(1, 768, 3),
                noise_scale=0.0,
                noise_scale_w=0.0,
                sdp_ratio=0.0,
                max_len=20,
            )[0][0, 0]
            torch.manual_seed(91)
            reloaded = runtime.generate(**kwargs)

        self.assertEqual(report.tensor_count, len(original.state_dict()))
        self.assertTrue(torch.equal(direct.float(), reloaded))

    def test_training_collator_pads_every_sequence_axis(self):
        collator = MeloTTSTrainingCollator()
        batch = collator([
            {
                "input_ids": [1, 2, 1],
                "tone_ids": [0, 1, 0],
                "language_ids": [0, 0, 0],
                "bert_features": torch.zeros(1024, 3),
                "ja_bert_features": torch.zeros(768, 3),
                "spectrogram": torch.zeros(9, 5),
                "audio_values": torch.zeros(20),
                "speaker_id": 0,
            },
            {
                "input_ids": [2, 1],
                "tone_ids": [1, 0],
                "language_ids": [1, 1],
                "bert_features": torch.zeros(1024, 2),
                "ja_bert_features": torch.zeros(768, 2),
                "spectrogram": torch.zeros(9, 4),
                "audio_values": torch.zeros(19),
                "speaker_id": 1,
            },
        ])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
        self.assertEqual(tuple(batch["bert_features"].shape), (2, 1024, 3))
        self.assertEqual(tuple(batch["spectrogram"].shape), (2, 9, 5))
        self.assertEqual(tuple(batch["audio_values"].shape), (2, 1, 20))
        self.assertEqual(batch["input_lengths"].tolist(), [3, 2])
        self.assertEqual(batch["spectrogram_lengths"].tolist(), [5, 4])
        self.assertEqual(batch["audio_lengths"].tolist(), [20, 19])

    def test_preprocessed_objective_backpropagates_all_generator_families(self):
        config = _tiny_config()
        objective = MeloTTSTrainingModel(
            build_melotts_model(config),
            config,
            enable_discriminators=False,
        )
        batch = _training_batch()
        torch.manual_seed(23)

        output = objective(**batch)
        output["loss"].backward()

        self.assertTrue(bool(torch.isfinite(output["loss"])))
        for prefix in DEPLOYABLE_MELOTTS_COMPONENTS:
            gradients = [
                parameter.grad for name, parameter in objective.model.named_parameters()
                if name.startswith(prefix + ".")
            ]
            self.assertTrue(
                any(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients),
                prefix,
            )

    def test_variable_length_padding_does_not_change_objective(self):
        config = _tiny_config()
        objective = MeloTTSTrainingModel(
            build_melotts_model(config),
            config,
            enable_discriminators=False,
        ).eval()
        original = _training_batch()
        changed = {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in original.items()
        }
        changed["input_ids"][1, 2] = 3
        changed["tone_ids"][1, 2] = 2
        changed["language_ids"][1, 2] = 1
        changed["bert_features"][1, :, 2] = 99.0
        changed["ja_bert_features"][1, :, 2] = -99.0
        changed["spectrogram"][1, :, 4] = 88.0
        changed["audio_values"][1, 0, 19] = 77.0

        torch.manual_seed(101)
        first = objective(**original)
        torch.manual_seed(101)
        second = objective(**changed)

        for name in ("loss", "mel_loss", "kl_loss", "duration_loss"):
            self.assertTrue(
                torch.allclose(first[name], second[name], atol=1e-6, rtol=1e-6),
                name,
            )

    def test_every_training_phase_routes_gradients_to_its_component(self):
        config = _tiny_config(use_duration_discriminator=True)
        objective = MeloTTSTrainingModel(
            build_melotts_model(config),
            config,
            mpd=_TinyWaveformDiscriminator(),
            duration_discriminator=_TinyDurationDiscriminator(),
        )
        phases = (
            ("discriminator", objective.mpd),
            ("duration_discriminator", objective.duration_discriminator),
            ("generator", objective.model),
        )
        for phase, component in phases:
            with self.subTest(phase=phase):
                objective.zero_grad(set_to_none=True)
                batch = _training_batch()
                batch["phase"] = phase
                torch.manual_seed(211)

                output = objective(**batch)
                output["loss"].backward()

                gradients = [
                    parameter.grad for parameter in component.parameters() if parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))

    def test_alignment_noise_uses_the_released_step_schedule(self):
        config = _tiny_config()
        objective = MeloTTSTrainingModel(
            build_melotts_model(config),
            config,
            enable_discriminators=False,
        )

        self.assertAlmostEqual(objective.set_step(3), 0.004)
        self.assertAlmostEqual(
            objective.model.current_mas_noise_scale,
            0.004,
        )
        self.assertEqual(objective.set_step(50), 0.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            objective.set_step(-1)

    def test_released_adversarial_formulas_are_exact(self):
        real = [torch.tensor([[0.5, 1.0]])]
        generated = [torch.tensor([[0.25, -0.5]])]
        real_features = [[torch.tensor([1.0, 3.0])]]
        generated_features = [[torch.tensor([2.0, 1.0])]]

        self.assertTrue(
            torch.allclose(
                discriminator_loss(real, generated),
                ((1 - real[0]).square().mean() + generated[0].square().mean()),
            ))
        self.assertTrue(torch.allclose(
            generator_loss(generated),
            (1 - generated[0]).square().mean(),
        ))
        self.assertTrue(
            torch.allclose(
                feature_matching_loss(real_features, generated_features),
                torch.tensor(3.0),
            ))

        duration_real = torch.tensor([
            [[0.5], [1.0]],
            [[0.25], [0.75]],
        ])
        duration_generated = torch.tensor([
            [[0.25], [0.5]],
            [[0.0], [1.0]],
        ])
        expected_duration_discriminator = sum(
            (1.0 - real_item).square().mean() + generated_item.square().mean()
            for real_item, generated_item in zip(
                duration_real,
                duration_generated,
            ))
        expected_duration_generator = sum(
            (1.0 - generated_item).square().mean() for generated_item in duration_generated)
        self.assertTrue(
            torch.allclose(
                discriminator_loss(
                    duration_real,
                    duration_generated,
                ),
                expected_duration_discriminator,
            ))
        self.assertTrue(torch.allclose(
            generator_loss(duration_generated),
            expected_duration_generator,
        ))


if __name__ == "__main__":
    unittest.main()
