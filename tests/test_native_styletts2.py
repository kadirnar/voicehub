from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from voicehub.models.kokoro.native.configuration import KokoroAlbertConfig
from voicehub.models.styletts2.modeling import StyleTTS2ForTextToSpeech
from voicehub.models.styletts2.native.checkpoint import (
    export_styletts2_checkpoint,
    load_styletts2_checkpoint,
    read_legacy_styletts2_checkpoint,
)
from voicehub.models.styletts2.native.configuration import StyleTTS2ArchitectureConfig, load_styletts2_config
from voicehub.models.styletts2.native.frontend import (
    STYLETTS2_SYMBOLS,
    NativeStyleTTS2Frontend,
    StyleTTS2MelSpectrogram,
    trim_reference_silence,
)
from voicehub.models.styletts2.native.modeling import DEPLOYABLE_STYLETTS2_COMPONENTS
from voicehub.models.styletts2.native.registration import create_styletts2_architecture_spec
from voicehub.models.styletts2.native.training import StyleTTS2LossWeights, StyleTTS2TrainingModel
from voicehub.models.styletts2.source.styletts2.models import StyleTTS2Modules
from voicehub.models.styletts2.training import StyleTTS2TrainingAdapter, StyleTTS2TrainingCollator
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import get_training_spec

ROOT = Path(__file__).parents[1]
ACTIVE_FILES = (
    *(ROOT / 'voicehub/models/styletts2/native').glob("*.py"),
    ROOT / 'voicehub/models/styletts2/modeling.py',
    ROOT / "voicehub" / "models" / "styletts2" / "runtime.py",
    ROOT / "voicehub" / "models" / "styletts2" / "training.py",
    ROOT / "voicehub" / "models" / "styletts2" / "source" / "styletts2" / "models.py",
    ROOT / "voicehub" / "models" / "styletts2" / "source" / "styletts2" / "Modules" / "hifigan.py",
    ROOT / "voicehub" / "models" / "styletts2" / "source" / "styletts2" / "Modules" / "istftnet.py",
    ROOT / "voicehub" / "models" / "styletts2" / "source" / "styletts2" / "Modules" / "discriminators.py",
    ROOT / "voicehub" / "models" / "styletts2" / "source" / "styletts2" / "Modules" / "utils.py",
    *(ROOT / "voicehub" / "models" / "styletts2" / "source" / "styletts2" / "Modules" /
      "diffusion").glob("*.py"),
)


def _tiny_config() -> StyleTTS2ArchitectureConfig:
    return StyleTTS2ArchitectureConfig(
        sample_rate=8_000,
        n_fft=16,
        win_length=8,
        hop_length=4,
        dim_in=2,
        hidden_dim=4,
        max_conv_dim=4,
        n_layer=1,
        n_mels=4,
        n_token=8,
        max_dur=3,
        style_dim=2,
        dropout=0.0,
        plbert=KokoroAlbertConfig(
            vocab_size=8,
            embedding_size=4,
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=8,
            max_position_embeddings=16,
        ),
    )


class _FakeTextEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)

    def forward(self, input_ids, lengths, mask):
        del lengths
        value = self.embedding(input_ids).transpose(1, 2)
        return value.masked_fill(mask.unsqueeze(1), 0.0)


class _FakeBert(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)

    def forward(self, input_ids, attention_mask):
        return self.embedding(input_ids) * attention_mask.unsqueeze(-1)


class _FakeStyleEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 2)

    def forward(self, mel):
        return self.projection(mel.squeeze(1).mean(dim=-1))


class _FakePredictor(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.duration = nn.Linear(4, 3)
        self.f0 = nn.Conv1d(4, 1, 1)
        self.noise = nn.Conv1d(4, 1, 1)

    def forward(self, encoded, style, lengths, alignment, mask):
        del style, lengths, mask
        logits = self.duration(encoded.transpose(1, 2))
        return logits, encoded @ alignment

    def F0Ntrain(self, encoded, style):
        del style
        encoded = functional.interpolate(encoded, scale_factor=2)
        return self.f0(encoded).squeeze(1), self.noise(encoded).squeeze(1)


class _FakeDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, encoded, f0, noise, style):
        value = encoded.mean(dim=1, keepdim=True)
        value = functional.interpolate(value, scale_factor=8)
        prosody = functional.interpolate(
            (f0 + noise).unsqueeze(1),
            size=value.shape[-1],
        )
        return self.scale * (value + prosody + style.mean().reshape(1, 1, 1))


class _FakeDiffusion(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, target, *, embedding, features):
        return (
            target.square().mean() + self.scale * embedding.square().mean() + 0.0 * features.square().mean())


def _fake_training_graph() -> nn.Module:
    graph = nn.Module()
    graph.text_encoder = _FakeTextEncoder()
    graph.bert = _FakeBert()
    graph.bert_encoder = nn.Linear(4, 4)
    graph.style_encoder = _FakeStyleEncoder()
    graph.predictor_encoder = _FakeStyleEncoder()
    graph.predictor = _FakePredictor()
    graph.decoder = _FakeDecoder()
    graph.diffusion = _FakeDiffusion()
    return graph


class NativeStyleTTS2Tests(unittest.TestCase):

    def test_configuration_import_keeps_styletts2_graph_lazy(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import voicehub.models.styletts2."
                    "configuration; "
                    "print(*(int(name in sys.modules) for name in ("
                    "'voicehub.models.styletts2.modeling', "
                    "'voicehub.models.styletts2.native.modeling', "
                    "'voicehub.models.styletts2.native.training')))"),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "0 0 0")

    def test_active_runtime_uses_only_stdlib_torch_and_voicehub(self):
        forbidden = set()
        for source in ACTIVE_FILES:
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = [alias.name.split(".", 1)[0] for alias in node.names]
                elif (isinstance(node, ast.ImportFrom) and node.module and node.level == 0):
                    roots = [node.module.split(".", 1)[0]]
                else:
                    continue
                forbidden.update(
                    root for root in roots
                    if root not in sys.stdlib_module_names and root not in {"torch", "voicehub"})
        self.assertEqual(forbidden, set())

    def test_pinned_config_is_typed_and_round_trips(self):
        config_path = (
            ROOT / "voicehub" / "models" / "styletts2" / "source" / "styletts2" / "Configs" /
            "config_libritts.yml")
        config = load_styletts2_config(config_path)
        self.assertEqual(
            StyleTTS2ArchitectureConfig.from_dict(config.to_dict()),
            config,
        )
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "config.yml"
            custom.write_text("model_params: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unpinned YAML"):
                load_styletts2_config(custom)

    def test_pinned_single_speaker_istft_profile_is_supported(self):
        config_path = (
            ROOT / "voicehub" / "models" / "styletts2" / "source" / "styletts2" / "Configs" / "config.yml")
        config = load_styletts2_config(config_path)
        self.assertFalse(config.multispeaker)
        self.assertEqual(config.decoder.type, "istftnet")
        self.assertEqual(config.decoder.gen_istft_n_fft, 20)
        self.assertEqual(
            StyleTTS2ArchitectureConfig.from_dict(config.to_dict()),
            config,
        )

    def test_pinned_yaml_digest_is_independent_of_checkout_newlines(self):
        config_path = (
            ROOT / "voicehub" / "models" / "styletts2" / "source" / "styletts2" / "Configs" /
            "config_libritts.yml")
        with tempfile.TemporaryDirectory() as directory:
            windows_copy = Path(directory) / "config.yml"
            source = config_path.read_bytes().replace(b"\r\n", b"\n")
            windows_copy.write_bytes(source.replace(b"\n", b"\r\n"))
            config = load_styletts2_config(windows_copy)
        self.assertTrue(config.multispeaker)
        self.assertEqual(config.decoder.type, "hifigan")

    def test_frontend_requires_explicit_phonemes(self):
        frontend = NativeStyleTTS2Frontend()
        self.assertEqual(len(STYLETTS2_SYMBOLS), 178)
        with self.assertRaisesRegex(ValueError, "raw-text"):
            frontend.encode_phonemes("hello", explicit=False)
        token_ids = frontend.encode_phonemes("həloʊ", explicit=True)
        self.assertEqual(token_ids.shape, (1, 6))
        self.assertEqual(int(token_ids[0, 0]), 0)
        with self.assertRaisesRegex(ValueError, "outside"):
            frontend.encode_phonemes("🙂", explicit=True)

    def test_public_wrapper_rejects_implicit_g2p_before_loading(self):
        model = StyleTTS2ForTextToSpeech()
        with self.assertRaisesRegex(ValueError, "explicit phonemes"):
            model.generate("raw English")
        self.assertFalse(model.is_loaded)

    def test_native_mel_and_trim_are_torch_differentiable(self):
        waveform = torch.cat([torch.zeros(64),
                              torch.linspace(-0.5, 0.5, 256),
                              torch.zeros(64)]).requires_grad_()
        trimmed = trim_reference_silence(waveform)
        self.assertGreater(trimmed.numel(), 0)
        self.assertLessEqual(trimmed.numel(), waveform.numel())
        mel = StyleTTS2MelSpectrogram(
            sample_rate=8_000,
            n_fft=32,
            win_length=24,
            hop_length=8,
            n_mels=8,
        )(trimmed)
        self.assertEqual(mel.shape[-2], 8)
        mel.mean().backward()
        self.assertIsNotNone(waveform.grad)

    def test_safe_checkpoint_exports_and_reloads_fresh_model(self):

        def model() -> StyleTTS2Modules:
            return StyleTTS2Modules(**{name: nn.Linear(2, 2) for name in DEPLOYABLE_STYLETTS2_COMPONENTS})

        original = model()
        fresh = model()
        with torch.no_grad():
            for parameter in fresh.parameters():
                parameter.zero_()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            export_styletts2_checkpoint(original, path)
            report = load_styletts2_checkpoint(fresh, path)
        self.assertGreater(report.tensor_count, 0)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, fresh.state_dict()[name])

    def test_legacy_import_is_explicit_and_strict(self):
        model = StyleTTS2Modules(**{name: nn.Linear(2, 2) for name in DEPLOYABLE_STYLETTS2_COMPONENTS})
        payload = {"net": {name: component.state_dict() for name, component in model.items()}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint"):
                read_legacy_styletts2_checkpoint(
                    model,
                    path,
                    trust_pickle_checkpoint=False,
                )
            state = read_legacy_styletts2_checkpoint(
                model,
                path,
                trust_pickle_checkpoint=True,
            )
        self.assertEqual(set(state), set(model.state_dict()))

    def test_preprocessed_objective_backpropagates(self):
        graph = _fake_training_graph()
        objective = StyleTTS2TrainingModel(
            graph,
            _tiny_config(),
            enable_discriminators=False,
            loss_weights=StyleTTS2LossWeights(
                adversarial=0.0,
                feature_matching=0.0,
            ),
        )
        alignments = torch.zeros(1, 3, 4)
        alignments[0, 0, :1] = 1
        alignments[0, 1, 1:3] = 1
        alignments[0, 2, 3:] = 1
        output = objective(
            torch.tensor([[0, 2, 3]]),
            input_lengths=torch.tensor([3]),
            alignments=alignments,
            alignment_lengths=torch.tensor([4]),
            normalized_mel=torch.randn(1, 1, 4, 16),
            normalized_mel_lengths=torch.tensor([8]),
            reference_mel=torch.randn(1, 1, 4, 16),
            reference_mel_lengths=torch.tensor([16]),
            f0_targets=torch.randn(1, 8),
            noise_targets=torch.randn(1, 8),
            audio_values=torch.randn(1, 1, 32),
            audio_lengths=torch.tensor([32]),
        )
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in graph.parameters()))
        self.assertIn("diffusion_loss", output)
        self.assertIn("duration_ce_loss", output)

    def test_preprocessed_objective_ignores_variable_length_padding(self):
        objective = StyleTTS2TrainingModel(
            _fake_training_graph(),
            _tiny_config(),
            enable_discriminators=False,
            loss_weights=StyleTTS2LossWeights(
                adversarial=0.0,
                feature_matching=0.0,
                relativistic=0.0,
            ),
        )
        alignments = torch.zeros(2, 4, 4)
        alignments[0, :3, :3] = torch.eye(3)
        alignments[1] = torch.eye(4)
        values = {
            "input_ids": torch.tensor([
                [0, 2, 3, 0],
                [0, 4, 5, 6],
            ]),
            "input_lengths": torch.tensor([3, 4]),
            "alignments": alignments,
            "alignment_lengths": torch.tensor([3, 4]),
            "normalized_mel": torch.randn(2, 1, 4, 8),
            "normalized_mel_lengths": torch.tensor([6, 8]),
            "reference_mel": torch.randn(2, 1, 4, 8),
            "reference_mel_lengths": torch.tensor([7, 8]),
            "f0_targets": torch.randn(2, 8),
            "noise_targets": torch.randn(2, 8),
            "audio_values": torch.randn(2, 1, 32),
            "audio_lengths": torch.tensor([24, 32]),
        }
        baseline = objective(**values)
        padded = {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in values.items()
        }
        padded["input_ids"][0, 3] = 7
        padded["normalized_mel"][0, :, :, 6:] = 50.0
        padded["reference_mel"][0, :, :, 7:] = 50.0
        padded["f0_targets"][0, 6:] = 50.0
        padded["noise_targets"][0, 6:] = 50.0
        padded["audio_values"][0, :, 24:] = 50.0

        changed = objective(**padded)

        self.assertEqual(
            baseline["waveform_lengths"].tolist(),
            [24, 32],
        )
        for name in (
                "loss",
                "mel_loss",
                "f0_loss",
                "noise_loss",
                "duration_loss",
                "duration_ce_loss",
                "diffusion_loss",
                "waveform_loss",
        ):
            torch.testing.assert_close(baseline[name], changed[name])

    def test_training_collator_pads_both_alignment_axes(self):
        collator = StyleTTS2TrainingCollator()
        batch = collator([
            {
                "input_ids": torch.tensor([0, 2, 3]),
                "alignments": torch.eye(3),
                "normalized_mel": torch.ones(4, 6),
                "reference_mel": torch.ones(4, 5),
                "f0_targets": torch.ones(6),
                "noise_targets": torch.ones(6),
                "audio_values": torch.ones(24),
            },
            {
                "input_ids": torch.tensor([0, 3, 4, 5]),
                "alignments": torch.eye(4),
                "normalized_mel": torch.ones(4, 8),
                "reference_mel": torch.ones(4, 6),
                "f0_targets": torch.ones(8),
                "noise_targets": torch.ones(8),
                "audio_values": torch.ones(32),
            },
        ])
        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["input_lengths"].tolist(), [3, 4])
        self.assertEqual(batch["alignments"].shape, (2, 4, 4))
        self.assertEqual(batch["alignment_lengths"].tolist(), [3, 4])
        self.assertEqual(batch["normalized_mel"].shape, (2, 1, 4, 8))
        self.assertEqual(
            batch["normalized_mel_lengths"].tolist(),
            [6, 8],
        )
        self.assertEqual(batch["reference_mel"].shape, (2, 1, 4, 6))
        self.assertEqual(
            batch["reference_mel_lengths"].tolist(),
            [5, 6],
        )
        self.assertEqual(batch["f0_targets"].shape, (2, 8))
        self.assertEqual(batch["audio_values"].shape, (2, 1, 32))
        self.assertEqual(batch["audio_lengths"].tolist(), [24, 32])

    def test_architecture_spec_is_truthful(self):
        spec = create_styletts2_architecture_spec()
        from voicehub.registry import get_model_spec

        model_spec = get_model_spec("styletts2")
        training_spec = get_training_spec("styletts2")
        self.assertEqual(spec.architecture_id, "styletts2")
        self.assertTrue(spec.capabilities.training)
        self.assertEqual(
            spec.capabilities.checkpoint_formats,
            ("safetensors", ),
        )
        self.assertIn(
            "preprocessed-teacher-forced-finetuning",
            spec.capabilities.features,
        )
        self.assertIn("native-istftnet", spec.capabilities.features)
        self.assertTrue(model_spec.is_voicehub_native)
        self.assertEqual(model_spec.architecture, "styletts2")
        self.assertTrue(training_spec.native_training)
        self.assertIs(training_spec.support, TrainingSupport.PREPROCESSED)
        self.assertEqual(
            tuple(phase.name for phase in training_spec.phases),
            ("generator", "discriminator"),
        )

    def test_disabled_discriminators_are_not_scheduled(self):
        wrapper = StyleTTS2ForTextToSpeech(
            enable_native_finetuning=True,
            training_enable_discriminators=False,
        )
        adapter = StyleTTS2TrainingAdapter(
            wrapper,
            get_training_spec("styletts2"),
        )

        self.assertEqual(
            tuple(phase.name for phase in adapter.plan_training_phases(0)),
            ("generator", ),
        )
        with self.assertRaisesRegex(
                ValueError,
                "discriminator training is disabled",
        ):
            adapter.select_training_phase("discriminator")


if __name__ == "__main__":
    unittest.main()
