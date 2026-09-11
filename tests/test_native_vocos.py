from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.components.audio.vocoders.vocos.configuration import parse_vocos_yaml
from voicehub.components.audio.vocoders.vocos.dataset import DataConfig, VocosDataset
from voicehub.components.audio.vocoders.vocos.discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
)
from voicehub.components.audio.vocoders.vocos.experiment import VocosExp
from voicehub.components.audio.vocoders.vocos.feature_extractors import EncodecFeatures, MelSpectrogramFeatures
from voicehub.components.audio.vocoders.vocos.heads import IMDCTCosHead, IMDCTSymExpHead, ISTFTHead
from voicehub.components.audio.vocoders.vocos.models import VocosBackbone
from voicehub.components.audio.vocoders.vocos.modules import AdaLayerNorm
from voicehub.components.audio.vocoders.vocos.pretrained import Vocos
from voicehub.components.audio.vocoders.vocos.spectral_ops import IMDCT, MDCT
from voicehub.models.f5tts.native.audio import htk_mel_filter_bank as f5_htk_bank
from voicehub.policies.architecture_dependencies import inspect_native_imports
from voicehub.processing import htk_mel_filter_bank, save_pcm_wave

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOCOS_ROOT = PROJECT_ROOT / "voicehub/components/audio/vocoders/vocos"

_TINY_CONFIG = """\
feature_extractor:
  class_path: vocos.feature_extractors.MelSpectrogramFeatures
  init_args:
    sample_rate: 24000
    n_fft: 64
    hop_length: 16
    n_mels: 8
    padding: center
backbone:
  class_path: vocos.models.VocosBackbone
  init_args:
    input_channels: 8
    dim: 16
    intermediate_dim: 32
    num_layers: 2
head:
  class_path: vocos.heads.ISTFTHead
  init_args:
    dim: 16
    n_fft: 64
    hop_length: 16
    padding: center
"""


class NativeVocosTests(unittest.TestCase):

    @staticmethod
    def _tiny_encodec():
        from voicehub.components.audio.codecs.encodec import EncodecConfig, EncodecModel

        return EncodecModel.from_config(
            EncodecConfig(
                target_bandwidths=(1.5, 3.0, 6.0, 12.0),
                sample_rate=24_000,
                channels=1,
                causal=True,
                model_norm="weight_norm",
                dimension=8,
                n_filters=4,
                n_residual_layers=1,
                ratios=(2, 2),
                kernel_size=3,
                last_kernel_size=3,
                residual_kernel_size=3,
                dilation_base=2,
                lstm=0,
                bins=8,
                n_q=4,
                kmeans_init=False,
                kmeans_iters=2,
                threshold_ema_dead_code=0,
            )).eval()

    def test_entire_runtime_has_no_external_imports(self):
        violations = ()
        for path in sorted(VOCOS_ROOT.glob("*.py")):
            violations += inspect_native_imports(path)
        self.assertEqual(
            violations,
            (),
            "\n".join(str(violation) for violation in violations),
        )

    def test_strict_yaml_parser_accepts_released_schema(self):
        config = parse_vocos_yaml(_TINY_CONFIG)

        self.assertEqual(
            config["feature_extractor"]["init_args"]["sample_rate"],
            24_000,
        )
        self.assertEqual(
            config["backbone"]["class_path"],
            "vocos.models.VocosBackbone",
        )

    def test_strict_yaml_parser_rejects_object_tags_and_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError, "forbidden"):
            parse_vocos_yaml("model: !!python/object:dangerous.Type {}\n")
        with self.assertRaisesRegex(ValueError, "repeats"):
            parse_vocos_yaml("model: one\nmodel: two\n")

    def test_shared_htk_filter_is_exactly_the_f5_checkpoint_filter(self):
        expected = f5_htk_bank(
            sample_rate=24_000,
            n_fft=1_024,
            n_mels=100,
        )
        actual = htk_mel_filter_bank(
            sample_rate=24_000,
            n_fft=1_024,
            n_mels=100,
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.shape, (513, 100))

    def test_native_mel_frontend_preserves_released_state_namespace(self):
        frontend = MelSpectrogramFeatures(
            sample_rate=24_000,
            n_fft=64,
            hop_length=16,
            n_mels=8,
        )
        waveform = torch.randn(2, 256, requires_grad=True)

        features = frontend(waveform)
        features.mean().backward()

        self.assertEqual(features.shape, (2, 8, 17))
        self.assertIsNotNone(waveform.grad)
        self.assertEqual(
            set(frontend.state_dict()),
            {
                "mel_spec.spectrogram.window",
                "mel_spec.mel_scale.fb",
            },
        )

    def test_encodec_frontend_fails_closed_until_weights_are_explicit(self):
        codec = self._tiny_encodec()
        frontend = EncodecFeatures(
            bandwidths=(1.5, 3.0, 6.0, 12.0),
            encodec=codec,
        )
        self.assertTrue(frontend.encodec_weights_available)
        frontend._encodec_weights_available = False

        with self.assertRaisesRegex(RuntimeError, "verified native Encodec"):
            frontend.get_encodec_codes(torch.zeros(1, 64))

        frontend.attach_encodec(self._tiny_encodec())
        self.assertTrue(frontend.encodec_weights_available)

    def test_mdct_center_round_trip_and_clipping_heads_return_audio(self):
        torch.manual_seed(0)
        waveform = torch.randn(2, 256)
        coefficients = MDCT(64, padding="center")(waveform)
        restored = IMDCT(64, padding="center")(coefficients)

        torch.testing.assert_close(restored, waveform, rtol=0, atol=2e-5)
        hidden = torch.randn(2, 9, 8)
        for head in (
                IMDCTSymExpHead(8, 64, padding="center", clip_audio=True),
                IMDCTCosHead(8, 64, padding="center", clip_audio=True),
        ):
            with self.subTest(head=head.__class__.__name__):
                audio = head(hidden)
                self.assertEqual(audio.shape, (2, 256))
                self.assertLessEqual(float(audio.abs().max().item()), 1.0)

    def test_adaptive_normalization_broadcasts_per_batch_not_per_frame(self):
        layer = AdaLayerNorm(num_embeddings=3, embedding_dim=4)
        with torch.no_grad():
            layer.scale.weight.copy_(
                torch.tensor((
                    (1.0, 1.0, 1.0, 1.0),
                    (2.0, 2.0, 2.0, 2.0),
                    (3.0, 3.0, 3.0, 3.0),
                )))
        hidden = torch.randn(2, 5, 4)

        output = layer(hidden, torch.tensor((0, 2)))

        self.assertEqual(output.shape, hidden.shape)
        torch.testing.assert_close(
            output[0],
            torch.nn.functional.layer_norm(hidden[0], (4, ), eps=1e-6),
        )

    def test_safetensors_export_loads_into_a_fresh_public_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.yaml").write_text(_TINY_CONFIG, encoding="utf-8")
            source = Vocos.from_hparams(root / "config.yaml").eval()
            checkpoint = source.export_safetensors(root / "model.safetensors")
            restored = Vocos.from_pretrained(root).eval()
            features = torch.randn(1, 8, 12)

            with torch.inference_mode():
                expected = source.decode(features)
                actual = restored.decode(features)

        self.assertEqual(checkpoint.name, "model.safetensors")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unpinned_legacy_checkpoint_requires_explicit_trust(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.yaml").write_text(_TINY_CONFIG, encoding="utf-8")
            source = Vocos.from_hparams(root / "config.yaml")
            torch.save(source.state_dict(), root / "pytorch_model.bin")

            with self.assertRaisesRegex(ValueError, "trust_legacy_checkpoint"):
                Vocos.from_pretrained(root)
            restored = Vocos.from_pretrained(
                root,
                trust_legacy_checkpoint=True,
            )

        self.assertIsInstance(restored, Vocos)

    def test_native_training_objective_backpropagates_and_builds_schedulers(self):
        feature_extractor = MelSpectrogramFeatures(
            sample_rate=24_000,
            n_fft=64,
            hop_length=16,
            n_mels=8,
        )
        backbone = VocosBackbone(
            input_channels=8,
            dim=16,
            intermediate_dim=32,
            num_layers=2,
        )
        head = ISTFTHead(
            dim=16,
            n_fft=64,
            hop_length=16,
            padding="center",
        )
        objective = VocosExp(
            feature_extractor,
            backbone,
            head,
            sample_rate=24_000,
            initial_learning_rate=1e-4,
            pretrain_mel_steps=10,
            multiperiod_discriminator=MultiPeriodDiscriminator(periods=(2, )),
            multiresolution_discriminator=MultiResolutionDiscriminator(fft_sizes=(64, ), ),
        )
        waveform = torch.randn(2, 1_024)

        loss = objective.training_step(
            waveform,
            optimizer_idx=1,
            global_step=0,
            total_steps=100,
        )
        loss.backward()
        optimizers, schedulers = objective.configure_optimizers(total_steps=100)

        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertTrue(any(parameter.grad is not None for parameter in backbone.parameters()))
        self.assertEqual(len(optimizers), 2)
        self.assertEqual(len(schedulers), 2)
        self.assertEqual(objective.last_step.optimizer, "generator")

    def test_pcm_dataset_resolves_relative_paths_and_has_exact_length(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_pcm_wave(
                root / "speaker.wav",
                torch.linspace(-0.5, 0.5, 400),
                16_000,
            )
            (root / "files.txt").write_text("speaker.wav\n", encoding="utf-8")
            config = DataConfig(
                filelist_path=root / "files.txt",
                sampling_rate=24_000,
                num_samples=800,
                batch_size=1,
            )

            validation = VocosDataset(config, train=False)[0]

        self.assertEqual(validation.shape, (800, ))
        self.assertTrue(bool(torch.isfinite(validation).all().item()))


if __name__ == "__main__":
    unittest.main()
