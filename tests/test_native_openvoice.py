from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.checkpointing import save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.openvoice import (
    OpenVoiceConfig,
    OpenVoiceForTextToSpeech,
    OpenVoiceTrainingAdapter,
    OpenVoiceTrainingCollator,
)
from voicehub.models.openvoice.native.artifacts import OpenVoiceArtifacts, resolve_openvoice_artifacts
from voicehub.models.openvoice.native.checkpoint import (
    load_openvoice_checkpoint,
    read_openvoice_checkpoint,
    save_openvoice_checkpoint,
)
from voicehub.models.openvoice.native.configuration import OpenVoiceConverterConfig
from voicehub.models.openvoice.native.metadata import (
    OPENVOICE_CHECKPOINT_REVISION,
    OPENVOICE_CONVERTER_CHECKPOINT,
    OPENVOICE_SOURCE_REVISION,
)
from voicehub.models.openvoice.native.modeling import OpenVoiceToneColorConverter
from voicehub.models.openvoice.native.processing import OpenVoiceAudioProcessor
from voicehub.models.openvoice.native.runtime import OpenVoiceRuntime, load_openvoice_runtime
from voicehub.runtime import get_architecture_spec
from voicehub.training import AutoTrainingAdapter, TrainingSupport


def _tiny_config() -> OpenVoiceConverterConfig:
    return OpenVoiceConverterConfig(
        n_fft=64,
        hop_length=4,
        win_length=16,
        inter_channels=8,
        hidden_channels=8,
        filter_channels=16,
        n_heads=2,
        n_layers=1,
        resblock_kernel_sizes=(3, ),
        resblock_dilation_sizes=((1, 3, 5), ),
        upsample_rates=(2, 2),
        upsample_initial_channel=16,
        upsample_kernel_sizes=(4, 4),
        speaker_embedding_size=8,
    )


def _artifacts(
    root: Path,
    config: OpenVoiceConverterConfig,
    checkpoint: Path,
    *,
    legacy: bool = False,
) -> OpenVoiceArtifacts:
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    return OpenVoiceArtifacts(
        source=str(root),
        revision=None,
        config_path=config_path,
        config=config,
        checkpoint_path=checkpoint,
        legacy_pytorch=legacy,
        expected_checkpoint_sha256=None,
    )


class OpenVoiceArchitectureTests(unittest.TestCase):

    def test_provenance_and_exact_released_inventory_are_immutable(self):
        self.assertEqual(
            OPENVOICE_SOURCE_REVISION,
            "74a1d147b17a8c3092dd5430504bd83ef6c7eb23",
        )
        self.assertEqual(
            OPENVOICE_CHECKPOINT_REVISION,
            "f36e7edfe1684461a8343844af60babc2efbb727",
        )
        self.assertEqual(OPENVOICE_CONVERTER_CHECKPOINT["tensors"], 486)
        self.assertEqual(
            OPENVOICE_CONVERTER_CHECKPOINT["parameters"],
            32_792_226,
        )
        self.assertEqual(
            OPENVOICE_CONVERTER_CHECKPOINT["sha256"],
            "9652c27e92b6b2a91632590ac9962ef7ae2b712e5c5b7f4c34ec55ee2b37ab9e",
        )

        source = (Path(__file__).resolve().parents[1] / 'voicehub/models/openvoice/native/SOURCE.json')
        document = json.loads(source.read_text(encoding="utf-8"))
        self.assertFalse(
            any(
                "upstream parity" in limitation.lower() and "no " not in limitation.lower()
                for limitation in document["verified_scope"]["limitations"]))

    def test_default_meta_graph_matches_the_audited_checkpoint(self):
        with torch.device("meta"):
            model = OpenVoiceToneColorConverter(OpenVoiceConverterConfig())

        state = model.state_dict()
        self.assertEqual(len(state), 486)
        self.assertEqual(
            sum(tensor.numel() for tensor in state.values()),
            32_792_226,
        )

    def test_tiny_raw_audio_training_reaches_the_entire_converter(self):
        torch.manual_seed(7)
        config = _tiny_config()
        model = OpenVoiceToneColorConverter(config)
        processor = OpenVoiceAudioProcessor(config)
        source = processor.spectrogram((torch.randn(192), torch.randn(176)))
        target = processor.waveform_batch((torch.randn(192), torch.randn(180)))
        target_reference = processor.spectrogram((torch.randn(192), torch.randn(180)))

        output = model(
            source_spectrogram=source.values,
            source_lengths=source.lengths,
            source_reference_spectrogram=source.values,
            source_reference_lengths=source.lengths,
            target_reference_spectrogram=target_reference.values,
            target_reference_lengths=target_reference.lengths,
            target_waveform=target.values,
            target_lengths=target.lengths,
            tau=0.0,
        )
        self.assertEqual(output.waveform.shape, (2, 1, 192))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))

        output.loss.backward()
        parameters = tuple(model.parameters())
        self.assertTrue(parameters)
        self.assertTrue(all(parameter.grad is not None for parameter in parameters))

    def test_reference_lengths_crop_padding_inside_autograd(self):
        torch.manual_seed(11)
        config = _tiny_config()
        model = OpenVoiceToneColorConverter(config)
        processor = OpenVoiceAudioProcessor(config)
        references = processor.spectrogram((torch.randn(192), torch.randn(176)))
        padded = references.values.detach().clone().requires_grad_(True)

        embeddings = model.extract_speaker_embeddings(
            padded,
            lengths=references.lengths,
        )
        embeddings.sum().backward()

        self.assertEqual(embeddings.shape, (2, 8, 1))
        self.assertIsNotNone(padded.grad)
        second_length = int(references.lengths[1])
        self.assertEqual(
            torch.count_nonzero(padded.grad[1, :, second_length:]).item(),
            0,
        )

    def test_native_checkpoint_round_trip_and_strict_inventory(self):
        config = _tiny_config()
        model = OpenVoiceToneColorConverter(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = save_openvoice_checkpoint(
                model,
                root / "model.safetensors",
            )
            artifacts = _artifacts(root, config, checkpoint)
            restored = OpenVoiceToneColorConverter(config)
            load_openvoice_checkpoint(restored, artifacts)

            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]))

            state = dict(model.state_dict())
            state.pop(next(iter(state)))
            bad_checkpoint = save_safetensors(
                state,
                root / "bad.safetensors",
            )
            bad_artifacts = _artifacts(
                root,
                config,
                bad_checkpoint,
            )
            with self.assertRaises(CheckpointCompatibilityError):
                read_openvoice_checkpoint(restored, bad_artifacts)

    def test_legacy_pickle_import_is_explicitly_trusted(self):
        config = _tiny_config()
        model = OpenVoiceToneColorConverter(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pth"
            torch.save({"model": model.state_dict()}, checkpoint)
            artifacts = _artifacts(
                root,
                config,
                checkpoint,
                legacy=True,
            )

            with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint"):
                read_openvoice_checkpoint(model, artifacts)
            state = read_openvoice_checkpoint(
                model,
                artifacts,
                trust_pickle_checkpoint=True,
            )
            self.assertEqual(set(state), set(model.state_dict()))

    def test_runtime_export_accepts_only_an_empty_destination(self):
        config = _tiny_config()
        model = OpenVoiceToneColorConverter(config)
        processor = OpenVoiceAudioProcessor(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_checkpoint = save_openvoice_checkpoint(
                model,
                root / "source.safetensors",
            )
            artifacts = _artifacts(root, config, source_checkpoint)
            runtime = OpenVoiceRuntime(
                model=model,
                processor=processor,
                artifacts=artifacts,
            )
            destination = root / "export"
            destination.mkdir()
            runtime.save_pretrained(destination)

            resolved = resolve_openvoice_artifacts(destination)
            self.assertFalse(resolved.legacy_pytorch)
            reloaded = load_openvoice_runtime(destination)
            self.assertEqual(
                set(reloaded.model.state_dict()),
                set(model.state_dict()),
            )
            self.assertIsNotNone(reloaded._weight_norm_cache)
            self.assertGreater(reloaded._weight_norm_cache.module_count, 0)
            reloaded.train()
            self.assertIsNone(reloaded._weight_norm_cache)
            reloaded.eval()
            self.assertIsNotNone(reloaded._weight_norm_cache)
            with self.assertRaises(FileExistsError):
                runtime.save_pretrained(destination)

    def test_training_collator_preserves_variable_length_audio(self):
        collator = OpenVoiceTrainingCollator()
        batch = collator([
            {
                "source_audio": torch.randn(180),
                "target_audio": torch.randn(192),
                "sampling_rate": 22_050,
            },
            {
                "source_audio": torch.randn(160),
                "target_audio": torch.randn(176),
                "sampling_rate": 22_050,
            },
        ])

        self.assertEqual(
            tuple(value.numel() for value in batch["source_audio"]),
            (180, 160),
        )
        self.assertEqual(
            tuple(value.numel() for value in batch["target_audio"]),
            (192, 176),
        )
        self.assertEqual(batch["sampling_rate"], 22_050)

    def test_public_training_requires_an_explicit_reconstructed_recipe_opt_in(self):
        disabled = OpenVoiceForTextToSpeech(
            OpenVoiceConfig(enable_reconstructed_finetuning=False),
            lazy_load=True,
        )
        enabled = OpenVoiceForTextToSpeech(
            OpenVoiceConfig(enable_reconstructed_finetuning=True),
            lazy_load=True,
        )

        with self.assertRaisesRegex(ValueError, "does not publish"):
            disabled._validate_training_runtime()
        enabled._validate_training_runtime()
        adapter = AutoTrainingAdapter.from_model(enabled)
        self.assertIsInstance(adapter, OpenVoiceTrainingAdapter)
        self.assertIs(adapter.spec.support, TrainingSupport.CUSTOM)

    def test_public_inference_fails_closed_at_external_boundaries(self):
        model = OpenVoiceForTextToSpeech(lazy_load=True)
        with self.assertRaisesRegex(ValueError, "base_audio"):
            model._validate_generation_inputs({
                "speaker_audio_path": torch.zeros(1_000),
                "speaker_audio_sampling_rate": 22_050,
            })
        with self.assertRaisesRegex(ValueError, "external VAD"):
            model._validate_generation_inputs({
                "base_audio": torch.zeros(1_000),
                "base_audio_sampling_rate": 22_050,
                "speaker_audio_path": torch.zeros(1_000),
                "speaker_audio_sampling_rate": 22_050,
                "vad": True,
            })

    def test_architecture_registration_is_lazy_and_truthful(self):
        spec = get_architecture_spec("openvoice")
        self.assertEqual(spec.architecture_id, "openvoice-v2-converter")
        self.assertTrue(spec.capabilities.training)
        self.assertEqual(spec.capabilities.dtypes, ("float32", ))
        self.assertFalse(spec.metadata["official_training_recipe_available"])
        self.assertFalse(spec.metadata["full_upstream_finetuning_parity"])

        code = """
import json
import sys
from voicehub.registry import get_model_spec
spec = get_model_spec("openvoice")
print(json.dumps({
    "native": spec.is_voicehub_native,
    "architecture": spec.architecture,
    "torch": "torch" in sys.modules,
    "provider": "voicehub.models.openvoice.modeling" in sys.modules,
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload["native"])
        self.assertEqual(
            payload["architecture"],
            "openvoice-v2-converter",
        )
        self.assertFalse(payload["torch"])
        self.assertFalse(payload["provider"])


if __name__ == "__main__":
    unittest.main()
