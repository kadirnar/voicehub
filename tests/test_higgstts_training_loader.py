import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from voicehub.models.higgstts.inference import HiggsTTSConfig, HiggsTTSForTextToSpeech
from voicehub.models.higgstts.native.runtime import HiggsAudioV2Runtime
from voicehub.models.higgstts.training import HiggsSFTDataset, HiggsTrainingCollator, load_higgs_training_backend

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HiggsTrainingImportTests(unittest.TestCase):

    def test_training_module_keeps_framework_dependencies_lazy(self):
        command = (
            "import sys; import voicehub.models.higgstts.training; "
            "print('torch' in sys.modules, 'transformers' in sys.modules, "
            "'huggingface_hub' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False False False")

    def test_legacy_backend_name_resolves_to_native_runtime(self):
        from voicehub.models.higgstts.training import HiggsTrainingBackend

        self.assertIs(HiggsTrainingBackend, HiggsAudioV2Runtime)


class HiggsTrainingCompatibilityTests(unittest.TestCase):

    def test_compatibility_loader_routes_to_native_runtime(self):
        sentinel = object()
        with patch(
                "voicehub.models.higgstts.native.runtime."
                "load_higgs_audio_v2_runtime",
                return_value=sentinel,
        ) as loader:
            result = load_higgs_training_backend(
                "model/source",
                "codec/source",
                device="cpu",
                torch_dtype="float32",
                revision="revision",
            )

        self.assertIs(result, sentinel)
        loader.assert_called_once_with(
            "model/source",
            codec_source="codec/source",
            device="cpu",
            dtype="float32",
            revision="revision",
        )

    def test_collator_preserves_raw_records_for_frozen_codec(self):
        records = [
            {
                "audio": "first.wav",
                "text": "first",
            },
            {
                "audio_codes": [[1, 2]],
                "text": "second",
            },
        ]

        batch = HiggsTrainingCollator()(records)

        self.assertEqual(batch, {"records": records})
        self.assertIsNot(batch["records"][0], records[0])

    def test_dataset_requires_target_audio_and_reference_transcript(self):
        with self.assertRaisesRegex(ValueError, "requires `audio`"):
            HiggsSFTDataset([{"text": "missing target"}])
        with self.assertRaisesRegex(ValueError, "reference_text"):
            HiggsSFTDataset([{
                "audio": "target.wav",
                "reference_audio": "speaker.wav",
                "text": "target",
            }])
        dataset = HiggsSFTDataset([{
            "audio_codes": [[1, 2]],
            "reference_codes": [[3, 4]],
            "reference_text": "speaker",
            "text": "target",
        }])
        self.assertEqual(len(dataset), 1)

    def test_wrapper_loader_uses_current_native_checkpoint_ids(self):
        import torch

        runtime = SimpleNamespace(
            audio_tokenizer=Mock(),
            model=torch.nn.Linear(1, 1),
            prepare_for_inference=Mock(),
            sample_rate=24_000,
        )
        model = HiggsTTSForTextToSpeech(device="cpu")
        with patch(
                "voicehub.models.higgstts.native.runtime."
                "load_higgs_audio_v2_runtime",
                return_value=runtime,
        ) as loader:
            model.load()

        loader.assert_called_once_with(
            "bosonai/higgs-tts-2-3b-base",
            revision=None,
            codec_source="bosonai/higgs-audio-v2-tokenizer",
            codec_revision=None,
            device="cpu",
            dtype=torch.float32,
            cache_dir=None,
            token=None,
            local_files_only=False,
            verify_integrity=True,
            verify_checkpoint_integrity=False,
        )
        self.assertIs(model.model, runtime.model)
        self.assertIs(model.native_runtime, runtime)
        runtime.prepare_for_inference.assert_called_once_with()

    def test_config_rejects_non_native_runtime_modes(self):
        with self.assertRaisesRegex(ValueError, "never executes"):
            HiggsTTSConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "Safetensors"):
            HiggsTTSConfig(use_safetensors=False)


if __name__ == "__main__":
    unittest.main()
