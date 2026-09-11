from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from tests._moonshine_test_utils import write_tiny_moonshine_artifact
from voicehub import AutoConfig, AutoModelForSpeechRecognition, get_model_spec
from voicehub.models.asr_moonshine import (
    MoonshineASRConfig,
    MoonshineForSpeechRecognition,
    NativeMoonshineTrainingAdapter,
)
from voicehub.models.asr_moonshine.native import resolve_moonshine_artifacts
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.specs import get_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DeterministicMoonshine(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def generate(
        self,
        input_values,
        attention_mask,
        **kwargs,
    ):
        del attention_mask, kwargs
        return torch.tensor(
            [[1, 7, 2]],
            dtype=torch.long,
            device=input_values.device,
        ).repeat(input_values.shape[0], 1)


class NativeMoonshineProviderTests(unittest.TestCase):

    def test_provider_import_does_not_load_external_or_tensor_runtimes(self):
        code = """
import json
import sys
import voicehub.models.asr_moonshine
names = ("torch", "transformers", "tokenizers", "safetensors", "numpy", "torchaudio")
print(json.dumps({name: name in sys.modules for name in names}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(result.stdout),
            {
                "torch": False,
                "transformers": False,
                "tokenizers": False,
                "safetensors": False,
                "numpy": False,
                "torchaudio": False,
            },
        )

    def test_local_safe_load_training_loss_and_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference = write_tiny_moonshine_artifact(root)
            wrapper = MoonshineForSpeechRecognition(
                MoonshineASRConfig(name_or_path=root),
                device="cpu",
            )

            wrapper.load_for_training()

            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(
                    wrapper.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )
            prepared = wrapper.prepare_training_inputs(
                {
                    "audio": torch.linspace(-1.0, 1.0, 1_000),
                    "sampling_rate": 16_000,
                    "text": "hi!",
                },
                phase="speech_recognition",
            )
            self.assertEqual(prepared["labels"].tolist(), [7, 8, 2])
            self.assertEqual(
                prepared["decoder_attention_mask"].tolist(),
                [1, 1, 1],
            )
            padded_labels = wrapper.moonshine_processor.encode_labels(("hi!", "hi"), )
            self.assertEqual(
                padded_labels["labels"].tolist(),
                [[7, 8, 2], [7, 2, -100]],
            )
            self.assertEqual(
                padded_labels["decoder_attention_mask"].tolist(),
                [[1, 1, 1], [1, 1, 0]],
            )
            output = wrapper.model(
                prepared["input_values"].unsqueeze(0),
                prepared["attention_mask"].unsqueeze(0),
                decoder_attention_mask=prepared["decoder_attention_mask"].unsqueeze(0),
                labels=prepared["labels"].unsqueeze(0),
            )
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(wrapper.model.model.decoder.embed_tokens.weight.grad)

    def test_inference_decodes_native_tokens_and_rejects_unsupported_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tiny_moonshine_artifact(root)
            wrapper = MoonshineForSpeechRecognition(
                MoonshineASRConfig(name_or_path=root),
                device="cpu",
            )
            wrapper.load_for_training()
            wrapper.model = _DeterministicMoonshine()

            result = wrapper.transcribe(
                torch.linspace(-0.5, 0.5, 1_000),
                sampling_rate=16_000,
                max_new_tokens=3,
            )

            self.assertEqual(result.text, "hi")
            self.assertEqual(result.language, "en")
            self.assertEqual(result.metadata["backend"], "voicehub-native")
            self.assertEqual(result.metadata["decoding"], "greedy")
            with self.assertRaisesRegex(ValueError, "greedy"):
                wrapper.transcribe(
                    torch.ones(1_000),
                    sampling_rate=16_000,
                    num_beams=2,
                )
            with self.assertRaisesRegex(ValueError, "timestamp"):
                wrapper.transcribe(
                    torch.ones(1_000),
                    sampling_rate=16_000,
                    return_timestamps=True,
                )
            with self.assertRaisesRegex(ValueError, "English"):
                wrapper.transcribe(
                    torch.ones(1_000),
                    sampling_rate=16_000,
                    language="fr",
                )

    def test_training_adapter_exports_reloadable_native_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference = write_tiny_moonshine_artifact(root)
            wrapper = MoonshineForSpeechRecognition(
                MoonshineASRConfig(name_or_path=root),
                device="cpu",
            )
            adapter = AutoTrainingAdapter.from_model(wrapper)
            self.assertIsInstance(adapter, NativeMoonshineTrainingAdapter)

            context = adapter.create_training_context({
                "audio": torch.randn(1_000),
                "sampling_rate": 16_000,
                "text": "hi!",
            })
            output = adapter.execute_training_phase(context)
            self.assertTrue(torch.isfinite(output.loss))

            export = root / "export"
            adapter.save_pretrained(export)
            exported = json.loads((export / "config.json").read_text(encoding="utf-8"))
            config = AutoConfig.from_pretrained(export)
            reloaded = AutoModelForSpeechRecognition.from_pretrained(
                export,
                config=config,
                device="cpu",
            )
            reloaded.load_for_training()

            self.assertEqual(exported["model_type"], "asr_moonshine")
            self.assertEqual(
                exported["voicehub_checkpoint_format"],
                "native-moonshine-seq2seq-v1",
            )
            self.assertIsInstance(config, MoonshineASRConfig)
            self.assertIsInstance(reloaded, MoonshineForSpeechRecognition)
            for filename in (
                    "config.json",
                    "generation_config.json",
                    "model.safetensors",
                    "preprocessor_config.json",
                    "tokenizer.json",
            ):
                self.assertTrue((export / filename).is_file(), filename)
            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(
                    reloaded.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )

    def test_registry_training_spec_and_adapter_are_native(self):
        spec = get_model_spec("asr_moonshine")
        training = get_training_spec("asr_moonshine")

        self.assertEqual(
            spec.module,
            "voicehub.models.asr_moonshine.modeling",
        )
        self.assertIn("voicehub-native", spec.capabilities)
        self.assertEqual(spec.architecture, "moonshine")
        self.assertEqual(
            training.source_entrypoints,
            ("voicehub.models.asr_moonshine.native."
             "MoonshineForConditionalGeneration", ),
        )

    def test_remote_code_pickle_and_unsafe_shards_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "never executes"):
            MoonshineASRConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "Safetensors"):
            MoonshineASRConfig(use_safetensors=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {
                    "model.encoder.conv1.weight": "../escape.safetensors"
                }}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                resolve_moonshine_artifacts(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.bin").write_bytes(b"pickle")
            with self.assertRaisesRegex(ValueError, "Safetensors"):
                resolve_moonshine_artifacts(root / "weights.bin")


if __name__ == "__main__":
    unittest.main()
