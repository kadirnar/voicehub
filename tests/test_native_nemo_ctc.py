from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.checkpointing import save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.asr_nemo import NativeNeMoCTCTrainingAdapter, NeMoASRConfig, NeMoASRForSpeechRecognition
from voicehub.models.asr_nemo.native.checkpoint import (
    NeMoCTCSafeTensorsCheckpointAdapter,
    convert_nemo_quartznet_checkpoint,
    native_nemo_ctc_tensor_shapes,
)
from voicehub.models.asr_nemo.native.configuration import JasperBlockConfig, NeMoQuartzNetCTCConfig
from voicehub.models.asr_nemo.native.metadata import (
    NEMO_SOURCE_REVISION,
    QUARTZNET_SHA256,
    QUARTZNET_STATE_VALUES,
    QUARTZNET_TENSOR_COUNT,
    QUARTZNET_TENSOR_FINGERPRINT,
)
from voicehub.models.asr_nemo.native.modeling import NeMoQuartzNetForCTC
from voicehub.models.asr_nemo.native.tokenization import NeMoCharacterTokenizer
from voicehub.registry import get_model_spec
from voicehub.training import get_training_spec
from voicehub.training.specs import TrainingFamily

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_config() -> NeMoQuartzNetCTCConfig:
    return NeMoQuartzNetCTCConfig(
        variant="custom",
        num_mel_bins=8,
        pad_to=1,
        vocabulary=(" ", "a", "b", "'"),
        encoder_blocks=(
            JasperBlockConfig(
                filters=8,
                repeat=1,
                kernel_size=3,
                stride=2,
            ),
            JasperBlockConfig(
                filters=12,
                repeat=2,
                kernel_size=3,
                residual=True,
            ),
        ),
        spec_cutout_masks=0,
    )


def _write_native_artifact(
    directory: Path,
    config: NeMoQuartzNetCTCConfig,
    model: NeMoQuartzNetForCTC,
) -> None:
    save_safetensors(
        model.state_dict(),
        directory / "model.safetensors",
        metadata={"format": "voicehub-nemo-quartznet-ctc-v1"},
    )
    values = config.to_dict()
    values.update({
        'architectures': ["NeMoQuartzNetForCTC"],
        "model_type": "asr_nemo",
        "voicehub_provider": "asr_nemo",
    })
    write_json_file(directory / "config.json", values)


class NativeNeMoArchitectureTests(unittest.TestCase):

    def test_released_graph_matches_real_checkpoint_inventory(self):
        shapes = native_nemo_ctc_tensor_shapes()

        self.assertEqual(len(shapes), QUARTZNET_TENSOR_COUNT)
        self.assertEqual(
            sum(torch.empty(shape).numel() for shape in shapes.values()),
            QUARTZNET_STATE_VALUES,
        )
        self.assertEqual(
            shapes["preprocessor.featurizer.window"],
            (320, ),
        )
        self.assertEqual(
            shapes["encoder.encoder.0.mconv.0.conv.weight"],
            (64, 1, 33),
        )
        self.assertEqual(
            shapes["encoder.encoder.17.mconv.0.conv.weight"],
            (1024, 512, 1),
        )
        self.assertEqual(
            shapes["decoder.decoder_layers.0.weight"],
            (29, 1024, 1),
        )
        self.assertEqual(len(NEMO_SOURCE_REVISION), 40)
        self.assertEqual(len(QUARTZNET_SHA256), 64)
        self.assertEqual(len(QUARTZNET_TENSOR_FINGERPRINT), 64)

    def test_tiny_graph_computes_ctc_loss_and_full_backward(self):
        model = NeMoQuartzNetForCTC(_tiny_config())
        features = torch.randn(2, 8, 24)
        labels = torch.tensor([
            [1, 0, 2],
            [2, 1, -1],
        ])

        output = model(
            processed_signal=features,
            processed_signal_length=torch.tensor([24, 20]),
            labels=labels,
            label_lengths=torch.tensor([3, 2]),
        )
        output.loss.backward()

        self.assertEqual(tuple(output.logits.shape), (2, 12, 5))
        self.assertEqual(output.encoded_lengths.tolist(), [12, 10])
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNotNone(model.encoder.encoder[0].mconv[0].conv.weight.grad)
        self.assertIsNotNone(model.decoder.decoder_layers[0].weight.grad)

    def test_character_tokenizer_normalizes_and_decodes_offsets(self):
        tokenizer = NeMoCharacterTokenizer((" ", "a", "b", "'"))

        self.assertEqual(
            tokenizer.encode("  A\u2019B  "),
            (1, 3, 2),
        )
        decoded = tokenizer.decode_ctc([4, 1, 1, 4, 0, 2, 2, 4], )

        self.assertEqual(decoded.text, "a b")
        self.assertEqual(decoded.words[0].start_offset, 1)
        self.assertEqual(decoded.words[1].end_offset, 6)
        with self.assertRaisesRegex(ValueError, "cannot encode"):
            tokenizer.encode("c")

    def test_native_artifact_loads_exports_and_reloads(self):
        torch.manual_seed(17)
        config = _tiny_config()
        source_model = NeMoQuartzNetForCTC(config).eval()
        features = torch.randn(1, 8, 24)
        lengths = torch.tensor([21])
        expected = source_model(
            processed_signal=features,
            processed_signal_length=lengths,
        ).logits

        with tempfile.TemporaryDirectory() as source_directory:
            source = Path(source_directory)
            _write_native_artifact(source, config, source_model)
            wrapper = NeMoASRForSpeechRecognition(
                NeMoASRConfig(name_or_path=source),
                device="cpu",
                lazy_load=False,
            )
            actual = wrapper.model(
                processed_signal=features,
                processed_signal_length=lengths,
            ).logits
            self.assertTrue(torch.equal(expected, actual))

            with tempfile.TemporaryDirectory() as export_directory:
                exported = wrapper.export_native_pretrained(export_directory)
                fresh = NeMoASRForSpeechRecognition(
                    NeMoASRConfig(name_or_path=exported),
                    device="cpu",
                    lazy_load=False,
                )
                reloaded = fresh.model(
                    processed_signal=features,
                    processed_signal_length=lengths,
                ).logits
                self.assertTrue(torch.equal(actual, reloaded))

    @unittest.skipUnless(
        os.environ.get("VOICEHUB_TEST_NEMO_QUARTZNET_CHECKPOINT"),
        "set VOICEHUB_TEST_NEMO_QUARTZNET_CHECKPOINT for real archive validation",
    )
    def test_real_archive_converts_with_exact_namespace(self):
        source = Path(os.environ["VOICEHUB_TEST_NEMO_QUARTZNET_CHECKPOINT"])
        with tempfile.TemporaryDirectory() as directory:
            output = convert_nemo_quartznet_checkpoint(source, directory)
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))

            self.assertEqual(
                config["source_checkpoint_sha256"],
                QUARTZNET_SHA256,
            )
            self.assertEqual(
                config["source_tensor_fingerprint"],
                QUARTZNET_TENSOR_FINGERPRINT,
            )


class NativeNeMoProviderTests(unittest.TestCase):

    def test_provider_and_training_registry_are_native_ctc(self):
        provider = get_model_spec("asr_nemo")
        training = get_training_spec("asr_nemo")

        self.assertEqual(
            provider.default_model_path,
            NeMoASRForSpeechRecognition.default_model_name_or_path,
        )
        self.assertIn("voicehub-native", provider.capabilities)
        self.assertEqual(provider.license.license_id, "NVIDIA-NGC-Terms")
        self.assertEqual(training.family, TrainingFamily.CTC)
        self.assertIn(
            "voicehub.models.asr_nemo.native",
            training.source_entrypoints[0],
        )
        adapter = NeMoASRForSpeechRecognition(NeMoASRConfig(), ).get_training_adapter()
        self.assertIsInstance(adapter, NativeNeMoCTCTrainingAdapter)

    def test_unverified_neural_families_fail_before_network_access(self):
        model = NeMoASRForSpeechRecognition(NeMoASRConfig(name_or_path="nvidia/parakeet-tdt-0.6b-v2", ))

        with self.assertRaisesRegex(ValueError, "not the verified QuartzNet15x5"):
            model.load()

    def test_native_provider_imports_no_external_model_framework(self):
        code = """
import json
import sys
from voicehub.models.asr_nemo import NeMoASRForSpeechRecognition
blocked = ("nemo", "transformers", "torchaudio", "huggingface_hub", "safetensors")
print(json.dumps({name: name in sys.modules for name in blocked}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(result.stdout.strip().splitlines()[-1]),
            {
                "nemo": False,
                "transformers": False,
                "torchaudio": False,
                "huggingface_hub": False,
                "safetensors": False,
            },
        )

    def test_checkpoint_adapter_is_declared_and_configuration_is_strict(self):
        adapter = NeMoCTCSafeTensorsCheckpointAdapter()

        self.assertEqual(adapter.architecture_id, "nemo-quartznet-ctc")
        with self.assertRaisesRegex(ValueError, "character-CTC"):
            NeMoASRConfig(model_class="EncDecRNNTModel")
        with self.assertRaisesRegex(ValueError, "Safetensors"):
            NeMoASRConfig(checkpoint_filename="model.ckpt")


if __name__ == "__main__":
    unittest.main()
