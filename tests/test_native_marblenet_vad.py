import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.vad_nemo import NeMoVADConfig, NeMoVADForVoiceActivityDetection
from voicehub.models.vad_nemo.native.checkpoint import (
    MarbleNetVADSafeTensorsCheckpointAdapter,
    convert_nemo_marblenet_checkpoint,
)
from voicehub.models.vad_nemo.native.configuration import MarbleNetVADConfig
from voicehub.models.vad_nemo.native.metadata import (
    MARBLENET_VAD_REPOSITORY,
    MARBLENET_VAD_REVISION,
    MARBLENET_VAD_SHA256,
    NEMO_SOURCE_REVISION,
)
from voicehub.models.vad_nemo.native.modeling import MarbleNetVADModel
from voicehub.registry import get_model_spec
from voicehub.runtime import get_architecture_spec
from voicehub.training import get_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_native_artifact(path):
    config = MarbleNetVADConfig(dither=0.0)
    model = MarbleNetVADModel(config)
    save_safetensors(
        model.state_dict(),
        Path(path) / "model.safetensors",
        metadata={"format": "voicehub-marblenet-vad-v1"},
    )
    write_json_file(Path(path) / "config.json", config.to_dict())
    return model


class NativeMarbleNetArchitectureTests(unittest.TestCase):

    def test_provider_import_does_not_import_external_architecture_runtimes(self):
        code = """
import json
import sys
from voicehub.models.vad_nemo import NeMoVADForVoiceActivityDetection
print(json.dumps({
    name: name in sys.modules
    for name in ("nemo", "librosa", "omegaconf", "pytorch_lightning", "torchaudio")
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "nemo": False,
                "librosa": False,
                "omegaconf": False,
                "pytorch_lightning": False,
                "torchaudio": False,
            },
        )

    def test_released_graph_preserves_the_complete_tensor_namespace(self):
        state = MarbleNetVADModel(MarbleNetVADConfig()).state_dict()
        self.assertEqual(len(state), 84)
        self.assertEqual(
            tuple(state["preprocessor.featurizer.fb"].shape),
            (1, 80, 257),
        )
        self.assertEqual(
            tuple(state["encoder.encoder.0.mconv.0.conv.weight"].shape),
            (80, 1, 11),
        )
        self.assertEqual(
            tuple(state["encoder.encoder.4.mconv.0.conv.weight"].shape),
            (64, 1, 29),
        )
        self.assertEqual(tuple(state["decoder.layer0.weight"].shape), (2, 128))
        self.assertEqual(sum(tensor.numel() for tensor in state.values()), 114_270)

    def test_raw_audio_forward_masks_lengths_and_backpropagates(self):
        model = MarbleNetVADModel(MarbleNetVADConfig(dither=0.0), )
        waveforms = torch.randn(2, 3_200)
        lengths = torch.tensor([3_200, 2_400])
        frontend_lengths = lengths // 160 + 1
        frame_lengths = (frontend_lengths + 1) // 2
        labels = torch.zeros(2, int(frame_lengths.max()), dtype=torch.long)
        label_mask = (torch.arange(labels.shape[1]).unsqueeze(0) < frame_lengths.unsqueeze(1))

        output = model(
            waveforms,
            waveform_lengths=lengths,
            labels=labels,
            label_mask=label_mask,
        )
        output.loss.backward()

        self.assertEqual(output.logits.shape, (2, 11, 2))
        self.assertEqual(output.frame_lengths.tolist(), [11, 8])
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad))

    def test_restricted_checkpoint_conversion_and_safe_reload(self):
        source_model = MarbleNetVADModel(MarbleNetVADConfig())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.ckpt"
            torch.save(source_model.state_dict(), checkpoint)
            with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint"):
                convert_nemo_marblenet_checkpoint(checkpoint, root / "native")
            output = convert_nemo_marblenet_checkpoint(
                checkpoint,
                root / "native",
                trust_pickle_checkpoint=True,
            )
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            restored = MarbleNetVADModel(MarbleNetVADConfig.from_dict(config))
            with SafeTensorReader(output / "model.safetensors") as reader:
                MarbleNetVADSafeTensorsCheckpointAdapter().load_streaming(
                    restored,
                    reader,
                    config,
                    strict=True,
                )
        for name, expected in source_model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], expected)

    def test_nemo_tar_conversion_refuses_unsafe_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "weights.ckpt"
            torch.save(MarbleNetVADModel(MarbleNetVADConfig()).state_dict(), checkpoint)
            config = root / "model_config.yaml"
            config.write_text("target: FrameVAD\n", encoding="utf-8")
            archive = root / "model.nemo"
            with tarfile.open(archive, "w") as handle:
                handle.add(checkpoint, arcname="model_weights.ckpt")
                handle.add(config, arcname="model_config.yaml")
                link = tarfile.TarInfo("unsafe")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../outside"
                handle.addfile(link)
            with self.assertRaisesRegex(ValueError, "links"):
                convert_nemo_marblenet_checkpoint(
                    archive,
                    root / "native",
                    trust_pickle_checkpoint=True,
                )

    def test_provenance_and_registry_are_native(self):
        self.assertEqual(
            MARBLENET_VAD_REPOSITORY,
            "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0",
        )
        self.assertEqual(len(MARBLENET_VAD_REVISION), 40)
        self.assertEqual(len(NEMO_SOURCE_REVISION), 40)
        self.assertEqual(len(MARBLENET_VAD_SHA256), 64)
        architecture = get_architecture_spec("marblenet-vad")
        self.assertTrue(architecture.capabilities.training)
        self.assertFalse(architecture.capabilities.streaming)
        model_spec = get_model_spec("vad_nemo")
        self.assertEqual(model_spec.architecture, "marblenet-vad")
        self.assertIn("voicehub-native", model_spec.capabilities)
        training = get_training_spec("vad_nemo")
        self.assertEqual(training.support.value, "native")
        self.assertEqual(training.default_phase, "voice_activity_detection")


class NativeMarbleNetProviderTests(unittest.TestCase):

    def test_config_rejects_unverified_graphs_and_provider_options(self):
        for kwargs in (
            {"sample_rate": 8_000},
            {"architecture_family": "window"},
            {"speech_class_id": 0},
            {"model_kwargs": {'trainer': object()}},
            {"model_kwargs": {"token": "secret"}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                NeMoVADConfig(**kwargs)

    def test_local_inference_training_export_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            _write_native_artifact(source)
            wrapper = NeMoVADForVoiceActivityDetection(
                NeMoVADConfig(
                    name_or_path=source,
                    training_dither=0.0,
                ),
                device="cpu",
                lazy_load=False,
            )
            output = wrapper.detect(
                torch.zeros(1_600),
                sampling_rate=16_000,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
                return_frames=True,
            )
            self.assertEqual(len(output.probabilities), 6)
            self.assertEqual(output.metadata["backend"], "voicehub-native")

            wrapper.load_for_training()
            batch = wrapper.prepare_training_inputs(
                {
                    "audio": torch.zeros(1_600),
                    "sampling_rate": 16_000,
                    "segments": [(0.02, 0.06)],
                },
                phase="voice_activity_detection",
            )
            result = wrapper.model(**batch)
            result.loss.backward()
            self.assertEqual(result.logits.shape, (1, 6, 2))

            export = Path(directory) / "export"
            wrapper.export_native_pretrained(export)
            restored = NeMoVADForVoiceActivityDetection(
                NeMoVADConfig(name_or_path=export),
                device="cpu",
                lazy_load=False,
            )
            self.assertEqual(len(restored.model.state_dict()), 84)

    def test_official_pickle_boundary_is_explicit(self):
        wrapper = NeMoVADForVoiceActivityDetection(
            NeMoVADConfig(),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint"):
            wrapper._load_pretrained_model()


if __name__ == "__main__":
    unittest.main()
