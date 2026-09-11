import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.vad_speechbrain import (
    NativeSpeechBrainVADTrainingAdapter,
    SpeechBrainVADConfig,
    SpeechBrainVADForVoiceActivityDetection,
)
from voicehub.models.vad_speechbrain.native.checkpoint import (
    NATIVE_SPEECHBRAIN_VAD_FORMAT,
    SpeechBrainVADSafeTensorsCheckpointAdapter,
    convert_speechbrain_vad_checkpoint,
    speechbrain_source_tensor_mapping,
)
from voicehub.models.vad_speechbrain.native.configuration import SpeechBrainCRDNNVADConfig
from voicehub.models.vad_speechbrain.native.inference import SpeechBrainVADInference
from voicehub.models.vad_speechbrain.native.metadata import (
    SPEECHBRAIN_TRAINING_SOURCE_REVISION,
    SPEECHBRAIN_VAD_CHECKPOINT_LICENSE,
    SPEECHBRAIN_VAD_MODEL_SHA256,
    SPEECHBRAIN_VAD_REVISION,
    SPEECHBRAIN_VAD_TENSOR_FINGERPRINT,
)
from voicehub.models.vad_speechbrain.native.modeling import SpeechBrainCRDNNVADModel
from voicehub.registry import get_model_spec
from voicehub.runtime import get_architecture_spec
from voicehub.training import AutoTrainingAdapter, get_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _upstream_state(model):
    native = model.state_dict()
    return {
        source_name: native[native_name].clone()
        for source_name, native_name in speechbrain_source_tensor_mapping().items()
    }


def _native_artifact(root, *, speech_probability=0.5):
    del speech_probability
    config = SpeechBrainCRDNNVADConfig()
    model = SpeechBrainCRDNNVADModel(config)
    root.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        model.state_dict(),
        root / "model.safetensors",
        metadata={"format": NATIVE_SPEECHBRAIN_VAD_FORMAT},
    )
    write_json_file(root / "config.json", config.to_dict())
    return model


class NativeSpeechBrainVADArchitectureTests(unittest.TestCase):

    def test_provider_import_does_not_import_external_architecture_runtimes(self):
        code = """
import json
import sys
from voicehub.models.vad_speechbrain import SpeechBrainVADForVoiceActivityDetection
print(json.dumps({
    name: name in sys.modules
    for name in ("speechbrain", "torchaudio", "hyperpyyaml", "transformers")
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
            json.loads(completed.stdout.strip()),
            {
                "speechbrain": False,
                "torchaudio": False,
                "hyperpyyaml": False,
                "transformers": False,
            },
        )

    def test_graph_preserves_the_complete_published_tensor_inventory(self):
        state = SpeechBrainCRDNNVADModel(SpeechBrainCRDNNVADConfig()).state_dict()
        self.assertEqual(len(state), 49)
        self.assertEqual(sum(tensor.numel() for tensor in state.values()), 109_810)
        self.assertEqual(
            tuple(state["cnn_blocks.0.norm_1.weight"].shape),
            (40, 16),
        )
        self.assertEqual(
            tuple(state["rnn.weight_ih_l0"].shape),
            (96, 320),
        )
        self.assertEqual(tuple(state["output.weight"].shape), (1, 16))

    def test_frontend_and_raw_audio_graph_are_deterministic(self):
        torch.manual_seed(11)
        model = SpeechBrainCRDNNVADModel(SpeechBrainCRDNNVADConfig()).eval()
        waveform = torch.randn(2, 3_200)
        with torch.inference_mode():
            first = model(waveform)
            second = model(waveform)
        torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
        self.assertEqual(first.logits.shape, (2, 21))
        self.assertEqual(first.frame_lengths.tolist(), [21, 21])

    def test_raw_audio_fine_tuning_backpropagates_through_crdnn_only(self):
        model = SpeechBrainCRDNNVADModel(SpeechBrainCRDNNVADConfig()).train()
        waveforms = torch.randn(2, 3_200)
        labels = torch.zeros(2, 20)
        labels[:, 3:12] = 1.0
        output = model(waveforms, labels=labels)
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNone(waveforms.grad)
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad))

    def test_restricted_checkpoint_conversion_and_safe_reload(self):
        source_model = SpeechBrainCRDNNVADModel(SpeechBrainCRDNNVADConfig())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.ckpt"
            torch.save(_upstream_state(source_model), checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint"):
                convert_speechbrain_vad_checkpoint(checkpoint, root / "native")
            output = convert_speechbrain_vad_checkpoint(
                checkpoint,
                root / "native",
                trust_pickle_checkpoint=True,
                expected_checkpoint_sha256=digest,
            )
            values = json.loads((output / "config.json").read_text(encoding="utf-8"))
            restored = SpeechBrainCRDNNVADModel(SpeechBrainCRDNNVADConfig.from_dict(values))
            with SafeTensorReader(output / "model.safetensors") as reader:
                SpeechBrainVADSafeTensorsCheckpointAdapter().load_streaming(
                    restored,
                    reader,
                    values,
                    strict=True,
                )
        for name, tensor in source_model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], tensor)

    def test_source_compatible_decoder_uses_hysteresis_and_strict_lengths(self):
        model = SpeechBrainCRDNNVADModel(SpeechBrainCRDNNVADConfig()).eval()
        inference = SpeechBrainVADInference(
            model,
            large_chunk_size=0.1,
            small_chunk_size=0.1,
        )
        probabilities = torch.tensor([0.1, 0.6, 0.4, 0.3, 0.2, 0.7, 0.1], )
        decisions = inference.threshold(
            probabilities,
            activation_threshold=0.5,
            deactivation_threshold=0.25,
        )
        self.assertEqual(
            decisions.tolist(),
            [False, True, True, True, False, True, False],
        )
        boundaries = inference.boundaries(decisions, probabilities=probabilities)
        self.assertEqual(
            [(item.start, item.end) for item in boundaries],
            [(0.01, 0.03)],
        )
        self.assertEqual(
            inference.remove_short(
                boundaries,
                minimum_duration=0.02,
            ),
            (),
        )

    def test_provenance_and_registry_are_explicit(self):
        self.assertEqual(len(SPEECHBRAIN_TRAINING_SOURCE_REVISION), 40)
        self.assertEqual(len(SPEECHBRAIN_VAD_REVISION), 40)
        self.assertEqual(len(SPEECHBRAIN_VAD_MODEL_SHA256), 64)
        self.assertEqual(len(SPEECHBRAIN_VAD_TENSOR_FINGERPRINT), 64)
        self.assertEqual(SPEECHBRAIN_VAD_CHECKPOINT_LICENSE, "not-declared")
        architecture = get_architecture_spec("speechbrain-crdnn-vad")
        self.assertTrue(architecture.capabilities.training)
        self.assertFalse(architecture.capabilities.streaming)
        self.assertEqual(
            get_model_spec("vad_speechbrain").architecture,
            "speechbrain-crdnn-vad",
        )
        training = get_training_spec("vad_speechbrain")
        self.assertEqual(training.support.value, "native")
        self.assertEqual(training.default_phase, "voice_activity_detection")


class NativeSpeechBrainVADProviderTests(unittest.TestCase):

    def test_local_inference_training_export_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _native_artifact(source)
            wrapper = SpeechBrainVADForVoiceActivityDetection(
                SpeechBrainVADConfig(
                    name_or_path=source,
                    large_chunk_size=0.1,
                    small_chunk_size=0.1,
                    double_check=False,
                ),
                device="cpu",
                lazy_load=False,
            )
            output = wrapper.detect(
                torch.zeros(1_600),
                sampling_rate=16_000,
                threshold=0.6,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
                return_frames=True,
            )
            self.assertEqual(len(output.probabilities), 10)
            self.assertEqual(output.metadata["backend"], "voicehub-native")
            batch = wrapper.prepare_training_inputs(
                {
                    "audio": torch.zeros(3_200),
                    "sampling_rate": 16_000,
                    "segments": [(0.02, 0.10)],
                },
                phase="voice_activity_detection",
            )
            training_output = wrapper.model(**batch)
            self.assertTrue(torch.isfinite(training_output.loss))
            training_output.loss.backward()

            destination = root / "export"
            wrapper.export_native_pretrained(destination)
            restored = SpeechBrainVADForVoiceActivityDetection(
                SpeechBrainVADConfig(
                    name_or_path=destination,
                    large_chunk_size=0.1,
                    small_chunk_size=0.1,
                    double_check=False,
                ),
                device="cpu",
                lazy_load=False,
            )
            for name, tensor in wrapper.model.state_dict().items():
                torch.testing.assert_close(restored.model.state_dict()[name], tensor)

    def test_auto_training_uses_specialized_native_adapter(self):
        wrapper = SpeechBrainVADForVoiceActivityDetection(SpeechBrainVADConfig(), )
        adapter = AutoTrainingAdapter.from_model(wrapper)
        self.assertIsInstance(adapter, NativeSpeechBrainVADTrainingAdapter)


if __name__ == "__main__":
    unittest.main()
