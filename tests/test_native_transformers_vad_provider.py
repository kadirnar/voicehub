from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from voicehub.checkpointing import save_safetensors
from voicehub.models.asr_wav2vec2.native import (
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2ForSequenceClassification,
)
from voicehub.models.vad_transformers import TransformersVADConfig, TransformersVADForVoiceActivityDetection

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_config(
    *,
    architecture: str,
    extra: dict | None = None,
) -> Wav2Vec2Config:
    return Wav2Vec2Config.from_dict({
        "model_type": "wav2vec2",
        'architectures': [architecture],
        "id2label": {
            "0": "silence",
            "1": "speech",
        },
        "vocab_size": 8,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "hidden_dropout": 0.0,
        "activation_dropout": 0.0,
        "attention_dropout": 0.0,
        "feat_proj_dropout": 0.0,
        "final_dropout": 0.0,
        "layerdrop": 0.0,
        "conv_dim": [4, 8],
        "conv_stride": [2, 2],
        "conv_kernel": [4, 2],
        "num_conv_pos_embeddings": 4,
        "num_conv_pos_embedding_groups": 2,
        "apply_spec_augment": False,
        "mask_time_prob": 0.0,
        "mask_time_min_masks": 0,
        "mask_feature_prob": 0.0,
        "mask_feature_min_masks": 0,
        "num_labels": 2,
        "classifier_proj_size": 5,
        "sampling_rate": 100,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        **(extra or {}),
    })


def _write_artifact(
    root: Path,
    *,
    family: str,
) -> tuple[Wav2Vec2Config, torch.nn.Module]:
    if family == "frame-classification":
        architecture = "Wav2Vec2ForAudioFrameClassification"
        config = _tiny_config(architecture=architecture)
        model = Wav2Vec2ForAudioFrameClassification(config)
    else:
        architecture = "Wav2Vec2ForSequenceClassification"
        config = _tiny_config(architecture=architecture)
        model = Wav2Vec2ForSequenceClassification(config)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([-6.0, 6.0]))
        projector = getattr(model, "projector", None)
        if projector is not None:
            projector.weight.zero_()
            projector.bias.zero_()
    values = config.to_dict()
    values['architectures'] = [architecture]
    (root / "config.json").write_text(
        json.dumps(values),
        encoding="utf-8",
    )
    save_safetensors(
        model.state_dict(),
        root / "model.safetensors",
    )
    Wav2Vec2FeatureExtractor(
        sampling_rate=100,
        do_normalize=True,
        return_attention_mask=True,
    ).save_pretrained(root)
    return config, model


class NativeTransformersVADProviderTests(unittest.TestCase):

    def test_public_import_does_not_import_transformers(self):
        code = (
            "import json,sys;"
            "from voicehub.models.vad_transformers import "
            "TransformersVADForVoiceActivityDetection;"
            "print(json.dumps({'torch': 'torch' in sys.modules, "
            "'transformers': 'transformers' in sys.modules}))")
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
            },
        )

    def test_frame_classifier_inference_training_and_export_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            config, _ = _write_artifact(
                root,
                family="frame-classification",
            )
            wrapper = TransformersVADForVoiceActivityDetection(
                TransformersVADConfig(
                    name_or_path=str(root),
                    architecture_family="auto",
                ),
                device="cpu",
                lazy_load=False,
            )
            output = wrapper.detect(
                torch.zeros(30),
                sampling_rate=100,
                threshold=0.5,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
                return_frames=True,
            )

            self.assertEqual(
                wrapper.architecture_family,
                "frame-classification",
            )
            self.assertEqual(output.metadata["backend"], "voicehub-native")
            self.assertEqual(output.metadata["architecture"], "wav2vec2")
            self.assertEqual(output.metadata["speech_class_id"], 1)
            self.assertEqual(output.metadata["frame_hop_samples"], 4)
            self.assertEqual(tuple(output.probabilities.shape), (7, ))
            self.assertEqual(
                [(segment.start, segment.end) for segment in output.segments],
                [(0.0, 0.28)],
            )

            prepared = wrapper.prepare_training_inputs(
                {
                    "audio": torch.randn(2, 30),
                    "sampling_rate": 100,
                    "labels": torch.zeros(
                        (2, config.feature_output_length(30)),
                        dtype=torch.long,
                    ),
                },
                phase="voice_activity_detection",
            )
            result = wrapper.model(
                prepared["input_values"],
                attention_mask=prepared["attention_mask"],
                labels=prepared["labels"],
            )
            self.assertTrue(torch.isfinite(result.loss))
            result.loss.backward()
            self.assertIsNotNone(wrapper.model.classifier.weight.grad)

            exported = Path(directory) / "exported"
            wrapper.export_native_pretrained(exported)
            restored = TransformersVADForVoiceActivityDetection(
                TransformersVADConfig(name_or_path=str(exported)),
                device="cpu",
                lazy_load=False,
            )
            restored_output = restored.detect(
                torch.zeros(30),
                sampling_rate=100,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
            )

        self.assertEqual(
            restored_output.metadata["architecture_family"],
            "frame-classification",
        )
        torch.testing.assert_close(
            restored.model.classifier.bias,
            wrapper.model.classifier.bias,
        )

    def test_clip_classifier_uses_native_windowing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_artifact(root, family="audio-classification")
            wrapper = TransformersVADForVoiceActivityDetection(
                TransformersVADConfig(
                    name_or_path=str(root),
                    architecture_family="audio-classification",
                    window_duration_s=0.1,
                    hop_duration_s=0.05,
                ),
                device="cpu",
                lazy_load=False,
            )
            output = wrapper.detect(
                torch.zeros(20),
                sampling_rate=100,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
                return_frames=True,
            )

        self.assertEqual(
            wrapper.architecture_family,
            "audio-classification",
        )
        self.assertEqual(tuple(output.probabilities.shape), (3, ))
        self.assertEqual(output.metadata["frame_hop_samples"], 5)
        self.assertEqual(output.metadata["frame_length_samples"], 10)
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.0, 0.2)],
        )

    def test_task_ambiguous_and_asr_checkpoints_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "task-ambiguous"):
            TransformersVADForVoiceActivityDetection._infer_architecture_family({
                "model_type": "wav2vec2",
                'architectures': [],
            })
        with self.assertRaisesRegex(ValueError, "ASR head"):
            TransformersVADForVoiceActivityDetection._infer_architecture_family({
                "model_type":
                "wav2vec2",
                'architectures': ["Wav2Vec2ForCTC"],
            })

    def test_remote_code_pickle_and_loader_options_are_rejected(self):
        invalid = (
            {
                "trust_remote_code": True,
            },
            {
                "use_safetensors": False,
            },
            {
                "model_kwargs": {
                    "device_map": "auto",
                },
            },
            {
                "processor_kwargs": {
                    "padding": True,
                },
            },
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                TransformersVADConfig(**values)

    def test_multiclass_checkpoint_requires_explicit_speech_label(self):
        wrapper = TransformersVADForVoiceActivityDetection(TransformersVADConfig())
        wrapper.native_config = SimpleNamespace(id2label={
            0: "music",
            1: "noise",
            2: "silence",
        })

        with self.assertRaisesRegex(ValueError, "speech_class_id"):
            wrapper._speech_class_id(3)

    def test_single_logit_and_negative_labels_resolve_the_positive_class(self):
        wrapper = TransformersVADForVoiceActivityDetection(TransformersVADConfig())
        wrapper.native_config = SimpleNamespace(id2label={
            0: "non-speech",
            1: "speech",
        })

        self.assertEqual(wrapper._speech_class_id(1), 0)
        self.assertEqual(wrapper._speech_class_id(2), 1)

    def test_frame_geometry_prefers_checkpoint_logit_stride(self):
        wrapper = TransformersVADForVoiceActivityDetection(TransformersVADConfig())
        wrapper.native_config = SimpleNamespace(inputs_to_logits_ratio=320)

        self.assertEqual(
            wrapper._frame_geometry(
                frame_count=4,
                waveform_samples=16_000,
            ),
            (320, 320),
        )

    def test_training_audio_batch_cannot_be_empty(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            TransformersVADForVoiceActivityDetection._audio_batch([])


if __name__ == "__main__":
    unittest.main()
