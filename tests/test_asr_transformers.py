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
from voicehub.models.asr_hubert import HubertForSpeechRecognition
from voicehub.models.asr_moonshine import MoonshineForSpeechRecognition
from voicehub.models.asr_transformers import TransformersASRConfig, TransformersASRForSpeechRecognition
from voicehub.models.asr_transformers.training_asr_transformers import TransformersASRTrainingAdapter
from voicehub.models.asr_wav2vec2 import Wav2Vec2ForSpeechRecognition
from voicehub.models.asr_wav2vec2.native import Wav2Vec2Config, Wav2Vec2ForCTC
from voicehub.models.asr_wavlm import WavLMForSpeechRecognition
from voicehub.models.asr_whisper_native import WhisperForSpeechRecognition
from voicehub.training.auto import AutoTrainingAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_tiny_wav2vec2_artifact(root: Path):
    torch.manual_seed(97)
    config = Wav2Vec2Config(
        vocab_size=8,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        final_dropout=0.0,
        layerdrop=0.0,
        conv_dim=(4, 8),
        conv_stride=(2, 2),
        conv_kernel=(4, 2),
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
        apply_spec_augment=False,
        mask_time_prob=0.0,
        mask_time_min_masks=0,
        mask_feature_prob=0.0,
        mask_feature_min_masks=0,
        ctc_loss_reduction="sum",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    reference = Wav2Vec2ForCTC(config)
    values = config.to_dict()
    values['architectures'] = ["Wav2Vec2ForCTC"]
    (root / "config.json").write_text(
        json.dumps(values),
        encoding="utf-8",
    )
    (root / "vocab.json").write_text(
        json.dumps({
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "|": 4,
            "A": 5,
            "B": 6,
            "'": 7,
        }),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({
            "do_lower_case": True,
            "word_delimiter_token": "|",
        }),
        encoding="utf-8",
    )
    (root / "special_tokens_map.json").write_text(
        json.dumps({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
        }),
        encoding="utf-8",
    )
    (root / "preprocessor_config.json").write_text(
        json.dumps({
            "do_normalize": True,
            "feature_size": 1,
            "padding_value": 0.0,
            "return_attention_mask": False,
            "sampling_rate": 16_000,
        }),
        encoding="utf-8",
    )
    save_safetensors(
        reference.state_dict(),
        root / "model.safetensors",
    )
    return config, reference


def _shard_wav2vec2_artifact(root: Path, model: Wav2Vec2ForCTC) -> None:
    state = model.state_dict()
    names = tuple(state)
    split = len(names) // 2
    shard_names = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    groups = (names[:split], names[split:])
    for shard_name, group in zip(shard_names, groups):
        save_safetensors(
            {name: state[name]
             for name in group},
            root / shard_name,
        )
    weight_map = {name: shard_name for shard_name, group in zip(shard_names, groups) for name in group}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {},
            "weight_map": weight_map
        }),
        encoding="utf-8",
    )
    (root / "model.safetensors").unlink()


class _DeterministicCTCModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_values, attention_mask=None):
        del attention_mask
        token_ids = (5, 5, 0, 4, 6, 6, 0)
        logits = torch.full(
            (input_values.shape[0], len(token_ids), 8),
            -10.0,
            dtype=input_values.dtype,
            device=input_values.device,
        )
        for index, token_id in enumerate(token_ids):
            logits[:, index, token_id] = 10.0 + self.anchor
        return SimpleNamespace(
            logits=logits,
            input_lengths=torch.full(
                (input_values.shape[0], ),
                len(token_ids),
                dtype=torch.long,
                device=input_values.device,
            ),
        )


class NativeTransformersASRConfigTests(unittest.TestCase):

    def test_public_import_is_free_of_external_model_runtimes(self):
        code = """
import json
import sys
import voicehub.models.asr_transformers
names = ("transformers", "safetensors", "tokenizers", "torchaudio")
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
                "transformers": False,
                "safetensors": False,
                "tokenizers": False,
                "torchaudio": False,
            },
        )

    def test_native_controls_reject_delegation_and_unsafe_artifacts(self):
        self.assertEqual(
            TransformersASRConfig().architecture_family,
            "auto",
        )
        for values, message in (
            ({"architecture_family": "rnnt"}, "architecture_family"),
            ({"architecture_family": "tdt"}, "architecture_family"),
            ({"trust_remote_code": True}, "never executes"),
            ({"use_safetensors": False}, "Safetensors only"),
            ({"config_name_or_path": "other/config"}, "coherent"),
            ({"processor_name_or_path": "other/processor"}, "coherent"),
            ({"model_kwargs": {"device_map": "auto"}}, "does not delegate"),
            ({"processor_kwargs": {"padding": True}}, "does not delegate"),
            ({"pipeline_kwargs": {"chunk": 10}}, "does not delegate"),
            ({"checkpoint_filename": "../weights.safetensors"}, "safe"),
            ({"tokenizer_filename": r"assets\\tokenizer.json"}, "safe"),
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    TransformersASRConfig(**values)

    def test_serialized_runtime_secrets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            TransformersASRConfig(token="do-not-serialize")
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            TransformersASRConfig(inference_config={"nested": {
                "api_key": "do-not-serialize",
            }})


class NativeTransformersASRDispatchTests(unittest.TestCase):

    def test_public_compatibility_class_has_no_external_loading_hooks(self):
        for name in (
                "_auto_model_class",
                "_ensure_pipeline",
                "_legacy_transformers_module",
                "_load_native_model",
                "_model_load_kwargs",
                "_normalize_pipeline_output",
                "_pipeline_call_options",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(TransformersASRForSpeechRecognition, name), )

    def test_dispatch_table_is_closed_to_verified_native_families(self):
        cases = {
            "whisper": "whisper",
            "asr_whisper": "whisper",
            "wav2vec2": "wav2vec2",
            "asr_wav2vec2": "wav2vec2",
            "hubert": "hubert",
            "asr_hubert": "hubert",
            "wavlm": "wavlm",
            "asr_wavlm": "wavlm",
            "moonshine": "moonshine",
            "asr_moonshine": "moonshine",
        }
        for model_type, expected in cases.items():
            with self.subTest(model_type=model_type):
                self.assertEqual(
                    TransformersASRForSpeechRecognition._native_model_type_from_config({
                        "model_type": model_type,
                        'architectures': [],
                    }),
                    expected,
                )

        with self.assertRaisesRegex(ValueError, "not an ASR head"):
            (
                TransformersASRForSpeechRecognition._native_model_type_from_config({
                    "model_type":
                    "wav2vec2",
                    'architectures': ["Wav2Vec2ForAudioFrameClassification"],
                }))
        with self.assertRaisesRegex(ValueError, "cannot dispatch"):
            (
                TransformersASRForSpeechRecognition._native_model_type_from_config({
                    "model_type":
                    "parakeet_tdt",
                    'architectures': ["ParakeetForTDT"],
                }))

    def test_delegate_factory_selects_voicehub_wrappers_only(self):
        wrapper = TransformersASRForSpeechRecognition(device="cpu")
        cases = {
            "whisper": WhisperForSpeechRecognition,
            "wav2vec2": Wav2Vec2ForSpeechRecognition,
            "hubert": HubertForSpeechRecognition,
            "wavlm": WavLMForSpeechRecognition,
            "moonshine": MoonshineForSpeechRecognition,
        }
        for model_type, expected_type in cases.items():
            with self.subTest(model_type=model_type):
                delegate = wrapper._build_native_delegate(
                    model_type,
                    source="publisher/checkpoint",
                    revision="immutable-revision",
                )
                self.assertIsInstance(delegate, expected_type)
                self.assertEqual(
                    delegate.config.revision,
                    "immutable-revision",
                )
                self.assertFalse(delegate.is_loaded)

    def test_requested_family_must_match_checkpoint_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_tiny_wav2vec2_artifact(root)
            wrapper = TransformersASRForSpeechRecognition(
                TransformersASRConfig(
                    name_or_path=root,
                    architecture_family="speech-seq2seq",
                ),
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "not requested family"):
                wrapper.load_for_training()


class NativeTransformersASRRuntimeTests(unittest.TestCase):

    def test_single_safetensors_inference_training_backward_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference = _write_tiny_wav2vec2_artifact(root)
            wrapper = TransformersASRForSpeechRecognition(
                TransformersASRConfig(name_or_path=root),
                device="cpu",
            )
            adapter = AutoTrainingAdapter.from_model(wrapper)
            self.assertIsInstance(adapter, TransformersASRTrainingAdapter)

            context = adapter.create_training_context({
                "audio": torch.linspace(-1.0, 1.0, 30),
                "sampling_rate": 16_000,
                "text": "AB",
            })
            training_output = adapter.execute_training_phase(context)
            self.assertEqual(wrapper.native_model_type, "wav2vec2")
            self.assertEqual(wrapper.architecture_family, "ctc")
            self.assertIsInstance(
                wrapper._delegate,
                Wav2Vec2ForSpeechRecognition,
            )
            self.assertTrue(torch.isfinite(training_output.loss))
            training_output.loss.backward()
            self.assertIsNotNone(wrapper.model.lm_head.weight.grad)

            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(
                    wrapper.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )

            wrapper.model = _DeterministicCTCModel()
            output = wrapper.transcribe(
                torch.linspace(-0.5, 0.5, 30),
                sampling_rate=16_000,
                return_timestamps=True,
            )
            self.assertEqual(output.text, "a b")
            self.assertEqual(output.metadata["backend"], "voicehub-native")
            self.assertEqual(
                output.metadata["architecture"],
                "wav2vec2-ctc",
            )

            # Restore the differentiable graph before writing the artifact.
            wrapper.model = reference
            wrapper._delegate.model = reference
            export = root / "export"
            adapter.save_pretrained(export)
            self.assertTrue((export / "model.safetensors").is_file())
            reloaded = TransformersASRForSpeechRecognition(
                TransformersASRConfig(name_or_path=export),
                device="cpu",
            )
            reloaded.load_for_training()
            self.assertEqual(reloaded.native_model_type, "wav2vec2")
            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(
                    reloaded.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )

    def test_sharded_and_direct_safetensors_use_the_same_native_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference = _write_tiny_wav2vec2_artifact(root)
            _shard_wav2vec2_artifact(root, reference)
            sharded = TransformersASRForSpeechRecognition(
                TransformersASRConfig(name_or_path=root),
                device="cpu",
            )
            sharded.load_for_training()
            self.assertTrue(sharded.artifacts.is_sharded)

            # A direct file remains coherent with sibling config/processor
            # assets and does not need a separate config source.
            single = root / "fine-tuned.safetensors"
            save_safetensors(reference.state_dict(), single)
            direct = TransformersASRForSpeechRecognition(
                TransformersASRConfig(name_or_path=single),
                device="cpu",
            )
            direct.load_for_training()
            self.assertFalse(direct.artifacts.is_sharded)
            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(
                    direct.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )

    def test_generic_provider_rejects_pickle_and_optimized_formats(self):
        for filename in (
                "model.bin",
                "model.pt",
                "model.gguf",
                "model.onnx",
        ):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(
                    (FileNotFoundError, ValueError),
                        "path was not found|fine-tuning|Safetensors",
                ):
                    wrapper = TransformersASRForSpeechRecognition(
                        TransformersASRConfig(name_or_path=f"/tmp/{filename}", ))
                    wrapper.load_for_training()


if __name__ == "__main__":
    unittest.main()
