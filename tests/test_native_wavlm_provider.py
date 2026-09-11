from __future__ import annotations

import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from voicehub import AutoConfig, AutoModelForSpeechRecognition, get_model_spec
from voicehub.checkpointing import save_safetensors
from voicehub.models.asr_wavlm import NativeWavLMTrainingAdapter, WavLMASRConfig, WavLMForSpeechRecognition
from voicehub.models.asr_wavlm.native import WavLMConfig, WavLMForCTC, resolve_wavlm_artifacts
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.specs import get_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_artifact(root: Path):
    torch.manual_seed(91)
    config = WavLMConfig(
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
        conv_bias=True,
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
        num_buckets=8,
        max_bucket_distance=16,
        do_stable_layer_norm=True,
        apply_spec_augment=True,
        mask_time_prob=0.5,
        mask_time_length=2,
        mask_time_min_masks=1,
        mask_feature_prob=0.0,
        mask_feature_min_masks=0,
        ctc_loss_reduction="sum",
        pad_token_id=4,
        bos_token_id=1,
        eos_token_id=2,
    )
    reference = WavLMForCTC(config)
    (root / "config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    (root / "vocab.json").write_text(
        json.dumps({
            "|": 0,
            "A": 1,
            "B": 2,
            "<unk>": 3,
            "<pad>": 4,
            "'": 5,
        }),
        encoding="utf-8",
    )
    (root / "added_tokens.json").write_text(
        json.dumps({
            "<s>": 6,
            "</s>": 7,
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
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }),
        encoding="utf-8",
    )
    (root / "preprocessor_config.json").write_text(
        json.dumps({
            "do_normalize": True,
            "feature_size": 1,
            "padding_side": "right",
            "padding_value": 0.0,
            "return_attention_mask": True,
            "sampling_rate": 16_000,
        }),
        encoding="utf-8",
    )
    save_safetensors(
        reference.state_dict(),
        root / "model.safetensors",
    )
    return config, reference


class NativeWavLMProviderTests(unittest.TestCase):

    def test_provider_import_is_lazy_and_dependency_free(self):
        code = """
import json
import sys
import voicehub.models.asr_wavlm
names = ("torch", "transformers", "tokenizers", "safetensors", "torchaudio")
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
                "torchaudio": False,
            },
        )

    def test_public_constructor_keeps_the_uniform_asr_signature(self):
        signature = inspect.signature(WavLMForSpeechRecognition)
        self.assertEqual(
            tuple(signature.parameters),
            (
                "config",
                "model_path",
                "device",
                "lazy_load",
                "token",
                "kwargs",
            ),
        )

    def test_default_checkpoint_uses_the_immutable_safe_conversion(self):

        class ResolutionReached(RuntimeError):
            pass

        wrapper = WavLMForSpeechRecognition(device="cpu")
        with patch(
                "voicehub.models.asr_wavlm.native.artifacts."
                "resolve_wavlm_artifacts",
                side_effect=ResolutionReached,
        ) as resolver:
            with self.assertRaises(ResolutionReached):
                wrapper._load_pretrained_model()

        self.assertEqual(
            resolver.call_args.kwargs["revision"],
            "561f43a6081f379876b6633a38526aabe140ba3b",
        )

    def test_sharded_artifacts_are_coherent_and_paths_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "vocab.json").write_text(
                '{"<pad>": 0}',
                encoding="utf-8",
            )
            for name in (
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
            ):
                (root / name).touch()
            index = root / "model.safetensors.index.json"
            index.write_text(
                json.dumps({
                    "weight_map": {
                        "wavlm.layer.one": "model-00001-of-00002.safetensors",
                        "wavlm.layer.two": "model-00002-of-00002.safetensors",
                    },
                }),
                encoding="utf-8",
            )

            artifacts = resolve_wavlm_artifacts(root)
            self.assertTrue(artifacts.is_sharded)
            self.assertEqual(artifacts.checkpoint, index.resolve())

            for unsafe_name in (
                    "../outside.safetensors",
                    "..\\outside.safetensors",
            ):
                with self.subTest(unsafe_name=unsafe_name):
                    index.write_text(
                        json.dumps({
                            "weight_map": {
                                "wavlm.layer": unsafe_name,
                            },
                        }),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "Unsafe"):
                        resolve_wavlm_artifacts(root)

            for unsafe_filename in (
                    "../weights.safetensors",
                    "..\\weights.safetensors",
                    "/tmp/weights.safetensors",
            ):
                with (
                        self.subTest(unsafe_filename=unsafe_filename),
                        self.assertRaisesRegex(ValueError, "safe"),
                ):
                    resolve_wavlm_artifacts(
                        root,
                        checkpoint_filename=unsafe_filename,
                    )

    def test_local_checkpoint_training_inference_and_export_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference = _tiny_artifact(root)
            wrapper = WavLMForSpeechRecognition(
                WavLMASRConfig(name_or_path=root),
                device="cpu",
            )
            inference = wrapper.transcribe(
                torch.randn(30),
                sampling_rate=16_000,
            )
            self.assertEqual(
                inference.metadata["backend"],
                "voicehub-native",
            )
            self.assertEqual(
                inference.metadata["architecture"],
                "wavlm-ctc",
            )

            adapter = AutoTrainingAdapter.from_model(wrapper)
            self.assertIsInstance(adapter, NativeWavLMTrainingAdapter)
            context = adapter.create_training_context({
                "audio": torch.randn(30),
                "sampling_rate": 16_000,
                "text": "AB",
                "generator": torch.Generator().manual_seed(7),
            })
            output = adapter.execute_training_phase(context)
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(
                wrapper.model.wavlm.encoder.layers[0].attention.gru_rel_pos_linear.weight.grad)

            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(
                    wrapper.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )

            export = root / "export"
            adapter.save_pretrained(export)
            exported = json.loads((export / "config.json").read_text(encoding="utf-8"), )
            self.assertEqual(exported["model_type"], "asr_wavlm")
            self.assertEqual(exported['architectures'], ["WavLMForCTC"])
            self.assertEqual(
                exported["voicehub_checkpoint_format"],
                "native-wavlm-ctc-v1",
            )

            auto_config = AutoConfig.from_pretrained(export)
            restored = AutoModelForSpeechRecognition.from_pretrained(
                export,
                config=auto_config,
                device="cpu",
            )
            restored.load_for_training()
            self.assertIsInstance(auto_config, WavLMASRConfig)
            self.assertIsInstance(restored, WavLMForSpeechRecognition)
            for name, expected in wrapper.model.state_dict().items():
                torch.testing.assert_close(
                    restored.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )

    def test_local_sharded_checkpoint_loads_strictly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference = _tiny_artifact(root)
            state = reference.state_dict()
            tensor_names = tuple(sorted(state))
            midpoint = len(tensor_names) // 2
            shards = (
                (
                    "model-00001-of-00002.safetensors",
                    tensor_names[:midpoint],
                ),
                (
                    "model-00002-of-00002.safetensors",
                    tensor_names[midpoint:],
                ),
            )
            weight_map = {}
            for filename, names in shards:
                save_safetensors(
                    {name: state[name]
                     for name in names},
                    root / filename,
                )
                weight_map.update(dict.fromkeys(names, filename))
            (root / "model.safetensors").unlink()
            (root / "model.safetensors.index.json").write_text(
                json.dumps({
                    "weight_map": weight_map,
                }),
                encoding="utf-8",
            )

            wrapper = WavLMForSpeechRecognition(
                WavLMASRConfig(name_or_path=root),
                device="cpu",
                lazy_load=False,
            )

            self.assertTrue(wrapper.artifacts.is_sharded)
            for name, expected in state.items():
                torch.testing.assert_close(
                    wrapper.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )

    def test_registry_and_training_spec_are_fully_native(self):
        model_spec = get_model_spec("asr_wavlm")
        training_spec = get_training_spec("asr_wavlm")

        self.assertEqual(
            model_spec.module,
            "voicehub.models.asr_wavlm.modeling",
        )
        self.assertEqual(
            model_spec.config_module,
            "voicehub.models.asr_wavlm.configuration",
        )
        self.assertTrue(model_spec.is_voicehub_native)
        self.assertEqual(model_spec.architecture, "wavlm")
        self.assertEqual(
            training_spec.source_entrypoints,
            ("voicehub.models.asr_wavlm.native.WavLMForCTC", ),
        )

    def test_external_runtime_and_pickle_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "never executes"):
            WavLMASRConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "Safetensors only"):
            WavLMASRConfig(use_safetensors=False)
        with self.assertRaisesRegex(ValueError, "does not delegate"):
            WavLMASRConfig(model_kwargs={"device_map": "auto"})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pytorch_model.bin").touch()
            with self.assertRaisesRegex(ValueError, "Safetensors"):
                WavLMForSpeechRecognition(
                    root / "pytorch_model.bin",
                    device="cpu",
                    lazy_load=False,
                )


if __name__ == "__main__":
    unittest.main()
