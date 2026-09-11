from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.asr_medasr.configuration import MedASRASRConfig
from voicehub.models.asr_medasr.modeling import MedASRForSpeechRecognition
from voicehub.models.asr_medasr.native.artifacts import _require_coherent_snapshot, resolve_medasr_artifacts
from voicehub.models.asr_medasr.native.checkpoint import (
    MedASRCheckpointAdapter,
    medasr_header_fingerprint,
    native_medasr_tensor_dtypes,
    native_medasr_tensor_shapes,
)
from voicehub.models.asr_medasr.native.configuration import MedASRConfig
from voicehub.models.asr_medasr.native.frontend import MedASRFeatureExtractor
from voicehub.models.asr_medasr.native.metadata import MEDASR_CHECKPOINT, MEDASR_MODEL_REVISION
from voicehub.models.asr_medasr.native.modeling import MedASRForCTC
from voicehub.models.asr_medasr.native.processing import MedASRProcessor
from voicehub.models.asr_medasr.native.tokenization import MedASRTokenizer
from voicehub.models.asr_medasr.training_asr_medasr import NativeMedASRTrainingAdapter
from voicehub.processing.waveform import save_pcm_wave
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.specs import get_training_spec

ROOT = Path(__file__).parents[1]
ARCHITECTURE_ROOT = ROOT / 'voicehub/models/asr_medasr/native'
PROVIDER_ROOT = ROOT / "voicehub" / "models" / "asr_medasr"


def _tiny_config() -> MedASRConfig:
    return MedASRConfig(
        variant="custom",
        vocab_size=12,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        conv_kernel_size=4,
        subsampling_conv_channels=8,
        num_mel_bins=8,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=128,
        feature_fft_size=32,
        feature_window_length=24,
        feature_hop_length=8,
        feature_lower_hertz=20.0,
        feature_upper_hertz=7_000.0,
    )


def _tokenizer_document() -> dict:
    vocabulary = [
        ["<epsilon>", 0.0],
        ["<s>", 0.0],
        ["</s>", 0.0],
        ["<unk>", 0.0],
        ["▁", -2.0],
        ["a", -1.0],
        ["b", -1.0],
        ["c", -1.0],
        ["▁a", -0.1],
        ["▁b", -0.1],
        ["ab", -0.2],
        ["▁ab", -0.05],
    ]
    return {
        "version":
        "1.0",
        "normalizer":
        None,
        "pre_tokenizer": {
            "type":
            "Sequence",
            "pretokenizers": [
                {
                    "type": "WhitespaceSplit"
                },
                {
                    "type": "Metaspace",
                    "replacement": "▁",
                    "prepend_scheme": "always",
                    "split": True,
                },
            ],
        },
        "decoder": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
            "split": True,
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{
                "Sequence": {
                    "id": "A",
                    "type_id": 0
                }
            }],
            "pair": [],
            "special_tokens": {},
        },
        "added_tokens": [{
            "id": token_id,
            "content": spelling,
            "special": True,
        } for spelling, token_id in (
            ("<epsilon>", 0),
            ("<s>", 1),
            ("</s>", 2),
            ("<unk>", 3),
        )],
        "model": {
            "type": "Unigram",
            "unk_id": 3,
            "vocab": vocabulary,
            "byte_fallback": False,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_tiny_artifact(root: Path) -> tuple[MedASRConfig, MedASRForCTC]:
    config = _tiny_config()
    torch.manual_seed(13)
    model = MedASRForCTC(config)
    save_safetensors(
        model.state_dict(),
        root / "model.safetensors",
        metadata={"format": "test-medasr"},
    )
    values = config.to_dict()
    values['architectures'] = ["LasrForCTC"]
    _write_json(root / "config.json", values)
    _write_json(root / "tokenizer.json", _tokenizer_document())
    _write_json(
        root / "tokenizer_config.json",
        {
            "eos_token": "</s>",
            "pad_token": "<epsilon>",
            "tokenizer_class": "LasrTokenizer",
            "unk_token": "<unk>",
        },
    )
    _write_json(
        root / "preprocessor_config.json",
        {
            "feature_extractor_type": "LasrFeatureExtractor",
            "feature_size": config.num_mel_bins,
            "hop_length": config.feature_hop_length,
            "n_fft": config.feature_fft_size,
            "padding_side": "right",
            "padding_value": 0.0,
            "processor_class": "LasrProcessor",
            "return_attention_mask": True,
            "sampling_rate": config.sampling_rate,
            "win_length": config.feature_window_length,
        },
    )
    _write_json(
        root / "processor_config.json",
        {"processor_class": "LasrProcessor"},
    )
    return config, model


class MedASRArchitectureTests(unittest.TestCase):

    def test_package_discovery_does_not_import_torch(self):
        command = (
            "import sys; "
            "import voicehub.models.asr_medasr.native as architecture; "
            "import voicehub.models.asr_medasr as provider; "
            "assert 'torch' not in sys.modules; "
            "assert 'MedASRForCTC' in architecture.__all__; "
            "assert 'MedASRForSpeechRecognition' in provider.__all__; "
            "assert 'NativeMedASRTrainingAdapter' in provider.__all__")
        subprocess.run(
            [sys.executable, "-c", command],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_official_graph_matches_audited_header_inventory(self):
        config = MedASRConfig()
        shapes = native_medasr_tensor_shapes(config)
        dtypes = native_medasr_tensor_dtypes(config)
        inventory = {name: (dtypes[name], shape) for name, shape in shapes.items()}
        self.assertEqual(len(inventory), MEDASR_CHECKPOINT["tensors"])
        self.assertEqual(
            sum(math.prod(shape) for shape in shapes.values()),
            MEDASR_CHECKPOINT["parameters"],
        )
        self.assertEqual(
            sum(dtype == "I64" for dtype in dtypes.values()),
            MEDASR_CHECKPOINT["dtype_counts"]["I64"],
        )
        self.assertEqual(
            medasr_header_fingerprint(inventory),
            MEDASR_CHECKPOINT["header_fingerprint"],
        )

    def test_frontend_is_deterministic_and_tracks_right_padding(self):
        config = _tiny_config()
        frontend = MedASRFeatureExtractor(config)
        first = torch.linspace(-0.5, 0.5, 240)
        second = torch.linspace(0.25, -0.25, 176)
        outputs = frontend((first, second))
        self.assertEqual(outputs["input_features"].shape, (2, 28, 8))
        self.assertEqual(outputs["input_features"].dtype, torch.float32)
        self.assertEqual(
            outputs["attention_mask"].sum(-1).tolist(),
            [28, 20],
        )
        repeated = frontend((first, second))
        torch.testing.assert_close(
            outputs["input_features"],
            repeated["input_features"],
            rtol=0.0,
            atol=0.0,
        )

    def test_full_ctc_backward_and_gradient_checkpointing(self):
        config = _tiny_config()
        model = MedASRForCTC(config)
        model.train()
        model.gradient_checkpointing_enable()
        features = torch.randn(2, 28, config.num_mel_bins)
        mask = torch.ones(2, 28, dtype=torch.bool)
        mask[1, 22:] = False
        labels = torch.tensor([
            [8, 6, 0],
            [9, 7, 5],
        ])
        output = model(
            features,
            attention_mask=mask,
            labels=labels,
            output_hidden_states=True,
        )
        self.assertEqual(output.logits.shape, (2, 4, config.vocab_size))
        self.assertEqual(output.encoded_lengths.tolist(), [4, 3])
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for parameter in (
                model.encoder.subsampler.dense_0.weight,
                model.encoder.layers[0].self_attn.q_proj.weight,
                model.ctc_head.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_strict_safetensors_round_trip_into_meta_graph(self):
        config = _tiny_config()
        torch.manual_seed(7)
        source = MedASRForCTC(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_safetensors(source.state_dict(), path)
            target = MedASRForCTC(config, initialize=False)
            with SafeTensorReader(path) as reader:
                report = MedASRCheckpointAdapter().load_assign_streaming(
                    target,
                    reader,
                    config,
                    device="cpu",
                    dtype=torch.float32,
                )
            self.assertTrue(report.is_compatible)
            for name, tensor in source.state_dict().items():
                torch.testing.assert_close(
                    tensor,
                    target.state_dict()[name],
                    rtol=0.0,
                    atol=0.0,
                )

    def test_strict_loader_rejects_missing_tensor(self):
        config = _tiny_config()
        model = MedASRForCTC(config)
        state = dict(model.state_dict())
        state.pop("ctc_head.bias")
        target = MedASRForCTC(config, initialize=False)
        with self.assertRaises(CheckpointCompatibilityError):
            MedASRCheckpointAdapter().load_assign_streaming(
                target,
                state,
                config,
                device="cpu",
            )

    def test_strict_loader_rejects_invalid_dtypes_before_assignment(self):
        config = _tiny_config()
        source = MedASRForCTC(config)
        for label, invalid in {
                "integer": torch.zeros_like(
                    source.ctc_head.weight,
                    dtype=torch.int64,
                ),
                "complex": source.ctc_head.weight.to(torch.complex64),
                "sparse": source.ctc_head.weight.to_sparse(),
                "quantized": torch.quantize_per_tensor(
                    source.ctc_head.weight,
                    scale=0.1,
                    zero_point=0,
                    dtype=torch.qint8,
                ),
        }.items():
            with self.subTest(validation=label):
                state = dict(source.state_dict())
                state["ctc_head.weight"] = invalid
                target = MedASRForCTC(config, initialize=False)
                with self.assertRaisesRegex(
                        CheckpointCompatibilityError,
                        "dtypes/layouts",
                ):
                    MedASRCheckpointAdapter().load_assign_streaming(
                        target,
                        state,
                        config,
                        device="cpu",
                    )
                self.assertTrue(all(tensor.device.type == "meta" for tensor in target.state_dict().values()))


class MedASRTokenizerTests(unittest.TestCase):

    def test_unigram_encoding_and_ctc_collapse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_json(root / "tokenizer.json", _tokenizer_document())
            _write_json(
                root / "tokenizer_config.json",
                {
                    "eos_token": "</s>",
                    "pad_token": "<epsilon>",
                    "unk_token": "<unk>",
                },
            )
            tokenizer = MedASRTokenizer.from_files(
                root / "tokenizer.json",
                tokenizer_config=root / "tokenizer_config.json",
                expected_vocabulary_size=12,
            )
            self.assertEqual(tokenizer.encode("ab"), (11, ))
            decoded = tokenizer.decode_ctc([11, 11, 0, 8, 8, 6])
            self.assertEqual(decoded.text, "ab ab")
            self.assertEqual(
                [(span.start_offset, span.end_offset) for span in decoded.tokens],
                [(0, 2), (3, 5), (5, 6)],
            )

    def test_tokenizer_rejects_changed_pipeline(self):
        document = _tokenizer_document()
        document["normalizer"] = {"type": "NFC"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            _write_json(path, document)
            with self.assertRaisesRegex(ValueError, "no normalizer"):
                MedASRTokenizer.from_files(path)


class MedASRProviderTests(unittest.TestCase):

    def test_remote_artifacts_require_an_immutable_coherent_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.touch()
            with (
                    patch(
                        "voicehub.models.asr_medasr.native.artifacts."
                        "resolve_pretrained_file",
                        return_value=config,
                    ),
                    patch(
                        "voicehub.models.asr_medasr.native.artifacts."
                        "get_cached_hugging_face_commit",
                        return_value=None,
                    ),
            ):
                with self.assertRaisesRegex(RuntimeError, "immutable"):
                    resolve_medasr_artifacts(
                        "example/medasr",
                        revision="main",
                    )

            other = root / "other"
            other.mkdir()
            with self.assertRaisesRegex(RuntimeError, "one immutable snapshot"):
                _require_coherent_snapshot(
                    root,
                    root / "config.json",
                    other / "model.safetensors",
                )

        self.assertEqual(len(MEDASR_MODEL_REVISION), 40)

    def test_config_fails_closed_for_external_runtime_options(self):
        with self.assertRaisesRegex(ValueError, "never executes"):
            MedASRASRConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "Safetensors"):
            MedASRASRConfig(use_safetensors=False)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            MedASRASRConfig(model_kwargs={"device_map": "auto"})

    def test_local_raw_inference_training_and_export_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_tiny_artifact(root)
            wrapper = MedASRForSpeechRecognition(
                model_path=root,
                device="cpu",
                lazy_load=False,
            )
            waveform = torch.sin(torch.linspace(0.0, 10.0, 320), )
            output = wrapper.transcribe(
                waveform,
                sampling_rate=16_000,
                language="en",
            )
            self.assertEqual(output.language, "en")
            self.assertEqual(
                output.metadata["backend"],
                "voicehub-native",
            )
            prepared = wrapper.prepare_training_inputs(
                {
                    "audio": [waveform, waveform[:240]],
                    "text": ["ab", "a"],
                    "sampling_rate": 16_000,
                },
                phase="train",
            )
            training_output = wrapper.model(**prepared)
            self.assertTrue(torch.isfinite(training_output.loss))
            training_output.loss.backward()
            self.assertIsNotNone(wrapper.model.encoder.layers[0].conv.depthwise_conv.weight.grad)

            path = save_pcm_wave(
                root / "padded.wav",
                torch.cat((torch.zeros(160), torch.ones(160))),
                8_000,
            )
            file_prepared = wrapper.prepare_training_inputs(
                {
                    "audio": str(path),
                    "audio_lengths": 160,
                    "sampling_rate": 8_000,
                    "text": "a",
                },
                phase="train",
            )
            tensor_prepared = wrapper.prepare_training_inputs(
                {
                    "audio": torch.zeros(160),
                    "sampling_rate": 8_000,
                    "text": "a",
                },
                phase="train",
            )
            torch.testing.assert_close(
                file_prepared["input_features"],
                tensor_prepared["input_features"],
            )

            export = root / "export"
            wrapper.export_native_pretrained(export)
            for filename in (
                    "MODEL_TERMS_NOTICE",
                    "NOTICE",
                    "THIRD_PARTY_LICENSE",
                    "config.json",
                    "model.safetensors",
                    "preprocessor_config.json",
                    "tokenizer.json",
            ):
                self.assertTrue((export / filename).is_file(), filename)
            reloaded = MedASRForSpeechRecognition(
                model_path=export,
                device="cpu",
                lazy_load=False,
            )
            repeated = reloaded.transcribe(
                waveform,
                sampling_rate=16_000,
            )
            self.assertEqual(repeated.text, output.text)

    def test_export_validates_before_creating_destination(self):
        config = _tiny_config()
        wrapper = object.__new__(MedASRForSpeechRecognition)
        wrapper.model = MedASRForCTC(config)
        wrapper.native_config = config
        wrapper.medasr_processor = SimpleNamespace()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = dict(wrapper.model.state_dict())
            missing.pop("ctc_head.bias")
            variants = {
                "missing": missing,
                "shape": {
                    **wrapper.model.state_dict(),
                    "ctc_head.bias":
                    torch.zeros(wrapper.model.ctc_head.bias.numel() + 1, ),
                },
                "complex": {
                    **wrapper.model.state_dict(),
                    "ctc_head.bias":
                    wrapper.model.ctc_head.bias.to(torch.complex64, ),
                },
                "sparse": {
                    **wrapper.model.state_dict(),
                    "ctc_head.weight": wrapper.model.ctc_head.weight.to_sparse(),
                },
                "quantized": {
                    **wrapper.model.state_dict(),
                    "ctc_head.weight":
                    torch.quantize_per_tensor(
                        wrapper.model.ctc_head.weight,
                        scale=0.1,
                        zero_point=0,
                        dtype=torch.qint8,
                    ),
                },
            }
            for label, state in variants.items():
                with self.subTest(validation=label):
                    destination = root / f"rejected-{label}"
                    with (
                            patch.object(
                                wrapper.model,
                                "state_dict",
                                return_value=state,
                            ),
                            self.assertRaises((TypeError, ValueError)),
                    ):
                        wrapper.export_native_pretrained(destination)
                    self.assertFalse(destination.exists())

    def test_training_adapter_does_not_create_partial_export_directory(self):
        adapter = object.__new__(NativeMedASRTrainingAdapter)
        adapter.model = SimpleNamespace(
            export_native_pretrained=Mock(side_effect=ValueError("invalid runtime state"), ))
        adapter.setup = lambda: adapter
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "rejected"
            with self.assertRaisesRegex(ValueError, "invalid runtime state"):
                adapter.save_pretrained(destination)
            self.assertFalse(destination.exists())

    def test_shared_training_registry_selects_native_full_model_adapter(self):
        wrapper = MedASRForSpeechRecognition(
            lazy_load=True,
            device="cpu",
        )
        adapter = AutoTrainingAdapter.from_model(wrapper)
        spec = get_training_spec("asr_medasr")

        self.assertIsInstance(adapter, NativeMedASRTrainingAdapter)
        self.assertEqual(spec.module_paths, ("model", ))
        self.assertEqual(
            spec.component_paths,
            (
                "model.encoder",
                "model.ctc_head",
            ),
        )
        self.assertTrue(all(entrypoint.startswith("voicehub.") for entrypoint in spec.source_entrypoints))

    def test_unsupported_semantics_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "English-only"):
            MedASRForSpeechRecognition._validate_inference_request(
                language="fr",
                task="transcribe",
                return_timestamps=False,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                num_beams=None,
                max_new_tokens=None,
                hotwords=None,
            )
        with self.assertRaisesRegex(ValueError, "timestamp"):
            MedASRForSpeechRecognition._validate_inference_request(
                language="en",
                task="transcribe",
                return_timestamps=True,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                num_beams=None,
                max_new_tokens=None,
                hotwords=None,
            )

    def test_official_recipe_metadata_is_truthful(self):
        self.assertTrue(NativeMedASRTrainingAdapter.supports_custom_recipe)
        self.assertIn(
            "medasr",
            NativeMedASRTrainingAdapter.native_export_semantics,
        )
        source = json.loads((ARCHITECTURE_ROOT / "SOURCE.json").read_text(encoding="utf-8", ))
        self.assertEqual(
            source["recipe"]["fine_tuning_scope"],
            "full-model",
        )
        self.assertEqual(source["recipe"]["learning_rate"], 3e-5)
        self.assertEqual(source["recipe"]["warmup_steps"], 300)


class MedASRPolicyTests(unittest.TestCase):

    def test_native_runtime_import_boundary(self):
        forbidden = {
            "flax",
            "jax",
            "librosa",
            "numpy",
            "onnx",
            "safetensors",
            "sentencepiece",
            "tokenizers",
            "torchaudio",
            "transformers",
        }
        files = tuple(ARCHITECTURE_ROOT.glob("*.py")) + tuple(PROVIDER_ROOT.glob("*.py"))
        for path in files:
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".", 1)[0])
            self.assertFalse(
                imports & forbidden,
                f"{path.name}: {sorted(imports & forbidden)}",
            )

    def test_provenance_and_license_files_are_complete(self):
        source = json.loads((ARCHITECTURE_ROOT / "SOURCE.json").read_text(encoding="utf-8", ))
        self.assertEqual(
            source["artifact"]["revision"],
            "ae1e4845b4b07479735d93e1e591e566435b7104",
        )
        self.assertEqual(
            source["source"]["revision"],
            "65dc261512cbdb1ee72b88ae5b222f2605aad8e5",
        )
        self.assertTrue(source["audit"]["no_accuracy_claim"])
        license_text = (ARCHITECTURE_ROOT / "THIRD_PARTY_LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", license_text)
        self.assertIn("END OF TERMS AND CONDITIONS", license_text)
        model_notice = (ARCHITECTURE_ROOT / "MODEL_TERMS_NOTICE").read_text(encoding="utf-8")
        self.assertIn(
            "Health AI Developer Foundations Terms of Use",
            model_notice,
        )


if __name__ == "__main__":
    unittest.main()
