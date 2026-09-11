from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import Mock

import torch

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.hub import resolve_pretrained_file, write_json_file
from voicehub.models.asr_espnet.native.checkpoint import (
    NATIVE_ESPNET_FILENAME,
    NATIVE_ESPNET_FORMAT,
    NATIVE_ESPNET_LM_FILENAME,
    convert_espnet_librispeech_checkpoints,
    extract_espnet_token_list,
    native_espnet_asr_tensor_shapes,
    native_espnet_lm_tensor_shapes,
    tensor_inventory_fingerprint,
)
from voicehub.models.asr_espnet.native.configuration import ESPnetLibriSpeechTransformerConfig
from voicehub.models.asr_espnet.native.decoding import ESPnetCTCPrefixScorer, ESPnetJointBeamSearch
from voicehub.models.asr_espnet.native.metadata import (
    ESPNET_ASR_STATE_VALUES,
    ESPNET_ASR_TENSOR_COUNT,
    ESPNET_ASR_TENSOR_FINGERPRINT,
    ESPNET_CONFIG_FILENAME,
    ESPNET_CONFIG_SHA256,
    ESPNET_CONFIG_SIZE,
    ESPNET_LM_NATIVE_TENSOR_FINGERPRINT,
    ESPNET_LM_SOURCE_TENSOR_FINGERPRINT,
    ESPNET_REPOSITORY,
    ESPNET_REVISION,
    ESPNET_SOURCE_REVISION,
    ESPNET_TOKEN_LIST_SHA256,
)
from voicehub.models.asr_espnet.native.modeling import (
    ESPnetLibriSpeechTransformerForASR,
    ESPnetSequentialRNNLanguageModel,
)
from voicehub.models.asr_espnet.native.registration import create_espnet_architecture_spec
from voicehub.models.asr_espnet.native.tokenization import ESPnetLibriSpeechTokenizer
from voicehub.models.asr_espnet.native.training import NativeESPnetASRTrainingAdapter
from voicehub.models.asr_native.configuration import ESPnetASRConfig
from voicehub.models.asr_native.espnet import ESPnetASRForSpeechRecognition
from voicehub.training import ASRDataset, get_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _varint(value: int) -> bytes:
    value &= (1 << 64) - 1
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _wire_varint(field: int, value: int) -> bytes:
    return _varint(field << 3) + _varint(value)


def _wire_bytes(field: int, value: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(value)) + value


def _piece(text: str, score: float, piece_type: int = 1) -> bytes:
    return b"".join((
        _wire_bytes(1, text.encode("utf-8")),
        _varint((2 << 3) | 5),
        struct.pack("<f", score),
        _wire_varint(3, piece_type),
    ))


def _tiny_tokenizer_model() -> bytes:
    pieces = (
        ("<unk>", 0.0, 2),
        ("\u2581", -1.0, 1),
        ("A", -1.1, 1),
        ("B", -1.2, 1),
        ("\u2581A", -0.1, 1),
        ("\u2581B", -0.2, 1),
    )
    trainer = b"".join((
        _wire_varint(40, 0),
        _wire_varint(41, -1),
        _wire_varint(42, -1),
        _wire_varint(43, -1),
        _wire_bytes(44, b" <unk> "),
    ))
    normalizer = b"".join((
        _wire_bytes(1, b"nmt_nfkc"),
        _wire_varint(3, 1),
        _wire_varint(4, 1),
        _wire_varint(5, 1),
    ))
    return b"".join((
        *(_wire_bytes(1, _piece(text, score, piece_type)) for text, score, piece_type in pieces),
        _wire_bytes(2, trainer),
        _wire_bytes(3, normalizer),
    ))


def _tiny_tokens() -> tuple[str, ...]:
    return (
        "<blank>",
        "<unk>",
        "\u2581",
        "A",
        "B",
        "\u2581A",
        "\u2581B",
        "<sos/eos>",
    )


def _tiny_config() -> ESPnetLibriSpeechTransformerConfig:
    return ESPnetLibriSpeechTransformerConfig(
        variant="custom",
        n_fft=16,
        win_length=16,
        hop_length=4,
        n_mels=13,
        vocabulary_size=8,
        blank_token_id=0,
        unknown_token_id=1,
        sos_eos_token_id=7,
        encoder_dimension=8,
        encoder_attention_heads=2,
        encoder_linear_units=16,
        encoder_blocks=2,
        decoder_attention_heads=2,
        decoder_linear_units=16,
        decoder_blocks=2,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
        apply_spec_augment=False,
        language_model_layers=1,
        language_model_units=8,
        language_model_dropout=0.0,
        beam_size=2,
        language_model_weight=0.0,
        maximum_decode_ratio=0.5,
    )


def _write_native_artifact(root: Path):
    config = _tiny_config()
    model = ESPnetLibriSpeechTransformerForASR(config).eval()
    language_model = ESPnetSequentialRNNLanguageModel(config).eval()
    save_safetensors(
        model.state_dict(),
        root / NATIVE_ESPNET_FILENAME,
        metadata={"format": NATIVE_ESPNET_FORMAT},
    )
    save_safetensors(
        language_model.state_dict(),
        root / NATIVE_ESPNET_LM_FILENAME,
        metadata={"format": NATIVE_ESPNET_FORMAT},
    )
    (root / "tokenizer.model").write_bytes(_tiny_tokenizer_model())
    (root / "tokens.txt").write_text(
        "\n".join(_tiny_tokens()) + "\n",
        encoding="utf-8",
    )
    values = config.to_dict()
    values['architectures'] = [
        "ESPnetASRForSpeechRecognition",
        "ESPnetLibriSpeechTransformerForASR",
    ]
    values["checkpoint_format"] = NATIVE_ESPNET_FORMAT
    write_json_file(root / "config.json", values)
    return model, language_model


class NativeESPnetArchitectureTests(unittest.TestCase):

    def test_shared_registry_and_trainer_select_the_native_recipe(self):
        from voicehub.registry import get_model_spec
        from voicehub.runtime import get_architecture_spec
        from voicehub.training.recipes import BUILTIN_MODEL_ADAPTERS
        from voicehub.training.specs import get_training_spec

        model_spec = get_model_spec("asr_espnet")
        architecture = get_architecture_spec("espnet-librispeech-transformer-e18")
        training = get_training_spec("asr_espnet")

        self.assertTrue(model_spec.is_voicehub_native)
        self.assertIs(model_spec.native_architecture, architecture)
        self.assertTrue(architecture.capabilities.distributed_training)
        self.assertTrue(training.native_training)
        self.assertEqual(training.support.value, "native")
        self.assertEqual(training.default_phase, "speech_recognition")
        self.assertEqual(
            training.adapter_factory,
            ("voicehub.models.asr_espnet.native.training:"
             "NativeESPnetASRTrainingAdapter"),
        )
        self.assertEqual(
            BUILTIN_MODEL_ADAPTERS["asr_espnet"].__name__,
            "NativeESPnetASRTrainingAdapter",
        )

    def test_release_graph_matches_range_audited_inventory(self):
        shapes = native_espnet_asr_tensor_shapes()
        meta = OrderedDict((name, torch.empty(shape, device="meta")) for name, shape in shapes.items())

        self.assertEqual(len(shapes), ESPNET_ASR_TENSOR_COUNT)
        self.assertEqual(
            sum(math.prod(shape) for shape in shapes.values()),
            ESPNET_ASR_STATE_VALUES,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(meta),
            ESPNET_ASR_TENSOR_FINGERPRINT,
        )
        self.assertEqual(
            shapes["encoder.embed.conv.2.weight"],
            (512, 512, 5, 5),
        )
        self.assertEqual(
            shapes["encoder.encoders.17.feed_forward.w_1.weight"],
            (2_048, 512),
        )
        self.assertEqual(
            shapes["decoder.output_layer.weight"],
            (5_000, 512),
        )
        self.assertEqual(len(ESPNET_SOURCE_REVISION), 40)
        self.assertEqual(len(ESPNET_REVISION), 40)

    def test_language_model_inventory_preserves_exact_source_mapping(self):
        shapes = native_espnet_lm_tensor_shapes()
        native = OrderedDict((name, torch.empty(shape, device="meta")) for name, shape in shapes.items())
        source = OrderedDict((f"lm.{name}", tensor) for name, tensor in native.items())

        self.assertEqual(len(shapes), 19)
        self.assertEqual(
            sum(math.prod(shape) for shape in shapes.values()),
            154_768_264,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(native),
            ESPNET_LM_NATIVE_TENSOR_FINGERPRINT,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(source),
            ESPNET_LM_SOURCE_TENSOR_FINGERPRINT,
        )

    def test_official_variant_rejects_graph_drift(self):
        with self.assertRaisesRegex(ValueError, "fixes these fields"):
            ESPnetLibriSpeechTransformerConfig(encoder_blocks=17)
        with self.assertRaisesRegex(ValueError, "finite"):
            ESPnetLibriSpeechTransformerConfig(
                variant="custom",
                ctc_weight=float("nan"),
            )
        custom = ESPnetLibriSpeechTransformerConfig(
            variant="custom",
            encoder_blocks=17,
        )
        self.assertEqual(custom.encoder_blocks, 17)

    def test_tiny_graph_computes_hybrid_loss_and_gradients(self):
        torch.manual_seed(11)
        config = _tiny_config()
        model = ESPnetLibriSpeechTransformerForASR(config).train()
        output = model(
            torch.randn(2, 256),
            torch.tensor([256, 224]),
            torch.tensor([[5, 6], [5, -1]]),
            torch.tensor([2, 1]),
        )
        output.loss.backward()

        expected = (
            config.ctc_weight * output.losses["ctc_loss"] +
            (1.0 - config.ctc_weight) * output.losses["attention_loss"])
        self.assertTrue(torch.equal(output.loss, expected))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(tuple(output.ctc_logits.shape), (2, 10, 8))
        self.assertIsNotNone(model.encoder.encoders[0].self_attn.linear_q.weight.grad)
        self.assertIsNotNone(model.decoder.output_layer.weight.grad)
        self.assertIsNotNone(model.ctc.ctc_lo.weight.grad)

    def test_ctc_prefix_scorer_handles_repeated_labels_and_eos(self):
        values = torch.tensor([
            [-0.2, -1.0, -2.0, -3.0],
            [-0.3, -0.7, -1.2, -3.0],
            [-0.4, -0.6, -1.4, -3.0],
        ])
        scorer = ESPnetCTCPrefixScorer(
            values,
            blank_token_id=0,
            eos_token_id=3,
        )
        scores, states = scorer.extend(
            (3, ),
            torch.tensor([1, 2, 3]),
            scorer.initial_state,
        )
        repeated_scores, _ = scorer.extend(
            (3, 1),
            torch.tensor([1, 2, 3]),
            states[0],
        )

        self.assertEqual(tuple(states.shape), (3, 3, 2))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertLess(repeated_scores[0], repeated_scores[1])
        self.assertTrue(torch.isfinite(repeated_scores[2]))

    def test_joint_beam_search_returns_bounded_sequences(self):
        torch.manual_seed(3)
        config = _tiny_config()
        model = ESPnetLibriSpeechTransformerForASR(config).eval()
        language_model = ESPnetSequentialRNNLanguageModel(config).eval()
        decoder = ESPnetJointBeamSearch(
            model,
            config,
            language_model=language_model,
        )
        with torch.inference_mode():
            states, lengths = model.encode(
                torch.randn(1, 128),
                torch.tensor([128]),
                apply_augmentation=False,
            )
            result = decoder(states, lengths, beam_size=2)

        self.assertEqual(len(result.token_ids), 1)
        self.assertLessEqual(
            len(result.token_ids[0]),
            math.ceil(lengths[0].item() * config.maximum_decode_ratio),
        )
        self.assertTrue(math.isfinite(result.scores[0]))

    def test_conv2d6_rejects_too_few_feature_frames_precisely(self):
        config = _tiny_config()
        model = ESPnetLibriSpeechTransformerForASR(config).eval()

        with self.assertRaisesRegex(ValueError, "at least 11 feature frames"):
            model(
                features=torch.randn(1, 10, config.n_mels),
                feature_lengths=torch.tensor([10]),
            )

    def test_architecture_spec_is_narrow_and_trainable(self):
        spec = create_espnet_architecture_spec()

        self.assertEqual(
            spec.architecture_id,
            "espnet-librispeech-transformer-e18",
        )
        self.assertTrue(spec.capabilities.training)
        self.assertFalse(spec.capabilities.streaming)
        self.assertIn("safetensors", spec.capabilities.checkpoint_formats)
        self.assertIn(
            "Only the LibriSpeech Transformer e18",
            spec.metadata["verified_scope"],
        )


class NativeESPnetArtifactTests(unittest.TestCase):

    def test_token_list_extraction_uses_no_yaml_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "seed: 0\n"
                "token_list:\n"
                '- "<blank>"\n'
                "- <unk>\n"
                '- "\\u2581A"\n'
                "- <sos/eos>\n"
                "encoder: transformer\n",
                encoding="utf-8",
            )
            tokens = extract_espnet_token_list(path)

        self.assertEqual(
            tokens,
            ("<blank>", "<unk>", "\u2581A", "<sos/eos>"),
        )

    @unittest.skipUnless(
        os.environ.get("VOICEHUB_TEST_ESPNET_CONFIG") or
        os.environ.get("VOICEHUB_TEST_RELEASE_ASSETS") == "1",
        "set VOICEHUB_TEST_ESPNET_CONFIG or VOICEHUB_TEST_RELEASE_ASSETS=1 "
        "for pinned release-asset verification",
    )
    def test_release_token_list_fingerprint(self):
        configured_path = os.environ.get("VOICEHUB_TEST_ESPNET_CONFIG")
        config_path = (
            Path(configured_path) if configured_path is not None else resolve_pretrained_file(
                ESPNET_REPOSITORY,
                ESPNET_CONFIG_FILENAME,
                revision=ESPNET_REVISION,
            ))
        config_payload = config_path.read_bytes()
        self.assertEqual(len(config_payload), ESPNET_CONFIG_SIZE)
        self.assertEqual(
            hashlib.sha256(config_payload).hexdigest(),
            ESPNET_CONFIG_SHA256,
        )
        tokens = extract_espnet_token_list(config_path)
        payload = ("\n".join(tokens) + "\n").encode("utf-8")
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            ESPNET_TOKEN_LIST_SHA256,
        )

    def test_custom_pickle_conversion_requires_trust_then_reloads(self):
        torch.manual_seed(5)
        config = _tiny_config()
        model = ESPnetLibriSpeechTransformerForASR(config).eval()
        language_model = ESPnetSequentialRNNLanguageModel(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asr = root / "54epoch.pth"
            lm = root / "17epoch.pth"
            tokenizer = root / "bpe.model"
            yaml = root / "config.yaml"
            torch.save(model.state_dict(), asr)
            torch.save(
                OrderedDict((f"lm.{name}", tensor) for name, tensor in language_model.state_dict().items()),
                lm,
            )
            tokenizer.write_bytes(_tiny_tokenizer_model())
            yaml.write_text(
                "token_list:\n" + "".join(f"- {token}\n"
                                          for token in _tiny_tokens()) + "encoder: transformer\n",
                encoding="utf-8",
            )
            destination = root / "native"
            with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint"):
                convert_espnet_librispeech_checkpoints(
                    asr_checkpoint=asr,
                    language_model_checkpoint=lm,
                    tokenizer_model=tokenizer,
                    config_yaml=yaml,
                    destination=destination,
                    config=config,
                )
            convert_espnet_librispeech_checkpoints(
                asr_checkpoint=asr,
                language_model_checkpoint=lm,
                tokenizer_model=tokenizer,
                config_yaml=yaml,
                destination=destination,
                config=config,
                trust_pickle_checkpoint=True,
            )

            with SafeTensorReader(destination / NATIVE_ESPNET_FILENAME) as reader:
                self.assertEqual(
                    reader.metadata["format"],
                    NATIVE_ESPNET_FORMAT,
                )
                self.assertEqual(set(reader.keys()), set(model.state_dict()))
            wrapper = ESPnetASRForSpeechRecognition(
                model_path=destination,
                device="cpu",
                lazy_load=False,
            )

        self.assertIsInstance(
            wrapper.model,
            ESPnetLibriSpeechTransformerForASR,
        )
        self.assertIsInstance(
            wrapper.language_model,
            ESPnetSequentialRNNLanguageModel,
        )

    def test_native_export_reloads_in_a_fresh_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            original, original_lm = _write_native_artifact(source)
            wrapper = ESPnetASRForSpeechRecognition(
                model_path=source,
                device="cpu",
                lazy_load=False,
            )
            exported = root / "exported"
            wrapper.export_native_pretrained(exported)
            fresh = ESPnetASRForSpeechRecognition(
                model_path=exported,
                device="cpu",
                lazy_load=False,
            )

            self.assertEqual(
                fresh.model.state_dict().keys(),
                original.state_dict().keys(),
            )
            self.assertEqual(
                fresh.language_model.state_dict().keys(),
                original_lm.state_dict().keys(),
            )
            self.assertTrue(torch.equal(
                fresh.model.ctc.ctc_lo.weight,
                wrapper.model.ctc.ctc_lo.weight,
            ))

    def test_tokenizer_maps_sentencepiece_pieces_to_espnet_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer.model").write_bytes(_tiny_tokenizer_model())
            (root / "tokens.txt").write_text(
                "\n".join(_tiny_tokens()) + "\n",
                encoding="utf-8",
            )
            tokenizer = ESPnetLibriSpeechTokenizer.from_files(
                root / "tokenizer.model",
                root / "tokens.txt",
                strict_release=False,
            )

        self.assertEqual(tokenizer.encode_as_ids("a b"), (5, 6))
        self.assertEqual(tokenizer.decode_ids((5, 6)), "A B")


class NativeESPnetWrapperTests(unittest.TestCase):

    def test_prebuilt_asr_dataset_preserves_lazy_transform_fingerprint(self):
        transform_calls = []

        def transform(record):
            transform_calls.append(record["text"])
            return record

        dataset = ASRDataset(
            [{
                "audio": "clip.wav",
                "text": "Lazy transcript.",
            }],
            model_type="asr_espnet",
            transform=transform,
            transform_fingerprint="espnet-normalization-v1",
        )
        adapter = NativeESPnetASRTrainingAdapter(
            object(),
            get_training_spec("asr_espnet"),
        )

        prepared = adapter.create_dataset(dataset)

        self.assertIs(prepared, dataset)
        self.assertEqual(transform_calls, [])
        self.assertEqual(
            prepared.resume_fingerprint()["transform_fingerprint"],
            "espnet-normalization-v1",
        )
        self.assertEqual(prepared[0]["text"], "Lazy transcript.")
        self.assertEqual(transform_calls, ["Lazy transcript."])

    def test_inference_contract_rejects_unverified_features(self):
        validator = ESPnetASRForSpeechRecognition._validate_inference_request
        with self.assertRaisesRegex(ValueError, "English-only"):
            validator(
                language="tr",
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
            validator(
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
        with self.assertRaisesRegex(ValueError, "hotwords"):
            validator(
                language="en",
                task="transcribe",
                return_timestamps=False,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                num_beams=None,
                max_new_tokens=None,
                hotwords=("VOICEHUB", ),
            )

    def test_provider_config_rejects_external_runtime_options(self):
        with self.assertRaisesRegex(ValueError, "loader options"):
            ESPnetASRConfig(model_kwargs={"beam_search": "external"})
        with self.assertRaisesRegex(ValueError, "float32"):
            ESPnetASRConfig(torch_dtype="float16")
        config = ESPnetASRConfig(
            beam_size=5,
            ctc_weight=0.4,
            language_model_weight=0.5,
        )
        self.assertEqual(config.beam_size, 5)
        self.assertEqual(config.ctc_weight, 0.4)
        self.assertEqual(config.language_model_weight, 0.5)

    def test_raw_audio_training_batch_uses_native_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_native_artifact(root)
            wrapper = ESPnetASRForSpeechRecognition(
                model_path=root,
                device="cpu",
                lazy_load=False,
            )
            prepared = wrapper.prepare_training_inputs(
                {
                    "audio": torch.randn(16),
                    "sampling_rate": 16_000,
                    "text": "a b",
                },
                phase="speech_recognition",
            )

        self.assertEqual(
            tuple(prepared["waveforms"].shape),
            (1, wrapper.native_config.minimum_waveform_samples),
        )
        self.assertEqual(prepared["label_lengths"].tolist(), [2])
        self.assertEqual(prepared["labels"].tolist(), [[5, 6]])

    def test_provider_import_has_no_external_runtime_dependency(self):
        script = """
import builtins
import sys

blocked = {
    "espnet",
    "espnet2",
    "numpy",
    "safetensors",
    "sentencepiece",
    "torchaudio",
    "transformers",
}
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError("blocked import: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from voicehub.models.asr_native.espnet import ESPnetASRForSpeechRecognition
from voicehub.models.asr_native.configuration import ESPnetASRConfig
ESPnetASRForSpeechRecognition(ESPnetASRConfig(), lazy_load=True)
print("ok")
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
