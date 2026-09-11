from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.models.asr_wenet import NativeWeNetU2PPTrainingAdapter, WeNetASRConfig, WeNetASRForSpeechRecognition
from voicehub.models.asr_wenet.artifacts import _official_archive_path, resolve_wenet_u2pp_artifacts
from voicehub.models.asr_wenet.native.checkpoint import (
    WeNetU2PPSafeTensorsCheckpointAdapter,
    _load_restricted_state,
    convert_wenet_gigaspeech_checkpoint,
    native_wenet_tensor_shapes,
    tensor_inventory_fingerprint,
)
from voicehub.models.asr_wenet.native.configuration import WeNetU2PPConfig
from voicehub.models.asr_wenet.native.decoding import (
    WeNetDecodeHypothesis,
    attention_rescore,
    ctc_greedy_decode,
    ctc_prefix_beam_search,
)
from voicehub.models.asr_wenet.native.metadata import (
    GIGASPEECH_ARCHIVE_SHA256,
    GIGASPEECH_ARCHIVE_SIZE,
    GIGASPEECH_CHECKPOINT_LICENSE,
    GIGASPEECH_MIRROR_FILENAME,
    GIGASPEECH_MIRROR_REPOSITORY,
    GIGASPEECH_MIRROR_REVISION,
    GIGASPEECH_MODEL_URL,
    GIGASPEECH_STATE_VALUES,
    GIGASPEECH_TENSOR_COUNT,
    GIGASPEECH_TENSOR_FINGERPRINT,
    GIGASPEECH_TENSOR_FINGERPRINT_FORMAT,
    WENET_SOURCE_REVISION,
)
from voicehub.models.asr_wenet.native.modeling import WeNetSpecAugment, WeNetU2PPForASR
from voicehub.models.asr_wenet.native.tokenization import WeNetGigaSpeechTokenizer
from voicehub.registry import get_model_spec
from voicehub.training import get_training_spec
from voicehub.training.specs import TrainingFamily, TrainingSupport

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_config(**overrides) -> WeNetU2PPConfig:
    values = {
        "variant": "custom",
        "input_dim": 20,
        "vocab_size": 24,
        "encoder_dim": 16,
        "encoder_heads": 4,
        "encoder_linear_units": 32,
        "encoder_layers": 2,
        "decoder_heads": 4,
        "decoder_linear_units": 32,
        "decoder_layers": 1,
        "reverse_decoder_layers": 1,
        "convolution_kernel_size": 7,
        "sos_eos_token_id": 23,
        "use_dynamic_chunk": False,
        "spec_augment": False,
    }
    values.update(overrides)
    return WeNetU2PPConfig(**values, )


class NativeWeNetArchitectureTests(unittest.TestCase):

    def test_tokenizer_preserves_gigaspeech_word_boundary_semantics(self):
        sentencepiece = SimpleNamespace(encode_as_pieces=lambda word: [f"\u2581{word}"], )
        tokenizer = WeNetGigaSpeechTokenizer(
            sentencepiece,
            (
                "<blank>",
                "<unk>",
                "HELLO,WORLD!",
                "\u2581HELLO",
                "\u2581WORLD",
                "<sos/eos>",
            ),
        )

        self.assertEqual(
            tokenizer.encode_as_pieces("hello world"),
            ["\u2581HELLO", "\u2581WORLD"],
        )
        self.assertEqual(
            tokenizer.encode_as_pieces("hello, world!"),
            ["HELLO,WORLD!"],
        )
        self.assertEqual(tokenizer.encode_as_ids("hello, world!"), [2])

    def test_released_graph_matches_audited_inventory(self):
        shapes = native_wenet_tensor_shapes()
        meta_state = {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}

        self.assertEqual(len(shapes), GIGASPEECH_TENSOR_COUNT)
        self.assertEqual(
            sum(math.prod(shape) for shape in shapes.values()),
            GIGASPEECH_STATE_VALUES,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(meta_state),
            GIGASPEECH_TENSOR_FINGERPRINT,
        )
        self.assertEqual(
            GIGASPEECH_TENSOR_FINGERPRINT,
            "c1956c8a895e342aa4f53f824b1729f41fa3a861b6b08a1fe5e3d55b48ff45c3",
        )
        self.assertEqual(
            GIGASPEECH_TENSOR_FINGERPRINT_FORMAT,
            "SHA-256 of sorted name|portable-dtype|dimxdim rows joined by LF",
        )
        self.assertEqual(
            shapes["encoder.embed.conv.0.weight"],
            (512, 1, 3, 3),
        )
        self.assertEqual(
            shapes["encoder.embed.conv.2.weight"],
            (512, 512, 5, 5),
        )
        self.assertEqual(
            shapes["decoder.right_decoder.output_layer.weight"],
            (4_999, 512),
        )
        self.assertEqual(shapes["ctc.ctc_lo.weight"], (4_999, 512))
        self.assertEqual(len(WENET_SOURCE_REVISION), 40)
        self.assertEqual(len(GIGASPEECH_ARCHIVE_SHA256), 64)
        self.assertEqual(GIGASPEECH_ARCHIVE_SIZE, 503_845_602)
        self.assertEqual(GIGASPEECH_MIRROR_REPOSITORY, "openspeech/wenet-models")
        self.assertEqual(
            GIGASPEECH_MIRROR_REVISION,
            "90acd57d17169a15d5ceab462c6e7db3bd003921",
        )
        self.assertEqual(
            GIGASPEECH_MODEL_URL,
            "https://huggingface.co/openspeech/wenet-models/resolve/"
            "90acd57d17169a15d5ceab462c6e7db3bd003921/"
            f"{GIGASPEECH_MIRROR_FILENAME}",
        )
        self.assertEqual(GIGASPEECH_CHECKPOINT_LICENSE, "NOT DECLARED")

    def test_tiny_graph_computes_exact_hybrid_loss_and_backward(self):
        torch.manual_seed(7)
        model = WeNetU2PPForASR(_tiny_config()).train()
        features = torch.randn(2, 70, 20)
        labels = torch.tensor([
            [5, 6, 7, -1],
            [8, 9, -1, -1],
        ])

        output = model(
            features=features,
            feature_lengths=torch.tensor([70, 63]),
            labels=labels,
            label_lengths=torch.tensor([3, 2]),
            decoding_chunk_size=-1,
        )
        output.loss.backward()

        expected = (0.3 * output.ctc_loss + 0.7 * output.attention_loss)
        self.assertTrue(torch.equal(output.loss, expected))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNotNone(model.encoder.encoders[0].self_attn.linear_q.weight.grad)
        self.assertIsNotNone(model.decoder.left_decoder.output_layer.weight.grad)
        self.assertIsNotNone(model.decoder.right_decoder.output_layer.weight.grad)
        self.assertIsNotNone(model.ctc.ctc_lo.weight.grad)

    def test_custom_graph_supports_forward_only_attention_training(self):
        model = WeNetU2PPForASR(_tiny_config(reverse_weight=0.0)).train()

        output = model(
            features=torch.randn(1, 70, 20),
            feature_lengths=torch.tensor([70]),
            labels=torch.tensor([[5, 6]]),
            label_lengths=torch.tensor([2]),
            decoding_chunk_size=-1,
        )
        output.loss.backward()

        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNotNone(model.decoder.left_decoder.output_layer.weight.grad)
        self.assertIsNone(model.decoder.right_decoder.output_layer.weight.grad)

    def test_spec_augment_masks_only_valid_source_frames(self):
        torch.manual_seed(11)
        augmenter = WeNetSpecAugment(
            _tiny_config(
                spec_augment=True,
                spec_max_time=4,
                spec_max_frequency=3,
            )).train()
        features = torch.ones(2, 10, 20)

        augmented = augmenter(features, torch.tensor([10, 5]))

        self.assertTrue((augmented[0] == 0.0).any())
        self.assertTrue((augmented[1, :5] == 0.0).any())
        self.assertTrue(torch.equal(augmented[1, 5:], features[1, 5:]))

    def test_short_raw_waveform_is_safely_extended_to_conv_context(self):
        model = WeNetU2PPForASR(_tiny_config()).eval()

        output = model(
            input_signal=torch.zeros(1, 800),
            input_signal_length=torch.tensor([800]),
        )

        self.assertEqual(tuple(output.log_probabilities.shape), (1, 1, 24))
        self.assertEqual(output.encoded_lengths.tolist(), [1])

    def test_native_safetensors_round_trip_preserves_outputs(self):
        torch.manual_seed(19)
        config = _tiny_config()
        source = WeNetU2PPForASR(config).eval()
        features = torch.randn(1, 70, 20)
        lengths = torch.tensor([65])
        expected = source(
            features=features,
            feature_lengths=lengths,
        ).log_probabilities

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            save_safetensors(source.state_dict(), checkpoint)
            fresh = WeNetU2PPForASR(config).eval()
            adapter = WeNetU2PPSafeTensorsCheckpointAdapter()
            with SafeTensorReader(checkpoint) as reader:
                adapter.load_streaming(
                    fresh,
                    reader,
                    config.to_dict(),
                    strict=True,
                )
            actual = fresh(
                features=features,
                feature_lengths=lengths,
            ).log_probabilities

        self.assertTrue(torch.equal(expected, actual))

    def test_pickle_conversion_requires_explicit_trust(self):
        with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint=True"):
            _load_restricted_state(
                Path("untrusted-final.pt"),
                trust_pickle_checkpoint=False,
            )

    def test_ctc_search_and_attention_rescoring_are_bounded(self):
        model = WeNetU2PPForASR(_tiny_config()).eval()
        log_probabilities = torch.log_softmax(
            torch.randn(1, 5, 24),
            dim=-1,
        )
        lengths = torch.tensor([5])

        greedy = ctc_greedy_decode(log_probabilities, lengths)[0]
        nbest = ctc_prefix_beam_search(
            log_probabilities,
            lengths,
            beam_size=3,
        )[0]
        rescored = attention_rescore(
            model,
            nbest,
            torch.randn(1, 4, 16),
        )

        self.assertLessEqual(len(nbest), 3)
        self.assertIsInstance(greedy.token_ids, tuple)
        self.assertIn(rescored.token_ids, {item.token_ids for item in nbest})

    def test_attention_rescoring_replaces_padding_with_eos(self):
        model = WeNetU2PPForASR(_tiny_config()).eval()
        nbest = (
            WeNetDecodeHypothesis((2, 3), -1.0),
            WeNetDecodeHypothesis((4, ), -2.0),
        )

        result = attention_rescore(
            model,
            nbest,
            torch.randn(1, 4, 16),
        )

        self.assertIn(result.token_ids, {(2, 3), (4, )})

    @unittest.skipUnless(
        os.environ.get("VOICEHUB_TEST_WENET_CHECKPOINT"),
        "set VOICEHUB_TEST_WENET_CHECKPOINT for real conversion validation",
    )
    def test_real_checkpoint_converts_with_exact_namespace(self):
        source = Path(os.environ["VOICEHUB_TEST_WENET_CHECKPOINT"])
        with tempfile.TemporaryDirectory() as directory:
            output = convert_wenet_gigaspeech_checkpoint(
                source,
                directory,
                trust_pickle_checkpoint=True,
            )
            values = json.loads((output / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(
            values["source_tensor_fingerprint"],
            GIGASPEECH_TENSOR_FINGERPRINT,
        )


class NativeWeNetProviderTests(unittest.TestCase):

    def test_registry_and_training_adapter_are_native(self):
        provider = get_model_spec("asr_wenet")
        training = get_training_spec("asr_wenet")

        self.assertEqual(
            provider.default_model_path,
            WeNetASRForSpeechRecognition.default_model_name_or_path,
        )
        self.assertIn("voicehub-native", provider.capabilities)
        self.assertEqual(provider.license.license_id, "NOT DECLARED")
        self.assertEqual(training.family, TrainingFamily.SPEECH_SEQ2SEQ)
        self.assertEqual(training.support, TrainingSupport.NATIVE)
        self.assertIn(
            "voicehub.models.asr_wenet.native",
            training.source_entrypoints[0],
        )
        adapter = WeNetASRForSpeechRecognition(WeNetASRConfig()).get_training_adapter()
        self.assertIsInstance(adapter, NativeWeNetU2PPTrainingAdapter)

    def test_training_adapter_preserves_adam_and_warmuplr(self):
        wrapper = SimpleNamespace(native_config=SimpleNamespace(warmup_steps=4), )
        adapter = NativeWeNetU2PPTrainingAdapter(
            wrapper,
            get_training_spec("asr_wenet"),
        )
        parameter = torch.nn.Parameter(torch.ones(()))
        arguments = SimpleNamespace(
            learning_rate=0.001,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            weight_decay=0.0,
            warmup_steps=0,
        )

        optimizer = adapter.create_optimizer(
            "default",
            [("weight", parameter)],
            arguments,
        )
        scheduler = adapter.create_scheduler(
            "default",
            optimizer,
            100,
            arguments,
        )

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.00025)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0005)

    def test_raw_audio_record_reaches_native_hybrid_backward(self):
        wrapper = WeNetASRForSpeechRecognition(
            WeNetASRConfig(),
            device="cpu",
        )
        wrapper.native_config = _tiny_config()
        wrapper.model = WeNetU2PPForASR(wrapper.native_config).train()
        wrapper.tokenizer = SimpleNamespace(encode_as_ids=lambda text: [5, 6], )
        adapter = NativeWeNetU2PPTrainingAdapter(
            wrapper,
            get_training_spec("asr_wenet"),
        )

        prepared = adapter.prepare_training_inputs(
            {
                "audio": torch.zeros(8_000),
                "sampling_rate": 16_000,
                "text": "HELLO",
            },
            SimpleNamespace(phase=SimpleNamespace(name="speech_recognition")),
        )
        output = wrapper.model(**prepared)
        output.loss.backward()

        self.assertEqual(tuple(prepared["input_signal"].shape), (1, 8_000))
        self.assertEqual(tuple(prepared["labels"].shape), (1, 2))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNotNone(wrapper.model.ctc.ctc_lo.weight.grad)

    def test_collated_raw_batch_uses_per_row_rates_and_audio_lengths(self):
        wrapper = WeNetASRForSpeechRecognition(
            WeNetASRConfig(),
            device="cpu",
        )
        wrapper.native_config = _tiny_config()
        wrapper.model = WeNetU2PPForASR(wrapper.native_config).train()
        wrapper.tokenizer = SimpleNamespace(encode_as_ids=lambda text: [5, 6], )

        prepared = wrapper.prepare_training_inputs(
            {
                "audio": torch.zeros(2, 8_000),
                "audio_lengths": torch.tensor([8_000, 4_000]),
                "sampling_rate": torch.tensor([16_000, 16_000]),
                "text": ["FIRST", "SECOND"],
            },
            phase="speech_recognition",
        )

        self.assertEqual(
            prepared["input_signal_length"].tolist(),
            [8_000, 4_000],
        )
        self.assertEqual(tuple(prepared["input_signal"].shape), (2, 8_000))
        self.assertEqual(tuple(prepared["labels"].shape), (2, 2))

    def test_configuration_rejects_upstream_loader_controls(self):
        with self.assertRaisesRegex(ValueError, "model_kwargs"):
            WeNetASRConfig(model_kwargs={"gpu": 0})
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            WeNetASRConfig(sample_rate=8_000)
        with self.assertRaisesRegex(ValueError, "decoding_strategy"):
            WeNetASRConfig(decoding_strategy="magic")
        with self.assertRaisesRegex(ValueError, "inside the vocabulary"):
            WeNetU2PPConfig(
                variant="custom",
                vocab_size=24,
                sos_eos_token_id=24,
            )
        with self.assertRaisesRegex(TypeError, "language"):
            WeNetASRForSpeechRecognition._validate_request(
                language=7,
                task="transcribe",
                return_timestamps=False,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                max_new_tokens=None,
                hotwords=None,
            )

    def test_native_provider_imports_no_external_model_framework(self):
        code = """
import json
import sys
from voicehub.models.asr_wenet import WeNetASRForSpeechRecognition
blocked = (
    "wenet", "transformers", "torchaudio", "huggingface_hub",
    "safetensors", "sentencepiece", "yaml",
)
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
                "wenet": False,
                "transformers": False,
                "torchaudio": False,
                "huggingface_hub": False,
                "safetensors": False,
                "sentencepiece": False,
                "yaml": False,
            },
        )

    def test_explicit_safetensors_path_keeps_its_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "custom-name.safetensors"
            direct.write_bytes(b"checkpoint")
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer.model").write_bytes(b"tokenizer")
            (root / "units.txt").write_text("unit 0\n", encoding="utf-8")

            artifacts = resolve_wenet_u2pp_artifacts(
                direct,
                checkpoint_filename="model.safetensors",
                tokenizer_filename="tokenizer.model",
                units_filename="units.txt",
                revision=None,
                cache_dir=None,
                token=None,
                local_files_only=True,
                trust_pickle_checkpoint=False,
            )

        self.assertEqual(artifacts.checkpoint.name, "custom-name.safetensors")

    def test_complete_native_cache_does_not_rehash_the_legacy_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch("voicehub.models.asr_wenet.artifacts."
                       "_download_official_archive") as download:
                archive = _official_archive_path(cache)
                native = (archive.parent / ".voicehub-native" / "wenet-u2pp")
                native.mkdir(parents=True)
                (native / "model.safetensors").write_bytes(b"checkpoint")
                (native / "config.json").write_text("{}", encoding="utf-8")
                (native / "tokenizer.model").write_bytes(b"tokenizer")
                (native / "units.txt").write_text("unit 0\n", encoding="utf-8")

                artifacts = resolve_wenet_u2pp_artifacts(
                    "wenet/gigaspeech-u2pp-conformer",
                    checkpoint_filename="model.safetensors",
                    tokenizer_filename="tokenizer.model",
                    units_filename="units.txt",
                    revision=None,
                    cache_dir=cache,
                    token=None,
                    local_files_only=True,
                    trust_pickle_checkpoint=False,
                )

        download.assert_not_called()
        self.assertTrue(artifacts.converted_from_pickle)

    @unittest.skipUnless(
        os.environ.get("VOICEHUB_TEST_WENET_ASSETS"),
        "set VOICEHUB_TEST_WENET_ASSETS for tokenizer validation",
    )
    def test_real_tokenizer_uses_sorted_wenet_unit_ids(self):
        root = Path(os.environ["VOICEHUB_TEST_WENET_ASSETS"])
        tokenizer = WeNetGigaSpeechTokenizer.from_files(
            root / "train_xl_unigram5000.model",
            root / "units.txt",
        )

        token_ids = tokenizer.encode_as_ids("HELLO WORLD")

        self.assertEqual(tokenizer.encode_as_ids("HELLO, WORLD!"), [1])
        self.assertEqual(tokenizer.decode_ids(token_ids), "HELLO WORLD")
        self.assertEqual(tokenizer.vocabulary_size, 4_999)


if __name__ == "__main__":
    unittest.main()
