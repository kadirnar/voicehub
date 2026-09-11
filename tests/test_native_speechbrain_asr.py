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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.hub import resolve_pretrained_file, write_json_file
from voicehub.models.asr_native.configuration import SpeechBrainASRConfig
from voicehub.models.asr_native.speechbrain import SpeechBrainASRForSpeechRecognition
from voicehub.models.asr_native.speechbrain_training import NativeSpeechBrainASRTrainingAdapter
from voicehub.models.asr_speechbrain.native.artifacts import resolve_speechbrain_asr_artifacts
from voicehub.models.asr_speechbrain.native.checkpoint import (
    NATIVE_SPEECHBRAIN_ASR_FORMAT,
    SpeechBrainASRSafeTensorsCheckpointAdapter,
    convert_speechbrain_asr_checkpoints,
    native_speechbrain_asr_tensor_shapes,
    speechbrain_asr_source_tensor_mapping,
    speechbrain_lm_source_tensor_mapping,
    tensor_inventory_fingerprint,
)
from voicehub.models.asr_speechbrain.native.configuration import SpeechBrainCRDNNASRConfig
from voicehub.models.asr_speechbrain.native.decoding import SpeechBrainRNNLMBeamSearch
from voicehub.models.asr_speechbrain.native.metadata import (
    SPEECHBRAIN_ASR_NATIVE_STATE_VALUES,
    SPEECHBRAIN_ASR_NATIVE_TENSOR_COUNT,
    SPEECHBRAIN_ASR_NATIVE_TENSOR_FINGERPRINT,
    SPEECHBRAIN_ASR_REPOSITORY,
    SPEECHBRAIN_ASR_REVISION,
    SPEECHBRAIN_ASR_SOURCE_REVISION,
    SPEECHBRAIN_ASR_TOKENIZER_SHA256,
    SPEECHBRAIN_ASR_TOKENIZER_SIZE,
)
from voicehub.models.asr_speechbrain.native.modeling import SpeechBrainCRDNNForASR
from voicehub.registry import get_model_spec
from voicehub.tokenization import SentencePieceUnigramTokenizer
from voicehub.training import ASRDataset, get_training_spec
from voicehub.training.adapters import BaseTrainingAdapter
from voicehub.training.arguments import TrainingArguments
from voicehub.training.specs import TrainingFamily, TrainingSupport
from voicehub.training.trainer import Trainer

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


def _tiny_config() -> SpeechBrainCRDNNASRConfig:
    return SpeechBrainCRDNNASRConfig(
        variant="custom",
        n_fft=16,
        win_length=16,
        hop_length=4,
        n_mels=4,
        cnn_channels=(2, ),
        inter_layer_pooling_size=(2, ),
        time_pooling_size=2,
        rnn_layers=1,
        rnn_neurons=3,
        dnn_blocks=1,
        dnn_neurons=4,
        embedding_size=3,
        decoder_neurons=5,
        attention_dim=5,
        attention_channels=2,
        attention_kernel_size=1,
        output_neurons=6,
        lm_rnn_layers=1,
        lm_rnn_neurons=7,
        lm_dnn_blocks=1,
        lm_dnn_neurons=4,
        dropout=0.0,
        lm_dropout=0.0,
        beam_size=2,
        maximum_attention_shift=4,
        lm_weight=0.0,
        coverage_penalty=0.0,
    )


def _write_native_artifact(directory: Path) -> SpeechBrainCRDNNForASR:
    config = _tiny_config()
    model = SpeechBrainCRDNNForASR(config).eval()
    save_safetensors(
        model.state_dict(),
        directory / "model.safetensors",
        metadata={"format": NATIVE_SPEECHBRAIN_ASR_FORMAT},
    )
    values = config.to_dict()
    values.update({
        'architectures': [
            "SpeechBrainASRForSpeechRecognition",
            "SpeechBrainCRDNNForASR",
        ],
        "checkpoint_format": NATIVE_SPEECHBRAIN_ASR_FORMAT,
    })
    write_json_file(directory / "config.json", values)
    (directory / "tokenizer.model").write_bytes(_tiny_tokenizer_model())
    return model


class NativeSpeechBrainArchitectureTests(unittest.TestCase):

    def test_released_graph_matches_the_audited_inventory(self):
        shapes = native_speechbrain_asr_tensor_shapes()
        meta_state = {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}

        self.assertEqual(len(shapes), SPEECHBRAIN_ASR_NATIVE_TENSOR_COUNT)
        self.assertEqual(
            sum(math.prod(shape) for shape in shapes.values()),
            SPEECHBRAIN_ASR_NATIVE_STATE_VALUES,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(meta_state),
            SPEECHBRAIN_ASR_NATIVE_TENSOR_FINGERPRINT,
        )
        self.assertEqual(
            shapes["encoder.rnn.weight_ih_l0"],
            (4_096, 2_560),
        )
        self.assertEqual(
            shapes["decoder.attention.location_convolution.weight"],
            (10, 1, 201),
        )
        self.assertEqual(
            shapes["language_model.rnn.weight_ih_l0"],
            (8_192, 128),
        )
        self.assertEqual(len(SPEECHBRAIN_ASR_SOURCE_REVISION), 40)

    def test_tiny_graph_computes_recipe_loss_and_backward(self):
        torch.manual_seed(7)
        config = _tiny_config()
        model = SpeechBrainCRDNNForASR(config).train()
        waveforms = torch.randn(2, 128)
        output = model(
            waveforms,
            torch.tensor([128, 112]),
            tokens_bos=torch.tensor([[0, 4, 5], [0, 4, 0]]),
            tokens_eos=torch.tensor([[4, 5, 0], [4, 0, 0]]),
            token_lengths=torch.tensor([3, 2]),
            ctc_tokens=torch.tensor([[4, 5], [4, 0]]),
            ctc_token_lengths=torch.tensor([2, 1]),
            epoch=1,
            update_normalization=True,
        )
        output.loss.backward()

        expected = (config.ctc_weight * output.ctc_loss + (1.0 - config.ctc_weight) * output.seq2seq_loss)
        self.assertTrue(torch.equal(output.loss, expected))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(
            tuple(output.encoder_states.shape),
            (2, 16, config.dnn_neurons),
        )
        self.assertIsNotNone(model.encoder.rnn.weight_ih_l0.grad)
        self.assertIsNotNone(model.decoder.rnn_cells[0].weight_ih.grad)
        self.assertIsNotNone(model.ctc_linear.weight.grad)
        self.assertIsNone(model.language_model.output.weight.grad)
        self.assertFalse(model.language_model.output.weight.requires_grad)

    def test_ctc_uses_floor_lengths_and_stops_after_recipe_epoch(self):
        config = _tiny_config()
        model = SpeechBrainCRDNNForASR(config)
        states = torch.randn(1, 3, config.dnn_neurons)
        relative_lengths = torch.tensor([0.6])
        rounded_lengths = torch.tensor([2])
        zero_loss = states.sum() * 0.0
        inputs = {
            "waveforms": torch.zeros(1, 32),
            "ctc_tokens": torch.tensor([[1]]),
            "ctc_token_lengths": torch.tensor([1]),
        }

        with (
                patch.object(
                    model,
                    "encode",
                    return_value=(states, rounded_lengths, relative_lengths),
                ),
                patch(
                    "voicehub.models.asr_speechbrain.native.modeling."
                    "functional.ctc_loss",
                    return_value=zero_loss,
                ) as ctc_loss,
        ):
            output = model(**inputs, epoch=config.number_of_ctc_epochs)

        self.assertEqual(ctc_loss.call_args.args[2].tolist(), [1])
        self.assertIs(output.ctc_loss, zero_loss)

        with (
                patch.object(
                    model,
                    "encode",
                    return_value=(states, rounded_lengths, relative_lengths),
                ),
                patch("voicehub.models.asr_speechbrain.native.modeling."
                      "functional.ctc_loss", ) as ctc_loss,
        ):
            output = model(**inputs, epoch=config.number_of_ctc_epochs + 1)

        ctc_loss.assert_not_called()
        self.assertIsNone(output.ctc_logits)
        self.assertIsNone(output.ctc_loss)

    def test_cnn_block_pools_frequency_without_decimating_time(self):
        block = SpeechBrainCRDNNForASR(_tiny_config()).encoder.cnn_blocks[0]
        block.eval()
        output = block(torch.randn(2, 11, 4, 1))
        self.assertEqual(tuple(output.shape), (2, 11, 2, 2))

    def test_native_safetensors_round_trip_preserves_outputs(self):
        torch.manual_seed(19)
        config = _tiny_config()
        source = SpeechBrainCRDNNForASR(config).eval()
        waveforms = torch.randn(1, 128)
        expected = source(
            waveforms,
            torch.tensor([121]),
            tokens_bos=torch.tensor([[0, 4, 5]]),
        ).sequence_logits

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            save_safetensors(source.state_dict(), checkpoint)
            restored = SpeechBrainCRDNNForASR(config).eval()
            with SafeTensorReader(checkpoint) as reader:
                SpeechBrainASRSafeTensorsCheckpointAdapter().load_streaming(
                    restored,
                    reader,
                    config.to_dict(),
                    strict=True,
                )
            actual = restored(
                waveforms,
                torch.tensor([121]),
                tokens_bos=torch.tensor([[0, 4, 5]]),
            ).sequence_logits

        self.assertTrue(torch.equal(expected, actual))

    def test_beam_search_state_is_request_local(self):
        torch.manual_seed(23)
        model = SpeechBrainCRDNNForASR(_tiny_config()).eval()
        decoder = SpeechBrainRNNLMBeamSearch(model)
        states = torch.randn(1, 4, 4)

        first = decoder(states, torch.ones(1), beam_size=2)
        second = decoder(states, torch.ones(1), beam_size=2)

        self.assertEqual(first, second)
        self.assertEqual(len(first.token_ids), 1)
        self.assertLessEqual(len(first.token_ids[0]), 4)

    def test_validation_beam_search_does_not_execute_the_language_model(self):
        torch.manual_seed(24)
        model = SpeechBrainCRDNNForASR(_tiny_config()).eval()
        decoder = SpeechBrainRNNLMBeamSearch(model)
        states = torch.randn(1, 4, 4)

        with patch.object(
                model.language_model,
                "forward",
                side_effect=AssertionError("validation must not use the RNNLM"),
        ):
            result = decoder(
                states,
                torch.ones(1),
                beam_size=2,
                lm_weight=0.0,
            )

        self.assertEqual(len(result.token_ids), 1)


class NativeSpeechBrainCheckpointTests(unittest.TestCase):

    def test_pickle_conversion_is_explicit_and_strict(self):
        with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint=True"):
            convert_speechbrain_asr_checkpoints(
                asr_checkpoint="asr.ckpt",
                lm_checkpoint="lm.ckpt",
                normalizer_checkpoint="normalizer.ckpt",
                tokenizer_model="tokenizer.ckpt",
                destination="native",
            )

        config = _tiny_config()
        source = SpeechBrainCRDNNForASR(config)
        source.frontend.normalizer.count.fill_(1)
        state = source.state_dict()
        asr_state = {
            upstream: state[native].clone()
            for upstream, native in speechbrain_asr_source_tensor_mapping(config, ).items()
        }
        lm_state = {
            upstream: state[native].clone()
            for upstream, native in speechbrain_lm_source_tensor_mapping(config, ).items()
        }
        normalizer = {
            "count": 1,
            "glob_mean": state["frontend.normalizer.glob_mean"].clone(),
            "glob_std": state["frontend.normalizer.glob_std"].clone(),
            "spk_dict_mean": {},
            "spk_dict_std": {},
            "spk_dict_count": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(asr_state, root / "asr.ckpt")
            torch.save(lm_state, root / "lm.ckpt")
            torch.save(normalizer, root / "normalizer.ckpt")
            (root / "tokenizer.ckpt").write_bytes(_tiny_tokenizer_model(), )
            destination = convert_speechbrain_asr_checkpoints(
                asr_checkpoint=root / "asr.ckpt",
                lm_checkpoint=root / "lm.ckpt",
                normalizer_checkpoint=root / "normalizer.ckpt",
                tokenizer_model=root / "tokenizer.ckpt",
                destination=root / "native",
                config=config,
                trust_pickle_checkpoint=True,
            )

            with SafeTensorReader(destination / "model.safetensors", ) as reader:
                self.assertEqual(
                    reader.metadata["format"],
                    NATIVE_SPEECHBRAIN_ASR_FORMAT,
                )
                for name, expected in state.items():
                    torch.testing.assert_close(
                        reader.get_tensor(name),
                        expected,
                        rtol=0,
                        atol=0,
                    )


class NativeSpeechBrainProviderTests(unittest.TestCase):

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
            model_type="asr_speechbrain",
            transform=transform,
            transform_fingerprint="speechbrain-normalization-v1",
        )
        adapter = NativeSpeechBrainASRTrainingAdapter(
            object(),
            get_training_spec("asr_speechbrain"),
        )

        prepared = adapter.create_dataset(dataset)

        self.assertIs(prepared, dataset)
        self.assertEqual(transform_calls, [])
        self.assertEqual(
            prepared.resume_fingerprint()["transform_fingerprint"],
            "speechbrain-normalization-v1",
        )
        self.assertEqual(prepared[0]["text"], "Lazy transcript.")
        self.assertEqual(transform_calls, ["Lazy transcript."])

    def test_registry_training_profile_and_adapter_are_native(self):
        provider = get_model_spec("asr_speechbrain")
        training = get_training_spec("asr_speechbrain")

        self.assertTrue(provider.is_voicehub_native)
        self.assertEqual(provider.architecture, "speechbrain-crdnn-asr")
        self.assertIn("fine-tuning", provider.capabilities)
        self.assertEqual(training.family, TrainingFamily.SPEECH_SEQ2SEQ)
        self.assertEqual(training.support, TrainingSupport.NATIVE)
        wrapper = SpeechBrainASRForSpeechRecognition(SpeechBrainASRConfig(), )
        self.assertIsInstance(
            wrapper.get_training_adapter(),
            NativeSpeechBrainASRTrainingAdapter,
        )

    def test_only_opted_in_adapter_steps_a_scheduler_after_evaluation(self):

        class RecordingStrategy:

            def __init__(self):
                self.calls = []

            def scheduler_step(self, scheduler, **kwargs):
                self.calls.append((scheduler, kwargs))

        scheduler = object()
        strategy = RecordingStrategy()
        default_adapter = BaseTrainingAdapter.__new__(BaseTrainingAdapter, )
        generic_trainer = SimpleNamespace(
            training_adapter=default_adapter,
            lr_scheduler=scheduler,
            training_strategy=strategy,
        )
        Trainer._step_recipe_scheduler_after_evaluation(
            generic_trainer,
            {"eval_wer": 0.2},
        )
        self.assertEqual(strategy.calls, [])

        speechbrain_adapter = SpeechBrainASRForSpeechRecognition(
            SpeechBrainASRConfig(), ).get_training_adapter()
        speechbrain_trainer = SimpleNamespace(
            training_adapter=speechbrain_adapter,
            lr_scheduler=scheduler,
            training_strategy=strategy,
        )
        Trainer._step_recipe_scheduler_after_evaluation(
            speechbrain_trainer,
            {"eval_loss": 1.0},
        )
        self.assertEqual(strategy.calls, [])
        Trainer._step_recipe_scheduler_after_evaluation(
            speechbrain_trainer,
            {"eval_wer": 0.2},
        )
        self.assertEqual(
            strategy.calls,
            [(scheduler, {
                "metric": 0.2
            })],
        )

    def test_provider_import_has_no_external_model_runtime(self):
        code = """
import json
import sys
from voicehub.models.asr_native.speechbrain import (
    SpeechBrainASRForSpeechRecognition,
)
blocked = (
    "speechbrain", "sentencepiece", "google.protobuf", "hyperpyyaml",
    "torchaudio", "transformers", "safetensors",
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
                "google.protobuf": False,
                "hyperpyyaml": False,
                "safetensors": False,
                "sentencepiece": False,
                "speechbrain": False,
                "torchaudio": False,
                "transformers": False,
            },
        )

    def test_cached_native_conversion_no_longer_requires_pickle_trust(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "asr.ckpt").write_bytes(b"cached-source-marker")
            native = (root / ".voicehub-native" / "speechbrain-crdnn-asr")
            native.mkdir(parents=True)
            _write_native_artifact(native)

            with patch(
                    "voicehub.models.asr_speechbrain.native.artifacts."
                    "resolve_pretrained_file",
                    return_value=root / "asr.ckpt",
            ) as resolver:
                artifacts = resolve_speechbrain_asr_artifacts(
                    "speechbrain/asr-crdnn-rnnlm-librispeech",
                    revision=None,
                    cache_dir=None,
                    token=None,
                    local_files_only=False,
                    trust_pickle_checkpoint=False,
                )

            self.assertEqual(
                artifacts.checkpoint.parent,
                native.resolve(),
            )
            self.assertTrue(artifacts.converted_from_pickle)
            resolver.assert_called_once()

    def test_local_inference_training_and_export_reload(self):
        torch.manual_seed(29)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            export = root / "export"
            source.mkdir()
            _write_native_artifact(source)
            wrapper = SpeechBrainASRForSpeechRecognition(
                model_path=source,
                device="cpu",
                lazy_load=False,
            )

            inference = wrapper(
                torch.zeros(128),
                sampling_rate=16_000,
                num_beams=1,
            )
            self.assertEqual(
                inference.metadata["backend"],
                "voicehub-native",
            )
            self.assertEqual(
                inference.metadata["checkpoint_format"],
                NATIVE_SPEECHBRAIN_ASR_FORMAT,
            )

            adapter = wrapper.get_training_adapter()
            context = adapter.create_training_context(
                {
                    "audio": torch.randn(2, 128),
                    "audio_lengths": torch.tensor([128, 112]),
                    "sampling_rate": 16_000,
                    "text": ["A B", "A"],
                },
                epoch=0.0,
            )
            output = adapter.execute_training_phase(context)
            output.loss.backward()
            self.assertTrue(torch.isfinite(output.loss))
            self.assertTrue(output.metadata["ctc_active"])
            self.assertEqual(
                output.metadata["native_objective"],
                "combined-ctc-seq2seq",
            )

            optimizer = adapter.create_optimizer(
                "model",
                tuple(adapter.named_parameters()),
                None,
            )
            scheduler = adapter.create_scheduler(
                "model",
                optimizer,
                10,
                None,
            )
            self.assertIsNone(scheduler.step())
            self.assertEqual(
                scheduler.step_validation_wer(0.20),
                (1.0, 1.0),
            )
            self.assertEqual(
                scheduler.step_validation_wer(0.20),
                (1.0, 0.8),
            )

            wrapper.model.zero_grad(set_to_none=True)
            dataset = wrapper.create_training_dataset([
                {
                    "audio": torch.randn(128),
                    "sampling_rate": 16_000,
                    "text": "A B",
                },
                {
                    "audio": torch.randn(112),
                    "sampling_rate": 16_000,
                    "text": "A",
                },
            ])
            trainer = Trainer(
                model=wrapper,
                args=TrainingArguments(
                    output_dir=str(root / "run"),
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=dataset,
            )
            training = trainer.train()
            self.assertEqual(training.global_step, 1)
            self.assertTrue(math.isfinite(training.training_loss))

            adapter.save_pretrained(export)
            restored = SpeechBrainASRForSpeechRecognition(
                model_path=export,
                device="cpu",
                lazy_load=False,
            )
            self.assertEqual(
                restored.checkpoint_adapter,
                "voicehub-speechbrain-crdnn-asr-safetensors@1",
            )
            self.assertEqual(
                restored.tokenizer.encode_as_ids("A B"),
                [4, 5],
            )

    def test_raw_evaluation_decodes_text_and_reports_corpus_wer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            _write_native_artifact(source)
            wrapper = SpeechBrainASRForSpeechRecognition(
                model_path=source,
                device="cpu",
                lazy_load=False,
            )
            wrapper.decoder = Mock(
                return_value=SimpleNamespace(
                    token_ids=((4, 5), (4, )),
                    scores=(0.0, 0.0),
                ))
            dataset = wrapper.create_training_dataset([
                {
                    "audio": torch.randn(128),
                    "sampling_rate": 16_000,
                    "text": "a   b",
                },
                {
                    "audio": torch.randn(112),
                    "sampling_rate": 16_000,
                    "text": "a",
                },
            ])
            trainer = Trainer(
                model=wrapper,
                args=TrainingArguments(
                    output_dir=str(root / "run"),
                    per_device_eval_batch_size=2,
                    save_strategy="no",
                    use_cpu=True,
                ),
                eval_dataset=dataset,
            )

            ctc_forward = wrapper.model.ctc_linear.forward
            with patch.object(
                    wrapper.model.ctc_linear,
                    "forward",
                    side_effect=ctc_forward,
            ) as ctc_linear:
                metrics = trainer.evaluate()
                prediction = trainer.predict(dataset)

            self.assertEqual(metrics["eval_wer"], 0.0)
            self.assertEqual(prediction.predictions, ["A B", "A"])
            self.assertEqual(prediction.label_ids, ["A B", "A"])
            self.assertEqual(prediction.metrics["test_wer"], 0.0)
            ctc_linear.assert_not_called()
            self.assertEqual(wrapper.decoder.call_count, 2)
            self.assertTrue(all(call.kwargs == {"lm_weight": 0.0} for call in wrapper.decoder.call_args_list))

    def test_raw_training_evaluation_advances_newbob_with_user_wer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            _write_native_artifact(source)
            wrapper = SpeechBrainASRForSpeechRecognition(
                model_path=source,
                device="cpu",
                lazy_load=False,
            )
            wrapper.decoder = Mock(return_value=SimpleNamespace(
                token_ids=((4, ), ),
                scores=(0.0, ),
            ))
            dataset = wrapper.create_training_dataset([{
                "audio": torch.randn(128),
                "sampling_rate": 16_000,
                "text": "A",
            }])
            trainer = Trainer(
                model=wrapper,
                args=TrainingArguments(
                    output_dir=str(root / "run"),
                    max_steps=2,
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    eval_strategy="steps",
                    eval_steps=1,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=dataset,
                eval_dataset=dataset,
                compute_metrics=lambda prediction: {"wer": 0.2},
            )

            trainer.train()

            self.assertEqual(trainer.get_learning_rate(), 0.8)
            self.assertEqual(
                [entry["eval_wer"] for entry in trainer.state.log_history if "eval_wer" in entry],
                [0.2, 0.2],
            )

    @unittest.skipUnless(
        os.environ.get("VOICEHUB_TEST_SPEECHBRAIN_ASR_ASSETS") or
        os.environ.get("VOICEHUB_TEST_RELEASE_ASSETS") == "1",
        "set VOICEHUB_TEST_SPEECHBRAIN_ASR_ASSETS or VOICEHUB_TEST_RELEASE_ASSETS=1 "
        "for release tokenizer checks",
    )
    def test_release_tokenizer_matches_published_sentencepiece(self):
        configured_root = os.environ.get("VOICEHUB_TEST_SPEECHBRAIN_ASR_ASSETS")
        if configured_root is not None:
            tokenizer_path = Path(configured_root) / "tokenizer.ckpt"
        else:
            tokenizer_path = resolve_pretrained_file(
                SPEECHBRAIN_ASR_REPOSITORY,
                "tokenizer.ckpt",
                revision=SPEECHBRAIN_ASR_REVISION,
            )
        tokenizer_payload = tokenizer_path.read_bytes()
        self.assertEqual(len(tokenizer_payload), SPEECHBRAIN_ASR_TOKENIZER_SIZE)
        self.assertEqual(
            hashlib.sha256(tokenizer_payload).hexdigest(),
            SPEECHBRAIN_ASR_TOKENIZER_SHA256,
        )
        tokenizer = SentencePieceUnigramTokenizer.from_model_file(tokenizer_path)

        self.assertEqual(
            tokenizer.encode_as_ids("HELLO WORLD"),
            [16, 83, 27, 472],
        )
        self.assertEqual(
            tokenizer.decode_ids([16, 83, 27, 472]),
            "HELLO WORLD",
        )


if __name__ == "__main__":
    unittest.main()
