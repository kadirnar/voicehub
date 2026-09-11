from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from voicehub.checkpointing.errors import CheckpointIntegrityError
from voicehub.converters.cosyvoice_speech_tokenizer import convert_audited_cosyvoice_speech_tokenizer
from voicehub.models.cosyvoice.configuration import CosyVoiceConfig
from voicehub.models.cosyvoice.modeling import CosyVoiceForTextToSpeech
from voicehub.models.cosyvoice.native.checkpoint import export_cosyvoice_checkpoint, load_cosyvoice_checkpoint
from voicehub.models.cosyvoice.native.configuration import CosyVoiceArchitectureConfig
from voicehub.models.cosyvoice.native.metadata import COSYVOICE3_SPEECH_TOKENIZER_FILE, S3TOKENIZER_SOURCE_REVISION
from voicehub.models.cosyvoice.native.modeling import CosyVoiceNativeModel
from voicehub.models.cosyvoice.native.runtime import CosyVoiceNativeRuntime, load_cosyvoice_runtime
from voicehub.models.cosyvoice.native.speech_tokenizer import CosyVoiceSpeechTokenizer, CosyVoiceSpeechTokenizerConfig
from voicehub.models.cosyvoice.native.tokenization import (
    END_OF_PROMPT,
    END_OF_TEXT,
    IM_END,
    IM_START,
    CosyVoiceTextTokenizer,
)
from voicehub.optimization import TorchCompileCapabilityReport, TTSOptimizationConfig
from voicehub.optimization.codecs import (
    CodecOptimizationConfig,
    discover_codec_compile_targets,
    resolve_codec_optimization,
)
from voicehub.tokenization import encode_gpt2_token


def _tiny_text_tokenizer(directory: Path) -> CosyVoiceTextTokenizer:
    vocabulary = {encode_gpt2_token(bytes((value, ))): value for value in range(256)}
    special_tokens = (END_OF_TEXT, IM_START, IM_END, END_OF_PROMPT)
    for token_id, spelling in enumerate(special_tokens, 256):
        vocabulary[spelling] = token_id
    vocabulary[encode_gpt2_token(b"hi")] = 260
    (directory / "vocab.json").write_text(
        json.dumps(vocabulary),
        encoding="utf-8",
    )
    (directory / "merges.txt").write_text(
        "#version: 0.2\nh i\n",
        encoding="utf-8",
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps({
            "added_tokens_decoder": {
                str(token_id): {
                    "content": spelling,
                    "special": True,
                }
                for token_id, spelling in enumerate(special_tokens, 256)
            },
        }),
        encoding="utf-8",
    )
    return CosyVoiceTextTokenizer.from_files(
        directory / "vocab.json",
        directory / "merges.txt",
        directory / "tokenizer_config.json",
        validate_published_ids=False,
    )


class CosyVoiceSpeechTokenizerGraphTests(unittest.TestCase):

    def test_official_graph_has_the_complete_audited_namespace(self):
        with torch.device("meta"):
            tokenizer = CosyVoiceSpeechTokenizer()
        state = tokenizer.state_dict()

        self.assertEqual(
            len(state),
            COSYVOICE3_SPEECH_TOKENIZER_FILE["initializer_count"],
        )
        self.assertEqual(
            sum(value.numel() for value in state.values()),
            COSYVOICE3_SPEECH_TOKENIZER_FILE["parameter_count"],
        )
        self.assertEqual(
            tuple(state)[:4],
            (
                "encoder.conv1.weight",
                "encoder.conv1.bias",
                "encoder.conv2.weight",
                "encoder.conv2.bias",
            ),
        )
        self.assertEqual(
            tuple(state)[-2:],
            (
                "quantizer._codebook.project_down.weight",
                "quantizer._codebook.project_down.bias",
            ),
        )
        self.assertFalse(any(name.startswith(("_mel_filters", "_window", "encoder._rope")) for name in state))
        self.assertEqual(
            S3TOKENIZER_SOURCE_REVISION,
            "9bf5d845b5e043ffaf4657f4942939091c7697a2",
        )

    def test_tiny_forward_and_fsq_match_the_published_equations(self):
        torch.manual_seed(31)
        tokenizer = CosyVoiceSpeechTokenizer(CosyVoiceSpeechTokenizerConfig.tiny())
        features = torch.randn(2, 8, 41)
        lengths = torch.tensor([41, 35])

        tokens, token_lengths = tokenizer(features, lengths)
        hidden, expected_lengths = tokenizer.encoder(features, lengths)
        projected = tokenizer.quantizer._codebook.project_down(hidden.reshape(-1, hidden.shape[-1])).float()
        digits = (projected.tanh() * 0.9990000128746033).round() + 1.0
        powers = 3.0**torch.arange(8, dtype=digits.dtype)
        expected = (digits * powers).sum(dim=-1).reshape(hidden.shape[:2]).long()

        self.assertEqual(token_lengths.tolist(), [11, 9])
        self.assertTrue(torch.equal(token_lengths, expected_lengths))
        self.assertTrue(torch.equal(tokens, expected))
        self.assertGreaterEqual(int(tokens.min()), 0)
        self.assertLess(int(tokens.max()), 6_561)

    def test_strict_safetensors_load_preserves_forward_results_and_buffers(self):
        torch.manual_seed(37)
        source = CosyVoiceSpeechTokenizer(CosyVoiceSpeechTokenizerConfig.tiny())
        features = torch.randn(1, 8, 37)
        lengths = torch.tensor([37])
        expected = source(features, lengths)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = export_cosyvoice_checkpoint(
                source,
                Path(temporary) / "speech_tokenizer.safetensors",
                component="speech_tokenizer",
            )
            with torch.device("meta"):
                restored = CosyVoiceSpeechTokenizer(CosyVoiceSpeechTokenizerConfig.tiny())
            load_cosyvoice_checkpoint(
                restored,
                checkpoint,
                component="speech_tokenizer",
            )
            actual = restored(features, lengths)

        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertFalse(any(value.device.type == "meta" for value in restored.buffers()))

    def test_converter_rejects_every_unaudited_onnx_before_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "speech_tokenizer_v3.onnx"
            source.write_bytes(b"not the immutable graph")
            with self.assertRaises(CheckpointIntegrityError):
                convert_audited_cosyvoice_speech_tokenizer(
                    source,
                    Path(temporary) / "speech_tokenizer.safetensors",
                )


class CosyVoiceSpeechTokenizerRuntimeTests(unittest.TestCase):

    def test_raw_training_audio_and_precomputed_tokens_share_one_boundary(self):
        torch.manual_seed(41)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            speech_tokenizer = CosyVoiceSpeechTokenizer(CosyVoiceSpeechTokenizerConfig.tiny())
            runtime = CosyVoiceNativeRuntime(
                CosyVoiceNativeModel(CosyVoiceArchitectureConfig.tiny()),
                _tiny_text_tokenizer(root),
                speech_tokenizer,
            )
            waveform = torch.randn(3_200)
            expected, expected_lengths = runtime.extract_speech_tokens(
                waveform,
                sampling_rate=16_000,
            )
            raw = runtime.prepare_language_batch([{
                "text": "hi",
                "speech_audio": waveform,
                "speech_sampling_rate": 16_000,
            }])
            precomputed = runtime.prepare_language_batch([{
                "text": "hi",
                "speech_tokens": expected[0, :expected_lengths[0]],
            }])

        self.assertTrue(torch.equal(raw["speech_tokens"], precomputed["speech_tokens"]))
        self.assertTrue(torch.equal(raw["speech_lengths"], precomputed["speech_lengths"]))
        self.assertTrue(runtime.supports_raw_speech_tokens)
        self.assertTrue(all(not parameter.requires_grad for parameter in speech_tokenizer.parameters()))

    def test_native_runtime_export_reloads_optional_tiny_tokenizer(self):
        torch.manual_seed(43)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = CosyVoiceNativeRuntime(
                CosyVoiceNativeModel(CosyVoiceArchitectureConfig.tiny()),
                _tiny_text_tokenizer(root),
                CosyVoiceSpeechTokenizer(CosyVoiceSpeechTokenizerConfig.tiny()),
            )
            export = runtime.save_pretrained(root / "export")
            restored = load_cosyvoice_runtime(export)

        self.assertTrue(restored.supports_raw_speech_tokens)
        self.assertEqual(
            restored.speech_tokenizer.config,
            CosyVoiceSpeechTokenizerConfig.tiny(),
        )
        self.assertEqual(
            tuple(restored.speech_tokenizer.state_dict()),
            tuple(runtime.speech_tokenizer.state_dict()),
        )

    def test_public_wrapper_optimizes_and_restores_all_native_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = CosyVoiceNativeRuntime(
                CosyVoiceNativeModel(CosyVoiceArchitectureConfig.tiny()),
                _tiny_text_tokenizer(root),
                CosyVoiceSpeechTokenizer(CosyVoiceSpeechTokenizerConfig.tiny()),
            )
            wrapper = CosyVoiceForTextToSpeech(
                CosyVoiceConfig(name_or_path="unused"),
                device="cpu",
            )
            wrapper._runtime = runtime
            expected_owners = (
                runtime.model.llm.llm.model.model,
                runtime.model.flow.decoder.estimator,
                runtime.model.hift,
                runtime.speech_tokenizer,
            )
            compiled_owners = []
            report = TorchCompileCapabilityReport(
                available=True,
                backend="eager",
                backend_available=True,
                torch_version=torch.__version__,
                available_backends=("eager", ),
            )

            with (
                    mock.patch(
                        "voicehub.optimization.torch_compile."
                        "inspect_torch_compile",
                        return_value=report,
                    ),
                    mock.patch.object(
                        torch,
                        "compile",
                        side_effect=lambda function, **_kwargs:
                        (compiled_owners.append(function.__self__) or function),
                    ),
            ):
                result = wrapper.optimize(
                    TTSOptimizationConfig(
                        attn_implementation="native",
                        kernel_backend="native",
                        compile=True,
                        compile_config={
                            "backend": "eager",
                        },
                    ))

            self.assertIs(result.model, runtime)
            self.assertEqual(
                tuple(compiled_owners),
                expected_owners,
            )
            self.assertTrue(all("forward" in owner.__dict__ for owner in expected_owners), )
            self.assertIs(
                wrapper.restore_tts_optimization(),
                runtime,
            )
            self.assertTrue(all("forward" not in owner.__dict__ for owner in expected_owners), )

    def test_shared_codec_optimizer_discovers_and_compiles_encoder_boundary(self):
        torch.manual_seed(47)
        tokenizer = CosyVoiceSpeechTokenizer(CosyVoiceSpeechTokenizerConfig.tiny())
        targets = discover_codec_compile_targets(tokenizer)
        self.assertEqual(
            tuple(target.attribute for target in targets),
            ("forward", ),
        )
        plan = resolve_codec_optimization(
            tokenizer,
            CodecOptimizationConfig(
                kernel_backend="native",
                compile=True,
                compile_config={
                    "backend": "eager",
                    "fullgraph": False,
                },
            ),
        )
        features = torch.randn(1, 8, 33)
        lengths = torch.tensor([33])
        expected = tokenizer(features, lengths)
        application = plan.apply(tokenizer)
        actual = application.model(features, lengths)
        application.restore()

        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))


if __name__ == "__main__":
    unittest.main()
