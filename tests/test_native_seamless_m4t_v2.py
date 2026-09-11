from __future__ import annotations

import ast
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.models.asr_seamless_m4t_v2 import SeamlessM4Tv2ASRConfig, SeamlessM4Tv2ForSpeechRecognition
from voicehub.models.asr_seamless_m4t_v2.native.artifacts import (
    SeamlessM4Tv2S2TArtifacts,
    resolve_seamless_m4t_v2_artifacts,
)
from voicehub.models.asr_seamless_m4t_v2.native.checkpoint import (
    SeamlessM4Tv2S2TCheckpointAdapter,
    native_seamless_m4t_v2_tensor_shapes,
    seamless_m4t_v2_header_fingerprint,
)
from voicehub.models.asr_seamless_m4t_v2.native.configuration import SeamlessM4Tv2S2TConfig
from voicehub.models.asr_seamless_m4t_v2.native.frontend import SeamlessM4Tv2FeatureExtractor
from voicehub.models.asr_seamless_m4t_v2.native.metadata import SEAMLESS_M4T_V2_CHECKPOINTS
from voicehub.models.asr_seamless_m4t_v2.native.modeling import SeamlessM4Tv2ForSpeechToText
from voicehub.models.asr_seamless_m4t_v2.native.processing import SeamlessM4Tv2Processor
from voicehub.models.asr_seamless_m4t_v2.native.registration import register_seamless_m4t_v2_architecture
from voicehub.models.asr_seamless_m4t_v2.native.runtime import (
    SeamlessM4Tv2S2TRuntime,
    load_seamless_m4t_v2_runtime,
    save_seamless_m4t_v2_runtime,
)
from voicehub.models.asr_seamless_m4t_v2.native.tokenization import (
    SEAMLESS_M4T_V2_LANGUAGE_TO_ID,
    SeamlessM4Tv2Tokenizer,
)
from voicehub.models.asr_seamless_m4t_v2.training_asr_seamless_m4t_v2 import NativeSeamlessM4Tv2TrainingAdapter
from voicehub.runtime.registry import ArchitectureRegistry
from voicehub.training import AutoTrainingAdapter


def _varint(value: int) -> bytes:
    value &= (1 << 64) - 1
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _integer(field: int, value: int) -> bytes:
    return _varint(field << 3) + _varint(value)


def _bytes(field: int, value: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(value)) + value


def _piece(text: str, score: float, piece_type: int = 1) -> bytes:
    return b"".join((
        _bytes(1, text.encode("utf-8")),
        _varint((2 << 3) | 5),
        struct.pack("<f", score),
        _integer(3, piece_type),
    ))


def _write_tokenizer(root: Path) -> SeamlessM4Tv2Tokenizer:
    pieces = (
        ("<unk>", 0.0, 2),
        ("<s>", 0.0, 3),
        ("</s>", 0.0, 3),
        ("▁", -10.0, 1),
        ("h", -11.0, 1),
        ("e", -12.0, 1),
        ("l", -13.0, 1),
        ("o", -14.0, 1),
        ("▁h", 5.0, 1),
        ("▁he", 4.0, 1),
        ("▁hel", 3.0, 1),
        ("▁hell", 2.0, 1),
        ("▁hello", 1.0, 1),
    )
    trainer = b"".join((
        _integer(3, 2),
        _integer(40, 0),
        _integer(41, 1),
        _integer(42, 2),
        _integer(43, -1),
        _bytes(44, b" <unk> "),
    ))
    normalizer = b"".join((
        _bytes(1, b"nmt_nfkc"),
        _integer(3, 1),
        _integer(4, 1),
        _integer(5, 1),
    ))
    model = root / "tokenizer.model"
    model.write_bytes(
        b"".join((
            *(_bytes(1, _piece(text, score, piece_type)) for text, score, piece_type in pieces),
            _bytes(2, trainer),
            _bytes(3, normalizer),
        )))
    added = root / "added_tokens.json"
    added.write_text(
        json.dumps({
            f"__{language}__": token_id
            for language, token_id in SEAMLESS_M4T_V2_LANGUAGE_TO_ID.items()
        }),
        encoding="utf-8",
    )
    return SeamlessM4Tv2Tokenizer.from_files(
        model,
        added_tokens=added,
        expected_sentencepiece_size=None,
    )


def _custom_config(*, vocab_size: int = 256_102) -> SeamlessM4Tv2S2TConfig:
    return SeamlessM4Tv2S2TConfig(
        variant="custom",
        vocab_size=vocab_size,
        hidden_size=8,
        feature_projection_input_dim=8,
        speech_encoder_layers=1,
        speech_encoder_attention_heads=2,
        speech_encoder_intermediate_size=16,
        speech_encoder_layerdrop=0.0,
        speech_encoder_chunk_size=64,
        conv_depthwise_kernel_size=3,
        left_max_position_embeddings=4,
        right_max_position_embeddings=2,
        num_adapter_layers=1,
        adaptor_kernel_size=2,
        adaptor_stride=1,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=16,
        decoder_layerdrop=0.0,
        dropout=0.0,
        attention_dropout=0.0,
        adaptor_dropout=0.0,
        max_position_embeddings=32,
        sampling_rate=16_000,
        num_mel_bins=4,
        feature_stride=2,
        feature_window_length=8,
        feature_hop_length=4,
        feature_fft_size=8,
        max_new_tokens=4,
    )


class NativeSeamlessM4Tv2Tests(unittest.TestCase):

    def test_published_graph_and_inventory_are_exact_and_immutable(self):
        config = SeamlessM4Tv2S2TConfig()
        shapes = native_seamless_m4t_v2_tensor_shapes(config)
        inventory = {name: ("F32", shape) for name, shape in shapes.items()}
        published = SEAMLESS_M4T_V2_CHECKPOINTS["facebook/seamless-m4t-v2-large"]

        self.assertEqual(len(shapes), 1_429)
        self.assertEqual(
            sum(math.prod(shape) for shape in shapes.values()),
            1_501_842_240,
        )
        self.assertEqual(
            seamless_m4t_v2_header_fingerprint(inventory),
            published["s2t_header_fingerprint"],
        )
        self.assertEqual(
            shapes["speech_encoder.encoder.layers.0."
                   "self_attn.distance_embedding.weight"],
            (73, 64),
        )
        self.assertEqual(
            shapes["text_decoder.layers.23.ffn.fc1.weight"],
            (8_192, 1_024),
        )
        with self.assertRaisesRegex(ValueError, "immutable"):
            SeamlessM4Tv2S2TConfig(hidden_size=512)

    def test_reduced_graph_backpropagates_and_checkpoints_every_family(self):
        config = _custom_config(vocab_size=32)
        model = SeamlessM4Tv2ForSpeechToText(config)
        model.gradient_checkpointing_enable()
        model.train()
        features = torch.randn(2, 6, 8)
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
        ])
        labels = torch.tensor([
            [3, 5, 6, 3],
            [3, 7, 3, -100],
        ])

        output = model(
            features,
            attention_mask=mask,
            labels=labels,
        )
        output.loss.backward()

        self.assertEqual(output.logits.shape, (2, 4, 32))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIs(
            model.shared.weight,
            model.text_decoder.embed_tokens.weight,
        )
        self.assertIs(model.shared.weight, model.lm_head.weight)
        self.assertGreater(model.shared.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            model.speech_encoder.feature_projection.projection.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertTrue(model.is_gradient_checkpointing)

    def test_frontend_is_deterministic_and_preserves_reference_geometry(self):
        config = _custom_config()
        extractor = SeamlessM4Tv2FeatureExtractor(config)
        first = torch.linspace(-0.2, 0.2, 48)
        second = torch.sin(torch.arange(40) * 0.2)

        left = extractor((first, second), sampling_rate=16_000)
        right = extractor((first, second), sampling_rate=16_000)

        self.assertEqual(left.input_features.shape, (2, 6, 8))
        self.assertEqual(left.attention_mask.tolist()[0], [1, 1, 1, 1, 1, 0])
        self.assertEqual(left.attention_mask.tolist()[1], [1, 1, 1, 1, 0, 0])
        self.assertTrue(torch.equal(left.input_features, right.input_features))
        self.assertTrue(torch.isfinite(left.input_features).all())
        with self.assertRaisesRegex(ValueError, "16000 Hz"):
            extractor(first, sampling_rate=8_000)

    def test_tokenizer_remaps_sentencepiece_and_adds_target_language(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = _write_tokenizer(Path(directory))

            content = tokenizer.encode_text("hello")
            target = tokenizer.encode_target("hello", language="tur")

        self.assertEqual(content, (13, ))
        self.assertEqual(
            target,
            (3, SEAMLESS_M4T_V2_LANGUAGE_TO_ID["tur"], 13, 3),
        )
        self.assertEqual(tokenizer.decode(target), "hello")
        self.assertEqual(
            tokenizer.generation_prompt("cmn_hant"),
            (SEAMLESS_M4T_V2_LANGUAGE_TO_ID["cmn_Hant"], ),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            tokenizer.language_token_id("zzz")

    def test_checkpoint_adapter_loads_persistent_subset_and_reties_aliases(self):
        config = _custom_config(vocab_size=32)
        source_model = SeamlessM4Tv2ForSpeechToText(config)
        source = {
            name: value.detach().clone()
            for name, value in source_model.state_dict().items() if name not in {
                "lm_head.weight",
                "text_decoder.embed_tokens.weight",
            }
        }
        with torch.device("meta"):
            target = SeamlessM4Tv2ForSpeechToText(
                config,
                initialize=False,
            )

        report = SeamlessM4Tv2S2TCheckpointAdapter().load_assign_streaming(
            target,
            source,
            config,
            device="cpu",
            dtype=torch.float32,
            strict=True,
        )

        self.assertTrue(report.is_compatible)
        self.assertIs(target.shared.weight, target.lm_head.weight)
        self.assertIs(
            target.shared.weight,
            target.text_decoder.embed_tokens.weight,
        )
        self.assertTrue(torch.equal(
            target.shared.weight,
            source_model.shared.weight,
        ))
        malformed = dict(source)
        malformed["unexpected.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(Exception, "unused"):
            SeamlessM4Tv2S2TCheckpointAdapter().load_assign_streaming(
                target,
                malformed,
                config,
                device="cpu",
                strict=True,
            )

    def test_local_artifact_resolution_rejects_unsafe_shard_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in (
                    "config.json",
                    "generation_config.json",
                    "preprocessor_config.json",
                    "tokenizer.model",
                    "tokenizer_config.json",
                    "added_tokens.json",
                    "special_tokens_map.json",
            ):
                (root / filename).write_text("{}", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {
                    "shared.weight": "../outside.safetensors"
                }}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unsafe"):
                resolve_seamless_m4t_v2_artifacts(root)

    def test_portable_s2t_export_reloads_without_unified_model_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = _write_tokenizer(root)
            config = _custom_config()
            processor = SeamlessM4Tv2Processor(config, tokenizer)
            model = SeamlessM4Tv2ForSpeechToText(config)
            artifacts = SeamlessM4Tv2S2TArtifacts(
                source="test",
                revision=None,
                root=root,
                config=root / "unused-config.json",
                generation_config=root / "unused-generation.json",
                preprocessor_config=root / "unused-preprocessor.json",
                tokenizer_model=root / "tokenizer.model",
                tokenizer_config=root / "unused-tokenizer-config.json",
                added_tokens=root / "added_tokens.json",
                special_tokens_map=root / "unused-special.json",
                checkpoint=root / "unused.safetensors",
            )
            runtime = SeamlessM4Tv2S2TRuntime(
                model=model,
                processor=processor,
                config=config,
                artifacts=artifacts,
                generation_config={},
            )
            export = root / "portable"
            export.mkdir()

            save_seamless_m4t_v2_runtime(
                runtime,
                export,
                maximum_shard_bytes=100_000_000,
            )
            restored = load_seamless_m4t_v2_runtime(
                export,
                device="cpu",
                local_files_only=True,
            )

            self.assertTrue((export / "model.safetensors").is_file())
            self.assertFalse(
                any(
                    name.startswith(("text_encoder.", "t2u_model.", "vocoder."))
                    for name in restored.model.state_dict()))
            self.assertTrue(torch.equal(
                restored.model.shared.weight,
                model.shared.weight,
            ))
            self.assertEqual(
                restored.processor.tokenizer.encode_text("hello"),
                (13, ),
            )
            self.assertFalse(restored.model.training)

    def test_public_wrapper_prepares_language_conditioned_training(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = _write_tokenizer(Path(directory))
            native_config = _custom_config()
            processor = SeamlessM4Tv2Processor(native_config, tokenizer)
            wrapper = SeamlessM4Tv2ForSpeechRecognition(
                target_language="eng",
                device="cpu",
            )
            wrapper.model = SeamlessM4Tv2ForSpeechToText(native_config)
            wrapper.native_config = native_config
            wrapper.seamless_processor = processor
            wrapper.training_processor = processor
            audio = torch.linspace(-0.1, 0.1, 48)

            prepared = wrapper.prepare_training_inputs(
                {
                    "audio": audio,
                    "sampling_rate": 16_000,
                    "text": "hello",
                },
                phase="full",
            )
            adapter = AutoTrainingAdapter.from_model(wrapper)
            output = wrapper.model(**prepared)
            output.loss.backward()
            batched = wrapper.prepare_training_inputs(
                {
                    "audio": torch.stack((audio, audio)),
                    "audio_lengths": torch.tensor([48, 40]),
                    "sampling_rate": torch.tensor([16_000, 16_000]),
                    "target_language": ["eng", "eng"],
                    "text": ["hello", "hello"],
                },
                phase="full",
            )
            with self.assertRaisesRegex(ValueError, "homogeneous"):
                wrapper.prepare_training_inputs(
                    {
                        "audio": torch.stack((audio, audio)),
                        "sampling_rate": torch.tensor([16_000, 16_000]),
                        "target_language": ["eng", "deu"],
                        "text": ["hello", "hello"],
                    },
                    phase="full",
                )

        self.assertEqual(
            prepared["labels"][0, :2].tolist(),
            [3, SEAMLESS_M4T_V2_LANGUAGE_TO_ID["eng"]],
        )
        self.assertEqual(batched["labels"].shape[0], 2)
        self.assertEqual(
            batched["labels"][:, 1].tolist(),
            [SEAMLESS_M4T_V2_LANGUAGE_TO_ID["eng"]] * 2,
        )
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsInstance(
            adapter,
            NativeSeamlessM4Tv2TrainingAdapter,
        )
        with self.assertRaisesRegex(ValueError, "translation"):
            wrapper._validate_request(
                language=None,
                task="translate",
                return_timestamps=False,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                num_beams=1,
                max_new_tokens=None,
                hotwords=None,
            )

    def test_public_config_and_architecture_spec_fail_closed(self):
        config = SeamlessM4Tv2ASRConfig(target_language=" TUR ")
        registry = ArchitectureRegistry()
        spec = register_seamless_m4t_v2_architecture(registry=registry)

        self.assertEqual(config.target_language, "tur")
        self.assertEqual(config.to_dict()["target_language"], "tur")
        self.assertTrue(spec.capabilities.training)
        self.assertFalse(spec.capabilities.streaming)
        self.assertEqual(
            spec.metadata["reference_checkpoint_license"],
            "CC-BY-NC-4.0",
        )
        with self.assertRaisesRegex(ValueError, "target language"):
            SeamlessM4Tv2ASRConfig(target_language="zzz")
        with self.assertRaisesRegex(ValueError, "Safetensors"):
            SeamlessM4Tv2ASRConfig(use_safetensors=False)

    def test_active_python_has_no_external_model_runtime_imports(self):
        root = Path(__file__).parents[1] / "voicehub"
        files = tuple((root / 'models/asr_seamless_m4t_v2/native').glob("*.py")) + tuple(
            (root / "models" / "asr_seamless_m4t_v2").glob("*.py"))
        forbidden = {
            "numpy",
            "sentencepiece",
            "tokenizers",
            "torchaudio",
            "transformers",
        }
        imported = set()
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".", 1)[0])

        self.assertFalse(imported & forbidden)


if __name__ == "__main__":
    unittest.main()
