from __future__ import annotations

import ast
import base64
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.asr_cohere import CohereASRConfig, CohereForSpeechRecognition, NativeCohereASRTrainingAdapter
from voicehub.models.asr_cohere.native.artifacts import CohereAsrArtifacts, resolve_cohere_asr_artifacts
from voicehub.models.asr_cohere.native.checkpoint import (
    CohereAsrCheckpointAdapter,
    cohere_asr_header_fingerprint,
    native_cohere_asr_tensor_shapes,
)
from voicehub.models.asr_cohere.native.configuration import SUPPORTED_LANGUAGES, CohereAsrConfig
from voicehub.models.asr_cohere.native.metadata import COHERE_ASR_CHECKPOINTS
from voicehub.models.asr_cohere.native.modeling import CohereAsrForConditionalGeneration
from voicehub.models.asr_cohere.native.processing import CohereAsrProcessor
from voicehub.models.asr_cohere.native.runtime import CohereAsrRuntime, load_cohere_asr_runtime, save_cohere_asr_runtime
from voicehub.models.asr_cohere.native.tokenization import CohereAsrTokenizer, load_cohere_tokenizer
from voicehub.models.asr_parakeet_tdt.native.configuration import ParakeetEncoderConfig
from voicehub.policies.architecture_dependencies import inspect_native_runtime
from voicehub.processing.waveform import save_pcm_wave
from voicehub.tasks import SpeechTask
from voicehub.tokenization.assets import TokenizerAssetError
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily, get_training_spec

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "voicehub"

_SPECIAL_TOKENS = (
    "<unk>",
    "<|nospeech|>",
    "<pad>",
    "<|endoftext|>",
    "<|startoftranscript|>",
    "<|pnc|>",
    "<|nopnc|>",
    "<|startofcontext|>",
    "<|itn|>",
    "<|noitn|>",
    "<|timestamp|>",
    "<|notimestamp|>",
    "<|diarize|>",
    "<|nodiarize|>",
    "<|spkchange|>",
    "<|audioseparator|>",
    "<|emo:undefined|>",
) + tuple(f"<|{language}|>" for language in SUPPORTED_LANGUAGES)
_TEXT_TOKENS = ("▁", "h", "i", "▁h", "▁hi")
_TOKENS = _SPECIAL_TOKENS + _TEXT_TOKENS
_VOCABULARY = {token: token_id for token_id, token in enumerate(_TOKENS)}


def _identity_precompiled_charsmap() -> str:
    payload = (struct.pack("<I", 4) + struct.pack("<I", 0) + b"\0")
    return base64.b64encode(payload).decode("ascii")


def _tokenizer_document() -> dict[str, object]:
    return {
        "version":
        "1.0",
        "truncation":
        None,
        "padding":
        None,
        "added_tokens": [{
            "id": _VOCABULARY[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        } for token in _SPECIAL_TOKENS],
        "normalizer": {
            "type":
            "Sequence",
            "normalizers": [
                {
                    "type": "Precompiled",
                    "precompiled_charsmap": (_identity_precompiled_charsmap()),
                },
                {
                    "type": "Prepend",
                    "prepend": "▁",
                },
                {
                    "type": "Replace",
                    "pattern": {
                        "String": " ",
                    },
                    "content": "▁",
                },
            ],
        },
        "pre_tokenizer":
        None,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {
                    "Sequence": {
                        "id": "A",
                        "type_id": 0,
                    },
                },
            ],
            "pair": [
                {
                    "Sequence": {
                        "id": "A",
                        "type_id": 0,
                    },
                },
                {
                    "Sequence": {
                        "id": "B",
                        "type_id": 1,
                    },
                },
            ],
            "special_tokens": {},
        },
        "decoder": {
            "type":
            "Sequence",
            "decoders": [
                {
                    "type": "Replace",
                    "pattern": {
                        "String": "▁",
                    },
                    "content": " ",
                },
                {
                    "type": "ByteFallback",
                },
                {
                    "type": "Fuse",
                },
                {
                    "type": "Strip",
                    "content": " ",
                    "start": 1,
                    "stop": 0,
                },
            ],
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": _VOCABULARY,
            "merges": [
                ["▁", "h"],
                ["▁h", "i"],
            ],
        },
    }


def _write_tokenizer_assets(directory: Path) -> CohereAsrTokenizer:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tokenizer.json").write_text(
        json.dumps(_tokenizer_document(), ensure_ascii=False),
        encoding="utf-8",
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps({
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "eos_token": "<|endoftext|>",
            "bos_token": "<|startoftranscript|>",
        }),
        encoding="utf-8",
    )
    return CohereAsrTokenizer.from_files(
        directory / "tokenizer.json",
        directory / "tokenizer_config.json",
    )


def _tiny_config(*, mask_prompt_loss: bool = False) -> CohereAsrConfig:
    encoder = ParakeetEncoderConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=16,
        hidden_act="silu",
        attention_bias=True,
        convolution_bias=True,
        conv_kernel_size=3,
        subsampling_factor=8,
        subsampling_conv_channels=2,
        num_mel_bins=8,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=64,
        scale_input=False,
        initializer_range=0.02,
    )
    return CohereAsrConfig(
        encoder_config=encoder,
        vocab_size=len(_VOCABULARY),
        decoder_hidden_size=8,
        decoder_num_hidden_layers=2,
        decoder_num_attention_heads=2,
        decoder_intermediate_size=16,
        decoder_max_position_embeddings=32,
        decoder_start_token_id=_VOCABULARY["▁"],
        sample_rate=16_000,
        hop_length=4,
        n_fft=16,
        win_length=8,
        dither=0.0,
        max_audio_clip_s=0.02,
        overlap_chunk_second=0.005,
        min_energy_window_samples=16,
        mask_prompt_loss=mask_prompt_loss,
    )


def _runtime(directory: Path, *, mask_prompt_loss: bool = False) -> CohereAsrRuntime:
    directory.mkdir(parents=True, exist_ok=True)
    config = _tiny_config(mask_prompt_loss=mask_prompt_loss)
    _write_tokenizer_assets(directory)
    model = CohereAsrForConditionalGeneration(config)
    processor = CohereAsrProcessor.from_files(
        featurizer=model.preprocessor.featurizer,
        config=config,
        tokenizer_path=directory / "tokenizer.json",
        tokenizer_config_path=directory / "tokenizer_config.json",
    )
    artifacts = CohereAsrArtifacts(
        source=str(directory),
        revision=None,
        config=directory / "config.json",
        generation_config=directory / "generation_config.json",
        preprocessor_config=directory / "preprocessor_config.json",
        processor_config=directory / "processor_config.json",
        tokenizer=directory / "tokenizer.json",
        tokenizer_config=directory / "tokenizer_config.json",
        checkpoint=directory / "model.safetensors",
    )
    return CohereAsrRuntime(
        model=model,
        processor=processor,
        config=config,
        artifacts=artifacts,
        generation_config={
            "bos_token_id": config.bos_token_id,
            "decoder_start_token_id": config.decoder_start_token_id,
            "eos_token_id": config.eos_token_id,
            "pad_token_id": config.pad_token_id,
        },
    )


class CohereAsrProvenanceTests(unittest.TestCase):

    def test_published_inventory_and_source_are_immutable(self):
        record = COHERE_ASR_CHECKPOINTS["CohereLabs/cohere-transcribe-03-2026"]
        self.assertEqual(
            record["revision"],
            "b1eacc2686a3d08ceaae5f24a88b1d519620bc09",
        )
        self.assertEqual(record["license"], "Apache-2.0")
        self.assertEqual(record["tensors"], 2_152)
        self.assertEqual(record["state_values"], 2_065_804_096)
        self.assertEqual(record["parameters"], 2_047_822_080)
        self.assertEqual(record["tensor_data_bytes"], 4_131_608_480)
        self.assertEqual(
            record["header_fingerprint"],
            "06a76e1e91f509c865013ce962a695a05"
            "b6a50ae0290d1258910c660ccb06292",
        )
        source = json.loads(
            (PACKAGE_ROOT / 'models/asr_cohere/native/SOURCE.json').read_text(encoding="utf-8"))
        self.assertEqual(source["implementation"], "voicehub-native")
        self.assertTrue(source["verified_scope"]["training"])
        self.assertTrue((PACKAGE_ROOT / 'models/asr_cohere/native/THIRD_PARTY_LICENSE').is_file())

    def test_default_graph_exactly_matches_published_header_namespace(self):
        config = CohereAsrConfig()
        shapes = native_cohere_asr_tensor_shapes(config)

        self.assertEqual(len(shapes), 2_152)
        self.assertEqual(
            sum(math.prod(shape) for shape in shapes.values()),
            2_065_804_096,
        )
        self.assertEqual(
            shapes["encoder.layers.47.self_attn.linear_q.weight"],
            (1_280, 1_280),
        )
        self.assertEqual(
            shapes["transf_decoder._decoder.layers.7."
                   "second_sub_layer.query_net.weight"],
            (1_024, 1_024),
        )
        self.assertEqual(
            shapes["log_softmax.mlp.layer0.weight"],
            (16_384, 1_024),
        )
        with torch.device("meta"):
            model = CohereAsrForConditionalGeneration(config)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            2_047_822_080,
        )
        self.assertEqual(
            sum(tensor.numel() for tensor in model.state_dict().values() if tensor.dtype == torch.int64),
            48,
        )

    def test_header_fingerprint_is_order_independent_and_shape_sensitive(self):
        first = {
            "b": ("BF16", (2, 3)),
            "a": ("I64", ()),
        }
        second = {
            "a": ("I64", ()),
            "b": ("BF16", (2, 3)),
        }
        changed = {
            **second,
            "b": ("BF16", (3, 2)),
        }

        self.assertEqual(
            cohere_asr_header_fingerprint(first),
            cohere_asr_header_fingerprint(second),
        )
        self.assertNotEqual(
            cohere_asr_header_fingerprint(first),
            cohere_asr_header_fingerprint(changed),
        )

    def test_active_graph_has_no_external_runtime_import(self):
        violations = inspect_native_runtime(
            PACKAGE_ROOT,
            directories=(
                'models/asr_cohere/native',
                "models/asr_cohere",
            ),
        )
        self.assertEqual(
            violations,
            (),
            "\n".join(str(value) for value in violations),
        )

    def test_public_facades_do_not_import_torch(self):
        script = """
import builtins
import sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("torch blocked by test")
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import voicehub.models.asr_cohere.native
import voicehub.models.asr_cohere
assert "torch" not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sources_do_not_hide_dynamic_external_imports(self):
        paths = tuple((PACKAGE_ROOT / 'models/asr_cohere/native').glob("*.py")) + tuple(
            (PACKAGE_ROOT / "models/asr_cohere").glob("*.py"))
        forbidden = {
            "datasets",
            "librosa",
            "nemo",
            "numpy",
            "safetensors",
            "sentencepiece",
            "torchaudio",
            "transformers",
        }
        discovered = set()
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    discovered.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    discovered.add(node.module.split(".", 1)[0])
        self.assertFalse(discovered & forbidden)


class CohereAsrTokenizerAndProcessorTests(unittest.TestCase):

    def test_tokenizer_round_trip_prompt_and_reserved_token_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer = _write_tokenizer_assets(Path(temporary))
            encoded = tokenizer.encode("hi")

            self.assertEqual(
                encoded.input_ids,
                (_VOCABULARY["▁hi"], ),
            )
            self.assertEqual(tokenizer.decode(encoded.input_ids), "hi")
            self.assertEqual(
                tokenizer.batch_decode([encoded.input_ids, encoded.input_ids]),
                ["hi", "hi"],
            )
            with self.assertRaisesRegex(ValueError, "reserved control"):
                tokenizer.encode("<|en|>")
            self.assertEqual(
                tokenizer.encode(
                    "<|en|>",
                    allow_special_tokens=True,
                ).input_ids,
                (_VOCABULARY["<|en|>"], ),
            )
            with self.assertRaisesRegex(ValueError, "automatic BOS"):
                tokenizer.encode("hi", add_special_tokens=True)

    def test_malformed_tokenizer_graph_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_tokenizer_assets(root)
            document = _tokenizer_document()
            document["model"]["byte_fallback"] = False
            (root / "tokenizer.json").write_text(
                json.dumps(document, ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                    TokenizerAssetError,
                    "byte_fallback",
            ):
                load_cohere_tokenizer(
                    root / "tokenizer.json",
                    root / "tokenizer_config.json",
                )

    def test_processor_builds_exact_prompt_and_teacher_forcing_shift(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            prompt = runtime.processor.get_decoder_prompt_ids(
                "en",
                punctuation=True,
            )
            expected_tokens = (
                "▁",
                "<|startofcontext|>",
                "<|startoftranscript|>",
                "<|emo:undefined|>",
                "<|en|>",
                "<|en|>",
                "<|pnc|>",
                "<|noitn|>",
                "<|notimestamp|>",
                "<|nodiarize|>",
            )
            self.assertEqual(
                prompt,
                tuple(_VOCABULARY[token] for token in expected_tokens),
            )
            prepared = runtime.processor(
                (
                    torch.linspace(-0.3, 0.3, 64),
                    torch.linspace(0.2, -0.2, 48),
                ),
                language="en",
                text=("hi", "hi"),
                sampling_rate=16_000,
            )

            self.assertEqual(prepared["input_features"].shape[0], 2)
            self.assertEqual(
                int(prepared["attention_mask"][0].sum()),
                64 // runtime.config.hop_length,
            )
            complete = (prompt + (_VOCABULARY["▁hi"], ) + (runtime.config.eos_token_id, ))
            self.assertEqual(
                tuple(prepared["decoder_input_ids"][0].tolist()),
                complete[:-1],
            )
            self.assertEqual(
                tuple(prepared["labels"][0].tolist()),
                complete[1:],
            )
            self.assertTrue(torch.all(prepared["labels"][~prepared["decoder_attention_mask"]] == -100))

    def test_prompt_loss_masking_is_explicit_and_source_configured(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _runtime(
                Path(temporary),
                mask_prompt_loss=True,
            )
            prepared = runtime.processor(
                torch.linspace(-0.3, 0.3, 64),
                language="en",
                text="hi",
                sampling_rate=16_000,
            )
            prompt_length = len(runtime.processor.get_decoder_prompt_ids("en"))

            self.assertTrue(torch.all(prepared["labels"][:, :prompt_length - 1] == -100))
            self.assertEqual(
                int(prepared["labels"][0, -1]),
                runtime.config.eos_token_id,
            )

    def test_long_form_splits_at_quiet_boundary_and_reassembles_by_language(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            waveform = torch.ones(400)
            waveform[240:272] = 0.0
            chunks = (runtime.processor.feature_extractor.split_long_waveform(waveform))
            prepared = runtime.processor(
                waveform,
                language="en",
                sampling_rate=16_000,
            )

            self.assertGreater(len(chunks), 1)
            self.assertEqual(
                sum(chunk.numel() for chunk in chunks),
                waveform.numel(),
            )
            self.assertEqual(
                len(prepared["audio_chunk_index"]),
                len(chunks),
            )
            index = ((0, 0), (0, 1), (1, None))
            self.assertEqual(
                runtime.processor.reassemble_chunk_texts(
                    ("hello", "world", "solo"),
                    index,
                    language="en",
                ),
                ("hello world", "solo"),
            )
            self.assertEqual(
                runtime.processor.reassemble_chunk_texts(
                    ("日", "本", "solo"),
                    index,
                    language="ja",
                ),
                ("日本", "solo"),
            )
            with self.assertRaisesRegex(
                    ValueError,
                    "single-clip duration",
            ):
                runtime.processor(
                    waveform,
                    language="en",
                    text="hi",
                    sampling_rate=16_000,
                )

    def test_frontend_is_batch_composition_invariant(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            waveform = torch.linspace(-0.4, 0.3, 64)
            alone = runtime.processor.feature_extractor(
                waveform,
                sampling_rate=16_000,
                chunk_long_audio=False,
            )
            batch = runtime.processor.feature_extractor(
                (waveform, torch.zeros(48)),
                sampling_rate=16_000,
                chunk_long_audio=False,
            )
            valid = int(alone["attention_mask"][0].sum())

            torch.testing.assert_close(
                alone["input_features"][0, :valid],
                batch["input_features"][0, :valid],
                rtol=0,
                atol=0,
            )

    def test_optional_transformers_frontend_parity(self):
        try:
            import numpy as np
            from transformers.models.cohere_asr import feature_extraction_cohere_asr

            TransformersFeatureExtractor = (feature_extraction_cohere_asr.CohereAsrFeatureExtractor)
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"Optional Transformers frontend unavailable: {error}")

        with tempfile.TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            try:
                upstream = TransformersFeatureExtractor(
                    feature_size=8,
                    sampling_rate=16_000,
                    hop_length=4,
                    n_fft=16,
                    win_length=8,
                    preemphasis=0.97,
                    dither=0.0,
                    max_audio_clip_s=0.02,
                    overlap_chunk_second=0.005,
                    min_energy_window_samples=16,
                )
            except (ImportError, ModuleNotFoundError) as error:
                self.skipTest(f"Optional librosa frontend unavailable: {error}")
            waveform = torch.linspace(-0.35, 0.25, 64)
            native = runtime.processor.feature_extractor(
                waveform,
                sampling_rate=16_000,
                chunk_long_audio=False,
            )
            reference = upstream(
                waveform.numpy().astype(np.float32),
                sampling_rate=16_000,
                return_tensors="pt",
            )

            torch.testing.assert_close(
                native["input_features"],
                reference["input_features"],
                rtol=2e-4,
                atol=2e-5,
            )
            torch.testing.assert_close(
                native["attention_mask"],
                reference["attention_mask"].bool(),
                rtol=0,
                atol=0,
            )


class CohereAsrModelTests(unittest.TestCase):

    def test_forward_loss_full_backward_and_gradient_checkpointing(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            runtime.model.train()
            runtime.model.gradient_checkpointing_enable()
            prepared = runtime.processor(
                (
                    torch.linspace(-0.3, 0.3, 96),
                    torch.linspace(0.3, -0.3, 80),
                ),
                language="en",
                text=("hi", "hi"),
                sampling_rate=16_000,
            )
            output = runtime.model(
                **{
                    name: prepared[name]
                    for name in (
                        "input_features",
                        "attention_mask",
                        "decoder_input_ids",
                        "decoder_attention_mask",
                        "labels",
                    )
                })

            self.assertTrue(runtime.model.gradient_checkpointing)
            self.assertEqual(
                output.logits.shape[:2],
                prepared["labels"].shape,
            )
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(runtime.model.encoder.pre_encode.conv[0].weight.grad)
            self.assertIsNotNone(runtime.model.encoder.layers[0].self_attn.linear_q.weight.grad)
            self.assertIsNotNone(
                runtime.model.transf_decoder._decoder.layers[1].second_sub_layer.query_net.weight.grad)
            self.assertIsNotNone(runtime.model.log_softmax.mlp.layer0.bias.grad)

    def test_greedy_generation_stops_at_eos_and_rejects_unverified_modes(self):
        config = _tiny_config()
        model = CohereAsrForConditionalGeneration(config).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.log_softmax.mlp.layer0.bias[config.eos_token_id] = 10.0
        features = torch.zeros(1, 32, 8)
        mask = torch.ones(1, 32, dtype=torch.bool)
        prompt = torch.tensor(
            [[config.decoder_start_token_id]],
            dtype=torch.long,
        )

        generated = model.generate(
            features,
            mask,
            prompt,
            max_new_tokens=3,
        )
        self.assertEqual(
            generated.sequences.tolist(),
            [[config.decoder_start_token_id, config.eos_token_id]],
        )
        with self.assertRaisesRegex(ValueError, "sampling"):
            model.generate(
                features,
                mask,
                prompt,
                do_sample=True,
            )
        with self.assertRaisesRegex(ValueError, "greedy"):
            model.generate(
                features,
                mask,
                prompt,
                num_beams=2,
            )
        with self.assertRaisesRegex(ValueError, "KV cache"):
            model.generate(
                features,
                mask,
                prompt,
                use_cache=True,
            )

    def test_forward_rejects_invalid_labels_without_silent_clamping(self):
        config = _tiny_config()
        model = CohereAsrForConditionalGeneration(config).eval()
        features = torch.zeros(1, 32, 8)
        mask = torch.ones(1, 32, dtype=torch.bool)
        invalid = torch.tensor([[config.vocab_size]])

        with self.assertRaisesRegex(ValueError, "out-of-vocabulary"):
            model(
                input_features=features,
                attention_mask=mask,
                labels=invalid,
            )
        with self.assertRaisesRegex(ValueError, "no supervised"):
            model(
                input_features=features,
                attention_mask=mask,
                labels=torch.tensor([[-100]]),
            )

    def test_checkpoint_loader_validates_before_assignment(self):
        config = _tiny_config()
        source_model = CohereAsrForConditionalGeneration(config)
        state = {name: value.detach().clone() for name, value in source_model.state_dict().items()}
        adapter = CohereAsrCheckpointAdapter()

        with torch.device("meta"):
            missing_target = CohereAsrForConditionalGeneration(config)
        missing = dict(state)
        missing.pop(next(iter(missing)))
        with self.assertRaises(CheckpointCompatibilityError):
            adapter.load_assign_streaming(
                missing_target,
                missing,
                config,
                device="cpu",
                dtype=torch.float32,
            )
        self.assertTrue(all(value.device.type == "meta" for value in missing_target.state_dict().values()))

        with torch.device("meta"):
            dtype_target = CohereAsrForConditionalGeneration(config)
        wrong_dtype = dict(state)
        name = "encoder.pre_encode.out.weight"
        wrong_dtype[name] = wrong_dtype[name].to(torch.int64)
        with self.assertRaisesRegex(
                CheckpointCompatibilityError,
                "incompatible tensor dtypes",
        ):
            adapter.load_assign_streaming(
                dtype_target,
                wrong_dtype,
                config,
                device="cpu",
                dtype=torch.float32,
            )
        self.assertTrue(all(value.device.type == "meta" for value in dtype_target.state_dict().values()))

        with torch.device("meta"):
            tied_target = CohereAsrForConditionalGeneration(config)
        wrong_tie = dict(state)
        output_name = "log_softmax.mlp.layer0.weight"
        wrong_tie[output_name] = wrong_tie[output_name] + 1.0
        with self.assertRaisesRegex(
                CheckpointCompatibilityError,
                "tied input/output",
        ):
            adapter.load_assign_streaming(
                tied_target,
                wrong_tie,
                config,
                device="cpu",
                dtype=torch.float32,
            )
        self.assertTrue(all(value.device.type == "meta" for value in tied_target.state_dict().values()))


class CohereAsrArtifactTests(unittest.TestCase):

    def test_strict_sharded_export_reloads_the_exact_graph(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _runtime(root / "source")
            destination = save_cohere_asr_runtime(
                runtime,
                root / "export",
                maximum_shard_bytes=8_192,
            )

            self.assertTrue((destination / "model.safetensors.index.json").is_file())
            self.assertFalse((destination / "model.safetensors").exists())
            reloaded = load_cohere_asr_runtime(
                destination,
                device="cpu",
                compute_dtype="float32",
                for_training=True,
            )
            self.assertTrue(reloaded.model.training)
            self.assertEqual(
                set(reloaded.model.state_dict()),
                set(runtime.model.state_dict()),
            )
            for name in runtime.model.state_dict():
                torch.testing.assert_close(
                    reloaded.model.state_dict()[name],
                    runtime.model.state_dict()[name],
                    rtol=0,
                    atol=0,
                )
            self.assertIs(
                reloaded.model.log_softmax.mlp.layer0.weight,
                reloaded.model.transf_decoder._embedding.token_embedding.weight,
            )

    def test_malformed_exports_leave_no_partial_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _runtime(root / "source")
            state = dict(runtime.model.state_dict())
            cases = []

            missing = dict(state)
            missing.pop(next(iter(missing)))
            cases.append(("missing", missing, ValueError))

            wrong_shape = dict(state)
            name = "encoder.pre_encode.out.weight"
            wrong_shape[name] = wrong_shape[name][:-1]
            cases.append(("shape", wrong_shape, ValueError))

            wrong_dtype = dict(state)
            wrong_dtype[name] = wrong_dtype[name].to(torch.float64)
            cases.append(("dtype", wrong_dtype, TypeError))

            sparse = dict(state)
            sparse[name] = sparse[name].to_sparse()
            cases.append(("sparse", sparse, TypeError))

            quantized = dict(state)
            quantized[name] = torch.quantize_per_tensor(
                quantized[name],
                scale=0.1,
                zero_point=0,
                dtype=torch.qint8,
            )
            cases.append(("quantized", quantized, TypeError))

            tied = dict(state)
            output_name = "log_softmax.mlp.layer0.weight"
            tied[output_name] = tied[output_name] + 1.0
            cases.append(("tied", tied, ValueError))

            for label, malformed, error_type in cases:
                destination = root / label
                with self.subTest(label=label):
                    with self.assertRaises(error_type):
                        save_cohere_asr_runtime(
                            runtime,
                            destination,
                            state_dict=malformed,
                        )
                    self.assertFalse(destination.exists())

    def test_local_artifact_resolution_rejects_ambiguity_and_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _runtime(root / "source")
            destination = save_cohere_asr_runtime(
                runtime,
                root / "export",
                maximum_shard_bytes=8_192,
            )
            (destination / "model.safetensors").write_bytes(b"placeholder")
            with self.assertRaisesRegex(ValueError, "both single-file"):
                resolve_cohere_asr_artifacts(destination)
            (destination / "model.safetensors").unlink()

            index_path = destination / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            first = next(iter(index["weight_map"]))
            index["weight_map"][first] = "../escape.safetensors"
            index_path.write_text(
                json.dumps(index),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                resolve_cohere_asr_artifacts(destination)

    def test_remote_resolution_requires_one_proven_immutable_snapshot(self):
        required = (
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshot"
            other = Path(temporary) / "other"
            root.mkdir()
            other.mkdir()
            for filename in required:
                (root / filename).write_text("{}", encoding="utf-8")
            (other / "tokenizer.json").write_text("{}", encoding="utf-8")

            def resolve(
                repo_id,
                filename,
                **kwargs,
            ):
                del repo_id, kwargs
                if filename == "model.safetensors.index.json":
                    raise FileNotFoundError(filename)
                return root / filename

            with (
                    patch(
                        "voicehub.models.asr_cohere.native.artifacts."
                        "resolve_pretrained_file",
                        side_effect=resolve,
                    ),
                    patch(
                        "voicehub.models.asr_cohere.native.artifacts."
                        "get_cached_hugging_face_commit",
                        return_value=None,
                    ),
            ):
                with self.assertRaisesRegex(RuntimeError, "immutable"):
                    resolve_cohere_asr_artifacts(
                        "organization/mutable-model",
                        revision="main",
                    )

            revision = "1" * 40
            with (
                    patch(
                        "voicehub.models.asr_cohere.native.artifacts."
                        "resolve_pretrained_file",
                        side_effect=resolve,
                    ),
                    patch(
                        "voicehub.models.asr_cohere.native.artifacts."
                        "get_cached_hugging_face_commit",
                        return_value=revision,
                    ),
            ):
                resolved = resolve_cohere_asr_artifacts(
                    "organization/model",
                    revision="main",
                )
            self.assertEqual(resolved.revision, revision)

            def incoherent(
                repo_id,
                filename,
                **kwargs,
            ):
                del repo_id, kwargs
                if filename == "model.safetensors.index.json":
                    raise FileNotFoundError(filename)
                if filename == "tokenizer.json":
                    return other / filename
                return root / filename

            with (
                    patch(
                        "voicehub.models.asr_cohere.native.artifacts."
                        "resolve_pretrained_file",
                        side_effect=incoherent,
                    ),
                    patch(
                        "voicehub.models.asr_cohere.native.artifacts."
                        "get_cached_hugging_face_commit",
                        return_value=revision,
                    ),
            ):
                with self.assertRaisesRegex(RuntimeError, "one immutable"):
                    resolve_cohere_asr_artifacts(
                        "organization/model",
                        revision="main",
                    )


class CohereAsrPublicTrainingTests(unittest.TestCase):

    def test_shared_training_registry_selects_native_full_model_adapter(self):
        wrapper = CohereForSpeechRecognition(
            lazy_load=True,
            device="cpu",
        )
        adapter = AutoTrainingAdapter.from_model(wrapper)
        spec = get_training_spec("asr_cohere")

        self.assertIsInstance(adapter, NativeCohereASRTrainingAdapter)
        self.assertEqual(spec.family, TrainingFamily.SPEECH_SEQ2SEQ)
        self.assertEqual(spec.module_paths, ("model", ))
        self.assertEqual(
            spec.component_paths,
            (
                "model.encoder",
                "model.encoder_decoder_proj",
                "model.transf_decoder",
                "model.log_softmax",
            ),
        )
        self.assertTrue(all(entrypoint.startswith("voicehub.") for entrypoint in spec.source_entrypoints))

    def test_public_config_rejects_delegation_and_non_safetensors(self):
        config = CohereASRConfig(name_or_path="CohereLabs/cohere-transcribe-03-2026")
        self.assertEqual(config.architecture_family, "speech-seq2seq")
        self.assertEqual(config.sample_rate, 16_000)
        self.assertFalse(config.trust_remote_code)
        with self.assertRaisesRegex(ValueError, "Safetensors only"):
            CohereASRConfig(
                name_or_path="model",
                use_safetensors=False,
            )
        with self.assertRaisesRegex(ValueError, "does not delegate"):
            CohereASRConfig(
                name_or_path="model",
                model_kwargs={"trust_remote_code": True},
            )
        with self.assertRaisesRegex(ValueError, "checkpoint root"):
            CohereASRConfig(
                name_or_path="model",
                processor_name_or_path="other",
            )

    def test_public_request_contract_fails_closed(self):
        validate = CohereForSpeechRecognition._validate_request
        resolved = validate(
            language="en",
            task="transcribe",
            return_timestamps=False,
            chunk_length_s=None,
            stride_length_s=None,
            batch_size=1,
            num_beams=1,
            max_new_tokens=None,
            hotwords=None,
            punctuation=True,
        )
        self.assertEqual(resolved, ("en", 256))
        common = {
            "language": "en",
            "task": "transcribe",
            "return_timestamps": False,
            "chunk_length_s": None,
            "stride_length_s": None,
            "batch_size": 1,
            "num_beams": 1,
            "max_new_tokens": 16,
            "hotwords": None,
            "punctuation": True,
        }
        for field, value, message in (
            ("language", None, "explicit language"),
            ("language", "tr", "Unsupported"),
            ("task", "translate", "translation"),
            ("return_timestamps", True, "timestamp"),
            ("chunk_length_s", 10.0, "manual chunk"),
            ("num_beams", 2, "greedy"),
            ("hotwords", "VoiceHub", "hotword"),
        ):
            values = {
                **common,
                field: value,
            }
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    validate(**values)

    def test_raw_training_preparation_is_batched_and_language_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _runtime(root / "runtime")
            wrapper = CohereForSpeechRecognition(
                CohereASRConfig(name_or_path=str(root / "runtime")),
                device="cpu",
                lazy_load=True,
            )
            wrapper.runtime = runtime
            wrapper.artifacts = runtime.artifacts
            wrapper.native_config = runtime.config
            wrapper.cohere_processor = runtime.processor
            wrapper.training_processor = runtime.processor
            wrapper.transformers_processor = runtime.processor
            wrapper.model = runtime.model

            prepared = wrapper.prepare_training_inputs(
                {
                    "audio": torch.linspace(-0.2, 0.2, 64),
                    "sampling_rate": 16_000,
                    "language": "en",
                    "punctuation": True,
                    "text": "hi",
                },
                phase="speech_recognition",
            )
            self.assertEqual(prepared["input_features"].ndim, 2)
            self.assertEqual(prepared["attention_mask"].ndim, 1)
            self.assertEqual(prepared["decoder_input_ids"].ndim, 1)
            self.assertEqual(prepared["labels"].ndim, 1)

            path = save_pcm_wave(
                root / "padded.wav",
                torch.cat((torch.zeros(32), torch.ones(32))),
                8_000,
            )
            file_prepared = wrapper.prepare_training_inputs(
                {
                    "audio": str(path),
                    "audio_lengths": 32,
                    "sampling_rate": 8_000,
                    "language": "en",
                    "punctuation": True,
                    "text": "hi",
                },
                phase="speech_recognition",
            )
            tensor_prepared = wrapper.prepare_training_inputs(
                {
                    "audio": torch.zeros(32),
                    "sampling_rate": 8_000,
                    "language": "en",
                    "punctuation": True,
                    "text": "hi",
                },
                phase="speech_recognition",
            )
            torch.testing.assert_close(
                file_prepared["input_features"],
                tensor_prepared["input_features"],
            )

            with self.assertRaisesRegex(ValueError, "shared language"):
                wrapper.prepare_training_inputs(
                    {
                        "audio": (
                            torch.zeros(64),
                            torch.zeros(64),
                        ),
                        "sampling_rate": (16_000, 16_000),
                        "language": ("en", "fr"),
                        "text": ("hi", "hi"),
                    },
                    phase="speech_recognition",
                )
            with self.assertRaisesRegex(TypeError, "booleans"):
                wrapper.prepare_training_inputs(
                    {
                        "audio": torch.zeros(64),
                        "sampling_rate": 16_000,
                        "language": "en",
                        "punctuation": "yes",
                        "text": "hi",
                    },
                    phase="speech_recognition",
                )

    def test_training_adapter_targets_exact_graph_and_restores_batch_rank(self):

        class Wrapper:

            architecture_family = "speech-seq2seq"

            def __init__(self):
                self.model = torch.nn.Linear(3, 3)
                self.config = SimpleNamespace(name_or_path=None)
                self.native_config = SimpleNamespace(mask_prompt_loss=False)
                self.exported = None

            @staticmethod
            def prepare_training_inputs(inputs, *, phase):
                del inputs, phase
                return {
                    "input_features": torch.zeros(8, 4),
                    "attention_mask": torch.ones(
                        8,
                        dtype=torch.bool,
                    ),
                    "decoder_input_ids": torch.ones(
                        5,
                        dtype=torch.long,
                    ),
                    "decoder_attention_mask": torch.ones(
                        5,
                        dtype=torch.bool,
                    ),
                    "labels": torch.ones(5, dtype=torch.long),
                    "ignored": "value",
                }

            def export_native_pretrained(self, destination):
                self.exported = Path(destination)

        spec = ModelTrainingSpec(
            model_type="asr_cohere",
            family=TrainingFamily.SPEECH_SEQ2SEQ,
            module_paths=("model", ),
            component_paths=("model", ),
            label_names=("labels", ),
            native_training=True,
            support=TrainingSupport.NATIVE,
            task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        )
        wrapper = Wrapper()
        adapter = NativeCohereASRTrainingAdapter(
            wrapper,
            spec,
        ).setup()
        context = adapter.create_training_context({})
        prepared = adapter.prepare_training_inputs({}, context)

        self.assertIs(adapter.primary_model, wrapper.model)
        self.assertEqual(prepared["input_features"].shape, (1, 8, 4))
        self.assertEqual(prepared["attention_mask"].shape, (1, 8))
        self.assertEqual(prepared["decoder_input_ids"].shape, (1, 5))
        self.assertNotIn("ignored", prepared)
        self.assertEqual(
            adapter.recipe_resume_configuration()["objective"],
            "prompt-conditioned-teacher-forced-cross-entropy",
        )
        self.assertEqual(
            adapter.artifact_manifest()["checkpoint_format"],
            "native-cohere-asr-v1",
        )
        adapter.save_pretrained("cohere-export")
        self.assertEqual(wrapper.exported, Path("cohere-export"))


if __name__ == "__main__":
    unittest.main()
