from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from voicehub.checkpointing import save_safetensors
from voicehub.checkpointing.errors import CheckpointIntegrityError
from voicehub.hub import write_json_file
from voicehub.models.fishtts.configuration import FishTTSConfig
from voicehub.models.fishtts.modeling import FishTTSForTextToSpeech
from voicehub.models.fishtts.native.artifacts import resolve_fish_codec_artifacts, resolve_fish_semantic_artifacts
from voicehub.models.fishtts.native.checkpoint import (
    convert_legacy_fish_codec,
    inspect_fish_checkpoint,
    load_fish_codec_checkpoint,
    load_fish_semantic_checkpoint,
    save_fish_codec_pretrained,
    save_fish_semantic_pretrained,
)
from voicehub.models.fishtts.native.codec import FishModifiedDAC
from voicehub.models.fishtts.native.configuration import FishCodecConfig, FishS2Config
from voicehub.models.fishtts.native.metadata import (
    FISH_ATTRIBUTION,
    FISH_LICENSE_NOTICE,
    FISH_S2_CHECKPOINT_REVISION,
    FISH_S2_CODEC_PARAMETER_COUNT,
    FISH_S2_CODEC_TENSOR_COUNT,
    FISH_S2_HEADER_FINGERPRINT,
    FISH_S2_PARAMETER_COUNT,
    FISH_S2_TENSOR_COUNT,
    FISH_SPEECH_SOURCE_LICENSE_SHA256,
    FISH_SPEECH_SOURCE_REVISION,
)
from voicehub.models.fishtts.native.modeling import FishS2ForConditionalGeneration
from voicehub.models.fishtts.native.prompting import FishConversationTurn, build_fish_prompt, split_speaker_turns
from voicehub.models.fishtts.native.registration import create_fish_s2_architecture_spec
from voicehub.models.fishtts.native.tokenization import (
    AUDIO_END,
    AUDIO_PAD,
    AUDIO_START,
    END_OF_TEXT,
    IM_END,
    IM_START,
    MODALITY_INTERLEAVE,
    MODALITY_TEXT,
    MODALITY_VOICE,
    PAD,
    PHONEME_END,
    PHONEME_START,
    SEMANTIC_TEMPLATE,
    FishTokenizer,
    normalize_fish_text,
)
from voicehub.models.fishtts.training import FishSpeechTrainingAdapter
from voicehub.tokenization import encode_gpt2_token

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN2_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
    r"\s+(?!\S)|\s+")


def _official_config_values() -> dict:
    common = {
        "attention_o_bias": False,
        "attention_qkv_bias": False,
        "audio_hidden_dim": 5_120,
        "dim": 2_560,
        "dropout": 0.0,
        "head_dim": 128,
        "initializer_range": 0.01976423537605237,
        "intermediate_size": 9_728,
        "n_head": 32,
        "n_local_heads": 8,
        "norm_eps": 1e-6,
        "rope_base": 1_000_000,
        "tie_word_embeddings": False,
        "use_gradient_checkpointing": True,
        "use_moe": False,
        "vocab_size": 4_096,
    }
    return {
        "audio_decoder_config": {
            **common,
            "attention_qk_norm": False,
            "max_seq_len": 11,
            "model_type": "fish_qwen3_audio_decoder",
            "n_layer": 4,
            "num_codebooks": 10,
            "text_dim": 2_560,
        },
        "audio_pad_token_id": 151_677,
        "dtype": "bfloat16",
        "eos_token_id": 151_645,
        "model_type": "fish_qwen3_omni",
        "pad_token_id": 151_669,
        "semantic_end_token_id": 155_773,
        "semantic_start_token_id": 151_678,
        "text_config": {
            **common,
            "attention_qk_norm": True,
            "max_seq_len": 32_768,
            "model_type": "fish_qwen3",
            "n_layer": 36,
            "tie_word_embeddings": True,
            "vocab_size": 155_776,
        },
    }


def _tokenizer_test_config() -> FishS2Config:
    return replace(
        FishS2Config.tiny(vocab_size=280),
        end_of_text_id=256,
        im_start_id=257,
        im_end_id=258,
        pad_token_id=259,
        audio_pad_token_id=267,
    )


def _write_test_tokenizer(path: Path, config: FishS2Config) -> None:
    vocabulary = {encode_gpt2_token(bytes((value, ))): value for value in range(256)}
    protocol = {
        END_OF_TEXT: config.end_of_text_id,
        IM_START: config.im_start_id,
        IM_END: config.im_end_id,
        PAD: config.pad_token_id,
        PHONEME_START: 260,
        PHONEME_END: 261,
        MODALITY_TEXT: 262,
        MODALITY_VOICE: 263,
        MODALITY_INTERLEAVE: 264,
        AUDIO_START: 265,
        AUDIO_END: 266,
        AUDIO_PAD: config.audio_pad_token_id,
    }
    protocol.update({
        SEMANTIC_TEMPLATE.format(code=code): config.semantic_begin_id + code
        for code in range(config.codebook_size)
    })
    added_tokens = [{
        "content": spelling,
        "id": token_id,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    } for spelling, token_id in protocol.items()]
    write_json_file(
        path, {
            "added_tokens": added_tokens,
            "model": {
                "byte_fallback": False,
                "continuing_subword_prefix": None,
                "dropout": None,
                "end_of_word_suffix": None,
                "merges": [],
                "type": "BPE",
                "unk_token": None,
                "vocab": vocabulary,
            },
            "normalizer": None,
            "pre_tokenizer": {
                "pretokenizers": [
                    {
                        "behavior": "Isolated",
                        "invert": False,
                        "pattern": {
                            "Regex": QWEN2_PATTERN,
                        },
                        "type": "Split",
                    },
                    {
                        "add_prefix_space": False,
                        "trim_offsets": False,
                        "type": "ByteLevel",
                        "use_regex": False,
                    },
                ],
                "type":
                "Sequence",
            },
            "version": "1.0",
        })


def _tiny_codec_config() -> FishCodecConfig:
    return FishCodecConfig(
        sample_rate=44_100,
        encoder_dim=4,
        encoder_rates=(2, ),
        decoder_dim=8,
        decoder_rates=(2, ),
        encoder_transformer_layers=(0, ),
        decoder_transformer_layers=(0, ),
        semantic_codebook_size=8,
        residual_codebook_size=4,
        residual_codebooks=2,
        codebook_dim=2,
        quantizer_dropout=0.0,
        downsample_factors=(2, ),
        transformer_layers=1,
        transformer_heads=2,
        transformer_hidden_size=8,
        transformer_intermediate_size=16,
        transformer_window_size=8,
    )


def _write_tiny_runtime(root: Path):
    semantic_config = _tokenizer_test_config()
    semantic_model = FishS2ForConditionalGeneration(semantic_config)
    save_fish_semantic_pretrained(semantic_model, root)
    _write_test_tokenizer(root / "tokenizer.json", semantic_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        codec = FishModifiedDAC(_tiny_codec_config())
    save_fish_codec_pretrained(codec, root / "codec")
    return semantic_model, codec


class NativeFishDependencyTests(unittest.TestCase):

    def test_active_runtime_has_no_provider_or_model_framework_imports(self):
        roots = (
            PROJECT_ROOT / 'voicehub/models/fishtts/native',
            PROJECT_ROOT / "voicehub" / "models" / "fishtts",
        )
        forbidden = {
            "einops",
            "fish_speech",
            "huggingface_hub",
            "hydra",
            "numpy",
            "omegaconf",
            "safetensors",
            "tokenizers",
            "torchaudio",
            "transformers",
            "vector_quantize_pytorch",
        }
        violations = []
        for root in roots:
            for path in root.glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module):
                        names = [node.module]
                    else:
                        names = []
                    for name in names:
                        if name.split(".", 1)[0] in forbidden:
                            violations.append((path.name, name))
        self.assertEqual(violations, [])

    def test_wrapper_import_keeps_external_model_frameworks_unloaded(self):
        command = (
            "import sys; "
            "import voicehub.models.fishtts.modeling; "
            "names=('transformers','huggingface_hub','safetensors',"
            "'tokenizers','numpy','torchaudio','einops','hydra',"
            "'omegaconf','vector_quantize_pytorch','fish_speech'); "
            "print(' '.join(str(name in sys.modules) for name in names))")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            " ".join(("False", ) * 11),
        )

    def test_provenance_and_license_obligations_are_pinned(self):
        directory = (PROJECT_ROOT / 'voicehub/models/fishtts/native')
        source = json.loads((directory / "SOURCE.json").read_text(encoding="utf-8"))
        self.assertEqual(
            source["implementation_sources"][0]["revision"],
            FISH_SPEECH_SOURCE_REVISION,
        )
        self.assertEqual(
            source["checkpoints"]["s2-pro"]["revision"],
            FISH_S2_CHECKPOINT_REVISION,
        )
        self.assertFalse(source["commercial_use"])
        self.assertTrue(source["license_obligations"]["commercial_license_required"])
        self.assertEqual(
            source["license_obligations"]["notice"],
            FISH_LICENSE_NOTICE,
        )
        self.assertEqual(
            source["license_obligations"]["attribution"],
            FISH_ATTRIBUTION,
        )
        license_bytes = (directory / "THIRD_PARTY_LICENSE").read_bytes()
        canonical_license_bytes = license_bytes.replace(b"\r\n", b"\n")
        digest = hashlib.sha256(canonical_license_bytes).hexdigest()
        self.assertEqual(digest, FISH_SPEECH_SOURCE_LICENSE_SHA256)


class NativeFishTopologyTests(unittest.TestCase):

    def test_published_config_and_semantic_header_inventory_are_exact(self):
        config = FishS2Config.from_dict(_official_config_values())
        self.assertEqual(config.end_of_text_id, 151_643)
        self.assertEqual(config.im_end_id, 151_645)
        self.assertEqual(config.to_dict()["eos_token_id"], 151_645)
        self.assertEqual(FishS2Config.from_dict(config.to_dict()), config)
        with torch.device("meta"):
            model = FishS2ForConditionalGeneration(
                config,
                initialize=False,
            )
        state = model.state_dict()
        self.assertEqual(len(state), FISH_S2_TENSOR_COUNT)
        self.assertEqual(
            sum(value.numel() for value in state.values()),
            FISH_S2_PARAMETER_COUNT,
        )

        def published_name(name: str) -> str:
            if name.startswith("codebook_embeddings."):
                return "audio_decoder." + name
            if name.startswith("fast_"):
                return "audio_decoder." + name[len("fast_"):]
            return "text_model.model." + name

        rows = [
            f"{published_name(name)}|BF16|" + "x".join(str(item) for item in value.shape)
            for name, value in state.items()
        ]
        fingerprint = hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()
        self.assertEqual(fingerprint, FISH_S2_HEADER_FINGERPRINT)

    def test_published_modified_dac_topology_is_exact(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with torch.device("meta"):
                codec = FishModifiedDAC(
                    FishCodecConfig(),
                    initialize=False,
                )
        state = codec.state_dict()
        self.assertEqual(len(state), FISH_S2_CODEC_TENSOR_COUNT)
        self.assertEqual(
            sum(value.numel() for value in state.values()),
            FISH_S2_CODEC_PARAMETER_COUNT,
        )
        self.assertEqual(
            tuple(state["encoder.block.0.conv.parametrizations.weight.original1"].shape),
            (64, 1, 7),
        )
        self.assertEqual(
            tuple(state["quantizer.semantic_quantizer.quantizers.0."
                        "codebook.weight"].shape),
            (4_096, 8),
        )

    def test_dense_s2_rejects_unreviewed_moe_variants(self):
        values = _official_config_values()
        values["text_config"]["use_moe"] = True
        with self.assertRaisesRegex(ValueError, "MoE"):
            FishS2Config.from_dict(values)
        values = _official_config_values()
        values["audio_decoder_config"]["text_dim"] = 1
        with self.assertRaisesRegex(ValueError, "text_dim"):
            FishS2Config.from_dict(values)


class NativeFishTokenizerAndModelTests(unittest.TestCase):

    def test_native_tokenizer_validates_protocol_and_blocks_injection(self):
        config = _tokenizer_test_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            _write_test_tokenizer(path, config)
            tokenizer = FishTokenizer.from_tokenizer_json(
                path,
                config=config,
            )
            self.assertEqual(len(tokenizer), config.text.vocab_size)
            self.assertEqual(
                tokenizer.get_token_id("<|end_of_text|>"),
                config.end_of_text_id,
            )
            self.assertEqual(
                tokenizer.semantic_code_to_token_id(7),
                config.semantic_end_id,
            )
            self.assertEqual(
                tokenizer.token_id_to_semantic_code(config.semantic_begin_id + 3),
                3,
            )
            self.assertEqual(normalize_fish_text("e\u0301"), "é")
            self.assertEqual(
                normalize_fish_text("<|speaker:2|> hello"),
                "<|speaker:2|> hello",
            )
            self.assertEqual(
                split_speaker_turns("intro <|speaker:1|> one <|speaker:2|> two"),
                (
                    "intro",
                    "<|speaker:1|> one",
                    "<|speaker:2|> two",
                ),
            )
            with self.assertRaisesRegex(ValueError, "reserved"):
                normalize_fish_text("hello <|im_end|>")

            reference = torch.tensor([
                [1, 2],
                [2, 3],
                [3, 1],
            ])
            prompt = build_fish_prompt(
                "hello",
                tokenizer,
                reference_text="sample",
                reference_codes=reference,
            )
            self.assertEqual(prompt.shape[0], config.num_codebooks + 1)
            first_semantic = (prompt[0] == config.semantic_begin_id + reference[0, 0]).nonzero().flatten()
            self.assertTrue(first_semantic.numel())
            index = int(first_semantic[0])
            torch.testing.assert_close(
                prompt[1:, index:index + 2],
                reference,
            )
            with self.assertRaisesRegex(TypeError, "integer dtype"):
                build_fish_prompt(
                    "hello",
                    tokenizer,
                    reference_text="sample",
                    reference_codes=reference.float(),
                )
            with self.assertRaisesRegex(ValueError, "reserved"):
                build_fish_prompt(
                    "hello",
                    tokenizer,
                    history=(FishConversationTurn(
                        role="user",
                        text="unsafe <|im_end|>",
                    ), ),
                )

    def test_semantic_objective_reaches_slow_and_fast_parameters(self):
        torch.manual_seed(7)
        config = FishS2Config.tiny()
        model = FishS2ForConditionalGeneration(config)
        inputs = torch.zeros(
            1,
            config.num_codebooks + 1,
            7,
            dtype=torch.long,
        )
        inputs[0, 0] = torch.tensor([4, 56, 57, 58, 9, 10, 11])
        inputs[0, 1:] = torch.randint(
            0,
            config.codebook_size,
            (config.num_codebooks, 7),
        )
        labels = torch.full_like(inputs, -100)
        labels[0, 0] = torch.tensor([56, 57, 58, 59, 3, -100, -100])
        labels[0, 1:, :4] = torch.randint(
            0,
            config.codebook_size,
            (config.num_codebooks, 4),
        )
        output = model(inp=inputs, labels=labels)
        losses, codebook_logits, codebook_targets = (
            FishSpeechTrainingAdapter.compute_source_losses(
                token_logits=output.token_logits,
                codebook_logits=output.codebook_logits,
                labels=labels,
                semantic_begin_id=config.semantic_begin_id,
                semantic_end_id=config.semantic_end_id,
                num_codebooks=config.num_codebooks,
            ))
        loss = losses["base_loss"] + losses["semantic_loss"]
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            tuple(codebook_logits.shape[:-1]),
            tuple(codebook_targets.shape),
        )
        loss.backward()
        parameters = dict(model.named_parameters())
        for name in (
                "embeddings.weight",
                "codebook_embeddings.weight",
                "layers.0.attention.wqkv.weight",
                "fast_embeddings.weight",
                "fast_layers.0.attention.wqkv.weight",
                "fast_output.weight",
        ):
            with self.subTest(name=name):
                gradient = parameters[name].grad
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_slow_cache_matches_full_causal_forward(self):
        torch.manual_seed(11)
        config = FishS2Config.tiny()
        model = FishS2ForConditionalGeneration(config).eval()
        prompt = torch.zeros(
            1,
            config.num_codebooks + 1,
            5,
            dtype=torch.long,
        )
        prompt[0, 0] = torch.tensor([4, 5, 6, 7, 8])
        with torch.no_grad():
            expected = model.forward_slow(prompt)
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=config.text.max_position_embeddings,
            )
            actual = model.forward_generate(
                prompt,
                torch.arange(prompt.shape[-1]),
                return_all=True,
            )
        torch.testing.assert_close(actual.logits, expected.logits)
        torch.testing.assert_close(
            actual.hidden_states,
            expected.hidden_states,
        )
        model.clear_caches()
        self.assertEqual(model.max_batch_size, -1)


class NativeFishCheckpointAndLifecycleTests(unittest.TestCase):

    def test_semantic_and_codec_safe_exports_strictly_reload(self):
        semantic = FishS2ForConditionalGeneration(_tokenizer_test_config())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            codec = FishModifiedDAC(_tiny_codec_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_fish_semantic_pretrained(semantic, root / "semantic")
            save_fish_codec_pretrained(codec, root / "codec")
            with torch.device("meta"):
                restored_semantic = FishS2ForConditionalGeneration(
                    semantic.config,
                    initialize=False,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    restored_codec = FishModifiedDAC(
                        codec.config,
                        initialize=False,
                    )
            load_fish_semantic_checkpoint(
                restored_semantic,
                root / "semantic" / "model.safetensors",
                device="cpu",
            )
            load_fish_codec_checkpoint(
                restored_codec,
                root / "codec" / "model.safetensors",
                device="cpu",
            )
            for source, restored in (
                (semantic, restored_semantic),
                (codec, restored_codec),
            ):
                for name, expected in source.state_dict().items():
                    torch.testing.assert_close(
                        restored.state_dict()[name],
                        expected,
                        rtol=0,
                        atol=0,
                    )
            notice = (root / "semantic" / "NOTICE").read_text(encoding="utf-8")
            self.assertIn(FISH_LICENSE_NOTICE, notice)
            self.assertIn(FISH_ATTRIBUTION, notice)
            self.assertTrue((root / "semantic" / "THIRD_PARTY_LICENSE").is_file())

    def test_sharded_inventory_inspection_reads_headers_without_record_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_safetensors(
                {"alpha": torch.zeros(2, 3)},
                root / "one.safetensors",
            )
            save_safetensors(
                {"beta": torch.zeros(4, dtype=torch.int64)},
                root / "two.safetensors",
            )
            index = root / "model.safetensors.index.json"
            write_json_file(
                index, {
                    "metadata": {
                        "total_size": 56,
                    },
                    "weight_map": {
                        "alpha": "one.safetensors",
                        "beta": "two.safetensors",
                    },
                })
            report = inspect_fish_checkpoint(index)
            self.assertEqual(report.tensor_count, 2)
            self.assertEqual(report.parameter_count, 10)
            expected = hashlib.sha256(b"alpha|F32|2x3\nbeta|I64|4").hexdigest()
            self.assertEqual(report.header_fingerprint, expected)

    def test_legacy_codec_conversion_is_explicit_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "codec.pth"
            legacy.write_bytes(b"not the official checkpoint")
            with patch("voicehub.models.fishtts.native.checkpoint.torch.load", ) as load:
                with self.assertRaisesRegex(
                        PermissionError,
                        "trust_legacy_pickle",
                ):
                    convert_legacy_fish_codec(legacy, root / "denied")
                load.assert_not_called()
            with patch("voicehub.models.fishtts.native.checkpoint.torch.load", ) as load:
                with self.assertRaises(CheckpointIntegrityError):
                    convert_legacy_fish_codec(
                        legacy,
                        root / "invalid",
                        trust_legacy_pickle=True,
                    )
                load.assert_not_called()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                codec = FishModifiedDAC(_tiny_codec_config())
            torch.save(
                {
                    "state_dict": {
                        "generator." + name: value
                        for name, value in codec.state_dict().items()
                    },
                },
                legacy,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                converted = convert_legacy_fish_codec(
                    legacy,
                    root / "converted",
                    trust_legacy_pickle=True,
                    verify_official_integrity=False,
                    config=codec.config,
                )
            self.assertEqual(converted.suffix, ".safetensors")
            self.assertTrue((converted.parent / "NOTICE").is_file())
            with torch.device("meta"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    restored = FishModifiedDAC(
                        codec.config,
                        initialize=False,
                    )
            load_fish_codec_checkpoint(
                restored,
                converted,
                device="cpu",
            )
            for name, expected in codec.state_dict().items():
                torch.testing.assert_close(
                    restored.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )

    def test_artifact_resolvers_reject_unsafe_or_pickle_steady_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.gguf", "codec.pth"):
                (root / name).write_bytes(b"unsafe")
            with self.assertRaisesRegex(ValueError, "Safetensors only"):
                resolve_fish_semantic_artifacts(root / "model.gguf")
            with self.assertRaisesRegex(ValueError, "Safetensors only"):
                resolve_fish_codec_artifacts(root / "codec.pth")

    def test_training_inference_training_preserves_semantic_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            source_model, _ = _write_tiny_runtime(root)
            wrapper = FishTTSForTextToSpeech(
                FishTTSConfig(
                    name_or_path=root,
                    torch_dtype="float32",
                ),
                device="cpu",
            )
            wrapper.load_for_training()
            semantic = wrapper.model
            parameter = next(semantic.parameters())
            self.assertTrue(semantic.training)
            self.assertIsNone(wrapper._runtime)
            self.assertIsNone(wrapper._codec)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                wrapper.load()
            self.assertIs(wrapper.model, semantic)
            self.assertIs(next(wrapper.model.parameters()), parameter)
            self.assertIsNotNone(wrapper._runtime)
            self.assertFalse(wrapper.model.training)
            self.assertTrue(all(not item.requires_grad for item in wrapper._codec.parameters()))

            wrapper.load_for_training()
            self.assertIs(wrapper.model, semantic)
            self.assertIs(next(wrapper.model.parameters()), parameter)
            self.assertTrue(wrapper.model.training)

            exported = Path(directory) / "exported"
            wrapper.export_native_pretrained(exported)
            fresh = FishTTSForTextToSpeech(
                FishTTSConfig(
                    name_or_path=exported,
                    torch_dtype="float32",
                ),
                device="cpu",
            )
            fresh.load_for_training()
            self.assertIsNone(fresh._runtime)
            self.assertIsNone(fresh._codec)
            for name, expected in source_model.state_dict().items():
                torch.testing.assert_close(
                    fresh.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )

    def test_security_config_and_architecture_registration(self):
        with self.assertRaisesRegex(ValueError, "trust_remote_code"):
            FishTTSConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "Safetensors"):
            FishTTSConfig(use_safetensors=False)
        with self.assertRaisesRegex(ValueError, "does not delegate"):
            FishTTSConfig(model_kwargs={"attn_implementation": "flash"})
        with self.assertRaisesRegex(ValueError, "full fine-tuning"):
            FishTTSConfig(training_lora_config={"rank": 8})
        spec = create_fish_s2_architecture_spec()
        self.assertEqual(spec.architecture_id, "fish-s2")
        self.assertEqual(spec.capabilities.checkpoint_formats, ("safetensors", ))
        self.assertTrue(spec.capabilities.training)
        self.assertFalse(spec.metadata["commercial_use"])


if __name__ == "__main__":
    unittest.main()
