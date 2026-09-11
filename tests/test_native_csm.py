import ast
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional

from voicehub.hub import write_json_file
from voicehub.models.csm.modeling import CSMForTextToSpeech
from voicehub.models.csm.native.checkpoint import (
    export_csm_checkpoint,
    load_csm_checkpoint,
    tensor_inventory_fingerprint,
)
from voicehub.models.csm.native.configuration import CSMArchitectureConfig, CSMTransformerConfig
from voicehub.models.csm.native.metadata import (
    CSM_CHECKPOINT_HEADER_FINGERPRINT,
    CSM_CHECKPOINT_PARAMETER_COUNT,
    CSM_CHECKPOINT_TENSOR_COUNT,
    CSM_SOURCE_REVISION,
    MIMI_CHECKPOINT_HEADER_FINGERPRINT,
    MIMI_CHECKPOINT_PARAMETER_COUNT,
    MIMI_CHECKPOINT_REVISION,
    MIMI_CHECKPOINT_TENSOR_COUNT,
)
from voicehub.models.csm.native.mimi import build_mimi, mimi_checkpoint_inventory
from voicehub.models.csm.native.modeling import CSMAttention, CSMLlama3ScaledRoPE, CSMModel, sample_top_k
from voicehub.models.csm.native.processing import CSMProcessor, CSMTextTokenizer
from voicehub.models.csm.native.runtime import CSMRuntime, load_csm_runtime
from voicehub.models.csm.training import CSMTrainingBackend, CSMTrainingCollator, prepare_csm_training_inputs
from voicehub.tokenization import ByteBPETokenizer, encode_gpt2_token

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tokenizer():
    vocabulary = {bytes((value, )): value for value in range(256)}
    tokenizer = ByteBPETokenizer(
        vocabulary,
        special_tokens={
            "<|begin_of_text|>": 128_000,
            "<|end_of_text|>": 128_001,
        },
        prefix_token_ids=(128_000, ),
        suffix_token_ids=(128_001, ),
        pad_token_id=128_001,
    )
    return CSMTextTokenizer(tokenizer)


def _write_test_tokenizer(path: Path) -> None:
    vocabulary = {encode_gpt2_token(bytes((value, ))): value for value in range(256)}
    write_json_file(
        path, {
            "version":
            "1.0",
            "added_tokens": [
                {
                    "id": 128_000,
                    "content": "<|begin_of_text|>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
                {
                    "id": 128_001,
                    "content": "<|end_of_text|>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
            ],
            "normalizer":
            None,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
            "model": {
                "type": "BPE",
                "vocab": vocabulary,
                "merges": [],
                "unk_token": None,
            },
        })


def _portable_test_config() -> CSMArchitectureConfig:
    return CSMArchitectureConfig(
        text_vocabulary_size=128_002,
        audio_vocabulary_size=19,
        num_audio_codebooks=4,
        backbone=CSMTransformerConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_sequence_length=32,
        ),
        depth_decoder=CSMTransformerConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_sequence_length=4,
        ),
    )


class NativeCSMDependencyTests(unittest.TestCase):

    def test_public_runtime_has_no_external_model_framework_imports(self):
        roots = (
            PROJECT_ROOT / 'voicehub/models/csm/native',
            PROJECT_ROOT / "voicehub" / "models" / "csm",
        )
        forbidden = {
            "transformers",
            "huggingface_hub",
            "torchtune",
            "tokenizers",
            "safetensors",
            "torchaudio",
            "numpy",
            "einops",
            "moshi",
        }
        violations = []
        for root in roots:
            for path in root.glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        names = [node.module]
                    for name in names:
                        if name.split(".", 1)[0] in forbidden:
                            violations.append((path.name, name))
        mimi_source = (PROJECT_ROOT / "voicehub" / "models" / "csm" / "source" / "moshi")
        native_mimi_paths = [
            mimi_source / "models" / "__init__.py",
            mimi_source / "models" / "compression.py",
            mimi_source / "utils" / "__init__.py",
            mimi_source / "utils" / "compile.py",
            mimi_source / "utils" / "quantize.py",
        ]
        for directory in ("modules", "quantization"):
            native_mimi_paths.extend(sorted((mimi_source / directory).glob("*.py")), )
        for path in native_mimi_paths:
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

    def test_import_does_not_load_transformers_or_hub_client(self):
        command = (
            "import sys; "
            "import voicehub.models.csm.modeling; "
            "print('transformers' in sys.modules, "
            "'huggingface_hub' in sys.modules, "
            "'torchtune' in sys.modules, "
            "'tokenizers' in sys.modules, "
            "'safetensors' in sys.modules, "
            "'torchaudio' in sys.modules, "
            "'sentencepiece' in sys.modules, "
            "'einops' in sys.modules, "
            "'bitsandbytes' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            "False False False False False False False False False",
        )

    def test_provenance_pins_source_model_and_codec(self):
        source = json.loads(
            (PROJECT_ROOT / 'voicehub/models/csm/native/SOURCE.json').read_text(encoding="utf-8"))
        self.assertEqual(source["sources"][0]["revision"], CSM_SOURCE_REVISION)
        revisions = {artifact["revision"] for artifact in source["artifacts"]}
        self.assertIn(MIMI_CHECKPOINT_REVISION, revisions)
        self.assertIn("training_boundary", source["implementation"])


class NativeCSMInventoryTests(unittest.TestCase):

    def test_official_graph_matches_audited_safe_header(self):
        model = CSMModel(
            CSMArchitectureConfig(),
            device="meta",
            dtype=torch.float32,
        )
        inventory = {name: ("F32", tuple(value.shape)) for name, value in model.state_dict().items()}
        self.assertEqual(len(inventory), CSM_CHECKPOINT_TENSOR_COUNT)
        self.assertEqual(
            sum(value.numel() for value in model.parameters()),
            CSM_CHECKPOINT_PARAMETER_COUNT,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(inventory),
            CSM_CHECKPOINT_HEADER_FINGERPRINT,
        )
        self.assertEqual(
            tuple(model.audio_head.shape),
            (31, 1024, 2051),
        )

    def test_config_rejects_incompatible_attention_geometry(self):
        with self.assertRaisesRegex(ValueError, "divide"):
            CSMTransformerConfig(
                hidden_size=24,
                intermediate_size=48,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=3,
                max_sequence_length=16,
            )

    def test_mimi_graph_matches_audited_safe_header_without_allocation(self):
        with torch.device("meta"):
            codec = build_mimi(device="meta")
        inventory = mimi_checkpoint_inventory(codec)
        self.assertEqual(len(inventory), MIMI_CHECKPOINT_TENSOR_COUNT)
        self.assertEqual(
            sum(math.prod(shape) for _, shape in inventory.values()),
            MIMI_CHECKPOINT_PARAMETER_COUNT,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(inventory),
            MIMI_CHECKPOINT_HEADER_FINGERPRINT,
        )
        self.assertIn(
            "encoder_transformer.transformer.layers.0.self_attn."
            "in_proj_weight",
            inventory,
        )
        self.assertNotIn(
            "encoder_transformer.transformer.layers.0.self_attn."
            "in_projs.0.weight",
            inventory,
        )


class NativeCSMParityTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(13)

    def test_llama3_scaled_rope_matches_direct_source_equation(self):
        config = CSMTransformerConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_sequence_length=16,
        )
        rope = CSMLlama3ScaledRoPE(config)
        inputs = torch.randn(2, 5, 4, 8)
        positions = torch.tensor([
            [0, 1, 2, 3, 4],
            [2, 3, 4, 5, 6],
        ])
        actual = rope(inputs, input_positions=positions)

        cache = rope.cache[positions].unsqueeze(2)
        paired = inputs.float().reshape(2, 5, 4, 4, 2)
        expected = torch.stack(
            (
                paired[..., 0] * cache[..., 0] - paired[..., 1] * cache[..., 1],
                paired[..., 1] * cache[..., 0] + paired[..., 0] * cache[..., 1],
            ),
            dim=-1,
        ).flatten(3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_grouped_query_attention_matches_direct_equation(self):
        config = CSMTransformerConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_sequence_length=8,
        )
        rope = CSMLlama3ScaledRoPE(config)
        attention = CSMAttention(config, rope).eval()
        inputs = torch.randn(2, 5, 16)
        actual = attention(inputs)

        batch, time, _ = inputs.shape
        queries = attention.q_proj(inputs).reshape(batch, time, 4, 4)
        keys = attention.k_proj(inputs).reshape(batch, time, 2, 4)
        values = attention.v_proj(inputs).reshape(batch, time, 2, 4)
        queries = rope(queries).transpose(1, 2)
        keys = (
            rope(keys)[:, :, :, None, :].expand(
                batch,
                time,
                2,
                2,
                4,
            ).reshape(batch, time, 4, 4).transpose(1, 2))
        values = (
            values[:, :, :, None, :].expand(
                batch,
                time,
                2,
                2,
                4,
            ).reshape(batch, time, 4, 4).transpose(1, 2))
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(4)
        causal = torch.tril(torch.ones(time, time, dtype=torch.bool))
        scores = scores.masked_fill(~causal, -torch.inf)
        expected = torch.matmul(scores.softmax(dim=-1), values)
        expected = attention.output_proj(expected.transpose(1, 2).reshape(batch, time, 16))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_masked_multimodal_embedding_matches_source_sum(self):
        config = CSMArchitectureConfig.tiny()
        model = CSMModel(config)
        tokens = torch.tensor([[
            [1, 2, 3, 4, 7],
            [5, 6, 7, 8, 9],
        ]])
        mask = torch.tensor([[
            [True, True, False, False, False],
            [False, False, False, False, True],
        ]])
        actual = model.embed_tokens(tokens, mask)
        expected_audio = (
            model.audio_embeddings(torch.tensor([1])) +
            model.audio_embeddings(torch.tensor([2 + config.audio_vocabulary_size])))
        expected_text = model.text_embeddings(torch.tensor([9]))
        torch.testing.assert_close(actual[0, 0], expected_audio[0])
        torch.testing.assert_close(actual[0, 1], expected_text[0])

    def test_greedy_generation_resets_cache_and_is_repeatable(self):
        config = CSMArchitectureConfig.tiny()
        model = CSMModel(config).eval()
        tokens = torch.zeros(
            1,
            4,
            config.num_audio_codebooks + 1,
            dtype=torch.long,
        )
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens[..., -1] = torch.tensor([[2, 3, 4, 5]])
        mask[..., -1] = True
        first = model.generate_audio_codes(
            tokens,
            mask,
            max_new_frames=3,
            temperature=0,
            top_k=0,
        )
        second = model.generate_audio_codes(
            tokens,
            mask,
            max_new_frames=3,
            temperature=0,
            top_k=0,
        )
        torch.testing.assert_close(first, second)

    def test_sampling_bounds_are_explicit(self):
        logits = torch.randn(2, 7)
        with self.assertRaisesRegex(ValueError, r"\[1, 7\]"):
            sample_top_k(logits, top_k=8, temperature=1)
        expected = logits.argmax(dim=-1, keepdim=True).int()
        torch.testing.assert_close(
            sample_top_k(logits, top_k=0, temperature=0),
            expected,
        )


class NativeCSMTrainingTests(unittest.TestCase):

    def test_native_objective_backpropagates_through_both_decoders(self):
        torch.manual_seed(7)
        config = CSMArchitectureConfig.tiny()
        model = CSMModel(config)
        batch, time = 2, 7
        tokens = torch.zeros(
            batch,
            time,
            config.num_audio_codebooks + 1,
            dtype=torch.long,
        )
        token_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens[:, :2, -1] = torch.randint(
            0,
            config.text_vocabulary_size,
            (batch, 2),
        )
        token_mask[:, :2, -1] = True
        codes = torch.randint(
            0,
            config.audio_vocabulary_size,
            (batch, time - 2, config.num_audio_codebooks),
        )
        tokens[:, 2:, :-1] = codes
        token_mask[:, 2:, :-1] = True
        labels = torch.full(
            (batch, time, config.num_audio_codebooks),
            -100,
            dtype=torch.long,
        )
        labels[:, 2:] = codes

        output = model(tokens, token_mask, labels=labels)
        self.assertEqual(output.loss.ndim, 0)
        torch.testing.assert_close(
            output.loss,
            output.backbone_loss + output.depth_decoder_loss,
        )
        output.loss.backward()
        self.assertIsNotNone(model.backbone.layers[0].attn.q_proj.weight.grad)
        self.assertIsNotNone(model.decoder.layers[0].attn.q_proj.weight.grad)
        self.assertIsNotNone(model.codebook0_head.weight.grad)
        self.assertIsNotNone(model.audio_head.grad)

    def test_processor_builds_labels_from_preencoded_mimi_codes(self):
        config = CSMArchitectureConfig(
            text_vocabulary_size=128_256,
            audio_vocabulary_size=19,
            num_audio_codebooks=4,
            backbone=CSMArchitectureConfig.tiny().backbone,
            depth_decoder=CSMTransformerConfig(
                hidden_size=24,
                intermediate_size=48,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_sequence_length=4,
            ),
        )
        processor = CSMProcessor(_tokenizer(), config)
        codes = torch.tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ])
        batch = processor.training_batch([{
            "speaker_id": 2,
            "text": "hi",
            "audio_codes": codes,
        }])
        self.assertEqual(batch["tokens"].shape[0], 1)
        audio_positions = (batch["labels"][0, :, 0] != -100).nonzero()
        self.assertEqual(audio_positions.shape[0], 4)
        self.assertTrue(torch.equal(
            batch["labels"][0, audio_positions[0], :].squeeze(0),
            codes[:, 0],
        ))
        self.assertTrue(
            torch.equal(
                batch["labels"][0, audio_positions[-1], :].squeeze(0),
                torch.zeros(4, dtype=torch.long),
            ))

    def test_columnar_code_tensor_splits_into_individual_records(self):
        config = _portable_test_config()
        processor = CSMProcessor(_tokenizer(), config)
        codes = torch.randint(
            0,
            config.audio_vocabulary_size,
            (2, config.num_audio_codebooks, 3),
        )
        batch = prepare_csm_training_inputs(
            processor,
            {
                "text": ["first", "second"],
                "speaker_id": [0, 1],
                "audio_codes": codes,
            },
        )
        self.assertEqual(batch["tokens"].shape[0], 2)

    def test_native_collator_places_batch_on_runtime_device(self):

        class Processor:
            sample_rate = 24_000

            def training_batch(self, records, **kwargs):
                return {
                    "records": records,
                    **kwargs,
                }

        class Runtime:
            device = torch.device("meta")

            @staticmethod
            def encode_training_records(records):
                return [dict(record) for record in records]

        collator = CSMTrainingCollator(
            Processor(),
            runtime=Runtime(),
        )
        output = collator([{
            "text": "prepared",
            "audio_codes": torch.ones(4, 2, dtype=torch.long),
        }])
        self.assertEqual(output["device"], torch.device("meta"))

    def test_backend_exposes_exact_differentiable_scalar_loss(self):
        config = CSMArchitectureConfig.tiny()
        model = CSMModel(config)
        backend = CSMTrainingBackend(
            model=model,
            processor=object(),
            sample_rate=24_000,
        )
        tokens = torch.zeros(
            1,
            3,
            config.num_audio_codebooks + 1,
            dtype=torch.long,
        )
        mask = torch.ones_like(tokens, dtype=torch.bool)
        tokens[..., -1] %= config.text_vocabulary_size
        labels = torch.randint(
            0,
            config.audio_vocabulary_size,
            (1, 3, config.num_audio_codebooks),
        )
        loss = backend.forward_loss(
            tokens=tokens,
            tokens_mask=mask,
            labels=labels,
        )
        self.assertEqual(loss.ndim, 0)
        loss.backward()
        self.assertIsNotNone(model.audio_head.grad)

    def test_fully_masked_batch_has_finite_zero_loss(self):
        config = CSMArchitectureConfig.tiny()
        model = CSMModel(config)
        tokens = torch.zeros(
            1,
            3,
            config.num_audio_codebooks + 1,
            dtype=torch.long,
        )
        mask = torch.ones_like(tokens, dtype=torch.bool)
        labels = torch.full(
            (1, 3, config.num_audio_codebooks),
            -100,
            dtype=torch.long,
        )
        output = model(tokens, mask, labels=labels)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.loss.item(), 0.0)
        output.loss.backward()
        self.assertIsNotNone(model.codebook0_head.weight.grad)

    def test_training_load_forwards_gated_artifact_controls(self):
        codec = SimpleNamespace(
            sample_rate=24_000,
            encode=lambda value: value,
            decode=lambda value: value,
        )
        backend = SimpleNamespace(
            model=object(),
            processor=object(),
            sample_rate=24_000,
            runtime=None,
        )
        wrapper = CSMForTextToSpeech(
            device="cpu",
            token="private-runtime-token",
            codec=codec,
            load_codec=False,
            local_files_only=True,
            verify_integrity=True,
            verify_checkpoint_integrity=True,
        )
        with patch(
                "voicehub.models.csm.training.load_csm_training_backend",
                return_value=backend,
        ) as loader:
            wrapper._loading_for_training = True
            try:
                wrapper._load_pretrained_model()
            finally:
                wrapper._loading_for_training = False
        loader.assert_called_once_with(
            "sesame/csm-1b",
            device="cpu",
            torch_dtype="bfloat16",
            codec=codec,
            include_codec=False,
            token="private-runtime-token",
            local_files_only=True,
            verify_integrity=True,
            verify_checkpoint_integrity=True,
        )


class NativeCSMCheckpointTests(unittest.TestCase):

    def test_safetensors_export_fresh_reload_has_exact_parity(self):
        torch.manual_seed(19)
        config = CSMArchitectureConfig.tiny()
        model = CSMModel(config).eval()
        tokens = torch.randint(
            0,
            config.audio_vocabulary_size,
            (1, 5, config.num_audio_codebooks + 1),
        )
        tokens[..., -1] %= config.text_vocabulary_size
        mask = torch.ones_like(tokens, dtype=torch.bool)
        expected = model(tokens, mask).logits
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = export_csm_checkpoint(
                model,
                Path(directory) / "model.safetensors",
            )
            restored = CSMModel(
                config,
                device="meta",
                dtype=torch.float32,
            )
            report = load_csm_checkpoint(
                restored,
                checkpoint,
                device="cpu",
            )
            actual = restored(tokens, mask).logits
        self.assertEqual(report.tensor_count, len(model.state_dict()))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_export_rejects_partial_state(self):
        model = CSMModel(CSMArchitectureConfig.tiny())
        state = dict(model.state_dict())
        state.pop(next(iter(state)))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                export_csm_checkpoint(
                    model,
                    Path(directory) / "model.safetensors",
                    state_override=state,
                )

    def test_flat_export_reloads_through_public_wrapper(self):

        class FakeCodec:
            sample_rate = 24_000

            def encode(self, waveform):
                return waveform

            def decode(self, codes):
                return codes

        torch.manual_seed(29)
        config = _portable_test_config()
        model = CSMModel(config).eval()
        tokens = torch.zeros(
            1,
            4,
            config.num_audio_codebooks + 1,
            dtype=torch.long,
        )
        tokens[..., -1] = torch.tensor([[2, 3, 4, 5]])
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask[..., -1] = True
        expected = model(tokens, mask).logits
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_path = root / "source-tokenizer.json"
            _write_test_tokenizer(tokenizer_path)
            processor = CSMProcessor(
                CSMTextTokenizer.from_file(tokenizer_path),
                config,
            )
            runtime = CSMRuntime(
                model,
                processor,
                codec=FakeCodec(),
            )
            export = runtime.save_pretrained(
                root / "export",
                include_codec=False,
            )
            restored = load_csm_runtime(
                export,
                dtype="float32",
                codec=FakeCodec(),
                include_codec=False,
            )
            wrapper = CSMForTextToSpeech.from_pretrained(
                export,
                device="cpu",
                lazy_load=False,
                codec=FakeCodec(),
                config_kwargs={
                    "torch_dtype": "float32",
                    "load_codec": False,
                },
            )
            restored_logits = restored.model(tokens, mask).logits
            wrapper_logits = wrapper.model(tokens, mask).logits
        torch.testing.assert_close(restored_logits, expected, rtol=0, atol=0)
        torch.testing.assert_close(wrapper_logits, expected, rtol=0, atol=0)


class NativeCSMRuntimeBoundaryTests(unittest.TestCase):

    @staticmethod
    def _runtime_components():
        config = _portable_test_config()
        model = CSMModel(config)
        processor = CSMProcessor(_tokenizer(), config)

        class Codec:
            sample_rate = 24_000

            def encode(self, waveform):
                return waveform

            def decode(self, codes):
                return codes

        return model, processor, Codec()

    def test_model_only_runtime_reports_raw_audio_boundary(self):
        config = CSMArchitectureConfig.tiny()
        model = CSMModel(config)

        class Processor:
            sample_rate = 24_000

        runtime = object.__new__(CSMRuntime)
        runtime.model = model
        runtime.processor = Processor()
        runtime.codec = None
        runtime.artifacts = None
        runtime.audio_postprocessor = None
        runtime.sample_rate = 24_000
        runtime.device = torch.device("cpu")
        with self.assertRaisesRegex(RuntimeError, "pre-encoded"):
            runtime.encode_audio(torch.ones(100), sampling_rate=24_000)

    def test_generic_postprocessor_does_not_claim_watermarking(self):
        model, processor, codec = self._runtime_components()
        runtime = CSMRuntime(
            model,
            processor,
            codec=codec,
            audio_postprocessor=lambda audio, sample_rate: audio,
        )
        self.assertFalse(runtime.audio_postprocessor_watermarks)

    def test_watermark_declaration_must_be_an_explicit_boolean(self):
        model, processor, codec = self._runtime_components()

        class InvalidPostprocessor:
            watermarks_audio = "yes"

            def __call__(self, audio, sample_rate):
                return audio

        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            CSMRuntime(
                model,
                processor,
                codec=codec,
                audio_postprocessor=InvalidPostprocessor(),
            )

    def test_nested_raw_segments_are_encoded_without_mutating_records(self):
        runtime = object.__new__(CSMRuntime)
        runtime.encode_audio = lambda audio, sampling_rate=None: torch.full(
            (4, 2),
            int(sampling_rate),
            dtype=torch.long,
        )
        records = [{
            "conversation_id":
            "sample",
            "segments": [{
                "speaker": 2,
                "text": "hello",
                "audio": torch.ones(8),
                "sampling_rate": 24_000,
            }],
        }]
        prepared = runtime.encode_training_records(records)
        self.assertIn("audio", records[0]["segments"][0])
        self.assertNotIn("audio", prepared[0]["segments"][0])
        self.assertEqual(
            prepared[0]["segments"][0]["audio_codes"].shape,
            (4, 2),
        )


if __name__ == "__main__":
    unittest.main()
