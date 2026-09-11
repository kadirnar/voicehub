from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from voicehub.models.orpheustts.configuration import OrpheusTTSConfig
from voicehub.models.orpheustts.modeling import OrpheusTTSForTextToSpeech
from voicehub.models.orpheustts.protocol import (
    AUDIO_TOKEN_OFFSET,
    END_SPEECH_TOKEN_ID,
    SNAC_CODEBOOK_SIZE,
    START_AI_TOKEN_ID,
    START_SPEECH_TOKEN_ID,
)
from voicehub.models.orpheustts.tokenization_orpheustts import (
    LLAMA3_SPLIT_PATTERN,
    OrpheusTokenizer,
    llama3_pretokenize,
)
from voicehub.policies.architecture_dependencies import inspect_native_imports
from voicehub.registry import get_model_spec
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.contracts import TrainingSupport
from voicehub.training.recipes import CodecCausalLMTrainingAdapter, OrpheusTrainingAdapter
from voicehub.training.specs import get_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _tiny_tokenizer_document() -> dict[str, object]:
    from voicehub.tokenization.assets import encode_gpt2_token

    vocabulary = {encode_gpt2_token(bytes((value, ))): value for value in range(256)}
    added_tokens = [
        {
            "id": 128000,
            "content": "<|begin_of_text|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        {
            "id": 128004,
            "content": "<|finetune_right_pad_id|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
    ]
    added_tokens.extend({
        "id": 128256 + index,
        "content": f"<custom_token_{index}>",
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": True,
        "special": False,
    } for index in range(11))
    # Retain the official model's ID-space upper bound without manufacturing
    # every unused spelling in this intentionally sparse test asset.
    added_tokens.append({
        "id": 156939,
        "content": "<custom_token_28683>",
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": True,
        "special": False,
    })
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type":
            "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {
                        "Regex": LLAMA3_SPLIT_PATTERN,
                    },
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type":
            "Sequence",
            "processors": [
                {
                    "type": "ByteLevel",
                    "add_prefix_space": True,
                    "trim_offsets": False,
                    "use_regex": True,
                },
                {
                    "type":
                    "TemplateProcessing",
                    "single": [
                        {
                            "SpecialToken": {
                                "id": "<|begin_of_text|>",
                                "type_id": 0,
                            },
                        },
                        {
                            "Sequence": {
                                "id": "A",
                                "type_id": 0,
                            },
                        },
                    ],
                    "pair": [],
                    "special_tokens": {
                        "<|begin_of_text|>": {
                            "id": "<|begin_of_text|>",
                            "ids": [128000],
                            "tokens": ["<|begin_of_text|>"],
                        },
                    },
                },
            ],
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocabulary,
            "merges": [],
        },
    }


def _tiny_lm_config():
    from voicehub.models.causal_lm.native.configuration import CausalLMConfig

    return CausalLMConfig.from_dict({
        "model_type": "llama",
        "vocab_size": 156940,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 131072,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        },
        "pad_token_id": 128004,
        "bos_token_id": 128000,
        "eos_token_id": 128009,
        "tie_word_embeddings": True,
    })


def _tiny_snac_config() -> dict[str, object]:
    return {
        "sampling_rate": 24000,
        "encoder_dim": 2,
        "encoder_rates": [2],
        "latent_dim": 4,
        "decoder_dim": 4,
        "decoder_rates": [2],
        "attn_window_size": None,
        "codebook_size": 4,
        "codebook_dim": 2,
        "vq_strides": [4, 2, 1],
        "noise": False,
        "depthwise": False,
    }


def _write_tiny_artifact(root: Path):
    import torch

    from voicehub.checkpointing import save_safetensors
    from voicehub.hub import write_json_file
    from voicehub.models.causal_lm.native.modeling import LlamaForCausalLM
    from voicehub.models.orpheustts.source.snac import SNAC

    torch.manual_seed(37)
    model = LlamaForCausalLM(_tiny_lm_config())
    model.save_pretrained(root)
    write_json_file(root / "tokenizer.json", _tiny_tokenizer_document())

    codec_root = root / "snac"
    codec_root.mkdir()
    codec_config = _tiny_snac_config()
    write_json_file(codec_root / "config.json", codec_config)
    codec = SNAC(**codec_config)
    save_safetensors(codec.state_dict(), codec_root / "model.safetensors")
    return model, codec


class NativeOrpheusDeclarationTests(unittest.TestCase):

    def test_public_import_loads_no_external_model_framework(self):
        code = """
import json
import sys
import voicehub.models.orpheustts
names = ("torch", "transformers", "tokenizers", "safetensors", "snac")
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
                "snac": False,
            },
        )

    def test_provider_source_obeys_native_import_policy(self):
        provider = PROJECT_ROOT / "voicehub" / "models" / "orpheustts"
        violations = tuple(
            violation for source in provider.rglob("*.py") for violation in inspect_native_imports(source))
        self.assertEqual(violations, ())

    def test_registry_and_training_profile_are_native(self):
        model_spec = get_model_spec("orpheustts")
        training_spec = get_training_spec("orpheustts")

        self.assertTrue(model_spec.is_voicehub_native)
        self.assertEqual(model_spec.architecture, "causal-lm")
        self.assertIn("fine-tuning", model_spec.capabilities)
        self.assertIs(training_spec.support, TrainingSupport.NATIVE)
        self.assertTrue(training_spec.native_training)
        self.assertEqual(
            training_spec.source_entrypoints,
            ("voicehub.models.causal_lm.native.modeling:"
             "CausalLMForCausalLM.forward", ),
        )

    def test_external_runtime_controls_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "never executes repository code"):
            OrpheusTTSConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "Safetensors only"):
            OrpheusTTSConfig(use_safetensors=False)
        with self.assertRaisesRegex(ValueError, "does not delegate"):
            OrpheusTTSConfig(model_kwargs={"device_map": "auto"})
        with self.assertRaisesRegex(ValueError, "sample_rate=24000"):
            OrpheusTTSConfig(sample_rate=16000)

    def test_llama3_scanner_preserves_official_boundaries(self):
        text = " Tara's 1234!\r\n  Merhaba 世界"
        pieces = llama3_pretokenize(text)

        self.assertEqual("".join(pieces), text)
        self.assertIn("'s", pieces)
        self.assertIn("123", pieces)
        self.assertIn("4", pieces)
        self.assertIn(" 世界", pieces)
        self.assertEqual(
            llama3_pretokenize(" \u2003>"),
            (" ", "\u2003", ">"),
        )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for native Orpheus")
class NativeOrpheusCheckpointTests(unittest.TestCase):

    def test_audited_real_snac_header_matches_vendored_graph(self):
        from voicehub.models.orpheustts.checkpoint import (
            REFERENCE_SNAC_CHECKPOINT,
            REFERENCE_SNAC_TENSOR_SHAPES,
            SNACCheckpointAdapter,
        )
        from voicehub.models.orpheustts.source.snac import SNAC

        codec = SNAC(
            sampling_rate=24000,
            encoder_dim=48,
            encoder_rates=[2, 4, 8, 8],
            latent_dim=768,
            decoder_dim=1024,
            decoder_rates=[8, 8, 4, 2],
            attn_window_size=None,
            codebook_size=4096,
            codebook_dim=8,
            vq_strides=[4, 2, 1],
            noise=True,
            depthwise=True,
        )
        state = codec.state_dict()

        self.assertEqual(
            len(state),
            REFERENCE_SNAC_CHECKPOINT["tensor_count"],
        )
        for name, shape in REFERENCE_SNAC_TENSOR_SHAPES.items():
            self.assertEqual(tuple(state[name].shape), shape, name)
        plan = SNACCheckpointAdapter.for_model(codec).tensor_plan({})
        self.assertEqual(len(plan.rules), len(state))
        self.assertEqual(
            REFERENCE_SNAC_CHECKPOINT["revision"],
            "c29a77c025506947a7ff15a678787b66b4c2ff47",
        )

    def test_tokenizer_uses_full_official_model_id_space(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            path.write_text(
                json.dumps(_tiny_tokenizer_document()),
                encoding="utf-8",
            )
            tokenizer = OrpheusTokenizer.from_tokenizer_json(path)

        encoding = tokenizer.encode("tara: Hello, world!")
        self.assertEqual(encoding.input_ids[0], 128000)
        self.assertEqual(tokenizer.token_id_space_size, 156940)
        self.assertLess(tokenizer.vocabulary_size, tokenizer.token_id_space_size)

    def test_tiny_inference_training_and_export_round_trip(self):
        import torch

        class DeterministicGenerator(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.input_ids = None
                self.generation_config = None

            def generate(
                self,
                input_ids,
                *,
                attention_mask,
                generation_config,
            ):
                del attention_mask
                self.input_ids = input_ids.detach().clone()
                self.generation_config = generation_config
                frame = [AUDIO_TOKEN_OFFSET + channel * SNAC_CODEBOOK_SIZE for channel in range(7)]
                continuation = torch.tensor(
                    [frame + [END_SPEECH_TOKEN_ID]],
                    dtype=torch.long,
                    device=input_ids.device,
                )
                return SimpleNamespace(sequences=torch.cat((input_ids, continuation), dim=1), )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_model, _ = _write_tiny_artifact(root)
            wrapper = OrpheusTTSForTextToSpeech(
                root,
                device="cpu",
                lazy_load=False,
                torch_dtype="float32",
            )
            for name, expected in reference_model.state_dict().items():
                torch.testing.assert_close(
                    wrapper.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )

            native_model = wrapper.model
            generator = DeterministicGenerator()
            wrapper.model = generator
            inference = wrapper._generate(
                "Hello",
                voice="tara",
                max_new_tokens=8,
                seed=19,
            )
            self.assertEqual(
                tuple(generator.input_ids[0, -2:].tolist()),
                (START_AI_TOKEN_ID, START_SPEECH_TOKEN_ID),
            )
            self.assertEqual(generator.generation_config.max_new_tokens, 8)
            self.assertEqual(generator.generation_config.top_p, 0.8)
            self.assertEqual(
                generator.generation_config.repetition_penalty,
                1.3,
            )
            self.assertEqual(inference.metadata["audio_tokens"], 7)
            self.assertEqual(inference.sample_rate, 24000)
            self.assertTrue(torch.isfinite(inference.audio).all())

            wrapper.model = native_model
            dataset = wrapper.create_training_dataset(
                [{
                    "text": "Hello",
                    "voice": "tara",
                    "audio_codes": (
                        [1],
                        [2, 3],
                        [0, 1, 2, 3],
                    ),
                }],
                completion_only=True,
                max_length=128,
            )
            batch = dataset.collate_fn([dataset[0]])
            adapter = AutoTrainingAdapter.from_model(wrapper)
            self.assertIsInstance(adapter, OrpheusTrainingAdapter)
            self.assertIsInstance(adapter, CodecCausalLMTrainingAdapter)
            output = adapter.execute_training_phase(adapter.create_training_context(batch), )
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(wrapper.model.lm_head.weight.grad)

            export = root / "export"
            adapter.save_pretrained(export)
            self.assertTrue((export / "model.safetensors").is_file())
            self.assertTrue((export / "tokenizer.json").is_file())
            self.assertTrue((export / "snac" / "model.safetensors").is_file())
            restored = OrpheusTTSForTextToSpeech(
                export,
                device="cpu",
                lazy_load=False,
                torch_dtype="float32",
            )
            for name, expected in wrapper.model.state_dict().items():
                torch.testing.assert_close(
                    restored.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
