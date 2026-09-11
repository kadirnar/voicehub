"""Executable contracts for VoiceHub-native OuteTTS."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from voicehub.models.causal_lm.native.configuration import LlamaConfig
from voicehub.models.outetts.inference import OuteTTSConfig, OuteTTSForTextToSpeech
from voicehub.models.outetts.native.artifacts import OuteTTSArtifacts
from voicehub.models.outetts.native.checkpoint import load_outetts_language_model
from voicehub.models.outetts.native.modeling import OuteTTSForCausalLM, RecentWindowRepetitionProcessor
from voicehub.models.outetts.native.prompting import OuteTTSPromptProcessor
from voicehub.models.outetts.native.tokenization import OuteTTSTokenizer
from voicehub.models.outetts.training import OuteTTSSFTDataset, OuteTTSTrainingAdapter
from voicehub.tokenization import ByteBPETokenizer
from voicehub.training import AutoTrainingAdapter, TrainingSupport, get_training_spec


def _tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )


def _protocol_tokenizer(path: Path) -> OuteTTSTokenizer:
    token_ids = {
        "<|im_start|>": 133_309,
        "<|im_end|>": 133_310,
        "<|text_start|>": 133_311,
        "<|text_end|>": 133_312,
        "<|audio_start|>": 133_317,
        "<|audio_end|>": 133_318,
        "<|word_start|>": 133_320,
        "<|word_end|>": 133_321,
        "<|features|>": 133_322,
        "<|global_features_start|>": 133_323,
        "<|global_features_end|>": 133_324,
        "<|code|>": 133_325,
        "<|t_0.20|>": 133_326,
        "<|energy_1|>": 133_327,
        "<|spectral_centroid_2|>": 133_328,
        "<|pitch_3|>": 133_329,
        "<|energy_4|>": 133_330,
        "<|spectral_centroid_5|>": 133_331,
        "<|pitch_6|>": 133_332,
        "<|c1_1|>": 128_257,
        "<|c1_2|>": 128_258,
        "<|c2_3|>": 129_284,
        "<|c2_4|>": 129_285,
    }
    tokenizer = ByteBPETokenizer(
        {bytes((value, )): value
         for value in range(256)},
        special_tokens=token_ids,
        pad_token_id=token_ids["<|im_end|>"],
        use_regex=False,
    )
    return OuteTTSTokenizer(
        tokenizer,
        family="llama",
        token_ids=token_ids,
        tokenizer_path=path,
    )


def _profile() -> dict:
    return {
        "text":
        "Hello",
        "words": [{
            "word": "Hello",
            "duration": 0.2,
            "c1": [1, 2],
            "c2": [3, 4],
            "features": {
                "energy": 1,
                "spectral_centroid": 2,
                "pitch": 3,
            },
        }],
        "global_features": {
            "energy": 4,
            "spectral_centroid": 5,
            "pitch": 6,
        },
        "interface_version":
        3,
    }


class NativeOuteTTSModelTests(unittest.TestCase):

    def test_causal_objective_backpropagates_through_language_model(self):
        model = OuteTTSForCausalLM(_tiny_config())
        input_ids = torch.tensor([[1, 5, 6, 7]], dtype=torch.long)
        labels = torch.tensor([[-100, 5, 6, 7]], dtype=torch.long)

        output = model(input_ids, labels=labels)
        self.assertIsNotNone(output.loss)
        output.loss.backward()

        self.assertIsNotNone(model.model.embed_tokens.weight.grad)
        self.assertGreater(
            float(model.model.embed_tokens.weight.grad.abs().sum()),
            0.0,
        )
        self.assertIsNotNone(model.lm_head.weight.grad)

    def test_repetition_penalty_is_scoped_to_recent_window(self):
        processor = RecentWindowRepetitionProcessor(2.0, window=2)
        logits = torch.ones((1, 8))
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        processed = processor(tokens, logits)

        self.assertEqual(float(processed[0, 1]), 1.0)
        self.assertEqual(float(processed[0, 2]), 1.0)
        self.assertEqual(float(processed[0, 3]), 0.5)
        self.assertEqual(float(processed[0, 4]), 0.5)

    def test_safetensors_export_loads_into_a_fresh_native_graph(self):
        torch.manual_seed(7)
        original = OuteTTSForCausalLM(_tiny_config())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original.save_pretrained(root)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("{}\n", encoding="utf-8")
            artifacts = OuteTTSArtifacts(
                source=str(root),
                revision=None,
                config=root / "config.json",
                tokenizer=tokenizer,
                checkpoint=root / "model.safetensors",
            )

            restored, restored_config = load_outetts_language_model(
                artifacts,
                device="cpu",
                dtype=torch.float32,
            )

        self.assertEqual(restored_config, original.config)
        for name, expected in original.state_dict().items():
            self.assertTrue(torch.equal(restored.state_dict()[name], expected))


class NativeOuteTTSPromptAndTrainingTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        path = Path(self.temporary.name) / "tokenizer.json"
        path.write_text("{}\n", encoding="utf-8")
        self.tokenizer = _protocol_tokenizer(path)
        self.processor = OuteTTSPromptProcessor(self.tokenizer)

    def tearDown(self):
        self.temporary.cleanup()

    def test_v3_profile_builds_completion_only_training_labels(self):
        runtime = type(
            "Runtime",
            (),
            {"prompt_processor": self.processor},
        )()
        dataset = OuteTTSSFTDataset(
            [{
                "speaker_profile": _profile()
            }],
            runtime=runtime,
            completion_only=True,
        )

        item = dataset[0]

        audio_start = self.tokenizer.convert_tokens_to_ids(OuteTTSPromptProcessor.AUDIO_START)
        completion_start = item["input_ids"].index(audio_start) + 1
        self.assertEqual(
            item["labels"][:completion_start],
            [-100] * completion_start,
        )
        self.assertTrue(all(value >= 0 for value in item["labels"][completion_start:]))

    def test_raw_audio_training_fails_closed(self):
        runtime = type(
            "Runtime",
            (),
            {"prompt_processor": self.processor},
        )()
        dataset = OuteTTSSFTDataset(
            [{
                "text": "Hello",
                "audio": "speaker.wav"
            }],
            runtime=runtime,
        )

        with self.assertRaisesRegex(ValueError, "raw audio.*V3 speaker profile"):
            dataset[0]

    def test_profile_validation_rejects_misaligned_codebooks(self):
        profile = _profile()
        profile["words"][0]["c2"] = [3]
        runtime = type(
            "Runtime",
            (),
            {"prompt_processor": self.processor},
        )()
        dataset = OuteTTSSFTDataset(
            [{
                "speaker_profile": profile
            }],
            runtime=runtime,
        )

        with self.assertRaisesRegex(ValueError, "different lengths"):
            dataset[0]


class NativeOuteTTSBoundaryTests(unittest.TestCase):

    def test_shared_registries_resolve_the_native_training_contract(self):
        from voicehub.registry import get_model_spec
        from voicehub.runtime import get_architecture_spec

        model_spec = get_model_spec("outetts")
        training_spec = get_training_spec("outetts")
        architecture = get_architecture_spec("outetts")
        wrapper = OuteTTSForTextToSpeech(device="cpu")

        self.assertTrue(model_spec.is_voicehub_native)
        self.assertEqual(model_spec.architecture, "outetts")
        self.assertIs(model_spec.native_architecture, architecture)
        self.assertTrue(architecture.capabilities.training)
        self.assertEqual(training_spec.support, TrainingSupport.PREPROCESSED)
        self.assertEqual(
            training_spec.module_paths,
            ("model.language_model", ),
        )
        adapter = AutoTrainingAdapter.from_model(wrapper, spec=training_spec)
        self.assertIsInstance(adapter, OuteTTSTrainingAdapter)
        self.assertFalse(wrapper.is_loaded)

    def test_training_adapter_resolves_only_the_language_model(self):
        language_model = OuteTTSForCausalLM(_tiny_config())
        codec = torch.nn.Linear(2, 2)
        runtime = torch.nn.Module()
        runtime.language_model = language_model
        runtime.codec = codec

        class Wrapper:

            def __init__(self):
                self.config = SimpleNamespace(
                    model_type="outetts",
                    name_or_path="",
                )
                self.model = runtime
                self.loaded_for_training = False

            def load_for_training(self):
                self.loaded_for_training = True

        wrapper = Wrapper()
        adapter = AutoTrainingAdapter.from_model(wrapper)
        adapter.setup()

        self.assertTrue(wrapper.loaded_for_training)
        self.assertIs(adapter.primary_model, language_model)
        self.assertEqual(adapter.primary_path, "model.language_model")
        self.assertTrue(all(not parameter.requires_grad for parameter in codec.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in language_model.parameters()))

    def test_external_and_quantized_backends_fail_before_loading(self):
        external = OuteTTSForTextToSpeech(
            OuteTTSConfig(backend="LLAMACPP"),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "external runtime"):
            external.generate("hello", generation_type="regular")
        self.assertFalse(external.is_loaded)

        quantized = OuteTTSForTextToSpeech(
            OuteTTSConfig(additional_model_config={"load_in_4bit": True}),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "quantized"):
            quantized.load_for_training()
        self.assertFalse(quantized.is_loaded)

    def test_active_modules_do_not_import_provider_runtimes(self):
        roots = (
            Path("voicehub/models/outetts/native"),
            Path("voicehub/models/outetts/inference.py"),
            Path("voicehub/models/outetts/training.py"),
        )
        prohibited = (
            "transformers",
            "huggingface_hub",
            "llama_cpp",
            "vllm",
            "exllamav2",
        )
        for root in roots:
            paths = sorted(root.glob("*.py")) if root.is_dir() else [root]
            for path in paths:
                source = path.read_text(encoding="utf-8")
                for dependency in prohibited:
                    self.assertNotIn(
                        f"import {dependency}",
                        source,
                        msg=f"{path} imports {dependency}",
                    )

    def test_provenance_document_is_valid_json(self):
        path = Path("voicehub/models/outetts/native/SOURCE.json")
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["architecture"], "outetts")
        self.assertEqual(
            len(document["implementation_sources"]),
            3,
        )


if __name__ == "__main__":
    unittest.main()
