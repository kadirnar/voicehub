from __future__ import annotations

import ast
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from voicehub.models.causal_lm.native import LlamaConfig, LlamaForCausalLM
from voicehub.models.llasa.artifacts import LLASA_MULTILINGUAL_REVISION, XCODEC2_HF_REVISION, resolve_xcodec2_artifacts
from voicehub.models.llasa.checkpoint import (
    REFERENCE_LLASSA_CHECKPOINT,
    REFERENCE_XCODEC2_CHECKPOINT,
    REFERENCE_XCODEC2_TENSOR_SHAPES,
)
from voicehub.models.llasa.configuration import LlasaConfig
from voicehub.models.llasa.modeling import LlasaForTextToSpeech
from voicehub.models.llasa.tokenization_llasa import (
    BOS_TOKEN,
    END_HEADER_TOKEN,
    EOT_TOKEN,
    LLASA_SPEECH_TOKEN_OFFSET,
    LLASA_VOCABULARY_SIZE,
    SPEECH_GENERATION_END,
    SPEECH_GENERATION_START,
    START_HEADER_TOKEN,
    LlasaTokenizer,
)
from voicehub.models.llasa.training import LLASA_TRAINING_SOURCE_REVISION, LlasaSFTDataset
from voicehub.models.llasa.xcodec2 import (
    XCODEC2_TRANSFORMERS_SOURCE_REVISION,
    Wav2Vec2BertSemanticConfig,
    XCodec2Config,
    XCodec2FiniteScalarQuantization,
    XCodec2Model,
)
from voicehub.processing.waveform import save_pcm_wave

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_xcodec_config() -> XCodec2Config:
    semantic = Wav2Vec2BertSemanticConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        conv_depthwise_kernel_size=3,
        output_hidden_size=32,
        layerdrop=0.0,
        conformer_conv_dropout=0.0,
        mask_time_prob=0.0,
    )
    return XCodec2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        encoder_hidden_size=1,
        quantization_dim=64,
        semantic_model_config=semantic,
        activation_dropout=0.0,
    )


class _FakeLlasaTokenizer:

    pad_token_id = 128_009
    eos_token_id = 128_009

    def __init__(self) -> None:
        self.last_messages = None

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        return_tensors=None,
        continue_final_message=False,
        add_generation_prompt=False,
    ):
        del tokenize, continue_final_message, add_generation_prompt
        self.last_messages = messages
        if return_tensors == "pt":
            return torch.tensor([[10, 11]], dtype=torch.long)
        assistant = messages[-1]["content"]
        codes = [int(value) for value in re.findall(r"<\|s_(\d+)\|>", assistant)]
        return [
            10,
            11,
            128_260,
            *(LLASA_SPEECH_TOKEN_OFFSET + value for value in codes),
            128_261,
            128_009,
        ]

    @staticmethod
    def convert_tokens_to_ids(token):
        return {
            SPEECH_GENERATION_START: 128_260,
            SPEECH_GENERATION_END: 128_261,
            EOT_TOKEN: 128_009,
        }[token]

    @staticmethod
    def convert_ids_to_tokens(token_ids):
        output = []
        for token_id in token_ids:
            if token_id == 128_261:
                output.append(SPEECH_GENERATION_END)
            elif LLASA_SPEECH_TOKEN_OFFSET <= token_id < LLASA_VOCABULARY_SIZE:
                output.append(f"<|s_{token_id - LLASA_SPEECH_TOKEN_OFFSET}|>")
            else:
                output.append(f"<token_{token_id}>")
        return output


class _FakeCodec(nn.Module):

    sampling_rate = 16_000
    hop_length = 320

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.last_waveform_shape = None

    def encode_code(self, input_waveform, sample_rate=16_000):
        if sample_rate != self.sampling_rate:
            raise ValueError("unexpected sample rate")
        self.last_waveform_shape = tuple(input_waveform.shape)
        return torch.tensor(
            [[[5, 6]]],
            dtype=torch.long,
            device=input_waveform.device,
        )

    def decode_code(self, audio_codes):
        samples = audio_codes.shape[-1] * self.hop_length
        return torch.linspace(
            -0.25,
            0.25,
            samples,
            device=audio_codes.device,
        ).view(1, 1, -1)


class _FakeLanguageModel:

    def __init__(self) -> None:
        self.generation_config = None

    def generate(
        self,
        *,
        input_ids,
        attention_mask,
        generation_config,
    ):
        if not torch.equal(attention_mask, torch.ones_like(input_ids)):
            raise AssertionError("prompt attention mask must be dense")
        self.generation_config = generation_config
        continuation = torch.tensor(
            [[LLASA_SPEECH_TOKEN_OFFSET + 7, 128_261]],
            dtype=torch.long,
            device=input_ids.device,
        )
        return SimpleNamespace(sequences=torch.cat((input_ids, continuation), dim=-1))


class LlasaDependencyAndProvenanceTests(unittest.TestCase):

    def test_registry_and_training_contract_expose_native_end_to_end_support(self):
        from voicehub.registry import get_model_spec
        from voicehub.runtime import get_architecture_spec
        from voicehub.training.recipes import BUILTIN_MODEL_ADAPTERS
        from voicehub.training.specs import get_training_spec

        model_spec = get_model_spec("llasa")
        architecture = get_architecture_spec("llasa")
        training = get_training_spec("llasa")

        self.assertTrue(model_spec.is_voicehub_native)
        self.assertIs(model_spec.native_architecture, architecture)
        self.assertTrue(architecture.capabilities.training)
        self.assertEqual(architecture.license_id, "CC-BY-NC-4.0")
        self.assertIn(
            "raw-audio-fine-tuning",
            architecture.capabilities.features,
        )
        self.assertTrue(training.native_training)
        self.assertEqual(training.support.value, "native")
        self.assertEqual(training.default_phase, "codec_language_model")
        self.assertEqual(
            training.adapter_factory,
            "voicehub.models.llasa.training:LlasaTrainingAdapter",
        )
        self.assertEqual(
            BUILTIN_MODEL_ADAPTERS["llasa"].__name__,
            "LlasaTrainingAdapter",
        )

    def test_public_execution_path_has_no_provider_imports(self):
        model_root = PROJECT_ROOT / "voicehub" / "models" / "llasa"
        public_files = (
            "artifacts.py",
            "checkpoint.py",
            "configuration.py",
            "modeling.py",
            "tokenization_llasa.py",
            "training.py",
            "xcodec2.py",
        )
        forbidden = {
            "einops",
            "huggingface_hub",
            "numpy",
            "soundfile",
            "torchaudio",
            "torchtune",
            "transformers",
            "vector_quantize_pytorch",
            "xcodec2",
        }
        findings = []
        for filename in public_files:
            path = model_root / filename
            tree = ast.parse(path.read_text(encoding="utf-8"), filename)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    root = name.split(".", 1)[0]
                    if root in forbidden:
                        findings.append((filename, node.lineno, name))
        self.assertEqual(findings, [])

    def test_pinned_sources_and_noncommercial_license_are_explicit(self):
        source = json.loads((PROJECT_ROOT / "voicehub" / "models" / "llasa" / "source" /
                             "SOURCE.json").read_text(encoding="utf-8"))
        revisions = {component["revision"] for component in source["components"]}
        self.assertEqual(source["license"], "CC-BY-NC-4.0")
        self.assertIn(LLASA_MULTILINGUAL_REVISION, revisions)
        self.assertIn(XCODEC2_HF_REVISION, revisions)
        self.assertIn(LLASA_TRAINING_SOURCE_REVISION, revisions)
        self.assertIn(XCODEC2_TRANSFORMERS_SOURCE_REVISION, revisions)

    def test_legacy_remote_code_codec_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "self-contained"):
            resolve_xcodec2_artifacts("HKUSTAudio/xcodec2")

    def test_config_rejects_external_execution_switches(self):
        with self.assertRaisesRegex(ValueError, "never executes"):
            LlasaConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "Safetensors only"):
            LlasaConfig(use_safetensors=False)


class LlasaProtocolTests(unittest.TestCase):

    def test_protocol_ids_cover_the_complete_xcodec_codebook(self):
        self.assertEqual(LlasaTokenizer.speech_code_to_token_id(0), 128_264)
        self.assertEqual(
            LlasaTokenizer.speech_code_to_token_id(65_535),
            193_799,
        )
        self.assertEqual(
            LlasaTokenizer.token_id_to_speech_code(193_799),
            65_535,
        )
        with self.assertRaisesRegex(ValueError, "speech token"):
            LlasaTokenizer.token_id_to_speech_code(128_263)

    def test_chat_rendering_preserves_the_published_llama32_template(self):
        rendered = LlasaTokenizer.format_chat(
            [
                {
                    "role": "user",
                    "content": " request "
                },
                {
                    "role": "assistant",
                    "content": " answer "
                },
            ],
            continue_final_message=True,
        )
        self.assertTrue(
            rendered.startswith(
                BOS_TOKEN + START_HEADER_TOKEN + "system" + END_HEADER_TOKEN + "\n\n" +
                "Cutting Knowledge Date: December 2023\n" + "Today Date: 26 Jul 2024\n\n" + EOT_TOKEN))
        self.assertTrue(rendered.endswith("\n\nanswer"))
        self.assertNotIn(EOT_TOKEN, rendered[-len(EOT_TOKEN):])

    def test_malformed_and_out_of_range_speech_tokens_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "malformed speech token"):
            LlasaForTextToSpeech._extract_speech_ids(["<|s_not-an-integer|>"])
        with self.assertRaisesRegex(RuntimeError, "out-of-range"):
            LlasaForTextToSpeech._extract_speech_ids(["<|s_65536|>"])


class XCodec2NativeGraphTests(unittest.TestCase):

    def test_official_llasa_language_model_namespace_matches_exactly(self):
        config = LlamaConfig(
            vocab_size=193_800,
            hidden_size=2_048,
            intermediate_size=8_192,
            num_hidden_layers=16,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=64,
            max_position_embeddings=131_072,
            rms_norm_eps=1e-5,
            rope_theta=500_000.0,
            rope_scaling={
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8_192,
                "rope_type": "llama3",
            },
            bos_token_id=128_000,
            eos_token_id=128_009,
            tie_word_embeddings=True,
        )
        with torch.device("meta"):
            model = LlamaForCausalLM(config, initialize=False)
        state = model.state_dict()
        self.assertEqual(
            len(state),
            REFERENCE_LLASSA_CHECKPOINT["tensor_count"],
        )
        self.assertEqual(
            tuple(state["model.embed_tokens.weight"].shape),
            (193_800, 2_048),
        )
        self.assertEqual(
            tuple(state["model.layers.15.self_attn.q_proj.weight"].shape),
            (2_048, 2_048),
        )
        self.assertEqual(
            tuple(state["lm_head.weight"].shape),
            (193_800, 2_048),
        )

    def test_official_checkpoint_namespace_and_shapes_match_exactly(self):
        with torch.device("meta"):
            model = XCodec2Model(XCodec2Config(), initialize=False)
        state = model.state_dict()
        self.assertEqual(
            len(state),
            REFERENCE_XCODEC2_CHECKPOINT["tensor_count"],
        )
        for name, expected_shape in REFERENCE_XCODEC2_TENSOR_SHAPES.items():
            with self.subTest(name=name):
                self.assertIn(name, state)
                self.assertEqual(tuple(state[name].shape), expected_shape)

    def test_fsq_indices_round_trip_and_keep_straight_through_gradient(self):
        quantizer = XCodec2FiniteScalarQuantization(_tiny_xcodec_config())
        hidden_states = torch.linspace(
            -2.0,
            2.0,
            24,
            dtype=torch.float32,
        ).reshape(3, 8)
        hidden_states.requires_grad_(True)
        codes, indices = quantizer(hidden_states)
        restored = quantizer.codes_from_indices(indices.long())
        self.assertTrue(torch.equal(codes.detach(), restored))
        codes.square().sum().backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(torch.isfinite(hidden_states.grad).all())
        self.assertGreater(hidden_states.grad.abs().sum().item(), 0.0)

    def test_tiny_raw_pcm_graph_encodes_and_decodes_aligned_frames(self):
        model = XCodec2Model(_tiny_xcodec_config()).eval()
        waveform = torch.linspace(-0.2, 0.2, 1_600).unsqueeze(0)
        with torch.inference_mode():
            features = model.feature_extractor(waveform)
            encoded = model.encode_audio(waveform)
            decoded = model.decode_code(encoded.audio_codes)
        self.assertEqual(tuple(features.input_values.shape), (1, 1, 1_920))
        self.assertEqual(tuple(features.input_features.shape), (1, 6, 160))
        self.assertEqual(tuple(encoded.audio_codes.shape), (1, 1, 6))
        self.assertEqual(tuple(decoded.shape), (1, 1, 1_920))
        self.assertTrue(torch.isfinite(decoded).all())

    def test_incompatible_frontend_metadata_fails_closed(self):
        model = XCodec2Model(_tiny_xcodec_config()).eval()
        with self.assertRaisesRegex(ValueError, "frontend metadata"):
            model.feature_extractor.validate_preprocessor_config({
                "sampling_rate": 24_000,
                "hop_length": 320,
            })

    def test_tiny_safetensors_export_reloads_strictly(self):
        model = XCodec2Model(_tiny_xcodec_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = XCodec2Model.from_pretrained(directory, strict=True)
        self.assertEqual(
            tuple(restored.quantizer.project_in.weight.shape),
            (8, 64),
        )
        self.assertTrue(
            torch.equal(
                model.quantizer.project_in.weight,
                restored.quantizer.project_in.weight,
            ))


class LlasaTrainingAndInferenceTests(unittest.TestCase):

    def test_precomputed_recipe_masks_text_and_preserves_speaker_prefix(self):
        tokenizer = _FakeLlasaTokenizer()
        dataset = LlasaSFTDataset(
            [{
                "text": "hello",
                "speaker": "Paimon",
                "audio_codes": [2, 3],
            }],
            tokenizer=tokenizer,
            codec=None,
        )
        example = dataset[0]
        speech_start = example["input_ids"].index(128_260)
        self.assertEqual(example["labels"][:speech_start], [-100] * speech_start)
        self.assertEqual(
            example["labels"][speech_start:],
            example["input_ids"][speech_start:],
        )
        self.assertTrue(
            tokenizer.last_messages[-1]["content"].startswith("Speaker Paimon" + SPEECH_GENERATION_START))

    def test_raw_audio_recipe_resamples_and_freezes_xcodec(self):
        tokenizer = _FakeLlasaTokenizer()
        codec = _FakeCodec()
        dataset = LlasaSFTDataset(
            [{
                "text": "hello",
                "waveform": torch.linspace(-0.5, 0.5, 80),
                "sampling_rate": 8_000,
            }],
            tokenizer=tokenizer,
            codec=codec,
        )
        example = dataset[0]
        self.assertEqual(codec.last_waveform_shape, (1, 160))
        self.assertFalse(codec.training)
        self.assertFalse(codec.anchor.requires_grad)
        self.assertIn(LLASA_SPEECH_TOKEN_OFFSET + 5, example["input_ids"])
        self.assertIn(LLASA_SPEECH_TOKEN_OFFSET + 6, example["input_ids"])

    def test_completion_batch_backpropagates_through_native_llama_only(self):
        tokenizer = _FakeLlasaTokenizer()
        codec = _FakeCodec()
        dataset = LlasaSFTDataset(
            [{
                "text": "hello",
                "audio_codes": [2, 3]
            }],
            tokenizer=tokenizer,
            codec=codec,
        )
        batch = dataset.collate_fn([dataset[0]])
        config = LlamaConfig(
            vocab_size=LLASA_VOCABULARY_SIZE,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            pad_token_id=128_009,
            bos_token_id=128_000,
            eos_token_id=128_009,
            tie_word_embeddings=True,
        )
        model = LlamaForCausalLM(config)
        model.tie_weights()
        output = model(**batch)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.model.embed_tokens.weight.grad)
        self.assertIsNone(codec.anchor.grad)

    def test_native_generation_and_voice_clone_slice_exact_codec_frames(self):
        wrapper = LlasaForTextToSpeech(device="cpu")
        tokenizer = _FakeLlasaTokenizer()
        codec = _FakeCodec()
        language_model = _FakeLanguageModel()
        wrapper.tokenizer = tokenizer
        wrapper.codec = codec
        wrapper.model = language_model
        wrapper._torch = torch
        with tempfile.TemporaryDirectory() as directory:
            reference = save_pcm_wave(
                Path(directory) / "reference.wav",
                torch.linspace(-0.2, 0.2, 160),
                16_000,
            )
            output = wrapper._generate(
                "target",
                speaker_audio_path=str(reference),
                reference_text="reference",
                max_new_tokens=4,
                seed=123,
            )
        self.assertEqual(tuple(output.audio.shape), (320, ))
        self.assertEqual(output.metadata["prompt_audio_tokens"], 2)
        self.assertEqual(output.metadata["audio_tokens"], 1)
        self.assertEqual(language_model.generation_config.max_new_tokens, 4)
        self.assertEqual(language_model.generation_config.seed, 123)


if __name__ == "__main__":
    unittest.main()
