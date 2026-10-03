from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_backbone_values(*, linear_rope: bool = False) -> dict:
    values = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 32,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "tie_word_embeddings": False,
    }
    if linear_rope:
        values["rope_scaling"] = {
            "factor": 2.0,
            "rope_type": "linear",
            "type": "linear",
        }
    return values


class _DatasetTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    speech_start_id = 20
    speech_end_id = 21
    speech_offset = 30

    @staticmethod
    def encode(text, *, add_special_tokens=False):
        from voicehub.tokenization import Encoding

        del add_special_tokens
        if "<|SPEECH_GENERATION_START|>" not in text:
            raise AssertionError("Dataset prompt omitted the speech marker.")
        return Encoding((11, _DatasetTokenizer.speech_start_id))

    @staticmethod
    def convert_tokens_to_ids(token):
        from voicehub.architectures.neutts.tokenization import SPEECH_GENERATION_END, SPEECH_GENERATION_START

        return {
            SPEECH_GENERATION_START: _DatasetTokenizer.speech_start_id,
            SPEECH_GENERATION_END: _DatasetTokenizer.speech_end_id,
        }[token]

    @staticmethod
    def speech_code_to_token_id(code):
        return _DatasetTokenizer.speech_offset + code


@unittest.skipUnless(TORCH_AVAILABLE, "Native NeuTTS requires PyTorch")
class NativeNeuTTSTests(unittest.TestCase):

    def test_neucodec_aligns_rounding_mismatch_like_published_runtime(self):
        import torch

        from voicehub.architectures.neutts.neucodec import NeuCodecModel

        semantic = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
        acoustic = torch.arange(2 * 5 * 5).reshape(2, 5, 5)
        aligned_semantic, aligned_acoustic = NeuCodecModel._align_encoder_frames(
            semantic,
            acoustic,
        )

        self.assertEqual(aligned_semantic.shape, (2, 3, 4))
        self.assertEqual(aligned_acoustic.shape, (2, 5, 4))
        self.assertTrue(torch.equal(aligned_semantic, semantic))
        self.assertTrue(torch.equal(aligned_acoustic, acoustic[..., :4]))

        with self.assertRaisesRegex(RuntimeError, "no alignable audio frames"):
            NeuCodecModel._align_encoder_frames(
                semantic[..., :0],
                acoustic[..., :0],
            )

    def test_configuration_import_keeps_neutts_graph_lazy(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import voicehub.models.neutts.configuration_neutts; "
                    "print(*(int(name in sys.modules) for name in ("
                    "'voicehub.models.neutts.inference', "
                    "'voicehub.architectures.neutts.modeling', "
                    "'voicehub.architectures.neutts.neucodec')))"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "0 0 0")

    def test_public_config_identity_is_canonical(self):
        from voicehub.models.neutts import NeuTTSConfig
        from voicehub.models.neutts.configuration_neutts import NeuTTSConfig as ConfigurationNeuTTSConfig
        from voicehub.models.neutts.inference import NeuTTSForTextToSpeech

        self.assertIs(NeuTTSConfig, ConfigurationNeuTTSConfig)
        self.assertIs(
            NeuTTSForTextToSpeech.config_class,
            ConfigurationNeuTTSConfig,
        )

    def test_active_runtime_has_no_provider_imports(self):
        roots = (
            PROJECT_ROOT / "voicehub" / "architectures" / "neutts",
            PROJECT_ROOT / "voicehub" / "models" / "neutts",
        )
        model_files = {
            "configuration_neutts.py",
            "inference.py",
            "modeling_neutts.py",
            "training.py",
        }
        forbidden = {
            "espeak",
            "huggingface_hub",
            "librosa",
            "neucodec",
            "neutts",
            "numpy",
            "onnxruntime",
            "phonemizer",
            "soundfile",
            "torchaudio",
            "transformers",
        }
        findings = []
        for root in roots:
            candidates = (
                root.glob("*.py") if root.name == "neutts" and root.parent.name == "architectures" else
                (root / name for name in model_files))
            for path in candidates:
                tree = ast.parse(path.read_text(encoding="utf-8"), path.name)
                for node in ast.walk(tree):
                    imported = []
                    if isinstance(node, ast.Import):
                        imported = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported = [node.module]
                    for module in imported:
                        if module.split(".", 1)[0] in forbidden:
                            findings.append(f"{path.name}:{module}")
                        if module.startswith("voicehub.models.neutts.source"):
                            findings.append(f"{path.name}:{module}")
        self.assertEqual(findings, [])

    def test_provenance_and_pinned_checkpoint_contracts_are_recorded(self):
        from voicehub.architectures.neutts.metadata import (
            NEUCODEC_REFERENCE,
            NEUCODEC_SOURCE_REVISION,
            NEUTTS_SOURCE_REVISION,
            NEUTTS_TRAINING_SOURCE,
            NEUTTS_VARIANTS,
        )

        source_path = (PROJECT_ROOT / "voicehub" / "architectures" / "neutts" / "SOURCE.json")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        codec = source["checkpoints"]["neucodec"]

        self.assertEqual(
            source["implementation_sources"][0]["revision"],
            NEUTTS_SOURCE_REVISION,
        )
        self.assertEqual(
            source["implementation_sources"][1]["revision"],
            NEUCODEC_SOURCE_REVISION,
        )
        self.assertEqual(codec["sha256"], NEUCODEC_REFERENCE["sha256"])
        self.assertEqual(codec["tensor_count"], 811)
        self.assertEqual(codec["state_values"], 629_937_706)
        self.assertEqual(
            NEUTTS_TRAINING_SOURCE["model_family"],
            "neutts-air",
        )
        self.assertEqual(
            set(source["checkpoints"]["neutts"]["repositories"]),
            set(NEUTTS_VARIANTS),
        )

    def test_lazy_architecture_spec_declares_truthful_native_boundaries(self):
        from voicehub.architectures.neutts.registration import create_neutts_architecture_spec
        from voicehub.registry import get_model_spec
        from voicehub.training.contracts import TrainingSupport
        from voicehub.training.specs import get_training_spec

        spec = create_neutts_architecture_spec()
        model_spec = get_model_spec("neutts")
        training_spec = get_training_spec("neutts")
        self.assertEqual(spec.architecture_id, "neutts")
        self.assertTrue(spec.capabilities.training)
        self.assertEqual(
            spec.capabilities.checkpoint_formats,
            ("safetensors", ),
        )
        self.assertEqual(
            spec.metadata["verified_training_family"],
            "neutts-air",
        )
        self.assertIn(
            "torch-compile-inference-unsafe",
            spec.capabilities.features,
        )
        self.assertIn(
            "Nano and 2E objectives fail closed",
            spec.metadata["training_boundary"],
        )
        self.assertEqual(
            spec.components["audio-codec"].path,
            "voicehub.architectures.neutts.neucodec:NeuCodecModel",
        )
        self.assertTrue(model_spec.is_voicehub_native)
        self.assertEqual(model_spec.architecture, "neutts")
        self.assertTrue(training_spec.native_training)
        self.assertIs(training_spec.support, TrainingSupport.NATIVE)
        self.assertEqual(
            training_spec.training_default_model_name_or_path,
            "neuphonic/neutts-air",
        )

    def test_artifact_boundary_rejects_unsafe_and_unpinned_formats(self):
        from voicehub.architectures.neutts.artifacts import resolve_neucodec_artifacts, resolve_neutts_artifacts

        for source in (
                "model.gguf",
                "model.onnx",
                "model.bin",
                "model.pt",
        ):
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "Safetensors"):
                    resolve_neutts_artifacts(source)
        with self.assertRaisesRegex(ValueError, "distilled NeuCodec"):
            resolve_neucodec_artifacts("neuphonic/neucodec-distill")
        with self.assertRaisesRegex(ValueError, "immutable"):
            resolve_neutts_artifacts("example/custom-neutts")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "weights.bin").write_bytes(b"unsafe")
            with self.assertRaisesRegex(ValueError, "Safetensors"):
                resolve_neutts_artifacts(
                    root,
                    checkpoint_filename="weights.bin",
                )

    def test_hugging_face_snapshot_symlinks_keep_logical_safetensors_identity(self):
        import torch

        from voicehub.architectures.causal_lm.checkpoint import open_causal_lm_tensor_source
        from voicehub.architectures.neutts.artifacts import resolve_neutts_artifacts
        from voicehub.checkpointing import SafeTensorReader, save_safetensors

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blobs = root / "blobs"
            snapshot = root / "snapshots" / "revision"
            blobs.mkdir()
            snapshot.mkdir(parents=True)
            blob = blobs / ("a" * 64)
            save_safetensors({"value": torch.zeros(1)}, blob)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").symlink_to(blob)

            artifacts = resolve_neutts_artifacts(snapshot)
            with open_causal_lm_tensor_source(snapshot / "model.safetensors", ) as reader:
                self.assertEqual(
                    artifacts.checkpoint,
                    (snapshot / "model.safetensors").absolute(),
                )
                self.assertIsInstance(reader, SafeTensorReader)

    def test_published_backbone_graphs_match_checkpoint_counts(self):
        import torch

        from voicehub.architectures.neutts.configuration import NeuTTSBackboneConfig
        from voicehub.architectures.neutts.metadata import NEUTTS_VARIANTS
        from voicehub.architectures.neutts.modeling import NeuTTSBackbone

        configurations = {
            "neuphonic/neutts-air": {
                "model_type": "qwen2",
                "vocab_size": 217_652,
                "hidden_size": 896,
                "intermediate_size": 4_864,
                "num_hidden_layers": 24,
                "num_attention_heads": 14,
                "num_key_value_heads": 2,
                "max_position_embeddings": 32_768,
                "rope_theta": 1_000_000.0,
                "bos_token_id": 151_643,
                "eos_token_id": 151_645,
                "tie_word_embeddings": True,
            },
            "neuphonic/neutts-nano": {
                "model_type": "llama",
                "vocab_size": 194_256,
                "hidden_size": 576,
                "intermediate_size": 2_304,
                "num_hidden_layers": 24,
                "num_attention_heads": 9,
                "num_key_value_heads": 3,
                "head_dim": 64,
                "max_position_embeddings": 2_048,
                "rms_norm_eps": 1e-5,
                "rope_theta": 500_000.0,
                "rope_scaling": {
                    "factor": 32.0,
                    "rope_type": "linear",
                    "type": "linear",
                },
                "bos_token_id": 128_000,
                "eos_token_id": 128_261,
                "pad_token_id": 128_001,
                "tie_word_embeddings": True,
            },
            "neuphonic/neutts-2e": {
                "model_type": "qwen3",
                "vocab_size": 217_232,
                "hidden_size": 512,
                "intermediate_size": 1_536,
                "num_hidden_layers": 28,
                "num_attention_heads": 12,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "max_position_embeddings": 32_768,
                "bos_token_id": None,
                "eos_token_id": 151_674,
                "pad_token_id": 151_645,
                "tie_word_embeddings": True,
                "neuphonic": {
                    "input_format": "BPE",
                    "supported_langs": ["en-us"],
                },
            },
        }
        expected_export = {
            "neuphonic/neutts-air": (291, 747_930_496),
            "neuphonic/neutts-nano": (218, 228_704_832),
            "neuphonic/neutts-2e": (310, 236_039_680),
        }
        for model_id, values in configurations.items():
            with self.subTest(model_id=model_id), torch.device("meta"):
                config = NeuTTSBackboneConfig.from_dict(values)
                model = NeuTTSBackbone(config, initialize=False)
                state = dict(model.state_dict())
                if model_id != "neuphonic/neutts-air":
                    state.pop("lm_head.weight")
            count, value_count = expected_export[model_id]
            self.assertEqual(len(state), count)
            self.assertEqual(
                sum(tensor.numel() for tensor in state.values()),
                value_count,
            )
            self.assertEqual(
                count,
                NEUTTS_VARIANTS[model_id]["tensor_count"],
            )
            self.assertEqual(
                value_count,
                NEUTTS_VARIANTS[model_id]["value_count"],
            )

    def test_neucodec_graph_matches_exact_safe_checkpoint_inventory(self):
        import torch

        from voicehub.architectures.neutts.configuration import NeuCodecConfig
        from voicehub.architectures.neutts.metadata import NEUCODEC_REFERENCE
        from voicehub.architectures.neutts.neucodec import NeuCodecModel

        with torch.device("meta"):
            model = NeuCodecModel(NeuCodecConfig(), initialize=False)
        state = model.state_dict()

        self.assertEqual(len(state), NEUCODEC_REFERENCE["tensor_count"])
        self.assertEqual(
            sum(tensor.numel() for tensor in state.values()),
            NEUCODEC_REFERENCE["value_count"],
        )
        self.assertEqual(
            tuple(state["acoustic_decoder.head.linear.weight"].shape),
            (1_922, 1_024),
        )
        self.assertEqual(model.input_sampling_rate, 16_000)
        self.assertEqual(model.output_sampling_rate, 24_000)
        self.assertEqual(model.config.encoder_hop_length, 320)
        self.assertEqual(model.config.hop_length, 480)
        self.assertEqual(model.config.n_fft, 1_920)

    def test_neucodec_frontend_uses_one_valid_zero_then_right_padding(self):
        import torch

        from voicehub.architectures.neutts.configuration import NeuCodecConfig
        from voicehub.architectures.neutts.neucodec import NeuCodecFeatureExtractor

        frontend = NeuCodecFeatureExtractor(NeuCodecConfig())
        observed = {}

        def fake_fbank(waveform, config):
            del config
            observed["shape"] = tuple(waveform.shape)
            return torch.arange(
                320,
                dtype=waveform.dtype,
                device=waveform.device,
            ).reshape(4, 80)

        with patch(
                "voicehub.architectures.neutts.neucodec.kaldi_fbank",
                side_effect=fake_fbank,
        ):
            features = frontend(torch.ones(640))

        self.assertEqual(observed["shape"], (1, 960))
        self.assertEqual(tuple(features.input_values.shape), (1, 1, 960))
        self.assertEqual(int(features.padding_mask.sum()), 641)
        self.assertEqual(
            features.input_values[0, 0, 640].item(),
            0.0,
        )
        self.assertEqual(tuple(features.input_features.shape), (1, 2, 160))

    def test_linear_rope_and_language_model_objective_are_native(self):
        import torch

        from voicehub.architectures.neutts.configuration import NeuTTSBackboneConfig
        from voicehub.architectures.neutts.modeling import LinearScalingRotaryEmbedding, NeuTTSBackbone

        config = NeuTTSBackboneConfig.from_dict(_tiny_backbone_values(linear_rope=True))
        model = NeuTTSBackbone(config)
        rotary = model.model.layers[0].self_attn.rotary
        self.assertIsInstance(rotary, LinearScalingRotaryEmbedding)
        self.assertEqual(rotary.factor, 2.0)

        tokens = torch.tensor([[1, 4, 5, 2]])
        output = model(tokens, labels=tokens)
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNotNone(model.model.embed_tokens.weight.grad)
        self.assertIsNotNone(model.lm_head.weight.grad)

    def test_rotary_frequencies_are_computed_like_the_published_cpu_load(self):
        import torch

        from voicehub.architectures.neutts.configuration import NeuTTSBackboneConfig
        from voicehub.architectures.neutts.modeling import NeuTTSBackbone

        values = dict(
            _tiny_backbone_values(),
            head_dim=128,
            num_attention_heads=1,
            hidden_size=128,
            rope_theta=10_000.0,
        )
        config = NeuTTSBackboneConfig.from_dict(values)
        # transformers builds `inv_freq` on the CPU before `.to(device)`.
        expected = 1.0 / (10_000.0**(torch.arange(0, 128, 2, dtype=torch.int64).float() / 128))
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model = NeuTTSBackbone(config, device=device)
                rotary = model.model.layers[0].self_attn.rotary
                self.assertEqual(rotary.inverse_frequency.device.type, device)
                self.assertTrue(torch.equal(rotary.inverse_frequency.cpu(), expected))
        if torch.cuda.is_available():
            on_device = 1.0 / (10_000.0**(torch.arange(0, 128, 2, device="cuda").float() / 128))
            # Documents why the CPU construction matters on CUDA hosts.
            self.assertFalse(torch.equal(on_device.cpu(), expected))

    def test_generation_projects_only_the_last_position(self):
        import torch

        from voicehub.architectures.neutts.configuration import NeuTTSBackboneConfig
        from voicehub.architectures.neutts.modeling import NeuTTSBackbone, NeuTTSRuntime

        torch.manual_seed(0)
        model = NeuTTSBackbone(NeuTTSBackboneConfig.from_dict(_tiny_backbone_values())).eval()
        tokens = torch.tensor([[1, 4, 5, 6, 7]])
        with torch.inference_mode():
            logits, cache = model.next_token_logits(tokens, cache=None, use_cache=True)
            full = model(tokens).logits
        self.assertEqual(tuple(logits.shape), (1, 1, 64))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.allclose(logits[:, -1], full[:, -1]))
        self.assertEqual(cache.sequence_length(), 5)

        projected_lengths = []
        model.lm_head.register_forward_hook(
            lambda module, inputs, output: projected_lengths.append(inputs[0].shape[1]))
        runtime = SimpleNamespace(
            backbone=model,
            tokenizer=SimpleNamespace(
                convert_tokens_to_ids=lambda token: 2,
                pad_token_id=0,
            ),
            max_context=12,
            min_new_tokens=3,
        )
        with torch.inference_mode():
            generated = NeuTTSRuntime._generate_tokens(
                runtime,
                tokens,
                max_new_tokens=4,
                temperature=1.0,
                top_k=5,
                seed=0,
            )
        self.assertGreaterEqual(generated.shape[1], 3)
        self.assertEqual(set(projected_lengths), {1})

    def test_text_and_phoneme_control_token_injection_is_rejected(self):
        from voicehub.architectures.neutts.modeling import NeuTTSRuntime
        from voicehub.architectures.neutts.tokenization import normalize_neutts_text

        with self.assertRaisesRegex(ValueError, "control tokens"):
            normalize_neutts_text("hello <|SPEECH_GENERATION_START|> world")
        with self.assertRaisesRegex(ValueError, "control-token"):
            NeuTTSRuntime._clean_phonemes(
                "h ə <|speech_1|>",
                name="phonemes",
            )

    def test_sft_dataset_masks_prompt_and_fails_closed_without_phonemes(self):
        from voicehub.architectures.neutts.modeling import NeuTTSRuntime
        from voicehub.models.neutts.training import NeuTTSSFTDataset

        runtime = SimpleNamespace(
            tokenizer=_DatasetTokenizer(),
            codec=None,
            input_format="phonemes",
            _resolve_phonemes=NeuTTSRuntime._resolve_phonemes,
        )
        dataset = NeuTTSSFTDataset(
            [{
                "text": "hello",
                "phonemes": "h ə l o",
                "codes": [3, 4]
            }],
            runtime=runtime,
        )
        example = dataset[0]
        self.assertEqual(
            example["input_ids"],
            [11, 20, 33, 34, 21],
        )
        self.assertEqual(
            example["labels"],
            [-100, 20, 33, 34, 21],
        )
        batch = dataset.collate_fn([example])
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 5))

        missing = NeuTTSSFTDataset(
            [{
                "text": "hello",
                "codes": [3]
            }],
            runtime=runtime,
        )
        with self.assertRaisesRegex(ValueError, "phonemes"):
            missing[0]

    def test_wrapper_enforces_reference_and_verified_training_boundaries(self):
        from voicehub.models.neutts.configuration_neutts import NeuTTSConfig
        from voicehub.models.neutts.inference import NeuTTSForTextToSpeech

        wrapper = NeuTTSForTextToSpeech(
            NeuTTSConfig(name_or_path="neuphonic/neutts-air"),
            lazy_load=True,
        )
        wrapper._validate_generation_inputs({
            "reference_codes": [1, 2],
            "reference_text": "reference",
        })
        with self.assertRaisesRegex(ValueError, "exactly one"):
            wrapper._validate_generation_inputs({
                "reference_codes": [1],
                "speaker_audio_path": "reference.wav",
                "reference_text": "reference",
            })
        wrapper._validate_training_runtime()

        for model_id in (
                "neuphonic/neutts-nano",
                "neuphonic/neutts-2e",
        ):
            candidate = NeuTTSForTextToSpeech(
                NeuTTSConfig(name_or_path=model_id),
                lazy_load=True,
            )
            with self.subTest(model_id=model_id):
                with self.assertRaisesRegex(ValueError, "NeuTTS-Air"):
                    candidate._validate_training_runtime()

        air_config = SimpleNamespace(
            causal_lm=SimpleNamespace(model_type="qwen2"),
            input_format="phonemes",
        )
        wrapper._validate_native_training_config(air_config)
        with self.assertRaisesRegex(ValueError, "local artifact"):
            wrapper._validate_native_training_config(
                SimpleNamespace(
                    causal_lm=SimpleNamespace(model_type="llama"),
                    input_format="phonemes",
                ))

    def test_cached_official_tokenizers_use_exact_native_split_engines(self):
        from voicehub.architectures.neutts.configuration import NeuTTSBackboneConfig
        from voicehub.architectures.neutts.metadata import NEUTTS_VARIANTS
        from voicehub.architectures.neutts.tokenization import SPEECH_CODEBOOK_SIZE, NeuTTSTokenizer

        cache = Path.home() / ".cache" / "huggingface" / "hub"
        checked = []
        for model_id in (
                "neuphonic/neutts-air",
                "neuphonic/neutts-nano",
                "neuphonic/neutts-2e",
        ):
            revision = NEUTTS_VARIANTS[model_id]["revision"]
            owner, name = model_id.split("/")
            snapshot = (cache / f"models--{owner}--{name}" / "snapshots" / revision)
            config_path = snapshot / "config.json"
            tokenizer_path = snapshot / "tokenizer.json"
            if not config_path.is_file() or not tokenizer_path.is_file():
                continue
            config = NeuTTSBackboneConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
            tokenizer = NeuTTSTokenizer.from_tokenizer_json(
                tokenizer_path,
                tokenizer_config_path=snapshot / "tokenizer_config.json",
                bos_token_id=config.causal_lm.bos_token_id,
                eos_token_id=(config.causal_lm.eos_token_ids[0] if config.causal_lm.eos_token_ids else None),
                pad_token_id=config.causal_lm.pad_token_id,
                expected_vocabulary_size=config.causal_lm.vocab_size,
            )
            self.assertLessEqual(
                tokenizer.token_id_space_size,
                config.causal_lm.vocab_size,
            )
            self.assertEqual(
                tokenizer.token_id_to_speech_code(
                    tokenizer.speech_code_to_token_id(SPEECH_CODEBOOK_SIZE - 1)),
                SPEECH_CODEBOOK_SIZE - 1,
            )
            checked.append(model_id)
        if not checked:
            self.skipTest("Official NeuTTS tokenizer snapshots are not cached.")


if __name__ == "__main__":
    unittest.main()
