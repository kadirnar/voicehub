from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from voicehub.checkpointing import save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.hub import write_json_file
from voicehub.models.higgstts.configuration import HiggsTTSConfig
from voicehub.models.higgstts.modeling import HiggsTTSForTextToSpeech
from voicehub.models.higgstts.native.checkpoint import (
    export_higgs_checkpoint,
    load_higgs_checkpoint,
    tensor_inventory_fingerprint,
    validate_higgs_checkpoint,
)
from voicehub.models.higgstts.native.configuration import HiggsAudioV2Config
from voicehub.models.higgstts.native.generation import HiggsAudioV2GenerationOutput, HiggsAudioV2Generator
from voicehub.models.higgstts.native.metadata import (
    HIGGS_AUDIO_V2_CHECKPOINT_HEADER_FINGERPRINT,
    HIGGS_AUDIO_V2_CHECKPOINT_PARAMETER_COUNT,
    HIGGS_AUDIO_V2_CHECKPOINT_TENSOR_COUNT,
    HIGGS_AUDIO_V2_CODEC_HEADER_FINGERPRINT,
    HIGGS_AUDIO_V2_CODEC_PARAMETER_COUNT,
    HIGGS_AUDIO_V2_CODEC_TENSOR_COUNT,
    HIGGS_AUDIO_V2_REPOSITORY,
    HIGGS_AUDIO_V2_REVISION,
    HIGGS_AUDIO_V2_SOURCE_REVISION,
    HIGGS_AUDIO_V2_TOKENIZER_REVISION,
)
from voicehub.models.higgstts.native.modeling import HiggsAudioV2ForConditionalGeneration
from voicehub.models.higgstts.native.processing import (
    HIGGS_SPECIAL_TOKEN_IDS,
    HiggsAudioV2Processor,
    HiggsAudioV2TextTokenizer,
)
from voicehub.models.higgstts.native.registration import register_higgs_audio_v2_architecture
from voicehub.models.higgstts.native.runtime import HiggsAudioV2Runtime, load_higgs_audio_v2_runtime
from voicehub.models.higgstts.native.tokenizer import HiggsAudioV2TokenizerModel
from voicehub.models.higgstts.native.tokenizer_configuration import HiggsAudioV2TokenizerConfig
from voicehub.models.higgstts.training import HiggsTrainingAdapter
from voicehub.runtime.registry import ArchitectureRegistry
from voicehub.tokenization.assets import encode_gpt2_token
from voicehub.training.contracts import TrainingPhaseSpec, TrainingSupport
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_model_config() -> HiggsAudioV2Config:
    return replace(
        HiggsAudioV2Config(),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        num_codebooks=2,
        codebook_size=18,
        audio_stream_bos_id=16,
        audio_stream_eos_id=17,
        rope_parameters={
            "factor": 2.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 128,
            "rope_theta": 10_000.0,
            "rope_type": "llama3",
        },
    )


def _write_tokenizer(path: Path) -> HiggsAudioV2TextTokenizer:
    vocabulary = {encode_gpt2_token(bytes((value, ))): value for value in range(256)}
    added_tokens = [{
        "content": spelling,
        "id": token_id,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    } for spelling, token_id in HIGGS_SPECIAL_TOKEN_IDS.items()]
    write_json_file(
        path,
        {
            "added_tokens": added_tokens,
            "model": {
                "merges": [],
                "type": "BPE",
                "unk_token": None,
                "vocab": vocabulary,
            },
            "normalizer": None,
            "pre_tokenizer": {
                "add_prefix_space": False,
                "trim_offsets": True,
                "type": "ByteLevel",
                "use_regex": True,
            },
            "version": "1.0",
        },
    )
    return HiggsAudioV2TextTokenizer.from_files(path)


def _small_runtime(directory: Path) -> HiggsAudioV2Runtime:
    model_config = _small_model_config()
    codec = HiggsAudioV2TokenizerModel(HiggsAudioV2TokenizerConfig.tiny())
    processor = HiggsAudioV2Processor(
        _write_tokenizer(directory / "tokenizer.json"),
        codec,
        model_config,
    )
    return HiggsAudioV2Runtime(
        HiggsAudioV2ForConditionalGeneration(model_config),
        processor,
    )


def _training_spec() -> ModelTrainingSpec:
    phase = TrainingPhaseSpec(
        name="codec_language_model",
        component_paths=("model", ),
        optimizer_names=("model", ),
        label_names=("labels", "audio_labels"),
        prediction_keys=("logits", "text_logits"),
        loss_keys=("loss", "text_loss", "audio_loss"),
        required_inputs=(
            "input_ids",
            "attention_mask",
            "audio_input_ids",
            "audio_input_ids_mask",
            "labels",
            "audio_labels",
        ),
    )
    return ModelTrainingSpec(
        model_type="higgstts",
        family=TrainingFamily.CAUSAL_LM,
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=(
            "voicehub.models.higgstts.native.modeling:"
            "HiggsAudioV2ForConditionalGeneration.forward", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(phase, ),
        default_phase=phase.name,
    )


class NativeHiggsDependencyTests(unittest.TestCase):

    def test_native_files_do_not_import_external_model_runtimes(self):
        roots = (
            PROJECT_ROOT / 'voicehub/models/higgstts/native',
            PROJECT_ROOT / "voicehub" / "models" / "higgstts",
        )
        active_provider_files = {
            "__init__.py",
            "configuration.py",
            "inference.py",
            "modeling.py",
            "training.py",
        }
        forbidden = {
            "accelerate",
            "einops",
            "huggingface_hub",
            "librosa",
            "numpy",
            "safetensors",
            "tokenizers",
            "torchaudio",
            "transformers",
        }
        violations = []
        for root in roots:
            paths = root.glob("*.py")
            if root.name == "higgstts":
                paths = (path for path in paths if path.name in active_provider_files)
            for path in paths:
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module):
                        modules = [node.module]
                    else:
                        modules = []
                    violations.extend(
                        (path.name, module) for module in modules if module.split(".", 1)[0] in forbidden)
        self.assertEqual(violations, [])

    def test_public_package_is_lazy_without_framework_clients(self):
        command = (
            "import sys; import voicehub.models.higgstts; "
            "print('torch' in sys.modules, 'transformers' in sys.modules, "
            "'huggingface_hub' in sys.modules, 'safetensors' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False False False False")

    def test_provenance_pins_code_weights_codec_and_training_boundary(self):
        source = json.loads(
            (PROJECT_ROOT / 'voicehub/models/higgstts/native/SOURCE.json').read_text(encoding="utf-8"))
        self.assertEqual(
            source["source"]["revision"],
            HIGGS_AUDIO_V2_SOURCE_REVISION,
        )
        self.assertEqual(
            source["checkpoint"]["revision"],
            HIGGS_AUDIO_V2_REVISION,
        )
        self.assertEqual(
            source["audio_tokenizer"]["revision"],
            HIGGS_AUDIO_V2_TOKENIZER_REVISION,
        )
        self.assertTrue(source["training"]["full_sft"])
        self.assertFalse(source["training"]["published_recipe"])
        self.assertIn("Community License", source["checkpoint"]["license"])
        self.assertTrue((PROJECT_ROOT / 'voicehub/models/higgstts/native/THIRD_PARTY_LICENSE').is_file())

    def test_architecture_registration_is_lazy_and_truthful(self):
        registry = ArchitectureRegistry()
        spec = register_higgs_audio_v2_architecture(registry=registry)

        self.assertIs(registry.get("native-higgstts"), spec)
        self.assertEqual(spec.architecture_id, "higgs-audio-v2")
        self.assertTrue(spec.capabilities.training)
        self.assertFalse(spec.capabilities.streaming)
        self.assertEqual(
            spec.capabilities.checkpoint_formats,
            ("safetensors", ),
        )
        self.assertEqual(
            spec.metadata["checkpoint_license"],
            "Boson-Higgs-Audio-2-Community-License",
        )


class NativeHiggsInventoryTests(unittest.TestCase):

    def test_official_model_graph_matches_audited_header(self):
        with torch.device("meta"):
            model = HiggsAudioV2ForConditionalGeneration(
                HiggsAudioV2Config(),
                initialize=False,
                dtype=torch.bfloat16,
            )
        state = model.state_dict()
        inventory = {name: ("BF16", tuple(value.shape)) for name, value in state.items()}

        self.assertEqual(len(state), HIGGS_AUDIO_V2_CHECKPOINT_TENSOR_COUNT)
        self.assertEqual(
            sum(value.numel() for value in state.values()),
            HIGGS_AUDIO_V2_CHECKPOINT_PARAMETER_COUNT,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(inventory),
            HIGGS_AUDIO_V2_CHECKPOINT_HEADER_FINGERPRINT,
        )

    def test_official_codec_graph_matches_audited_header(self):
        with torch.device("meta"):
            codec = HiggsAudioV2TokenizerModel(
                HiggsAudioV2TokenizerConfig(),
                initialize=False,
            )
        remapped = {
            "semantic_model.encoder.pos_conv_embed.conv.weight_g":
            ("semantic_model.encoder.pos_conv_embed.conv."
             "parametrizations.weight.original0"),
            "semantic_model.encoder.pos_conv_embed.conv.weight_v":
            ("semantic_model.encoder.pos_conv_embed.conv."
             "parametrizations.weight.original1"),
        }
        state = codec.state_dict()
        inventory = {remapped.get(name, name): ("F32", tuple(value.shape)) for name, value in state.items()}

        self.assertEqual(len(state), HIGGS_AUDIO_V2_CODEC_TENSOR_COUNT)
        self.assertEqual(
            sum(value.numel() for value in state.values()),
            HIGGS_AUDIO_V2_CODEC_PARAMETER_COUNT,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(inventory),
            HIGGS_AUDIO_V2_CODEC_HEADER_FINGERPRINT,
        )

    def test_tiny_checkpoint_roundtrip_and_mismatch_rejection(self):
        model = HiggsAudioV2ForConditionalGeneration(HiggsAudioV2Config.tiny())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = export_higgs_checkpoint(
                model,
                root / "model.safetensors",
            )
            report = validate_higgs_checkpoint(model, checkpoint)
            with torch.device("meta"):
                restored = HiggsAudioV2ForConditionalGeneration(
                    HiggsAudioV2Config.tiny(),
                    initialize=False,
                )
            load_higgs_checkpoint(
                restored,
                checkpoint,
                device="cpu",
                dtype=torch.float32,
            )
            first_name = next(iter(model.state_dict()))
            self.assertTrue(torch.equal(
                model.state_dict()[first_name],
                restored.state_dict()[first_name],
            ))
            self.assertEqual(report.tensor_count, len(model.state_dict()))

            bad = save_safetensors(
                {"unexpected.weight": torch.ones(1)},
                root / "bad.safetensors",
            )
            with self.assertRaises(CheckpointCompatibilityError):
                validate_higgs_checkpoint(model, bad)


class NativeHiggsProcessingAndTrainingTests(unittest.TestCase):

    def test_source_label_masking_and_delay_pattern_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = _small_runtime(Path(directory))
            target = torch.tensor([[1, 2, 3], [4, 5, 6]])
            reference = torch.tensor([[7, 8], [9, 10]])
            batch = runtime.prepare_training_inputs([{
                "audio_codes": target,
                "reference_codes": reference,
                "reference_text": "reference",
                "text": "target",
            }])

        self.assertEqual(
            batch.labels[batch.labels != -100].tolist(),
            [128_013, 128_012, 128_009],
        )
        self.assertEqual(
            batch.audio_labels[0, :, 0][batch.audio_labels[0, :, 0] != -100].tolist(),
            [1, 2, 3, 17],
        )
        self.assertEqual(
            batch.audio_labels[0, :, 1][batch.audio_labels[0, :, 1] != -100].tolist(),
            [4, 5, 6, 17],
        )
        self.assertTrue(
            (batch.audio_labels[
                0,
                :batch.audio_input_ids.shape[1] - (target.shape[1] + 3),
            ] == -100).all())

    def test_full_sft_uses_native_losses_and_keeps_codec_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = _small_runtime(Path(directory))
            wrapper = HiggsTTSForTextToSpeech(
                HiggsTTSConfig(
                    training_audio_loss_weight=3.0,
                    training_text_loss_weight=2.0,
                ),
                device="cpu",
            )
            wrapper._runtime = runtime
            adapter = HiggsTrainingAdapter(wrapper, _training_spec())
            output = adapter(
                records=[{
                    "audio_codes": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                    "text": "target",
                }])

        expected = (output.losses["text_loss"] * 2.0 + output.losses["audio_loss"] * 3.0)
        self.assertTrue(torch.allclose(output.loss, expected))
        output.loss.backward()
        self.assertTrue(
            all(not parameter.requires_grad for parameter in runtime.audio_tokenizer.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in runtime.model.parameters()))
        self.assertEqual(
            output.metadata["objective"],
            "source-joint-text-plus-delayed-codebook-ce",
        )

    def test_native_runtime_export_reloads_without_external_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = _small_runtime(root)
            destination = runtime.save_pretrained(root / "export")
            restored = load_higgs_audio_v2_runtime(
                destination,
                device="cpu",
                dtype="float32",
            )

        self.assertEqual(restored.model.config, runtime.model.config)
        self.assertEqual(
            restored.audio_tokenizer.config,
            runtime.audio_tokenizer.config,
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in restored.audio_tokenizer.parameters()))

    def test_generation_constraints_force_only_selected_codebooks(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = _small_runtime(Path(directory))
            generator = HiggsAudioV2Generator(
                runtime.model,
                runtime.processor,
            )
            logits = torch.zeros(1, 2, 18)
            forced = generator._force_token(
                logits,
                torch.tensor([[True, False]]),
                16,
            )

        self.assertEqual(forced[0, 0].argmax().item(), 16)
        self.assertTrue(torch.isneginf(forced[0, 0, :16]).all())
        self.assertTrue(torch.isneginf(forced[0, 0, 17:]).all())
        self.assertTrue(torch.equal(forced[0, 1], logits[0, 1]))

    def test_public_wrapper_routes_native_generation_options(self):
        response = HiggsAudioV2GenerationOutput(
            waveform=torch.tensor([[[0.25, -0.25]]]),
            audio_codes=torch.ones(1, 2, 1, dtype=torch.long),
            delayed_audio_codes=torch.ones(1, 2, 2, dtype=torch.long),
            text_sequence=torch.ones(1, 2, dtype=torch.long),
            sample_rate=24_000,
            generated_steps=2,
        )
        runtime = SimpleNamespace(generate=Mock(return_value=response))
        wrapper = HiggsTTSForTextToSpeech(device="cpu")
        wrapper._runtime = runtime

        output = wrapper._generate(
            "hello",
            seed=41,
            temperature=0.0,
            top_k=None,
            ras_win_len=None,
        )

        runtime.generate.assert_called_once()
        options = runtime.generate.call_args.kwargs
        self.assertEqual(options["seed"], 41)
        self.assertEqual(options["temperature"], 0.0)
        self.assertIsNone(options["top_k"])
        self.assertIsNone(options["ras_window"])
        self.assertEqual(output.audio.tolist(), [0.25, -0.25])
        self.assertEqual(output.metadata["backend"], "voicehub-native")
        self.assertEqual(output.metadata["seed"], 41)

    def test_public_config_rejects_remote_code_and_pickle_checkpoints(self):
        self.assertEqual(
            HiggsTTSForTextToSpeech.default_model_name_or_path,
            HIGGS_AUDIO_V2_REPOSITORY,
        )
        with self.assertRaisesRegex(ValueError, "never executes"):
            HiggsTTSConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "Safetensors"):
            HiggsTTSConfig(use_safetensors=False)


if __name__ == "__main__":
    unittest.main()
