from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from voicehub.checkpointing import save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.voxcpm.configuration import VoxCPMConfig
from voicehub.models.voxcpm.modeling import VoxCPMForTextToSpeech
from voicehub.models.voxcpm.native.checkpoint import (
    export_voxcpm_checkpoint,
    load_voxcpm_checkpoint,
    tensor_inventory_fingerprint,
    validate_voxcpm_checkpoint,
)
from voicehub.models.voxcpm.native.codec import VoxCPMAudioVAE
from voicehub.models.voxcpm.native.configuration import VoxCPM2ArchitectureConfig
from voicehub.models.voxcpm.native.lora import (
    VoxCPMLoRAConfig,
    export_voxcpm_lora,
    inject_voxcpm_lora,
    load_voxcpm_lora,
    merged_voxcpm_state_dict,
    read_voxcpm_lora_config,
)
from voicehub.models.voxcpm.native.metadata import (
    VOXCPM2_CHECKPOINT_HEADER_FINGERPRINT,
    VOXCPM2_CHECKPOINT_PARAMETER_COUNT,
    VOXCPM2_CHECKPOINT_REVISION,
    VOXCPM2_CHECKPOINT_TENSOR_COUNT,
    VOXCPM2_CODEC_HEADER_FINGERPRINT,
    VOXCPM2_CODEC_PARAMETER_COUNT,
    VOXCPM2_CODEC_TENSOR_COUNT,
    VOXCPM2_SOURCE_REVISION,
)
from voicehub.models.voxcpm.native.modeling import VoxCPM2Model
from voicehub.models.voxcpm.native.processing import VoxCPM2Processor, VoxCPM2Tokenizer
from voicehub.models.voxcpm.native.registration import register_voxcpm2_architecture
from voicehub.models.voxcpm.native.runtime import VoxCPM2Runtime
from voicehub.models.voxcpm.training_voxcpm import VoxCPMTrainingAdapter
from voicehub.registry import get_model_spec
from voicehub.runtime.registry import ArchitectureRegistry
from voicehub.tokenization import SentencePieceBPEAssets, SentencePieceBPETokenizer
from voicehub.training import AutoTrainingAdapter
from voicehub.training.contracts import TrainingPhaseSpec, TrainingSupport
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily, get_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_config() -> VoxCPM2ArchitectureConfig:
    config = VoxCPM2ArchitectureConfig.tiny()
    return replace(
        config,
        audio_vae_config=replace(
            config.audio_vae_config,
            out_sample_rate=48_000,
        ),
    )


def _tiny_tokenizer(path: Path) -> VoxCPM2Tokenizer:
    path.write_text("{}\n", encoding="utf-8")
    assets = SentencePieceBPEAssets(
        vocabulary={
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
            "\u2581": 3,
            "a": 105,
        },
        merges=(),
        special_tokens={},
        added_tokens={},
        unk_token_id=0,
        prefix_token_ids=(1, ),
        prepend=" ",
        replacement_source=" ",
        replacement_target="\u2581",
        byte_fallback=False,
        fuse_unk=False,
        original_document={},
    )
    tokenizer = SentencePieceBPETokenizer(
        assets,
        pad_token_id=2,
        bos_token_id=1,
        eos_token_id=2,
    )
    return VoxCPM2Tokenizer(
        tokenizer,
        split_map={},
        source_path=path,
    )


def _tiny_runtime(directory: Path) -> VoxCPM2Runtime:
    config = _tiny_config()
    model = VoxCPM2Model(config)
    codec = VoxCPMAudioVAE(config.audio_vae_config)
    processor = VoxCPM2Processor(
        _tiny_tokenizer(directory / "tokenizer.json"),
        config,
        codec=codec,
    )
    return VoxCPM2Runtime(model, processor, codec)


def _training_spec() -> ModelTrainingSpec:
    phase = TrainingPhaseSpec(
        name="source_flow_and_stop",
        component_paths=("model", ),
        optimizer_names=("model", ),
        prediction_keys=("target_features", ),
        loss_keys=("diffusion_loss", "stop_loss"),
        required_inputs=(
            "text_tokens",
            "text_mask",
            "audio_feats",
            "audio_mask",
            "loss_mask",
            "position_ids",
            "labels",
        ),
    )
    return ModelTrainingSpec(
        model_type="voxcpm",
        family=TrainingFamily.FLOW_MATCHING,
        module_paths=("model", ),
        component_paths=("model", ),
        prediction_keys=("target_features", ),
        loss_keys=("diffusion_loss", "stop_loss"),
        source_entrypoints=("voicehub.models.voxcpm.native.modeling:VoxCPM2Model.forward", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(phase, ),
        default_phase=phase.name,
    )


class NativeVoxCPMDependencyTests(unittest.TestCase):

    def test_native_files_do_not_import_external_model_runtimes(self):
        roots = (
            PROJECT_ROOT / 'voicehub/models/voxcpm/native',
            PROJECT_ROOT / 'voicehub/models/voxcpm',
        )
        forbidden = {
            "accelerate",
            "einops",
            "huggingface_hub",
            "numpy",
            "peft",
            "safetensors",
            "sentencepiece",
            "tokenizers",
            "torchaudio",
            "transformers",
            "voxcpm",
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
                    violations.extend(
                        (path.name, name) for name in names if name.split(".", 1)[0] in forbidden)
        self.assertEqual(violations, [])

    def test_public_package_is_lazy_without_torch_or_framework_clients(self):
        command = (
            "import sys; import voicehub.models.voxcpm; "
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

    def test_provenance_pins_source_checkpoint_and_training_boundary(self):
        document = json.loads(
            (PROJECT_ROOT / 'voicehub/models/voxcpm/native/SOURCE.json').read_text(encoding="utf-8"))
        self.assertEqual(document["source"]["revision"], VOXCPM2_SOURCE_REVISION)
        self.assertEqual(
            document["checkpoint"]["revision"],
            VOXCPM2_CHECKPOINT_REVISION,
        )
        self.assertTrue(document["training"]["full_sft"])
        self.assertTrue(document["training"]["lora"])
        self.assertEqual(
            document["training"]["codec_policy"],
            "AudioVAE V2 is frozen and used only for preprocessing/validation decode.",
        )
        self.assertTrue((PROJECT_ROOT / 'voicehub/models/voxcpm/native/THIRD_PARTY_LICENSE').is_file())

    def test_architecture_registration_is_lazy_and_truthful(self):
        registry = ArchitectureRegistry()
        spec = register_voxcpm2_architecture(registry=registry)

        self.assertIs(registry.get("native-voxcpm2"), spec)
        self.assertTrue(spec.capabilities.training)
        self.assertFalse(spec.capabilities.streaming)
        self.assertEqual(spec.capabilities.checkpoint_formats, ("safetensors", ))
        self.assertEqual(spec.metadata["implementation"], "voicehub-native")


class NativeVoxCPMInventoryTests(unittest.TestCase):

    def test_official_model_graph_matches_audited_safetensors_header(self):
        config = VoxCPM2ArchitectureConfig()
        with torch.device("meta"):
            model = VoxCPM2Model(config, dtype=torch.bfloat16)
        state = model.state_dict()
        inventory = {name: ("BF16", tuple(value.shape)) for name, value in state.items()}

        self.assertEqual(len(state), VOXCPM2_CHECKPOINT_TENSOR_COUNT)
        self.assertEqual(
            sum(value.numel() for value in state.values()),
            VOXCPM2_CHECKPOINT_PARAMETER_COUNT,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(inventory),
            VOXCPM2_CHECKPOINT_HEADER_FINGERPRINT,
        )

    def test_official_codec_graph_matches_audited_archive_inventory(self):
        with torch.device("meta"):
            codec = VoxCPMAudioVAE(
                VoxCPM2ArchitectureConfig().audio_vae_config,
                dtype=torch.float32,
            )
        state = codec.state_dict()
        names = {
            torch.float32: "F32",
            torch.int32: "I32",
        }
        inventory = {name: (names[value.dtype], tuple(value.shape)) for name, value in state.items()}

        self.assertEqual(len(state), VOXCPM2_CODEC_TENSOR_COUNT)
        self.assertEqual(
            sum(value.numel() for value in state.values()),
            VOXCPM2_CODEC_PARAMETER_COUNT,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(inventory),
            VOXCPM2_CODEC_HEADER_FINGERPRINT,
        )


class NativeVoxCPMTrainingTests(unittest.TestCase):

    def test_full_sft_runs_published_losses_and_keeps_codec_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = _tiny_runtime(Path(directory))
            wrapper = VoxCPMForTextToSpeech(
                VoxCPMConfig(
                    training_diffusion_loss_weight=2.0,
                    training_stop_loss_weight=3.0,
                ),
                device="cpu",
            )
            wrapper._runtime = runtime
            adapter = VoxCPMTrainingAdapter(wrapper, _training_spec())
            output = adapter(
                records=[{
                    "text": "a",
                    "audio_features": torch.randn(2, 2, 8),
                }], )

        expected = (output.losses["diffusion_loss"] * 2.0 + output.losses["stop_loss"] * 3.0)
        self.assertTrue(torch.allclose(output.loss, expected))
        self.assertEqual(output.metadata["objective"], "source-cfm-plus-stop-ce")
        output.loss.backward()
        self.assertTrue(all(not parameter.requires_grad for parameter in runtime.codec.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in runtime.model.parameters()))

    def test_lora_trains_only_published_targets_and_exports_merged_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = _tiny_runtime(root)
            wrapper = VoxCPMForTextToSpeech(
                VoxCPMConfig(training_lora_config={
                    "rank": 2,
                    "alpha": 4.0
                }),
                device="cpu",
            )
            wrapper._runtime = runtime
            adapter = VoxCPMTrainingAdapter(wrapper, _training_spec())
            output = adapter(
                records=[{
                    "text": "a",
                    "audio_features": torch.randn(2, 2, 8),
                }], )
            output.loss.backward()
            trainable = {
                name
                for name, parameter in runtime.model.named_parameters() if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(all(name.endswith((".lora_A", ".lora_B")) for name in trainable))
            self.assertTrue(
                any(
                    parameter.grad is not None for name, parameter in runtime.model.named_parameters()
                    if name.endswith(".lora_B")))

            export_directory = root / "export"
            adapter.save_pretrained(export_directory)
            self.assertTrue((export_directory / "model.safetensors").is_file())
            self.assertTrue((export_directory / "lora_adapter" / "lora_weights.safetensors").is_file())
            with torch.device("meta"):
                reloaded = VoxCPM2Model(_tiny_config())
            load_voxcpm_checkpoint(
                reloaded,
                export_directory / "model.safetensors",
                device="cpu",
            )
            self.assertFalse(any("lora_" in name for name in reloaded.state_dict()))

    def test_lora_adapter_roundtrip_and_merge_preserve_standard_namespace(self):
        config = VoxCPM2ArchitectureConfig.tiny()
        model = VoxCPM2Model(config)
        lora = VoxCPMLoRAConfig(rank=2, alpha=4.0)
        inject_voxcpm_lora(model, lora)
        for name, parameter in model.named_parameters():
            if name.endswith(".lora_B"):
                parameter.data.fill_(0.125)
        merged = merged_voxcpm_state_dict(model)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = export_voxcpm_checkpoint(
                model,
                root / "model.safetensors",
                state_override=merged,
            )
            adapter_directory = export_voxcpm_lora(model, root / "adapter", lora)
            self.assertEqual(read_voxcpm_lora_config(adapter_directory), lora)
            with torch.device("meta"):
                fresh = VoxCPM2Model(config)
            load_voxcpm_checkpoint(fresh, checkpoint, device="cpu")

            adapter_target = VoxCPM2Model(config)
            inject_voxcpm_lora(adapter_target, lora)
            load_voxcpm_lora(adapter_target, adapter_directory, lora)
            for name, value in model.state_dict().items():
                if name.endswith((".lora_A", ".lora_B")):
                    self.assertTrue(torch.equal(value, adapter_target.state_dict()[name]))

        self.assertEqual(set(merged), set(fresh.state_dict()))

    def test_strict_checkpoint_validation_rejects_partial_state(self):
        config = VoxCPM2ArchitectureConfig.tiny()
        model = VoxCPM2Model(config)
        with tempfile.TemporaryDirectory() as directory:
            path = save_safetensors(
                {"base_lm.embed_tokens.weight": model.base_lm.embed_tokens.weight},
                Path(directory) / "partial.safetensors",
            )
            blob = path.with_name("checkpoint-blob")
            path.rename(blob)
            path.symlink_to(blob.name)
            with self.assertRaises(CheckpointCompatibilityError):
                validate_voxcpm_checkpoint(model, path)


class NativeVoxCPMProviderTests(unittest.TestCase):

    def test_public_registry_selects_native_architecture_and_adapter(self):
        model_spec = get_model_spec("voxcpm")
        training_spec = get_training_spec("voxcpm")
        wrapper = VoxCPMForTextToSpeech(VoxCPMConfig(), device="cpu")
        adapter = AutoTrainingAdapter.from_model(wrapper)

        self.assertTrue(model_spec.is_voicehub_native)
        self.assertEqual(model_spec.architecture, "voxcpm2")
        self.assertNotIn("streaming", model_spec.capabilities)
        self.assertEqual(training_spec.default_phase, "source_flow_and_stop")
        self.assertEqual(
            training_spec.source_entrypoints,
            ("voicehub.models.voxcpm.native.modeling:"
             "VoxCPM2Model.forward", ),
        )
        self.assertIsInstance(adapter, VoxCPMTrainingAdapter)

    def test_config_is_safe_and_rejects_legacy_or_remote_code_modes(self):
        config = VoxCPMConfig()
        self.assertEqual(config.sample_rate, 48_000)
        self.assertEqual(config.torch_dtype, "bfloat16")
        self.assertNotIn("token", config.to_dict())
        with self.assertRaisesRegex(ValueError, "never executes repository code"):
            VoxCPMConfig(trust_remote_code=True)
        with self.assertRaises(PermissionError):
            VoxCPMConfig(codec_path="audiovae.pth")
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            VoxCPMConfig(token="secret")

    def test_generation_uses_native_runtime_and_reports_conditioning_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = _tiny_runtime(Path(directory))
            wrapper = VoxCPMForTextToSpeech(
                VoxCPMConfig(),
                device="cpu",
            )
            wrapper._runtime = runtime
            with patch.object(
                    runtime,
                    "generate",
                    return_value=torch.zeros(1, 32),
            ) as generate:
                output = wrapper.generate("a", seed=7)

        self.assertEqual(output.sample_rate, 48_000)
        self.assertEqual(output.audio.shape, (32, ))
        self.assertEqual(output.metadata["backend"], "voicehub-native")
        self.assertEqual(generate.call_args.kwargs["seed"], 7)

    def test_external_postprocessing_options_fail_closed(self):
        wrapper = VoxCPMForTextToSpeech(
            VoxCPMConfig(),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "postprocessing"):
            wrapper.generate("a", denoise=True)


if __name__ == "__main__":
    unittest.main()
