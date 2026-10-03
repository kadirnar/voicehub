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

from voicehub.architectures.registry import ArchitectureRegistry
from voicehub.architectures.voxcpm2.checkpoint import (
    export_voxcpm_checkpoint,
    load_voxcpm_checkpoint,
    tensor_inventory_fingerprint,
    validate_voxcpm_checkpoint,
)
from voicehub.architectures.voxcpm2.codec import VoxCPMAudioVAE
from voicehub.architectures.voxcpm2.configuration import VoxCPM2ArchitectureConfig
from voicehub.architectures.voxcpm2.lora import (
    VoxCPMLoRAConfig,
    export_voxcpm_lora,
    inject_voxcpm_lora,
    load_voxcpm_lora,
    merged_voxcpm_state_dict,
    read_voxcpm_lora_config,
)
from voicehub.architectures.voxcpm2.metadata import (
    VOXCPM2_CHECKPOINT_HEADER_FINGERPRINT,
    VOXCPM2_CHECKPOINT_PARAMETER_COUNT,
    VOXCPM2_CHECKPOINT_REVISION,
    VOXCPM2_CHECKPOINT_TENSOR_COUNT,
    VOXCPM2_CODEC_HEADER_FINGERPRINT,
    VOXCPM2_CODEC_PARAMETER_COUNT,
    VOXCPM2_CODEC_TENSOR_COUNT,
    VOXCPM2_SOURCE_REVISION,
)
from voicehub.architectures.voxcpm2.modeling import VoxCPM2Model
from voicehub.architectures.voxcpm2.processing import VoxCPM2Processor, VoxCPM2Tokenizer
from voicehub.architectures.voxcpm2.registration import register_voxcpm2_architecture
from voicehub.architectures.voxcpm2.runtime import VoxCPM2Runtime
from voicehub.checkpointing import save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.voxcpm_native.configuration_voxcpm import VoxCPMConfig
from voicehub.models.voxcpm_native.modeling_voxcpm import VoxCPMForTextToSpeech
from voicehub.models.voxcpm_native.training_voxcpm import VoxCPMTrainingAdapter
from voicehub.registry import get_model_spec
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
        source_entrypoints=("voicehub.architectures.voxcpm2.modeling:VoxCPM2Model.forward", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(phase, ),
        default_phase=phase.name,
    )


class NativeVoxCPMDependencyTests(unittest.TestCase):

    def test_native_files_do_not_import_external_model_runtimes(self):
        roots = (
            PROJECT_ROOT / "voicehub" / "architectures" / "voxcpm2",
            PROJECT_ROOT / "voicehub" / "models" / "voxcpm_native",
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
            "import sys; import voicehub.models.voxcpm_native; "
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
        document = json.loads((PROJECT_ROOT / "voicehub" / "architectures" / "voxcpm2" /
                               "SOURCE.json").read_text(encoding="utf-8"))
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
        self.assertTrue(
            (PROJECT_ROOT / "voicehub" / "architectures" / "voxcpm2" / "THIRD_PARTY_LICENSE").is_file())

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


def _non_unit_rope_config() -> VoxCPM2ArchitectureConfig:
    config = _tiny_config()
    factors = (0.9977997200264581, 1.5, 4.0, 31.0)
    rope = replace(config.lm_config.rope_scaling, long_factor=factors, short_factor=factors)
    return replace(config, lm_config=replace(config.lm_config, rope_scaling=rope))


def _source_rope_tables(config, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream MiniCPMLongRoPE tables after the source's `model.to(dtype)`."""
    dimension = config.lm_config.head_dim
    inverse = 1.0 / (config.lm_config.rope_theta**(torch.arange(0, dimension, 2).float() / dimension))
    positions = torch.arange(config.lm_config.max_position_embeddings, dtype=torch.float32)
    factors = torch.tensor(config.lm_config.rope_scaling.short_factor, dtype=torch.float32)
    embedding = torch.outer(positions, 1.0 / factors) * inverse
    embedding = torch.cat((embedding, embedding), dim=-1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


class NativeVoxCPMSourceNumericsTests(unittest.TestCase):

    def test_local_transformers_inherit_backbone_longrope_factors(self):
        # Upstream builds the local encoder and DiT from
        # `lm_config.model_copy()`, so they keep the backbone LongRoPE factors.
        config = _non_unit_rope_config()
        model = VoxCPM2Model(config)
        cosine, sine = _source_rope_tables(config, torch.float32)
        for decoder in (
                model.base_lm,
                model.feat_encoder.encoder,
                model.feat_decoder.estimator.decoder,
        ):
            self.assertEqual(
                tuple(decoder.rope_emb.short_factor),
                config.lm_config.rope_scaling.short_factor,
            )
            torch.testing.assert_close(decoder.rope_emb.cos_cached, cosine, rtol=0, atol=0)
            torch.testing.assert_close(decoder.rope_emb.sin_cached, sine, rtol=0, atol=0)
        self.assertIsNone(model.residual_lm.rope_emb)

    def test_rope_tables_follow_low_precision_parameter_dtype(self):
        # Upstream casts the whole model (including the non-persistent RoPE
        # tables) with `model.to(bfloat16)` before inference.
        config = _non_unit_rope_config()
        cosine, sine = _source_rope_tables(config, torch.bfloat16)
        direct = VoxCPM2Model(config, dtype=torch.bfloat16)
        with torch.device("meta"):
            streamed = VoxCPM2Model(config, dtype=torch.bfloat16)
        streamed.to_empty(device="cpu")
        streamed.materialize_runtime_buffers("cpu")
        for model in (direct, streamed):
            for decoder in (
                    model.base_lm,
                    model.feat_encoder.encoder,
                    model.feat_decoder.estimator.decoder,
            ):
                self.assertEqual(decoder.rope_emb.cos_cached.dtype, torch.bfloat16)
                torch.testing.assert_close(decoder.rope_emb.cos_cached, cosine, rtol=0, atol=0)
                torch.testing.assert_close(decoder.rope_emb.sin_cached, sine, rtol=0, atol=0)


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
            ("voicehub.architectures.voxcpm2.modeling:"
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

    def test_continuation_prompt_is_left_padded_and_decoded_as_codec_context(self):
        # Source: prompt audio uses padding_mode="left", and the non-streaming
        # decode prepends the last (streaming_prefix_len - 1) = 3 prefix audio
        # patches, then trims their samples from the waveform.
        torch.manual_seed(0)
        waveform = torch.randn(29)
        with tempfile.TemporaryDirectory() as directory:
            runtime = _tiny_runtime(Path(directory))
            codec = runtime.codec
            patch_size = runtime.model.patch_size
            patch_samples = codec.hop_length * patch_size
            pad = -waveform.numel() % patch_samples
            with torch.no_grad():
                expected_prompt = codec.encode(
                    torch.nn.functional.pad(waveform, (pad, 0))[None, None],
                    codec.sample_rate,
                )
            generated = torch.randn(1, runtime.model.feat_dim, 2 * patch_size)
            with patch.object(
                    runtime.model,
                    "generate_features",
                    return_value=generated,
            ) as generate, patch.object(
                    codec,
                    "decode",
                    wraps=codec.decode,
            ) as decode:
                audio = runtime.generate(
                    "a",
                    prompt_audio=waveform.numpy(),
                    prompt_sampling_rate=codec.sample_rate,
                    prompt_text="a",
                )
            prefix_features = generate.call_args.kwargs["audio_feats"][0]
            prompt_patches = expected_prompt.shape[-1] // patch_size
            context = min(3, prompt_patches)
            torch.testing.assert_close(
                prefix_features[-prompt_patches:].permute(2, 0, 1).flatten(1),
                expected_prompt[0],
            )
            decoded_latents = decode.call_args.args[0]
            torch.testing.assert_close(
                decoded_latents,
                torch.cat((expected_prompt[..., -context * patch_size:], generated), dim=-1),
            )
            with torch.no_grad():
                full = codec.decode(decoded_latents).squeeze(1)
        self.assertEqual(context, 3)
        torch.testing.assert_close(audio, full[..., context * patch_size * codec.decode_chunk_size:])
        self.assertEqual(audio.shape[-1], 2 * patch_size * codec.decode_chunk_size)

    def test_reference_audio_keeps_right_padding_and_no_codec_context(self):
        waveform = torch.randn(29)
        with tempfile.TemporaryDirectory() as directory:
            runtime = _tiny_runtime(Path(directory))
            codec = runtime.codec
            pad = -waveform.numel() % (codec.hop_length * runtime.model.patch_size)
            with torch.no_grad():
                expected = codec.encode(
                    torch.nn.functional.pad(waveform, (0, pad))[None, None],
                    codec.sample_rate,
                )
            generated = torch.randn(1, runtime.model.feat_dim, 4)
            with patch.object(
                    runtime.model,
                    "generate_features",
                    return_value=generated,
            ) as generate, patch.object(codec, "decode", wraps=codec.decode) as decode:
                runtime.generate(
                    "a",
                    reference_audio=waveform.numpy(),
                    reference_sampling_rate=codec.sample_rate,
                )
            features = generate.call_args.kwargs["audio_feats"][0]
            reference = features[1:1 + expected.shape[-1] // runtime.model.patch_size]
            torch.testing.assert_close(reference.permute(2, 0, 1).flatten(1), expected[0])
            torch.testing.assert_close(decode.call_args.args[0], generated)

    def test_generation_length_is_bounded_by_target_text_tokens(self):
        # Source `_generate_with_prompt_cache` always uses
        # max_len=min(int(len(target_tokens) * 6.0 + 10), max_len).
        with tempfile.TemporaryDirectory() as directory:
            runtime = _tiny_runtime(Path(directory))
            tokens = len(runtime.processor.tokenizer.encode("a a"))
            generated = torch.randn(1, runtime.model.feat_dim, 4)
            for requested, expected in ((2_000, int(tokens * 6.0 + 10)), (12, 12)):
                with patch.object(
                        runtime.model,
                        "generate_features",
                        return_value=generated,
                ) as generate:
                    runtime.generate("a a", max_length=requested)
                self.assertEqual(generate.call_args.kwargs["max_length"], expected)

    def test_public_generation_defaults_match_source_voxcpm_generate(self):
        # Source `VoxCPM.generate`: cfg_value=2.0, inference_timesteps=10,
        # min_len=2, max_len=4096.
        with tempfile.TemporaryDirectory() as directory:
            runtime = _tiny_runtime(Path(directory))
            wrapper = VoxCPMForTextToSpeech(VoxCPMConfig(), device="cpu")
            wrapper._runtime = runtime
            with patch.object(runtime, "generate", return_value=torch.zeros(1, 32)) as generate:
                wrapper.generate("a")
        options = generate.call_args.kwargs
        self.assertEqual(options["guidance"], 2.0)
        self.assertEqual(options["diffusion_steps"], 10)
        self.assertEqual(options["min_length"], 2)
        self.assertEqual(options["max_length"], 4_096)

    def test_generation_folds_target_text_whitespace_like_source(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = _tiny_runtime(Path(directory))
            wrapper = VoxCPMForTextToSpeech(VoxCPMConfig(), device="cpu")
            wrapper._runtime = runtime
            with patch.object(runtime, "generate", return_value=torch.zeros(1, 32)) as generate:
                wrapper.generate("a\n  a\t a")
        self.assertEqual(generate.call_args.args[0], "a a a")

    def test_external_postprocessing_options_fail_closed(self):
        wrapper = VoxCPMForTextToSpeech(
            VoxCPMConfig(),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "postprocessing"):
            wrapper.generate("a", denoise=True)


if __name__ == "__main__":
    unittest.main()
