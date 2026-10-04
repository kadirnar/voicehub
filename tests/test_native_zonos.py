from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from voicehub.architectures.registry import ArchitectureRegistry
from voicehub.architectures.zonos.artifacts import ZonosArtifacts, resolve_zonos_artifacts
from voicehub.architectures.zonos.checkpoint import (
    export_zonos_checkpoint,
    load_zonos_checkpoint,
    save_zonos_pretrained,
    zonos_inventory_fingerprint,
)
from voicehub.architectures.zonos.configuration import ZonosArchitectureConfig, ZonosBackboneConfig
from voicehub.architectures.zonos.frontend import (
    PHONEME_SYMBOLS,
    UNK_ID,
    PrecomputedPhonemeFrontend,
    batch_phoneme_ids,
    make_condition_dict,
    resolve_phonemes,
    tokenize_phonemes,
)
from voicehub.architectures.zonos.metadata import (
    NATIVE_ZONOS_FORMAT,
    ZONOS_HYBRID_REPOSITORY,
    ZONOS_SOURCE_REVISION,
    ZONOS_TRANSFORMER_HEADER_FINGERPRINT,
    ZONOS_TRANSFORMER_PARAMETER_COUNT,
    ZONOS_TRANSFORMER_REPOSITORY,
    ZONOS_TRANSFORMER_REVISION,
    ZONOS_TRANSFORMER_TENSOR_COUNT,
)
from voicehub.architectures.zonos.modeling import ZonosForCausalLM, ZonosFourierConditioner
from voicehub.architectures.zonos.pattern import apply_delay_pattern
from voicehub.architectures.zonos.registration import create_zonos_architecture_spec, register_zonos_architecture
from voicehub.architectures.zonos.runtime import NativeZonosRuntime, ZonosGeneration
from voicehub.architectures.zonos.sampling import ZonosSamplingOptions, sample_zonos_token
from voicehub.checkpointing import save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.zonos.inference import ZonosConfig, ZonosForTextToSpeech
from voicehub.models.zonos.training import ZonosTrainingAdapter
from voicehub.training.specs import get_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_ROOT = PROJECT_ROOT / "voicehub" / "architectures" / "zonos"


def _tiny_config() -> ZonosArchitectureConfig:
    return ZonosArchitectureConfig(
        backbone=ZonosBackboneConfig(
            d_model=16,
            attn_mlp_d_intermediate=32,
            n_layer=2,
            attn_layer_idx=(0, 1),
            attn_cfg={
                "causal": True,
                "num_heads": 4,
                "num_heads_kv": 2,
                "rotary_emb_dim": 4,
                "rotary_emb_interleaved": True,
                "qkv_proj_bias": False,
                "out_proj_bias": False,
            },
        ), )


def _tiny_prefix(
    model: ZonosForCausalLM,
    *,
    batch_size: int = 2,
) -> torch.Tensor:
    ids = tokenize_phonemes("həlo").repeat(batch_size, 1)
    values = make_condition_dict(
        ids,
        language=["en-us"] * batch_size,
        device=model.device,
    )
    return model.prefix_conditioner(values)


class _Wrapper:

    def __init__(self, model: ZonosForCausalLM):
        self.model = model
        self.config = ZonosConfig(name_or_path="local")

    def load_for_training(self):
        self.model.train()
        return self


class _FakeCodec:
    sample_rate = 44_100
    hop_length = 512
    num_codebooks = 9
    codebook_size = 1_024

    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        return self

    def encode(self, waveform, *, sample_rate):
        del waveform, sample_rate
        return torch.zeros(1, 9, 2, dtype=torch.long, device=self.device)

    def decode(self, codes):
        return torch.zeros(
            codes.shape[0],
            1,
            codes.shape[-1] * 512,
            device=self.device,
        )


class NativeZonosConfigurationTests(unittest.TestCase):

    def test_pinned_provenance_is_present(self):
        source = json.loads((ARCHITECTURE_ROOT / "SOURCE.json").read_text(encoding="utf-8"), )
        self.assertEqual(
            source["source"]["revision"],
            ZONOS_SOURCE_REVISION,
        )
        self.assertEqual(
            source["checkpoint"]["revision"],
            ZONOS_TRANSFORMER_REVISION,
        )
        self.assertEqual(
            source["checkpoint"]["tensor_count"],
            ZONOS_TRANSFORMER_TENSOR_COUNT,
        )
        self.assertTrue((ARCHITECTURE_ROOT / "THIRD_PARTY_LICENSE").is_file(), )

    def test_default_meta_graph_matches_the_official_safe_header(self):
        with torch.device("meta"):
            model = ZonosForCausalLM(ZonosArchitectureConfig())
        state = model.state_dict()
        self.assertEqual(len(state), ZONOS_TRANSFORMER_TENSOR_COUNT)
        self.assertEqual(
            sum(value.numel() for value in state.values()),
            ZONOS_TRANSFORMER_PARAMETER_COUNT,
        )
        inventory = {name: ("BF16", tuple(value.shape)) for name, value in state.items()}
        self.assertEqual(
            zonos_inventory_fingerprint(inventory),
            ZONOS_TRANSFORMER_HEADER_FINGERPRINT,
        )
        self.assertEqual(
            tuple(state["backbone.layers.0.mixer.in_proj.weight"].shape),
            (3_072, 2_048),
        )
        self.assertEqual(
            tuple(state["prefix_conditioner.conditioners.0."
                        "phoneme_embedder.weight"].shape),
            (189, 2_048),
        )

    def test_hybrid_configuration_fails_closed(self):
        values = ZonosArchitectureConfig().to_dict()
        values["backbone"]["ssm_cfg"] = {"layer": "Mamba2"}
        with self.assertRaisesRegex(NotImplementedError, "hybrid Mamba-2"):
            ZonosArchitectureConfig.from_dict(values)
        with self.assertRaisesRegex(NotImplementedError, "hybrid"):
            resolve_zonos_artifacts(ZONOS_HYBRID_REPOSITORY)

    def test_architecture_registration_is_lazy_and_truthful(self):
        spec = create_zonos_architecture_spec()
        self.assertEqual(spec.architecture_id, "zonos")
        self.assertTrue(spec.capabilities.training)
        self.assertIn("safetensors", spec.capabilities.checkpoint_formats)
        self.assertFalse(spec.metadata["hybrid_support"])
        registry = ArchitectureRegistry()
        register_zonos_architecture(registry=registry)
        self.assertIs(
            registry.get("native-zonos"),
            registry.get("zonos"),
        )


class NativeZonosFrontendTests(unittest.TestCase):

    def test_published_phoneme_vocabulary_is_exact(self):
        self.assertEqual(len(PHONEME_SYMBOLS), 185)
        ids = tokenize_phonemes("həlo")
        self.assertEqual(ids[0].item(), 2)
        self.assertEqual(ids[-1].item(), 3)
        self.assertEqual(ids.numel(), 6)

    def test_batch_tokenization_left_pads_like_upstream(self):
        ids, lengths = batch_phoneme_ids(("a", "həlo"))
        self.assertEqual(lengths.tolist(), [3, 6])
        self.assertEqual(ids.shape, (2, 6))
        self.assertEqual(ids[0, :3].tolist(), [0, 0, 0])

    def test_raw_text_requires_an_explicit_frontend(self):
        with self.assertRaisesRegex(RuntimeError, "requires eSpeak-compatible"):
            resolve_phonemes("hello", language="en-us")
        value, frontend_id = resolve_phonemes(
            "həlo",
            language="en-us",
            frontend=PrecomputedPhonemeFrontend(),
        )
        self.assertEqual(value, "həlo")
        self.assertEqual(frontend_id, "precomputed-phonemes")

    def test_out_of_vocabulary_phonemes_map_to_unk_like_upstream(self):
        # eSpeak output for supported languages (here French "bonjour")
        # contains symbols outside the published table; the released
        # tokenizer maps them to UNK instead of rejecting the request.
        ids = tokenize_phonemes("bɔ̃ʒˈuʁ")
        self.assertEqual(ids.numel(), 9)
        self.assertEqual(ids[3].item(), UNK_ID)
        self.assertNotIn(UNK_ID, ids[[1, 2, 4, 5, 6, 7]].tolist())
        with self.assertRaisesRegex(ValueError, "no symbol from the published"):
            tokenize_phonemes("你好")

    def test_condition_dictionary_validates_features(self):
        ids = tokenize_phonemes("həlo").unsqueeze(0)
        condition = make_condition_dict(
            ids,
            language="en-us",
            speaker_embedding=torch.ones(128),
        )
        self.assertEqual(condition["espeak"].shape, (1, 6))
        self.assertEqual(condition["speaker"].shape, (1, 1, 128))
        self.assertAlmostEqual(
            condition["emotion"].sum().item(),
            1.0,
            places=6,
        )


class NativeZonosModelTests(unittest.TestCase):

    def test_delay_pattern_does_not_leak_between_batches(self):
        codes = torch.tensor([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ])
        delayed = apply_delay_pattern(codes, 99)
        self.assertEqual(delayed[1, 0, 0].item(), 99)
        self.assertEqual(delayed[1, 0, 1].item(), 7)

    def test_teacher_forcing_is_batch_safe_and_fully_differentiable(self):
        model = ZonosForCausalLM(_tiny_config())
        prefix = _tiny_prefix(model)
        full = torch.arange(36).reshape(9, 4) + 10
        short = torch.full((9, 4), -100)
        short[:, :2] = torch.arange(18).reshape(9, 2) + 100
        codes = torch.stack((full, short))
        lengths = torch.tensor([4, 2])
        output = model(
            prefix,
            codes,
            audio_code_lengths=lengths,
        )
        self.assertEqual(output.logits.shape[:3], output.labels.shape)
        for batch_index, length in enumerate(lengths.tolist()):
            for codebook in range(9):
                self.assertEqual(
                    output.labels[
                        batch_index,
                        codebook,
                        length + codebook,
                    ].item(),
                    model.eos_token_id,
                )
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNotNone(model.backbone.layers[0].mixer.in_proj.weight.grad, )
        self.assertGreater(
            model.heads[0].weight.grad[model.eos_token_id].abs().sum().item(),
            0,
        )
        self.assertGreater(
            model.prefix_conditioner.conditioners[1].uncond_vector.grad.abs().sum().item(),
            0,
        )

    def test_fourier_conditioner_normalizes_before_casting(self):
        torch.manual_seed(0)
        conditioner = ZonosFourierConditioner(
            16,
            name="speaking_rate",
            min_val=0,
            max_val=40,
            uncond_type="learned",
        ).to(torch.bfloat16)
        value = torch.tensor([[[3.7]]])
        # Released order: normalize in float32, then cast to the weight dtype.
        angles = 2 * torch.pi * (value / 40).to(torch.bfloat16) @ conditioner.weight.T
        expected = torch.cat((angles.cos(), angles.sin()), dim=-1)
        self.assertTrue(torch.equal(conditioner(value), expected))
        cast_first = 2 * torch.pi * (value.to(torch.bfloat16) / 40) @ conditioner.weight.T
        self.assertFalse(torch.equal(cast_first, angles))

    def test_generation_logits_keep_released_padded_layout(self):
        config = _tiny_config()
        self.assertEqual(config.output_vocab_size, 1_025)
        self.assertEqual(config.generation_logits_width, 1_026)
        model = ZonosForCausalLM(config).eval()
        prefix = torch.cat((_tiny_prefix(model, batch_size=1), _tiny_prefix(model, batch_size=1)))
        cache = model.setup_cache(batch_size=2, max_sequence_length=prefix.shape[1] + 4)
        codes = torch.full((1, 9, 1), model.masked_token_id)
        with torch.inference_mode():
            logits = model.prefill(prefix, codes, cache, cfg_scale=2.0)
        self.assertEqual(logits.shape, (1, 9, 1_026))
        self.assertTrue(torch.isneginf(logits[..., 1_025:]).all())
        self.assertTrue(torch.isfinite(logits[..., :1_025]).all())
        # Same seed, same noise layout as the released sampler; the masked
        # column can never be drawn.
        generator = torch.Generator().manual_seed(7)
        token = sample_zonos_token(logits, options=ZonosSamplingOptions(), generator=generator)
        noise = torch.empty_like(logits).exponential_(1, generator=torch.Generator().manual_seed(7))
        probabilities = torch.softmax(logits, dim=-1)
        top = probabilities.max(dim=-1, keepdim=True).values
        probabilities = probabilities.masked_fill(probabilities < 0.1 * top, 0.0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        self.assertTrue(torch.equal(token, torch.argmax(probabilities / noise, dim=-1, keepdim=True)))
        self.assertTrue(bool((token < 1_025).all()))

    def test_decode_reuses_rotary_table_without_host_sync(self):
        from voicehub.architectures.zonos import modeling

        torch.manual_seed(0)
        model = ZonosForCausalLM(_tiny_config()).eval()
        prefix = torch.cat((_tiny_prefix(model, batch_size=1), _tiny_prefix(model, batch_size=1)))
        codes = torch.full((1, 9, 1), model.masked_token_id)
        cache = model.setup_cache(batch_size=2, max_sequence_length=prefix.shape[1] + 4)
        build = modeling.precompute_rotary_frequencies
        with torch.inference_mode(), patch.object(
                modeling,
                "precompute_rotary_frequencies",
                side_effect=build,
        ) as precompute:
            model.prefill(prefix, codes, cache, cfg_scale=2.0)
            cache.sequence_offset += prefix.shape[1] + 1
            cache.lengths_per_sample.add_(prefix.shape[1] + 1)
            # Decode steps must not read device values back to the host.
            with patch.object(torch.Tensor, "item", side_effect=AssertionError("host sync")):
                while cache.sequence_offset < cache.max_sequence_length:
                    logits = model.decode_step(codes, cache, cfg_scale=2.0)
                    cache.sequence_offset += 1
                    cache.lengths_per_sample.add_(1)
        self.assertEqual(precompute.call_count, 1)
        self.assertTrue(
            torch.equal(
                cache.rotary_frequencies,
                build(cache.max_sequence_length, model.config.backbone.head_dim, device="cpu"),
            ))
        self.assertTrue(torch.isfinite(logits[..., :1_025]).all())
        with self.assertRaisesRegex(RuntimeError, "rotary position exceeds"):
            model.decode_step(codes, cache, cfg_scale=2.0)

    def test_seeded_sampling_is_request_local(self):
        logits = torch.randn(1, 9, 1_025)
        options = ZonosSamplingOptions(min_p=0.1)
        first = sample_zonos_token(
            logits,
            options=options,
            generator=torch.Generator().manual_seed(41),
        )
        second = sample_zonos_token(
            logits,
            options=options,
            generator=torch.Generator().manual_seed(41),
        )
        self.assertTrue(torch.equal(first, second))


class NativeZonosCheckpointTests(unittest.TestCase):

    def test_export_strict_reload_and_fresh_runtime(self):
        config = _tiny_config()
        source = ZonosForCausalLM(config)
        with tempfile.TemporaryDirectory() as directory:
            destination = save_zonos_pretrained(source, directory)
            with torch.device("meta"):
                target = ZonosForCausalLM(config)
            report = load_zonos_checkpoint(
                target,
                destination / "model.safetensors",
                device="cpu",
                dtype=torch.float32,
            )
            self.assertEqual(report.tensor_count, len(source.state_dict()))
            for name, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[name]))
            runtime = NativeZonosRuntime.from_pretrained(
                destination,
                device="cpu",
                dtype="float32",
                codec=_FakeCodec(),
            )
            self.assertEqual(runtime.config.backbone.d_model, 16)
            self.assertEqual(
                runtime.model.state_dict().keys(),
                source.state_dict().keys(),
            )

    def test_loader_rejects_an_incomplete_checkpoint(self):
        config = _tiny_config()
        source = ZonosForCausalLM(config)
        state = dict(source.state_dict())
        state.pop(next(iter(state)))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "broken.safetensors"
            save_safetensors(state, checkpoint)
            with torch.device("meta"):
                target = ZonosForCausalLM(config)
            with self.assertRaises(CheckpointCompatibilityError):
                load_zonos_checkpoint(
                    target,
                    checkpoint,
                    device="cpu",
                )

    def test_export_rejects_partial_state_override(self):
        model = ZonosForCausalLM(_tiny_config())
        state = dict(model.state_dict())
        state.pop(next(iter(state)))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                export_zonos_checkpoint(
                    model,
                    Path(directory) / "partial.safetensors",
                    state_override=state,
                )

    def test_local_artifact_resolution_is_unambiguous(self):
        model = ZonosForCausalLM(_tiny_config())
        with tempfile.TemporaryDirectory() as directory:
            save_zonos_pretrained(model, directory)
            artifacts = resolve_zonos_artifacts(directory)
            self.assertEqual(artifacts.config.name, "config.json")
            self.assertEqual(artifacts.checkpoint.name, "model.safetensors")

    def test_official_resolution_is_pinned(self):
        with patch(
                "voicehub.architectures.zonos.artifacts."
                "resolve_pretrained_file",
                side_effect=lambda source, filename, **kwargs: Path(filename),
        ) as resolver:
            artifacts = resolve_zonos_artifacts(ZONOS_TRANSFORMER_REPOSITORY, )
        self.assertEqual(artifacts.revision, ZONOS_TRANSFORMER_REVISION)
        self.assertEqual(resolver.call_count, 2)
        for call in resolver.call_args_list:
            self.assertEqual(
                call.kwargs["revision"],
                ZONOS_TRANSFORMER_REVISION,
            )


class NativeZonosRuntimeTests(unittest.TestCase):

    def test_runtime_accepts_precomputed_phonemes_and_native_codec(self):
        model = ZonosForCausalLM(_tiny_config())
        runtime = NativeZonosRuntime(
            artifacts=ZonosArtifacts(
                config=Path("config.json"),
                checkpoint=Path("model.safetensors"),
                source="test",
                revision=None,
            ),
            config=model.config,
            model=model,
            codec=_FakeCodec(),
        )
        expected_codes = torch.zeros(1, 9, 3, dtype=torch.long)
        with patch(
                "voicehub.architectures.zonos.runtime.generate_zonos_codes",
                return_value=expected_codes,
        ):
            result = runtime.generate(
                "ignored raw text",
                phonemes="həlo",
            )
        self.assertEqual(result.text_frontend, "precomputed-phonemes")
        self.assertTrue(torch.equal(result.codes, expected_codes))
        self.assertEqual(result.audio.shape, (1, 1, 1_536))

    def test_training_adapter_exports_fresh_inference_runtime(self):
        model = ZonosForCausalLM(_tiny_config())
        adapter = ZonosTrainingAdapter(
            _Wrapper(model),
            get_training_spec("zonos"),
        )
        prefix = _tiny_prefix(model)
        output = adapter(
            prefix_conditioning=prefix,
            audio_codes=torch.randint(0, 1_024, (2, 9, 4)),
        )
        output.loss.backward()
        self.assertEqual(output.metadata["supervised_tokens"], 90)
        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            runtime = NativeZonosRuntime.from_pretrained(
                directory,
                device="cpu",
                dtype="float32",
                codec=_FakeCodec(),
            )
            self.assertEqual(runtime.config.backbone.d_model, 16)

    def test_raw_audio_training_batches_use_the_frozen_codec_boundary(self):
        model = ZonosForCausalLM(_tiny_config())
        runtime = NativeZonosRuntime(
            artifacts=ZonosArtifacts(
                config=Path("config.json"),
                checkpoint=Path("model.safetensors"),
                source="test",
                revision=None,
            ),
            config=model.config,
            model=model,
            codec=_FakeCodec(),
        )
        wrapper = ZonosForTextToSpeech(device="cpu")
        wrapper.model = model
        wrapper._runtime = runtime
        adapter = ZonosTrainingAdapter(
            wrapper,
            get_training_spec("zonos"),
        )

        def encode(example, *, sampling_rate):
            self.assertEqual(sampling_rate, 44_100)
            frames = example.shape[-1] // 4
            return torch.zeros(1, 9, frames, dtype=torch.long)

        with patch.object(runtime, "encode_audio", side_effect=encode) as codec:
            output = adapter(
                texts=("first", "second"),
                phonemes=("həlo", "a"),
                audio=torch.zeros(2, 16),
                audio_lengths=torch.tensor([16, 8]),
                sampling_rates=torch.tensor([44_100, 44_100]),
            )

        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(codec.call_count, 2)
        self.assertEqual(
            [call.args[0].shape[-1] for call in codec.call_args_list],
            [16, 8],
        )
        self.assertEqual(output.metadata["supervised_tokens"], 72)

    def test_public_wrapper_uses_native_runtime_contract(self):
        wrapper = ZonosForTextToSpeech(device="cpu")
        runtime = unittest.mock.Mock()
        runtime.generate.return_value = ZonosGeneration(
            codes=torch.zeros(1, 9, 2, dtype=torch.long),
            audio=torch.zeros(1, 1, 1_024),
            sample_rate=44_100,
            text_frontend="precomputed-phonemes",
        )
        wrapper._runtime = runtime
        wrapper.model = ZonosForCausalLM(_tiny_config())
        output = wrapper._generate(
            "hello",
            phonemes="həlo",
            seed=7,
        )
        self.assertEqual(output.sample_rate, 44_100)
        self.assertEqual(
            output.metadata["architecture"],
            "voicehub-native-zonos-transformer",
        )


class NativeZonosResamplingTests(unittest.TestCase):

    def setUp(self):
        try:
            import torchaudio
        except ImportError:  # pragma: no cover - optional reference
            self.skipTest("torchaudio is not installed")
        self.torchaudio = torchaudio
        generator = torch.Generator().manual_seed(0)
        self.waveform = torch.randn(2, 1, 1_001, generator=generator)

    def test_codec_preprocess_matches_upstream_functional_resample(self):
        from voicehub.architectures.zonos.codec import ZonosDACCodec

        expected = self.torchaudio.functional.resample(self.waveform, 16_000, 44_100)
        expected = torch.nn.functional.pad(expected, (0, -expected.shape[-1] % 512))
        actual = ZonosDACCodec()._preprocess(self.waveform, sample_rate=16_000)
        self.assertTrue(torch.equal(actual, expected))

    def test_vendored_autoencoder_preprocess_matches_upstream(self):
        from voicehub.models.zonos.source.zonos.autoencoder import DACAutoencoder

        expected = self.torchaudio.functional.resample(self.waveform, 24_000, 44_100)
        expected = torch.nn.functional.pad(expected, (0, -expected.shape[-1] % 512))
        actual = DACAutoencoder.preprocess(DACAutoencoder.__new__(DACAutoencoder), self.waveform, 24_000)
        self.assertTrue(torch.equal(actual, expected))

    def test_speaker_input_matches_upstream_resample_transform(self):
        from voicehub.models.zonos.source.zonos.speaker_cloning import SpeakerEmbedding

        stereo = self.waveform[:, 0]
        expected = self.torchaudio.transforms.Resample(22_050, 16_000)(stereo.mean(0, keepdim=True))
        actual = SpeakerEmbedding.prepare_input(None, stereo, 22_050)
        self.assertTrue(torch.equal(actual, expected))


class NativeZonosDependencyTests(unittest.TestCase):

    def test_native_runtime_has_no_external_architecture_imports(self):
        allowed = {
            "__future__",
            "collections",
            "copy",
            "dataclasses",
            "hashlib",
            "importlib",
            "json",
            "math",
            "numbers",
            "pathlib",
            "torch",
            "types",
            "typing",
            "voicehub",
        }
        files = tuple(ARCHITECTURE_ROOT.glob("*.py")) + (
            PROJECT_ROOT / "voicehub" / "models" / "zonos" / "inference.py",
            PROJECT_ROOT / "voicehub" / "models" / "zonos" / "training.py",
        )
        violations = []
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    root = module.split(".", 1)[0]
                    if root not in allowed:
                        violations.append((path.name, module))
        self.assertEqual(violations, [])

    def test_native_format_is_stable(self):
        self.assertEqual(
            NATIVE_ZONOS_FORMAT,
            "voicehub-zonos-v0.1-transformer-v1",
        )


if __name__ == "__main__":
    unittest.main()
