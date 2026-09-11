from __future__ import annotations

import ast
import gc
import hashlib
import inspect
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from voicehub.checkpointing import save_safetensors
from voicehub.components.audio.codecs.encodec import EncodecConfig, EncodecModel, ResidualVectorQuantization
from voicehub.components.audio.codecs.encodec import checkpoint as checkpoint_module
from voicehub.components.audio.codecs.encodec import (
    convert_official_encodec_checkpoint,
    encodec_24khz_config,
    encodec_48khz_config,
    linear_overlap_add,
    load_encodec_model_from_safetensors,
    load_encodec_safetensors,
    resolve_encodec_checkpoint,
    save_encodec_safetensors,
    verify_native_graph_contract,
)
from voicehub.components.audio.codecs.encodec.metadata import ENCODEC_24KHZ_RELEASE, ENCODEC_48KHZ_RELEASE
from voicehub.optimization.codecs import discover_codec_compile_targets
from voicehub.runtime.catalog import register_builtin_architectures
from voicehub.runtime.registry import ArchitectureRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENCODEC_ROOT = (PROJECT_ROOT / "voicehub" / "components" / "audio" / "codecs" / "encodec")


def _tiny_config(*, segmented: bool = False) -> EncodecConfig:
    return EncodecConfig(
        target_bandwidths=(0.1, ),
        sample_rate=100,
        channels=1,
        normalize=segmented,
        segment=0.1 if segmented else None,
        overlap=0.2,
        name="tiny_encodec",
        causal=True,
        model_norm="weight_norm",
        dimension=8,
        n_filters=4,
        n_residual_layers=1,
        ratios=(2, 2),
        kernel_size=3,
        last_kernel_size=3,
        residual_kernel_size=3,
        dilation_base=2,
        true_skip=False,
        compress=2,
        lstm=0,
        bins=8,
        n_q=2,
        decay=0.99,
        kmeans_init=False,
        kmeans_iters=2,
        threshold_ema_dead_code=0,
    )


def _fingerprint(state: dict[str, torch.Tensor]) -> str:
    rows = [f"{name}|F32|{'x'.join(str(value) for value in state[name].shape)}" for name in sorted(state)]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


class NativeEncodecGraphTests(unittest.TestCase):

    def test_native_catalog_declares_exact_encodec_runtime(self):
        registry = ArchitectureRegistry()
        register_builtin_architectures(registry=registry)

        spec = registry.get("native-encodec")

        self.assertEqual(spec.architecture_id, "encodec")
        self.assertTrue(spec.capabilities.training)
        self.assertEqual(spec.capabilities.checkpoint_formats, ("safetensors", ))
        self.assertIn("straight-through-fine-tuning", spec.capabilities.features)
        self.assertEqual(
            spec.upstream_revision,
            "0e2d0aed29362c8e8f52494baf3e6f99056b214f",
        )
        self.assertEqual(
            tuple(release["tensor_count"] for release in spec.metadata["reference_releases"]),
            (252, 224),
        )

    def test_package_import_is_lazy(self):
        code = (
            "import sys\n"
            "import voicehub.components.audio.codecs.encodec\n"
            "print(int('torch' in sys.modules))\n")
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "0")

    def test_runtime_has_no_external_model_or_audio_imports(self):
        forbidden = {
            "einops",
            "encodec",
            "huggingface_hub",
            "librosa",
            "numpy",
            "requests",
            "safetensors",
            "soundfile",
            "torchaudio",
            "transformers",
        }
        violations = []
        for path in ENCODEC_ROOT.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module):
                    names = [node.module]
                else:
                    names = []
                for name in names:
                    if name.partition(".")[0] in forbidden:
                        violations.append((path.name, name))
        self.assertEqual(violations, [])

    def test_24khz_graph_matches_published_inventory(self):
        with torch.device("meta"):
            model = EncodecModel.encodec_model_24khz()
        state = dict(model.state_dict())

        self.assertEqual(model.sample_rate, 24_000)
        self.assertEqual(model.channels, 1)
        self.assertEqual(model.frame_rate, 75)
        self.assertEqual(model.quantizer.n_q, 32)
        self.assertEqual(len(state), ENCODEC_24KHZ_RELEASE.tensor_count)
        self.assertEqual(
            sum(math.prod(tensor.shape) for tensor in state.values()),
            ENCODEC_24KHZ_RELEASE.state_values,
        )
        self.assertEqual(
            _fingerprint(state),
            ENCODEC_24KHZ_RELEASE.inventory_fingerprint,
        )
        self.assertEqual(
            state["encoder.model.0.conv.conv.weight_v"].shape,
            (32, 1, 7),
        )
        self.assertEqual(
            state["quantizer.vq.layers.0._codebook.embed"].shape,
            (1024, 128),
        )
        self.assertNotIn("encoder.model.0.conv.conv.weight", state)

    def test_48khz_graph_matches_published_inventory(self):
        with torch.device("meta"):
            model = EncodecModel.encodec_model_48khz()
        state = dict(model.state_dict())

        self.assertEqual(model.sample_rate, 48_000)
        self.assertEqual(model.channels, 2)
        self.assertEqual(model.frame_rate, 150)
        self.assertEqual(model.quantizer.n_q, 16)
        self.assertEqual(model.segment_length, 48_000)
        self.assertEqual(model.segment_stride, 47_520)
        self.assertEqual(len(state), ENCODEC_48KHZ_RELEASE.tensor_count)
        self.assertEqual(
            sum(math.prod(tensor.shape) for tensor in state.values()),
            ENCODEC_48KHZ_RELEASE.state_values,
        )
        self.assertEqual(
            _fingerprint(state),
            ENCODEC_48KHZ_RELEASE.inventory_fingerprint,
        )
        self.assertEqual(
            state["encoder.model.0.conv.conv.weight"].shape,
            (32, 2, 7),
        )
        self.assertEqual(
            state["encoder.model.0.conv.norm.weight"].shape,
            (32, ),
        )
        self.assertNotIn("encoder.model.0.conv.conv.weight_v", state)

    def test_native_contract_checks_both_official_graphs(self):
        verify_native_graph_contract("encodec_24khz")
        verify_native_graph_contract("encodec_48khz")

    def test_bandwidths_select_exact_official_codebook_counts(self):
        config_24 = encodec_24khz_config()
        config_48 = encodec_48khz_config()
        with torch.device("meta"):
            model_24 = EncodecModel.from_config(config_24)
            model_48 = EncodecModel.from_config(config_48)
        self.assertEqual(
            [
                model_24.quantizer.get_num_quantizers_for_bandwidth(
                    model_24.frame_rate,
                    value,
                ) for value in config_24.target_bandwidths
            ],
            [2, 4, 8, 16, 32],
        )
        self.assertEqual(
            [
                model_48.quantizer.get_num_quantizers_for_bandwidth(
                    model_48.frame_rate,
                    value,
                ) for value in config_48.target_bandwidths
            ],
            [2, 4, 8, 16],
        )
        del model_24, model_48
        gc.collect()

    def test_tiny_graph_encodes_and_decodes_arbitrary_length(self):
        torch.manual_seed(7)
        model = EncodecModel.from_config(_tiny_config()).eval()
        waveform = torch.randn(2, 1, 33)

        frames = model.encode(waveform)
        decoded = model.decode(frames)
        output = model(waveform)

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][0].shape[:2], (2, 2))
        self.assertEqual(decoded.shape[:2], waveform.shape[:2])
        self.assertGreaterEqual(decoded.shape[-1], waveform.shape[-1])
        self.assertEqual(output.shape, waveform.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_segmented_normalized_graph_uses_visible_overlap_add(self):
        torch.manual_seed(8)
        model = EncodecModel.from_config(_tiny_config(segmented=True), ).eval()
        waveform = torch.randn(1, 1, 23)

        frames = model.encode(waveform)
        decoded = model.decode(frames)

        self.assertEqual(len(frames), 3)
        self.assertTrue(all(scale is not None for _, scale in frames))
        self.assertGreaterEqual(decoded.shape[-1], waveform.shape[-1])
        self.assertEqual(model(waveform).shape, waveform.shape)
        visible = linear_overlap_add(
            [
                torch.ones(1, 1, 6),
                torch.full((1, 1, 6), 3.0),
            ],
            4,
        )
        self.assertEqual(visible.shape[-1], 10)
        self.assertGreater(float(visible[..., 4]), 1.0)
        self.assertLess(float(visible[..., 4]), 3.0)

    def test_fine_tuning_reaches_encoder_decoder_and_later_rvq_stages(self):
        torch.manual_seed(9)
        model = EncodecModel.from_config(_tiny_config()).train()
        waveform = torch.randn(2, 1, 35)
        result = model.forward_quantized(waveform)
        objective = ((result.audio_values - waveform).square().mean() + result.commitment_loss)
        objective.backward()

        encoder_gradient = (model.encoder.model[0].conv.conv.weight_v.grad)
        decoder_gradient = (model.decoder.model[-1].conv.conv.weight_v.grad)
        self.assertIsNotNone(encoder_gradient)
        self.assertIsNotNone(decoder_gradient)
        self.assertTrue(torch.isfinite(encoder_gradient).all())
        self.assertTrue(torch.isfinite(decoder_gradient).all())
        self.assertGreater(float(encoder_gradient.abs().sum()), 0)
        self.assertGreater(float(decoder_gradient.abs().sum()), 0)

        rvq = ResidualVectorQuantization(
            num_quantizers=2,
            dim=4,
            codebook_size=8,
            codebook_dim=2,
            kmeans_init=False,
            kmeans_iters=2,
            threshold_ema_dead_code=0,
        ).train()
        quantized, _, losses = rvq(torch.randn(2, 4, 5, requires_grad=True), )
        (quantized.square().mean() + losses.mean()).backward()
        later_gradient = rvq.layers[1].project_in.weight.grad
        self.assertIsNotNone(later_gradient)
        self.assertGreater(float(later_gradient.abs().sum()), 0)

    def test_public_validation_rejects_invalid_audio_codes_and_bandwidth(self):
        model = EncodecModel.from_config(_tiny_config()).eval()
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.zeros(1, 20))
        with self.assertRaisesRegex(ValueError, "expects 1 channel"):
            model(torch.zeros(1, 2, 20))
        with self.assertRaisesRegex(TypeError, "floating-point"):
            model(torch.zeros(1, 1, 20, dtype=torch.int16))
        with self.assertRaisesRegex(ValueError, "supports bandwidths"):
            model.set_target_bandwidth(4.0)
        with self.assertRaisesRegex(ValueError, "outside"):
            model.decode([
                (
                    torch.full((1, 1, 2), model.quantizer.bins),
                    None,
                ),
            ])

    def test_decoder_optimizer_targets_capture_safe_inner_boundary(self):
        torch.manual_seed(10)
        model = EncodecModel.from_config(_tiny_config()).eval()
        frame, = model.encode(torch.randn(1, 1, 20))
        codes, scale = frame
        target, = discover_codec_compile_targets(
            model,
            mode="inference",
        )
        self.assertEqual(target.label, "codec.decode.encodec.decode_frame")
        self.assertEqual(target.attribute, "_decode_frame_unchecked")
        self.assertEqual(target.component, "decode")
        source = inspect.getsource(getattr(type(model), target.attribute))
        for value_sync in (
                "bool(",
                ".item(",
                ".any(",
                ".tolist(",
                ".numpy(",
        ):
            self.assertNotIn(value_sync, source)

        with patch.object(
                model,
                "_decode_frame_unchecked",
                wraps=model._decode_frame_unchecked,
        ) as inner:
            expected = model.decode([frame])
        self.assertEqual(inner.call_count, 1)
        self.assertIs(inner.call_args.args[0], codes)
        self.assertIs(inner.call_args.args[1], scale)

        with patch.object(
                model,
                "_validate_encoded_frame",
                side_effect=AssertionError("validation entered capture target"),
        ):
            actual = getattr(target.owner, target.attribute)(codes, scale)
        torch.testing.assert_close(actual, expected)

        invalid = codes.clone()
        invalid[..., 0] = model.quantizer.bins
        with patch.object(
                model,
                "_decode_frame_unchecked",
                wraps=model._decode_frame_unchecked,
        ) as inner:
            with self.assertRaisesRegex(ValueError, "outside"):
                model.decode([(invalid, scale)])
        inner.assert_not_called()

        training_target, = discover_codec_compile_targets(
            model.train(),
            mode="training",
        )
        self.assertEqual(training_target.attribute, "forward")
        self.assertEqual(training_target.component, "forward")


class NativeEncodecCheckpointTests(unittest.TestCase):

    def test_safetensors_export_reconstructs_a_fresh_graph(self):
        torch.manual_seed(10)
        model = EncodecModel.from_config(_tiny_config()).eval()
        reference = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as temporary:
            path = save_encodec_safetensors(
                model,
                Path(temporary) / "model.safetensors",
            )
            restored = load_encodec_model_from_safetensors(path)

        self.assertEqual(restored.config, model.config)
        self.assertEqual(set(restored.state_dict()), set(reference))
        for name, tensor in restored.state_dict().items():
            torch.testing.assert_close(tensor, reference[name])

    def test_safetensors_loader_rejects_incomplete_namespaces(self):
        model = EncodecModel.from_config(_tiny_config()).eval()
        name, tensor = next(iter(model.state_dict().items()))
        with tempfile.TemporaryDirectory() as temporary:
            path = save_safetensors(
                {name: tensor},
                Path(temporary) / "incomplete.safetensors",
            )
            with self.assertRaisesRegex(ValueError, "namespace mismatch"):
                load_encodec_safetensors(model, path)

    def test_legacy_conversion_is_explicitly_trust_gated(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / ENCODEC_24KHZ_RELEASE.filename
            source.write_bytes(b"not a checkpoint")
            with self.assertRaisesRegex(
                    PermissionError,
                    "trust_official_pickle=True",
            ):
                convert_official_encodec_checkpoint(
                    source,
                    Path(temporary) / "model.safetensors",
                    model_name="encodec_24khz",
                )

    def test_trusted_legacy_reader_still_forces_weights_only(self):
        sentinel = {"weight": torch.ones(1)}
        with (
                patch.object(
                    checkpoint_module,
                    "verify_official_checkpoint",
                    return_value="d7cc33bc" + "0" * 56,
                ),
                patch.object(
                    checkpoint_module,
                    "_validate_official_state",
                    return_value=sentinel,
                ),
                patch.object(
                    torch,
                    "load",
                    return_value=sentinel,
                ) as load,
        ):
            state, digest = checkpoint_module._restricted_official_state(
                "official.th",
                ENCODEC_24KHZ_RELEASE,
                trust_official_pickle=True,
            )

        self.assertIs(state, sentinel)
        self.assertTrue(digest.startswith("d7cc33bc"))
        load.assert_called_once_with(
            Path("official.th").resolve(),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )

    def test_local_resolver_prefers_unambiguous_safetensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / ENCODEC_24KHZ_RELEASE.filename
            native = root / "encodec_24khz.safetensors"
            legacy.touch()
            native.touch()

            resolved = resolve_encodec_checkpoint(
                "24khz",
                repository=root,
                local_files_only=True,
            )

        self.assertEqual(resolved, native.resolve())

    def test_provenance_records_pinned_source_and_audited_inventories(self):
        provenance = json.loads((ENCODEC_ROOT / "SOURCE.json").read_text(encoding="utf-8"), )
        self.assertEqual(
            provenance["source"]["revision"],
            "0e2d0aed29362c8e8f52494baf3e6f99056b214f",
        )
        artifacts = provenance["checkpoint_release"]["artifacts"]
        self.assertEqual(
            [artifact["tensor_count"] for artifact in artifacts],
            [252, 224],
        )
        self.assertTrue((ENCODEC_ROOT / "THIRD_PARTY_LICENSE").is_file())
        self.assertTrue((ENCODEC_ROOT / "VECTOR_QUANTIZE_THIRD_PARTY_LICENSE").is_file(), )


if __name__ == "__main__":
    unittest.main()
