from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from voicehub.checkpointing import save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.components.audio.codecs.base import codec_is_stochastic_vae, separate_audio_codec
from voicehub.models.xtts.configuration import XTTSConfig
from voicehub.models.xtts.modeling import XTTSForTextToSpeech
from voicehub.models.xtts.native.dvae import XTTS2DVAE, XTTS2DVAEConfig, XTTS2TrainingAudioEncoder
from voicehub.models.xtts.native.dvae_checkpoint import (
    NATIVE_XTTS2_DVAE_FILENAME,
    NATIVE_XTTS2_DVAE_FORMAT,
    NATIVE_XTTS2_DVAE_MEL_STATS_FILENAME,
    convert_trusted_legacy_xtts2_dvae_checkpoint,
    inspect_xtts2_dvae_checkpoint,
    load_xtts2_dvae_checkpoint,
    load_xtts2_training_audio_encoder,
    save_xtts2_dvae_checkpoint,
    save_xtts2_dvae_mel_stats,
)
from voicehub.models.xtts.native.metadata import XTTS2_DVAE_STORED_ELEMENT_COUNT, XTTS2_DVAE_TENSOR_COUNT
from voicehub.models.xtts.training_xtts import XTTSTrainingAdapter
from voicehub.optimization import OptimizationContext, OptimizationPassManager, TorchCompilePass
from voicehub.optimization.codecs import discover_codec_compile_targets
from voicehub.training.specs import get_training_spec
from voicehub.training.tts_datasets import TTSDataset
from voicehub.training.utils import NATIVE_EXPORT_DIR


def _tiny_config(**overrides) -> XTTS2DVAEConfig:
    values = {
        "mel_channels": 4,
        "num_tokens": 16,
        "codebook_dim": 8,
        "hidden_dim": 8,
        "num_layers": 2,
        "num_resnet_blocks": 1,
        "kernel_size": 3,
    }
    values.update(overrides)
    return XTTS2DVAEConfig(**values)


def _tiny_training_audio_encoder() -> XTTS2TrainingAudioEncoder:
    config = _tiny_config()
    return XTTS2TrainingAudioEncoder(
        XTTS2DVAE(config),
        -torch.ones(config.mel_channels),
    )


class _TinyXTTSRuntime(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.gpt = nn.Linear(2, 2)
        self.hifigan_decoder = nn.Linear(2, 1)


class NativeXTTS2DVAETests(unittest.TestCase):

    def test_published_graph_has_exact_standalone_namespace(self):
        with torch.device("meta"):
            model = XTTS2DVAE()
        state = model.state_dict()

        self.assertEqual(len(state), XTTS2_DVAE_TENSOR_COUNT)
        self.assertEqual(
            sum(value.numel() for value in state.values()),
            XTTS2_DVAE_STORED_ELEMENT_COUNT,
        )
        self.assertEqual(state["encoder.0.0.weight"].shape, (512, 80, 3))
        self.assertEqual(state["encoder.5.weight"].shape, (512, 1_024, 1))
        self.assertEqual(state["codebook.embed"].shape, (512, 1_024))
        self.assertEqual(state["decoder.4.0.conv.weight"].shape, (1_024, 1_024, 3))
        self.assertEqual(state["decoder.6.weight"].shape, (80, 512, 1))

    def test_encoder_quantizer_and_decoder_boundaries_have_tiny_gradients(self):
        model = XTTS2DVAE(_tiny_config())
        mel = torch.randn(2, 4, 17, requires_grad=True)

        encoder_latents = model.encode_latents(mel)
        encoding = model.encode_mel(mel)
        reconstruction = model.decode_codes(encoding.audio_codes)

        self.assertEqual(encoder_latents.shape, (2, 5, 8))
        self.assertEqual(encoding.audio_codes.shape, (2, 5))
        self.assertEqual(encoding.audio_codes.dtype, torch.long)
        self.assertEqual(encoding.quantized_latents.shape, (2, 5, 8))
        self.assertEqual(reconstruction.shape, (2, 4, 20))
        encoder_latents.square().mean().backward()
        self.assertIsNotNone(model.encoder[0][0].weight.grad)
        self.assertFalse(any(parameter.requires_grad for parameter in model.codebook.parameters()))

        view = separate_audio_codec(model)
        self.assertIs(view.encoder, model.encoder)
        self.assertIs(view.bottleneck, model.codebook)
        self.assertIs(view.decoder, model.decoder)
        self.assertFalse(codec_is_stochastic_vae(model, view=view))
        self.assertEqual(
            tuple(target.attribute for target in discover_codec_compile_targets(
                model,
                mode="training",
            )),
            ("forward", ),
        )
        self.assertEqual(
            tuple(target.attribute for target in discover_codec_compile_targets(
                model,
                mode="inference",
            )),
            ("decode_codes", ),
        )

    def test_strict_safetensors_round_trip_materializes_meta_graph(self):
        source = XTTS2DVAE(_tiny_config())
        with tempfile.TemporaryDirectory() as directory:
            path = save_xtts2_dvae_checkpoint(
                source,
                Path(directory) / "dvae.safetensors",
            )
            inventory = inspect_xtts2_dvae_checkpoint(path)
            with torch.device("meta"):
                target = XTTS2DVAE(_tiny_config())
            load_xtts2_dvae_checkpoint(target, path)

        self.assertEqual(inventory.tensor_count, len(source.state_dict()))
        self.assertEqual(len(inventory.header_fingerprint), 64)
        self.assertFalse(any(value.is_meta for value in target.state_dict().values()))
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[name])

    def test_loader_rejects_config_drift_even_when_tensor_shapes_match(self):
        source = XTTS2DVAE(_tiny_config())
        with tempfile.TemporaryDirectory() as directory:
            path = save_xtts2_dvae_checkpoint(
                source,
                Path(directory) / "dvae.safetensors",
            )
            with torch.device("meta"):
                target = XTTS2DVAE(_tiny_config(sample_rate=24_000))
            with self.assertRaisesRegex(
                    CheckpointCompatibilityError,
                    "architecture metadata",
            ):
                load_xtts2_dvae_checkpoint(target, path)

    def test_loader_rejects_partial_or_extended_namespaces(self):
        model = XTTS2DVAE(_tiny_config())
        state = dict(model.state_dict())
        state.pop("encoder.0.0.bias")
        state["unexpected"] = torch.zeros(1)
        metadata = {
            "format":
            NATIVE_XTTS2_DVAE_FORMAT,
            "voicehub.xtts2_dvae_config":
            json.dumps(
                asdict(model.config),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = save_safetensors(
                state,
                Path(directory) / "dvae.safetensors",
                metadata=metadata,
            )
            with self.assertRaisesRegex(
                    CheckpointCompatibilityError,
                    "missing=.*encoder.0.0.bias.*unexpected",
            ):
                load_xtts2_dvae_checkpoint(model, path)

    def test_legacy_conversion_is_explicit_and_shape_strict(self):
        model = XTTS2DVAE(_tiny_config())
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "dvae.pth"
            native = Path(directory) / "dvae.safetensors"
            torch.save(model.state_dict(), legacy)
            with self.assertRaises(PermissionError):
                convert_trusted_legacy_xtts2_dvae_checkpoint(
                    legacy,
                    native,
                    config=model.config,
                    expected_sha256=None,
                )
            converted = convert_trusted_legacy_xtts2_dvae_checkpoint(
                legacy,
                native,
                config=model.config,
                trust_legacy_pickle=True,
                expected_sha256=None,
            )
            self.assertEqual(
                inspect_xtts2_dvae_checkpoint(converted).tensor_count,
                len(model.state_dict()),
            )

    def test_waveform_boundary_and_adapter_preserve_precomputed_path(self):
        config = _tiny_config()
        model = XTTS2DVAE(config)
        with tempfile.TemporaryDirectory() as directory:
            dvae_path = save_xtts2_dvae_checkpoint(
                model,
                Path(directory) / "dvae.safetensors",
            )
            stats_path = save_xtts2_dvae_mel_stats(
                -torch.ones(config.mel_channels),
                Path(directory) / "mel_stats.safetensors",
            )
            boundary = load_xtts2_training_audio_encoder(
                dvae_path,
                stats_path,
                config=config,
            )

        waveform = torch.randn(2, 2_048)
        codes = boundary(waveform)
        self.assertEqual(codes.shape, (2, 3))
        self.assertFalse(any(parameter.requires_grad for parameter in boundary.parameters()))

        wrapper = SimpleNamespace(
            training_audio_encoder=boundary,
            _runtime_config=SimpleNamespace(audio=SimpleNamespace(sample_rate=22_050), ),
        )
        adapter = XTTSTrainingAdapter(
            wrapper,
            get_training_spec("xtts"),
        )
        prepared = adapter.prepare_training_inputs(
            {
                "text_inputs": torch.tensor([[1, 2], [3, 4]]),
                "text_lengths": torch.tensor([2, 2]),
                "wav": waveform,
                "cond_latents": torch.randn(2, 2, 4),
            },
            None,
        )
        self.assertEqual(prepared["audio_codes"].shape, (2, 3))
        self.assertEqual(prepared["wav_lengths"].tolist(), [2_048, 2_048])
        self.assertNotIn("wav", prepared)

        precomputed = torch.tensor([[1, 2], [3, 4]])
        prepared = adapter.prepare_training_inputs(
            {
                "text_inputs": torch.tensor([[1], [2]]),
                "text_lengths": torch.tensor([1, 1]),
                "audio_codes": precomputed,
                "wav_lengths": torch.tensor([2_048, 2_048]),
                "cond_latents": torch.randn(2, 2, 4),
            },
            None,
        )
        self.assertIs(prepared["audio_codes"], precomputed)

    def test_public_config_requires_both_safe_training_artifacts(self):
        with self.assertRaisesRegex(ValueError, "requires both"):
            XTTSConfig(training_dvae_checkpoint="dvae.safetensors")
        with self.assertRaisesRegex(ValueError, "safetensors"):
            XTTSConfig(
                training_dvae_checkpoint="dvae.pth",
                training_mel_stats_checkpoint="mel_stats.pth",
            )

    def test_dataset_contract_accepts_waveform_target_preparation(self):
        dataset = TTSDataset(
            [{
                "text_inputs": [1, 2, 3],
                "text_lengths": 3,
                "wav": [0.0, 0.1, -0.1],
                "cond_latents": [[0.1, 0.2]],
            }],
            model_type="xtts",
        )
        self.assertEqual(dataset.variant_names, ("native-gpt-waveform", ))

    def test_training_optimizer_compiles_and_restores_frozen_dvae_boundary(self):
        if not callable(getattr(torch, "compile", None)):
            self.skipTest("torch.compile requires PyTorch 2 or newer")
        runtime = _TinyXTTSRuntime()
        boundary = _tiny_training_audio_encoder()
        wrapper = SimpleNamespace(
            native_runtime=runtime,
            training_audio_encoder=boundary,
        )
        adapter = XTTSTrainingAdapter(
            wrapper,
            get_training_spec("xtts"),
        )
        adapter.primary_model = runtime.gpt
        adapter.primary_path = "model.gpt"
        adapter._components = [("model.gpt", runtime.gpt)]

        roots = adapter.optimization_module_roots()
        self.assertEqual(
            tuple(label for label, _module in roots),
            (
                "xtts.gpt",
                "xtts.training_audio_encoder",
            ),
        )
        self.assertIs(roots[1][1], boundary)
        self.assertFalse(boundary.training)
        self.assertFalse(any(parameter.requires_grad for parameter in boundary.parameters()))

        compiled_owners = []

        def compile_method(function, **kwargs):
            del kwargs
            compiled_owners.append(function.__self__)
            return function

        original_state_topology = adapter.state_dict()["topology"]
        with mock.patch.object(
                torch,
                "compile",
                side_effect=compile_method,
        ):
            result = OptimizationPassManager().apply(
                adapter,
                (TorchCompilePass(
                    backend="eager",
                    requirement="required",
                ), ),
                OptimizationContext(
                    mode="training",
                    device="cpu",
                    dtype="float32",
                    persist_result=True,
                ),
            )

        self.assertEqual(compiled_owners, [runtime.gpt, boundary])
        self.assertIn("forward", runtime.gpt.__dict__)
        self.assertIn("forward", boundary.__dict__)
        self.assertEqual(
            result.manifest_metadata()[0]["metadata"]["execution_targets"],
            [
                "xtts.gpt.forward",
                "xtts.training_audio_encoder.forward",
            ],
        )
        self.assertEqual(
            adapter.state_dict()["topology"],
            original_state_topology,
        )

        self.assertIs(result.restore(), adapter)
        self.assertNotIn("forward", runtime.gpt.__dict__)
        self.assertNotIn("forward", boundary.__dict__)
        self.assertFalse(boundary.training)
        self.assertFalse(any(parameter.requires_grad for parameter in boundary.parameters()))

    def test_portable_export_rediscovers_moved_bundled_training_artifacts(self):
        boundary = _tiny_training_audio_encoder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_source = root / "runtime-source"
            runtime_source.mkdir()
            (runtime_source / "config.json").write_text(
                json.dumps({}),
                encoding="utf-8",
            )
            (runtime_source / "vocab.json").write_text(
                json.dumps({}),
                encoding="utf-8",
            )
            original_artifacts = root / "original-artifacts"
            original_artifacts.mkdir()
            original_dvae = save_xtts2_dvae_checkpoint(
                boundary.dvae,
                original_artifacts / NATIVE_XTTS2_DVAE_FILENAME,
            )
            original_stats = save_xtts2_dvae_mel_stats(
                boundary.mel_processor.mel_stats,
                original_artifacts / NATIVE_XTTS2_DVAE_MEL_STATS_FILENAME,
            )

            wrapper = XTTSForTextToSpeech(
                XTTSConfig(
                    name_or_path=runtime_source,
                    training_dvae_checkpoint=original_dvae,
                    training_mel_stats_checkpoint=original_stats,
                ),
                device="cpu",
                lazy_load=True,
            )
            wrapper.model = _TinyXTTSRuntime()
            wrapper._model_directory = runtime_source
            wrapper._training_audio_encoder = boundary
            export_directory = root / "export"
            wrapper.save_pretrained(export_directory)

            portable_config = json.loads((export_directory / "config.json").read_text(encoding="utf-8"))
            self.assertIsNone(portable_config["training_dvae_checkpoint"])
            self.assertIsNone(portable_config["training_mel_stats_checkpoint"])
            self.assertTrue((export_directory / NATIVE_EXPORT_DIR / NATIVE_XTTS2_DVAE_FILENAME).is_file())
            self.assertTrue(
                (export_directory / NATIVE_EXPORT_DIR / NATIVE_XTTS2_DVAE_MEL_STATS_FILENAME).is_file())

            original_dvae.unlink()
            original_stats.unlink()
            moved_directory = root / "moved-export"
            export_directory.rename(moved_directory)

            fresh = XTTSForTextToSpeech.from_pretrained(
                moved_directory,
                device="cpu",
                lazy_load=True,
            )
            resolved = fresh._training_audio_artifacts()
            expected = (
                (moved_directory / NATIVE_EXPORT_DIR / NATIVE_XTTS2_DVAE_FILENAME).resolve(),
                (moved_directory / NATIVE_EXPORT_DIR / NATIVE_XTTS2_DVAE_MEL_STATS_FILENAME).resolve(),
            )
            self.assertEqual(resolved, expected)

            fresh.model = _TinyXTTSRuntime()
            fresh._model_directory = moved_directory / NATIVE_EXPORT_DIR
            attached = nn.Identity()

            def attach(dvae_checkpoint, mel_stats_checkpoint):
                self.assertEqual(
                    (
                        Path(dvae_checkpoint),
                        Path(mel_stats_checkpoint),
                    ),
                    expected,
                )
                fresh._training_audio_encoder = attached
                return attached

            with mock.patch.object(
                    fresh,
                    "configure_training_audio_encoder",
                    side_effect=attach,
            ) as configure:
                fresh._prepare_for_training()
            configure.assert_called_once_with(*expected)
            self.assertIs(fresh.training_audio_encoder, attached)


if __name__ == "__main__":
    unittest.main()
