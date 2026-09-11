from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from voicehub.checkpointing import save_safetensors
from voicehub.models.f5tts.modeling import F5TTSConfig, F5TTSForTextToSpeech
from voicehub.models.f5tts.native.audio import F5MelSpectrogram
from voicehub.models.f5tts.native.checkpoint import (
    convert_legacy_f5tts_checkpoint,
    export_f5tts_checkpoint,
    load_f5tts_checkpoint,
    load_vocos_checkpoint,
)
from voicehub.models.f5tts.native.configuration import F5TTSArchitectureConfig
from voicehub.models.f5tts.native.frontend import F5Vocabulary, NativeF5TextFrontend
from voicehub.models.f5tts.native.metadata import (
    F5TTS_CHECKPOINT_LICENSE,
    F5TTS_CHECKPOINT_REVISION,
    F5TTS_SOURCE_REVISION,
    F5TTS_V1_BASE_GRAPH_TENSOR_COUNT,
    F5TTS_V1_BASE_PARAMETER_COUNT,
    VOCOS_CHECKPOINT_REVISION,
    VOCOS_SOURCE_REVISION,
)
from voicehub.models.f5tts.native.modeling import F5ConditionalFlowMatcher
from voicehub.models.f5tts.native.modules import RotaryEmbedding, apply_rotary_position_embedding
from voicehub.models.f5tts.native.registration import create_f5tts_architecture_spec
from voicehub.models.f5tts.native.runtime import NativeF5TTSRuntime
from voicehub.models.f5tts.native.vocoder import ISTFTHead, NativeVocos
from voicehub.training.arguments import TrainingArguments
from voicehub.training.recipes import F5TTSTrainingAdapter
from voicehub.training.specs import get_training_spec
from voicehub.training.trainer import Trainer


def _tiny_config() -> F5TTSArchitectureConfig:
    return F5TTSArchitectureConfig(
        model_name="test-f5",
        mel_dim=8,
        dim=32,
        depth=2,
        heads=4,
        dim_head=8,
        text_dim=16,
        text_num_embeds=12,
        conv_layers=1,
        n_fft=32,
        win_length=32,
        hop_length=8,
        sample_rate=8_000,
        dropout=0.0,
    )


def _tiny_vocabulary() -> F5Vocabulary:
    return F5Vocabulary((" ", ".", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"))


class NativeF5TTSRuntimeTests(unittest.TestCase):

    def test_registration_is_lazy_and_records_distinct_artifact_license(self):
        code = (
            "import json,sys;"
            "from voicehub.models.f5tts.native.registration import "
            "create_f5tts_architecture_spec;"
            "s=create_f5tts_architecture_spec();"
            "print(json.dumps({'torch': 'torch' in sys.modules,"
            "'license': s.license_id,"
            "'checkpoint': s.metadata['checkpoint_license']}))")
        result = subprocess.run(
            (sys.executable, "-c", code),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertFalse(payload["torch"])
        self.assertEqual(payload["license"], "MIT")
        self.assertEqual(payload["checkpoint"], F5TTS_CHECKPOINT_LICENSE)
        spec = create_f5tts_architecture_spec()
        self.assertTrue(spec.capabilities.training)
        self.assertTrue(spec.metadata["full_finetuning_ready"])
        self.assertEqual(
            spec.metadata["quality_validated_inference_dtypes"],
            ("float32", ),
        )
        self.assertIn(
            "Explicit opt-in",
            spec.metadata["reduced_precision_inference_policy"],
        )
        self.assertEqual(len(F5TTS_SOURCE_REVISION), 40)
        self.assertEqual(len(F5TTS_CHECKPOINT_REVISION), 40)
        self.assertEqual(len(VOCOS_SOURCE_REVISION), 40)
        self.assertEqual(len(VOCOS_CHECKPOINT_REVISION), 40)

    def test_released_graph_has_the_audited_tensor_inventory_on_meta(self):
        with torch.device("meta"):
            model = F5ConditionalFlowMatcher(F5TTSArchitectureConfig())
        state = model.state_dict()
        self.assertEqual(len(state), F5TTS_V1_BASE_GRAPH_TENSOR_COUNT)
        self.assertEqual(
            sum(tensor.numel() for tensor in state.values()),
            F5TTS_V1_BASE_PARAMETER_COUNT,
        )
        self.assertEqual(
            tuple(state["transformer.text_embed.text_embed.weight"].shape),
            (2_546, 512),
        )
        self.assertEqual(
            tuple(state["transformer.rotary_embed.inv_freq"].shape),
            (32, ),
        )

    def test_rotary_embedding_uses_x_transformers_adjacent_pairs(self):
        rotary = RotaryEmbedding(4)
        frequencies, scale = rotary.forward_from_seq_len(2)
        hidden = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]])
        actual = apply_rotary_position_embedding(hidden, frequencies, scale)
        angle = frequencies[1]
        expected_last = (hidden[0, 0, 1] * angle.cos() + torch.tensor([-6.0, 5.0, -8.0, 7.0]) * angle.sin())
        self.assertTrue(torch.allclose(actual[0, 0, 0], hidden[0, 0, 0]))
        self.assertTrue(torch.allclose(actual[0, 0, 1], expected_last))

    def test_native_flow_objective_is_differentiable(self):
        model = F5ConditionalFlowMatcher(_tiny_config())
        mel = torch.randn(2, 20, 8)
        text = torch.tensor(
            [[2, 3, 4, 5, -1], [6, 7, 8, -1, -1]],
            dtype=torch.long,
        )
        loss, conditioning, prediction = model(
            mel,
            text,
            lens=torch.tensor((20, 17)),
        )
        loss.backward()
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(conditioning.shape, mel.shape)
        self.assertEqual(prediction.shape, mel.shape)
        self.assertIsNotNone(model.transformer.input_embed.proj.weight.grad)

    def test_training_arguments_enable_native_gradient_checkpointing(self):
        architecture = _tiny_config()
        flow = F5ConditionalFlowMatcher(architecture)
        runtime = NativeF5TTSRuntime(
            flow_model=flow,
            vocoder=None,
            frontend=NativeF5TextFrontend(_tiny_vocabulary()),
        )
        wrapper = F5TTSForTextToSpeech(
            F5TTSConfig(
                architecture=architecture.to_dict(),
                model_name=architecture.model_name,
                use_ema=False,
            ),
            device="cpu",
        )
        wrapper.model = runtime
        adapter = F5TTSTrainingAdapter(
            wrapper,
            get_training_spec("f5tts"),
        )
        record = {
            "mel": torch.randn(8, 20),
            "mel_lengths": torch.tensor(20),
            "input_ids": torch.tensor((2, 3, 4, 5)),
        }

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=wrapper,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    gradient_checkpointing=True,
                    use_cpu=True,
                ),
                train_dataset=[record],
                training_adapter=adapter,
            )
            trainer.train()

        self.assertTrue(flow.gradient_checkpointing)
        self.assertTrue(flow.transformer.checkpoint_activations)
        adapter.gradient_checkpointing_disable()
        self.assertFalse(flow.gradient_checkpointing)

    def test_native_sampler_is_seeded_and_preserves_reference_frames(self):
        model = F5ConditionalFlowMatcher(_tiny_config()).eval()
        reference = torch.randn(1, 7, 8)
        text = torch.tensor([[2, 3, 4, 5]])
        first, first_path = model.sample(
            reference,
            text,
            12,
            lengths=torch.tensor((7, )),
            steps=3,
            seed=31,
        )
        second, second_path = model.sample(
            reference,
            text,
            12,
            lengths=torch.tensor((7, )),
            steps=3,
            seed=31,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first_path, second_path))
        self.assertTrue(torch.equal(first[:, :7], reference))
        self.assertEqual(tuple(first_path.shape), (4, 1, 12, 8))

    def test_sampler_controller_compacts_steps_and_runs_stork2(self):
        model = F5ConditionalFlowMatcher(_tiny_config()).eval()
        reference = torch.randn(1, 7, 8)
        text = torch.tensor([[2, 3, 4, 5]])
        model.transformer.enable_diffusion_sampling({
            "target_steps": 2,
            "solver": "stork2",
            "stork_stages": 5,
        })

        sampled, trajectory = model.sample(
            reference,
            text,
            12,
            lengths=torch.tensor((7, )),
            steps=5,
            seed=31,
            use_epss=False,
        )
        stats = model.transformer.diffusion_sampling_stats()

        self.assertEqual(sampled.shape, (1, 12, 8))
        self.assertEqual(trajectory.shape[0], 3)
        self.assertEqual(stats["native_steps"], 5)
        self.assertEqual(stats["prepared_steps"], 2)
        self.assertEqual(stats["solver_steps"], 2)
        self.assertEqual(stats["solver_startup_steps"], 1)
        self.assertEqual(stats["solver_stabilized_steps"], 1)

    def test_compile_targets_keep_sampler_and_vocoder_eager_for_inference(self):

        class TinyVocoder(torch.nn.Module):

            def __init__(self, input_channels: int):
                super().__init__()
                self.backbone = SimpleNamespace(input_channels=input_channels)

            @staticmethod
            def decode(features):
                return features

        flow = F5ConditionalFlowMatcher(_tiny_config())
        vocoder = TinyVocoder(flow.num_channels)
        runtime = NativeF5TTSRuntime(
            flow_model=flow,
            vocoder=vocoder,
            frontend=NativeF5TextFrontend(_tiny_vocabulary()),
        )

        inference = runtime.optimization_compile_targets("inference")
        training = runtime.optimization_compile_targets("training")

        self.assertEqual(
            tuple(target.label for target in inference),
            ("flow_model.transformer.forward", ),
        )
        self.assertIs(inference[0].owner, flow.transformer)
        self.assertEqual(inference[0].attribute, "forward")
        self.assertNotIn(flow, tuple(target.owner for target in inference))
        self.assertNotIn(vocoder, tuple(target.owner for target in inference))
        self.assertEqual(
            tuple(target.label for target in training),
            ("flow_model.forward", ),
        )
        self.assertIs(training[0].owner, flow)
        self.assertEqual(training[0].attribute, "forward")
        with self.assertRaisesRegex(ValueError, "Unsupported optimization mode"):
            runtime.optimization_compile_targets("export")

    def test_reduced_precision_inference_fails_closed(self):

        class TinyVocoder(torch.nn.Module):

            def __init__(self, input_channels: int):
                super().__init__()
                self.backbone = SimpleNamespace(input_channels=input_channels)
                self.projection = torch.nn.Conv1d(
                    input_channels,
                    input_channels,
                    kernel_size=1,
                )

            def decode(self, features):
                return self.projection(features)

        for flow_dtype in (torch.float16, torch.bfloat16):
            for vocoder_dtype in (torch.float32, flow_dtype):
                with self.subTest(
                        flow_dtype=flow_dtype,
                        vocoder_dtype=vocoder_dtype,
                ):
                    flow = F5ConditionalFlowMatcher(_tiny_config()).to(dtype=flow_dtype)
                    vocoder = TinyVocoder(flow.num_channels).to(dtype=vocoder_dtype)
                    runtime = NativeF5TTSRuntime(
                        flow_model=flow,
                        vocoder=vocoder,
                        frontend=NativeF5TextFrontend(_tiny_vocabulary()),
                    )

                    with self.assertRaisesRegex(
                            RuntimeError,
                            "reduced-precision inference is disabled.*DiT",
                    ):
                        runtime.prepare_for_inference()

    def test_reduced_precision_requires_explicit_quality_acknowledgement(self):

        class TinyVocoder(torch.nn.Module):

            def __init__(self, input_channels: int):
                super().__init__()
                self.backbone = SimpleNamespace(input_channels=input_channels)
                self.projection = torch.nn.Conv1d(
                    input_channels,
                    input_channels,
                    kernel_size=1,
                )

            def decode(self, features):
                return self.projection(features)

        flow = F5ConditionalFlowMatcher(_tiny_config()).bfloat16()
        runtime = NativeF5TTSRuntime(
            flow_model=flow,
            vocoder=TinyVocoder(flow.num_channels).bfloat16(),
            frontend=NativeF5TextFrontend(_tiny_vocabulary()),
            allow_unvalidated_reduced_precision_inference=True,
        )

        runtime.prepare_for_inference()

        self.assertFalse(runtime.training)
        self.assertFalse(runtime.ema_model.training)
        self.assertFalse(runtime.vocoder.training)

    def test_reduced_precision_acknowledgement_is_boolean(self):
        with self.assertRaisesRegex(
                TypeError,
                "allow_unvalidated_reduced_precision_inference.*boolean",
        ):
            F5TTSConfig(allow_unvalidated_reduced_precision_inference="yes", )

    def test_reduced_precision_fails_before_checkpoint_resolution(self):
        model = F5TTSForTextToSpeech(
            F5TTSConfig(torch_dtype="float16"),
            device="cpu",
            lazy_load=True,
        )

        with (
                patch("voicehub.models.f5tts.modeling.resolve_f5tts_artifacts", ) as resolver,
                patch(
                    "voicehub.models.f5tts.modeling.resolve_torch_dtype",
                    return_value=torch.float16,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "reduced-precision inference is disabled",
                ),
        ):
            model.load()

        resolver.assert_not_called()
        self.assertIsNone(model.model)
        self.assertIsNone(model.artifacts)

    def test_checkpoint_round_trip_is_strict_and_prefix_compatible(self):
        source = F5ConditionalFlowMatcher(_tiny_config())
        target = F5ConditionalFlowMatcher(_tiny_config())
        with torch.no_grad():
            for index, parameter in enumerate(source.parameters(), start=1):
                parameter.fill_((index % 7) / 10)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            export_f5tts_checkpoint(source, checkpoint)
            report = load_f5tts_checkpoint(target, checkpoint)
            self.assertEqual(report.prefix, "ema_model.")
            self.assertEqual(report.tensor_count, len(source.state_dict()))
            for name, tensor in source.state_dict().items():
                self.assertTrue(torch.equal(tensor, target.state_dict()[name]))

            incompatible = F5ConditionalFlowMatcher(
                F5TTSArchitectureConfig(**{
                    **_tiny_config().to_dict(),
                    "text_num_embeds": 13,
                }))
            with self.assertRaisesRegex(ValueError, "shape"):
                load_f5tts_checkpoint(incompatible, checkpoint)

    def test_f5_checkpoint_accepts_hugging_face_snapshot_symlink(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        with torch.no_grad():
            target.weight.zero_()
            target.bias.zero_()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blobs = root / "blobs"
            snapshot = root / "snapshots" / "revision"
            blobs.mkdir()
            snapshot.mkdir(parents=True)
            blob = blobs / ("a" * 64)
            save_safetensors(
                {
                    f"ema_model.{name}": tensor
                    for name, tensor in source.state_dict().items()
                },
                blob,
            )
            checkpoint = snapshot / "model.safetensors"
            checkpoint.symlink_to(Path("../../blobs") / blob.name)

            report = load_f5tts_checkpoint(target, checkpoint)

            self.assertEqual(report.path, blob.resolve())
            self.assertEqual(report.prefix, "ema_model.")
            for name, tensor in source.state_dict().items():
                self.assertTrue(torch.equal(tensor, target.state_dict()[name]))

            unsafe_alias = snapshot / "model.bin"
            unsafe_alias.symlink_to(Path("../../blobs") / blob.name)
            with self.assertRaisesRegex(ValueError, "Safetensors only"):
                load_f5tts_checkpoint(target, unsafe_alias)

    def test_legacy_conversion_is_weights_only(self):
        model = F5ConditionalFlowMatcher(_tiny_config())
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.pt"
            native = Path(directory) / "model.safetensors"
            torch.save(
                {"state_dict": {
                    f"ema_model.{name}": value
                    for name, value in model.state_dict().items()
                }},
                legacy,
            )
            with patch.object(torch, "load", wraps=torch.load) as load:
                convert_legacy_f5tts_checkpoint(legacy, native)
            self.assertTrue(load.call_args.kwargs["weights_only"])
            restored = F5ConditionalFlowMatcher(_tiny_config())
            load_f5tts_checkpoint(restored, native)

    def test_vocos_namespace_and_decoder_are_native(self):
        with torch.device("meta"):
            vocoder = NativeVocos()
        state = vocoder.state_dict()
        self.assertEqual(len(state), 83)
        self.assertEqual(
            tuple(state["feature_extractor.mel_spec.mel_scale.fb"].shape),
            (513, 100),
        )
        self.assertEqual(
            tuple(state["head.out.weight"].shape),
            (1_026, 512),
        )
        actual = NativeVocos()
        waveform = actual.decode(torch.randn(1, 100, 8))
        self.assertEqual(tuple(waveform.shape), (1, 1_792))
        self.assertTrue(torch.isfinite(waveform).all())

    def test_vocos_checkpoint_accepts_hugging_face_snapshot_symlink(self):
        source = torch.nn.Linear(4, 3)
        target = torch.nn.Linear(4, 3)
        with torch.no_grad():
            target.weight.zero_()
            target.bias.zero_()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blobs = root / "blobs"
            snapshot = root / "snapshots" / "revision"
            blobs.mkdir()
            snapshot.mkdir(parents=True)
            blob = blobs / ("b" * 64)
            save_safetensors(source.state_dict(), blob)
            checkpoint = snapshot / "vocos.safetensors"
            checkpoint.symlink_to(Path("../../blobs") / blob.name)

            report = load_vocos_checkpoint(target, checkpoint)

            self.assertEqual(report.path, blob.resolve())
            for name, tensor in source.state_dict().items():
                self.assertTrue(torch.equal(tensor, target.state_dict()[name]))

            unsafe_alias = snapshot / "vocos.bin"
            unsafe_alias.symlink_to(Path("../../blobs") / blob.name)
            with self.assertRaisesRegex(ValueError, "converted Safetensors only"):
                load_vocos_checkpoint(target, unsafe_alias)

    def test_frontend_requires_explicit_chinese_normalization(self):
        frontend = NativeF5TextFrontend(_tiny_vocabulary())
        self.assertEqual(frontend.encode("abc.").tolist(), [2, 3, 4, 1])
        with self.assertRaisesRegex(ValueError, "pinyin-with-tone"):
            frontend.encode("你好")
        normalized = NativeF5TextFrontend(
            _tiny_vocabulary(),
            normalizer=lambda text: ("a", "b"),
        )
        self.assertEqual(normalized.encode("你好").tolist(), [2, 3])

    def test_ema_training_export_reloads_into_a_fresh_flow_graph(self):
        architecture = _tiny_config()
        flow = F5ConditionalFlowMatcher(architecture)
        runtime = NativeF5TTSRuntime(
            flow_model=flow,
            vocoder=None,
            frontend=NativeF5TextFrontend(_tiny_vocabulary()),
        )
        wrapper = F5TTSForTextToSpeech(
            F5TTSConfig(
                architecture=architecture.to_dict(),
                model_name=architecture.model_name,
            ),
            device="cpu",
        )
        wrapper.model = runtime
        adapter = F5TTSTrainingAdapter(
            wrapper,
            get_training_spec("f5tts"),
        ).setup()
        with torch.no_grad():
            next(flow.parameters()).add_(0.25)
        adapter.on_optimizer_step(optimizer_names=("model", ), step=1)

        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            self.assertTrue((Path(directory) / "config.json").is_file())
            self.assertTrue((Path(directory) / "vocab.txt").is_file())
            fresh = F5ConditionalFlowMatcher(architecture)
            load_f5tts_checkpoint(
                fresh,
                Path(directory) / "model.safetensors",
            )
            shadow = adapter.recipe_state_dict()["ema"]["shadow"]
            for name, tensor in fresh.state_dict().items():
                expected = shadow.get(name, flow.state_dict()[name])
                self.assertTrue(torch.equal(tensor, expected))

    def test_ema_disabled_exports_explicit_raw_weights_without_recipe_state(self):
        architecture = _tiny_config()
        flow = F5ConditionalFlowMatcher(architecture)
        runtime = NativeF5TTSRuntime(
            flow_model=flow,
            vocoder=None,
            frontend=NativeF5TextFrontend(_tiny_vocabulary()),
        )
        wrapper = F5TTSForTextToSpeech(
            F5TTSConfig(
                architecture=architecture.to_dict(),
                model_name=architecture.model_name,
                use_ema=False,
            ),
            device="cpu",
        )
        wrapper.model = runtime
        adapter = F5TTSTrainingAdapter(
            wrapper,
            get_training_spec("f5tts"),
        ).setup()
        with torch.no_grad():
            next(flow.parameters()).add_(0.25)
        raw_state = {name: tensor.detach().clone() for name, tensor in flow.state_dict().items()}

        adapter.on_optimizer_step(optimizer_names=("model", ), step=1)

        self.assertIsNone(adapter._ema)
        self.assertEqual(adapter.recipe_state_dict(), {})
        self.assertFalse(adapter.recipe_resume_configuration()["resolved_use_ema"], )
        with self.assertRaisesRegex(ValueError, "use_ema=False"):
            adapter.load_recipe_state_dict({"ema": {}}, strict=True)

        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            configuration = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
            self.assertFalse(configuration["use_ema"])
            fresh = F5ConditionalFlowMatcher(architecture)
            report = load_f5tts_checkpoint(
                fresh,
                Path(directory) / "model.safetensors",
                use_ema=False,
            )

        self.assertEqual(report.prefix, "")
        for name, tensor in fresh.state_dict().items():
            self.assertTrue(torch.equal(tensor, raw_state[name]))

    def test_mel_frontend_is_pure_torch_and_has_expected_shape(self):
        frontend = F5MelSpectrogram(
            sample_rate=8_000,
            n_fft=32,
            hop_length=8,
            win_length=32,
            n_mels=8,
        )
        waveform = torch.randn(2, 128)
        mel = frontend(waveform)
        self.assertEqual(mel.shape[:2], (2, 8))
        self.assertTrue(torch.isfinite(mel).all())

        reduced_precision = frontend(waveform.bfloat16())
        self.assertEqual(reduced_precision.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(reduced_precision).all())
        self.assertTrue(torch.allclose(
            reduced_precision.float(),
            mel,
            atol=0.08,
            rtol=0.02,
        ))

    def test_vocoder_istft_boundary_promotes_bfloat16_to_float32(self):
        head = ISTFTHead(dim=8, n_fft=16, hop_length=4).bfloat16()
        waveform = head(torch.randn(2, 12, 8, dtype=torch.bfloat16))
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertEqual(tuple(waveform.shape), (2, 44))
        self.assertTrue(torch.isfinite(waveform).all())


if __name__ == "__main__":
    unittest.main()
