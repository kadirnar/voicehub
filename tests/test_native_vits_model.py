from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from voicehub.models.vits.native.checkpoint import (
    FACEBOOK_MMS_TTS_ENG_HEADER_FINGERPRINT,
    FACEBOOK_MMS_TTS_ENG_REVISION,
    HuggingFaceVitsCheckpointAdapter,
    native_vits_tensor_shapes,
    safetensors_header_fingerprint,
)
from voicehub.models.vits.native.configuration import VitsConfig
from voicehub.models.vits.native.frontend import VitsFrontendCapabilityError, VitsFrontendConfig, VitsTokenizer
from voicehub.models.vits.native.registration import create_vits_architecture_spec, register_vits_architecture
from voicehub.runtime.registry import ArchitectureRegistry
from voicehub.tasks import SpeechTask

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def _tiny_config(*, stochastic: bool = False) -> VitsConfig:
    return VitsConfig(
        vocab_size=8,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        window_size=2,
        ffn_dim=16,
        ffn_kernel_size=3,
        flow_size=8,
        spectrogram_bins=5,
        layerdrop=0.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        use_stochastic_duration_prediction=stochastic,
        upsample_initial_channel=16,
        upsample_rates=(2, ),
        upsample_kernel_sizes=(4, ),
        resblock_kernel_sizes=(3, ),
        resblock_dilation_sizes=((1, ), ),
        depth_separable_channels=2,
        depth_separable_num_layers=1,
        duration_predictor_flow_bins=4,
        duration_predictor_tail_bound=2.0,
        duration_predictor_kernel_size=3,
        duration_predictor_dropout=0.0,
        duration_predictor_num_flows=2,
        duration_predictor_filter_channels=8,
        prior_encoder_num_flows=2,
        prior_encoder_num_wavenet_layers=2,
        posterior_encoder_num_wavenet_layers=2,
        wavenet_kernel_size=3,
        wavenet_dilation_rate=2,
        wavenet_dropout=0.0,
        pad_token_id=0,
    )


class NativeVitsDeclarationTests(unittest.TestCase):

    def test_registration_is_lazy_and_capability_boundary_is_explicit(self):
        script = """
import sys
from voicehub.models.vits.native.registration import create_vits_architecture_spec
spec = create_vits_architecture_spec()
print("voicehub.models.vits.native.modeling" in sys.modules)
print(spec.metadata["full_finetuning_ready"])
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.splitlines(), ["False", "True"])

        spec = create_vits_architecture_spec()
        self.assertEqual(spec.architecture_id, "vits")
        self.assertTrue(spec.supports_task(SpeechTask.TEXT_TO_SPEECH))
        self.assertTrue(spec.capabilities.training)
        self.assertIn(
            "fine-tuning-requires-explicit-acoustic-config",
            spec.capabilities.features,
        )
        self.assertIn(
            "full-adversarial-fine-tuning",
            spec.capabilities.features,
        )
        self.assertIn(
            "torch-compile-inference-unsafe",
            spec.capabilities.features,
        )
        self.assertEqual(
            spec.capabilities.optimization_passes,
            ("compile", "custom-kernels"),
        )
        registry = ArchitectureRegistry()
        register_vits_architecture(registry=registry)
        self.assertIs(registry.get("mms-tts"), registry.get("vits"))

    def test_vits_modules_do_not_import_architecture_frameworks(self):
        package = Path(__file__).parents[1] / 'voicehub/models/vits/native'
        allowed_roots = {
            "__future__",
            "collections",
            "copy",
            "dataclasses",
            "functools",
            "hashlib",
            "importlib",
            "json",
            "math",
            "numbers",
            "operator",
            "pathlib",
            "re",
            "torch",
            "types",
            "typing",
            "voicehub",
        }
        violations = []
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = [alias.name.partition(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots = [node.module.partition(".")[0]]
                else:
                    continue
                violations.extend(
                    f"{path.name}:{node.lineno}:{root}" for root in roots if root not in allowed_roots)
        self.assertEqual(violations, [])

    def test_source_provenance_pins_training_and_checkpoint_revisions(self):
        source = json.loads((Path(__file__).parents[1] /
                             'voicehub/models/vits/native/SOURCE.json').read_text(encoding="utf-8"))
        self.assertEqual(
            source["sources"][0]["revision"],
            "2e561ba58618d021b5b8323d3765880f7e0ecfdb",
        )
        self.assertEqual(
            source["reference_checkpoint"]["revision"],
            FACEBOOK_MMS_TTS_ENG_REVISION,
        )
        self.assertIn(
            "requires training_acoustic_config explicitly",
            source["implementation"]["training_boundary"],
        )


class VitsConfigurationTests(unittest.TestCase):

    def test_defaults_match_mms_tts_generator(self):
        config = VitsConfig()
        self.assertEqual(config.vocab_size, 38)
        self.assertEqual(config.hidden_size, 192)
        self.assertEqual(config.num_hidden_layers, 6)
        self.assertEqual(config.num_attention_heads, 2)
        self.assertEqual(config.flow_size, 192)
        self.assertEqual(config.spectrogram_bins, 513)
        self.assertEqual(config.upsample_rates, (8, 8, 2, 2))
        self.assertEqual(config.upsample_factor, 256)
        self.assertEqual(config.fft_size, 1024)
        self.assertTrue(config.use_stochastic_duration_prediction)

    def test_huggingface_config_round_trip_preserves_unknown_metadata(self):
        config = VitsConfig.from_dict({
            "model_type": "vits",
            "vocab_size": 40,
            "torch_dtype": "float32",
            "noise_scale": 0,
        }, )
        payload = config.to_dict()
        self.assertEqual(payload["vocab_size"], 40)
        self.assertEqual(payload["torch_dtype"], "float32")
        self.assertEqual(payload["noise_scale"], 0.0)
        self.assertEqual(payload["model_type"], "vits")

    def test_configuration_rejects_invalid_architecture_geometry(self):
        with self.assertRaisesRegex(ValueError, "flow_size"):
            VitsConfig(flow_size=191)
        with self.assertRaisesRegex(ValueError, "halving"):
            VitsConfig(
                upsample_initial_channel=8,
                upsample_rates=(2, 2, 2, 2),
                upsample_kernel_sizes=(4, 4, 4, 4),
            )
        with self.assertRaisesRegex(ValueError, "spectrogram_bins"):
            VitsConfig(spectrogram_bins=1)
        with self.assertRaisesRegex(ValueError, "speaker_embedding_size"):
            VitsConfig(num_speakers=2, speaker_embedding_size=0)


class VitsFrontendTests(unittest.TestCase):

    def test_mms_character_frontend_adds_blank_token_zero(self):
        tokenizer = VitsTokenizer(
            {
                "k": 0,
                "a": 1,
                "b": 2,
                " ": 3
            },
            config=VitsFrontendConfig(
                language="eng",
                add_blank=True,
                normalize=True,
                phonemize=False,
                pad_token="k",
            ),
        )
        encoded = tokenizer.encode("A b!")
        self.assertEqual(encoded.input_ids, (0, 1, 0, 3, 0, 2, 0))
        self.assertEqual(tokenizer.decode(encoded), "a b")
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertIsNone(tokenizer.unk_token_id)
        self.assertEqual(tokenizer.encode("").input_ids, ())
        self.assertEqual(tokenizer.encode("ka").input_ids, (0, 0, 1, 0))

    def test_required_language_provider_is_never_imported_implicitly(self):
        tokenizer = VitsTokenizer(
            {
                "_": 0,
                "a": 1
            },
            config=VitsFrontendConfig(
                add_blank=True,
                normalize=False,
                phonemize=True,
                pad_token="_",
            ),
        )
        with self.assertRaisesRegex(
                VitsFrontendCapabilityError,
                "TextPhonemizer",
        ):
            tokenizer.encode("a")

    def test_declarative_frontend_assets_load_without_upstream_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab = root / "vocab.json"
            metadata = root / "tokenizer_config.json"
            vocab.write_text(
                json.dumps({
                    "k": 0,
                    "x": 1
                }),
                encoding="utf-8",
            )
            metadata.write_text(
                json.dumps({
                    "language": "eng",
                    "add_blank": True,
                    "normalize": True,
                    "phonemize": False,
                    "pad_token": "k",
                }, ),
                encoding="utf-8",
            )
            tokenizer = VitsTokenizer.from_files(
                vocab,
                tokenizer_config_file=metadata,
            )
        self.assertEqual(tokenizer.encode("X").input_ids, (0, 1, 0))

    def test_frontend_assets_reject_duplicate_json_keys(self):
        from voicehub.models.vits.native.frontend import VitsFrontendAssetError

        with tempfile.TemporaryDirectory() as directory:
            vocab = Path(directory) / "vocab.json"
            vocab.write_text('{"k": 0, "k": 1}', encoding="utf-8")
            with self.assertRaisesRegex(
                    VitsFrontendAssetError,
                    "repeats key",
            ):
                VitsTokenizer.from_files(vocab)


class VitsCheckpointTests(unittest.TestCase):

    def test_real_mms_tts_header_inventory_is_fully_covered(self):
        shapes = native_vits_tensor_shapes(VitsConfig())
        self.assertEqual(len(shapes), 762)
        self.assertEqual(
            safetensors_header_fingerprint(shapes),
            FACEBOOK_MMS_TTS_ENG_HEADER_FINGERPRINT,
        )
        self.assertEqual(
            FACEBOOK_MMS_TTS_ENG_REVISION,
            "c71de0fe7204c83f1c10820a7d696d0b450048ba",
        )

    def test_huggingface_adapter_has_a_strict_identity_plan(self):
        config = VitsConfig().to_dict()
        adapter = HuggingFaceVitsCheckpointAdapter()
        self.assertTrue(adapter.probe((Path("model.safetensors"), ), config), )
        plan = adapter.tensor_plan(config)
        names = frozenset(native_vits_tensor_shapes(config))
        self.assertEqual(plan.source_names, names)
        self.assertEqual(plan.target_names, names)
        self.assertEqual(plan.ignored_source_patterns, ())

    @unittest.skipUnless(TORCH_AVAILABLE, "native VITS requires PyTorch")
    def test_declared_shapes_match_the_executable_tiny_graph(self):
        from voicehub.models.vits.native.modeling import VitsModel

        config = _tiny_config(stochastic=True)
        actual = {name: tuple(tensor.shape) for name, tensor in VitsModel(config).state_dict().items()}
        self.assertEqual(native_vits_tensor_shapes(config), actual)


@unittest.skipUnless(TORCH_AVAILABLE, "native VITS requires PyTorch")
class VitsWaveNetKernelTests(unittest.TestCase):

    def test_torch_backend_matches_the_original_gated_activation(self):
        from voicehub.kernels import KernelBackend
        from voicehub.models.vits.native.modeling import VitsWaveNet

        torch.manual_seed(19)
        wavenet = VitsWaveNet(_tiny_config(), num_layers=2).eval()
        inputs = torch.randn(2, 8, 5)
        padding_mask = torch.tensor([
            [[1.0, 1.0, 1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0, 1.0, 0.0]],
        ])

        expected_inputs = inputs
        expected_outputs = torch.zeros_like(inputs)
        for index, (input_layer, output_layer) in enumerate(zip(wavenet.in_layers, wavenet.res_skip_layers)):
            hidden = input_layer(expected_inputs)
            activation = torch.tanh(hidden[:, :wavenet.hidden_size])
            gate = torch.sigmoid(hidden[:, wavenet.hidden_size:])
            residual_skip = output_layer(wavenet.dropout(activation * gate))
            if index < wavenet.num_layers - 1:
                expected_inputs = (expected_inputs + residual_skip[:, :wavenet.hidden_size]) * padding_mask
                expected_outputs = (expected_outputs + residual_skip[:, wavenet.hidden_size:])
            else:
                expected_outputs = expected_outputs + residual_skip
        expected = expected_outputs * padding_mask

        self.assertIs(wavenet.kernel_backend, KernelBackend.TORCH)
        torch.testing.assert_close(
            wavenet(inputs, padding_mask),
            expected,
            rtol=0,
            atol=0,
        )

    def test_backend_selection_does_not_change_checkpoint_keys(self):
        from voicehub.kernels import KernelBackend
        from voicehub.models.vits.native.modeling import VitsWaveNet

        wavenet = VitsWaveNet(_tiny_config(), num_layers=2)
        before = {name: tensor.detach().clone() for name, tensor in wavenet.state_dict().items()}

        self.assertIsNone(wavenet.set_kernel_backend("triton"))
        self.assertIs(wavenet.kernel_backend, KernelBackend.TRITON)
        self.assertEqual(tuple(wavenet.state_dict()), tuple(before))
        for name, expected in before.items():
            torch.testing.assert_close(wavenet.state_dict()[name], expected)

        with self.assertRaisesRegex(ValueError, "Unknown kernel backend"):
            wavenet.set_kernel_backend("not-a-kernel")


@unittest.skipUnless(TORCH_AVAILABLE, "native VITS requires PyTorch")
class VitsWeightNormCacheTests(unittest.TestCase):

    def test_inference_cache_is_exact_nonpersistent_and_reversible(self):
        from voicehub.models.vits.native.modeling import VitsModel, VitsSamplingConfig, WeightNormalizedConv1d

        torch.manual_seed(29)
        model = VitsModel(_tiny_config()).eval()
        input_ids = torch.tensor([[1, 2, 3]])
        sampling = VitsSamplingConfig(
            seed=41,
            noise_scale=0.3,
            noise_scale_duration=0.0,
            max_output_frames=50,
        )
        state_keys = tuple(model.state_dict())
        with torch.inference_mode():
            expected = model.synthesize(input_ids, sampling=sampling)

        cached_bytes = model.cache_weight_norm_for_inference()
        cached_modules = [module for module in model.modules() if isinstance(module, WeightNormalizedConv1d)]
        self.assertGreater(cached_bytes, 0)
        self.assertTrue(cached_modules)
        self.assertTrue(all(module._inference_weight is not None for module in cached_modules))
        self.assertEqual(tuple(model.state_dict()), state_keys)
        self.assertFalse(any("_inference_weight" in key for key in model.state_dict()))

        with torch.inference_mode():
            actual = model.synthesize(input_ids, sampling=sampling)
        self.assertTrue(torch.equal(actual.waveform, expected.waveform))
        self.assertTrue(torch.equal(actual.durations, expected.durations))

        model.train()
        self.assertTrue(all(module._inference_weight is None for module in cached_modules))
        with self.assertRaisesRegex(RuntimeError, "requires eval mode"):
            model.cache_weight_norm_for_inference()

    def test_cache_invalidates_after_parameter_or_dtype_mutation(self):
        from voicehub.models.vits.native.modeling import VitsModel, WeightNormalizedConv1d

        model = VitsModel(_tiny_config()).eval()
        model.cache_weight_norm_for_inference()
        layer = next(module for module in model.modules() if isinstance(module, WeightNormalizedConv1d))
        cached = layer._inference_weight
        self.assertIsNotNone(cached)
        with torch.no_grad():
            layer.weight_v.add_(0.125)
        with torch.inference_mode():
            materialized = layer.normalized_weight()
        self.assertIsNone(layer._inference_weight)
        self.assertFalse(torch.equal(materialized, cached))

        model.cache_weight_norm_for_inference()
        cached = layer._inference_weight
        layer.weight_v = torch.nn.Parameter(layer.weight_v.detach().clone().add_(0.125))
        with torch.inference_mode():
            materialized = layer.normalized_weight()
        self.assertIsNone(layer._inference_weight)
        self.assertFalse(torch.equal(materialized, cached))

        model.cache_weight_norm_for_inference()
        model.to(dtype=torch.float64)
        self.assertTrue(
            all(
                module._inference_weight is None for module in model.modules()
                if isinstance(module, WeightNormalizedConv1d)))

    def test_cache_never_changes_gradient_semantics(self):
        from voicehub.models.vits.native.modeling import WeightNormalizedConv1d

        layer = WeightNormalizedConv1d(2, 3, 3, padding=1).eval()
        layer.cache_weight_norm_for_inference()
        inputs = torch.randn(1, 2, 5, requires_grad=True)
        layer(inputs).sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(layer.weight_g.grad)
        self.assertIsNotNone(layer.weight_v.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "native VITS requires PyTorch")
class VitsAlignmentTests(unittest.TestCase):

    def test_duration_expansion_and_monotonic_search_agree(self):
        from voicehub.models.vits.native.alignment import generate_path, maximum_path

        durations = torch.tensor([[[2.0, 1.0, 2.0]]])
        mask = torch.ones(1, 1, 5, 3)
        expected = torch.tensor([[
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]], )
        self.assertTrue(torch.equal(generate_path(durations, mask).squeeze(1), expected), )
        self.assertTrue(
            torch.equal(
                generate_path(durations.long(), mask.bool()).squeeze(1),
                expected.bool(),
            ), )
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            generate_path(torch.tensor([[[2.0, 0.0, 3.0]]]), mask)
        scores = torch.where(
            expected.bool(),
            torch.tensor(8.0),
            torch.tensor(-8.0),
        )
        self.assertTrue(torch.equal(maximum_path(scores, mask[:, 0]), expected), )

    def test_spline_duration_flow_is_numerically_reversible(self):
        from voicehub.models.vits.native.modeling import VitsConvFlow

        torch.manual_seed(4)
        flow = VitsConvFlow(_tiny_config(stochastic=True)).eval()
        value = torch.randn(2, 2, 4) * 0.5
        mask = torch.ones(2, 1, 4)
        condition = torch.randn(2, 8, 4)
        transformed, logdet = flow(value, mask, condition)
        restored, reverse_logdet = flow(
            transformed,
            mask,
            condition,
            reverse=True,
        )
        self.assertTrue(torch.allclose(restored, value, atol=1e-5, rtol=1e-5))
        self.assertTrue(
            torch.allclose(
                logdet + reverse_logdet,
                torch.zeros_like(logdet),
                atol=1e-5,
                rtol=1e-5,
            ), )


@unittest.skipUnless(TORCH_AVAILABLE, "native VITS requires PyTorch")
class VitsRuntimeTests(unittest.TestCase):

    def test_request_seed_is_repeatable_and_preserves_global_rng(self):
        from voicehub.models.vits.native.modeling import VitsModel, VitsSamplingConfig

        torch.manual_seed(91)
        model = VitsModel(_tiny_config()).eval()
        input_ids = torch.tensor([[1, 2, 3]])
        state = torch.random.get_rng_state().clone()
        first = model.synthesize(
            input_ids,
            sampling=VitsSamplingConfig(
                seed=7,
                noise_scale=0.3,
                noise_scale_duration=0.0,
                max_output_frames=50,
            ),
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        second = model.synthesize(
            input_ids,
            sampling=VitsSamplingConfig(
                seed=7,
                noise_scale=0.3,
                noise_scale_duration=0.0,
                max_output_frames=50,
            ),
        )
        third = model.synthesize(
            input_ids,
            sampling=VitsSamplingConfig(
                seed=8,
                noise_scale=0.3,
                noise_scale_duration=0.0,
                max_output_frames=50,
            ),
        )
        self.assertTrue(torch.equal(first.waveform, second.waveform))
        self.assertFalse(torch.equal(first.waveform, third.waveform))
        self.assertEqual(first.waveform.shape[0], 1)
        self.assertEqual(
            first.waveform.shape[1],
            first.sequence_lengths.item(),
        )
        self.assertEqual(
            first.sequence_lengths.item(),
            int(first.durations.sum().item()) * 2,
        )

    def test_supervised_generator_graph_runs_mas_and_backward(self):
        from voicehub.models.vits.native.losses import vits_kl_loss
        from voicehub.models.vits.native.modeling import VitsModel

        model = VitsModel(_tiny_config()).train()
        output = model(
            torch.tensor([[1, 2, 3]]),
            spectrogram=torch.randn(1, 5, 5),
            generator=torch.Generator().manual_seed(2),
        )
        self.assertEqual(output.alignment.shape, (1, 1, 5, 3))
        self.assertEqual(output.durations.sum().item(), 5.0)
        self.assertEqual(output.waveform.shape, (1, 10))
        torch.testing.assert_close(
            output.segment_start_frames,
            torch.zeros(1, dtype=torch.long),
        )
        kl = vits_kl_loss(
            output.prior_latents,
            output.posterior_log_variances,
            output.expanded_prior_means,
            output.expanded_prior_log_variances,
            output.spectrogram_mask,
        )
        loss = output.duration_loss + kl + output.waveform.square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.text_encoder.embed_tokens.weight.grad)
        self.assertIsNotNone(model.decoder.conv_post.weight.grad)

    def test_windowed_training_slices_latents_before_decoder_deterministically(self):
        from voicehub.models.vits.native.modeling import VitsModel

        model = VitsModel(_tiny_config()).train()
        spectrogram = torch.randn(2, 5, 7)
        spectrogram_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0],
        ])
        durations = torch.tensor([
            [[2.0, 2.0, 3.0]],
            [[2.0, 2.0, 2.0]],
        ])
        decoder_inputs = []
        handle = model.decoder.register_forward_pre_hook(
            lambda _module, inputs: decoder_inputs.append(inputs[0].detach().clone()))
        try:
            first = model(
                torch.tensor([
                    [1, 2, 3],
                    [1, 2, 3],
                ]),
                spectrogram=spectrogram,
                spectrogram_attention_mask=spectrogram_mask,
                durations=durations,
                segment_frames=4,
                generator=torch.Generator().manual_seed(17),
            )
            second = model(
                torch.tensor([
                    [1, 2, 3],
                    [1, 2, 3],
                ]),
                spectrogram=spectrogram,
                spectrogram_attention_mask=spectrogram_mask,
                durations=durations,
                segment_frames=4,
                generator=torch.Generator().manual_seed(17),
            )
        finally:
            handle.remove()

        self.assertEqual(
            [value.shape[-1] for value in decoder_inputs],
            [4, 4],
        )
        self.assertEqual(first.posterior_latents.shape[-1], 7)
        self.assertEqual(first.waveform.shape, (2, 8))
        torch.testing.assert_close(
            first.sequence_lengths,
            torch.tensor([8, 8]),
        )
        torch.testing.assert_close(
            first.segment_start_frames,
            second.segment_start_frames,
        )
        self.assertTrue((first.segment_start_frames >= 0).all())
        self.assertTrue((first.segment_start_frames + 4 <= torch.tensor([7, 6])).all())
        expected_decoder_input = torch.stack([
            first.posterior_latents[index, :, start:start + 4]
            for index, start in enumerate(first.segment_start_frames.tolist())
        ])
        torch.testing.assert_close(
            decoder_inputs[0],
            expected_decoder_input,
        )

    def test_stochastic_duration_training_graph_is_differentiable(self):
        from voicehub.models.vits.native.modeling import VitsModel

        model = VitsModel(_tiny_config(stochastic=True)).train()
        output = model(
            torch.tensor([[1, 2, 3]]),
            spectrogram=torch.randn(1, 5, 5),
            durations=torch.tensor([[[1.0, 2.0, 2.0]]]),
            generator=torch.Generator().manual_seed(3),
        )
        loss = output.duration_loss + output.waveform.square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.duration_predictor.conv_pre.weight.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "native VITS requires PyTorch")
class VitsLossTests(unittest.TestCase):

    def test_generator_objective_combines_every_source_term(self):
        from voicehub.models.vits.native.losses import VitsGeneratorLoss
        from voicehub.models.vits.native.modeling import VitsTrainingOutput

        latent = torch.zeros(1, 2, 3, requires_grad=True)
        generated_output = torch.zeros(1, 2, requires_grad=True)
        generated_feature = torch.ones(1, 2, requires_grad=True)
        output = VitsTrainingOutput(
            waveform=torch.zeros(1, 6),
            sequence_lengths=torch.tensor([6]),
            alignment=torch.ones(1, 1, 3, 1),
            durations=torch.tensor([[[3.0]]]),
            duration_loss=torch.tensor(2.0),
            posterior_latents=torch.zeros_like(latent),
            prior_latents=latent,
            expanded_prior_means=torch.zeros_like(latent),
            expanded_prior_log_variances=torch.zeros_like(latent),
            posterior_means=torch.zeros_like(latent),
            posterior_log_variances=torch.zeros_like(latent),
            text_mask=torch.ones(1, 1, 1),
            spectrogram_mask=torch.ones(1, 1, 3),
        )
        losses = VitsGeneratorLoss()(
            output,
            mel_reconstruction_loss=torch.tensor(0.5),
            generated_discriminator_outputs=(generated_output, ),
            real_feature_maps=((torch.zeros(1, 2), ), ),
            generated_feature_maps=((generated_feature, ), ),
        )
        self.assertTrue(torch.allclose(losses.total, torch.tensor(26.5)))
        losses.total.backward()
        self.assertIsNotNone(latent.grad)
        self.assertIsNotNone(generated_output.grad)
        self.assertIsNotNone(generated_feature.grad)

    def test_source_loss_equations_and_gradient_boundaries(self):
        from voicehub.models.vits.native.losses import (
            discriminator_loss,
            feature_matching_loss,
            generator_adversarial_loss,
            vits_kl_loss,
        )

        real = torch.tensor([[1.0, 0.0]], requires_grad=True)
        generated = torch.tensor([[0.5, -0.5]], requires_grad=True)
        disc, real_terms, generated_terms = discriminator_loss(
            (real, ),
            (generated, ),
        )
        self.assertTrue(torch.allclose(disc, torch.tensor(0.75)))
        self.assertEqual(len(real_terms), 1)
        self.assertEqual(len(generated_terms), 1)

        adversarial, terms = generator_adversarial_loss((generated, ))
        self.assertTrue(torch.allclose(adversarial, torch.tensor(1.25)))
        self.assertEqual(len(terms), 1)
        feature = feature_matching_loss(
            ((real, ), ),
            ((generated, ), ),
        )
        self.assertTrue(torch.allclose(feature, torch.tensor(1.0)))
        feature.backward(retain_graph=True)
        self.assertIsNone(real.grad)
        self.assertIsNotNone(generated.grad)

        zeros = torch.zeros(1, 2, 3, requires_grad=True)
        kl = vits_kl_loss(
            zeros,
            torch.zeros_like(zeros),
            torch.zeros_like(zeros),
            torch.zeros_like(zeros),
            torch.ones(1, 1, 3),
        )
        self.assertEqual(kl.item(), -1.0)

    def test_small_discriminators_execute_the_reference_topology(self):
        from voicehub.models.vits.native.losses import VitsPeriodDiscriminator, VitsScaleDiscriminator

        waveform = torch.randn(2, 1, 64, requires_grad=True)
        scale = VitsScaleDiscriminator(
            channels=(2, 4, 4, 4, 4, 4),
            groups=(1, 1, 1, 1, 1, 1),
        )
        period = VitsPeriodDiscriminator(
            3,
            channels=(2, 4, 4, 4, 4),
        )
        scale_output, scale_features = scale(waveform)
        period_output, period_features = period(waveform)
        self.assertEqual(len(scale_features), 7)
        self.assertEqual(len(period_features), 6)
        (scale_output.mean() + period_output.mean()).backward()
        self.assertIsNotNone(waveform.grad)

    def test_training_support_distinguishes_recipe_from_checkpoint_metadata(self):
        from voicehub.models.vits.native.losses import VITS_TRAINING_SUPPORT

        self.assertTrue(VITS_TRAINING_SUPPORT.differentiable_generator_graph)
        self.assertTrue(VITS_TRAINING_SUPPORT.monotonic_alignment_search)
        self.assertTrue(VITS_TRAINING_SUPPORT.discriminator_architecture)
        self.assertTrue(VITS_TRAINING_SUPPORT.source_acoustic_frontend)
        self.assertTrue(VITS_TRAINING_SUPPORT.adversarial_optimizer_phases)
        self.assertTrue(VITS_TRAINING_SUPPORT.random_discriminator_initialization)
        self.assertFalse(VITS_TRAINING_SUPPORT.checkpoint_discriminator_weights)
        self.assertFalse(VITS_TRAINING_SUPPORT.checkpoint_acoustic_frontend)
        self.assertTrue(VITS_TRAINING_SUPPORT.full_finetuning_ready)
        self.assertGreaterEqual(
            len(VITS_TRAINING_SUPPORT.blocking_requirements),
            2,
        )


@unittest.skipUnless(TORCH_AVAILABLE, "native VITS requires PyTorch")
class VitsAdversarialTrainingTests(unittest.TestCase):

    @staticmethod
    def _acoustic_config(**overrides):
        values = {
            "sampling_rate": 16_000,
            "filter_length": 8,
            "hop_length": 2,
            "win_length": 8,
            "num_mel_channels": 3,
            "mel_fmin": 0.0,
            "mel_fmax": 8_000.0,
            "segment_size": 8,
        }
        values.update(overrides)
        return values

    def test_acoustic_frontend_matches_its_spectrogram_projection(self):
        from voicehub.models.vits.native.training import VitsAcousticConfig, VitsAcousticFrontend

        config = VitsAcousticConfig.from_mapping(self._acoustic_config())
        frontend = VitsAcousticFrontend(config)
        waveform = torch.linspace(-0.8, 0.8, 16).unsqueeze(0)
        spectrogram = frontend.spectrogram(waveform)
        self.assertEqual(spectrogram.shape, (1, 5, 8))
        torch.testing.assert_close(
            frontend.mel_spectrogram(waveform),
            frontend.spectrogram_to_mel(spectrogram),
        )
        torch.testing.assert_close(
            frontend.spectrogram_lengths(torch.tensor([16])),
            torch.tensor([8]),
        )

    def test_acoustic_config_rejects_unverifiable_or_misaligned_values(self):
        from voicehub.models.vits.native.modeling import VitsModel
        from voicehub.models.vits.native.training import VitsAcousticConfig

        with self.assertRaisesRegex(ValueError, "incomplete"):
            VitsAcousticConfig.from_mapping({"sampling_rate": 16_000})
        with self.assertRaisesRegex(ValueError, "divisible"):
            VitsAcousticConfig.from_mapping(self._acoustic_config(segment_size=7), )
        with self.assertRaisesRegex(ValueError, "upsample factor"):
            VitsAcousticConfig.from_mapping(self._acoustic_config(hop_length=1), ).validate_model(
                VitsModel(_tiny_config()))

    def test_both_source_phases_are_differentiable(self):
        from torch import nn

        from voicehub.models.vits.native.modeling import VitsModel
        from voicehub.models.vits.native.training import VitsAdversarialTrainingModel

        class TinyDiscriminator(nn.Module):

            def __init__(self):
                super().__init__()
                self.hidden = nn.Conv1d(1, 2, 3, padding=1)
                self.output = nn.Conv1d(2, 1, 1)

            def _one(self, waveform):
                hidden = torch.nn.functional.leaky_relu(
                    self.hidden(waveform.unsqueeze(1)),
                    0.1,
                )
                output = self.output(hidden)
                return output.flatten(1), (hidden, output)

            def forward(self, real_waveform, generated_waveform):
                real_output, real_features = self._one(real_waveform)
                generated_output, generated_features = self._one(generated_waveform)
                return (
                    (real_output, ),
                    (generated_output, ),
                    (real_features, ),
                    (generated_features, ),
                )

        training_model = VitsAdversarialTrainingModel(
            VitsModel(_tiny_config()),
            self._acoustic_config(),
            discriminator=TinyDiscriminator(),
        )
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "audio_values": torch.randn(1, 16),
            "generator": torch.Generator().manual_seed(3),
        }

        discriminator_output = training_model.discriminator_step(**inputs)
        self.assertTrue(torch.isfinite(discriminator_output["loss"]))
        discriminator_output["loss"].backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in training_model.discriminator.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in training_model.native_model.parameters()))

        training_model.zero_grad(set_to_none=True)
        generator_output = training_model.generator_step(**inputs)
        self.assertEqual(
            set(generator_output["losses"]),
            {
                "adversarial_loss",
                "duration_loss",
                "feature_matching_loss",
                "generator_loss",
                "kl_loss",
                "mel_reconstruction_loss",
            },
        )
        self.assertTrue(torch.isfinite(generator_output["loss"]))
        generator_output["loss"].backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in training_model.native_model.parameters()))

    def test_windowed_training_aligns_real_waveform_and_mel_to_model_offsets(self):
        from torch import nn

        from voicehub.models.vits.native.modeling import VitsModel
        from voicehub.models.vits.native.training import VitsAdversarialTrainingModel

        training_model = VitsAdversarialTrainingModel(
            VitsModel(_tiny_config()),
            self._acoustic_config(),
            discriminator=nn.Identity(),
        )
        audio = torch.arange(32, dtype=torch.float32).reshape(2, 16)
        spectrogram = torch.linspace(
            0.1,
            1.0,
            steps=80,
        ).reshape(2, 5, 8)
        batch = training_model._training_batch(
            torch.tensor([
                [1, 2, 3],
                [1, 2, 3],
            ]),
            audio_values=audio,
            spectrogram=spectrogram,
            durations=torch.tensor([
                [[2.0, 3.0, 3.0]],
                [[3.0, 2.0, 3.0]],
            ]),
            generator=torch.Generator().manual_seed(23),
        )

        starts = batch.output.segment_start_frames.tolist()
        expected_real = torch.stack(
            [audio[index, start * 2:start * 2 + 8] for index, start in enumerate(starts)])
        target_mel = training_model.acoustic_frontend.spectrogram_to_mel(spectrogram, )
        expected_mel = torch.stack(
            [target_mel[index, :, start:start + 4] for index, start in enumerate(starts)])
        torch.testing.assert_close(batch.real_waveform, expected_real)
        torch.testing.assert_close(batch.target_mel, expected_mel)
        torch.testing.assert_close(
            batch.generated_waveform,
            batch.output.waveform,
        )
        self.assertEqual(batch.generated_waveform.shape, (2, 8))

    def test_adapter_routes_and_freezes_the_two_optimizer_phases(self):
        from torch import nn

        from voicehub.models.vits.configuration import VitsConfig as PublicVitsConfig
        from voicehub.models.vits.native.modeling import VitsModel
        from voicehub.models.vits.native.training import VitsAdversarialTrainingModel
        from voicehub.models.vits.training import NativeVitsGeneratorTrainingAdapter
        from voicehub.training.specs import get_training_spec

        class TinyDiscriminator(nn.Module):

            def __init__(self):
                super().__init__()
                self.hidden = nn.Conv1d(1, 2, 3, padding=1)
                self.output = nn.Conv1d(2, 1, 1)

            def _one(self, waveform):
                hidden = self.hidden(waveform.unsqueeze(1)).tanh()
                output = self.output(hidden)
                return output.flatten(1), (hidden, output)

            def forward(self, real_waveform, generated_waveform):
                real_output, real_features = self._one(real_waveform)
                generated_output, generated_features = self._one(generated_waveform)
                return (
                    (real_output, ),
                    (generated_output, ),
                    (real_features, ),
                    (generated_features, ),
                )

        class Wrapper:

            def __init__(self):
                self.config = PublicVitsConfig(
                    enable_native_adversarial_training=True,
                    training_acoustic_config=(VitsAdversarialTrainingTests._acoustic_config()),
                )
                self.model = VitsModel(_tiny_config())
                self.training_model = VitsAdversarialTrainingModel(
                    self.model,
                    self.config.training_acoustic_config,
                    discriminator=TinyDiscriminator(),
                )

            def load_for_training(self):
                return self

            @staticmethod
            def prepare_training_inputs(inputs, *, phase):
                if phase not in {"discriminator", "generator"}:
                    raise ValueError(phase)
                return dict(inputs)

        wrapper = Wrapper()
        adapter = NativeVitsGeneratorTrainingAdapter(
            wrapper,
            get_training_spec("vits"),
        )
        self.assertTrue(all(phase.optimizer_step_after_phase for phase in adapter.spec.phases))
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "audio_values": torch.randn(1, 16),
            "generator": torch.Generator().manual_seed(7),
        }
        discriminator = adapter(
            training_phase="discriminator",
            **inputs,
        )
        self.assertEqual(discriminator.optimizer_names, ("discriminator", ))
        discriminator.loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in wrapper.training_model.discriminator.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in wrapper.model.parameters()))

        wrapper.training_model.zero_grad(set_to_none=True)
        generator = adapter(
            training_phase="generator",
            **inputs,
        )
        self.assertEqual(generator.optimizer_names, ("generator", ))
        generator.loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in wrapper.model.parameters()))
        self.assertTrue(
            all(parameter.grad is None for parameter in wrapper.training_model.discriminator.parameters()))


if __name__ == "__main__":
    unittest.main()
