from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from voicehub.components.audio.codecs.base import (
    AudioCodec,
    DenseCodecCodes,
    RaggedCodecCodes,
    coerce_codec_codes,
    separate_audio_codec,
)
from voicehub.components.audio.codecs.dac.model.dac import DAC
from voicehub.kernels import KernelBackend, cute_dsl_capability
from voicehub.models.chatterbox.models.s3gen.s3gen import S3Token2Wav
from voicehub.models.chatterbox.models.s3tokenizer.model_v2 import S3TokenizerV2
from voicehub.models.cosyvoice.native.flow import CosyVoiceFlowMatchingModel
from voicehub.models.cosyvoice.native.vocoder import CosyVoiceHiFTGenerator
from voicehub.models.csm.source.moshi.models.compression import MimiModel
from voicehub.models.qwen3tts.native.codec import Qwen3TTSSpeechDecoder
from voicehub.models.voxcpm.native.codec import VoxCPMAudioVAE
from voicehub.models.xtts.source.TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder
from voicehub.optimization.capabilities import OptimizationContext
from voicehub.optimization.codecs import (
    CodecCompileComponent,
    CodecCUDAGraphCaptureError,
    CodecKernelBackend,
    CodecKernelPass,
    CodecOptimizationCompatibilityError,
    CodecOptimizationConfig,
    CodecOptimizationPolicy,
    capture_codec_cuda_graph,
    discover_codec_compile_targets,
    resolve_codec_optimization,
)


class _TinyCodec(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.quantizer = nn.Identity()
        self.decoder = nn.Linear(4, 4)

    def encode(self, audio):
        return self.quantizer(self.encoder(audio))

    def decode(self, encoded):
        return self.decoder(encoded)

    def forward(self, audio):
        return self.decode(self.encode(audio))


class _KernelBlock(nn.Module):

    supported_kernel_operations = ("audio.codec.test", )
    supported_codec_kernel_backends = (
        CodecKernelBackend.TORCH,
        CodecKernelBackend.TRITON,
        CodecKernelBackend.CUDA_EXTENSION,
    )

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.kernel_backend = KernelBackend.TORCH

    def set_kernel_backend(self, backend):
        self.kernel_backend = KernelBackend.coerce(backend)

    def resolve_kernel_backend(self, backend, *, device, dtype):
        del device, dtype
        backend = KernelBackend.coerce(backend)
        return KernelBackend.TORCH if backend is KernelBackend.AUTO else backend

    @property
    def codec_kernel_backend(self):
        return CodecKernelBackend.coerce(self.kernel_backend)

    def set_codec_kernel_backend(self, backend):
        backend = CodecKernelBackend.coerce(backend)
        self.set_kernel_backend(backend.generic_backend())

    def resolve_codec_kernel_backend(self, backend, *, device, dtype):
        backend = CodecKernelBackend.coerce(backend)
        return CodecKernelBackend.coerce(
            self.resolve_kernel_backend(
                backend.generic_backend(),
                device=device,
                dtype=dtype,
            ))

    def forward(self, value):
        return value * self.weight


class _KernelCodec(_TinyCodec):

    def __init__(self) -> None:
        super().__init__()
        self.kernel_block = _KernelBlock()


class _NonCodecKernelBlock(nn.Module):

    supported_kernel_operations = ("tts.diffusion.test", )

    def __init__(self) -> None:
        super().__init__()
        self.kernel_backend = KernelBackend.AUTO

    def set_kernel_backend(self, backend):
        self.kernel_backend = KernelBackend.coerce(backend)

    def resolve_kernel_backend(self, backend, *, device, dtype):
        del device, dtype
        backend = KernelBackend.coerce(backend)
        return KernelBackend.TORCH if backend is KernelBackend.AUTO else backend


class _MixedKernelCodec(_KernelCodec):

    def __init__(self) -> None:
        super().__init__()
        self.diffusion_block = _NonCodecKernelBlock()


class _StochasticVAE(nn.Module):

    is_stochastic_vae = True

    def forward(self, value, *, epsilon=None):
        if epsilon is None:
            epsilon = torch.randn_like(value)
        return value + epsilon

    def decode(self, value):
        return value * 2


class _IndexCodec(_TinyCodec):

    def from_indices(self, values):
        return self.decoder(values)


class _Codebook(nn.Module):

    def from_codes(self, values):
        return values.float(), None, None

    def forward(self, values):
        return values


class _EncoderCodebook(nn.Module):

    def encode(self, values):
        return values

    def decode(self, values):
        raise NotImplementedError("inverse path is unavailable")


class CodecCodeContainerTests(unittest.TestCase):

    def test_dense_and_multirate_codes_retain_typed_geometry(self):
        dense_values = torch.zeros(2, 3, 5, dtype=torch.int64)
        dense = DenseCodecCodes(
            dense_values,
            lengths=torch.tensor([5, 3]),
        )
        ragged = RaggedCodecCodes(
            (
                torch.zeros(2, 2, dtype=torch.int32),
                torch.zeros(2, 1, 4, dtype=torch.int32),
            ),
            strides=(2, 1),
        )

        self.assertEqual(dense.batch_size, 2)
        self.assertEqual(dense.num_codebooks, 3)
        self.assertEqual(dense.num_frames, 5)
        self.assertEqual(ragged.batch_size, 2)
        self.assertEqual(ragged.num_levels, 2)
        self.assertEqual(ragged.num_codebooks, 2)
        self.assertEqual(ragged.temporal_lengths, (2, 4))
        self.assertEqual(ragged.strides, (2, 1))
        self.assertIs(coerce_codec_codes(dense_values).values, dense_values)

    def test_code_containers_reject_float_and_mismatched_batches(self):
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            DenseCodecCodes(torch.zeros(1, 2, 3))
        with self.assertRaisesRegex(ValueError, "batch dimension"):
            RaggedCodecCodes((
                torch.zeros(1, 2, dtype=torch.long),
                torch.zeros(2, 4, dtype=torch.long),
            ))


class CodecStructuralProtocolTests(unittest.TestCase):

    def test_component_view_does_not_reparent_or_change_checkpoint_keys(self):
        codec = _TinyCodec()
        before = tuple(codec.state_dict())

        view = separate_audio_codec(codec)

        self.assertIsInstance(codec, AudioCodec)
        self.assertIs(view.encoder, codec.encoder)
        self.assertIs(view.bottleneck, codec.quantizer)
        self.assertIs(view.decoder, codec.decoder)
        self.assertNotIsInstance(view, nn.Module)
        self.assertEqual(tuple(codec.state_dict()), before)
        self.assertEqual(
            tuple(label for label, _module in view.optimization_module_roots()),
            ("codec.encoder", "codec.bottleneck", "codec.decoder"),
        )

    def test_component_view_finds_an_already_owned_nested_codec(self):
        wrapper = nn.Module()
        wrapper.model = _TinyCodec()

        view = separate_audio_codec(wrapper)

        self.assertIs(view.encoder, wrapper.model.encoder)
        self.assertIs(view.bottleneck, wrapper.model.quantizer)
        self.assertIs(view.decoder, wrapper.model.decoder)
        self.assertEqual(view.encoder_attribute, "model.encoder")
        self.assertEqual(view.bottleneck_attribute, "model.quantizer")
        self.assertEqual(view.decoder_attribute, "model.decoder")

    def test_compile_target_discovery_is_mode_and_component_aware(self):
        codec = _TinyCodec()

        inference = discover_codec_compile_targets(codec, mode="inference")
        training = discover_codec_compile_targets(codec, mode="training")
        all_targets = discover_codec_compile_targets(
            codec,
            components=CodecCompileComponent.ALL,
        )

        self.assertEqual(
            tuple(target.attribute for target in inference),
            ("decode", ),
        )
        self.assertEqual(
            tuple(target.attribute for target in training),
            ("forward", ),
        )
        self.assertEqual(
            tuple(target.attribute for target in all_targets),
            ("encode", "forward", "decode", "forward"),
        )

    def test_code_to_waveform_boundary_wins_over_latent_decode(self):
        targets = discover_codec_compile_targets(_IndexCodec(), mode="inference")
        self.assertEqual(
            tuple(target.attribute for target in targets),
            ("from_indices", ),
        )

    def test_component_view_exposes_quantizer_as_an_explicit_stage(self):
        codec = _TinyCodec()
        view = separate_audio_codec(codec)

        targets = discover_codec_compile_targets(
            view,
            components="quantizer",
        )

        self.assertEqual(len(targets), 1)
        self.assertIs(targets[0].owner, codec.quantizer)
        self.assertEqual(targets[0].attribute, "forward")
        self.assertEqual(targets[0].component, "quantizer")

    def test_declared_flow_vocoder_and_decoder_stages_are_executable(self):
        flow = object.__new__(CosyVoiceFlowMatchingModel)
        nn.Module.__init__(flow)
        flow.decoder = SimpleNamespace(estimator=nn.Identity())
        hift = object.__new__(CosyVoiceHiFTGenerator)
        nn.Module.__init__(hift)
        chatterbox = object.__new__(S3Token2Wav)
        nn.Module.__init__(chatterbox)
        mimi = object.__new__(MimiModel)
        nn.Module.__init__(mimi)
        mimi.encoder = nn.Identity()
        mimi.quantizer = _Codebook()
        mimi.decoder = nn.Identity()
        qwen = object.__new__(Qwen3TTSSpeechDecoder)
        nn.Module.__init__(qwen)

        cases = (
            (
                flow,
                ("codec.flow.cosyvoice.estimator.forward", ),
                ("flow", ),
            ),
            (
                hift,
                ("codec.vocoder.cosyvoice_hift.forward", ),
                ("vocoder", ),
            ),
            (
                chatterbox,
                (
                    "codec.flow.chatterbox_s3gen",
                    "codec.vocoder.chatterbox_hift",
                ),
                ("flow", "vocoder"),
            ),
            (
                mimi,
                ("codec.mimi.decode", ),
                ("decode", ),
            ),
            (
                qwen,
                ("codec.decode.qwen3_tts.decode_codes", ),
                ("decode", ),
            ),
        )
        for codec, labels, components in cases:
            with self.subTest(codec=type(codec).__name__):
                targets = discover_codec_compile_targets(codec)
                self.assertEqual(
                    tuple(target.label for target in targets),
                    labels,
                )
                self.assertEqual(
                    tuple(target.component for target in targets),
                    components,
                )

        self.assertEqual(
            tuple(target.label for target in discover_codec_compile_targets(
                chatterbox,
                components="flow",
            )),
            ("codec.flow.chatterbox_s3gen", ),
        )
        self.assertEqual(
            tuple(
                target.label for target in discover_codec_compile_targets(
                    chatterbox,
                    components="decode",
                )),
            (
                "codec.flow.chatterbox_s3gen",
                "codec.vocoder.chatterbox_hift",
            ),
        )
        self.assertEqual(
            tuple(target.label for target in discover_codec_compile_targets(
                flow,
                components="all",
            )),
            ("codec.flow.cosyvoice.estimator.forward", ),
        )
        self.assertEqual(
            {target.component
             for target in discover_codec_compile_targets(
                 mimi,
                 components="all",
             )},
            {"decode", "encode", "quantizer"},
        )

    def test_architecture_hooks_avoid_ambiguous_generic_boundaries(self):
        tokenizer = object.__new__(S3TokenizerV2)
        nn.Module.__init__(tokenizer)
        tokenizer.encoder = nn.Identity()
        tokenizer.quantizer = _EncoderCodebook()
        hifigan = object.__new__(HifiDecoder)
        nn.Module.__init__(hifigan)

        tokenizer_targets = discover_codec_compile_targets(
            tokenizer,
            components="all",
        )
        vocoder_targets = discover_codec_compile_targets(
            hifigan,
            components="vocoder",
        )

        self.assertEqual(
            tuple((target.component, target.attribute) for target in tokenizer_targets),
            (("encode", "forward"), ("quantizer", "encode")),
        )
        self.assertEqual(
            tuple((target.component, target.attribute) for target in vocoder_targets),
            (("vocoder", "forward"), ),
        )

    def test_cuda_graph_auto_resolves_forward_only_decoder_before_cuda_check(self):
        decoder = object.__new__(Qwen3TTSSpeechDecoder)
        nn.Module.__init__(decoder)

        with self.assertRaises(CodecCUDAGraphCaptureError) as captured:
            capture_codec_cuda_graph(
                decoder,
                torch.zeros(1, 16, 2, dtype=torch.long),
            )

        self.assertIn("CUDA", str(captured.exception))
        self.assertNotIn(".auto is not a callable", str(captured.exception))

    def test_dac_code_to_waveform_and_quantizer_targets_match_live_calls(self):
        codec = object.__new__(DAC)
        nn.Module.__init__(codec)
        codec.quantizer = _Codebook()
        codec.decoder = nn.Identity()

        inference = discover_codec_compile_targets(codec)
        quantizer = discover_codec_compile_targets(
            codec,
            components="quantizer",
        )
        training_quantizer = discover_codec_compile_targets(
            codec,
            mode="training",
            components="quantizer",
        )
        codes = torch.ones(1, 2, 4, dtype=torch.long)

        self.assertEqual(
            tuple(target.attribute for target in inference),
            ("decode_codes", ),
        )
        self.assertEqual(
            tuple(target.attribute for target in quantizer),
            ("from_codes", ),
        )
        self.assertEqual(
            tuple(target.attribute for target in training_quantizer),
            ("forward", ),
        )
        torch.testing.assert_close(
            codec.decode_codes(codes),
            codes.float(),
        )


class CodecOptimizationPlanTests(unittest.TestCase):

    def test_config_and_plan_are_strict_json_serializable(self):
        codec = _TinyCodec()
        config = CodecOptimizationConfig(
            policy="relaxed",
            kernel_backend="native",
            compile=False,
            compile_components="decode",
        )

        restored = CodecOptimizationConfig.from_dict(json.loads(config.to_json_string()))
        plan = resolve_codec_optimization(codec, restored)
        manifest = plan.manifest()

        self.assertIs(restored.policy, CodecOptimizationPolicy.RELAXED)
        self.assertEqual(manifest["config"], config.to_dict())
        self.assertEqual(json.loads(plan.to_json_string()), manifest)
        self.assertEqual(manifest["passes"], [])
        self.assertEqual(
            [decision["selected"] for decision in manifest["decisions"]],
            ["relaxed", "native", "eager"],
        )

    def test_codec_kernel_selector_is_applied_before_compile_and_reversible(self):
        codec = _KernelCodec()
        keys = tuple(codec.state_dict())
        plan = resolve_codec_optimization(
            codec,
            CodecOptimizationConfig(
                kernel_backend="auto",
                compile=False,
            ),
        )

        self.assertEqual(
            [optimization_pass.pass_id for optimization_pass in plan],
            ["codec-kernels"],
        )
        self.assertIsInstance(plan.passes[0], CodecKernelPass)
        result = plan.apply(codec)
        self.assertIs(codec.kernel_block.kernel_backend, KernelBackend.TORCH)
        self.assertEqual(tuple(codec.state_dict()), keys)
        self.assertIs(result.restore(), codec)
        self.assertEqual(tuple(codec.state_dict()), keys)

    def test_codec_kernel_pass_does_not_mutate_non_codec_selectors(self):
        codec = _MixedKernelCodec()
        codec.kernel_block.set_codec_kernel_backend("auto")
        before_diffusion = codec.diffusion_block.kernel_backend

        plan = resolve_codec_optimization(
            codec,
            CodecOptimizationConfig(
                kernel_backend="auto",
                compile=False,
            ),
        )
        result = plan.apply(codec)

        self.assertIs(
            codec.kernel_block.codec_kernel_backend,
            CodecKernelBackend.TORCH,
        )
        self.assertIs(codec.diffusion_block.kernel_backend, before_diffusion)
        metadata = result.manifest_metadata()[0]["metadata"]
        self.assertEqual(metadata["domain"], "codec")
        self.assertEqual(metadata["targets"], ["model.kernel_block"])
        result.restore()
        self.assertIs(
            codec.kernel_block.codec_kernel_backend,
            CodecKernelBackend.AUTO,
        )
        self.assertIs(codec.diffusion_block.kernel_backend, before_diffusion)

    def test_exact_policy_pins_auto_kernels_and_rejects_accelerator_math(self):
        codec = _KernelCodec()
        plan = resolve_codec_optimization(
            codec,
            CodecOptimizationConfig(
                policy="exact",
                kernel_backend="auto",
                compile=False,
            ),
        )

        self.assertIs(plan.passes[0].backend, CodecKernelBackend.TORCH)
        self.assertEqual(plan.decisions[1].selected, "torch")
        with self.assertRaisesRegex(
                ValueError,
                "requires policy='relaxed'",
        ):
            resolve_codec_optimization(
                codec,
                CodecOptimizationConfig(
                    policy="exact",
                    kernel_backend="triton",
                    compile=False,
                ),
            )

    def test_relaxed_codec_auto_resolves_an_available_accelerator_explicitly(self):
        codec = _KernelCodec()
        context = OptimizationContext(
            mode="inference",
            device="cuda",
            dtype="float16",
        )

        with (
                mock.patch(
                    "voicehub.kernels.cuda_extensions."
                    "CUDA_EXTENSIONS.is_loaded",
                    return_value=False,
                ),
                mock.patch(
                    "voicehub.kernels.capabilities.triton_capability",
                    return_value=SimpleNamespace(
                        available=True,
                        reason="",
                    ),
                ),
        ):
            plan = resolve_codec_optimization(
                codec,
                CodecOptimizationConfig(
                    policy="relaxed",
                    kernel_backend="auto",
                    compile=False,
                ),
                context=context,
            )

        self.assertIs(plan.passes[0].backend, CodecKernelBackend.TRITON)
        self.assertEqual(plan.decisions[1].selected, "triton")

    def test_cute_is_codec_scoped_and_fails_closed_without_an_operation(self):
        codec = _KernelCodec()
        config = CodecOptimizationConfig(
            policy="relaxed",
            kernel_backend="cutlass",
            compile=False,
        )

        self.assertIs(config.kernel_backend, CodecKernelBackend.CUTE)
        self.assertEqual(config.to_dict()["kernel_backend"], "cute")
        with self.assertRaisesRegex(
                CodecOptimizationCompatibilityError,
                "exposes no operation with a registered 'cute' implementation",
        ):
            resolve_codec_optimization(codec, config)

    def test_cute_capability_probe_is_lazy_and_fails_closed_off_linux(self):
        with (
                mock.patch(
                    "voicehub.kernels.capabilities.sys.platform",
                    "darwin",
                ),
                mock.patch("voicehub.kernels.capabilities.import_module", ) as import_module,
        ):
            capability = cute_dsl_capability("cuda")

        self.assertFalse(capability.available)
        self.assertIn("only on Linux", capability.reason)
        import_module.assert_not_called()

    def test_required_compile_uses_the_discovered_decoder_boundary(self):
        codec = _TinyCodec()
        plan = resolve_codec_optimization(
            codec,
            CodecOptimizationConfig(
                kernel_backend="native",
                compile=True,
                compile_components="decode",
                compile_config={
                    "backend": "eager",
                    "fullgraph": True,
                },
            ),
        )

        self.assertEqual(
            tuple(target.attribute for target in plan.compile_targets),
            ("decode", ),
        )
        self.assertEqual(
            [optimization_pass.pass_id for optimization_pass in plan],
            ["torch.compile"],
        )
        application = plan.apply(codec)
        value = torch.randn(2, 4)
        torch.testing.assert_close(
            application.model.decode(value),
            codec.decoder(value),
        )
        application.restore()


class CodecCUDAGraphSafetyTests(unittest.TestCase):

    def test_stochastic_vae_uses_graph_aware_default_rng_without_epsilon(self):
        codec = _StochasticVAE()
        with self.assertRaises(CodecCUDAGraphCaptureError) as captured:
            capture_codec_cuda_graph(
                codec,
                torch.zeros(1, 4),
                target="forward",
            )
        self.assertNotIn("explicit `epsilon`", str(captured.exception))

    def test_decoder_only_vae_path_passes_randomness_check(self):
        codec = _StochasticVAE()
        with self.assertRaises(CodecCUDAGraphCaptureError) as captured:
            capture_codec_cuda_graph(
                codec,
                torch.zeros(1, 4),
                target="decode",
            )
        self.assertNotIn("Stochastic VAE", str(captured.exception))

    def test_decoder_only_flag_cannot_disguise_a_forward_capture(self):
        with self.assertRaisesRegex(
                CodecCUDAGraphCaptureError,
                "recognized decoder",
        ):
            capture_codec_cuda_graph(
                _StochasticVAE(),
                torch.zeros(1, 4),
                target="forward",
                decoder_only=True,
            )

    def test_explicit_epsilon_is_checked_before_cuda_capture(self):
        codec = _StochasticVAE()
        with self.assertRaises(CodecCUDAGraphCaptureError) as captured:
            capture_codec_cuda_graph(
                codec,
                torch.zeros(1, 4),
                target="forward",
                epsilon=torch.zeros(1, 4),
            )
        self.assertNotIn("explicit `epsilon`", str(captured.exception))

    def test_deterministic_vae_encoder_does_not_require_epsilon(self):
        codec = object.__new__(VoxCPMAudioVAE)
        nn.Module.__init__(codec)
        with self.assertRaises(CodecCUDAGraphCaptureError) as captured:
            capture_codec_cuda_graph(
                codec,
                torch.zeros(1, 1, 16),
                target="encode",
            )
        self.assertNotIn("explicit `epsilon`", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
