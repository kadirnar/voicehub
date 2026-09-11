from __future__ import annotations

import unittest

import torch
from torch import nn

from voicehub.components.audio.codecs.dac.nn.layers import Snake1d as DACSnake
from voicehub.kernels import (
    AUDIO_CODEC_SNAKE,
    AUDIO_CODEC_SNAKE_BETA,
    DIFFUSION_FUSED_MODULATE,
    KernelBackend,
    codec_snake,
    codec_snake_beta,
    codec_snake_beta_reference,
    codec_snake_reference,
    fused_modulate,
    fused_modulate_reference,
    resolve_kernel,
)
from voicehub.models.chatterbox.models.s3gen.hifigan import Snake as ChatterboxSnake
from voicehub.models.cosyvoice.native.flow import AdaLayerNormZeroFinal as CosyVoiceAdaLayerNormFinal
from voicehub.models.cosyvoice.native.vocoder import Snake as CosyVoiceSnake
from voicehub.models.echo.autoencoder import Snake1d as EchoSnake
from voicehub.models.echo.model import LowRankAdaLN as EchoLowRankAdaLN
from voicehub.models.f5tts.native.modules import AdaLayerNormFinal as F5AdaLayerNormFinal
from voicehub.models.higgstts.native.tokenizer import Snake1d as HiggsSnake
from voicehub.models.irodoritts.native.codec import IrodoriDACVAECodec
from voicehub.models.irodoritts.native.codec_layers import Snake1d as IrodoriSnake
from voicehub.models.irodoritts.native.modeling import LowRankAdaLN as IrodoriLowRankAdaLN
from voicehub.models.irodoritts.native.runtime import InferenceRuntime as IrodoriRuntime
from voicehub.models.llasa.xcodec2 import XCodec2SnakeBeta
from voicehub.models.omnivoice.native.codec import Snake1d as OmniVoiceSnake
from voicehub.models.orpheustts.source.snac.layers import Snake1d as SNACSnake
from voicehub.models.qwen3tts.native.codec import SnakeBeta as QwenSnakeBeta
from voicehub.models.vibevoice.native.diffusion import VibeVoiceDiffusionLayer
from voicehub.models.voxcpm.native.codec import _Snake1d as VoxCPMSnake
from voicehub.models.vui.fluac import Snake1d as VUISnake
from voicehub.models.zonos.native.codec import ZonosDACCodec
from voicehub.optimization import (
    CodecOptimizationConfig,
    CustomKernelPass,
    OptimizationContext,
    OptimizationPassManager,
    resolve_codec_optimization,
)


def _assert_tree_close(
    test: unittest.TestCase,
    actual,
    expected,
) -> None:
    if isinstance(expected, tuple):
        test.assertIsInstance(actual, tuple)
        test.assertEqual(len(actual), len(expected))
        for actual_item, expected_item in zip(actual, expected):
            _assert_tree_close(test, actual_item, expected_item)
        return
    torch.testing.assert_close(actual, expected)


def _diffusion_blocks() -> tuple[nn.Module, ...]:
    return (
        F5AdaLayerNormFinal(4),
        CosyVoiceAdaLayerNormFinal(4),
        EchoLowRankAdaLN(4, 2, 1e-6),
        IrodoriLowRankAdaLN(4, 2, 1e-6),
        VibeVoiceDiffusionLayer(
            4,
            8,
            norm_epsilon=1e-6,
        ),
    )


def _diffusion_inputs() -> tuple[tuple[torch.Tensor, ...], ...]:
    hidden_states = torch.randn(2, 3, 4)
    embedding = torch.randn(2, 4)
    low_rank_condition = torch.randn(2, 3, 12)
    vibe_condition = torch.randn(2, 3, 4)
    return (
        (hidden_states, embedding),
        (hidden_states, embedding),
        (hidden_states, low_rank_condition),
        (hidden_states, low_rank_condition),
        (hidden_states, vibe_condition),
    )


def _codec_snakes() -> tuple[nn.Module, ...]:
    return (
        DACSnake(3),
        ChatterboxSnake(3),
        CosyVoiceSnake(3),
        IrodoriSnake(3),
        HiggsSnake(3),
        OmniVoiceSnake(3),
        VoxCPMSnake(3),
        EchoSnake(3),
        SNACSnake(3),
        VUISnake(3),
    )


def _codec_snake_betas() -> tuple[nn.Module, ...]:
    return (
        XCodec2SnakeBeta(3),
        QwenSnakeBeta(
            3,
            factory_kwargs={
                "device": None,
                "dtype": None,
            },
        ),
    )


class DiffusionFusedModulateTests(unittest.TestCase):

    def test_broadcast_noncontiguous_forward_and_backward_match_reference(self):
        torch.manual_seed(101)
        hidden_states = (torch.randn(2, 7, 4, dtype=torch.float64).transpose(1, 2).requires_grad_())
        shift = torch.randn(
            2,
            1,
            7,
            dtype=torch.float64,
            requires_grad=True,
        )
        scale = torch.randn(
            1,
            4,
            1,
            dtype=torch.float64,
            requires_grad=True,
        )
        self.assertFalse(hidden_states.is_contiguous())

        actual = fused_modulate(
            hidden_states,
            shift,
            scale,
            backend="torch",
        )
        reference_hidden = hidden_states.detach().clone().requires_grad_()
        reference_shift = shift.detach().clone().requires_grad_()
        reference_scale = scale.detach().clone().requires_grad_()
        expected = fused_modulate_reference(
            reference_hidden,
            reference_shift,
            reference_scale,
        )
        gradient = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(
            actual,
            (hidden_states, shift, scale),
            gradient,
        )
        expected_gradients = torch.autograd.grad(
            expected,
            (reference_hidden, reference_shift, reference_scale),
            gradient,
        )

        torch.testing.assert_close(actual, expected)
        for actual_gradient, expected_gradient in zip(
                actual_gradients,
                expected_gradients,
        ):
            torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertEqual(actual_gradients[1].shape, shift.shape)
        self.assertEqual(actual_gradients[2].shape, scale.shape)

    def test_empty_tensor_retains_shape_and_zero_broadcast_gradients(self):
        hidden_states = torch.empty(
            2,
            0,
            4,
            requires_grad=True,
        )
        shift = torch.randn(2, 1, 4, requires_grad=True)
        scale = torch.randn(4, requires_grad=True)

        output = fused_modulate(
            hidden_states,
            shift,
            scale,
            backend="torch",
        )
        output.sum().backward()

        self.assertEqual(output.shape, (2, 0, 4))
        self.assertEqual(hidden_states.grad.shape, hidden_states.shape)
        torch.testing.assert_close(shift.grad, torch.zeros_like(shift))
        torch.testing.assert_close(scale.grad, torch.zeros_like(scale))

    def test_invalid_contracts_fail_before_dispatch(self):
        hidden_states = torch.ones(2, 3, 4)
        shift = torch.ones(2, 1, 4)
        scale = torch.ones(4)

        with self.assertRaisesRegex(TypeError, "must be torch.Tensor"):
            fused_modulate(hidden_states, None, scale, backend="torch")
        with self.assertRaisesRegex(ValueError, "at least one tensor dimension"):
            fused_modulate(
                torch.tensor(1.0),
                torch.tensor(1.0),
                torch.tensor(1.0),
                backend="torch",
            )
        with self.assertRaisesRegex(ValueError, "same dtype"):
            fused_modulate(
                hidden_states,
                shift.double(),
                scale,
                backend="torch",
            )
        with self.assertRaisesRegex(TypeError, "floating-point"):
            fused_modulate(
                hidden_states.to(torch.int64),
                shift.to(torch.int64),
                scale.to(torch.int64),
                backend="torch",
            )
        with self.assertRaisesRegex(ValueError, "broadcast"):
            fused_modulate(
                hidden_states,
                torch.ones(5),
                scale,
                backend="torch",
            )
        with self.assertRaisesRegex(ValueError, "cannot expand"):
            fused_modulate(
                torch.ones(1, 4),
                torch.ones(2, 4),
                scale,
                backend="torch",
            )


class CodecSnakeKernelTests(unittest.TestCase):

    def test_native_ranks_and_alpha_layouts_match_reference(self):
        torch.manual_seed(202)
        cases = (
            (
                torch.randn(2, 3, dtype=torch.float64),
                torch.rand(3, dtype=torch.float64) + 0.5,
            ),
            (
                torch.randn(2, 5, 3, dtype=torch.float64).transpose(1, 2),
                torch.rand(1, 3, dtype=torch.float64) + 0.5,
            ),
            (
                torch.randn(2, 3, 2, 4, dtype=torch.float64),
                torch.rand(1, 3, 1, 1, dtype=torch.float64) + 0.5,
            ),
        )

        for inputs, alpha in cases:
            with self.subTest(
                    input_shape=tuple(inputs.shape),
                    alpha_shape=tuple(alpha.shape),
            ):
                torch.testing.assert_close(
                    codec_snake(inputs, alpha, backend="torch"),
                    codec_snake_reference(inputs, alpha),
                )

    def test_noncontiguous_forward_and_gradients_match_reference(self):
        torch.manual_seed(303)
        inputs = (torch.randn(2, 8, 3, dtype=torch.float64).transpose(1, 2).requires_grad_())
        alpha = ((torch.rand(3, 1, dtype=torch.float64) + 0.5).requires_grad_())
        self.assertFalse(inputs.is_contiguous())

        actual = codec_snake(inputs, alpha, backend="torch")
        reference_inputs = inputs.detach().clone().requires_grad_()
        reference_alpha = alpha.detach().clone().requires_grad_()
        expected = codec_snake_reference(
            reference_inputs,
            reference_alpha,
        )
        gradient = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(
            actual,
            (inputs, alpha),
            gradient,
        )
        expected_gradients = torch.autograd.grad(
            expected,
            (reference_inputs, reference_alpha),
            gradient,
        )

        torch.testing.assert_close(actual, expected)
        for actual_gradient, expected_gradient in zip(
                actual_gradients,
                expected_gradients,
        ):
            torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertEqual(actual_gradients[1].shape, alpha.shape)

    def test_invalid_contracts_fail_before_dispatch(self):
        inputs = torch.ones(2, 3, 4)
        alpha = torch.ones(1, 3, 1)

        with self.assertRaisesRegex(TypeError, "must be torch.Tensor"):
            codec_snake(inputs, None, backend="torch")
        with self.assertRaisesRegex(ValueError, "batch, channels"):
            codec_snake(
                torch.ones(3),
                torch.ones(3),
                backend="torch",
            )
        with self.assertRaisesRegex(ValueError, "one alpha value per input channel"):
            codec_snake(inputs, torch.ones(2), backend="torch")
        with self.assertRaisesRegex(ValueError, "same dtype"):
            codec_snake(inputs, alpha.double(), backend="torch")
        with self.assertRaisesRegex(TypeError, "floating-point"):
            codec_snake(
                inputs.to(torch.int64),
                alpha.to(torch.int64),
                backend="torch",
            )

    def test_snake_beta_forward_and_gradients_match_reference(self):
        inputs = torch.randn(
            2,
            3,
            7,
            dtype=torch.float64,
            requires_grad=True,
        )
        alpha = (torch.rand(3, dtype=torch.float64) + 0.5).requires_grad_()
        beta = (torch.rand(1, 3, 1, dtype=torch.float64) + 0.5).requires_grad_()
        actual = codec_snake_beta(
            inputs,
            alpha,
            beta,
            backend="torch",
        )
        expected_inputs = inputs.detach().clone().requires_grad_()
        expected_alpha = alpha.detach().clone().requires_grad_()
        expected_beta = beta.detach().clone().requires_grad_()
        expected = codec_snake_beta_reference(
            expected_inputs,
            expected_alpha,
            expected_beta,
        )
        gradient = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(
            actual,
            (inputs, alpha, beta),
            gradient,
        )
        expected_gradients = torch.autograd.grad(
            expected,
            (expected_inputs, expected_alpha, expected_beta),
            gradient,
        )
        torch.testing.assert_close(actual, expected)
        for actual_gradient, expected_gradient in zip(
                actual_gradients,
                expected_gradients,
        ):
            torch.testing.assert_close(actual_gradient, expected_gradient)


class KernelDispatchAndCompileTests(unittest.TestCase):

    def test_auto_dispatch_resolves_portable_torch_kernels_on_cpu(self):
        hidden_states = torch.randn(2, 3, 4)
        shift = torch.randn(2, 1, 4)
        scale = torch.randn(4)
        codec_inputs = torch.randn(2, 3, 5)
        alpha = torch.rand(1, 3, 1) + 0.5

        modulation = resolve_kernel(
            DIFFUSION_FUSED_MODULATE,
            hidden_states,
            shift,
            scale,
            backend="auto",
        )
        snake = resolve_kernel(
            AUDIO_CODEC_SNAKE,
            codec_inputs,
            alpha,
            backend="auto",
        )

        self.assertIs(modulation.backend, KernelBackend.TORCH)
        self.assertIs(snake.backend, KernelBackend.TORCH)
        torch.testing.assert_close(
            fused_modulate(
                hidden_states,
                shift,
                scale,
                backend="auto",
            ),
            fused_modulate_reference(hidden_states, shift, scale),
        )
        torch.testing.assert_close(
            codec_snake(codec_inputs, alpha, backend="auto"),
            codec_snake_reference(codec_inputs, alpha),
        )

    def test_real_selector_boundaries_compile_fullgraph_with_aot_eager(self):
        diffusion = F5AdaLayerNormFinal(4).eval()
        diffusion.set_kernel_backend("torch")
        diffusion_inputs = (
            torch.randn(2, 3, 4),
            torch.randn(2, 4),
        )
        compiled_diffusion = torch.compile(
            diffusion,
            backend="aot_eager",
            fullgraph=True,
        )
        torch.testing.assert_close(
            compiled_diffusion(*diffusion_inputs),
            diffusion(*diffusion_inputs),
        )

        codec = DACSnake(3).eval()
        codec.set_kernel_backend("torch")
        codec_inputs = torch.randn(2, 3, 7)
        compiled_codec = torch.compile(
            codec,
            backend="aot_eager",
            fullgraph=True,
        )
        torch.testing.assert_close(
            compiled_codec(codec_inputs),
            codec(codec_inputs),
        )


class StructuralKernelSelectorTests(unittest.TestCase):

    def test_real_diffusion_blocks_preserve_state_and_auto_matches_torch(self):
        torch.manual_seed(404)
        for block, inputs in zip(
                _diffusion_blocks(),
                _diffusion_inputs(),
        ):
            with self.subTest(block=f"{type(block).__module__}.{type(block).__name__}"):
                before = {key: value.detach().clone() for key, value in block.state_dict().items()}
                self.assertEqual(
                    block.supported_kernel_operations,
                    (DIFFUSION_FUSED_MODULATE, ),
                )
                self.assertIs(
                    block.resolve_kernel_backend(
                        "auto",
                        device="cpu",
                        dtype="float32",
                    ),
                    KernelBackend.TORCH,
                )
                block.set_kernel_backend("torch")
                expected = block(*inputs)
                block.set_kernel_backend("auto")
                actual = block(*inputs)
                _assert_tree_close(self, actual, expected)
                self.assertIs(block.kernel_backend, KernelBackend.AUTO)
                self.assertEqual(tuple(block.state_dict()), tuple(before))
                for key, value in block.state_dict().items():
                    torch.testing.assert_close(value, before[key])

    def test_real_codec_snakes_preserve_state_and_auto_matches_torch(self):
        torch.manual_seed(505)
        inputs = torch.randn(2, 3, 7)
        for snake in _codec_snakes():
            with self.subTest(snake=f"{type(snake).__module__}.{type(snake).__name__}"):
                before = {key: value.detach().clone() for key, value in snake.state_dict().items()}
                self.assertEqual(
                    snake.supported_kernel_operations,
                    (AUDIO_CODEC_SNAKE, ),
                )
                self.assertIs(
                    snake.resolve_kernel_backend(
                        "auto",
                        device="cpu",
                        dtype="float32",
                    ),
                    KernelBackend.TORCH,
                )
                snake.set_kernel_backend("torch")
                expected = snake(inputs)
                snake.set_kernel_backend("auto")
                actual = snake(inputs)
                torch.testing.assert_close(actual, expected)
                self.assertIs(snake.kernel_backend, KernelBackend.AUTO)
                self.assertEqual(tuple(snake.state_dict()), tuple(before))
                for key, value in snake.state_dict().items():
                    torch.testing.assert_close(value, before[key])

    def test_universal_auto_keeps_periodic_codec_math_on_torch(self):
        for snake in (*_codec_snakes(), *_codec_snake_betas()):
            with self.subTest(snake=f"{type(snake).__module__}.{type(snake).__name__}", ):
                self.assertIs(
                    snake.resolve_kernel_backend(
                        "auto",
                        device="cuda",
                        dtype="float16",
                    ),
                    KernelBackend.TORCH,
                )

    def test_real_codec_snake_betas_share_the_structural_protocol(self):
        inputs = torch.randn(2, 3, 7)
        for snake in _codec_snake_betas():
            with self.subTest(snake=f"{type(snake).__module__}.{type(snake).__name__}", ):
                keys = tuple(snake.state_dict())
                self.assertEqual(
                    snake.supported_kernel_operations,
                    (AUDIO_CODEC_SNAKE_BETA, ),
                )
                snake.set_kernel_backend("torch")
                expected = snake(inputs)
                snake.set_kernel_backend("auto")
                torch.testing.assert_close(snake(inputs), expected)
                self.assertEqual(tuple(snake.state_dict()), keys)

    def test_custom_kernel_pass_configures_and_restores_diffusion_modules(self):
        model = nn.ModuleList(_diffusion_blocks())
        for block in model:
            block.set_kernel_backend("auto")
        keys = tuple(model.state_dict())

        result = OptimizationPassManager().apply(
            model,
            (CustomKernelPass(backend="auto"), ),
            OptimizationContext(
                mode="training",
                device="cpu",
                dtype="float32",
                persist_result=True,
            ),
        )

        self.assertTrue(all(block.kernel_backend is KernelBackend.TORCH for block in model), )
        metadata = result.manifest_metadata()[0]["metadata"]
        self.assertEqual(metadata["target_count"], len(model))
        self.assertEqual(metadata["changed_target_count"], len(model))
        self.assertEqual(
            metadata["kernel_operations"],
            [DIFFUSION_FUSED_MODULATE],
        )
        self.assertEqual(tuple(model.state_dict()), keys)

        result.restore()
        self.assertTrue(all(block.kernel_backend is KernelBackend.AUTO for block in model), )
        self.assertEqual(tuple(model.state_dict()), keys)

    def test_irodori_runtime_exposes_kernel_roots_and_checkpoint_keys(self):
        runtime = object.__new__(IrodoriRuntime)
        block = IrodoriLowRankAdaLN(4, 2, 1e-6)
        block.set_kernel_backend("auto")
        runtime.model = nn.Sequential(block)
        codec_block = IrodoriSnake(3)
        codec_block.set_kernel_backend("auto")
        runtime.codec = object.__new__(IrodoriDACVAECodec)
        runtime.codec.model = nn.Sequential(codec_block)
        keys = tuple(runtime.state_dict())

        result = OptimizationPassManager().apply(
            runtime,
            (CustomKernelPass(backend="auto"), ),
            OptimizationContext(
                mode="inference",
                device="cpu",
                dtype="float32",
                persist_result=True,
            ),
        )

        self.assertIs(block.kernel_backend, KernelBackend.TORCH)
        self.assertIs(codec_block.kernel_backend, KernelBackend.TORCH)
        self.assertEqual(tuple(runtime.state_dict()), keys)
        result.restore()
        self.assertIs(block.kernel_backend, KernelBackend.AUTO)
        self.assertIs(codec_block.kernel_backend, KernelBackend.AUTO)
        self.assertEqual(tuple(runtime.state_dict()), keys)

    def test_real_non_module_codec_wrappers_expose_nested_kernel_graphs(self):
        irodori = object.__new__(IrodoriDACVAECodec)
        irodori_snake = IrodoriSnake(3)
        irodori_snake.set_kernel_backend("auto")
        irodori.model = nn.Sequential(irodori_snake)

        zonos = object.__new__(ZonosDACCodec)
        zonos_snake = DACSnake(3)
        zonos_snake.set_kernel_backend("auto")
        zonos._model = nn.Sequential(zonos_snake)

        for codec, snake in (
            (irodori, irodori_snake),
            (zonos, zonos_snake),
        ):
            with self.subTest(codec=type(codec).__name__):
                plan = resolve_codec_optimization(
                    codec,
                    CodecOptimizationConfig(
                        policy="exact",
                        kernel_backend="auto",
                        compile=False,
                    ),
                )
                self.assertEqual(
                    [optimization_pass.pass_id for optimization_pass in plan],
                    ["codec-kernels"],
                )
                result = plan.apply(codec)
                self.assertIs(snake.kernel_backend, KernelBackend.TORCH)
                result.restore()
                self.assertIs(snake.kernel_backend, KernelBackend.AUTO)

    def test_custom_kernel_pass_configures_and_restores_codec_modules(self):
        model = nn.ModuleList(_codec_snakes())
        for snake in model:
            snake.set_kernel_backend("auto")
        keys = tuple(model.state_dict())

        result = OptimizationPassManager().apply(
            model,
            (CustomKernelPass(backend="auto"), ),
            OptimizationContext(
                mode="inference",
                device="cpu",
                dtype="float32",
                persist_result=True,
            ),
        )

        self.assertTrue(all(snake.kernel_backend is KernelBackend.TORCH for snake in model), )
        metadata = result.manifest_metadata()[0]["metadata"]
        self.assertEqual(metadata["target_count"], len(model))
        self.assertEqual(metadata["changed_target_count"], len(model))
        self.assertEqual(
            metadata["kernel_operations"],
            [AUDIO_CODEC_SNAKE],
        )
        self.assertEqual(tuple(model.state_dict()), keys)

        result.restore()
        self.assertTrue(all(snake.kernel_backend is KernelBackend.AUTO for snake in model), )
        self.assertEqual(tuple(model.state_dict()), keys)


if __name__ == "__main__":
    unittest.main()
