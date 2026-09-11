from __future__ import annotations

import json
import subprocess
import sys
import unittest

import torch

from voicehub.kernels import KernelBackend
from voicehub.models.conversationtts.native.decoder import build_llama32_decoder
from voicehub.models.conversationtts.native.registration import create_conversationtts_architecture_spec
from voicehub.models.f5tts.native.registration import create_f5tts_architecture_spec
from voicehub.models.qwen3tts.native.registration import create_qwen3_tts_architecture_spec
from voicehub.models.vits.native.registration import create_vits_architecture_spec
from voicehub.neural.backends import FlashAttention4Policy
from voicehub.optimization import (
    CustomKernelPass,
    FlashAttention4Pass,
    OptimizationContext,
    OptimizationPassManager,
    TorchCompilePass,
)
from voicehub.training import (
    DiffusionTTSOptimizationConfig,
    LLMTTSOptimizationConfig,
    VITSOptimizationConfig,
    diffusion_tts_acceleration_plan,
    llm_tts_acceleration_plan,
    vits_acceleration_plan,
)


class TTSAccelerationPlanTests(unittest.TestCase):

    def test_each_family_builds_its_semantically_compatible_ordered_plan(self):
        vits = vits_acceleration_plan(kernel_backend="triton")
        llm = llm_tts_acceleration_plan(
            kernel_backend="triton",
            attention_policy="required",
        )
        diffusion = diffusion_tts_acceleration_plan(
            kernel_backend="cuda-extension",
            attention_policy="disabled",
        )

        self.assertEqual(
            [type(item) for item in vits],
            [CustomKernelPass, TorchCompilePass],
        )
        self.assertEqual(
            [type(item) for item in llm],
            [CustomKernelPass, FlashAttention4Pass, TorchCompilePass],
        )
        self.assertEqual(
            [type(item) for item in diffusion],
            [CustomKernelPass, FlashAttention4Pass, TorchCompilePass],
        )
        self.assertIs(vits[0].backend, KernelBackend.TRITON)
        self.assertIs(llm[1].policy, FlashAttention4Policy.REQUIRED)
        self.assertIs(
            diffusion[0].backend,
            KernelBackend.CUDA_EXTENSION,
        )
        self.assertIs(
            diffusion[1].policy,
            FlashAttention4Policy.DISABLED,
        )
        self.assertEqual(vits[-1].config.mode, "default")
        self.assertTrue(vits[-1].config.dynamic)
        for plan in (llm, diffusion):
            compile_pass = plan[-1]
            self.assertEqual(
                compile_pass.config.mode,
                "max-autotune-no-cudagraphs",
            )
            self.assertTrue(compile_pass.config.dynamic)
            self.assertFalse(compile_pass.config.fullgraph)

    def test_profile_conveniences_delegate_without_forcing_compile(self):
        plans = (
            VITSOptimizationConfig().acceleration_plan(use_torch_compile=False, ),
            LLMTTSOptimizationConfig().acceleration_plan(use_torch_compile=False, ),
            DiffusionTTSOptimizationConfig().acceleration_plan(use_torch_compile=False, ),
        )
        self.assertEqual(
            [[type(item) for item in plan] for plan in plans],
            [
                [CustomKernelPass],
                [CustomKernelPass, FlashAttention4Pass],
                [CustomKernelPass, FlashAttention4Pass],
            ],
        )

    def test_plan_construction_does_not_import_or_build_optional_backends(self):
        code = '\nimport json\nimport sys\nfrom voicehub.training import llm_tts_acceleration_plan\nplan = llm_tts_acceleration_plan(use_torch_compile=False)\nprint(json.dumps({\n    "count": len(plan),\n    "flash_attn": "flash_attn" in sys.modules,\n    "triton": "triton" in sys.modules,\n    "cpp_extension": "torch.utils.cpp_extension" in sys.modules,\n}))\n'
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(result.stdout),
            {
                "count": 2,
                "flash_attn": False,
                "triton": False,
                "cpp_extension": False,
            },
        )

    def test_architecture_capabilities_match_the_acceleration_matrix(self):
        specifications = {
            "vits": create_vits_architecture_spec(),
            "conversation": create_conversationtts_architecture_spec(),
            "qwen": create_qwen3_tts_architecture_spec(),
            "f5": create_f5tts_architecture_spec(),
        }
        self.assertEqual(
            specifications["vits"].capabilities.optimization_passes,
            ("compile", "custom-kernels"),
        )
        for name in ("conversation", "qwen", "f5"):
            capabilities = (specifications[name].capabilities.optimization_passes)
            self.assertIn("compile", capabilities)
            self.assertIn("custom-kernels", capabilities)
            self.assertIn("attention-backend", capabilities)

    def test_llm_plan_configures_real_conversation_graph_and_restores(self):
        torch.manual_seed(23)
        decoder = build_llama32_decoder(
            vocabulary_size=19,
            number_of_layers=2,
            number_of_heads=4,
            number_of_kv_heads=2,
            embedding_dimension=16,
            maximum_sequence_length=12,
            intermediate_dimension=32,
            normalization_epsilon=1e-5,
        ).eval()
        decoder.tok_embeddings = torch.nn.Identity()
        decoder.output = torch.nn.Identity()
        inputs = torch.randn(2, 5, 16)
        original_keys = tuple(decoder.state_dict())
        expected = decoder(inputs)

        result = OptimizationPassManager().apply(
            decoder,
            LLMTTSOptimizationConfig().acceleration_plan(use_torch_compile=False, ),
            OptimizationContext(
                mode="training",
                architecture="conversationtts",
                device="cpu",
                dtype="float32",
                persist_result=True,
            ),
        )
        actual = decoder(inputs)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(decoder.state_dict()), original_keys)
        self.assertTrue(all(layer.mlp.kernel_backend is KernelBackend.AUTO for layer in decoder.layers))
        self.assertTrue(
            all(layer.attn.flash_attention4_policy is FlashAttention4Policy.AUTO for layer in decoder.layers))

        self.assertIs(result.restore(), decoder)
        self.assertTrue(all(layer.mlp.kernel_backend is KernelBackend.TORCH for layer in decoder.layers))
        self.assertTrue(
            all(
                layer.attn.flash_attention4_policy is FlashAttention4Policy.DISABLED
                for layer in decoder.layers))


if __name__ == "__main__":
    unittest.main()
