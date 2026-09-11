from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from tests.test_native_supertonic_runtime import _runtime
from voicehub.models.chatterbox.models.s3gen.flow_matching import ConditionalCFM
from voicehub.models.chatterbox.native.registration import create_chatterbox_architecture_spec
from voicehub.models.chatterbox.tts import ChatterboxTTS
from voicehub.models.styletts2.native.registration import create_styletts2_architecture_spec
from voicehub.models.styletts2.native.runtime import StyleTTS2Runtime
from voicehub.models.styletts2.native.sampling import StyleTTS2DiffusionSampler
from voicehub.models.styletts2.source.styletts2.Modules.diffusion.sampler import (
    ADPM2Sampler,
    KarrasSchedule,
    KDiffusion,
    LogNormalDistribution,
)
from voicehub.models.supertonic.native.frontend import SupertonicStyle
from voicehub.models.supertonic.native.registration import create_supertonic_architecture_spec
from voicehub.optimization import (
    OptimizationContext,
    TTSOptimizationCompatibilityError,
    TTSOptimizationConfig,
    resolve_tts_optimization,
)
from voicehub.optimization.diffusion_sampling import DiffusionSamplingCompatibilityError, DiffusionSamplingConfig


class _ChatterboxEstimator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, x, mask, mu, time, speakers, conditioning):
        del mask, time, speakers, conditioning
        self.calls += 1
        return (x + mu) * self.scale


class _StyleDenoiser(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.calls = 0

    def forward(self, values, time, **_kwargs):
        del time
        self.calls += 1
        return values * self.scale


def _chatterbox_flow() -> ConditionalCFM:
    return ConditionalCFM(
        in_channels=80,
        cfm_params=SimpleNamespace(
            inference_cfg_rate=0.7,
            sigma_min=1e-4,
            solver="euler",
            t_scheduler="linear",
            training_cfg_rate=0.0,
        ),
        n_spks=1,
        spk_emb_dim=80,
        estimator=_ChatterboxEstimator(),
    )


def _style_sampler(denoiser: nn.Module, ) -> StyleTTS2DiffusionSampler:
    diffusion = KDiffusion(
        net=denoiser,
        sigma_distribution=LogNormalDistribution(0.0, 1.0),
        sigma_data=1.0,
    )
    return StyleTTS2DiffusionSampler(
        diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(
            sigma_min=0.0001,
            sigma_max=3.0,
            rho=9.0,
        ),
        clamp=False,
    )


class SpecializedDiffusionSamplingAdapterTests(unittest.TestCase):

    def test_enabled_exact_paths_preserve_native_outputs(self) -> None:
        native_chatterbox = _chatterbox_flow()
        adapted_chatterbox = _chatterbox_flow()
        adapted_chatterbox.load_state_dict(native_chatterbox.state_dict())
        adapted_chatterbox.enable_diffusion_sampling(DiffusionSamplingConfig())
        chatterbox_inputs = {
            "mu": torch.zeros(1, 80, 3),
            "mask": torch.ones(1, 1, 3),
            "n_timesteps": 4,
            "spks": torch.zeros(1, 80),
            "cond": torch.zeros(1, 80, 3),
        }
        torch.manual_seed(3)
        native_mel, _ = native_chatterbox(**chatterbox_inputs)
        torch.manual_seed(3)
        adapted_mel, _ = adapted_chatterbox(**chatterbox_inputs)
        torch.testing.assert_close(
            adapted_mel,
            native_mel,
            rtol=0,
            atol=0,
        )

        native_style = _style_sampler(_StyleDenoiser())
        adapted_style = _style_sampler(_StyleDenoiser())
        adapted_style.load_state_dict(native_style.state_dict())
        adapted_style.enable_diffusion_sampling(DiffusionSamplingConfig())
        style_noise = torch.ones(1, 1, 4)
        torch.manual_seed(5)
        native_style_output = native_style(style_noise, num_steps=5)
        torch.manual_seed(5)
        adapted_style_output = adapted_style(style_noise, num_steps=5)
        torch.testing.assert_close(
            adapted_style_output,
            native_style_output,
            rtol=0,
            atol=0,
        )

        native_supertonic = _runtime()
        adapted_supertonic = _runtime()
        adapted_supertonic.load_state_dict(native_supertonic.state_dict())
        adapted_supertonic.enable_diffusion_sampling(DiffusionSamplingConfig())
        style = SupertonicStyle(
            ttl=torch.zeros(1, 50, 256),
            duration=torch.zeros(1, 8, 16),
        )
        native_audio, native_duration = native_supertonic.infer_batch(
            ("hello", ),
            ("en", ),
            style,
            total_steps=5,
            generator=torch.Generator().manual_seed(7),
        )
        adapted_audio, adapted_duration = adapted_supertonic.infer_batch(
            ("hello", ),
            ("en", ),
            style,
            total_steps=5,
            generator=torch.Generator().manual_seed(7),
        )
        torch.testing.assert_close(
            adapted_audio,
            native_audio,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            adapted_duration,
            native_duration,
            rtol=0,
            atol=0,
        )

    def test_chatterbox_compacts_euler_schedule_and_reuses_guided_velocity(self, ) -> None:
        flow = _chatterbox_flow()
        state_keys = tuple(flow.state_dict())
        flow.enable_diffusion_sampling(
            DiffusionSamplingConfig(
                target_steps=2,
                prediction_cache="fora",
                cache_interval=2,
                cache_warmup_steps=0,
            ))

        output, _cache = flow(
            mu=torch.zeros(1, 80, 3),
            mask=torch.ones(1, 1, 3),
            n_timesteps=4,
            spks=torch.zeros(1, 80),
            cond=torch.zeros(1, 80, 3),
        )

        self.assertEqual(output.shape, (1, 80, 3))
        self.assertEqual(flow.estimator.calls, 1)
        self.assertEqual(tuple(flow.state_dict()), state_keys)
        stats = flow.diffusion_sampling_stats()
        self.assertEqual(stats["native_steps"], 4)
        self.assertEqual(stats["prepared_steps"], 2)
        self.assertEqual(stats["model_calls"], 1)
        self.assertEqual(stats["predicted_calls"], 1)

    def test_chatterbox_rejects_cfg_pruning_and_stork(self) -> None:
        flow = _chatterbox_flow()
        with self.assertRaisesRegex(
                DiffusionSamplingCompatibilityError,
                "fixed two-branch CFG",
        ):
            flow.enable_diffusion_sampling(DiffusionSamplingConfig(guidance="limited_interval"))
        with self.assertRaisesRegex(
                DiffusionSamplingCompatibilityError,
                "STORK-2",
        ):
            flow.enable_diffusion_sampling(DiffusionSamplingConfig(solver="stork2"))

    def test_styletts2_compacts_the_active_adpm2_sigma_span(self) -> None:
        denoiser = _StyleDenoiser()
        sampler = _style_sampler(denoiser)
        state_keys = tuple(sampler.state_dict())
        sampler.enable_diffusion_sampling(DiffusionSamplingConfig(target_steps=2))

        torch.manual_seed(11)
        result = sampler(
            torch.ones(1, 1, 4),
            num_steps=5,
        )

        self.assertEqual(result.shape, (1, 1, 4))
        self.assertEqual(denoiser.calls, 4)
        self.assertEqual(tuple(sampler.state_dict()), state_keys)
        stats = sampler.diffusion_sampling_stats()
        self.assertEqual(stats["native_steps"], 4)
        self.assertEqual(stats["prepared_steps"], 2)

    def test_styletts2_rejects_hidden_cfg_cache_and_stork_surfaces(self, ) -> None:
        sampler = _style_sampler(_StyleDenoiser())
        configurations = (
            (
                DiffusionSamplingConfig(guidance="adaptive"),
                "CFG inside each ADPM2",
            ),
            (
                DiffusionSamplingConfig(prediction_cache="fora"),
                "stochastic midpoint",
            ),
            (
                DiffusionSamplingConfig(solver="stork2"),
                "direct velocity",
            ),
        )
        for config, message in configurations:
            with self.subTest(config=config), self.assertRaisesRegex(
                    DiffusionSamplingCompatibilityError,
                    message,
            ):
                sampler.enable_diffusion_sampling(config)

    def test_supertonic_rebuilds_its_discrete_step_recurrence(self) -> None:
        runtime = _runtime()
        state_keys = tuple(runtime.state_dict())
        runtime.enable_diffusion_sampling(DiffusionSamplingConfig(target_steps=2))
        style = SupertonicStyle(
            ttl=torch.zeros(1, 50, 256),
            duration=torch.zeros(1, 8, 16),
        )

        with patch.object(
                runtime.vector_estimator,
                "forward",
                wraps=runtime.vector_estimator.forward,
        ) as estimator:
            waveform, duration = runtime.infer_batch(
                ("hello", ),
                ("en", ),
                style,
                total_steps=5,
                generator=torch.Generator().manual_seed(17),
            )

        self.assertEqual(estimator.call_count, 2)
        self.assertEqual(waveform.shape[0], 1)
        self.assertEqual(duration.shape, (1, ))
        self.assertEqual(tuple(runtime.state_dict()), state_keys)
        stats = runtime.diffusion_sampling_stats()
        self.assertEqual(stats["native_steps"], 5)
        self.assertEqual(stats["prepared_steps"], 2)

    def test_supertonic_rejects_non_discrete_sampling_methods(self) -> None:
        runtime = _runtime()
        configurations = (
            (
                DiffusionSamplingConfig(schedule="quadratic"),
                "discrete current_step",
            ),
            (
                DiffusionSamplingConfig(guidance="limited_interval"),
                "no classifier-free-guidance",
            ),
            (
                DiffusionSamplingConfig(prediction_cache="taylor"),
                "next absolute latent",
            ),
            (
                DiffusionSamplingConfig(solver="stork2"),
                "direct continuous velocity",
            ),
        )
        for config, message in configurations:
            with self.subTest(config=config), self.assertRaisesRegex(
                    DiffusionSamplingCompatibilityError,
                    message,
            ):
                runtime.enable_diffusion_sampling(config)

    def test_architectures_declare_only_the_reviewed_sampling_surface(self, ) -> None:
        expected = (
            (
                create_chatterbox_architecture_spec(),
                (
                    "schedule",
                    "prediction-cache",
                ),
            ),
            (
                create_styletts2_architecture_spec(),
                ("schedule", ),
            ),
            (
                create_supertonic_architecture_spec(),
                ("discrete-step-count", ),
            ),
        )
        for spec, capabilities in expected:
            with self.subTest(architecture=spec.architecture_id):
                self.assertIn(
                    "diffusion-sampling",
                    spec.capabilities.optimization_passes,
                )
                self.assertEqual(
                    spec.metadata["diffusion_sampling_capabilities"],
                    capabilities,
                )

    def test_universal_resolver_accepts_only_declared_techniques(self) -> None:
        context = OptimizationContext(
            mode="inference",
            device="cpu",
            dtype="float32",
        )
        supported = {
            "chatterbox": {
                "target_steps": 4,
                "prediction_cache": "fora",
            },
            "styletts2": {
                "target_steps": 3,
            },
            "supertonic": {
                "target_steps": 2,
            },
        }
        for model_type, sampling_config in supported.items():
            with self.subTest(model_type=model_type):
                plan = resolve_tts_optimization(
                    model_type,
                    TTSOptimizationConfig(
                        compile=False,
                        attn_implementation="native",
                        kernel_backend="native",
                        diffusion_sampling="required",
                        diffusion_sampling_config=sampling_config,
                    ),
                    context=context,
                )
                self.assertIn(
                    "voicehub.diffusion-sampling@1",
                    tuple(item.qualified_id for item in plan),
                )

        unsupported = (
            ("chatterbox", {
                "guidance": "limited_interval",
            }),
            ("styletts2", {
                "prediction_cache": "fora",
            }),
            ("supertonic", {
                "solver": "stork2",
            }),
        )
        for model_type, sampling_config in unsupported:
            with self.subTest(model_type=model_type), self.assertRaises(TTSOptimizationCompatibilityError):
                resolve_tts_optimization(
                    model_type,
                    TTSOptimizationConfig(
                        compile=False,
                        attn_implementation="native",
                        kernel_backend="native",
                        diffusion_sampling="required",
                        diffusion_sampling_config=sampling_config,
                    ),
                    context=context,
                )

    def test_plain_runtimes_expose_sampler_modules_to_the_pass(self) -> None:
        chatterbox = object.__new__(ChatterboxTTS)
        chatterbox.t3 = nn.Linear(1, 1)
        chatterbox.s3gen = nn.Sequential(_chatterbox_flow())
        chatterbox.ve = nn.Linear(1, 1)
        self.assertEqual(
            tuple(label for label, _module in chatterbox.optimization_module_roots()),
            ("t3", "s3gen", "voice_encoder"),
        )

        styletts2 = object.__new__(StyleTTS2Runtime)
        styletts2.model = nn.Linear(1, 1)
        styletts2.sampler = _style_sampler(_StyleDenoiser())
        self.assertEqual(
            tuple(root.label for root in styletts2.optimization_module_roots()),
            ("model", "sampler"),
        )


if __name__ == "__main__":
    unittest.main()
