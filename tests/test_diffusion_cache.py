from __future__ import annotations

import json
import unittest
from dataclasses import replace

import torch
from torch import nn

from voicehub.optimization import (
    DiffusionBlockResidualCache,
    DiffusionCacheConfig,
    DiffusionCacheMethod,
    DiffusionCacheMixin,
    DiffusionCachePass,
    DiffusionCachePolicy,
    DiffusionCachePredictor,
    DiffusionCacheRefreshPolicy,
    DiffusionCacheStepPolicy,
    OptimizationContext,
    TTSOptimizationCompatibilityError,
    TTSOptimizationConfig,
    diffusion_cache_request,
    diffusion_cache_summary,
    reset_diffusion_cache_metrics,
    resolve_tts_optimization,
)


def _context(mode: str = "inference") -> OptimizationContext:
    return OptimizationContext(
        mode=mode,
        device="cpu",
        dtype="float32",
    )


class _CountingBlock(nn.Module):

    def __init__(self, operation: str = "add", value: float = 1.0):
        super().__init__()
        self.operation = operation
        self.value = nn.Parameter(torch.tensor(float(value)))
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.operation == "add":
            return hidden_states + self.value
        if self.operation == "multiply":
            return hidden_states * self.value
        raise AssertionError(f"Unknown test operation: {self.operation}")


def _run(
    cache: DiffusionBlockResidualCache,
    blocks: tuple[_CountingBlock, ...],
    hidden_states: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return cache.run(
        blocks,
        hidden_states,
        lambda block, value: block(value),
        **kwargs,
    )


class _TinyCachedDiT(DiffusionCacheMixin, nn.Module):

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            _CountingBlock("add", 1.0),
            _CountingBlock("add", 2.0),
            _CountingBlock("add", 3.0),
            _CountingBlock("add", 4.0),
        ], )
        self._initialize_diffusion_cache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        lane: str = "default",
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._run_diffusion_blocks(
            self.blocks,
            hidden_states,
            lambda block, value: block(value),
            cache_lane=lane,
            valid_mask=valid_mask,
        )


class DiffusionCacheConfigurationTests(unittest.TestCase):

    def test_config_round_trips_as_strict_json(self):
        config = DiffusionCacheConfig.from_dict({
            "front_blocks": 2,
            "back_blocks": 1,
            "residual_diff_threshold": 0.125,
            "warmup_steps": 3,
            "warmup_interval": 2,
            "max_cached_steps": 11,
            "max_consecutive_cached_steps": 2,
            "max_accumulated_relative_error": 0.4,
            "predictor": "taylorseer",
            "taylor_order": 2,
            "compute_step_mask": [True, False, True],
            "compute_step_policy": "static",
            "force_refresh_step_hint": 7,
            "force_refresh_step_policy": "repeat",
            "probe_downsample_factor": 2,
            "metrics_history_size": 32,
            "synchronize_distributed": False,
            "epsilon": 1e-5,
        })

        payload = config.to_dict()
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)

        self.assertEqual(config.predictor, DiffusionCachePredictor.TAYLOR)
        self.assertEqual(config.taylor_order, 2)
        self.assertEqual(config.compute_step_policy, DiffusionCacheStepPolicy.STATIC)
        self.assertEqual(
            config.force_refresh_step_policy,
            DiffusionCacheRefreshPolicy.REPEAT,
        )
        self.assertEqual(config.method, DiffusionCacheMethod.DBCACHE)
        self.assertEqual(config.compute_step_mask, (True, False, True))
        self.assertEqual(json.loads(encoded), payload)
        self.assertEqual(DiffusionCacheConfig.from_dict(payload), config)
        self.assertEqual(
            DiffusionCachePolicy.coerce(True),
            DiffusionCachePolicy.REQUIRED,
        )
        self.assertEqual(
            DiffusionCachePolicy.coerce(False),
            DiffusionCachePolicy.DISABLED,
        )

    def test_invalid_configurations_fail_closed(self):
        invalid = (
            ({
                "front_blocks": 0
            }, ValueError, "front_blocks"),
            ({
                "back_blocks": -1
            }, ValueError, "back_blocks"),
            ({
                "residual_diff_threshold": float("nan")
            }, ValueError, "residual_diff_threshold"),
            ({
                "max_cached_steps": -2
            }, ValueError, "max_cached_steps"),
            ({
                "max_consecutive_cached_steps": -2
            }, ValueError, "max_consecutive_cached_steps"),
            ({
                "max_accumulated_relative_error": 0.0
            }, ValueError, "max_accumulated_relative_error"),
            ({
                "compute_step_mask": [True, 1]
            }, TypeError, "compute_step_mask"),
            ({
                "epsilon": 0.0
            }, ValueError, "epsilon"),
            ({
                "predictor": "not-a-predictor"
            }, ValueError, "predictor"),
            ({
                "taylor_order": 4
            }, ValueError, "taylor_order"),
            ({
                "warmup_interval": 0
            }, ValueError, "warmup_interval"),
            ({
                "probe_downsample_factor": 0
            }, ValueError, "probe_downsample_factor"),
            ({
                "metrics_history_size": -1
            }, ValueError, "metrics_history_size"),
            ({
                "compute_step_policy": "sometimes"
            }, ValueError, "step policy"),
            ({
                "force_refresh_step_policy": "sometimes"
            }, ValueError, "refresh policy"),
        )
        for values, error_type, message in invalid:
            with self.subTest(values=values):
                with self.assertRaisesRegex(error_type, message):
                    DiffusionCacheConfig(**values)
        with self.assertRaisesRegex(ValueError, "Unknown diffusion-cache"):
            DiffusionCachePolicy.coerce("sometimes")
        with self.assertRaisesRegex(ValueError, "First-block cache"):
            DiffusionCacheConfig(
                method="fbcache",
                front_blocks=2,
            )

    def test_official_cache_dit_option_names_are_accepted(self):
        config = DiffusionCacheConfig.from_dict({
            "Fn_compute_blocks": 2,
            "Bn_compute_blocks": 1,
            "max_warmup_steps": 4,
            "warmup_interval": 2,
            "max_continuous_cached_steps": 5,
            "max_accumulated_residual_diff_threshold": 0.3,
            "steps_computation_mask": [1, 0, 1],
            "steps_computation_policy": "static",
            "downsample_factor": 4,
            "taylorseer_order": 2,
        })

        self.assertEqual(config.front_blocks, 2)
        self.assertEqual(config.back_blocks, 1)
        self.assertEqual(config.warmup_steps, 4)
        self.assertEqual(config.max_consecutive_cached_steps, 5)
        self.assertEqual(config.compute_step_mask, (True, False, True))
        self.assertEqual(config.probe_downsample_factor, 4)
        with self.assertRaisesRegex(ValueError, "cannot set both"):
            DiffusionCacheConfig.from_dict({
                "Fn_compute_blocks": 2,
                "front_blocks": 1,
            })

    def test_keyword_overrides_win_across_official_aliases(self):
        config = DiffusionCacheConfig.from_dict(
            {"Fn_compute_blocks": 2},
            front_blocks=3,
        )

        self.assertEqual(config.front_blocks, 3)


class DiffusionBlockResidualCacheTests(unittest.TestCase):

    @staticmethod
    def _additive_blocks(count: int = 4) -> tuple[_CountingBlock, ...]:
        return tuple(_CountingBlock("add", float(index + 1)) for index in range(count))

    def test_cache_executes_first_and_last_but_skips_middle_on_hit(self):
        blocks = self._additive_blocks()
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                back_blocks=1,
                warmup_steps=0,
                max_consecutive_cached_steps=-1,
            ), )
        value = torch.ones(1, 2, 3)

        with torch.inference_mode():
            eager = _run(cache, blocks, value)
            cached = _run(cache, blocks, value)

        self.assertTrue(torch.equal(cached, eager))
        self.assertEqual([block.calls for block in blocks], [2, 1, 1, 2])

    def test_warmup_computes_middle_before_first_cache_hit(self):
        blocks = self._additive_blocks()
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                back_blocks=1,
                warmup_steps=2,
                max_consecutive_cached_steps=-1,
            ), )
        value = torch.ones(1, 2, 3)

        with torch.inference_mode():
            for _ in range(3):
                _run(cache, blocks, value)

        self.assertEqual([block.calls for block in blocks], [3, 2, 2, 3])
        self.assertEqual(cache.stats()["warmup_misses"], 1)
        self.assertEqual(cache.stats()["cached_steps"], 1)

    def test_threshold_selects_hits_and_recomputes_changed_probes(self):
        blocks = self._additive_blocks(3)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                residual_diff_threshold=0.1,
                warmup_steps=0,
                max_consecutive_cached_steps=-1,
            ), )
        baseline = torch.ones(1, 2, 2)

        with torch.inference_mode():
            _run(cache, blocks, baseline)
            _run(cache, blocks, baseline.clone())
            _run(cache, blocks, baseline * 4)

        self.assertEqual([block.calls for block in blocks], [3, 2, 2])
        stats = cache.stats()
        self.assertEqual(stats["cached_steps"], 1)
        self.assertEqual(stats["computed_steps"], 2)
        self.assertEqual(stats["threshold_misses"], 1)
        self.assertGreater(stats["last_relative_difference"], 0.1)

    def test_max_consecutive_hits_forces_a_periodic_refresh(self):
        blocks = self._additive_blocks(3)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=0,
                max_consecutive_cached_steps=1,
            ), )
        value = torch.ones(1, 2, 2)

        with torch.inference_mode():
            for _ in range(4):
                _run(cache, blocks, value)

        self.assertEqual([block.calls for block in blocks], [4, 2, 2])
        stats = cache.stats()
        self.assertEqual(stats["computed_steps"], 2)
        self.assertEqual(stats["cached_steps"], 2)
        self.assertEqual(stats["limit_misses"], 1)

    def test_cfg_lanes_do_not_share_probe_or_residual_state(self):
        blocks = self._additive_blocks(3)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=0,
                max_consecutive_cached_steps=-1,
            ), )

        with torch.inference_mode():
            for lane, value in (
                ("conditional", torch.ones(1, 2, 2)),
                ("unconditional", torch.full((1, 2, 2), 3.0)),
                ("conditional", torch.ones(1, 2, 2)),
                ("unconditional", torch.full((1, 2, 2), 3.0)),
            ):
                _run(cache, blocks, value, lane=lane)

        self.assertEqual([block.calls for block in blocks], [4, 2, 2])
        self.assertEqual(cache.stats()["computed_steps"], 2)
        self.assertEqual(cache.stats()["cached_steps"], 2)

    def test_request_session_cleans_state_even_after_an_exception(self):
        blocks = self._additive_blocks(3)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=0,
                max_consecutive_cached_steps=-1,
            ), )
        value = torch.ones(1, 2, 2)

        with self.assertRaisesRegex(RuntimeError, "request failed"):
            with cache.session():
                with torch.inference_mode():
                    _run(cache, blocks, value)
                    _run(cache, blocks, value)
                raise RuntimeError("request failed")
        self.assertEqual(cache._states, {})

        with cache.session():
            with torch.inference_mode():
                _run(cache, blocks, value)

        self.assertEqual(cache._states, {})
        self.assertEqual([block.calls for block in blocks], [3, 2, 2])
        self.assertEqual(cache.stats()["sessions"], 2)

    def test_valid_mask_ignores_padding_but_detects_valid_changes(self):
        blocks = self._additive_blocks(2)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=0,
                residual_diff_threshold=0.1,
                max_consecutive_cached_steps=-1,
                probe_downsample_factor=2,
            ), )
        mask = torch.tensor([[True, True, False]])
        baseline = torch.ones(1, 3, 1)
        padding_changed = baseline.clone()
        padding_changed[:, 2] = 100.0
        valid_changed = padding_changed.clone()
        valid_changed[:, 0] = 3.0

        with torch.inference_mode():
            _run(cache, blocks, baseline, valid_mask=mask)
            _run(cache, blocks, padding_changed, valid_mask=mask)
            _run(cache, blocks, valid_changed, valid_mask=mask)

        self.assertEqual([block.calls for block in blocks], [3, 2])
        self.assertEqual(cache.stats()["cached_steps"], 1)
        self.assertEqual(cache.stats()["threshold_misses"], 1)

    def test_gradient_and_training_calls_bypass_cache(self):
        gradient_blocks = self._additive_blocks(2)
        gradient_cache = DiffusionBlockResidualCache(DiffusionCacheConfig(front_blocks=1, warmup_steps=0), )
        first = torch.ones(1, 2, 2, requires_grad=True)
        second = torch.ones(1, 2, 2, requires_grad=True)

        with torch.enable_grad():
            output = _run(gradient_cache, gradient_blocks, first)
            output.sum().backward()
            _run(gradient_cache, gradient_blocks, second)

        self.assertIsNotNone(first.grad)
        self.assertEqual(
            [block.calls for block in gradient_blocks],
            [2, 2],
        )
        self.assertEqual(gradient_cache.stats()["bypassed_steps"], 2)
        self.assertEqual(gradient_cache.stats()["computed_steps"], 0)

        training_blocks = self._additive_blocks(2)
        training_cache = DiffusionBlockResidualCache(DiffusionCacheConfig(front_blocks=1, warmup_steps=0), )
        with torch.no_grad():
            _run(
                training_cache,
                training_blocks,
                torch.ones(1, 2, 2),
                training=True,
            )
            _run(
                training_cache,
                training_blocks,
                torch.ones(1, 2, 2),
                training=True,
            )
        self.assertEqual(
            [block.calls for block in training_blocks],
            [2, 2],
        )
        self.assertEqual(training_cache.stats()["bypassed_steps"], 2)
        self.assertEqual(training_cache.stats()["cached_steps"], 0)

    def test_taylor_predictor_uses_computed_step_distance(self):
        front = _CountingBlock("add", 0.0)
        middle = _CountingBlock("multiply", 2.0)
        blocks = (front, middle)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=2,
                residual_diff_threshold=1.0,
                max_consecutive_cached_steps=-1,
                predictor="taylor",
            ), )

        with torch.inference_mode():
            first = _run(cache, blocks, torch.tensor([[[1.0]]]))
            second = _run(cache, blocks, torch.tensor([[[2.0]]]))
            predicted = _run(cache, blocks, torch.tensor([[[3.0]]]))

        self.assertEqual(first.item(), 2.0)
        self.assertEqual(second.item(), 4.0)
        self.assertEqual(predicted.item(), 6.0)
        self.assertEqual(middle.calls, 2)
        self.assertEqual(cache.stats()["cached_steps"], 1)

    def test_higher_order_taylor_predictor_uses_only_computed_history(self):
        blocks = (
            _CountingBlock("add", 0.0),
            _CountingBlock("multiply", 2.0),
        )
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=3,
                residual_diff_threshold=1.0,
                max_consecutive_cached_steps=-1,
                predictor="taylor",
                taylor_order=2,
            ), )

        with torch.inference_mode():
            _run(cache, blocks, torch.tensor([[[1.0]]]))
            _run(cache, blocks, torch.tensor([[[2.0]]]))
            _run(cache, blocks, torch.tensor([[[3.0]]]))
            predicted = _run(cache, blocks, torch.tensor([[[4.0]]]))

        self.assertEqual(predicted.item(), 8.0)
        self.assertEqual(blocks[1].calls, 3)
        self.assertEqual(cache.stats()["taylor_predictions"], 1)
        self.assertEqual(cache.stats()["maximum_taylor_order_used"], 2)

    def test_warmup_interval_allows_cache_between_refresh_steps(self):
        blocks = self._additive_blocks(3)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=4,
                warmup_interval=2,
                max_consecutive_cached_steps=-1,
            ), )

        with torch.inference_mode():
            for _ in range(4):
                _run(cache, blocks, torch.ones(1, 2, 2))

        stats = cache.stats()
        self.assertEqual(stats["computed_steps"], 2)
        self.assertEqual(stats["cached_steps"], 2)
        self.assertEqual(stats["warmup_misses"], 1)

    def test_static_step_mask_skips_threshold_but_retains_forced_compute_steps(self):
        blocks = self._additive_blocks(3)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=0,
                residual_diff_threshold=1e-9,
                max_consecutive_cached_steps=-1,
                compute_step_mask=(True, False, True, False),
                compute_step_policy="static",
            ), )

        with torch.inference_mode():
            for scale in (1.0, 10.0, 20.0, 30.0):
                _run(cache, blocks, torch.full((1, 2, 2), scale))

        stats = cache.stats()
        self.assertEqual(stats["computed_steps"], 2)
        self.assertEqual(stats["cached_steps"], 2)
        self.assertEqual(stats["static_hits"], 2)
        self.assertEqual(stats["mask_misses"], 1)
        self.assertEqual(stats["threshold_misses"], 0)

    def test_inference_and_forced_refreshes_start_new_cache_segments(self):
        blocks = self._additive_blocks(3)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=0,
                max_consecutive_cached_steps=-1,
                num_inference_steps=2,
            ), )
        with torch.inference_mode():
            for _ in range(5):
                _run(cache, blocks, torch.ones(1, 2, 2))
        self.assertEqual(cache.stats()["inference_refreshes"], 2)
        self.assertEqual(cache.stats()["computed_steps"], 3)

        forced = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=0,
                max_consecutive_cached_steps=-1,
                force_refresh_step_hint=2,
                force_refresh_step_policy="repeat",
            ), )
        forced_blocks = self._additive_blocks(3)
        with torch.inference_mode():
            for _ in range(5):
                _run(forced, forced_blocks, torch.ones(1, 2, 2))
        self.assertEqual(forced.stats()["forced_refreshes"], 2)

    def test_runtime_stats_report_cache_effectiveness(self):
        blocks = self._additive_blocks(3)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=0,
                max_consecutive_cached_steps=-1,
            ), )
        value = torch.ones(1, 2, 2)

        with torch.inference_mode():
            _run(cache, blocks, value)
            _run(cache, blocks, value)

        stats = cache.stats(details=True)
        self.assertEqual(stats["calls"], 2)
        self.assertEqual(stats["computed_steps"], 1)
        self.assertEqual(stats["cached_steps"], 1)
        self.assertEqual(stats["cold_misses"], 1)
        self.assertEqual(stats["hit_rate"], 0.5)
        self.assertEqual(stats["total_block_evaluations"], 6)
        self.assertEqual(stats["executed_block_evaluations"], 4)
        self.assertEqual(stats["skipped_block_evaluations"], 2)
        self.assertAlmostEqual(stats["block_compute_reduction"], 1 / 3)
        self.assertEqual(stats["estimated_block_speedup"], 1.5)
        self.assertEqual(stats["residual_differences"], [0.0])
        self.assertEqual(stats["cached_step_indices"], [1])
        self.assertEqual(stats["computed_step_indices"], [0])
        self.assertEqual(stats["lanes"]["default"]["cached_steps"], 1)
        self.assertGreater(stats["peak_cache_bytes"], 0)
        self.assertEqual(reset_diffusion_cache_metrics(_TinyCachedDiT()), 1)
        cache.reset_stats()
        self.assertEqual(cache.stats()["calls"], 0)


class DiffusionCachePassTests(unittest.TestCase):

    def test_request_context_isolates_and_releases_every_target(self):
        model = _TinyCachedDiT().eval()
        model.enable_diffusion_cache({
            "warmup_steps": 0,
            "max_consecutive_cached_steps": -1,
        })

        with diffusion_cache_request(model):
            with torch.inference_mode():
                model(torch.ones(1, 2, 2))
                model(torch.ones(1, 2, 2))
            self.assertEqual(
                model.diffusion_cache_stats()["active_cache_entries"],
                1,
            )

        stats = model.diffusion_cache_stats()
        self.assertEqual(stats["sessions"], 1)
        self.assertEqual(stats["cached_steps"], 1)
        self.assertEqual(stats["active_cache_entries"], 0)

    def test_pass_is_reversible_and_preserves_state_dict_keys(self):
        model = _TinyCachedDiT().eval()
        original_keys = tuple(model.state_dict())
        previous = DiffusionCacheConfig(
            front_blocks=1,
            warmup_steps=1,
            residual_diff_threshold=0.2,
        )
        model.enable_diffusion_cache(previous)
        optimization_pass = DiffusionCachePass({
            "front_blocks": 1,
            "back_blocks": 1,
            "warmup_steps": 0,
            "max_consecutive_cached_steps": -1,
        })
        context = _context()

        optimization_pass.validate(model, context)
        result = optimization_pass.apply(model, context)

        self.assertIs(result.model, model)
        self.assertEqual(
            model.diffusion_cache_config,
            optimization_pass.config,
        )
        self.assertEqual(tuple(model.state_dict()), original_keys)
        self.assertEqual(result.metadata["targets"], ["model"])

        with torch.inference_mode():
            model(torch.ones(1, 2, 2))
            model(torch.ones(1, 2, 2))
        runtime = optimization_pass.runtime_manifest_status(result)
        self.assertEqual(runtime["model"]["calls"], 2)
        self.assertEqual(runtime["model"]["cached_steps"], 1)
        self.assertEqual(
            diffusion_cache_summary(model, details=True)["model"]["cached_step_indices"],
            [1],
        )

        restored = optimization_pass.restore(model, result.state, context)

        self.assertIs(restored, model)
        self.assertEqual(model.diffusion_cache_config, previous)
        self.assertEqual(tuple(model.state_dict()), original_keys)


class NativeDiffusionCacheAdapterTests(unittest.TestCase):

    @staticmethod
    def _cache_config() -> DiffusionCacheConfig:
        return DiffusionCacheConfig(
            front_blocks=1,
            warmup_steps=0,
            residual_diff_threshold=1.0,
            max_consecutive_cached_steps=-1,
        )

    def test_f5_and_cosyvoice_flat_dit_adapters_take_cache_hits(self):
        from voicehub.models.cosyvoice.native.configuration import CosyVoiceArchitectureConfig
        from voicehub.models.cosyvoice.native.flow import DiTEstimator
        from voicehub.models.f5tts.native.configuration import F5TTSArchitectureConfig
        from voicehub.models.f5tts.native.modeling import F5DiT

        f5_config = F5TTSArchitectureConfig(
            model_name="cache-test",
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
        f5 = F5DiT(f5_config).eval()
        f5.enable_diffusion_cache(self._cache_config())
        hidden = torch.randn(1, 4, 8)
        conditioning = torch.randn_like(hidden)
        token_ids = torch.ones(1, 4, dtype=torch.long)
        timestep = torch.tensor([0.5])
        mask = torch.ones(1, 4, dtype=torch.bool)

        cosy_config = CosyVoiceArchitectureConfig.tiny().flow
        cosy = DiTEstimator(cosy_config).eval()
        cosy.enable_diffusion_cache(self._cache_config())
        values = torch.randn(1, cosy_config.mel_channels, 4)
        flow_mask = torch.ones(1, 1, 4)
        means = torch.randn_like(values)
        speakers = torch.randn(1, cosy_config.mel_channels)
        flow_conditioning = torch.randn_like(values)

        with torch.inference_mode():
            first_f5 = f5(
                hidden,
                conditioning,
                token_ids,
                timestep,
                mask=mask,
            )
            second_f5 = f5(
                hidden,
                conditioning,
                token_ids,
                timestep,
                mask=mask,
            )
            first_cosy = cosy(
                values,
                flow_mask,
                means,
                timestep,
                speakers,
                flow_conditioning,
                diffusion_cache_lane="conditional",
            )
            second_cosy = cosy(
                values,
                flow_mask,
                means,
                timestep,
                speakers,
                flow_conditioning,
                diffusion_cache_lane="conditional",
            )

        torch.testing.assert_close(second_f5, first_f5)
        torch.testing.assert_close(second_cosy, first_cosy)
        self.assertEqual(f5.diffusion_cache_stats()["cached_steps"], 1)
        self.assertEqual(cosy.diffusion_cache_stats()["cached_steps"], 1)

    def test_irodori_and_vibevoice_custom_adapters_take_cache_hits(self):
        from voicehub.models.irodoritts.native.configuration import IrodoriModelConfig
        from voicehub.models.irodoritts.native.modeling import TextToLatentRFDiT
        from voicehub.models.vibevoice.native.configuration import VibeVoiceDiffusionConfig
        from voicehub.models.vibevoice.native.diffusion import VibeVoiceDiffusionHead

        irodori_config = IrodoriModelConfig(
            adaln_rank=8,
            latent_dim=4,
            mlp_ratio=2.0,
            model_dim=16,
            num_heads=2,
            num_layers=2,
            speaker_dim=16,
            speaker_heads=2,
            speaker_layers=1,
            speaker_mlp_ratio=2.0,
            text_dim=16,
            text_heads=2,
            text_layers=1,
            text_mlp_ratio=2.0,
            text_vocab_size=266,
            timestep_embed_dim=8,
            variant="custom",
        )
        irodori = TextToLatentRFDiT(irodori_config).eval()
        irodori.enable_diffusion_cache(self._cache_config())
        latent = torch.randn(1, 3, 4)
        timestep = torch.tensor([0.5])
        text_ids = torch.ones(1, 3, dtype=torch.long)
        text_mask = torch.ones(1, 3, dtype=torch.bool)
        reference = torch.randn(1, 2, 4)
        reference_mask = torch.ones(1, 2, dtype=torch.bool)
        latent_mask = torch.ones(1, 3, dtype=torch.bool)

        vibe_config = VibeVoiceDiffusionConfig(
            hidden_size=8,
            head_layers=2,
            head_ffn_ratio=2.0,
            latent_size=2,
            ddpm_num_steps=10,
            ddpm_num_inference_steps=2,
            ddpm_batch_mul=1,
        )
        vibe = VibeVoiceDiffusionHead(vibe_config).eval()
        vibe.enable_diffusion_cache(self._cache_config())
        noisy_latents = torch.randn(2, 2)
        vibe_timesteps = torch.ones(2)
        vibe_condition = torch.randn(2, 8)

        with torch.inference_mode():
            first_irodori = irodori(
                latent,
                timestep,
                text_ids,
                text_mask,
                reference,
                reference_mask,
                latent_mask=latent_mask,
            )
            second_irodori = irodori(
                latent,
                timestep,
                text_ids,
                text_mask,
                reference,
                reference_mask,
                latent_mask=latent_mask,
            )
            first_vibe = vibe(
                noisy_latents,
                vibe_timesteps,
                vibe_condition,
            )
            second_vibe = vibe(
                noisy_latents,
                vibe_timesteps,
                vibe_condition,
            )

        torch.testing.assert_close(second_irodori, first_irodori)
        torch.testing.assert_close(second_vibe, first_vibe)
        self.assertEqual(irodori.diffusion_cache_stats()["cached_steps"], 1)
        self.assertEqual(vibe.diffusion_cache_stats()["cached_steps"], 1)

    def test_chatterbox_mid_blocks_take_cache_hits(self):
        from voicehub.models.chatterbox.models.s3gen.decoder import ConditionalDecoder

        decoder = ConditionalDecoder(
            in_channels=8,
            out_channels=4,
            causal=False,
            channels=[8],
            dropout=0.0,
            attention_head_dim=4,
            n_blocks=1,
            num_mid_blocks=3,
            num_heads=2,
        ).eval()
        state_keys = tuple(decoder.state_dict())
        decoder.enable_diffusion_cache(self._cache_config())
        values = torch.randn(1, 4, 5)
        means = torch.randn_like(values)
        mask = torch.ones(1, 1, 5)
        timestep = torch.tensor([0.5])

        with torch.inference_mode():
            first = decoder(values, mask, means, timestep)
            second = decoder(values, mask, means, timestep)

        torch.testing.assert_close(second, first)
        self.assertEqual(decoder.diffusion_cache_stats()["cached_steps"], 1)
        self.assertEqual(tuple(decoder.state_dict()), state_keys)

    def test_style_transformers_keep_cfg_lanes_separate_and_take_hits(self):
        from voicehub.models.styletts2.source.styletts2.Modules.diffusion.modules import (
            StyleTransformer1d,
            Transformer1d,
        )

        for transformer_type in (Transformer1d, StyleTransformer1d):
            with self.subTest(transformer=transformer_type.__name__):
                transformer = transformer_type(
                    num_layers=3,
                    channels=4,
                    num_heads=1,
                    head_features=8,
                    multiplier=2,
                    context_features=4,
                    context_embedding_features=4,
                    embedding_max_length=8,
                ).eval()
                state_keys = tuple(transformer.state_dict())
                transformer.enable_diffusion_cache(self._cache_config())
                values = torch.randn(1, 1, 4)
                timestep = torch.tensor([0.5])
                embedding = torch.randn(1, 3, 4)
                features = torch.randn(1, 4)

                with torch.inference_mode():
                    first = transformer(
                        values,
                        timestep,
                        embedding=embedding,
                        features=features,
                        embedding_scale=2.0,
                    )
                    second = transformer(
                        values,
                        timestep,
                        embedding=embedding,
                        features=features,
                        embedding_scale=2.0,
                    )

                torch.testing.assert_close(second, first)
                self.assertEqual(
                    transformer.diffusion_cache_stats()["cached_steps"],
                    2,
                )
                self.assertEqual(tuple(transformer.state_dict()), state_keys)

    def test_supertonic_caches_latent_residual_without_stalling(self):
        from tests.test_native_supertonic_runtime import _runtime
        from voicehub.models.supertonic.native.frontend import SupertonicStyle

        runtime = _runtime().eval()
        state_keys = tuple(runtime.state_dict())
        runtime.enable_diffusion_cache(
            DiffusionCacheConfig(
                front_blocks=1,
                back_blocks=1,
                warmup_steps=0,
                residual_diff_threshold=1_000_000.0,
                max_consecutive_cached_steps=-1,
            ), )
        style = SupertonicStyle(
            ttl=torch.zeros(1, 50, 256),
            duration=torch.zeros(1, 8, 16),
        )

        with torch.inference_mode():
            waveform, duration = runtime.infer_batch(
                ("hello", ),
                ("en", ),
                style,
                total_steps=3,
                generator=torch.Generator().manual_seed(11),
            )

        self.assertEqual(waveform.shape[0], 1)
        self.assertEqual(duration.shape, (1, ))
        self.assertEqual(runtime.diffusion_cache_stats()["computed_steps"], 1)
        self.assertEqual(runtime.diffusion_cache_stats()["cached_steps"], 2)
        self.assertEqual(tuple(runtime.state_dict()), state_keys)

    def test_echo_and_voxcpm_native_dits_take_cache_hits(self):
        from voicehub.models.echo.model import EchoDiT
        from voicehub.models.voxcpm.native.configuration import VoxCPM2ArchitectureConfig
        from voicehub.models.voxcpm.native.modeling import VoxCPM2Model

        echo = EchoDiT(
            latent_size=4,
            model_size=8,
            num_layers=3,
            num_heads=2,
            intermediate_size=16,
            norm_eps=1e-5,
            text_vocab_size=32,
            text_model_size=8,
            text_num_layers=1,
            text_num_heads=2,
            text_intermediate_size=16,
            speaker_patch_size=2,
            speaker_model_size=8,
            speaker_num_layers=1,
            speaker_num_heads=2,
            speaker_intermediate_size=16,
            timestep_embed_size=8,
            adaln_rank=4,
        ).eval()
        echo.enable_diffusion_cache(self._cache_config())
        text_ids = torch.ones(1, 3, dtype=torch.long)
        text_mask = torch.ones(1, 3, dtype=torch.bool)
        speaker = torch.randn(1, 4, 4)
        speaker_mask = torch.ones(1, 4, dtype=torch.bool)
        text_cache = echo.get_kv_cache_text(text_ids, text_mask)
        speaker_cache = echo.get_kv_cache_speaker(speaker)
        latent = torch.randn(1, 5, 4)
        timestep = torch.tensor([0.5])

        voxcpm_config = VoxCPM2ArchitectureConfig.tiny()
        voxcpm_config = replace(
            voxcpm_config,
            dit_config=replace(voxcpm_config.dit_config, num_layers=3),
        )
        voxcpm = VoxCPM2Model(voxcpm_config).eval().feat_decoder.estimator
        voxcpm.enable_diffusion_cache(self._cache_config())
        values = torch.randn(1, voxcpm_config.feat_dim, voxcpm_config.patch_size)
        condition = torch.randn(1, voxcpm_config.dit_config.hidden_dim)
        prefix = torch.randn_like(values)

        with torch.inference_mode():
            first_echo = echo(
                latent,
                timestep,
                text_mask,
                speaker_mask,
                text_cache,
                speaker_cache,
            )
            second_echo = echo(
                latent,
                timestep,
                text_mask,
                speaker_mask,
                text_cache,
                speaker_cache,
            )
            first_voxcpm = voxcpm(
                values,
                condition,
                timestep,
                prefix,
                torch.zeros_like(timestep),
            )
            second_voxcpm = voxcpm(
                values,
                condition,
                timestep,
                prefix,
                torch.zeros_like(timestep),
            )

        torch.testing.assert_close(second_echo, first_echo)
        torch.testing.assert_close(second_voxcpm, first_voxcpm)
        self.assertEqual(echo.diffusion_cache_stats()["cached_steps"], 1)
        self.assertEqual(voxcpm.diffusion_cache_stats()["cached_steps"], 1)


class UniversalDiffusionCacheResolutionTests(unittest.TestCase):

    def test_all_registered_diffusion_families_resolve_required_cache(self):
        model_types = (
            "chatterbox",
            "cosyvoice",
            "echo",
            "f5tts",
            "irodoritts",
            "styletts2",
            "supertonic",
            "vibevoice",
            "voxcpm",
        )
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                plan = resolve_tts_optimization(
                    model_type,
                    TTSOptimizationConfig(
                        attn_implementation="native",
                        kernel_backend="native",
                        diffusion_cache="required",
                        compile=False,
                    ),
                    context=_context(),
                )
                self.assertEqual(
                    [item.compatibility_kind for item in plan.passes],
                    ["diffusion-cache"],
                )

    def test_supported_cache_is_ordered_before_compile(self):
        plan = resolve_tts_optimization(
            "f5tts",
            TTSOptimizationConfig(
                attn_implementation="native",
                kernel_backend="native",
                diffusion_cache="required",
                diffusion_cache_config={
                    "front_blocks": 2,
                    "predictor": "taylor",
                },
                compile=True,
                compile_config={"backend": "eager"},
            ),
            context=_context(),
        )

        self.assertEqual(
            [item.compatibility_kind for item in plan.passes],
            ["diffusion-cache", "compile"],
        )
        self.assertEqual(
            [decision.feature for decision in plan.decisions],
            [
                "kernels",
                "attention",
                "diffusion_sampling",
                "diffusion_cache",
                "compile",
            ],
        )
        cache_decision = plan.decisions[3]
        self.assertEqual(cache_decision.requested, "required")
        self.assertEqual(
            cache_decision.selected,
            "block-residual-cache:dbcache:taylor",
        )

    def test_unsupported_auto_falls_back_and_required_fails(self):
        automatic = resolve_tts_optimization(
            "vits",
            TTSOptimizationConfig(
                attn_implementation="native",
                kernel_backend="native",
                diffusion_cache="auto",
                compile=False,
            ),
            context=_context(),
        )
        decision = next(item for item in automatic.decisions if item.feature == "diffusion_cache")

        self.assertEqual(automatic.passes, ())
        self.assertEqual(decision.selected, "unsupported")
        self.assertIn("does not declare", decision.reason)
        with self.assertRaisesRegex(
                TTSOptimizationCompatibilityError,
                "does not declare an architecture-owned diffusion-cache",
        ):
            resolve_tts_optimization(
                "vits",
                TTSOptimizationConfig(
                    attn_implementation="native",
                    kernel_backend="native",
                    diffusion_cache="required",
                    compile=False,
                ),
                context=_context(),
            )


if __name__ == "__main__":
    unittest.main()
