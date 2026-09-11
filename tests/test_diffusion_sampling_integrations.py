from __future__ import annotations

from types import SimpleNamespace

import torch

from voicehub.models.echo.model import EchoDiT
from voicehub.models.echo.sampling import sample_euler_cfg_independent_guidances
from voicehub.models.irodoritts.native.flow_matching import sample_euler_rf_cfg
from voicehub.models.irodoritts.native.modeling import TextToLatentRFDiT
from voicehub.optimization.diffusion_sampling import (
    DiffusionSamplingConfig,
    DiffusionSamplingController,
    DiffusionSamplingMixin,
)


def _sampling_controller() -> DiffusionSamplingController:
    return DiffusionSamplingController(
        DiffusionSamplingConfig(
            target_steps=2,
            guidance="limited_interval",
            guidance_start=0.0,
            guidance_end=0.0,
        ))


class _FakeEcho:

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self) -> None:
        self.diffusion_sampling_controller = _sampling_controller()
        self.calls: list[tuple[int, str]] = []

    def reset_diffusion_cache(self) -> None:
        pass

    def get_kv_cache_text(
        self,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        del text_input_ids, text_mask
        return []

    def get_kv_cache_speaker(
        self,
        speaker_latent: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        del speaker_latent
        return []

    def __call__(
        self,
        *,
        x: torch.Tensor,
        diffusion_cache_lane: str,
        **_kwargs,
    ) -> torch.Tensor:
        self.calls.append((x.shape[0], diffusion_cache_lane))
        return torch.ones_like(x)


class _FakeIrodori:

    device = torch.device("cpu")
    dtype = torch.float32
    cfg = SimpleNamespace(
        patched_latent_dim=4,
        use_speaker_condition_resolved=False,
        use_caption_condition=False,
    )

    def __init__(self) -> None:
        self.diffusion_sampling_controller = _sampling_controller()
        self.calls: list[tuple[int, str]] = []

    def reset_diffusion_cache(self) -> None:
        pass

    def encode_conditions(
        self,
        *,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        **_kwargs,
    ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            None,
            None,
            None,
            None,
    ]:
        text_state = text_input_ids.to(dtype=self.dtype).unsqueeze(-1)
        return text_state, text_mask, None, None, None, None

    def forward_with_encoded_conditions(
        self,
        *,
        x_t: torch.Tensor,
        diffusion_cache_lane: str,
        **_kwargs,
    ) -> torch.Tensor:
        self.calls.append((x_t.shape[0], diffusion_cache_lane))
        return torch.ones_like(x_t)


def test_native_models_expose_the_shared_diffusion_sampling_protocol() -> None:
    assert issubclass(EchoDiT, DiffusionSamplingMixin)
    assert issubclass(TextToLatentRFDiT, DiffusionSamplingMixin)


def test_echo_sampler_rebuilds_schedule_and_narrows_guidance() -> None:
    model = _FakeEcho()

    result = sample_euler_cfg_independent_guidances(
        model=model,
        speaker_latent=torch.zeros(1, 2, 80),
        speaker_mask=torch.ones(1, 2, dtype=torch.bool),
        text_input_ids=torch.ones(1, 2, dtype=torch.long),
        text_mask=torch.ones(1, 2, dtype=torch.bool),
        rng_seed=7,
        num_steps=4,
        cfg_scale_text=3.0,
        cfg_scale_speaker=5.0,
        cfg_min_t=0.0,
        cfg_max_t=1.0,
        truncation_factor=None,
        rescale_k=None,
        rescale_sigma=None,
        speaker_kv_scale=None,
        speaker_kv_max_layers=None,
        speaker_kv_min_t=None,
        sequence_length=3,
    )

    assert result.shape == (1, 3, 80)
    assert model.calls == [(3, "packed-cfg"), (1, "conditional")]
    stats = model.diffusion_sampling_controller.stats()
    assert stats["native_steps"] == 4
    assert stats["prepared_steps"] == 2
    assert stats["model_calls"] == 2
    assert stats["guidance_calls"] == 1
    assert stats["guidance_skips"] == 1


def test_irodori_sampler_rebuilds_schedule_and_narrows_guidance() -> None:
    model = _FakeIrodori()

    result = sample_euler_rf_cfg(
        model=model,
        text_input_ids=torch.ones(1, 2, dtype=torch.long),
        text_mask=torch.ones(1, 2, dtype=torch.bool),
        ref_latent=None,
        ref_mask=None,
        sequence_length=3,
        num_steps=4,
        cfg_scale_text=3.0,
        cfg_scale_caption=0.0,
        cfg_scale_speaker=0.0,
        cfg_guidance_mode="independent",
        cfg_min_t=0.0,
        cfg_max_t=1.0,
        use_context_kv_cache=False,
    )

    assert result.shape == (1, 3, 4)
    assert model.calls == [(2, "packed-cfg"), (1, "conditional")]
    stats = model.diffusion_sampling_controller.stats()
    assert stats["native_steps"] == 4
    assert stats["prepared_steps"] == 2
    assert stats["model_calls"] == 2
    assert stats["guidance_calls"] == 1
    assert stats["guidance_skips"] == 1
