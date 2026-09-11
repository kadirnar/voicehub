"""Fail-closed sampler acceleration for StyleTTS 2 style diffusion."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from torch import Tensor

from voicehub.models.styletts2.source.styletts2.Modules.diffusion.sampler import (
    Diffusion,
    DiffusionSampler,
    Sampler,
    Schedule,
)
from voicehub.optimization.diffusion_sampling import (
    DiffusionGuidanceStrategy,
    DiffusionPredictionCacheMethod,
    DiffusionSamplingCompatibilityError,
    DiffusionSamplingMixin,
    DiffusionSolverStrategy,
    coerce_diffusion_sampling_config,
)


class StyleTTS2DiffusionSampler(DiffusionSamplingMixin, DiffusionSampler):
    """Adapt only the Karras schedule boundary owned by StyleTTS 2.

    The released ADPM2 solver evaluates a denoised prediction at both
    the start and stochastic midpoint of every transition. Classifier-
    free guidance is also hidden inside each denoiser call. Whole-
    prediction caching and guidance pruning are therefore rejected until
    those stage and branch boundaries are represented explicitly.
    """

    diffusion_sampling_capabilities = frozenset({"schedule"})

    def __init__(
        self,
        diffusion: Diffusion,
        *,
        sampler: Sampler,
        sigma_schedule: Schedule,
        num_steps: int | None = None,
        clamp: bool = True,
    ) -> None:
        super().__init__(
            diffusion,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            num_steps=num_steps,
            clamp=clamp,
        )
        object.__setattr__(self, "_diffusion_cache_target", getattr(diffusion, "net", None))
        self._initialize_diffusion_sampling()

    def enable_diffusion_sampling(self, config=None):
        resolved = coerce_diffusion_sampling_config(config)
        if resolved.guidance is not DiffusionGuidanceStrategy.NATIVE:
            raise DiffusionSamplingCompatibilityError(
                "StyleTTS 2 keeps CFG inside each ADPM2 denoiser call; "
                "limited and adaptive guidance are not a safe sampler "
                "boundary.")
        if (resolved.prediction_cache is not DiffusionPredictionCacheMethod.DISABLED):
            raise DiffusionSamplingCompatibilityError(
                "StyleTTS 2 ADPM2 uses distinct start and stochastic "
                "midpoint denoiser stages; whole-prediction caching is "
                "not enabled for this adapter.")
        if resolved.solver is not DiffusionSolverStrategy.NATIVE:
            raise DiffusionSamplingCompatibilityError(
                "StyleTTS 2 exposes denoised x0 predictions to ADPM2, not "
                "the direct velocity required by STORK-2.")
        return super().enable_diffusion_sampling(resolved)

    def forward(
        self,
        noise: Tensor,
        num_steps: int | None = None,
        **kwargs: Any,
    ) -> Tensor:
        controller = self.diffusion_sampling_controller
        requested_steps = self.num_steps if num_steps is None else num_steps
        if requested_steps is None:
            raise ValueError("StyleTTS 2 `num_steps` must be provided.")
        if (isinstance(requested_steps, bool) or not isinstance(requested_steps, int) or requested_steps < 2):
            raise ValueError("StyleTTS 2 `num_steps` must be an integer >= 2.")

        native_sigmas = self.sigma_schedule(
            requested_steps,
            noise.device,
        )
        if controller is None:
            prepared_sigmas = native_sigmas
            prepared_steps = requested_steps
        else:
            controller.reset()
            # The released sampler consumes only indices [0, num_steps - 1].
            # Rebuild that exact active span so an enabled controller with no
            # target-step reduction remains bit-for-bit on the native path.
            active_sigmas = native_sigmas[:requested_steps]
            prepared_sigmas = controller.prepare_schedule(active_sigmas)
            prepared_steps = int(prepared_sigmas.numel())
        denoise_call = 0
        cache_target = getattr(self, "_diffusion_cache_target", None)
        cache_enabled = getattr(cache_target, "diffusion_cache_config", None) is not None

        def denoise(*args, **call_kwargs):
            nonlocal denoise_call
            stage = "main" if denoise_call % 2 == 0 else "midpoint"
            denoise_call += 1
            cache_kwargs = ({"diffusion_cache_lane": f"adpm2:{stage}"} if cache_enabled else {})
            return self.denoise_fn(
                *args,
                **{
                    **call_kwargs,
                    **kwargs,
                    **cache_kwargs,
                },
            )

        reset_cache = getattr(cache_target, "reset_diffusion_cache", None)
        if callable(reset_cache):
            reset_cache()
        cache_session = getattr(cache_target, "diffusion_cache_session", None)
        session = cache_session() if callable(cache_session) else nullcontext()
        with session:
            result = self.sampler(
                noise,
                fn=denoise,
                sigmas=prepared_sigmas,
                num_steps=prepared_steps,
            )
        return result.clamp(-1.0, 1.0) if self.clamp else result


__all__ = ["StyleTTS2DiffusionSampler"]
