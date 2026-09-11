"""VibeVoice diffusion head and its cosine DPM-Solver++ schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.kernels.diffusion import DiffusionModulationKernelOptimizable
from voicehub.models.vibevoice.native.configuration import VibeVoiceDiffusionConfig
from voicehub.neural.normalization import RMSNorm
from voicehub.optimization.diffusion_cache import DiffusionCacheMixin
from voicehub.optimization.diffusion_sampling import DiffusionSamplingMixin


class _RMSNormNoAffine(nn.Module):

    def __init__(self, *, epsilon: float) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        working = hidden_states.float()
        normalized = working * torch.rsqrt(working.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return normalized.to(input_dtype)


class VibeVoiceTimestepEmbedder(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        *,
        frequency_embedding_size: int = 256,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_size,
                hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        timesteps: Tensor,
        dimension: int,
        *,
        max_period: int = 10_000,
    ) -> Tensor:
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(
                half,
                dtype=torch.float32,
                device=timesteps.device,
            ) / half)
        arguments = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat(
            (torch.cos(arguments), torch.sin(arguments)),
            dim=-1,
        )
        if dimension % 2:
            embedding = torch.cat(
                (embedding, torch.zeros_like(embedding[:, :1])),
                dim=-1,
            )
        return embedding.to(timesteps.dtype)

    def forward(self, timesteps: Tensor) -> Tensor:
        frequencies = self.timestep_embedding(
            timesteps,
            self.frequency_embedding_size,
        )
        return self.mlp(frequencies)


class VibeVoiceDiffusionFFN(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def _modulate(hidden_states: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return hidden_states * (1 + scale) + shift


class VibeVoiceDiffusionLayer(DiffusionModulationKernelOptimizable, nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        norm_epsilon: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.ffn = VibeVoiceDiffusionFFN(
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        self.norm = RMSNorm(
            hidden_size,
            epsilon=norm_epsilon,
            device=device,
            dtype=dtype,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                hidden_size,
                3 * hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )
        self._initialize_diffusion_kernel_backend()

    def forward(self, hidden_states: Tensor, condition: Tensor) -> Tensor:
        shift, scale, gate = self.adaLN_modulation(condition).chunk(3, dim=-1)
        normalized = self._diffusion_modulate(
            self.norm(hidden_states),
            shift,
            scale,
        )
        return hidden_states + gate * self.ffn(normalized)


class VibeVoiceDiffusionFinalLayer(DiffusionModulationKernelOptimizable, nn.Module):

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        *,
        norm_epsilon: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.norm_final = _RMSNormNoAffine(epsilon=norm_epsilon)
        self.linear = nn.Linear(
            hidden_size,
            output_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                hidden_size,
                2 * hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )
        self._initialize_diffusion_kernel_backend()

    def forward(self, hidden_states: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        hidden_states = self._diffusion_modulate(
            self.norm_final(hidden_states),
            shift,
            scale,
        )
        return self.linear(hidden_states)


class VibeVoiceDiffusionHead(
        DiffusionCacheMixin,
        DiffusionSamplingMixin,
        nn.Module,
):
    """Checkpoint-compatible latent velocity predictor."""

    def __init__(
        self,
        config: VibeVoiceDiffusionConfig | dict[str, Any],
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = VibeVoiceDiffusionConfig.from_dict(config)
        if not isinstance(config, VibeVoiceDiffusionConfig):
            raise TypeError("VibeVoice diffusion head requires its config.")
        self.config = config
        self.noisy_images_proj = nn.Linear(
            config.latent_size,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.cond_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.t_embedder = VibeVoiceTimestepEmbedder(
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        intermediate_size = int(config.hidden_size * config.head_ffn_ratio)
        self.layers = nn.ModuleList(
            VibeVoiceDiffusionLayer(
                config.hidden_size,
                intermediate_size,
                norm_epsilon=config.rms_norm_eps,
                device=device,
                dtype=dtype,
            ) for _ in range(config.head_layers))
        self.final_layer = VibeVoiceDiffusionFinalLayer(
            config.hidden_size,
            config.latent_size,
            norm_epsilon=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        if initialize:
            self.initialize_weights()
        self._initialize_diffusion_cache()
        self._initialize_diffusion_sampling()

    def initialize_weights(self) -> None:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for layer in self.layers:
            nn.init.zeros_(layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.linear.weight)

    def forward(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        condition: Tensor,
        *,
        diffusion_cache_lane: str = "packed-cfg",
    ) -> Tensor:
        if noisy_latents.ndim != 2:
            raise ValueError("Diffusion latents must have shape [tokens, latent].")
        if condition.shape != (
                noisy_latents.shape[0],
                self.config.hidden_size,
        ):
            raise ValueError("Diffusion conditions do not align with latents.")
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(noisy_latents.shape[0])
        if timesteps.shape != (noisy_latents.shape[0], ):
            raise ValueError("Diffusion timesteps must contain one value per latent.")
        hidden_states = self.noisy_images_proj(noisy_latents)
        modulation = self.cond_proj(condition) + self.t_embedder(timesteps)
        hidden_states = self._run_diffusion_blocks(
            self.layers,
            hidden_states,
            lambda layer, value: layer(value, modulation),
            cache_lane=diffusion_cache_lane,
        )
        return self.final_layer(hidden_states, modulation)


def cosine_betas(
    num_train_timesteps: int,
    *,
    maximum_beta: float = 0.999,
) -> Tensor:
    """Return the exact Glide cosine beta discretization."""
    if (isinstance(num_train_timesteps, bool) or not isinstance(num_train_timesteps, int) or
            num_train_timesteps <= 0):
        raise ValueError("Diffusion step count must be a positive integer.")

    def alpha_bar(time: float) -> float:
        return math.cos((time + 0.008) / 1.008 * math.pi / 2)**2

    values = []
    for index in range(num_train_timesteps):
        first = index / num_train_timesteps
        second = (index + 1) / num_train_timesteps
        values.append(min(1 - alpha_bar(second) / alpha_bar(first), maximum_beta))
    return torch.tensor(values, dtype=torch.float32)


@dataclass(frozen=True)
class VibeVoiceDPMStepOutput:
    prev_sample: Tensor


class VibeVoiceDPMSolver:
    """Dependency-free default DPM-Solver++(2M) used by VibeVoice.

    Only the published schedule is represented: cosine betas, velocity
    prediction, midpoint second order, linspace timesteps, and a zero final
    sigma. Unsupported solver variations are not silently approximated.
    """

    def __init__(self, config: VibeVoiceDiffusionConfig) -> None:
        if not isinstance(config, VibeVoiceDiffusionConfig):
            raise TypeError("DPM solver requires a VibeVoice diffusion config.")
        if config.prediction_type != "v_prediction":
            raise ValueError("Published VibeVoice sampling requires velocity prediction.")
        self.config = config
        self.betas = cosine_betas(config.ddpm_num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_t = self.alphas_cumprod.sqrt()
        self.sigma_t = (1.0 - self.alphas_cumprod).sqrt()
        self.training_sigmas = ((1.0 - self.alphas_cumprod) / self.alphas_cumprod).sqrt()
        self.timesteps = torch.empty(0, dtype=torch.long)
        self.sigmas = torch.empty(0, dtype=torch.float32)
        self._model_outputs: list[Tensor | None] = [None, None]
        self._step_index: int | None = None
        self._lower_order_steps = 0

    def inference_timestep_schedule(
        self,
        num_inference_steps: int | None = None,
    ) -> Tensor:
        """Build evaluation timesteps plus the virtual clean-data endpoint."""
        steps = (self.config.ddpm_num_inference_steps if num_inference_steps is None else num_inference_steps)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("Inference step count must be a positive integer.")
        if steps > self.config.ddpm_num_steps:
            raise ValueError("Inference steps cannot exceed training steps.")
        # NumPy's linspace+round is reproduced using float64 before rounding.
        values = (
            torch.linspace(
                0,
                self.config.ddpm_num_steps - 1,
                steps + 1,
                dtype=torch.float64,
            ).round().flip(0)[:-1])
        if values.numel() > 1 and not bool(torch.all(values[1:] < values[:-1]).item()):
            raise ValueError(
                "Inference steps collapse onto duplicate training timesteps; "
                "use fewer inference steps.")
        # DPM-Solver++ terminates at sigma=0, represented by a virtual
        # timestep below the training grid rather than training timestep 0.
        return torch.cat((values, values.new_tensor([-1.0])))

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        *,
        device: str | torch.device | None = None,
        timestep_schedule: Tensor | None = None,
    ) -> None:
        if timestep_schedule is None:
            schedule = self.inference_timestep_schedule(num_inference_steps)
        else:
            if not isinstance(timestep_schedule, Tensor):
                raise TypeError("DPM timestep schedule must be a torch.Tensor.")
            if timestep_schedule.ndim != 1 or timestep_schedule.numel() < 2:
                raise ValueError(
                    "DPM timestep schedule must contain at least one step "
                    "and its terminal endpoint.")
            if not bool(torch.isfinite(timestep_schedule).all().item()):
                raise ValueError("DPM timestep schedule must contain only finite values.")
            schedule = timestep_schedule.detach().to(
                device="cpu",
                dtype=torch.float64,
            )
            differences = schedule[1:] - schedule[:-1]
            if not bool(torch.all(differences < 0).item()):
                raise ValueError("DPM timestep schedule must be strictly decreasing.")
            if not math.isclose(
                    float(schedule[-1].item()),
                    -1.0,
                    rel_tol=0.0,
                    abs_tol=1e-9,
            ):
                raise ValueError(
                    "DPM timestep schedule must terminate at the virtual "
                    "clean-data timestep -1.")
            steps = schedule.numel() - 1
            if num_inference_steps is not None and (isinstance(num_inference_steps, bool) or
                                                    not isinstance(num_inference_steps, int) or
                                                    num_inference_steps != steps):
                raise ValueError("`num_inference_steps` must match the explicit DPM "
                                 "timestep schedule.")
            if steps > self.config.ddpm_num_steps:
                raise ValueError("Inference steps cannot exceed training steps.")

        values = schedule[:-1].round().to(torch.long)
        if bool(torch.any((values < 0) | (values >= self.config.ddpm_num_steps)).item()):
            raise ValueError("DPM evaluation timesteps must lie inside the training grid.")
        if values.numel() > 1 and not bool(torch.all(values[1:] < values[:-1]).item()):
            raise ValueError("Rounded DPM evaluation timesteps must be strictly decreasing.")
        sigma_schedule = self.training_sigmas[values]
        self.timesteps = values.to(device=device)
        self.sigmas = torch.cat(
            (sigma_schedule, sigma_schedule.new_zeros(1)),
            dim=0,
        ).cpu()
        self._model_outputs = [None, None]
        self._step_index = None
        self._lower_order_steps = 0

    @staticmethod
    def _alpha_sigma(sigma: Tensor) -> tuple[Tensor, Tensor]:
        alpha = 1 / torch.sqrt(sigma.square() + 1)
        return alpha, sigma * alpha

    def add_noise(
        self,
        original_samples: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        if original_samples.shape != noise.shape:
            raise ValueError("Original samples and diffusion noise must align.")
        alpha = self.alpha_t.to(
            device=original_samples.device,
            dtype=original_samples.dtype,
        )[timesteps.to(original_samples.device)]
        sigma = self.sigma_t.to(
            device=original_samples.device,
            dtype=original_samples.dtype,
        )[timesteps.to(original_samples.device)]
        while alpha.ndim < original_samples.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        return alpha * original_samples + sigma * noise

    def get_velocity(
        self,
        original_samples: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        if original_samples.shape != noise.shape:
            raise ValueError("Original samples and diffusion noise must align.")
        alpha = self.alpha_t.to(
            device=original_samples.device,
            dtype=original_samples.dtype,
        )[timesteps.to(original_samples.device)]
        sigma = self.sigma_t.to(
            device=original_samples.device,
            dtype=original_samples.dtype,
        )[timesteps.to(original_samples.device)]
        while alpha.ndim < original_samples.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        return alpha * noise - sigma * original_samples

    def _index(self, timestep: Tensor | int) -> int:
        value = int(timestep)
        matches = (self.timesteps.cpu() == value).nonzero().flatten()
        return int(matches[0]) if matches.numel() else len(self.timesteps) - 1

    def _first_order(self, model_output: Tensor, sample: Tensor) -> Tensor:
        assert self._step_index is not None
        sigma_target = self.sigmas[self._step_index + 1]
        sigma_source = self.sigmas[self._step_index]
        alpha_target, sigma_target = self._alpha_sigma(sigma_target)
        alpha_source, sigma_source = self._alpha_sigma(sigma_source)
        lambda_target = torch.log(alpha_target) - torch.log(sigma_target)
        lambda_source = torch.log(alpha_source) - torch.log(sigma_source)
        step = lambda_target - lambda_source
        return (sigma_target / sigma_source * sample - alpha_target * (torch.exp(-step) - 1.0) * model_output)

    def _second_order(self, sample: Tensor) -> Tensor:
        assert self._step_index is not None
        current = self._model_outputs[-1]
        previous = self._model_outputs[-2]
        if current is None or previous is None:
            raise RuntimeError("Second-order DPM state is incomplete.")
        sigma_target = self.sigmas[self._step_index + 1]
        sigma_source = self.sigmas[self._step_index]
        sigma_previous = self.sigmas[self._step_index - 1]
        alpha_target, sigma_target = self._alpha_sigma(sigma_target)
        alpha_source, sigma_source = self._alpha_sigma(sigma_source)
        alpha_previous, sigma_previous = self._alpha_sigma(sigma_previous)
        lambda_target = torch.log(alpha_target) - torch.log(sigma_target)
        lambda_source = torch.log(alpha_source) - torch.log(sigma_source)
        lambda_previous = torch.log(alpha_previous) - torch.log(sigma_previous)
        step = lambda_target - lambda_source
        previous_step = lambda_source - lambda_previous
        ratio = previous_step / step
        derivative = (current - previous) / ratio
        return (
            sigma_target / sigma_source * sample - alpha_target * (torch.exp(-step) - 1.0) * current -
            0.5 * alpha_target * (torch.exp(-step) - 1.0) * derivative)

    def step(
        self,
        model_output: Tensor,
        timestep: Tensor | int,
        sample: Tensor,
    ) -> VibeVoiceDPMStepOutput:
        if self.timesteps.numel() == 0:
            raise RuntimeError("Call `set_timesteps` before DPM sampling.")
        if self._step_index is None:
            self._step_index = self._index(timestep)
        self._model_outputs[0] = self._model_outputs[1]
        sigma = self.sigmas[self._step_index].to(
            device=sample.device,
            dtype=sample.dtype,
        )
        alpha, sigma_time = self._alpha_sigma(sigma)
        # DPM-Solver++ integrates the predicted clean sample.
        clean_prediction = alpha * sample - sigma_time * model_output
        self._model_outputs[1] = clean_prediction
        final = self._step_index == len(self.timesteps) - 1
        use_first_order = self._lower_order_steps < 1 or final
        float_sample = sample.float()
        if use_first_order:
            previous = self._first_order(
                clean_prediction.float(),
                float_sample,
            )
        else:
            previous = self._second_order(float_sample)
        self._lower_order_steps = min(2, self._lower_order_steps + 1)
        self._step_index += 1
        return VibeVoiceDPMStepOutput(previous.to(model_output.dtype))


__all__ = [
    "VibeVoiceDPMSolver",
    "VibeVoiceDPMStepOutput",
    "VibeVoiceDiffusionHead",
    "cosine_betas",
]
