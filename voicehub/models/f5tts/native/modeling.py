"""Native F5-TTS DiT and conditional-flow objective."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from voicehub.models.f5tts.native.audio import F5MelSpectrogram
from voicehub.models.f5tts.native.configuration import F5TTSArchitectureConfig
from voicehub.models.f5tts.native.modules import (
    AdaLayerNormFinal,
    DiTBlock,
    InputEmbedding,
    RotaryEmbedding,
    TextEmbedding,
    TimestepEmbedding,
)
from voicehub.optimization.diffusion_cache import DiffusionCacheMixin
from voicehub.optimization.diffusion_sampling import DiffusionSamplingMixin, DiffusionStepContext


def lengths_to_mask(
    lengths: torch.Tensor,
    *,
    length: int | None = None,
) -> torch.Tensor:
    maximum = int(lengths.max().item()) if length is None else int(length)
    return (torch.arange(maximum, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1))


def random_span_mask(
    lengths: torch.Tensor,
    fractions: torch.Tensor,
    *,
    sequence_length: int,
) -> torch.Tensor:
    span_lengths = (fractions * lengths).long()
    maximum_start = lengths - span_lengths
    starts = (maximum_start * torch.rand_like(fractions)).long().clamp_min(0)
    ends = starts + span_lengths
    positions = torch.arange(sequence_length, device=lengths.device)
    return ((positions.unsqueeze(0) >= starts.unsqueeze(1)) & (positions.unsqueeze(0) < ends.unsqueeze(1)))


def epss_timesteps(
    steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    schedules = {
        5: (0, 2, 4, 8, 16, 32),
        6: (0, 2, 4, 6, 8, 16, 32),
        7: (0, 2, 4, 6, 8, 16, 24, 32),
        10: (0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32),
        12: (0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32),
        16: (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32),
    }
    schedule = schedules.get(steps)
    if schedule is None:
        return torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    return torch.tensor(schedule, device=device, dtype=dtype) / 32


class F5DiT(DiffusionCacheMixin, DiffusionSamplingMixin, nn.Module):
    """State-compatible F5-TTS diffusion transformer."""

    diffusion_sampling_capabilities = frozenset({
        "schedule",
        "guidance",
        "prediction-cache",
        "stork2",
    })

    def __init__(
        self,
        config: F5TTSArchitectureConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.time_embed = TimestepEmbedding(config.dim)
        self.text_embed = TextEmbedding(
            config.text_num_embeds,
            config.text_dim,
            mask_padding=config.text_mask_padding,
            average_upsampling=config.text_embedding_average_upsampling,
            conv_layers=config.conv_layers,
            conv_mult=config.conv_mult,
        )
        self.input_embed = InputEmbedding(
            config.mel_dim,
            config.text_dim,
            config.dim,
        )
        self.rotary_embed = RotaryEmbedding(config.dim_head)
        self.dim = config.dim
        self.depth = config.depth
        self.transformer_blocks = nn.ModuleList(
            DiTBlock(
                config.dim,
                heads=config.heads,
                dim_head=config.dim_head,
                ff_mult=config.ff_mult,
                dropout=config.dropout,
                qk_norm=config.qk_norm,
                pe_attn_head=config.pe_attn_head,
                attn_mask_enabled=config.attn_mask_enabled,
            ) for _ in range(config.depth))
        self.long_skip_connection = (
            nn.Linear(config.dim * 2, config.dim, bias=False) if config.long_skip_connection else None)
        self.norm_out = AdaLayerNormFinal(config.dim)
        self.proj_out = nn.Linear(config.dim, config.mel_dim)
        self.checkpoint_activations = config.checkpoint_activations
        self._text_cond: torch.Tensor | None = None
        self._text_uncond: torch.Tensor | None = None
        self._initialize_diffusion_cache()
        self._initialize_diffusion_sampling()
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def clear_cache(self) -> None:
        self._text_cond = None
        self._text_uncond = None

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable or disable non-reentrant transformer-block checkpointing."""
        if not isinstance(enabled, bool):
            raise TypeError("`enabled` must be a boolean.")
        self.checkpoint_activations = enabled

    def _input_embedding(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        drop_audio_cond: bool,
        drop_text: bool,
        cache: bool,
        audio_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cached = self._text_uncond if drop_text else self._text_cond
        if cached is None or not cache:
            sequence_length: int | torch.Tensor = (
                hidden_states.shape[1] if audio_mask is None else audio_mask.sum(dim=1))
            cached = self.text_embed(
                token_ids,
                sequence_length,
                drop_text=drop_text,
            )
            if cache:
                if drop_text:
                    self._text_uncond = cached
                else:
                    self._text_cond = cached
        return self.input_embed(
            hidden_states,
            conditioning,
            cached,
            drop_audio_cond=drop_audio_cond,
            audio_mask=audio_mask,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        token_ids: torch.Tensor,
        time: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
    ) -> torch.Tensor:
        batch, sequence_length = hidden_states.shape[:2]
        if time.ndim == 0:
            time = time.repeat(batch)
        time_embedding = self.time_embed(time)
        if cfg_infer:
            conditional = self._input_embedding(
                hidden_states,
                conditioning,
                token_ids,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
                audio_mask=mask,
            )
            unconditional = self._input_embedding(
                hidden_states,
                conditioning,
                token_ids,
                drop_audio_cond=True,
                drop_text=True,
                cache=cache,
                audio_mask=mask,
            )
            hidden_states = torch.cat((conditional, unconditional), dim=0)
            time_embedding = torch.cat((time_embedding, time_embedding), dim=0)
            if mask is not None:
                mask = torch.cat((mask, mask), dim=0)
        else:
            hidden_states = self._input_embedding(
                hidden_states,
                conditioning,
                token_ids,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                audio_mask=mask,
            )
        rope = self.rotary_embed.forward_from_seq_len(sequence_length)
        residual = hidden_states

        def apply_block(block: nn.Module, value: torch.Tensor) -> torch.Tensor:
            if self.checkpoint_activations and self.training:
                return activation_checkpoint(
                    block,
                    value,
                    time_embedding,
                    mask,
                    rope,
                    use_reentrant=False,
                )
            return block(
                value,
                time_embedding,
                mask=mask,
                rope=rope,
            )

        cache_lane = (
            "packed-cfg" if cfg_infer else
            ("unconditional" if drop_text or drop_audio_cond else "conditional"))
        hidden_states = self._run_diffusion_blocks(
            self.transformer_blocks,
            hidden_states,
            apply_block,
            cache_lane=cache_lane,
            valid_mask=mask,
        )
        if self.long_skip_connection is not None:
            hidden_states = self.long_skip_connection(torch.cat((hidden_states, residual), dim=-1))
        return self.proj_out(self.norm_out(hidden_states, time_embedding))


class F5ConditionalFlowMatcher(nn.Module):
    """F5-TTS conditional-flow model with native ODE integration."""

    def __init__(
        self,
        config: F5TTSArchitectureConfig,
        *,
        transformer: F5DiT | None = None,
        mel_spec: nn.Module | None = None,
        ode_method: str = "euler",
    ) -> None:
        super().__init__()
        if ode_method not in {"euler", "midpoint"}:
            raise ValueError("Native F5-TTS supports 'euler' and 'midpoint'.")
        self.config = config
        self.frac_lengths_mask = (
            config.mask_fraction_min,
            config.mask_fraction_max,
        )
        self.mel_spec = mel_spec or F5MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.mel_dim,
        )
        self.num_channels = config.mel_dim
        self.audio_drop_prob = config.audio_drop_prob
        self.cond_drop_prob = config.cond_drop_prob
        self.transformer = transformer or F5DiT(config)
        self.dim = self.transformer.dim
        self.sigma = config.sigma
        self.ode_method = ode_method

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def gradient_checkpointing(self) -> bool:
        return self.transformer.checkpoint_activations

    def gradient_checkpointing_enable(self) -> None:
        """Delegate activation checkpointing to the native DiT."""
        self.transformer.set_gradient_checkpointing(True)

    def gradient_checkpointing_disable(self) -> None:
        """Disable activation checkpointing on the native DiT."""
        self.transformer.set_gradient_checkpointing(False)

    def _velocity(
        self,
        time: torch.Tensor,
        state: torch.Tensor,
        *,
        conditioning: torch.Tensor,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None,
        cfg_strength: float,
        guidance_context: DiffusionStepContext | None = None,
    ) -> torch.Tensor:
        if cfg_strength < 1e-5:
            return self.transformer(
                state,
                conditioning,
                token_ids,
                time,
                mask=mask,
                cache=True,
            )
        packed = self.transformer(
            state,
            conditioning,
            token_ids,
            time,
            mask=mask,
            cfg_infer=True,
            cache=True,
        )
        conditional, unconditional = packed.chunk(2, dim=0)
        controller = self.transformer.diffusion_sampling_controller
        if controller is not None and guidance_context is not None:
            controller.observe_guidance(
                guidance_context,
                conditional,
                unconditional,
            )
        return conditional + (conditional - unconditional) * cfg_strength

    @torch.no_grad()
    def sample(
        self,
        conditioning: torch.Tensor,
        token_ids: torch.Tensor,
        duration: int | torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
        steps: int = 32,
        cfg_strength: float = 1.0,
        sway_sampling_coef: float | None = None,
        seed: int | None = None,
        max_duration: int = 65_536,
        vocoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        use_epss: bool = True,
        no_reference_audio: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if steps <= 0:
            raise ValueError("`steps` must be positive.")
        self.transformer.reset_diffusion_cache()
        self.transformer.reset_diffusion_sampling()
        if conditioning.ndim == 2:
            conditioning = self.mel_spec(conditioning).transpose(1, 2)
        if conditioning.ndim != 3 or conditioning.shape[-1] != self.num_channels:
            raise ValueError("F5-TTS conditioning must be waveform or `[batch, frames, mel]`.")
        conditioning = conditioning.to(
            device=self.device,
            dtype=next(self.parameters()).dtype,
        )
        token_ids = token_ids.to(device=self.device, dtype=torch.long)
        batch, prompt_length = conditioning.shape[:2]
        if lengths is None:
            lengths = torch.full(
                (batch, ),
                prompt_length,
                device=self.device,
                dtype=torch.long,
            )
        else:
            lengths = lengths.to(device=self.device, dtype=torch.long)
        if isinstance(duration, int):
            durations = torch.full(
                (batch, ),
                duration,
                device=self.device,
                dtype=torch.long,
            )
        else:
            durations = duration.to(device=self.device, dtype=torch.long)
        text_lengths = (token_ids != -1).sum(dim=-1)
        durations = torch.maximum(
            torch.maximum(text_lengths, lengths) + 1,
            durations,
        ).clamp(max=max_duration)
        maximum_duration = int(durations.max().item())
        prompt_mask = lengths_to_mask(lengths, length=maximum_duration)
        conditioning = F.pad(
            conditioning,
            (0, 0, 0, maximum_duration - prompt_length),
        )
        if no_reference_audio:
            conditioning = torch.zeros_like(conditioning)
        step_conditioning = torch.where(
            prompt_mask.unsqueeze(-1),
            conditioning,
            torch.zeros_like(conditioning),
        )
        attention_mask = (lengths_to_mask(durations, length=maximum_duration) if batch > 1 else None)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        initial = [
            torch.randn(
                int(item.item()),
                self.num_channels,
                device=self.device,
                dtype=conditioning.dtype,
                generator=generator,
            ) for item in durations
        ]
        state = pad_sequence(initial, batch_first=True, padding_value=0.0)
        times = (
            epss_timesteps(
                steps,
                device=self.device,
                dtype=conditioning.dtype,
            ) if use_epss else torch.linspace(
                0,
                1,
                steps + 1,
                device=self.device,
                dtype=conditioning.dtype,
            ))
        if sway_sampling_coef is not None:
            times = times + sway_sampling_coef * (torch.cos(torch.pi / 2 * times) - 1 + times)
        controller = self.transformer.diffusion_sampling_controller
        if controller is not None:
            times = controller.prepare_schedule(times)
            if (controller.config.solver.value == "stork2" and self.ode_method != "euler"):
                raise ValueError("STORK-2 requires F5-TTS `ode_method='euler'`.")
        trajectory = [state]
        try:
            for index in range(times.numel() - 1):
                start = times[index]
                step_size = times[index + 1] - start
                total_steps = times.numel() - 1
                guidance_context = DiffusionStepContext(
                    index=index,
                    total_steps=total_steps,
                    timestep=start,
                    next_timestep=times[index + 1],
                    lane="guidance",
                    solver=self.ode_method,
                    stage="main",
                )
                native_guidance = cfg_strength >= 1e-5
                use_guidance = (
                    native_guidance if controller is None else controller.should_use_guidance(
                        guidance_context,
                        native=native_guidance,
                    ))
                effective_cfg_strength = cfg_strength if use_guidance else 0.0
                evaluation_context = DiffusionStepContext(
                    index=index,
                    total_steps=total_steps,
                    timestep=start,
                    next_timestep=times[index + 1],
                    lane="packed-cfg" if use_guidance else "conditional",
                    solver=self.ode_method,
                    stage="main",
                )

                def evaluate_main() -> torch.Tensor:
                    return self._velocity(
                        start,
                        state,
                        conditioning=step_conditioning,
                        token_ids=token_ids,
                        mask=attention_mask,
                        cfg_strength=effective_cfg_strength,
                        guidance_context=guidance_context,
                    )

                velocity = (
                    evaluate_main() if controller is None else controller.evaluate(
                        evaluation_context,
                        state,
                        evaluate_main,
                    ))
                if self.ode_method == "midpoint":
                    midpoint = state + 0.5 * step_size * velocity
                    midpoint_time = start + 0.5 * step_size
                    midpoint_guidance_context = DiffusionStepContext(
                        index=index,
                        total_steps=total_steps,
                        timestep=midpoint_time,
                        next_timestep=times[index + 1],
                        lane="guidance",
                        solver=self.ode_method,
                        stage="midpoint",
                    )
                    midpoint_guidance = (
                        native_guidance if controller is None else controller.should_use_guidance(
                            midpoint_guidance_context,
                            native=native_guidance,
                        ))
                    midpoint_cfg_strength = (cfg_strength if midpoint_guidance else 0.0)
                    midpoint_context = DiffusionStepContext(
                        index=index,
                        total_steps=total_steps,
                        timestep=midpoint_time,
                        next_timestep=times[index + 1],
                        lane=("packed-cfg" if midpoint_guidance else "conditional"),
                        solver=self.ode_method,
                        stage="midpoint",
                    )

                    def evaluate_midpoint() -> torch.Tensor:
                        return self._velocity(
                            midpoint_time,
                            midpoint,
                            conditioning=step_conditioning,
                            token_ids=token_ids,
                            mask=attention_mask,
                            cfg_strength=midpoint_cfg_strength,
                            guidance_context=midpoint_guidance_context,
                        )

                    velocity = (
                        evaluate_midpoint() if controller is None else controller.evaluate(
                            midpoint_context,
                            midpoint,
                            evaluate_midpoint,
                        ))
                state = (
                    state + step_size * velocity
                    if controller is None or self.ode_method == "midpoint" else controller.advance(
                        evaluation_context,
                        state,
                        velocity,
                    ))
                trajectory.append(state)
        finally:
            self.transformer.clear_cache()
        sampled = torch.where(
            prompt_mask.unsqueeze(-1),
            conditioning,
            state,
        )
        if vocoder is not None:
            sampled = vocoder(sampled.transpose(1, 2))
        return sampled, torch.stack(trajectory)

    def forward(
        self,
        inp: torch.Tensor,
        text: torch.Tensor,
        *,
        lens: torch.Tensor | None = None,
        noise_scheduler: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if noise_scheduler not in {None, "linear"}:
            raise ValueError("Native F5-TTS currently implements the linear flow path.")
        if inp.ndim == 2:
            inp = self.mel_spec(inp).transpose(1, 2)
        if inp.ndim != 3 or inp.shape[-1] != self.num_channels:
            raise ValueError("`inp` must be waveform or `[batch, frames, mel]`.")
        batch, sequence_length = inp.shape[:2]
        text = text.to(device=inp.device, dtype=torch.long)
        if lens is None:
            lens = torch.full(
                (batch, ),
                sequence_length,
                device=inp.device,
                dtype=torch.long,
            )
        else:
            lens = lens.to(device=inp.device, dtype=torch.long)
        mask = lengths_to_mask(lens, length=sequence_length)
        fractions = torch.empty(
            batch,
            device=inp.device,
            dtype=torch.float32,
        ).uniform_(*self.frac_lengths_mask)
        span_mask = random_span_mask(
            lens,
            fractions,
            sequence_length=sequence_length,
        ) & mask
        target = inp
        noise = torch.randn_like(target)
        time = torch.rand(batch, dtype=inp.dtype, device=inp.device)
        expanded_time = time[:, None, None]
        path = (1 - expanded_time) * noise + expanded_time * target
        flow = target - noise
        conditioning = torch.where(
            span_mask.unsqueeze(-1),
            torch.zeros_like(target),
            target,
        )
        drop_audio = bool(torch.rand((), device=inp.device).item() < self.audio_drop_prob)
        drop_text = False
        if torch.rand((), device=inp.device).item() < self.cond_drop_prob:
            drop_audio = True
            drop_text = True
        prediction = self.transformer(
            path,
            conditioning,
            text,
            time,
            drop_audio_cond=drop_audio,
            drop_text=drop_text,
            mask=mask,
        )
        loss = F.mse_loss(prediction, flow, reduction="none")[span_mask]
        if loss.numel() == 0:
            raise RuntimeError("F5-TTS sampled an empty training span.")
        return loss.mean(), conditioning, prediction


def build_f5tts_model(
    config: F5TTSArchitectureConfig | dict[str, object] | None = None,
    *,
    ode_method: str = "euler",
) -> F5ConditionalFlowMatcher:
    resolved = (
        config
        if isinstance(config, F5TTSArchitectureConfig) else F5TTSArchitectureConfig.from_mapping(config))
    return F5ConditionalFlowMatcher(resolved, ode_method=ode_method)


__all__ = [
    "F5ConditionalFlowMatcher",
    "F5DiT",
    "build_f5tts_model",
    "epss_timesteps",
    "lengths_to_mask",
    "random_span_mask",
]
