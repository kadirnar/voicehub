"""VoiceHub-owned PyTorch implementation of the VoxCPM2 TTS graph."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.voxcpm.native.configuration import VoxCPM2ArchitectureConfig, VoxCPMCFMConfig
from voicehub.models.voxcpm.native.minicpm import MiniCPMModel, local_transformer_config
from voicehub.optimization.diffusion_cache import DiffusionCacheMixin
from voicehub.optimization.diffusion_sampling import DiffusionSamplingMixin, DiffusionStepContext
from voicehub.optimization.protocols import OptimizationCompileTarget


class VoxCPMScalarQuantizer(nn.Module):
    """Finite scalar quantizer with the source straight-through estimator."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        latent_dimension: int,
        scale: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.in_proj = nn.Linear(
            input_dimension,
            latent_dimension,
            device=device,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(
            latent_dimension,
            output_dimension,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = torch.tanh(self.in_proj(hidden_states))
        quantized = torch.round(hidden_states * self.scale) / self.scale
        if self.training:
            hidden_states = hidden_states + (quantized - hidden_states).detach()
        else:
            hidden_states = quantized
        return self.out_proj(hidden_states)


class VoxCPMLocalEncoder(nn.Module):
    """Encode one AudioVAE patch into one LM embedding."""

    def __init__(
        self,
        config,
        *,
        input_dimension: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.special_token = nn.Parameter(
            torch.randn(
                1,
                1,
                1,
                config.hidden_size,
                device=device,
                dtype=dtype,
            ))
        self.in_proj = nn.Linear(
            input_dimension,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.encoder = MiniCPMModel(config, device=device, dtype=dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("VoxCPM local features must have shape [batch, time, patch, dim].")
        batch, time, _, _ = inputs.shape
        embedded = self.in_proj(inputs)
        special = self.special_token.expand(batch, time, 1, -1)
        embedded = torch.cat((special, embedded), dim=2).flatten(0, 1)
        output, _ = self.encoder(embedded, is_causal=False)
        return output[:, 0].unflatten(0, (batch, time))


class _SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("VoxCPM time-embedding dimensions must be even.")
        self.dimension = dimension

    def forward(self, values: Tensor, scale: float = 1_000.0) -> Tensor:
        if values.ndim < 1:
            values = values.unsqueeze(0)
        half = self.dimension // 2
        frequency = math.log(10_000.0) / (half - 1)
        frequency = torch.exp(torch.arange(
            half,
            dtype=values.dtype,
            device=values.device,
        ) * -frequency)
        embedding = scale * values.unsqueeze(1) * frequency.unsqueeze(0)
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class _TimeMLP(nn.Module):

    def __init__(
        self,
        dimension: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(
            dimension,
            dimension,
            device=device,
            dtype=dtype,
        )
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(
            dimension,
            dimension,
            device=device,
            dtype=dtype,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(inputs)))


class VoxCPMLocalDiT(DiffusionCacheMixin, DiffusionSamplingMixin, nn.Module):
    """Source-compatible local diffusion transformer."""

    def __init__(
        self,
        config,
        *,
        input_channels: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_channels = input_channels
        self.out_channels = input_channels
        self.in_proj = nn.Linear(
            input_channels,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.cond_proj = nn.Linear(
            input_channels,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(
            config.hidden_size,
            input_channels,
            device=device,
            dtype=dtype,
        )
        self.time_embeddings = _SinusoidalTimeEmbedding(config.hidden_size)
        self.time_mlp = _TimeMLP(
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.delta_time_mlp = _TimeMLP(
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.decoder = MiniCPMModel(config, device=device, dtype=dtype)
        self._initialize_diffusion_cache()
        self._initialize_diffusion_sampling()

    def forward(
        self,
        inputs: Tensor,
        conditioning_embedding: Tensor,
        time: Tensor,
        prefix_condition: Tensor,
        delta_time: Tensor,
        *,
        diffusion_cache_lane: str = "packed-cfg",
    ) -> Tensor:
        inputs = self.in_proj(inputs.transpose(1, 2).contiguous())
        condition = self.cond_proj(prefix_condition.transpose(1, 2).contiguous())
        prefix_length = condition.shape[1]
        time_embedding = self.time_mlp(self.time_embeddings(time).to(inputs.dtype))
        delta_embedding = self.delta_time_mlp(self.time_embeddings(delta_time).to(inputs.dtype))
        time_embedding = time_embedding + delta_embedding
        conditioning_embedding = conditioning_embedding.view(
            inputs.shape[0],
            -1,
            inputs.shape[-1],
        )
        sequence = torch.cat(
            (
                conditioning_embedding,
                time_embedding.unsqueeze(1),
                condition,
                inputs,
            ),
            dim=1,
        )
        position_embedding = None
        if self.decoder.rope_emb is not None:
            positions = torch.arange(
                sequence.shape[1],
                dtype=torch.long,
                device=sequence.device,
            )
            position_embedding = self.decoder.rope_emb(positions)
        hidden = self._run_diffusion_blocks(
            self.decoder.layers,
            sequence,
            lambda layer, value: layer(
                value,
                position_embedding,
                is_causal=False,
            )[0],
            cache_lane=diffusion_cache_lane,
        )
        hidden = self.decoder.norm(hidden)
        hidden = hidden[
            :,
            prefix_length + conditioning_embedding.shape[1] + 1:,
        ]
        return self.out_proj(hidden).transpose(1, 2).contiguous()


class VoxCPMConditionalFlowMatcher(nn.Module):
    """Published conditional flow-matching objective and Euler sampler."""

    def __init__(
        self,
        input_channels: int,
        config: VoxCPMCFMConfig,
        estimator: VoxCPMLocalDiT,
        *,
        mean_mode: bool,
    ) -> None:
        super().__init__()
        self.solver = config.solver
        self.sigma_min = config.sigma_min
        self.t_scheduler = config.t_scheduler
        self.training_cfg_rate = config.training_cfg_rate
        self.inference_cfg_rate = config.inference_cfg_rate
        self.reg_loss_type = config.reg_loss_type
        self.ratio_r_neq_t_range = config.ratio_r_neq_t_range
        self.noise_cond_prob_range = config.noise_cond_prob_range
        self.noise_cond_scale = config.noise_cond_scale
        self.in_channels = input_channels
        self.mean_mode = mean_mode
        self.estimator = estimator

    def sample(
        self,
        conditioning_embedding: Tensor,
        *,
        patch_size: int,
        prefix_condition: Tensor,
        steps: int,
        temperature: float = 1.0,
        guidance: float = 1.0,
        sway: float = 1.0,
        use_cfg_zero_star: bool = True,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if steps <= 0:
            raise ValueError("VoxCPM diffusion steps must be positive.")
        # The outer autoregressive frame changes prefix conditioning.  Every
        # local solve starts a new cache history.
        self.estimator.reset_diffusion_cache()
        self.estimator.reset_diffusion_sampling()
        batch = conditioning_embedding.shape[0]
        state = torch.randn(
            (batch, self.in_channels, patch_size),
            device=conditioning_embedding.device,
            dtype=conditioning_embedding.dtype,
            generator=generator,
        ) * float(temperature)
        time_span = torch.linspace(
            1,
            0,
            steps + 1,
            device=state.device,
            dtype=state.dtype,
        )
        time_span = time_span + sway * (torch.cos(torch.pi / 2 * time_span) - 1 + time_span)
        controller = self.estimator.diffusion_sampling_controller
        if controller is not None:
            time_span = controller.prepare_schedule(time_span)
        time = time_span[0]
        delta = time_span[0] - time_span[1]
        zero_steps = max(1, int(len(time_span) * 0.04))
        for step in range(1, len(time_span)):
            index = step - 1
            total_steps = len(time_span) - 1
            if use_cfg_zero_star and step <= zero_steps:
                velocity = torch.zeros_like(state)
            else:
                guidance_context = DiffusionStepContext(
                    index=index,
                    total_steps=total_steps,
                    timestep=time,
                    next_timestep=time_span[step],
                    lane="guidance",
                    solver="euler",
                )
                use_guidance = (
                    True if controller is None else controller.should_use_guidance(
                        guidance_context,
                        native=True,
                    ))
                evaluation_context = DiffusionStepContext(
                    index=index,
                    total_steps=total_steps,
                    timestep=time,
                    next_timestep=time_span[step],
                    lane="packed-cfg" if use_guidance else "conditional",
                    solver="euler",
                )

                def evaluate_velocity() -> Tensor:
                    if not use_guidance:
                        model_time = time.expand(batch)
                        model_delta = (
                            torch.zeros_like(model_time) if not self.mean_mode else delta.expand(batch))
                        return self.estimator(
                            state,
                            conditioning_embedding,
                            model_time,
                            prefix_condition,
                            model_delta,
                            diffusion_cache_lane="conditional",
                        )
                    model_state = torch.cat((state, state), dim=0)
                    model_embedding = torch.cat(
                        (
                            conditioning_embedding,
                            torch.zeros_like(conditioning_embedding),
                        ),
                        dim=0,
                    )
                    model_time = time.expand(2 * batch)
                    model_delta = (
                        torch.zeros_like(model_time) if not self.mean_mode else delta.expand(2 * batch))
                    model_prefix = torch.cat(
                        (prefix_condition, prefix_condition),
                        dim=0,
                    )
                    positive, negative = self.estimator(
                        model_state,
                        model_embedding,
                        model_time,
                        model_prefix,
                        model_delta,
                        diffusion_cache_lane="packed-cfg",
                    ).chunk(2)
                    if controller is not None:
                        controller.observe_guidance(
                            guidance_context,
                            positive,
                            negative,
                        )
                    if use_cfg_zero_star:
                        positive_flat = positive.flatten(1)
                        negative_flat = negative.flatten(1)
                        scale = ((positive_flat * negative_flat).sum(1, keepdim=True) /
                                 (negative_flat.pow(2).sum(1, keepdim=True) + 1e-8)).view(batch, 1, 1)
                    else:
                        scale = 1.0
                    return (negative * scale + guidance * (positive - negative * scale))

                velocity = (
                    evaluate_velocity() if controller is None else controller.evaluate(
                        evaluation_context,
                        state,
                        evaluate_velocity,
                    ))
            state = state - delta * velocity
            time = time - delta
            if step < len(time_span) - 1:
                delta = time - time_span[step + 1]
        return state

    def _sample_r_t(
        self,
        inputs: Tensor,
        *,
        ratio_r_neq_t: float,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        batch = inputs.shape[0]
        if self.t_scheduler == "log-norm":
            first = (
                torch.randn(
                    batch,
                    device=inputs.device,
                    dtype=inputs.dtype,
                    generator=generator,
                ) - 0.4)
            second = (
                torch.randn(
                    batch,
                    device=inputs.device,
                    dtype=inputs.dtype,
                    generator=generator,
                ) - 0.4)
            first, second = torch.sigmoid(first), torch.sigmoid(second)
        else:
            first = torch.rand(
                batch,
                device=inputs.device,
                dtype=inputs.dtype,
                generator=generator,
            )
            second = torch.rand(
                batch,
                device=inputs.device,
                dtype=inputs.dtype,
                generator=generator,
            )
        unequal = (
            torch.rand(
                batch,
                device=inputs.device,
                dtype=inputs.dtype,
                generator=generator,
            ) < ratio_r_neq_t)
        lower = torch.minimum(first, second)
        upper = torch.maximum(first, second)
        return (
            torch.where(unequal, lower, second),
            torch.where(unequal, upper, second),
        )

    def compute_loss(
        self,
        target: Tensor,
        conditioning_embedding: Tensor,
        *,
        prefix_condition: Tensor | None = None,
        target_mask: Tensor | None = None,
        progress: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        batch = target.shape[0]
        if self.training_cfg_rate > 0:
            keep = (torch.rand(
                batch,
                device=target.device,
                generator=generator,
            ) > self.training_cfg_rate)
            conditioning_embedding = (conditioning_embedding * keep.view(-1, 1))
        if prefix_condition is None:
            prefix_condition = torch.zeros_like(target)
        noise_probability = (
            self.noise_cond_prob_range[0] + float(progress) *
            (self.noise_cond_prob_range[1] - self.noise_cond_prob_range[0]))
        noisy = (torch.rand(
            batch,
            device=target.device,
            generator=generator,
        ) < noise_probability)
        prefix_condition = prefix_condition + (
            noisy.view(-1, 1, 1) * torch.randn(
                prefix_condition.shape,
                device=prefix_condition.device,
                dtype=prefix_condition.dtype,
                generator=generator,
            ) * self.noise_cond_scale)
        ratio = (
            self.ratio_r_neq_t_range[0] + float(progress) *
            (self.ratio_r_neq_t_range[1] - self.ratio_r_neq_t_range[0]) if self.mean_mode else 0.0)
        first, second = self._sample_r_t(
            target,
            ratio_r_neq_t=ratio,
            generator=generator,
        )
        first_detached = first.detach().clone()
        second_detached = second.detach().clone()
        noise = torch.randn(
            target.shape,
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        )
        interpolated = ((1 - second_detached.view(-1, 1, 1)) * target +
                        second_detached.view(-1, 1, 1) * noise)
        target_velocity = noise - target

        def model_fn(
            sample: Tensor,
            first_time: Tensor,
            second_time: Tensor,
        ) -> Tensor:
            return self.estimator(
                sample,
                conditioning_embedding,
                second_time,
                prefix_condition,
                second_time - first_time,
            )

        if self.mean_mode:
            predicted, derivative = torch.func.jvp(
                model_fn,
                (interpolated, first, second),
                (
                    target_velocity,
                    torch.zeros_like(first),
                    torch.ones_like(second),
                ),
            )
            supervised = target_velocity - (second_detached - first_detached).view(-1, 1, 1) * derivative
        else:
            predicted = model_fn(interpolated, first, second)
            supervised = target_velocity
        losses = functional.mse_loss(
            predicted,
            supervised.detach(),
            reduction="none",
        ).mean(dim=1)
        weights = (losses + 1e-3).pow(0.0).reciprocal().detach()
        if target_mask is not None:
            mask = target_mask.squeeze(1).to(losses.dtype)
            weights = weights * mask
            return (weights * losses).sum() / target_mask.sum().clamp_min(1.0)
        return (weights * losses).mean()


@dataclass(frozen=True)
class VoxCPM2Output:
    loss: Tensor
    diffusion_loss: Tensor
    stop_loss: Tensor
    target_features: Tensor
    generated_features: Tensor | None = None


class VoxCPM2Model(nn.Module):
    """Exact 577-tensor VoxCPM2 language/flow graph.

    AudioVAE is intentionally a separate module: the official model
    Safetensors file does not contain codec tensors and upstream also freezes
    that codec during fine-tuning.
    """

    def __init__(
        self,
        config: VoxCPM2ArchitectureConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not isinstance(config, VoxCPM2ArchitectureConfig):
            raise TypeError("`config` must be a VoxCPM2ArchitectureConfig.")
        self.config = config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
        self.base_lm = MiniCPMModel(
            config.lm_config,
            device=device,
            dtype=dtype,
        )
        residual_config = local_transformer_config(
            config.lm_config,
            hidden_size=config.lm_config.hidden_size,
            intermediate_size=config.lm_config.intermediate_size,
            num_attention_heads=config.lm_config.num_attention_heads,
            num_hidden_layers=config.residual_lm_num_layers,
            kv_channels=config.lm_config.kv_channels,
            no_rope=config.residual_lm_no_rope,
        )
        self.residual_lm = MiniCPMModel(
            residual_config,
            device=device,
            dtype=dtype,
        )
        encoder_config = local_transformer_config(
            config.lm_config,
            hidden_size=config.encoder_config.hidden_dim,
            intermediate_size=config.encoder_config.ffn_dim,
            num_attention_heads=config.encoder_config.num_heads,
            num_hidden_layers=config.encoder_config.num_layers,
            kv_channels=config.encoder_config.kv_channels,
        )
        self.feat_encoder = VoxCPMLocalEncoder(
            encoder_config,
            input_dimension=config.feat_dim,
            device=device,
            dtype=dtype,
        )
        decoder_config = local_transformer_config(
            config.lm_config,
            hidden_size=config.dit_config.hidden_dim,
            intermediate_size=config.dit_config.ffn_dim,
            num_attention_heads=config.dit_config.num_heads,
            num_hidden_layers=config.dit_config.num_layers,
            kv_channels=config.dit_config.kv_channels,
        )
        estimator = VoxCPMLocalDiT(
            decoder_config,
            input_channels=config.feat_dim,
            device=device,
            dtype=dtype,
        )
        self.feat_decoder = VoxCPMConditionalFlowMatcher(
            config.feat_dim,
            config.dit_config.cfm_config,
            estimator,
            mean_mode=config.dit_config.dit_mean_mode,
        )
        hidden = config.lm_config.hidden_size
        dit_hidden = config.dit_config.hidden_dim
        self.fsq_layer = VoxCPMScalarQuantizer(
            hidden,
            hidden,
            config.scalar_quantization_latent_dim,
            config.scalar_quantization_scale,
            device=device,
            dtype=dtype,
        )
        self.enc_to_lm_proj = nn.Linear(
            config.encoder_config.hidden_dim,
            hidden,
            device=device,
            dtype=dtype,
        )
        self.lm_to_dit_proj = nn.Linear(
            hidden,
            dit_hidden,
            device=device,
            dtype=dtype,
        )
        self.res_to_dit_proj = nn.Linear(
            hidden,
            dit_hidden,
            device=device,
            dtype=dtype,
        )
        self.fusion_concat_proj = nn.Linear(
            hidden * 2,
            hidden,
            device=device,
            dtype=dtype,
        )
        self.stop_proj = nn.Linear(
            hidden,
            hidden,
            device=device,
            dtype=dtype,
        )
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(
            hidden,
            2,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.stop_loss = nn.CrossEntropyLoss(reduction="none")

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Compile the exact public graph used by inference or training."""
        if mode == "inference":
            attribute = "generate_features"
        elif mode == "training":
            attribute = "forward"
        else:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (OptimizationCompileTarget(
            attribute,
            self,
            attribute,
        ), )

    @property
    def parameter_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def materialize_runtime_buffers(self, device) -> None:
        self.base_lm.materialize_runtime_buffers(device)
        self.residual_lm.materialize_runtime_buffers(device)
        self.feat_encoder.encoder.materialize_runtime_buffers(device)
        self.feat_decoder.estimator.decoder.materialize_runtime_buffers(device)

    def setup_generation_cache(
        self,
        *,
        batch_size: int,
        device,
        dtype,
    ) -> None:
        for model in (self.base_lm, self.residual_lm):
            model.setup_cache(
                batch_size=batch_size,
                max_length=self.config.max_length,
                device=device,
                dtype=dtype,
            )

    def forward(
        self,
        text_tokens: Tensor,
        text_mask: Tensor,
        audio_feats: Tensor,
        audio_mask: Tensor,
        loss_mask: Tensor,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        *,
        progress: float = 0.0,
        sample_generate: bool = False,
        generator: torch.Generator | None = None,
    ) -> VoxCPM2Output:
        del position_ids
        if audio_feats.ndim != 4:
            raise ValueError("`audio_feats` must have shape [batch, time, patch, dim].")
        batch, time, patch, dimension = audio_feats.shape
        if patch != self.patch_size or dimension != self.feat_dim:
            raise ValueError("VoxCPM audio feature shape does not match its configuration.")
        expected = (batch, time)
        for name, value in (
            ("text_tokens", text_tokens),
            ("text_mask", text_mask),
            ("audio_mask", audio_mask),
            ("loss_mask", loss_mask),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"`{name}` must have shape {expected}.")
        if labels is None:
            labels = torch.zeros_like(text_tokens)
            final = loss_mask.bool().long().sum(dim=1).sub(1).clamp_min(0)
            labels.scatter_(1, final.unsqueeze(1), 1)
        if tuple(labels.shape) != expected:
            raise ValueError(f"`labels` must have shape {expected}.")
        target_dtype = self.parameter_dtype
        text_tokens = text_tokens.to(dtype=torch.long)
        text_mask = text_mask.to(dtype=target_dtype)
        audio_feats = audio_feats.to(dtype=target_dtype)
        audio_mask = audio_mask.to(dtype=target_dtype)
        loss_mask = loss_mask.to(dtype=target_dtype)
        labels = labels.to(dtype=torch.long)

        feature_embedding = self.enc_to_lm_proj(self.feat_encoder(audio_feats))
        scale = (self.config.lm_config.scale_emb if self.config.lm_config.use_mup else 1.0)
        text_embedding = self.base_lm.embed_tokens(text_tokens) * scale
        combined = (text_mask.unsqueeze(-1) * text_embedding + audio_mask.unsqueeze(-1) * feature_embedding)
        encoded, _ = self.base_lm(combined, is_causal=True)
        encoded = (self.fsq_layer(encoded) * audio_mask.unsqueeze(-1) + encoded * text_mask.unsqueeze(-1))
        lm_hidden = torch.cat(
            (torch.zeros_like(encoded[:, :1]), encoded[:, :-1]),
            dim=1,
        )
        residual_input = self.fusion_concat_proj(
            torch.cat(
                (encoded, audio_mask.unsqueeze(-1) * feature_embedding),
                dim=-1,
            ))
        residual, _ = self.residual_lm(
            residual_input,
            is_causal=True,
        )
        residual_hidden = torch.cat(
            (torch.zeros_like(residual[:, :1]), residual[:, :-1]),
            dim=1,
        )
        dit_hidden = torch.cat(
            (
                self.lm_to_dit_proj(lm_hidden),
                self.res_to_dit_proj(residual_hidden),
            ),
            dim=-1,
        ).flatten(0, 1)
        feature_target = audio_feats.flatten(0, 1)
        feature_condition = torch.cat(
            (torch.zeros_like(audio_feats[:, :1]), audio_feats[:, :-1]),
            dim=1,
        ).flatten(0, 1)
        sequence_mask = (loss_mask.unsqueeze(-1).expand(-1, -1, self.patch_size).flatten(0, 1).unsqueeze(1))
        diffusion_loss = self.feat_decoder.compute_loss(
            feature_target.transpose(1, 2).contiguous(),
            dit_hidden,
            prefix_condition=feature_condition.transpose(1, 2).contiguous(),
            target_mask=sequence_mask,
            progress=progress,
            generator=generator,
        )
        stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))
        stop_losses = self.stop_loss(
            stop_logits.transpose(1, 2),
            labels,
        )
        stop_loss = ((stop_losses * loss_mask).sum() / loss_mask.sum().clamp_min(1.0))
        generated = None
        if sample_generate:
            generated_sequence = self.feat_decoder.sample(
                dit_hidden,
                patch_size=self.patch_size,
                prefix_condition=feature_condition.transpose(1, 2).contiguous(),
                steps=10,
                generator=generator,
            )
            generated = (
                generated_sequence.transpose(1, 2).unflatten(0, (batch, time)).permute(0, 3, 1, 2).flatten(2))
        target_features = (feature_target.unflatten(0, (batch, time)).permute(0, 3, 1, 2).flatten(2))
        return VoxCPM2Output(
            loss=diffusion_loss + stop_loss,
            diffusion_loss=diffusion_loss,
            stop_loss=stop_loss,
            target_features=target_features,
            generated_features=generated,
        )

    @torch.inference_mode()
    def generate_features(
        self,
        text_tokens: Tensor,
        text_mask: Tensor,
        audio_feats: Tensor,
        audio_mask: Tensor,
        *,
        min_length: int = 2,
        max_length: int = 2_000,
        diffusion_steps: int = 10,
        guidance: float = 2.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate AudioVAE latent patches from a prepared source prefix."""
        if text_tokens.ndim != 2 or text_tokens.shape[0] != 1:
            raise ValueError("Native VoxCPM2 generation currently supports batch size one.")
        if tuple(text_mask.shape) != tuple(text_tokens.shape):
            raise ValueError("VoxCPM text mask shape must match text tokens.")
        if audio_feats.ndim != 4 or audio_feats.shape[:2] != text_tokens.shape:
            raise ValueError("VoxCPM audio prefix must align with text tokens.")
        if tuple(audio_mask.shape) != tuple(text_tokens.shape):
            raise ValueError("VoxCPM audio mask shape must match text tokens.")
        if min_length < 0 or max_length <= min_length:
            raise ValueError("VoxCPM generation length bounds are invalid.")
        if text_tokens.shape[1] + max_length > self.config.max_length:
            raise ValueError(
                "VoxCPM generation prefix and `max_length` exceed the "
                f"{self.config.max_length}-position context window.")
        device = next(self.parameters()).device
        dtype = self.parameter_dtype
        text_tokens = text_tokens.to(device=device, dtype=torch.long)
        text_mask = text_mask.to(device=device, dtype=dtype)
        audio_feats = audio_feats.to(device=device, dtype=dtype)
        audio_mask = audio_mask.to(device=device, dtype=dtype)
        feature_embedding = self.enc_to_lm_proj(self.feat_encoder(audio_feats))
        scale = (self.config.lm_config.scale_emb if self.config.lm_config.use_mup else 1.0)
        text_embedding = self.base_lm.embed_tokens(text_tokens) * scale
        combined = (text_mask.unsqueeze(-1) * text_embedding + audio_mask.unsqueeze(-1) * feature_embedding)
        self.setup_generation_cache(
            batch_size=1,
            device=device,
            dtype=dtype,
        )
        encoded, cache = self.base_lm(combined, is_causal=True)
        self.base_lm.kv_cache.fill(cache)
        encoded = (self.fsq_layer(encoded) * audio_mask.unsqueeze(-1) + encoded * text_mask.unsqueeze(-1))
        language_hidden = encoded[:, -1]
        residual_input = self.fusion_concat_proj(
            torch.cat(
                (encoded, audio_mask.unsqueeze(-1) * feature_embedding),
                dim=-1,
            ))
        residual, cache = self.residual_lm(
            residual_input,
            is_causal=True,
        )
        self.residual_lm.kv_cache.fill(cache)
        residual_hidden = residual[:, -1]
        previous = audio_feats[:, -1]
        generated = []
        position = text_tokens.shape[1]
        for step in range(max_length):
            condition = torch.cat(
                (
                    self.lm_to_dit_proj(language_hidden),
                    self.res_to_dit_proj(residual_hidden),
                ),
                dim=-1,
            )
            prediction = self.feat_decoder.sample(
                condition,
                patch_size=self.patch_size,
                prefix_condition=previous.transpose(1, 2).contiguous(),
                steps=diffusion_steps,
                guidance=guidance,
                generator=generator,
            ).transpose(1, 2)
            generated.append(prediction)
            current_embedding = self.enc_to_lm_proj(self.feat_encoder(prediction.unsqueeze(1)))[:, 0]
            stop = self.stop_head(self.stop_actn(self.stop_proj(language_hidden))).argmax(dim=-1)
            if step > min_length and int(stop.item()) == 1:
                break
            language_hidden = self.fsq_layer(self.base_lm.forward_step(
                current_embedding,
                position,
            ))
            residual_hidden = self.residual_lm.forward_step(
                self.fusion_concat_proj(torch.cat((language_hidden, current_embedding), dim=-1)),
                position,
            )
            previous = prediction
            position += 1
        if not generated:
            raise RuntimeError("VoxCPM2 generated no AudioVAE features.")
        return torch.cat(generated, dim=1).transpose(1, 2).flatten(2)


__all__ = [
    "VoxCPM2Model",
    "VoxCPM2Output",
    "VoxCPMConditionalFlowMatcher",
    "VoxCPMLocalDiT",
    "VoxCPMLocalEncoder",
    "VoxCPMScalarQuantizer",
]
