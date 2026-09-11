"""Causal DiT conditional-flow matcher for VoiceHub-native CosyVoice 3.

This is a PyTorch-only implementation of the author graph.  Module names
and tensor shapes intentionally follow the published ``flow.pt``
inventory so an explicit legacy conversion can create a strict native
Safetensors artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.kernels.diffusion import DiffusionModulationKernelOptimizable
from voicehub.models.cosyvoice.native.configuration import CosyVoiceFlowConfig
from voicehub.optimization.diffusion_cache import DiffusionCacheMixin
from voicehub.optimization.diffusion_sampling import DiffusionSamplingMixin, DiffusionStepContext
from voicehub.optimization.protocols import OptimizationCompileTarget


def sequence_mask(lengths: Tensor, maximum: int) -> Tensor:
    """Return ``[batch, maximum]`` validity without device synchronization."""
    if not isinstance(lengths, Tensor) or lengths.ndim != 1:
        raise ValueError("Lengths must have shape [batch].")
    if (lengths < 0).any() or (lengths > maximum).any():
        raise ValueError("A sequence length is outside the padded extent.")
    return torch.arange(maximum, device=lengths.device)[None] < lengths[:, None]


class PreLookaheadLayer(nn.Module):
    """Two-convolution residual lookahead used before mel upsampling."""

    def __init__(self, in_channels: int, channels: int, lookahead: int) -> None:
        super().__init__()
        self.pre_lookahead_len = lookahead
        self.conv1 = nn.Conv1d(in_channels, channels, lookahead + 1)
        self.conv2 = nn.Conv1d(channels, in_channels, 3)

    def forward(self, inputs: Tensor, context: Tensor | None = None) -> Tensor:
        values = inputs.transpose(1, 2)
        if context is None:
            values = functional.pad(values, (0, self.pre_lookahead_len))
        else:
            if self.training:
                raise ValueError("Explicit lookahead context is inference-only.")
            if context.shape[1] != self.pre_lookahead_len:
                raise ValueError("Lookahead context length is invalid.")
            values = torch.cat((values, context.transpose(1, 2)), dim=-1)
        values = functional.leaky_relu(self.conv1(values), negative_slope=0.01)
        values = functional.pad(values, (2, 0))
        return self.conv2(values).transpose(1, 2) + inputs


class SinusPositionEmbedding(nn.Module):

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timestep: Tensor, scale: float = 1_000.0) -> Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            torch.arange(
                half,
                device=timestep.device,
                dtype=torch.float32,
            ) * (-math.log(10_000.0) / (half - 1)))
        phase = scale * timestep.float()[:, None] * frequencies[None]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class TimestepEmbedding(nn.Module):

    def __init__(self, dimension: int, frequency_dimension: int = 256) -> None:
        super().__init__()
        self.time_embed = SinusPositionEmbedding(frequency_dimension)
        self.time_mlp = nn.Sequential(
            nn.Linear(frequency_dimension, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, timestep: Tensor) -> Tensor:
        return self.time_mlp(self.time_embed(timestep).to(timestep.dtype))


class CausalConvPositionEmbedding(nn.Module):

    def __init__(self, dimension: int, kernel_size: int = 31, groups: int = 16) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Position convolution kernel must be odd.")
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                dimension,
                dimension,
                kernel_size,
                groups=groups,
            ),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                dimension,
                dimension,
                kernel_size,
                groups=groups,
            ),
            nn.Mish(),
        )

    def forward(self, values: Tensor) -> Tensor:
        values = values.transpose(1, 2)
        values = self.conv1(functional.pad(values, (self.kernel_size - 1, 0)))
        values = self.conv2(functional.pad(values, (self.kernel_size - 1, 0)))
        return values.transpose(1, 2)


class InputEmbedding(nn.Module):

    def __init__(self, mel_dim: int, mu_dim: int, out_dim: int, spk_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + mu_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(out_dim)

    def forward(
        self,
        values: Tensor,
        conditioning: Tensor,
        means: Tensor,
        speakers: Tensor,
    ) -> Tensor:
        speakers = speakers[:, None, :].expand(-1, values.shape[1], -1)
        result = self.proj(torch.cat((values, conditioning, means, speakers), dim=-1))
        return result + self.conv_pos_embed(result)


class RotaryEmbedding(nn.Module):
    """State-compatible rotary frequencies without x-transformers."""

    def __init__(self, dimension: int, theta: float = 10_000.0) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / (theta**(torch.arange(0, dimension, 2).float() / dimension)),
        )

    def forward_from_seq_len(
        self,
        length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        positions = torch.arange(length, device=device, dtype=self.inv_freq.dtype)
        frequencies = torch.outer(positions, self.inv_freq.to(device))
        return frequencies.cos().to(dtype), frequencies.sin().to(dtype)


def _apply_rotary(values: Tensor, rotary: tuple[Tensor, Tensor]) -> Tensor:
    cosine, sine = rotary
    even = values[..., ::2]
    odd = values[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = odd * cosine + even * sine
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class AdaLayerNormZero(DiffusionModulationKernelOptimizable, nn.Module):

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dimension, dimension * 6)
        self.norm = nn.LayerNorm(
            dimension,
            elementwise_affine=False,
            eps=1e-6,
        )
        self._initialize_diffusion_kernel_backend()

    def forward(self, values: Tensor, embedding: Tensor):
        parameters = self.linear(self.silu(embedding)).chunk(6, dim=-1)
        shift_attention, scale_attention, gate_attention, shift_ff, scale_ff, gate_ff = parameters
        normalized = self._diffusion_modulate(
            self.norm(values),
            shift_attention[:, None],
            scale_attention[:, None],
        )
        return normalized, gate_attention, shift_ff, scale_ff, gate_ff


class AdaLayerNormZeroFinal(DiffusionModulationKernelOptimizable, nn.Module):

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dimension, dimension * 2)
        self.norm = nn.LayerNorm(
            dimension,
            elementwise_affine=False,
            eps=1e-6,
        )
        self._initialize_diffusion_kernel_backend()

    def forward(self, values: Tensor, embedding: Tensor) -> Tensor:
        scale, shift = self.linear(self.silu(embedding)).chunk(2, dim=-1)
        return self._diffusion_modulate(
            self.norm(values),
            shift[:, None],
            scale[:, None],
        )


class Attention(nn.Module):

    def __init__(
        self,
        dimension: int,
        heads: int,
        head_dimension: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner = heads * head_dimension
        self.heads = heads
        self.head_dimension = head_dimension
        self.to_q = nn.Linear(dimension, inner)
        self.to_k = nn.Linear(dimension, inner)
        self.to_v = nn.Linear(dimension, inner)
        self.to_out = nn.ModuleList((
            nn.Linear(inner, dimension),
            nn.Dropout(dropout),
        ))

    def forward(
        self,
        values: Tensor,
        *,
        mask: Tensor | None,
        rotary: tuple[Tensor, Tensor] | None,
    ) -> Tensor:
        batch_size, sequence_length, _ = values.shape

        def project(layer: nn.Linear) -> Tensor:
            result = layer(values).view(
                batch_size,
                sequence_length,
                self.heads,
                self.head_dimension,
            ).transpose(1, 2)
            return result

        query, key, value = project(self.to_q), project(self.to_k), project(self.to_v)
        if rotary is not None:
            query = _apply_rotary(query, rotary)
            key = _apply_rotary(key, rotary)
        attention_mask = mask
        if attention_mask is not None and attention_mask.ndim == 3:
            attention_mask = attention_mask[:, None]
        output = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
        )
        output = output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            -1,
        )
        return self.to_out[1](self.to_out[0](output))


class FeedForward(nn.Module):

    def __init__(self, dimension: int, multiplier: int) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Sequential(
                nn.Linear(dimension, dimension * multiplier),
                nn.GELU(approximate="tanh"),
            ),
            nn.Dropout(0.0),
            nn.Linear(dimension * multiplier, dimension),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.ff(values)


class DiTBlock(nn.Module):

    def __init__(
        self,
        dimension: int,
        heads: int,
        head_dimension: int,
        feed_forward_multiplier: int,
    ) -> None:
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dimension)
        self.attn = Attention(dimension, heads, head_dimension)
        self.ff_norm = nn.LayerNorm(
            dimension,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.ff = FeedForward(dimension, feed_forward_multiplier)

    def forward(
        self,
        values: Tensor,
        timestep: Tensor,
        *,
        mask: Tensor | None,
        rotary: tuple[Tensor, Tensor],
    ) -> Tensor:
        normalized, attention_gate, shift, scale, ff_gate = self.attn_norm(
            values,
            timestep,
        )
        values = values + attention_gate[:, None] * self.attn(
            normalized,
            mask=mask,
            rotary=rotary,
        )
        normalized = self.attn_norm._diffusion_modulate(
            self.ff_norm(values),
            shift[:, None],
            scale[:, None],
        )
        return values + ff_gate[:, None] * self.ff(normalized)


def _chunk_attention_mask(
    valid: Tensor,
    *,
    streaming: bool,
    chunk_size: int,
    left_chunks: int,
) -> Tensor:
    batch_size, sequence_length = valid.shape
    if not streaming:
        return valid[:, None, :].expand(batch_size, sequence_length, sequence_length)
    positions = torch.arange(sequence_length, device=valid.device)
    query_chunk = positions[:, None] // chunk_size
    key_chunk = positions[None, :] // chunk_size
    allowed = key_chunk <= query_chunk
    if left_chunks >= 0:
        allowed &= key_chunk >= query_chunk - left_chunks
    return allowed[None].expand(batch_size, -1, -1) & valid[:, None, :]


class DiTEstimator(DiffusionCacheMixin, DiffusionSamplingMixin, nn.Module):

    diffusion_sampling_capabilities = frozenset({
        "schedule",
        "guidance",
        "prediction-cache",
        "stork2",
    })

    def __init__(self, config: CosyVoiceFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.time_embed = TimestepEmbedding(config.model_dim)
        self.input_embed = InputEmbedding(
            config.mel_channels,
            config.mel_channels,
            config.model_dim,
            config.mel_channels,
        )
        self.rotary_embed = RotaryEmbedding(config.head_dim)
        self.transformer_blocks = nn.ModuleList(
            DiTBlock(
                config.model_dim,
                config.heads,
                config.head_dim,
                config.feed_forward_multiplier,
            ) for _ in range(config.depth))
        self.norm_out = AdaLayerNormZeroFinal(config.model_dim)
        self.proj_out = nn.Linear(config.model_dim, config.mel_channels)
        self._initialize_diffusion_cache()
        self._initialize_diffusion_sampling()

    def forward(
        self,
        values: Tensor,
        mask: Tensor,
        means: Tensor,
        timestep: Tensor,
        speakers: Tensor,
        conditioning: Tensor,
        *,
        streaming: bool = False,
        diffusion_cache_lane: str = "default",
    ) -> Tensor:
        values = values.transpose(1, 2)
        means = means.transpose(1, 2)
        conditioning = conditioning.transpose(1, 2)
        time_embedding = self.time_embed(timestep)
        values = self.input_embed(
            values,
            conditioning,
            means,
            speakers,
        )
        valid = mask[:, 0].to(dtype=torch.bool)
        attention_mask = _chunk_attention_mask(
            valid,
            streaming=streaming,
            chunk_size=self.config.static_chunk_size,
            left_chunks=self.config.num_decoding_left_chunks,
        )
        rotary = self.rotary_embed.forward_from_seq_len(
            values.shape[1],
            device=values.device,
            dtype=values.dtype,
        )
        values = self._run_diffusion_blocks(
            self.transformer_blocks,
            values,
            lambda block, hidden_states: block(
                hidden_states,
                time_embedding,
                mask=attention_mask,
                rotary=rotary,
            ),
            cache_lane=diffusion_cache_lane,
            valid_mask=valid,
        )
        values = self.norm_out(values, time_embedding)
        return self.proj_out(values).transpose(1, 2)


class CausalConditionalFlowMatcher(nn.Module):
    """Rectified conditional-flow objective and Euler sampler."""

    def __init__(self, config: CosyVoiceFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.estimator = DiTEstimator(config)

    def compute_loss(
        self,
        target: Tensor,
        mask: Tensor,
        means: Tensor,
        speakers: Tensor,
        conditioning: Tensor,
        *,
        streaming: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = target.shape[0]
        timestep = torch.rand(
            (batch_size, 1, 1),
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        )
        noise = torch.randn(
            target.shape,
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        )
        path = ((1 - (1 - self.config.sigma_min) * timestep) * noise + timestep * target)
        velocity = target - (1 - self.config.sigma_min) * noise
        if self.training and self.config.training_cfg_rate:
            retain = torch.rand(
                batch_size,
                device=target.device,
                generator=generator,
            ) > self.config.training_cfg_rate
            means = means * retain[:, None, None]
            speakers = speakers * retain[:, None]
            conditioning = conditioning * retain[:, None, None]
        prediction = self.estimator(
            path,
            mask,
            means,
            timestep.flatten(),
            speakers,
            conditioning,
            streaming=streaming,
        )
        denominator = (mask.sum() * target.shape[1]).clamp_min(1)
        loss = ((prediction - velocity).square() * mask).sum() / denominator
        return loss, path

    @torch.inference_mode()
    def sample(
        self,
        means: Tensor,
        mask: Tensor,
        speakers: Tensor,
        conditioning: Tensor,
        *,
        steps: int = 10,
        temperature: float = 1.0,
        streaming: bool = False,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if steps <= 0 or temperature <= 0:
            raise ValueError("Flow steps and temperature must be positive.")
        self.estimator.reset_diffusion_cache()
        self.estimator.reset_diffusion_sampling()
        values = torch.randn(
            means.shape,
            device=means.device,
            dtype=means.dtype,
            generator=generator,
        ) * temperature
        times = 1 - torch.cos(
            torch.linspace(
                0,
                1,
                steps + 1,
                device=means.device,
                dtype=means.dtype,
            ) * (math.pi / 2))
        controller = self.estimator.diffusion_sampling_controller
        if controller is not None:
            times = controller.prepare_schedule(times)
        total_steps = times.numel() - 1
        for index, (start, end) in enumerate(zip(times[:-1], times[1:])):
            timestep = start.expand(means.shape[0])
            guidance_context = DiffusionStepContext(
                index=index,
                total_steps=total_steps,
                timestep=start,
                next_timestep=end,
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
                timestep=start,
                next_timestep=end,
                lane="guided" if use_guidance else "conditional",
                solver="euler",
            )

            def evaluate_velocity() -> Tensor:
                conditioned = self.estimator(
                    values,
                    mask,
                    means,
                    timestep,
                    speakers,
                    conditioning,
                    streaming=streaming,
                    diffusion_cache_lane="conditional",
                )
                if not use_guidance:
                    return conditioned
                unconditioned = self.estimator(
                    values,
                    mask,
                    torch.zeros_like(means),
                    timestep,
                    torch.zeros_like(speakers),
                    torch.zeros_like(conditioning),
                    streaming=streaming,
                    diffusion_cache_lane="unconditional",
                )
                if controller is not None:
                    controller.observe_guidance(
                        guidance_context,
                        conditioned,
                        unconditioned,
                    )
                return ((1 + self.config.inference_cfg_rate) * conditioned -
                        self.config.inference_cfg_rate * unconditioned)

            velocity = (
                evaluate_velocity() if controller is None else controller.evaluate(
                    evaluation_context,
                    values,
                    evaluate_velocity,
                ))
            values = (
                values + (end - start) * velocity if controller is None else controller.advance(
                    evaluation_context,
                    values,
                    velocity,
                ))
        return values.float()


@dataclass(frozen=True)
class CosyVoiceFlowOutput:
    loss: Tensor
    path: Tensor
    means: Tensor
    mask: Tensor


class CosyVoiceFlowMatchingModel(nn.Module):
    """Speech-token conditioning, causal upsampling, and CFM objective."""

    def __init__(
        self,
        config: CosyVoiceFlowConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not isinstance(config, CosyVoiceFlowConfig):
            raise TypeError("`config` must be CosyVoiceFlowConfig.")
        self.config = config
        self.input_embedding = nn.Embedding(
            config.speech_vocab_size,
            config.mel_channels,
            device=device,
            dtype=dtype,
        )
        self.pre_lookahead_layer = PreLookaheadLayer(
            config.mel_channels,
            config.lookahead_hidden_size,
            config.lookahead_frames,
        )
        self.spk_embed_affine_layer = nn.Linear(
            config.speaker_embedding_dim,
            config.mel_channels,
            device=device,
            dtype=dtype,
        )
        self.decoder = CausalConditionalFlowMatcher(config)

    def codec_optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose the callable that is actually used for each flow mode."""
        if mode == "inference":
            return (
                OptimizationCompileTarget(
                    "codec.flow.cosyvoice.estimator.forward",
                    self.decoder.estimator,
                    "forward",
                    component="flow",
                ), )
        if mode != "training":
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (
            OptimizationCompileTarget(
                "codec.flow.cosyvoice.forward",
                self,
                "forward",
                component="flow",
            ), )

    def encode_tokens(
        self,
        speech_tokens: Tensor,
        speech_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(speech_tokens, Tensor) or speech_tokens.ndim != 2:
            raise ValueError("`speech_tokens` must have shape [batch, sequence].")
        if (speech_tokens < 0).any() or (speech_tokens >= self.config.speech_vocab_size).any():
            raise ValueError("Flow speech token is outside the vocabulary.")
        valid = sequence_mask(speech_lengths, speech_tokens.shape[1])
        embeddings = self.input_embedding(speech_tokens) * valid[..., None]
        means = self.pre_lookahead_layer(embeddings)
        means = means.repeat_interleave(self.config.token_mel_ratio, dim=1)
        mask = valid.repeat_interleave(self.config.token_mel_ratio, dim=1)
        return means.transpose(1, 2), mask

    def forward(
        self,
        *,
        speech_tokens: Tensor,
        speech_lengths: Tensor,
        speech_features: Tensor,
        feature_lengths: Tensor,
        speaker_embeddings: Tensor,
        conditioning: Tensor | None = None,
        streaming: bool = False,
        generator: torch.Generator | None = None,
    ) -> CosyVoiceFlowOutput:
        means, token_mask = self.encode_tokens(speech_tokens, speech_lengths)
        if not isinstance(speech_features, Tensor) or speech_features.ndim != 3:
            raise ValueError("`speech_features` must have shape [batch, frames, mel].")
        if speech_features.shape[-1] != self.config.mel_channels:
            raise ValueError("Speech feature channel count is invalid.")
        valid = sequence_mask(feature_lengths, speech_features.shape[1])
        target = speech_features.transpose(1, 2)
        if means.shape[-1] != target.shape[-1]:
            raise ValueError(
                "CosyVoice 3 requires mel frames to equal "
                "`speech token frames * token_mel_ratio`.")
        valid &= token_mask
        mask = valid[:, None].to(target.dtype)
        speakers = functional.normalize(speaker_embeddings, dim=-1)
        speakers = self.spk_embed_affine_layer(speakers)
        if conditioning is None:
            conditioning = torch.zeros_like(target)
        elif conditioning.shape != target.shape:
            raise ValueError("`conditioning` must match transposed speech features.")
        loss, path = self.decoder.compute_loss(
            target,
            mask,
            means,
            speakers,
            conditioning,
            streaming=streaming,
            generator=generator,
        )
        return CosyVoiceFlowOutput(
            loss=loss,
            path=path,
            means=means,
            mask=mask,
        )

    @torch.inference_mode()
    def generate(
        self,
        speech_tokens: Tensor,
        speech_lengths: Tensor,
        speaker_embeddings: Tensor,
        *,
        prompt_features: Tensor | None = None,
        steps: int = 10,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        means, valid = self.encode_tokens(speech_tokens, speech_lengths)
        mask = valid[:, None].to(means.dtype)
        speakers = self.spk_embed_affine_layer(functional.normalize(speaker_embeddings, dim=-1))
        conditioning = torch.zeros_like(means)
        if prompt_features is not None:
            if prompt_features.ndim != 3 or (prompt_features.shape[0] != means.shape[0] or
                                             prompt_features.shape[2] != means.shape[1]):
                raise ValueError("`prompt_features` must have shape [batch, frames, mel].")
            prompt = prompt_features.transpose(1, 2)
            if prompt.shape[-1] > means.shape[-1]:
                raise ValueError("Prompt features exceed the generated mel extent.")
            conditioning[..., :prompt.shape[-1]] = prompt
        return self.decoder.sample(
            means,
            mask,
            speakers,
            conditioning,
            steps=steps,
            temperature=temperature,
            generator=generator,
        )


__all__ = [
    "CosyVoiceFlowMatchingModel",
    "CosyVoiceFlowOutput",
    "CausalConditionalFlowMatcher",
    "DiTEstimator",
    "PreLookaheadLayer",
    "sequence_mask",
]
