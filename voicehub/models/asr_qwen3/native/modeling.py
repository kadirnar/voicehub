"""VoiceHub-native Qwen3-ASR audio encoder and conditional decoder.

The graph follows the official Qwen3-ASR implementation reviewed at
``7c6daf77a2421100f5fb066495372c00129d39ff``.  It uses PyTorch and
VoiceHub's shared Qwen3 decoder, generation cache, and sequence
objective; no upstream model runtime is imported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from voicehub.generation.config import GenerationConfig
from voicehub.generation.engine import (
    AutoregressiveGenerator,
    GenerationOutput,
    GenerationStepInput,
    GenerationStepOutput,
)
from voicehub.generation.stopping import StoppingCriterion
from voicehub.models.asr_qwen3.native.configuration import Qwen3ASRArchitectureConfig, Qwen3ASRAudioConfig
from voicehub.models.causal_lm.native.modeling import CausalLMModel
from voicehub.neural.cache import DynamicKVCache
from voicehub.objectives.sequence import sequence_cross_entropy


def qwen3_asr_audio_output_lengths(input_lengths: Tensor | int) -> Tensor:
    """Return audio-token counts after the three stride-two convolutions.

    Qwen3-ASR first partitions mel frames into 100-frame convolution
    chunks; each full chunk produces 13 tokens.  The tail is transformed
    with the exact three-layer convolution length equation.
    """
    lengths = torch.as_tensor(input_lengths)
    if lengths.dtype == torch.bool or lengths.is_floating_point():
        raise TypeError("Audio feature lengths must use an integer dtype.")
    if (lengths <= 0).any():
        raise ValueError("Audio feature lengths must be positive.")
    tail = lengths.remainder(100)
    tail_after_first = (tail - 1).div(2, rounding_mode="floor") + 1
    tail_after_second = ((tail_after_first - 1).div(2, rounding_mode="floor") + 1)
    tail_after_third = ((tail_after_second - 1).div(2, rounding_mode="floor") + 1)
    return tail_after_third + lengths.div(100, rounding_mode="floor") * 13


def _sinusoidal_positions(
    length: int,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    maximum_timescale: float = 10_000.0,
) -> Tensor:
    if length < 1:
        raise ValueError("Sinusoidal position length must be positive.")
    half = channels // 2
    increment = math.log(maximum_timescale) / (half - 1)
    inverse_timescales = torch.exp(-increment * torch.arange(half, device=device, dtype=torch.float32))
    positions = torch.arange(
        length,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(1)
    angles = positions * inverse_timescales.unsqueeze(0)
    return torch.cat((angles.sin(), angles.cos()), dim=1).to(dtype=dtype)


class Qwen3ASRAudioAttention(nn.Module):
    """Bidirectional multi-head attention within bounded audio windows."""

    def __init__(
        self,
        config: Qwen3ASRAudioConfig,
        *,
        device: Any = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        factory = {"device": device, "dtype": dtype}
        self.k_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            **factory,
        )
        self.v_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            **factory,
        )
        self.q_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            **factory,
        )
        self.out_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            **factory,
        )

    def forward(
        self,
        hidden_states: Tensor,
        cumulative_sequence_lengths: Tensor,
    ) -> Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.embed_dim:
            raise ValueError("Audio attention expects [time, d_model] hidden states.")
        if (not isinstance(cumulative_sequence_lengths, Tensor) or cumulative_sequence_lengths.ndim != 1 or
                cumulative_sequence_lengths.numel() < 2):
            raise ValueError("`cumulative_sequence_lengths` must contain window "
                             "boundaries.")
        boundaries = tuple(int(value) for value in cumulative_sequence_lengths.detach().cpu().tolist())
        if (boundaries[0] != 0 or boundaries[-1] != hidden_states.shape[0] or
                any(right <= left for left, right in pairwise(boundaries))):
            raise ValueError("Invalid audio attention window boundaries.")

        sequence_length = hidden_states.shape[0]
        query = self.q_proj(hidden_states).view(
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(0, 1)
        key = self.k_proj(hidden_states).view(
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(0, 1)
        value = self.v_proj(hidden_states).view(
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(0, 1)

        attended_windows: list[Tensor] = []
        for left, right in pairwise(boundaries):
            query_window = query[:, left:right]
            key_window = key[:, left:right]
            value_window = value[:, left:right]
            scores = torch.matmul(
                query_window.float(),
                key_window.float().transpose(-2, -1),
            ) * self.scaling
            probabilities = torch.softmax(scores, dim=-1).to(dtype=query.dtype)
            probabilities = functional.dropout(
                probabilities,
                p=self.attention_dropout,
                training=self.training,
            )
            attended_windows.append(torch.matmul(probabilities, value_window).transpose(0, 1))
        attended = torch.cat(attended_windows, dim=0)
        attended = attended.reshape(
            sequence_length,
            self.embed_dim,
        )
        return self.out_proj(attended)


class Qwen3ASRAudioEncoderLayer(nn.Module):
    """Pre-norm audio Transformer block with the official tensor names."""

    def __init__(
        self,
        config: Qwen3ASRAudioConfig,
        *,
        device: Any = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.self_attn = Qwen3ASRAudioAttention(
            config,
            **factory,
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            config.d_model,
            **factory,
        )
        self.fc1 = nn.Linear(
            config.d_model,
            config.encoder_ffn_dim,
            **factory,
        )
        self.fc2 = nn.Linear(
            config.encoder_ffn_dim,
            config.d_model,
            **factory,
        )
        self.final_layer_norm = nn.LayerNorm(
            config.d_model,
            **factory,
        )

    def forward(
        self,
        hidden_states: Tensor,
        cumulative_sequence_lengths: Tensor,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = residual + self.self_attn(
            self.self_attn_layer_norm(hidden_states),
            cumulative_sequence_lengths,
        )
        residual = hidden_states
        hidden_states = residual + self.fc2(functional.gelu(self.fc1(self.final_layer_norm(hidden_states))))
        if hidden_states.dtype == torch.float16:
            limit = torch.finfo(hidden_states.dtype).max - 1_000
            hidden_states = hidden_states.clamp(-limit, limit)
        return hidden_states


class Qwen3ASRAudioEncoder(nn.Module):
    """Chunked convolutional frontend, audio Transformer, and projector."""

    def __init__(
        self,
        config: Qwen3ASRAudioConfig,
        *,
        initialize: bool = True,
        device: Any = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        factory = {"device": device, "dtype": dtype}
        self.layers = nn.ModuleList(
            Qwen3ASRAudioEncoderLayer(config, **factory) for _ in range(config.encoder_layers))
        self.ln_post = nn.LayerNorm(config.d_model, **factory)
        self.conv2d1 = nn.Conv2d(
            1,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
            **factory,
        )
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
            **factory,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
            **factory,
        )
        frequency_bins = config.num_mel_bins
        for _ in range(3):
            frequency_bins = (frequency_bins + 1) // 2
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * frequency_bins,
            config.d_model,
            bias=False,
            **factory,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model, **factory)
        self.proj2 = nn.Linear(config.d_model, config.output_dim, **factory)
        self.gradient_checkpointing = False
        if initialize:
            self.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def _window_boundaries(
        self,
        output_length: int,
        *,
        device: torch.device,
    ) -> Tensor:
        convolution_chunk = self.config.n_window * 2
        full_chunk_output = int(qwen3_asr_audio_output_lengths(convolution_chunk).item())
        window_length = full_chunk_output * (self.config.n_window_infer // convolution_chunk)
        boundaries = list(range(0, output_length, window_length))
        boundaries.append(output_length)
        return torch.tensor(
            boundaries,
            device=device,
            dtype=torch.int32,
        )

    def forward(
        self,
        input_features: Tensor,
        *,
        feature_lengths: Tensor,
    ) -> Tensor:
        if (not isinstance(input_features, Tensor) or input_features.ndim != 2 or
                input_features.shape[0] != self.config.num_mel_bins):
            raise ValueError(
                "Qwen3-ASR audio features must have shape "
                f"[{self.config.num_mel_bins}, frames].")
        if (not isinstance(feature_lengths, Tensor) or feature_lengths.numel() != 1):
            raise ValueError("The audio tower processes exactly one length.")
        feature_length = int(feature_lengths.reshape(-1)[0].item())
        if not 0 < feature_length <= input_features.shape[-1]:
            raise ValueError("Audio feature length is outside the input.")
        input_features = input_features[:, :feature_length]

        convolution_chunk = self.config.n_window * 2
        chunk_lengths = []
        remaining = feature_length
        while remaining:
            length = min(convolution_chunk, remaining)
            chunk_lengths.append(length)
            remaining -= length
        chunks = input_features.transpose(0, 1).split(chunk_lengths, dim=0)
        padded = nn.utils.rnn.pad_sequence(
            chunks,
            batch_first=True,
        ).transpose(1, 2)
        lengths = torch.tensor(
            chunk_lengths,
            device=input_features.device,
            dtype=torch.long,
        )
        after_convolution = qwen3_asr_audio_output_lengths(lengths)
        maximum_output = int(after_convolution.max().item())
        valid = (
            torch.arange(maximum_output, device=input_features.device).unsqueeze(0)
            < after_convolution.unsqueeze(1))

        padded = padded.unsqueeze(1)
        convolution_outputs: list[Tensor] = []
        for batch in padded.split(self.config.conv_chunksize, dim=0):
            hidden = functional.gelu(self.conv2d1(batch))
            hidden = functional.gelu(self.conv2d2(hidden))
            hidden = functional.gelu(self.conv2d3(hidden))
            convolution_outputs.append(hidden)
        hidden = torch.cat(convolution_outputs, dim=0)
        batch_size, channels, frequencies, frames = hidden.shape
        hidden = self.conv_out(
            hidden.permute(0, 3, 1, 2).contiguous().view(batch_size, frames, channels * frequencies))
        positions = _sinusoidal_positions(
            hidden.shape[1],
            self.config.d_model,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        hidden = hidden + positions.unsqueeze(0)
        hidden = hidden[valid]

        expected_length = int(qwen3_asr_audio_output_lengths(feature_length).item())
        if hidden.shape[0] != expected_length:
            raise RuntimeError(
                "Audio convolution produced an inconsistent token count: "
                f"{hidden.shape[0]} != {expected_length}.")
        boundaries = self._window_boundaries(
            expected_length,
            device=hidden.device,
        )
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    layer,
                    hidden,
                    boundaries,
                    use_reentrant=False,
                )
            else:
                hidden = layer(hidden, boundaries)
        hidden = self.ln_post(hidden)
        hidden = functional.gelu(self.proj1(hidden))
        return self.proj2(hidden)


@dataclass(frozen=True)
class Qwen3ASROutput:
    """Conditional decoder logits, native loss, and optional KV cache."""

    logits: Tensor
    loss: Tensor | None = None
    past_key_values: DynamicKVCache | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


class Qwen3ASRThinkerForConditionalGeneration(nn.Module):
    """Audio tower plus a shared dense Qwen3 language model."""

    def __init__(
        self,
        config: Qwen3ASRArchitectureConfig,
        *,
        initialize: bool = True,
        tie_weights: bool = True,
        device: Any = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.audio_tower = Qwen3ASRAudioEncoder(
            config.audio_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.model = CausalLMModel(
            config.text_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if initialize:
            nn.init.normal_(
                self.lm_head.weight,
                mean=0.0,
                std=config.initializer_range,
            )
        if tie_weights:
            self.tie_weights()

    def tie_weights(self) -> None:
        """Share the official input/output embedding parameter."""
        self.lm_head.weight = self.model.embed_tokens.weight

    def gradient_checkpointing_enable(self) -> None:
        self.audio_tower.gradient_checkpointing_enable()
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.audio_tower.gradient_checkpointing_disable()
        self.model.gradient_checkpointing_disable()

    def get_audio_features(
        self,
        input_features: Tensor,
        *,
        feature_attention_mask: Tensor | None = None,
        audio_feature_lengths: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(input_features, Tensor) or input_features.ndim != 3:
            raise ValueError("`input_features` must have shape [audio, mel, frames].")
        if input_features.shape[1] != self.config.audio_config.num_mel_bins:
            raise ValueError("Input mel dimension does not match the checkpoint.")
        if feature_attention_mask is not None:
            if (not isinstance(feature_attention_mask, Tensor) or tuple(feature_attention_mask.shape)
                    != (input_features.shape[0], input_features.shape[-1])):
                raise ValueError("`feature_attention_mask` must have shape "
                                 "[audio, frames].")
            lengths = feature_attention_mask.long().sum(dim=-1)
        elif audio_feature_lengths is not None:
            lengths = torch.as_tensor(
                audio_feature_lengths,
                device=input_features.device,
                dtype=torch.long,
            )
            if tuple(lengths.shape) != (input_features.shape[0], ):
                raise ValueError("`audio_feature_lengths` must contain one length per "
                                 "audio.")
        else:
            lengths = torch.full(
                (input_features.shape[0], ),
                input_features.shape[-1],
                device=input_features.device,
                dtype=torch.long,
            )
        features = [
            self.audio_tower(
                value,
                feature_lengths=length.reshape(1),
            ) for value, length in zip(input_features, lengths)
        ]
        return torch.cat(features, dim=0)

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        input_features: Tensor | None = None,
        attention_mask: Tensor | None = None,
        feature_attention_mask: Tensor | None = None,
        audio_feature_lengths: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ) -> Qwen3ASROutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")
        if inputs_embeds is None:
            if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
                raise ValueError("`input_ids` must have shape [batch, sequence].")
            inputs_embeds = self.model.embed_tokens(input_ids)
        if input_features is not None:
            if input_ids is None:
                raise ValueError("Audio placeholder replacement requires `input_ids`.")
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            ).to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            placeholder = input_ids == self.config.audio_token_id
            placeholder_count = int(placeholder.sum().item())
            if placeholder_count != audio_features.shape[0]:
                raise ValueError(
                    "Qwen3-ASR audio placeholder count does not match the "
                    "audio tower output: "
                    f"{placeholder_count} != {audio_features.shape[0]}.")
            expanded = placeholder.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                expanded,
                audio_features.reshape(-1),
            )
        elif input_ids is not None and past_key_values is None:
            placeholders = int((input_ids == self.config.audio_token_id).sum().item())
            if placeholders:
                raise ValueError(
                    "The prompt contains audio placeholders but no "
                    "`input_features` were provided.")

        if labels is not None and use_cache is None:
            use_cache = False
        output = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(output.last_hidden_state).float()
        loss = None
        if labels is not None:
            if (not isinstance(labels, Tensor) or labels.ndim != 2 or
                    tuple(labels.shape) != tuple(logits.shape[:2])):
                raise ValueError("`labels` must match [batch, sequence] model inputs.")
            if labels.shape[1] < 2:
                raise ValueError("Qwen3-ASR training requires at least two tokens.")
            loss_mask = (attention_mask[:, 1:] if attention_mask is not None else None)
            loss = sequence_cross_entropy(
                logits[:, :-1],
                labels[:, 1:],
                attention_mask=loss_mask,
                ignore_index=ignore_index,
                label_smoothing=label_smoothing,
            )
        return Qwen3ASROutput(
            logits=logits,
            loss=loss,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def generate(
            self,
            input_ids: Tensor,
            *,
            input_features: Tensor,
            attention_mask: Tensor | None = None,
            feature_attention_mask: Tensor | None = None,
            audio_feature_lengths: Tensor | None = None,
            generation_config: GenerationConfig | None = None,
            stopping_criteria: Sequence[StoppingCriterion] = (),
    ) -> GenerationOutput:
        """Generate with audio encoded only on the initial cached step."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if (not isinstance(attention_mask, Tensor) or tuple(attention_mask.shape) != tuple(input_ids.shape)):
            raise ValueError("`attention_mask` must have the same shape as `input_ids`.")
        config = generation_config or GenerationConfig(
            eos_token_id=(151_643, 151_645),
            pad_token_id=151_643,
            use_cache=True,
        )
        if not isinstance(config, GenerationConfig):
            raise TypeError("`generation_config` must be a VoiceHub GenerationConfig.")
        prompt_mask = attention_mask.to(
            device=input_ids.device,
            dtype=torch.bool,
        )

        def decoder_step(step: GenerationStepInput) -> GenerationStepOutput:
            past_length = (step.cache.sequence_length() if isinstance(step.cache, DynamicKVCache) else 0)
            key_length = past_length + step.token_ids.shape[1]
            generated = key_length - prompt_mask.shape[1]
            if generated < 0:
                raise RuntimeError("Decoder cache length is shorter than the prompt.")
            step_mask = prompt_mask
            if generated:
                step_mask = torch.cat(
                    (
                        prompt_mask,
                        torch.ones(
                            prompt_mask.shape[0],
                            generated,
                            device=prompt_mask.device,
                            dtype=torch.bool,
                        ),
                    ),
                    dim=-1,
                )
            include_audio = step.step_index == 0 or not step.use_cache
            output = self(
                step.token_ids,
                input_features=input_features if include_audio else None,
                feature_attention_mask=(feature_attention_mask if include_audio else None),
                audio_feature_lengths=(audio_feature_lengths if include_audio else None),
                attention_mask=step_mask,
                past_key_values=step.cache,
                use_cache=step.use_cache,
            )
            return GenerationStepOutput(
                logits=output.logits,
                cache=output.past_key_values,
            )

        return AutoregressiveGenerator().generate(
            decoder_step,
            input_ids,
            config,
            stopping_criteria=stopping_criteria,
        )


class Qwen3ASRForConditionalGeneration(nn.Module):
    """Top-level official checkpoint namespace and trainable forward."""

    def __init__(
        self,
        config: Qwen3ASRArchitectureConfig | dict[str, Any],
        *,
        initialize: bool = True,
        tie_weights: bool = True,
        device: Any = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = Qwen3ASRArchitectureConfig.coerce(config)
        self.thinker = Qwen3ASRThinkerForConditionalGeneration(
            self.config,
            initialize=initialize,
            tie_weights=tie_weights,
            device=device,
            dtype=dtype,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def tie_weights(self) -> None:
        self.thinker.tie_weights()

    def gradient_checkpointing_enable(self) -> None:
        self.thinker.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.thinker.gradient_checkpointing_disable()

    def forward(self, *args: Any, **kwargs: Any) -> Qwen3ASROutput:
        return self.thinker(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> GenerationOutput:
        return self.thinker.generate(*args, **kwargs)


def materialize_qwen3_asr_nonpersistent_buffers(
    model: Qwen3ASRForConditionalGeneration,
    *,
    device: str | torch.device,
) -> None:
    """Recreate non-persistent RoPE buffers after meta-parameter loading."""
    from voicehub.neural.rotary import RotaryEmbedding

    target = torch.device(device)
    for module in model.modules():
        if not isinstance(module, RotaryEmbedding):
            continue
        exponents = torch.arange(
            0,
            module.dimension,
            2,
            dtype=torch.float32,
            device=target,
        ) / module.dimension
        inverse_frequency = 1.0 / torch.pow(
            module.base,
            exponents,
        )
        module.inverse_frequency = inverse_frequency


__all__ = [
    "Qwen3ASRAudioAttention",
    "Qwen3ASRAudioEncoder",
    "Qwen3ASRAudioEncoderLayer",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASROutput",
    "Qwen3ASRThinkerForConditionalGeneration",
    "materialize_qwen3_asr_nonpersistent_buffers",
    "qwen3_asr_audio_output_lengths",
]
