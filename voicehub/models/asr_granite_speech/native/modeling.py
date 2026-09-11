"""PyTorch-only IBM Granite Speech architecture.

The executable graph is checkpoint-compatible with the released
``granite-speech-4.1-2b`` Safetensors inventory: a block-local Conformer
encoder, a BLIP-2-style Q-Former projector, and a Granite causal
language model.  The module hierarchy deliberately preserves the public
checkpoint names ``encoder.*``, ``projector.*``, and
``language_model.*``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
from voicehub.models.asr_granite_speech.native.configuration import (
    GraniteSpeechArchitectureConfig,
    GraniteSpeechEncoderConfig,
    GraniteSpeechProjectorConfig,
)
from voicehub.models.causal_lm.native.modeling import CausalLMOutput, GraniteForCausalLM
from voicehub.neural.cache import DynamicKVCache


@dataclass(frozen=True)
class GraniteSpeechEncoderOutput:
    """Audio encoder states and projected language-model embeddings."""

    last_hidden_state: Tensor
    projected_hidden_state: Tensor | None = None


@dataclass(frozen=True)
class GraniteSpeechOutput:
    """Native Granite Speech logits, loss, cache, and audio diagnostics."""

    logits: Tensor
    loss: Tensor | None = None
    past_key_values: DynamicKVCache | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None
    audio_hidden_states: Tensor | None = None


class GraniteSpeechConformerFeedForward(nn.Module):
    """Macaron feed-forward branch used by one Conformer block."""

    def __init__(self, config: GraniteSpeechEncoderConfig) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.up_proj = nn.Linear(
            config.hidden_dim,
            config.hidden_dim * config.feedforward_mult,
        )
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)
        self.down_proj = nn.Linear(
            config.hidden_dim * config.feedforward_mult,
            config.hidden_dim,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.pre_norm(hidden_states)
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.dropout(self.silu(hidden_states))
        hidden_states = self.down_proj(hidden_states)
        return self.dropout(hidden_states)


class GraniteSpeechConformerAttention(nn.Module):
    """Block-local attention with Shaw relative position embeddings."""

    def __init__(self, config: GraniteSpeechEncoderConfig) -> None:
        super().__init__()
        inner_dim = config.dim_head * config.num_heads
        self.max_pos_emb = config.max_pos_emb
        self.context_size = config.context_size
        self.num_heads = config.num_heads
        self.dim_head = config.dim_head
        self.scale = self.dim_head**-0.5
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.to_q = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(
            config.hidden_dim,
            inner_dim * 2,
            bias=False,
        )
        self.to_out = nn.Linear(inner_dim, config.hidden_dim)
        self.rel_pos_emb = nn.Embedding(
            2 * self.max_pos_emb + 1,
            self.dim_head,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_dists: Tensor,
    ) -> Tensor:
        normalized = self.pre_norm(hidden_states)
        batch_size, original_length, _ = normalized.shape
        block_count = math.ceil(original_length / self.context_size)
        remainder = original_length % self.context_size
        if remainder:
            normalized = functional.pad(
                normalized,
                (0, 0, 0, self.context_size - remainder),
            )

        queries = self.to_q(normalized)
        keys, values = self.to_kv(normalized).chunk(2, dim=-1)
        flat_batch = batch_size * block_count
        queries = queries.reshape(
            flat_batch,
            self.context_size,
            self.num_heads,
            self.dim_head,
        ).transpose(1, 2)
        keys = keys.reshape(
            flat_batch,
            self.context_size,
            self.num_heads,
            self.dim_head,
        ).transpose(1, 2)
        values = values.reshape(
            flat_batch,
            self.context_size,
            self.num_heads,
            self.dim_head,
        ).transpose(1, 2)

        relative = self.rel_pos_emb(attention_dists)
        positional_bias = torch.einsum(
            "b h c d, c r d -> b h c r",
            queries,
            relative,
        ) * self.scale
        if remainder:
            invalid = torch.ones(
                self.context_size,
                self.context_size,
                dtype=torch.bool,
                device=normalized.device,
            )
            invalid[:remainder, :remainder] = False
            mask_value = -torch.finfo(positional_bias.dtype).max
            positional_bias[block_count - 1:positional_bias.shape[0]:block_count].masked_fill_(
                invalid, mask_value)

        attended = functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=positional_bias,
            dropout_p=0.0,
            scale=self.scale,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            normalized.shape[1],
            -1,
        )
        output = self.to_out(attended[:, :original_length])
        return self.dropout(output)


class GraniteSpeechConformerDepthWiseConv1d(nn.Module):
    """Depthwise convolution with the reference implementation's padding."""

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        pad_offset = (kernel_size + 1) % 2
        self.padding = (pad, pad - pad_offset)
        self.conv = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size,
            groups=channels_in,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.conv(functional.pad(hidden_states, self.padding))


class GraniteSpeechConformerConvModule(nn.Module):
    """Pointwise/depthwise convolution branch of a Conformer block."""

    def __init__(self, config: GraniteSpeechEncoderConfig) -> None:
        super().__init__()
        inner_dim = config.hidden_dim * config.conv_expansion_factor
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.up_conv = nn.Conv1d(
            config.hidden_dim,
            inner_dim * 2,
            kernel_size=1,
        )
        self.glu = nn.GLU(dim=1)
        self.depth_conv = GraniteSpeechConformerDepthWiseConv1d(
            inner_dim,
            inner_dim,
            config.conv_kernel_size,
        )
        self.silu = nn.SiLU()
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.down_conv = nn.Conv1d(
            inner_dim,
            config.hidden_dim,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.up_conv(hidden_states.transpose(1, 2))
        hidden_states = self.glu(hidden_states)
        hidden_states = self.depth_conv(hidden_states)
        hidden_states = self.silu(self.batch_norm(hidden_states))
        hidden_states = self.down_conv(hidden_states).transpose(1, 2)
        return self.dropout(hidden_states)


class GraniteSpeechConformerBlock(nn.Module):
    """One checkpoint-compatible Granite Speech Conformer block."""

    def __init__(self, config: GraniteSpeechEncoderConfig) -> None:
        super().__init__()
        self.ff1 = GraniteSpeechConformerFeedForward(config)
        self.attn = GraniteSpeechConformerAttention(config)
        self.conv = GraniteSpeechConformerConvModule(config)
        self.ff2 = GraniteSpeechConformerFeedForward(config)
        self.post_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        hidden_states: Tensor,
        attention_dists: Tensor,
    ) -> Tensor:
        hidden_states = hidden_states + 0.5 * self.ff1(hidden_states)
        hidden_states = hidden_states + self.attn(
            hidden_states,
            attention_dists,
        )
        hidden_states = hidden_states + self.conv(hidden_states)
        hidden_states = hidden_states + 0.5 * self.ff2(hidden_states)
        return self.post_norm(hidden_states)


class GraniteSpeechCTCEncoder(nn.Module):
    """Conformer audio encoder retained under the official tensor names."""

    def __init__(self, config: GraniteSpeechEncoderConfig) -> None:
        super().__init__()
        self.config = config
        positions = torch.arange(config.context_size)
        distances = positions.view(-1, 1) - positions.view(1, -1)
        attention_dists = (distances.clamp(-config.context_size, config.context_size) + config.max_pos_emb)
        self.register_buffer(
            "attention_dists",
            attention_dists,
            persistent=False,
        )
        self.input_linear = nn.Linear(
            config.input_dim,
            config.hidden_dim,
            bias=True,
        )
        self.layers = nn.ModuleList(GraniteSpeechConformerBlock(config) for _ in range(config.num_layers))
        self.out = nn.Linear(
            config.hidden_dim,
            config.output_dim,
            bias=True,
        )
        self.out_mid = nn.Linear(
            config.output_dim,
            config.hidden_dim,
            bias=True,
        )
        self.num_layers = config.num_layers

    def forward(self, hidden_states: Tensor) -> Tensor:
        if (not isinstance(hidden_states, Tensor) or hidden_states.ndim != 3 or
                hidden_states.shape[-1] != self.config.input_dim):
            raise ValueError(
                "`input_features` must have shape "
                f"[batch, frames, {self.config.input_dim}].")
        hidden_states = self.input_linear(hidden_states)
        for index, layer in enumerate(self.layers, start=1):
            hidden_states = layer(
                hidden_states,
                self.attention_dists,
            )
            if index == self.num_layers // 2:
                midpoint = self.out(hidden_states.clone())
                hidden_states = hidden_states + self.out_mid(midpoint.softmax(dim=-1), )
        return hidden_states


class GraniteSpeechQFormerMultiHeadAttention(nn.Module):
    """Bidirectional self/cross attention used by the Q-Former."""

    def __init__(
        self,
        config: GraniteSpeechProjectorConfig,
        *,
        is_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = (config.hidden_size // config.num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.attention_head_size)
        self.scaling = self.attention_head_size**-0.5
        self.attention_dropout = config.attention_probs_dropout_prob
        key_value_size = (config.encoder_hidden_size if is_cross_attention else config.hidden_size)
        self.query = nn.Linear(
            config.hidden_size,
            self.all_head_size,
        )
        self.key = nn.Linear(
            key_value_size,
            self.all_head_size,
        )
        self.value = nn.Linear(
            key_value_size,
            self.all_head_size,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        current_states = (encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        active_mask = (encoder_attention_mask if encoder_hidden_states is not None else attention_mask)
        batch_size, query_length, _ = hidden_states.shape
        key_length = current_states.shape[1]
        queries = self.query(hidden_states).view(
            batch_size,
            query_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        keys = self.key(current_states).view(
            batch_size,
            key_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        values = self.value(current_states).view(
            batch_size,
            key_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        if active_mask is not None:
            if active_mask.ndim == 2:
                active_mask = active_mask[:, None, None, :].to(dtype=torch.bool, )
            elif active_mask.ndim != 4:
                raise ValueError("Q-Former attention masks must have rank two or four.")
            if active_mask.dtype == torch.bool:
                invalid = ~active_mask
                active_mask = torch.zeros(
                    (),
                    device=queries.device,
                    dtype=queries.dtype,
                ).expand_as(invalid).masked_fill(
                    invalid,
                    -torch.finfo(queries.dtype).max,
                )
            else:
                active_mask = active_mask.to(
                    device=queries.device,
                    dtype=queries.dtype,
                )
        attended = functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=active_mask,
            dropout_p=(self.attention_dropout if self.training else 0.0),
            scale=self.scaling,
        )
        return attended.transpose(1, 2).reshape(
            batch_size,
            query_length,
            self.all_head_size,
        )


class GraniteSpeechQFormerSelfOutput(nn.Module):

    def __init__(self, config: GraniteSpeechProjectorConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor,
    ) -> Tensor:
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.LayerNorm(hidden_states + residual)


class GraniteSpeechQFormerAttention(nn.Module):

    def __init__(
        self,
        config: GraniteSpeechProjectorConfig,
        *,
        is_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.attention = GraniteSpeechQFormerMultiHeadAttention(
            config,
            is_cross_attention=is_cross_attention,
        )
        self.output = GraniteSpeechQFormerSelfOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        attended = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return self.output(attended, hidden_states)


class GraniteSpeechQFormerIntermediate(nn.Module):

    def __init__(self, config: GraniteSpeechProjectorConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return functional.gelu(self.dense(hidden_states))


class GraniteSpeechQFormerOutput(nn.Module):

    def __init__(self, config: GraniteSpeechProjectorConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor,
    ) -> Tensor:
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.LayerNorm(hidden_states + residual)


class GraniteSpeechQFormerLayer(nn.Module):

    def __init__(
        self,
        config: GraniteSpeechProjectorConfig,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.attention = GraniteSpeechQFormerAttention(config)
        if layer_index % config.cross_attention_frequency == 0:
            self.crossattention = GraniteSpeechQFormerAttention(
                config,
                is_cross_attention=True,
            )
        else:
            self.crossattention = None
        self.intermediate_query = GraniteSpeechQFormerIntermediate(config)
        self.output_query = GraniteSpeechQFormerOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
        )
        if self.crossattention is not None:
            hidden_states = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
        intermediate = self.intermediate_query(hidden_states)
        return self.output_query(intermediate, hidden_states)


class GraniteSpeechQFormerEncoder(nn.Module):

    def __init__(self, config: GraniteSpeechProjectorConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            GraniteSpeechQFormerLayer(config, index) for index in range(config.num_hidden_layers))

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layer:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
        return hidden_states


class GraniteSpeechQFormer(nn.Module):
    """Query-only BLIP-2 Q-Former with official parameter names."""

    def __init__(self, config: GraniteSpeechProjectorConfig) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = GraniteSpeechQFormerEncoder(config)

    def forward(
        self,
        query_embeds: Tensor,
        *,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden_states = self.dropout(self.layernorm(query_embeds))
        return self.encoder(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )


class GraniteSpeechEncoderProjector(nn.Module):
    """Windowed learned-query projection from audio to language space."""

    def __init__(
        self,
        config: GraniteSpeechArchitectureConfig,
    ) -> None:
        super().__init__()
        projector = config.projector_config
        self.hidden_size = projector.hidden_size
        self.downsample_rate = config.downsample_rate
        self.window_size = config.window_size
        self.num_queries = config.projector_tokens_per_window
        self.query = nn.Parameter(torch.empty(
            1,
            self.num_queries,
            projector.hidden_size,
        ))
        self.qformer = GraniteSpeechQFormer(projector)
        self.linear = nn.Linear(
            projector.hidden_size,
            config.text_config.hidden_size,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if not isinstance(hidden_states, Tensor) or hidden_states.ndim != 3:
            raise ValueError("Granite Speech projector input must have shape "
                             "[batch, frames, hidden].")
        batch_size, sequence_length, hidden_size = hidden_states.shape
        block_count = math.ceil(sequence_length / self.window_size)
        padding = block_count * self.window_size - sequence_length
        if padding:
            hidden_states = functional.pad(
                hidden_states,
                (0, 0, 0, padding),
            )
        hidden_states = hidden_states.view(
            batch_size * block_count,
            self.window_size,
            hidden_size,
        )
        queries = self.query.expand(
            batch_size * block_count,
            -1,
            -1,
        )
        query_output = self.qformer(
            queries,
            encoder_hidden_states=hidden_states,
        )
        query_output = query_output.view(
            batch_size,
            block_count * self.num_queries,
            self.hidden_size,
        )
        return self.linear(query_output)


class GraniteSpeechForConditionalGeneration(nn.Module):
    """Trainable native Granite Speech model and cache-aware decoder."""

    def __init__(
        self,
        config: GraniteSpeechArchitectureConfig | dict[str, Any],
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        self.config = (
            config if isinstance(config, GraniteSpeechArchitectureConfig) else
            GraniteSpeechArchitectureConfig.from_dict(config))
        self.encoder = GraniteSpeechCTCEncoder(self.config.encoder_config)
        self.projector = GraniteSpeechEncoderProjector(self.config)
        self.language_model = GraniteForCausalLM(
            self.config.text_config,
            initialize=initialize,
        )
        self.gradient_checkpointing = False
        if initialize:
            self._initialize_audio_modules()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def tie_weights(self) -> None:
        if self.config.text_config.tie_word_embeddings:
            self.language_model.tie_weights()

    def _initialize_audio_modules(self) -> None:
        for module in (self.encoder, self.projector):
            for child in module.modules():
                if isinstance(child, (nn.Linear, nn.Conv1d)):
                    nn.init.normal_(
                        child.weight,
                        mean=0.0,
                        std=self.config.initializer_range,
                    )
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                elif isinstance(child, nn.Embedding):
                    nn.init.normal_(
                        child.weight,
                        mean=0.0,
                        std=self.config.initializer_range,
                    )
                elif isinstance(child, (nn.LayerNorm, nn.BatchNorm1d)):
                    if child.weight is not None:
                        nn.init.ones_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                    if isinstance(child, nn.BatchNorm1d):
                        child.reset_running_stats()
        nn.init.normal_(self.projector.query, mean=0.0, std=1.0)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True
        self.language_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False
        self.language_model.gradient_checkpointing_disable()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Linear:
        return self.language_model.get_output_embeddings()

    def get_audio_features(
        self,
        input_features: Tensor,
    ) -> GraniteSpeechEncoderOutput:
        if self.gradient_checkpointing and self.training:
            hidden_states = checkpoint(
                self.encoder,
                input_features,
                use_reentrant=False,
            )
            projected = checkpoint(
                self.projector,
                hidden_states,
                use_reentrant=False,
            )
        else:
            hidden_states = self.encoder(input_features)
            projected = self.projector(hidden_states)
        return GraniteSpeechEncoderOutput(
            last_hidden_state=hidden_states,
            projected_hidden_state=projected,
        )

    def get_merged_audio_embeddings(
        self,
        input_ids: Tensor,
        audio_features: Tensor,
        *,
        input_features_mask: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise ValueError("`input_ids` must have shape [batch, sequence].")
        is_audio = input_ids == self.config.audio_token_index
        safe_ids = input_ids.masked_fill(is_audio, 0)
        embeddings = self.get_input_embeddings()(safe_ids)
        audio_features = audio_features.to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        if input_features_mask is not None:
            if (not isinstance(input_features_mask, Tensor) or input_features_mask.ndim != 2 or
                    tuple(input_features_mask.shape) != tuple(audio_features.shape[:2])):
                raise ValueError("`input_features_mask` must match projected audio "
                                 "[batch, frames].")
            flattened_audio = audio_features[input_features_mask.to(
                device=audio_features.device,
                dtype=torch.bool,
            )]
        else:
            flattened_audio = audio_features.reshape(
                -1,
                audio_features.shape[-1],
            )
        audio_tokens = int(is_audio.sum().item())
        if audio_tokens != flattened_audio.shape[0]:
            raise ValueError(
                "Audio features and placeholder tokens do not match: "
                f"{flattened_audio.shape[0]} features for "
                f"{audio_tokens} tokens.")
        mask = is_audio.unsqueeze(-1).expand_as(embeddings)
        return embeddings.masked_scatter(
            mask,
            flattened_audio.reshape(-1),
        )

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        input_features: Tensor | None = None,
        input_features_mask: Tensor | None = None,
        audio_lengths: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ) -> GraniteSpeechOutput:
        if audio_lengths is not None:
            if (not isinstance(audio_lengths, Tensor) or audio_lengths.ndim != 1):
                raise ValueError("`audio_lengths` must be a rank-one tensor when "
                                 "provided.")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")
        if input_features is not None and inputs_embeds is not None:
            raise ValueError("`input_features` and precomputed `inputs_embeds` are "
                             "mutually exclusive.")
        audio_hidden_states = None
        if inputs_embeds is None and input_features is not None:
            parameter = next(self.encoder.parameters())
            input_features = input_features.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            audio = self.get_audio_features(input_features)
            audio_hidden_states = audio.projected_hidden_state
            if audio_hidden_states is None:
                raise RuntimeError("The Granite Speech projector returned no embeddings.")
            inputs_embeds = self.get_merged_audio_embeddings(
                input_ids,
                audio_hidden_states,
                input_features_mask=input_features_mask,
            )
            input_ids = None
        output: CausalLMOutput = self.language_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )
        return GraniteSpeechOutput(
            logits=output.logits,
            loss=output.loss,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            audio_hidden_states=audio_hidden_states,
        )

    def generate(
            self,
            input_ids: Tensor,
            *,
            input_features: Tensor,
            input_features_mask: Tensor | None = None,
            attention_mask: Tensor | None = None,
            generation_config: GenerationConfig | None = None,
            stopping_criteria: tuple[StoppingCriterion, ...] = (),
    ) -> GenerationOutput:
        """Generate text while presenting audio only on the first step."""
        if attention_mask is None:
            attention_mask = torch.ones_like(
                input_ids,
                dtype=torch.bool,
            )
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError("`attention_mask` must have the same shape as `input_ids`.")
        generation = generation_config or GenerationConfig(
            eos_token_id=self.config.text_config.eos_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            use_cache=self.config.text_config.use_cache,
        )
        prompt_mask = attention_mask.to(device=input_ids.device)

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
                            dtype=prompt_mask.dtype,
                            device=prompt_mask.device,
                        ),
                    ),
                    dim=-1,
                )
            if step.step_index == 0:
                output = self(
                    step.token_ids,
                    input_features=input_features,
                    input_features_mask=input_features_mask,
                    attention_mask=step_mask,
                    past_key_values=step.cache,
                    use_cache=step.use_cache,
                )
            else:
                output = self.language_model(
                    step.token_ids,
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
            generation,
            stopping_criteria=stopping_criteria,
        )


def materialize_granite_speech_nonpersistent_buffers(
    model: GraniteSpeechForConditionalGeneration,
    *,
    device: str | torch.device,
) -> None:
    """Recreate relative-position and RoPE buffers after meta loading."""
    if not isinstance(model, GraniteSpeechForConditionalGeneration):
        raise TypeError("`model` must be GraniteSpeechForConditionalGeneration.")
    from voicehub.neural.rotary import RotaryEmbedding

    target = torch.device(device)
    encoder = model.encoder
    positions = torch.arange(
        encoder.config.context_size,
        device=target,
    )
    distances = positions.view(-1, 1) - positions.view(1, -1)
    encoder.attention_dists = (
        distances.clamp(
            -encoder.config.context_size,
            encoder.config.context_size,
        ) + encoder.config.max_pos_emb)
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
        module.inverse_frequency = (
            1.0 / torch.pow(
                torch.tensor(
                    module.base,
                    dtype=torch.float32,
                    device=target,
                ),
                exponents,
            ))


__all__ = [
    "GraniteSpeechCTCEncoder",
    "GraniteSpeechConformerAttention",
    "GraniteSpeechConformerBlock",
    "GraniteSpeechEncoderOutput",
    "GraniteSpeechEncoderProjector",
    "GraniteSpeechForConditionalGeneration",
    "GraniteSpeechOutput",
    "GraniteSpeechQFormer",
    "materialize_granite_speech_nonpersistent_buffers",
]
