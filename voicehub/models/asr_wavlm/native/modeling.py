"""PyTorch-only WavLM CTC architecture owned by VoiceHub.

The implementation follows Microsoft's official WavLM graph at revision
``833df7e7832e5064a281131ee64a481afa8e5b95`` and the checkpoint-compatible
Hugging Face graph at Transformers revision
``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``.  Structurally identical
Wav2Vec2 frontend components are reused, while WavLM's learned mask vector,
bucketed relative-position bias, GRU-style bias gate, LayerDrop semantics,
and ``wavlm.*`` parameter namespace remain architecture-specific.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.asr_wav2vec2.native.modeling import (
    Float32LayerNorm,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2FeedForward,
    Wav2Vec2PositionalConvEmbedding,
    _span_mask,
    _validate_floating_input,
    _validated_raw_attention_mask,
    downsample_wav2vec2_lengths,
    feature_attention_mask,
)
from voicehub.models.asr_wavlm.native.configuration import WavLMConfig
from voicehub.objectives.ctc import ctc_loss


class WavLMAttention(nn.Module):
    """Self-attention with WavLM's gated bucketed relative-position bias."""

    def __init__(
        self,
        config: WavLMConfig,
        *,
        has_relative_position_bias: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_heads
        self.dropout = config.attention_dropout
        self.num_buckets = config.num_buckets
        self.max_distance = config.max_bucket_distance
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, self.num_heads, 1, 1), )
        self.gru_rel_pos_linear = nn.Linear(self.head_size, 8)
        if has_relative_position_bias:
            self.rel_attn_embed = nn.Embedding(
                self.num_buckets,
                self.num_heads,
            )

    def _split_heads(self, value: Tensor) -> Tensor:
        batch_size, steps, _ = value.shape
        return value.reshape(
            batch_size,
            steps,
            self.num_heads,
            self.head_size,
        ).transpose(1, 2)

    def relative_position_buckets(self, relative_positions: Tensor) -> Tensor:
        """Map signed distances to the exact WavLM logarithmic buckets."""
        if not isinstance(relative_positions, Tensor):
            raise TypeError("`relative_positions` must be a PyTorch tensor.")
        if (relative_positions.dtype == torch.bool or relative_positions.is_floating_point() or
                relative_positions.is_complex()):
            raise TypeError("`relative_positions` must use an integer dtype.")

        buckets_per_direction = self.num_buckets // 2
        buckets = (relative_positions > 0).to(dtype=torch.long) * buckets_per_direction
        distances = relative_positions.abs()
        maximum_exact = buckets_per_direction // 2
        is_exact = distances < maximum_exact
        # Clamp only the inactive logarithmic branch at zero. This avoids a
        # transient log(0) while preserving every selected upstream value.
        logarithmic = torch.log(distances.to(dtype=torch.float32).clamp_min(maximum_exact) / maximum_exact)
        logarithmic = logarithmic / math.log(self.max_distance / maximum_exact)
        logarithmic = logarithmic * (buckets_per_direction - maximum_exact)
        logarithmic = maximum_exact + logarithmic.to(dtype=torch.long)
        logarithmic = logarithmic.clamp_max(buckets_per_direction - 1)
        return buckets + torch.where(is_exact, distances, logarithmic)

    def compute_position_bias(
        self,
        query_length: int,
        key_length: int,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Create raw per-head position bias for the first encoder layer."""
        for name, value in (
            ("query_length", query_length),
            ("key_length", key_length),
            ("batch_size", batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"`{name}` must be an integer.")
            if value < 1:
                raise ValueError(f"`{name}` must be positive.")
        if not hasattr(self, "rel_attn_embed"):
            raise RuntimeError("Only WavLM's first encoder layer can create position bias.")
        query_positions = torch.arange(
            query_length,
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        key_positions = torch.arange(
            key_length,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        buckets = self.relative_position_buckets(key_positions - query_positions, )
        values = self.rel_attn_embed(buckets).permute(2, 0, 1)
        return values.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def _gated_position_bias(
        self,
        hidden_states: Tensor,
        position_bias: Tensor,
    ) -> Tensor:
        batch_size, steps, _ = hidden_states.shape
        expected_shape = (
            batch_size,
            self.num_heads,
            steps,
            steps,
        )
        if tuple(position_bias.shape) != expected_shape:
            raise ValueError(
                "WavLM position bias must have shape "
                f"{expected_shape}; found {tuple(position_bias.shape)}.")
        gated_hidden = hidden_states.reshape(
            batch_size,
            steps,
            self.num_heads,
            self.head_size,
        ).permute(0, 2, 1, 3)
        projected = self.gru_rel_pos_linear(gated_hidden)
        projected = projected.reshape(
            batch_size,
            self.num_heads,
            steps,
            2,
            4,
        ).sum(dim=-1)
        gate_a, gate_b = torch.sigmoid(projected).chunk(2, dim=-1)
        gate = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
        return gate * position_bias

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        position_bias: Tensor | None,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("WavLM attention expects [batch, time, hidden].")
        batch_size, steps, width = hidden_states.shape
        if width != self.hidden_size:
            raise ValueError("WavLM attention hidden width is incompatible.")
        if position_bias is None:
            position_bias = self.compute_position_bias(
                steps,
                steps,
                batch_size=batch_size,
                device=hidden_states.device,
            )
        elif position_bias.device != hidden_states.device:
            raise ValueError("WavLM position bias must share the hidden-state device.")

        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))
        working_query = (query.float() if query.dtype in (torch.float16, torch.bfloat16) else query)
        working_key = (key.float() if key.dtype in (torch.float16, torch.bfloat16) else key)
        scores = torch.matmul(
            working_query,
            working_key.transpose(-1, -2),
        ) * (self.head_size**-0.5)
        gated_bias = self._gated_position_bias(
            hidden_states,
            position_bias,
        )
        scores = scores + gated_bias.to(dtype=scores.dtype)

        if attention_mask is not None:
            if (attention_mask.dtype != torch.bool or tuple(attention_mask.shape) != (batch_size, steps)):
                raise ValueError("Encoder attention mask must be boolean [batch, time].")
            scores = scores.masked_fill(
                ~attention_mask[:, None, None, :],
                -torch.inf,
            )

        probabilities = torch.softmax(scores, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        dropped_probabilities = functional.dropout(
            probabilities,
            p=self.dropout,
            training=self.training,
        ).to(dtype=value.dtype)
        attended = torch.matmul(dropped_probabilities, value)
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            steps,
            self.hidden_size,
        )
        output = self.out_proj(attended)
        returned_attention = (dropped_probabilities if output_attentions else None)
        return output, returned_attention, position_bias


class WavLMEncoderLayer(nn.Module):
    """Post-normalized WavLM encoder layer."""

    def __init__(
        self,
        config: WavLMConfig,
        *,
        has_relative_position_bias: bool,
    ) -> None:
        super().__init__()
        self.attention = WavLMAttention(
            config,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = Float32LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = Float32LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor,
        position_bias: Tensor | None,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        residual = hidden_states
        attended, attention, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = self.layer_norm(residual + self.dropout(attended), )
        hidden_states = self.final_layer_norm(hidden_states + self.feed_forward(hidden_states), )
        return hidden_states, attention, position_bias


class WavLMEncoderLayerStableLayerNorm(nn.Module):
    """Pre-normalized WavLM encoder layer."""

    def __init__(
        self,
        config: WavLMConfig,
        *,
        has_relative_position_bias: bool,
    ) -> None:
        super().__init__()
        self.attention = WavLMAttention(
            config,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = Float32LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = Float32LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor,
        position_bias: Tensor | None,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        residual = hidden_states
        attended, attention, position_bias = self.attention(
            self.layer_norm(hidden_states),
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = residual + self.dropout(attended)
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states), )
        return hidden_states, attention, position_bias


@dataclass(frozen=True)
class WavLMEncoderOutput:
    """Result of the native WavLM encoder."""

    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor | None, ...] | None = None
    executed_layers: tuple[bool, ...] = ()


class WavLMEncoder(nn.Module):
    """Positional convolution and gated relative-position Transformer."""

    def __init__(self, config: WavLMConfig) -> None:
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = Float32LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        layer_type = (WavLMEncoderLayerStableLayerNorm if config.do_stable_layer_norm else WavLMEncoderLayer)
        self.layers = nn.ModuleList(
            layer_type(
                config,
                has_relative_position_bias=index == 0,
            ) for index in range(config.num_hidden_layers))

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        generator: torch.Generator | None = None,
    ) -> WavLMEncoderOutput:
        if hidden_states.ndim != 3:
            raise ValueError("WavLM encoder input must be [batch, time, hidden].")
        if (attention_mask.dtype != torch.bool or
                tuple(attention_mask.shape) != tuple(hidden_states.shape[:2])):
            raise ValueError("Encoder attention mask must be boolean [batch, time].")

        hidden_states = hidden_states.masked_fill(
            ~attention_mask.unsqueeze(-1),
            0.0,
        )
        hidden_states = hidden_states + self.pos_conv_embed(hidden_states)
        if not self.config.do_stable_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        collected_states: list[Tensor] | None = ([] if output_hidden_states else None)
        collected_attentions: list[Tensor | None] | None = ([] if output_attentions else None)
        executed_layers: list[bool] = []
        position_bias = None
        for index, layer in enumerate(self.layers):
            if collected_states is not None:
                collected_states.append(hidden_states)
            skip_layer = False
            if (self.training and index > 0 and self.config.layerdrop > 0.0):
                probability = torch.rand(
                    (),
                    device=hidden_states.device,
                    generator=generator,
                )
                skip_layer = bool(probability < self.config.layerdrop)
            if skip_layer:
                attention = None
            else:
                hidden_states, attention, position_bias = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                    output_attentions=output_attentions,
                )
            executed_layers.append(not skip_layer)
            if collected_attentions is not None:
                collected_attentions.append(attention)

        if self.config.do_stable_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        if collected_states is not None:
            collected_states.append(hidden_states)
        return WavLMEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=(None if collected_states is None else tuple(collected_states)),
            attentions=(None if collected_attentions is None else tuple(collected_attentions)),
            executed_layers=tuple(executed_layers),
        )


@dataclass(frozen=True)
class WavLMModelOutput:
    """Native WavLM output with exact frame lengths and diagnostics."""

    last_hidden_state: Tensor
    extract_features: Tensor
    feature_attention_mask: Tensor
    input_lengths: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor | None, ...] | None = None
    executed_layers: tuple[bool, ...] = ()
    past_key_values: None = None


class WavLMModel(nn.Module):
    """Native WavLM raw-waveform frontend and bidirectional encoder."""

    def __init__(
        self,
        config: WavLMConfig | Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.config = WavLMConfig.coerce(config)
        self.feature_extractor = Wav2Vec2FeatureEncoder(self.config)
        self.feature_projection = Wav2Vec2FeatureProjection(self.config)
        self.encoder = WavLMEncoder(self.config)
        if (self.config.mask_time_prob > 0.0 or self.config.mask_feature_prob > 0.0):
            self.masked_spec_embed = nn.Parameter(torch.empty(self.config.hidden_size), )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        config = self.config

        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    bound = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]), )
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        # Match the published family initialization exceptions.
        self.feature_projection.projection.reset_parameters()
        self.encoder.pos_conv_embed.conv.reset_parameters()
        for layer in self.encoder.layers:
            nn.init.ones_(layer.attention.gru_rel_pos_const)
        if hasattr(self, "masked_spec_embed"):
            nn.init.uniform_(self.masked_spec_embed)

    @staticmethod
    def _reject_cache(
        use_cache: bool | None,
        past_key_values: Any | None,
    ) -> None:
        if use_cache not in (None, False):
            raise ValueError(
                "WavLM is a bidirectional encoder and does not support "
                "causal key/value caching.")
        if past_key_values is not None:
            raise ValueError("`past_key_values` is invalid for the bidirectional WavLM "
                             "encoder.")

    def _apply_spec_augment(
        self,
        hidden_states: Tensor,
        feature_mask: Tensor,
        *,
        mask_time_indices: Tensor | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        if not self.config.apply_spec_augment:
            return hidden_states
        if mask_time_indices is not None:
            if (not isinstance(mask_time_indices, Tensor) or
                    tuple(mask_time_indices.shape) != tuple(feature_mask.shape)):
                raise ValueError("`mask_time_indices` must have shape "
                                 "[batch, feature_time].")
            if mask_time_indices.device != hidden_states.device:
                raise ValueError("`mask_time_indices` must be on the model input device.")
            if not (mask_time_indices.dtype == torch.bool or
                    ((mask_time_indices == 0) | (mask_time_indices == 1)).all()):
                raise ValueError("`mask_time_indices` must contain only zero and one.")
            time_mask = mask_time_indices.to(dtype=torch.bool)
            if (time_mask & ~feature_mask).any():
                raise ValueError("`mask_time_indices` cannot select padded feature frames.")
        elif self.training and self.config.mask_time_prob > 0.0:
            time_mask = _span_mask(
                feature_mask,
                probability=self.config.mask_time_prob,
                span_length=self.config.mask_time_length,
                minimum_spans=self.config.mask_time_min_masks,
                generator=generator,
            )
        else:
            time_mask = None

        if time_mask is not None and time_mask.any():
            if not hasattr(self, "masked_spec_embed"):
                raise ValueError(
                    "WavLM cannot apply time masking because this "
                    "configuration has no learned mask embedding.")
            replacement = self.masked_spec_embed.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            hidden_states = torch.where(
                time_mask.unsqueeze(-1),
                replacement.reshape(1, 1, -1),
                hidden_states,
            )

        if (self.training and self.config.mask_feature_prob > 0.0):
            valid_features = torch.ones(
                (hidden_states.shape[0], hidden_states.shape[2]),
                dtype=torch.bool,
                device=hidden_states.device,
            )
            feature_spans = _span_mask(
                valid_features,
                probability=self.config.mask_feature_prob,
                span_length=self.config.mask_feature_length,
                minimum_spans=self.config.mask_feature_min_masks,
                generator=generator,
            )
            hidden_states = hidden_states.masked_fill(
                feature_spans.unsqueeze(1),
                0.0,
            )
        return hidden_states

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None = None,
        *,
        mask_time_indices: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool | None = None,
        past_key_values: Any | None = None,
        generator: torch.Generator | None = None,
    ) -> WavLMModelOutput:
        _validate_floating_input(input_values, self.config)
        self._reject_cache(use_cache, past_key_values)
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("`generator` must be a PyTorch Generator or None.")
        for name, value in (
            ("output_attentions", output_attentions),
            ("output_hidden_states", output_hidden_states),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"`{name}` must be a boolean.")

        raw_mask, raw_lengths = _validated_raw_attention_mask(
            attention_mask,
            input_values=input_values,
            minimum_input_samples=self.config.minimum_input_samples,
        )
        masked_input = input_values.masked_fill(~raw_mask, 0.0)
        extract_features = self.feature_extractor(masked_input).transpose(1, 2)
        output_lengths = downsample_wav2vec2_lengths(
            raw_lengths,
            self.config,
        )
        encoded_mask = feature_attention_mask(
            extract_features.shape[1],
            output_lengths,
        )
        hidden_states, normalized_features = self.feature_projection(extract_features, )
        hidden_states = self._apply_spec_augment(
            hidden_states,
            encoded_mask,
            mask_time_indices=mask_time_indices,
            generator=generator,
        )
        encoded = self.encoder(
            hidden_states,
            attention_mask=encoded_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generator=generator,
        )
        return WavLMModelOutput(
            last_hidden_state=encoded.last_hidden_state,
            extract_features=normalized_features,
            feature_attention_mask=encoded_mask,
            input_lengths=output_lengths,
            hidden_states=encoded.hidden_states,
            attentions=encoded.attentions,
            executed_layers=encoded.executed_layers,
        )

    def freeze_feature_encoder(self) -> None:
        """Freeze only the raw-waveform convolutional frontend."""
        self.feature_extractor.freeze()


@dataclass(frozen=True)
class WavLMCTCOutput:
    """WavLM CTC logits, optional loss, and encoder diagnostics."""

    logits: Tensor
    loss: Tensor | None
    feature_attention_mask: Tensor
    input_lengths: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor | None, ...] | None = None
    executed_layers: tuple[bool, ...] = ()
    past_key_values: None = None


class WavLMForCTC(nn.Module):
    """WavLM encoder with a trainable native CTC projection head."""

    def __init__(
        self,
        config: WavLMConfig | Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.config = WavLMConfig.coerce(config)
        self.wavlm = WavLMModel(self.config)
        self.dropout = nn.Dropout(self.config.final_dropout)
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
        )
        nn.init.normal_(
            self.lm_head.weight,
            mean=0.0,
            std=self.config.initializer_range,
        )
        nn.init.zeros_(self.lm_head.bias)

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None = None,
        *,
        labels: Tensor | None = None,
        mask_time_indices: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool | None = None,
        past_key_values: Any | None = None,
        generator: torch.Generator | None = None,
    ) -> WavLMCTCOutput:
        outputs = self.wavlm(
            input_values,
            attention_mask,
            mask_time_indices=mask_time_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            generator=generator,
        )
        logits = self.lm_head(self.dropout(outputs.last_hidden_state))

        loss = None
        if labels is not None:
            if not isinstance(labels, Tensor):
                raise TypeError("`labels` must be a PyTorch tensor.")
            if (labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex()):
                raise TypeError("`labels` must use an integer dtype.")
            if labels.ndim != 2 or labels.shape[0] != input_values.shape[0]:
                raise ValueError("`labels` must have shape [batch, target_time].")
            if labels.device != logits.device:
                raise ValueError("`labels` must be on the model input device.")
            if ((labels < 0) & (labels != -100)).any():
                raise ValueError("Negative CTC labels must use the -100 ignore index.")
            valid_labels = labels >= 0
            if (labels.masked_select(valid_labels) >= self.config.vocab_size).any():
                raise ValueError("WavLM CTC labels must be smaller than `vocab_size`.")
            target_lengths = valid_labels.sum(dim=-1, dtype=torch.long)
            targets = labels.masked_select(valid_labels)
            loss = ctc_loss(
                logits,
                targets,
                outputs.input_lengths,
                target_lengths,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )

        return WavLMCTCOutput(
            logits=logits,
            loss=loss,
            feature_attention_mask=outputs.feature_attention_mask,
            input_lengths=outputs.input_lengths,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            executed_layers=outputs.executed_layers,
        )

    def freeze_feature_encoder(self) -> None:
        """Freeze only the raw-waveform convolutional frontend."""
        self.wavlm.freeze_feature_encoder()

    def freeze_base_model(self) -> None:
        """Freeze the WavLM base graph while keeping the CTC head trainable."""
        for parameter in self.wavlm.parameters():
            parameter.requires_grad_(False)


__all__ = [
    "WavLMAttention",
    "WavLMCTCOutput",
    "WavLMEncoder",
    "WavLMEncoderLayer",
    "WavLMEncoderLayerStableLayerNorm",
    "WavLMEncoderOutput",
    "WavLMForCTC",
    "WavLMModel",
    "WavLMModelOutput",
]
