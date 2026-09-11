"""PyTorch-only HuBERT CTC architecture owned by VoiceHub.

The implementation follows Hugging Face Transformers' Apache-2.0 HuBERT
graph at revision ``ebea912f0bb6f9e28ad2df04acd9b4df035933a9`` without
importing that runtime. Structurally identical Wav2Vec2 frontend and
encoder blocks are reused, while the ``hubert.*`` namespace, learned
mask embedding, feature-projection switch, and CTC head remain HuBERT-
specific.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.models.asr_hubert.native.configuration import HubertConfig
from voicehub.models.asr_wav2vec2.native.modeling import (
    Float32LayerNorm,
    Wav2Vec2Encoder,
    Wav2Vec2FeatureEncoder,
    _span_mask,
    _validate_floating_input,
    _validated_raw_attention_mask,
    downsample_wav2vec2_lengths,
    feature_attention_mask,
)
from voicehub.objectives.ctc import ctc_loss


class HubertFeatureProjection(nn.Module):
    """Optionally normalize convolutional features before projection."""

    def __init__(self, config: HubertConfig) -> None:
        super().__init__()
        self.layer_norm = (
            Float32LayerNorm(
                config.conv_dim[-1],
                eps=config.layer_norm_eps,
            ) if config.feat_proj_layer_norm else None)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        normalized = (hidden_states if self.layer_norm is None else self.layer_norm(hidden_states))
        return self.dropout(self.projection(normalized)), normalized


@dataclass(frozen=True)
class HubertModelOutput:
    """Native HuBERT output with exact frame lengths and diagnostics."""

    last_hidden_state: Tensor
    extract_features: Tensor
    feature_attention_mask: Tensor
    input_lengths: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor | None, ...] | None = None
    executed_layers: tuple[bool, ...] = ()
    past_key_values: None = None


class HubertModel(nn.Module):
    """Native HuBERT feature frontend and bidirectional encoder."""

    def __init__(
        self,
        config: HubertConfig | Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.config = HubertConfig.coerce(config)
        self.feature_extractor = Wav2Vec2FeatureEncoder(self.config)
        self.feature_projection = HubertFeatureProjection(self.config)
        self.encoder = Wav2Vec2Encoder(self.config)
        if (self.config.mask_time_prob > 0.0 or self.config.mask_feature_prob > 0.0):
            self.masked_spec_embed = nn.Parameter(torch.empty(self.config.hidden_size))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        config = self.config

        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        # Match the source family initialization exceptions.
        self.feature_projection.projection.reset_parameters()
        self.encoder.pos_conv_embed.conv.reset_parameters()
        if hasattr(self, "masked_spec_embed"):
            nn.init.uniform_(self.masked_spec_embed)

    @staticmethod
    def _reject_cache(
        use_cache: bool | None,
        past_key_values: Any | None,
    ) -> None:
        if use_cache not in (None, False):
            raise ValueError(
                "HuBERT is a bidirectional encoder and does not support "
                "causal key/value caching.")
        if past_key_values is not None:
            raise ValueError("`past_key_values` is invalid for the bidirectional HuBERT "
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
                    "HuBERT cannot apply time masking because this "
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
    ) -> HubertModelOutput:
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
        hidden_states, normalized_features = self.feature_projection(extract_features)
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
        )
        return HubertModelOutput(
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
class HubertCTCOutput:
    """HuBERT CTC logits, optional loss, and encoder diagnostics."""

    logits: Tensor
    loss: Tensor | None
    feature_attention_mask: Tensor
    input_lengths: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor | None, ...] | None = None
    executed_layers: tuple[bool, ...] = ()
    past_key_values: None = None


class HubertForCTC(nn.Module):
    """HuBERT encoder with a trainable native CTC projection head."""

    def __init__(
        self,
        config: HubertConfig | Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.config = HubertConfig.coerce(config)
        self.hubert = HubertModel(self.config)
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
    ) -> HubertCTCOutput:
        outputs = self.hubert(
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
                raise ValueError("HuBERT CTC labels must be smaller than `vocab_size`.")
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

        return HubertCTCOutput(
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
        self.hubert.freeze_feature_encoder()

    def freeze_base_model(self) -> None:
        """Freeze the HuBERT base graph while keeping the CTC head
        trainable."""
        for parameter in self.hubert.parameters():
            parameter.requires_grad_(False)


__all__ = [
    "HubertCTCOutput",
    "HubertFeatureProjection",
    "HubertForCTC",
    "HubertModel",
    "HubertModelOutput",
]
