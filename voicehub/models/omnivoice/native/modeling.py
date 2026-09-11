"""Native masked-token OmniVoice model.

OmniVoice uses the parameter layout of a dense Qwen3 base model but
replaces causal attention with bidirectional attention.  Reusing a
causal decoder would silently change both training and iterative
generation, so this module makes that distinction explicit while
retaining the published tensor names.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.causal_lm.native.modeling import (
    CausalSelfAttention,
    GatedMLP,
    _expand_key_values,
    _normalize_position_ids,
)
from voicehub.models.omnivoice.native.configuration import OmniVoiceArchitectureConfig
from voicehub.neural.normalization import RMSNorm
from voicehub.neural.rotary import apply_rotary_embedding


def _bidirectional_attention_bias(
    attention_mask: Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> Tensor:
    allowed = torch.ones(
        (batch_size, 1, sequence_length, sequence_length),
        dtype=torch.bool,
        device=device,
    )
    additive = None
    if attention_mask is not None:
        if not isinstance(attention_mask, Tensor):
            raise TypeError("`attention_mask` must be a PyTorch tensor.")
        if attention_mask.device != device:
            raise ValueError("`attention_mask` must be on the model input device.")
        if attention_mask.ndim == 2:
            if tuple(attention_mask.shape) != (
                    batch_size,
                    sequence_length,
            ):
                raise ValueError("Rank-two attention mask must be [batch, sequence].")
            allowed &= attention_mask.to(torch.bool)[:, None, None, :]
        elif attention_mask.ndim == 4:
            if (attention_mask.shape[0] not in (1, batch_size) or attention_mask.shape[1] != 1 or
                    attention_mask.shape[2] not in (1, sequence_length) or
                    attention_mask.shape[3] != sequence_length):
                raise ValueError(
                    "Rank-four attention mask is not broadcast-compatible "
                    "with [batch, 1, sequence, sequence].")
            if attention_mask.dtype == torch.bool:
                allowed &= attention_mask
            elif attention_mask.is_floating_point():
                if (torch.isnan(attention_mask).any() or torch.isposinf(attention_mask).any()):
                    raise ValueError("Additive attention masks cannot contain NaN or +inf.")
                additive = attention_mask.float()
            else:
                allowed &= attention_mask.to(torch.bool)
        else:
            raise ValueError("`attention_mask` must have rank two or four.")

    has_visible_key = allowed.any(dim=-1, keepdim=True)
    if not bool(has_visible_key.all()):
        fallback = torch.eye(
            sequence_length,
            dtype=torch.bool,
            device=device,
        ).reshape(1, 1, sequence_length, sequence_length)
        allowed = torch.where(has_visible_key, allowed, fallback)

    bias = torch.zeros(
        allowed.shape,
        dtype=torch.float32,
        device=device,
    )
    bias.masked_fill_(~allowed, torch.finfo(torch.float32).min)
    if additive is not None:
        bias = (bias + additive).clamp_min(torch.finfo(torch.float32).min)
    return bias


class OmniVoiceSelfAttention(CausalSelfAttention):
    """Qwen3 grouped-query attention without a causal triangle."""

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        position_ids: Tensor,
        cache=None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        if cache is not None or use_cache:
            raise ValueError("OmniVoice is bidirectional and does not support a KV cache.")
        batch_size, sequence_length, _ = hidden_states.shape
        query = self._shape(
            self.q_proj(hidden_states),
            self.num_attention_heads,
        )
        key = self._shape(
            self.k_proj(hidden_states),
            self.num_key_value_heads,
        )
        value = self._shape(
            self.v_proj(hidden_states),
            self.num_key_value_heads,
        )
        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        cosine, sine = self.rotary(position_ids, dtype=query.dtype)
        query, key = apply_rotary_embedding(
            query,
            key,
            cosine,
            sine,
        )
        key = _expand_key_values(key, self.num_key_value_groups)
        value = _expand_key_values(value, self.num_key_value_groups)
        bias = _bidirectional_attention_bias(
            attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=hidden_states.device,
        )
        scores = torch.matmul(
            query.float(),
            key.float().transpose(-1, -2),
        )
        scores.mul_(self.scaling)
        scores.add_(bias)
        probabilities = functional.softmax(scores, dim=-1).to(query.dtype)
        probabilities = functional.dropout(
            probabilities,
            p=self.attention_dropout if self.training else 0.0,
            training=self.training,
        )
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.num_attention_heads * self.head_dim,
        )
        return (
            self.o_proj(attended),
            probabilities if output_attentions else None,
            None,
        )


class OmniVoiceDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_index: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.self_attn = OmniVoiceSelfAttention(
            config,
            layer_index,
            **factory,
        )
        self.mlp = GatedMLP(config, **factory)
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            **factory,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            **factory,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        position_ids: Tensor,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None]:
        residual = hidden_states
        attention_output, attention, _ = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, attention


@dataclass(frozen=True, slots=True)
class OmniVoiceBackboneOutput:
    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


class OmniVoiceQwen3Backbone(nn.Module):
    """Dense Qwen3 base namespace with bidirectional decoder layers."""

    def __init__(
        self,
        config,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        factory = {"device": device, "dtype": dtype}
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            **factory,
        )
        self.layers = nn.ModuleList(
            OmniVoiceDecoderLayer(
                config,
                index,
                **factory,
            ) for index in range(config.num_hidden_layers))
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            **factory,
        )
        self.gradient_checkpointing = False
        if initialize:
            self.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> OmniVoiceBackboneOutput:
        if (not isinstance(inputs_embeds, Tensor) or inputs_embeds.ndim != 3 or
                inputs_embeds.shape[-1] != self.config.hidden_size):
            raise ValueError("`inputs_embeds` must be [batch, sequence, hidden_size].")
        if inputs_embeds.shape[0] == 0 or inputs_embeds.shape[1] == 0:
            raise ValueError("OmniVoice embeddings cannot have an empty axis.")
        hidden_states = inputs_embeds
        batch_size, sequence_length, _ = hidden_states.shape
        position_ids = _normalize_position_ids(
            position_ids,
            attention_mask=attention_mask,
            batch_size=batch_size,
            query_length=sequence_length,
            past_length=0,
            max_position_embeddings=self.config.max_position_embeddings,
            device=hidden_states.device,
        )

        hidden_history = [] if output_hidden_states else None
        attention_history = [] if output_attentions else None
        for layer in self.layers:
            if hidden_history is not None:
                hidden_history.append(hidden_states)
            if (self.gradient_checkpointing and self.training and not output_attentions):

                def custom_forward(
                    states: Tensor,
                    current_layer: OmniVoiceDecoderLayer = layer,
                ) -> Tensor:
                    output, _ = current_layer(
                        states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                    )
                    return output

                hidden_states = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    use_reentrant=False,
                )
                attention = None
            else:
                hidden_states, attention = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                )
            if attention_history is not None:
                if attention is None:
                    raise RuntimeError("OmniVoice layer omitted requested attention.")
                attention_history.append(attention)
        hidden_states = self.norm(hidden_states)
        if hidden_history is not None:
            hidden_history.append(hidden_states)
        return OmniVoiceBackboneOutput(
            last_hidden_state=hidden_states,
            hidden_states=(tuple(hidden_history) if hidden_history is not None else None),
            attentions=(tuple(attention_history) if attention_history is not None else None),
        )


@dataclass(frozen=True, slots=True)
class OmniVoiceModelOutput:
    loss: Tensor | None
    logits: Tensor
    codebook_losses: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


class OmniVoiceModel(nn.Module):
    """Checkpoint-exact OmniVoice masked-token model."""

    def __init__(
        self,
        config: OmniVoiceArchitectureConfig,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not isinstance(config, OmniVoiceArchitectureConfig):
            raise TypeError("`config` must be an OmniVoiceArchitectureConfig.")
        self.config = config
        self.llm = OmniVoiceQwen3Backbone(
            config.llm_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.audio_embeddings = nn.Embedding(
            config.num_audio_codebook * config.audio_vocab_size,
            config.llm_config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "codebook_layer_offsets",
            torch.arange(
                config.num_audio_codebook,
                device=device,
            ) * config.audio_vocab_size,
        )
        self.audio_heads = nn.Linear(
            config.llm_config.hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if initialize:
            nn.init.normal_(
                self.audio_embeddings.weight,
                mean=0.0,
                std=config.llm_config.initializer_range,
            )
            nn.init.normal_(
                self.audio_heads.weight,
                mean=0.0,
                std=config.llm_config.initializer_range,
            )

    @property
    def device(self) -> torch.device:
        return self.audio_embeddings.weight.device

    def get_input_embeddings(self) -> nn.Embedding:
        return self.llm.embed_tokens

    def gradient_checkpointing_enable(self) -> None:
        self.llm.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.llm.gradient_checkpointing_disable()

    def _validate_inputs(
        self,
        input_ids: Tensor,
        audio_mask: Tensor,
    ) -> None:
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 3:
            raise ValueError("`input_ids` must have shape [batch, codebook, sequence].")
        if (input_ids.dtype == torch.bool or input_ids.is_floating_point() or input_ids.is_complex()):
            raise TypeError("`input_ids` must use an integer dtype.")
        if input_ids.shape[1] != self.config.num_audio_codebook:
            raise ValueError("`input_ids` codebook axis does not match the model config.")
        if (not isinstance(audio_mask, Tensor) or audio_mask.dtype != torch.bool or
                tuple(audio_mask.shape) != (input_ids.shape[0], input_ids.shape[2])):
            raise ValueError("`audio_mask` must be boolean [batch, sequence].")
        if input_ids.device != audio_mask.device:
            raise ValueError("OmniVoice inputs must share a device.")
        audio_ids = input_ids.masked_select(audio_mask[:, None, :])
        text_ids = input_ids[:, 0, :].masked_select(~audio_mask)
        if (audio_ids.numel() and ((audio_ids < 0).any() or
                                   (audio_ids >= self.config.audio_vocab_size).any())):
            raise ValueError("An audio token ID is outside the audio vocabulary.")
        if (text_ids.numel() and ((text_ids < 0).any() or
                                  (text_ids >= self.config.llm_config.vocab_size).any())):
            raise ValueError("A text token ID is outside the text vocabulary.")

    def prepare_embeddings(
        self,
        input_ids: Tensor,
        audio_mask: Tensor,
    ) -> Tensor:
        self._validate_inputs(input_ids, audio_mask)
        text_embeddings = self.get_input_embeddings()(input_ids[:, 0, :])
        shifted_ids = (input_ids * audio_mask.unsqueeze(1) + self.codebook_layer_offsets.view(1, -1, 1))
        audio_embeddings = self.audio_embeddings(shifted_ids).sum(dim=1)
        return torch.where(
            audio_mask.unsqueeze(-1),
            audio_embeddings,
            text_embeddings,
        )

    def forward(
        self,
        input_ids: Tensor,
        audio_mask: Tensor,
        *,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        document_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> OmniVoiceModelOutput:
        inputs_embeds = self.prepare_embeddings(input_ids, audio_mask)
        if attention_mask is not None and document_ids is not None:
            raise ValueError("Pass either `attention_mask` or `document_ids`, not both.")
        if document_ids is not None:
            if (not isinstance(document_ids, Tensor) or document_ids.ndim != 2 or
                    tuple(document_ids.shape) != (input_ids.shape[0], input_ids.shape[2])):
                raise ValueError("`document_ids` must have shape [batch, sequence].")
            attention_mask = (document_ids[:, None, :, None] == document_ids[:, None, None, :])

        backbone = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        batch_size, sequence_length, _ = (backbone.last_hidden_state.shape)
        logits = self.audio_heads(backbone.last_hidden_state)
        logits = logits.view(
            batch_size,
            sequence_length,
            self.config.num_audio_codebook,
            self.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)

        loss = None
        codebook_losses = None
        if labels is not None:
            if (not isinstance(labels, Tensor) or tuple(labels.shape) != tuple(input_ids.shape)):
                raise ValueError("`labels` must match [batch, codebook, sequence].")
            if (labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex()):
                raise TypeError("`labels` must use an integer dtype.")
            invalid = (labels != -100) & ((labels < 0) | (labels >= self.config.audio_vocab_size))
            if invalid.any():
                raise ValueError("OmniVoice labels must be -100 or an audio token ID.")
            per_token = functional.cross_entropy(
                logits.permute(0, 3, 1, 2),
                labels,
                reduction="none",
                ignore_index=-100,
            )
            valid = (labels != -100).to(per_token.dtype)
            codebook_losses = ((per_token * valid).sum(dim=(0, 2)) / valid.sum(dim=(0, 2)).clamp_min(1.0))
            weights = logits.new_tensor(self.config.normalized_audio_codebook_weights, )
            loss = (codebook_losses * weights).sum()

        return OmniVoiceModelOutput(
            loss=loss,
            logits=logits,
            codebook_losses=codebook_losses,
            hidden_states=backbone.hidden_states,
            attentions=backbone.attentions,
        )


__all__ = [
    "OmniVoiceBackboneOutput",
    "OmniVoiceModel",
    "OmniVoiceModelOutput",
    "OmniVoiceQwen3Backbone",
]
