"""PyTorch-only Higgs Audio v2 dual-FFN language model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.models.causal_lm.native.modeling import CausalSelfAttention, GatedMLP, _normalize_position_ids
from voicehub.models.higgstts.native.configuration import HiggsAudioV2Config
from voicehub.neural.cache import DynamicKVCache
from voicehub.neural.normalization import RMSNorm
from voicehub.neural.rotary import RotaryEmbedding
from voicehub.objectives.sequence import sequence_cross_entropy


@dataclass(frozen=True)
class HiggsAudioV2ModelOutput:
    """Hidden states and optional decoder cache."""

    last_hidden_state: Tensor
    past_key_values: DynamicKVCache | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass(frozen=True)
class HiggsAudioV2Output:
    """Joint text/audio logits and source-aligned causal objectives."""

    logits: Tensor
    text_logits: Tensor
    loss: Tensor | None = None
    audio_loss: Tensor | None = None
    text_loss: Tensor | None = None
    codebook_losses: tuple[Tensor, ...] = ()
    past_key_values: DynamicKVCache | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


class HiggsAudioV2Embeddings(nn.Module):
    """Sum offset codebook embeddings for one delayed audio frame."""

    def __init__(
        self,
        config: HiggsAudioV2Config,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_audio_tokens = nn.Embedding(
            config.audio_vocabulary_size,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "audio_tokens_offsets",
            torch.arange(
                config.num_codebooks,
                device=device,
                dtype=torch.long,
            ) * config.codebook_size,
            persistent=False,
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        if (not isinstance(input_ids, Tensor) or input_ids.ndim != 3 or
                input_ids.shape[-1] != self.config.num_codebooks):
            raise ValueError("Higgs audio IDs must have shape "
                             "[batch, frames, num_codebooks].")
        if (input_ids.dtype == torch.bool or input_ids.is_floating_point() or input_ids.is_complex()):
            raise TypeError("Higgs audio IDs must use an integer dtype.")
        if input_ids.numel() and (int(input_ids.min()) < 0 or
                                  int(input_ids.max()) >= self.config.codebook_size):
            raise ValueError("A Higgs audio code is outside the codebook.")
        offsets = self.audio_tokens_offsets.to(device=input_ids.device)
        embedded = self.embed_audio_tokens(input_ids.long() + offsets)
        return embedded.sum(dim=-2)


class HiggsAudioV2DecoderLayer(nn.Module):
    """Shared attention with separate text and audio norms/MLPs."""

    def __init__(
        self,
        config: HiggsAudioV2Config,
        layer_index: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        shared = config.as_causal_lm_config()
        factory = {"device": device, "dtype": dtype}
        self.self_attn = CausalSelfAttention(
            shared,
            layer_index,
            **factory,
        )
        self.mlp = GatedMLP(shared, **factory)
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
        self.audio_mlp = GatedMLP(shared, **factory)
        self.audio_input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            **factory,
        )
        self.audio_post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            **factory,
        )

    @staticmethod
    def _select(
        hidden_states: Tensor,
        audio_token_mask: Tensor | None,
        *,
        text_module: nn.Module,
        audio_module: nn.Module,
    ) -> Tensor:
        if audio_token_mask is None:
            return audio_module(hidden_states)
        result = torch.empty_like(hidden_states)
        if audio_token_mask.any():
            result[audio_token_mask] = audio_module(hidden_states[audio_token_mask])
        if (~audio_token_mask).any():
            result[~audio_token_mask] = text_module(hidden_states[~audio_token_mask])
        return result

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        audio_token_mask: Tensor | None,
        position_ids: Tensor,
        cache: DynamicKVCache | None,
        use_cache: bool,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None, DynamicKVCache | None]:
        residual = hidden_states
        normalized = self._select(
            hidden_states,
            audio_token_mask,
            text_module=self.input_layernorm,
            audio_module=self.audio_input_layernorm,
        )
        attended, attention, cache = self.self_attn(
            normalized,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attended
        residual = hidden_states
        projected = self._select(
            hidden_states,
            audio_token_mask,
            text_module=lambda value: self.mlp(self.post_attention_layernorm(value)),
            audio_module=lambda value: self.audio_mlp(self.audio_post_attention_layernorm(value)),
        )
        return residual + projected, attention, cache


class HiggsAudioV2Model(nn.Module):
    """Exact 28-layer dual-path backbone for the official 3B checkpoint."""

    def __init__(
        self,
        config: HiggsAudioV2Config | dict[str, Any],
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = HiggsAudioV2Config.coerce(config)
        factory = {"device": device, "dtype": dtype}
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            self.config.pad_token_id,
            **factory,
        )
        self.layers = nn.ModuleList(
            HiggsAudioV2DecoderLayer(
                self.config,
                index,
                **factory,
            ) for index in range(self.config.num_hidden_layers))
        self.norm = RMSNorm(
            self.config.hidden_size,
            epsilon=self.config.rms_norm_eps,
            **factory,
        )
        self.embed_audio_tokens = HiggsAudioV2Embeddings(
            self.config,
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

    def materialize_runtime_buffers(self, device) -> None:
        """Rebuild non-persistent RoPE/offset buffers after meta loading."""
        target = torch.device(device)
        for module in self.modules():
            if isinstance(module, RotaryEmbedding):
                replacement = RotaryEmbedding(
                    module.dimension,
                    base=module.base,
                    scaling=dict(self.config.rope_parameters),
                    device=target,
                )
                module.inverse_frequency = replacement.inverse_frequency
            elif isinstance(module, HiggsAudioV2Embeddings):
                module.audio_tokens_offsets = (
                    torch.arange(
                        self.config.num_codebooks,
                        device=target,
                        dtype=torch.long,
                    ) * self.config.codebook_size)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def _embeddings(
        self,
        input_ids: Tensor | None,
        audio_input_ids: Tensor | None,
        audio_input_ids_mask: Tensor | None,
        inputs_embeds: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        supplied = sum(value is not None for value in (input_ids, audio_input_ids, inputs_embeds))
        if supplied == 0:
            raise ValueError("Specify text IDs, audio IDs, or input embeddings.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify only one of `input_ids` and `inputs_embeds`.")
        audio_token_mask = None
        if input_ids is not None:
            if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
                raise ValueError("`input_ids` must have shape [batch, sequence].")
            if (input_ids.dtype == torch.bool or input_ids.is_floating_point() or input_ids.is_complex()):
                raise TypeError("`input_ids` must use an integer dtype.")
            if input_ids.numel() and (int(input_ids.min()) < 0 or
                                      int(input_ids.max()) >= self.config.vocab_size):
                raise ValueError("A Higgs text ID is outside the vocabulary.")
            inputs_embeds = self.embed_tokens(input_ids.long())
            audio_token_mask = ((input_ids == self.config.audio_token_id)
                                | (input_ids == self.config.audio_delay_token_id))
        elif inputs_embeds is not None:
            if (not isinstance(inputs_embeds, Tensor) or inputs_embeds.ndim != 3 or
                    inputs_embeds.shape[-1] != self.config.hidden_size):
                raise ValueError(
                    "`inputs_embeds` must have shape "
                    f"[batch, sequence, {self.config.hidden_size}].")

        if audio_input_ids is not None:
            audio_embeds = self.embed_audio_tokens(audio_input_ids)
            if audio_input_ids_mask is not None:
                if (not isinstance(audio_input_ids_mask, Tensor) or
                        tuple(audio_input_ids_mask.shape) != tuple(audio_input_ids.shape[:2])):
                    raise ValueError("`audio_input_ids_mask` must match audio batch/frames.")
                valid_audio = audio_input_ids_mask.to(
                    device=audio_embeds.device,
                    dtype=torch.bool,
                )
            else:
                valid_audio = torch.ones(
                    audio_input_ids.shape[:2],
                    device=audio_embeds.device,
                    dtype=torch.bool,
                )
            if inputs_embeds is None:
                inputs_embeds = audio_embeds
                audio_token_mask = None
            else:
                assert audio_token_mask is not None
                expected = int(audio_token_mask.sum())
                supplied_audio = int(valid_audio.sum())
                if expected != supplied_audio:
                    raise ValueError(
                        "Higgs text audio placeholders and valid delayed "
                        f"frames disagree ({expected} != {supplied_audio}).")
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[audio_token_mask] = (audio_embeds[valid_audio].to(inputs_embeds.dtype))
        elif (audio_token_mask is not None and audio_token_mask.any()):
            raise ValueError("Higgs audio placeholders require `audio_input_ids`.")
        assert inputs_embeds is not None
        return inputs_embeds, audio_token_mask

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        audio_input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        audio_input_ids_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        inputs_embeds: Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> HiggsAudioV2ModelOutput:
        hidden_states, audio_token_mask = self._embeddings(
            input_ids,
            audio_input_ids,
            audio_input_ids_mask,
            inputs_embeds,
        )
        batch_size, query_length, _ = hidden_states.shape
        if (past_key_values is not None and not isinstance(past_key_values, DynamicKVCache)):
            raise TypeError("`past_key_values` must be a DynamicKVCache or None.")
        past_length = (0 if past_key_values is None else past_key_values.sequence_length())
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if not isinstance(use_cache, bool):
            raise TypeError("`use_cache` must be a boolean.")
        if self.gradient_checkpointing and self.training and use_cache:
            raise ValueError(
                "Gradient checkpointing and mutable KV caching are "
                "incompatible during training.")
        if use_cache and past_key_values is None:
            past_key_values = DynamicKVCache()
        position_ids = _normalize_position_ids(
            position_ids,
            attention_mask=attention_mask,
            batch_size=batch_size,
            query_length=query_length,
            past_length=past_length,
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
                    current_layer: HiggsAudioV2DecoderLayer = layer,
                ) -> Tensor:
                    result, _, _ = current_layer(
                        states,
                        attention_mask=attention_mask,
                        audio_token_mask=audio_token_mask,
                        position_ids=position_ids,
                        cache=None,
                        use_cache=False,
                        output_attentions=False,
                    )
                    return result

                hidden_states = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    use_reentrant=False,
                )
                attention = None
            else:
                hidden_states, attention, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    audio_token_mask=audio_token_mask,
                    position_ids=position_ids,
                    cache=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            if attention_history is not None:
                if attention is None:
                    raise RuntimeError("A Higgs layer omitted requested attention weights.")
                attention_history.append(attention)
        hidden_states = self.norm(hidden_states)
        if hidden_history is not None:
            hidden_history.append(hidden_states)
        return HiggsAudioV2ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=(tuple(hidden_history) if hidden_history is not None else None),
            attentions=(tuple(attention_history) if attention_history is not None else None),
        )


class HiggsAudioV2ForConditionalGeneration(nn.Module):
    """Joint delayed-audio and optional text causal language model."""

    def __init__(
        self,
        config: HiggsAudioV2Config | dict[str, Any],
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = HiggsAudioV2Config.coerce(config)
        factory = {"device": device, "dtype": dtype}
        self.model = HiggsAudioV2Model(
            self.config,
            initialize=initialize,
            **factory,
        )
        self.audio_lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.audio_vocabulary_size,
            bias=False,
            **factory,
        )
        self.text_lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
            **factory,
        )
        if initialize:
            nn.init.normal_(
                self.audio_lm_head.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            nn.init.normal_(
                self.text_lm_head.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.model.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        attention_mask: Tensor | None = None,
        audio_input_ids: Tensor | None = None,
        audio_input_ids_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        audio_labels: Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> HiggsAudioV2Output:
        if labels is not None or audio_labels is not None:
            if past_key_values is not None:
                raise ValueError("Higgs losses require a complete uncached sequence.")
            if use_cache is None:
                use_cache = False
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            audio_input_ids=audio_input_ids,
            audio_input_ids_mask=audio_input_ids_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs.last_hidden_state
        audio_logits = self.audio_lm_head(hidden_states)
        text_logits = self.text_lm_head(hidden_states)
        text_loss = None
        audio_loss = None
        codebook_losses: tuple[Tensor, ...] = ()
        if labels is not None:
            if (not isinstance(labels, Tensor) or tuple(labels.shape) != tuple(text_logits.shape[:2])):
                raise ValueError("Higgs text labels must match [batch, sequence].")
            if labels.shape[1] < 2:
                raise ValueError("Higgs causal losses require at least two frames.")
            text_mask = (None if attention_mask is None else attention_mask[:, 1:])
            text_loss = sequence_cross_entropy(
                text_logits[:, :-1],
                labels[:, 1:],
                attention_mask=text_mask,
            )
        if audio_labels is not None:
            if input_ids is None or audio_input_ids is None:
                raise ValueError("Higgs audio labels require text placeholders and "
                                 "delayed audio inputs.")
            if (not isinstance(audio_labels, Tensor) or
                    tuple(audio_labels.shape) != tuple(audio_input_ids.shape)):
                raise ValueError("`audio_labels` must match delayed audio input IDs.")
            audio_mask = (
                torch.ones(
                    audio_input_ids.shape[:2],
                    device=audio_input_ids.device,
                    dtype=torch.bool,
                ) if audio_input_ids_mask is None else audio_input_ids_mask.to(dtype=torch.bool))
            placeholders = ((input_ids == self.config.audio_token_id)
                            | (input_ids == self.config.audio_delay_token_id))
            expanded = torch.full(
                (
                    input_ids.shape[0],
                    input_ids.shape[1],
                    self.config.num_codebooks,
                ),
                -100,
                device=input_ids.device,
                dtype=torch.long,
            )
            if int(placeholders.sum()) != int(audio_mask.sum()):
                raise ValueError("Higgs audio labels do not align with text placeholders.")
            expanded[placeholders] = audio_labels[audio_mask].long()
            structured = audio_logits.view(
                *audio_logits.shape[:2],
                self.config.num_codebooks,
                self.config.codebook_size,
            )
            losses = []
            for index in range(self.config.num_codebooks):
                losses.append(sequence_cross_entropy(
                    structured[:, :-1, index],
                    expanded[:, 1:, index],
                ))
            codebook_losses = tuple(losses)
            audio_loss = torch.stack(losses).sum()
        loss = None
        if text_loss is not None:
            loss = text_loss
        if audio_loss is not None:
            loss = audio_loss if loss is None else loss + audio_loss
        return HiggsAudioV2Output(
            logits=audio_logits,
            text_logits=text_logits,
            loss=loss,
            audio_loss=audio_loss,
            text_loss=text_loss,
            codebook_losses=codebook_losses,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "HiggsAudioV2Config",
    "HiggsAudioV2DecoderLayer",
    "HiggsAudioV2Embeddings",
    "HiggsAudioV2ForConditionalGeneration",
    "HiggsAudioV2Model",
    "HiggsAudioV2ModelOutput",
    "HiggsAudioV2Output",
]
