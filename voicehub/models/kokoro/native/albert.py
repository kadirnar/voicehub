"""VoiceHub-native ALBERT graph used by Kokoro's PL-BERT encoder.

The tensor namespace and eager forward semantics match
``transformers==4.48.3`` for the subset exercised by the released Kokoro
checkpoint. No Transformers runtime is imported.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from voicehub.models.kokoro.native.configuration import KokoroAlbertConfig


def _gelu_new(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * value * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * torch.pow(value, 3.0))))


class AlbertEmbeddings(nn.Module):
    """Word, position, and token-type embeddings with HF-compatible names."""

    def __init__(self, config: KokoroAlbertConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.embedding_size,
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.embedding_size,
        )
        # Capitalization is checkpoint compatibility, not style.
        self.LayerNorm = nn.LayerNorm(
            config.embedding_size,
            eps=config.layer_norm_eps,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(
                (1, config.max_position_embeddings),
                dtype=torch.long,
            ),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` and `inputs_embeds`.")
        input_shape = (input_ids.shape if input_ids is not None else inputs_embeds.shape[:-1])
        sequence_length = input_shape[1]
        if sequence_length > self.position_ids.shape[1]:
            raise ValueError("Kokoro PL-BERT input exceeds its position embedding limit.")
        if position_ids is None:
            position_ids = self.position_ids[:, :sequence_length]
        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :sequence_length].expand(
                input_shape[0],
                sequence_length,
            )
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds + self.token_type_embeddings(token_type_ids)
        embeddings = embeddings + self.position_embeddings(position_ids)
        return self.dropout(self.LayerNorm(embeddings))


class AlbertAttention(nn.Module):
    """PyTorch SDPA attention selected by ALBERT 4.48.3 on Torch 2.8."""

    def __init__(self, config: KokoroAlbertConfig) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = (config.hidden_size // config.num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.attention_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def _transpose_for_scores(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return value.view(shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self._transpose_for_scores(self.query(hidden_states))
        key = self._transpose_for_scores(self.key(hidden_states))
        value = self._transpose_for_scores(self.value(hidden_states))
        context = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            dropout_p=(self.attention_dropout.p if self.training else 0.0),
            is_causal=False,
        )
        context = context.transpose(2, 1).flatten(2)
        projected = self.output_dropout(self.dense(context))
        return self.LayerNorm(hidden_states + projected)


class AlbertLayer(nn.Module):
    """One shared ALBERT attention/feed-forward layer."""

    def __init__(self, config: KokoroAlbertConfig) -> None:
        super().__init__()
        self.full_layer_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.attention(
            hidden_states,
            attention_mask,
        )
        feed_forward = self.ffn_output(_gelu_new(self.ffn(attention_output)))
        return self.full_layer_layer_norm(feed_forward + attention_output)


class AlbertLayerGroup(nn.Module):
    """Checkpoint-compatible ALBERT inner-layer group."""

    def __init__(self, config: KokoroAlbertConfig) -> None:
        super().__init__()
        self.albert_layers = nn.ModuleList(AlbertLayer(config) for _ in range(config.inner_group_num))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.albert_layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class AlbertTransformer(nn.Module):
    """Factorized ALBERT encoder with source-compatible parameter sharing."""

    def __init__(self, config: KokoroAlbertConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(
            config.embedding_size,
            config.hidden_size,
        )
        self.albert_layer_groups = nn.ModuleList(
            AlbertLayerGroup(config) for _ in range(config.num_hidden_groups))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        for layer_index in range(self.config.num_hidden_layers):
            group_index = int(layer_index / (self.config.num_hidden_layers / self.config.num_hidden_groups))
            hidden_states = self.albert_layer_groups[group_index](
                hidden_states,
                attention_mask,
            )
        return hidden_states


class KokoroAlbertModel(nn.Module):
    """Bare ALBERT model returning only its final hidden state."""

    def __init__(
        self,
        config: KokoroAlbertConfig | dict[str, Any],
    ) -> None:
        super().__init__()
        if not isinstance(config, KokoroAlbertConfig):
            config = KokoroAlbertConfig.from_dict(
                config,
                vocab_size=int(config.get("vocab_size", 178)),
            )
        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        # The released checkpoint contains the unused standard ALBERT pooler.
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self._initialize_weights()

    @property
    def device(self) -> torch.device:
        return self.embeddings.word_embeddings.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.word_embeddings.weight.dtype

    def _initialize_weights(self) -> None:
        config = self.config
        for module in self.modules():
            if isinstance(module, nn.Linear):
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
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported Kokoro ALBERT argument(s): {names}.")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` and `inputs_embeds`.")
        input_shape = (input_ids.shape if input_ids is not None else inputs_embeds.shape[:-1])
        device = (input_ids.device if input_ids is not None else inputs_embeds.device)
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if attention_mask.shape != input_shape:
            raise ValueError("Kokoro ALBERT `attention_mask` must match `input_ids`.")
        embeddings = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        extended_mask = attention_mask[:, None, None, :].to(dtype=embeddings.dtype)
        extended_mask = (1.0 - extended_mask) * torch.finfo(embeddings.dtype).min
        return self.encoder(embeddings, extended_mask)


__all__ = [
    "AlbertAttention",
    "AlbertEmbeddings",
    "AlbertLayer",
    "AlbertLayerGroup",
    "AlbertTransformer",
    "KokoroAlbertModel",
]
