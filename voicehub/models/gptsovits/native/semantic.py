"""Native GPT-SoVITS V1/V2-family S1 autoregressive semantic model."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.gptsovits.native.configuration import GPTSoVITSS1Config


def make_pad_mask(lengths: Tensor, max_length: int | None = None, *, left: bool = False) -> Tensor:
    """Return the exact upstream boolean padding convention."""
    if lengths.ndim != 1:
        raise ValueError("Lengths must be one-dimensional.")
    maximum = int(lengths.max().item()) if max_length is None else max_length
    positions = torch.arange(maximum, device=lengths.device).unsqueeze(0)
    if left:
        return positions < (maximum - lengths).unsqueeze(1)
    return positions >= lengths.unsqueeze(1)


class TokenEmbedding(nn.Module):

    def __init__(self, embedding_dim: int, vocabulary_size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_embeddings = nn.Embedding(vocabulary_size, embedding_dim)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.dropout(self.word_embeddings(input_ids))


class SinePositionalEmbedding(nn.Module):

    def __init__(self, embedding_dim: int, *, dropout: float, alpha: bool) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.x_scale = 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("_position_cache", torch.empty(0), persistent=False)

    def _positions(self, length: int, reference: Tensor) -> Tensor:
        cache = self._position_cache
        if (cache.ndim != 3 or cache.shape[1] < length or cache.device != reference.device or
                cache.dtype != reference.dtype):
            position = torch.arange(length, dtype=torch.float32, device=reference.device).unsqueeze(1)
            divisor = torch.exp(
                torch.arange(
                    0,
                    self.embedding_dim,
                    2,
                    dtype=torch.float32,
                    device=reference.device,
                ) * -(math.log(10_000.0) / self.embedding_dim))
            encoded = torch.zeros(
                length,
                self.embedding_dim,
                dtype=torch.float32,
                device=reference.device,
            )
            encoded[:, 0::2] = torch.sin(position * divisor)
            encoded[:, 1::2] = torch.cos(position * divisor)
            cache = encoded.unsqueeze(0).to(dtype=reference.dtype)
            self._position_cache = cache
        return cache[:, :length]

    def forward(self, hidden_states: Tensor, *, offset: int = 0) -> Tensor:
        length = hidden_states.shape[1]
        positions = self._positions(length + offset, hidden_states)[:, offset:offset + length]
        return self.dropout(hidden_states * self.x_scale + self.alpha * positions)


def _top_k_top_p(
    logits: Tensor,
    *,
    top_k: int,
    top_p: float,
) -> Tensor:
    filtered = logits.clone()
    if top_k > 0:
        threshold = torch.topk(filtered, min(top_k, filtered.shape[-1])).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative = torch.cumsum(functional.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove.scatter(1, sorted_indices, remove)
        filtered.masked_fill_(remove, -torch.inf)
    return filtered


class Text2SemanticDecoder(nn.Module):
    """Checkpoint-exact S1 graph with source-equivalent CE training."""

    def __init__(self, config: GPTSoVITSS1Config) -> None:
        super().__init__()
        self.config = config
        self.bert_proj = nn.Linear(config.bert_feature_dim, config.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(
            config.embedding_dim,
            config.phoneme_vocabulary_size,
            config.dropout,
        )
        self.ar_text_position = SinePositionalEmbedding(
            config.embedding_dim,
            dropout=0.1,
            alpha=True,
        )
        self.ar_audio_embedding = TokenEmbedding(
            config.embedding_dim,
            config.vocabulary_size,
            config.dropout,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            config.embedding_dim,
            dropout=0.1,
            alpha=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=False,
        )
        self.h = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.ar_predict_layer = nn.Linear(
            config.hidden_dim,
            config.vocabulary_size,
            bias=False,
        )

    def _validate(
        self,
        phoneme_ids: Any,
        phoneme_lengths: Any,
        semantic_ids: Any,
        semantic_lengths: Any,
        bert_features: Any,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        reference = next(self.parameters())
        phoneme_ids = torch.as_tensor(phoneme_ids, device=reference.device, dtype=torch.long)
        phoneme_lengths = torch.as_tensor(
            phoneme_lengths,
            device=reference.device,
            dtype=torch.long,
        )
        semantic_ids = torch.as_tensor(
            semantic_ids,
            device=reference.device,
            dtype=torch.long,
        )
        semantic_lengths = torch.as_tensor(
            semantic_lengths,
            device=reference.device,
            dtype=torch.long,
        )
        bert_features = torch.as_tensor(
            bert_features,
            device=reference.device,
            dtype=reference.dtype,
        )
        if phoneme_ids.ndim != 2 or semantic_ids.ndim != 2:
            raise ValueError("S1 phoneme and semantic IDs must have shape [batch, time].")
        batch, phoneme_steps = phoneme_ids.shape
        if semantic_ids.shape[0] != batch:
            raise ValueError("S1 phoneme and semantic batch sizes differ.")
        if tuple(phoneme_lengths.shape) != (batch, ) or tuple(semantic_lengths.shape) != (batch, ):
            raise ValueError("S1 length tensors must have shape [batch].")
        if bert_features.shape != (
                batch,
                self.config.bert_feature_dim,
                phoneme_steps,
        ):
            raise ValueError(
                "S1 BERT features must have shape "
                f"[batch, {self.config.bert_feature_dim}, phoneme_time].")
        if bool(((phoneme_ids < 0) | (phoneme_ids >= self.config.phoneme_vocabulary_size)).any()):
            raise ValueError(f"S1 phoneme IDs are outside the {self.config.version} vocabulary.")
        if bool(((semantic_ids < 0) | (semantic_ids >= self.config.eos_token_id)).any()):
            raise ValueError("S1 training semantic IDs must exclude EOS and padding IDs.")
        if bool(((phoneme_lengths < 1) | (phoneme_lengths > phoneme_steps)).any()):
            raise ValueError("S1 phoneme lengths are invalid.")
        if bool(((semantic_lengths < 1) | (semantic_lengths > semantic_ids.shape[1])).any()):
            raise ValueError("S1 semantic lengths are invalid.")
        return (
            phoneme_ids,
            phoneme_lengths,
            semantic_ids,
            semantic_lengths,
            bert_features,
        )

    def _text_hidden(
        self,
        phoneme_ids: Tensor,
        bert_features: Tensor,
    ) -> Tensor:
        hidden = self.ar_text_embedding(phoneme_ids)
        hidden = hidden + self.bert_proj(bert_features.transpose(1, 2))
        return self.ar_text_position(hidden)

    def _training_inputs(
        self,
        phoneme_ids: Tensor,
        phoneme_lengths: Tensor,
        semantic_ids: Tensor,
        semantic_lengths: Tensor,
        bert_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        text = self._text_hidden(phoneme_ids, bert_features)
        text_mask = make_pad_mask(phoneme_lengths, left=True)
        semantic_mask = make_pad_mask(semantic_lengths)
        semantic_mask_int = semantic_mask.to(torch.long)
        codes = semantic_ids * (1 - semantic_mask_int)
        targets = functional.pad(codes, (0, 1), value=0)
        targets = targets + self.config.eos_token_id * functional.pad(
            semantic_mask_int,
            (0, 1),
            value=1,
        )
        semantic_input = targets[:, :-1]
        semantic_hidden = self.ar_audio_position(self.ar_audio_embedding(semantic_input))
        text_steps = int(phoneme_lengths.max().item())
        semantic_steps = int(semantic_lengths.max().item())
        text_attention = functional.pad(
            torch.zeros(
                text_steps,
                text_steps,
                dtype=torch.bool,
                device=text.device,
            ),
            (0, semantic_steps),
            value=True,
        )
        semantic_attention = functional.pad(
            torch.triu(
                torch.ones(
                    semantic_steps,
                    semantic_steps,
                    dtype=torch.bool,
                    device=text.device,
                ),
                diagonal=1,
            ),
            (text_steps, 0),
            value=False,
        )
        attention = torch.cat([text_attention, semantic_attention])
        padding = torch.cat([text_mask, semantic_mask], dim=1)
        source_steps = text_steps + semantic_steps
        padding = padding.view(text.shape[0], 1, 1, source_steps)
        padding = padding.expand(-1, self.config.attention_heads, -1, -1)
        padding = padding.reshape(text.shape[0] * self.config.attention_heads, 1, source_steps)
        attention = attention.logical_or(padding)
        float_attention = torch.zeros_like(attention, dtype=text.dtype)
        float_attention.masked_fill_(attention, -torch.inf)
        return (
            torch.cat([text, semantic_hidden], dim=1),
            float_attention,
            targets,
            text_steps,
        )

    def training_objective(
        self,
        *,
        phoneme_ids: Any,
        phoneme_lengths: Any,
        semantic_ids: Any,
        semantic_lengths: Any,
        bert_features: Any,
    ) -> dict[str, Tensor]:
        values = self._validate(
            phoneme_ids,
            phoneme_lengths,
            semantic_ids,
            semantic_lengths,
            bert_features,
        )
        hidden, attention, targets, text_steps = self._training_inputs(*values)
        decoded = self.h(hidden, mask=attention)
        logits = self.ar_predict_layer(decoded[:, text_steps - 1:]).transpose(1, 2)
        loss = functional.cross_entropy(logits, targets, reduction="sum")
        valid = targets != self.config.eos_token_id
        top_k = min(3, logits.shape[1])
        predictions = torch.topk(logits.detach(), top_k, dim=1).indices
        correct = (predictions == targets.unsqueeze(1)).any(dim=1) & valid
        denominator = valid.sum().clamp_min(1)
        accuracy = correct.sum().to(logits.dtype) / denominator
        return {
            "loss": loss,
            "semantic_loss": loss,
            "top_3_accuracy": accuracy,
            "logits": logits,
        }

    @torch.no_grad()
    def generate(
        self,
        *,
        phoneme_ids: Any,
        phoneme_lengths: Any,
        bert_features: Any,
        prompt_semantic_ids: Any | None,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        maximum_new_tokens: int | None = None,
    ) -> Tensor:
        reference = next(self.parameters())
        phoneme_ids = torch.as_tensor(
            phoneme_ids,
            device=reference.device,
            dtype=torch.long,
        )
        phoneme_lengths = torch.as_tensor(
            phoneme_lengths,
            device=reference.device,
            dtype=torch.long,
        )
        bert_features = torch.as_tensor(
            bert_features,
            device=reference.device,
            dtype=reference.dtype,
        )
        if phoneme_ids.ndim != 2 or phoneme_ids.shape[0] != 1:
            raise ValueError("Native S1 generation currently supports one prepared item.")
        if bert_features.shape != (
                1,
                self.config.bert_feature_dim,
                phoneme_ids.shape[1],
        ):
            raise ValueError("Prepared S1 BERT feature shape does not match phoneme IDs.")
        if prompt_semantic_ids is None:
            generated = torch.empty(
                1,
                0,
                device=reference.device,
                dtype=torch.long,
            )
        else:
            generated = torch.as_tensor(
                prompt_semantic_ids,
                device=reference.device,
                dtype=torch.long,
            )
            if generated.ndim == 1:
                generated = generated.unsqueeze(0)
            if generated.ndim != 2 or generated.shape[0] != 1:
                raise ValueError("S1 prompt semantic IDs must have shape [1, time].")
        prefix_length = generated.shape[1]
        text = self._text_hidden(phoneme_ids, bert_features)
        limit = maximum_new_tokens or self.config.maximum_generated_tokens
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("`maximum_new_tokens` must be a positive integer.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("`top_k` must be a positive integer.")
        if not 0 < top_p <= 1:
            raise ValueError("`top_p` must be in (0, 1].")
        if temperature <= 0 or repetition_penalty <= 0:
            raise ValueError("Temperature and repetition penalty must be positive.")
        for step in range(limit):
            semantic_hidden = self.ar_audio_position(self.ar_audio_embedding(generated), )
            hidden = torch.cat([text, semantic_hidden], dim=1)
            text_steps = text.shape[1]
            semantic_steps = generated.shape[1]
            text_attention = functional.pad(
                torch.zeros(
                    text_steps,
                    text_steps,
                    dtype=torch.bool,
                    device=text.device,
                ),
                (0, semantic_steps),
                value=True,
            )
            semantic_attention = functional.pad(
                torch.triu(
                    torch.ones(
                        semantic_steps,
                        semantic_steps,
                        dtype=torch.bool,
                        device=text.device,
                    ),
                    diagonal=1,
                ),
                (text_steps, 0),
            )
            attention = torch.cat([text_attention, semantic_attention])
            decoded = self.h(hidden, mask=attention)
            logits = self.ar_predict_layer(decoded[:, -1])
            if step < 11:
                logits = logits[:, :-1]
            if generated.numel() and repetition_penalty != 1:
                previous = generated.unique()
                selected = logits[:, previous]
                selected = torch.where(
                    selected < 0,
                    selected * repetition_penalty,
                    selected / repetition_penalty,
                )
                logits[:, previous] = selected
            filtered = _top_k_top_p(
                logits / temperature,
                top_k=top_k,
                top_p=top_p,
            )
            sample = torch.multinomial(functional.softmax(filtered, dim=-1), 1)
            greedy = logits.argmax(dim=-1, keepdim=True)
            if (sample == self.config.eos_token_id).any() or (greedy == self.config.eos_token_id).any():
                break
            generated = torch.cat([generated, sample], dim=1)
        result = generated[:, prefix_length:]
        if result.shape[1] == 0:
            raise RuntimeError("GPT-SoVITS S1 generated no semantic tokens.")
        return result


class GPTSoVITSSemanticModel(nn.Module):
    """Wrapper retaining the released ``model.*`` checkpoint namespace."""

    def __init__(self, config: GPTSoVITSS1Config | None = None) -> None:
        super().__init__()
        self.config = config or GPTSoVITSS1Config()
        self.model = Text2SemanticDecoder(self.config)

    def forward(self, **batch: Any) -> dict[str, Tensor]:
        return self.model.training_objective(**batch)

    def generate(self, **inputs: Any) -> Tensor:
        return self.model.generate(**inputs)


__all__ = [
    "GPTSoVITSSemanticModel",
    "SinePositionalEmbedding",
    "Text2SemanticDecoder",
    "TokenEmbedding",
    "make_pad_mask",
]
