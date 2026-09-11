"""Exact PyTorch graph for Parler-TTS Mini v1 inference and fine-tuning."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicehub.generation import GenerationConfig, create_generator, sample_next_token
from voicehub.models.dac.native.modeling import DacModel
from voicehub.models.parlertts.native.configuration import ParlerDecoderConfig, ParlerTTSArchitectureConfig
from voicehub.models.parlertts.native.t5 import NativeT5EncoderModel
from voicehub.optimization.protocols import OptimizationCompileTarget


def apply_delay_pattern_mask(
    input_ids: Tensor,
    delay_pattern_mask: Tensor,
) -> Tensor:
    """Apply fixed BOS/EOS positions to delayed parallel codebook tokens."""
    sequence_length = input_ids.shape[-1]
    pattern = delay_pattern_mask[..., :sequence_length]
    return torch.where(pattern == -1, input_ids, pattern)


def build_delay_pattern_mask(
    input_ids: Tensor,
    *,
    bos_token_id: int,
    pad_token_id: int,
    max_length: int,
    num_codebooks: int,
) -> tuple[Tensor, Tensor]:
    """Build Parler-TTS's one-step-per-codebook autoregressive offset."""
    if not isinstance(input_ids, Tensor) or input_ids.ndim not in {2, 3}:
        raise ValueError(
            "`input_ids` must have shape [batch * codebooks, time] or "
            "[batch, codebooks, time].")
    if input_ids.ndim == 2:
        if input_ids.shape[0] % num_codebooks:
            raise ValueError("Decoder batch is not divisible by codebook count.")
        shaped = input_ids.reshape(
            -1,
            num_codebooks,
            input_ids.shape[-1],
        )
    else:
        shaped = input_ids
        if shaped.shape[1] != num_codebooks:
            raise ValueError("Decoder input has the wrong codebook count.")
    batch_size, _, sequence_length = shaped.shape
    shifted = torch.full(
        (batch_size, num_codebooks, max_length),
        -1,
        dtype=torch.long,
        device=input_ids.device,
    )
    if max_length < 2 * num_codebooks - 1:
        return (
            shaped.reshape(batch_size * num_codebooks, -1),
            shifted.reshape(batch_size * num_codebooks, -1),
        )
    for codebook in range(num_codebooks):
        stop = min(sequence_length + codebook, max_length)
        amount = stop - codebook
        if amount > 0:
            shifted[:, codebook, codebook:stop] = shaped[
                :,
                codebook,
                :amount,
            ]
    eos_pattern = torch.triu(
        torch.ones(
            num_codebooks,
            max_length,
            dtype=torch.bool,
            device=input_ids.device,
        ),
        diagonal=max_length - num_codebooks + 1,
    )
    bos_pattern = torch.tril(
        torch.ones(
            num_codebooks,
            max_length,
            dtype=torch.bool,
            device=input_ids.device,
        ))
    valid = ~(bos_pattern | eos_pattern)
    pattern = (valid * shifted + bos_pattern * bos_token_id + eos_pattern * pad_token_id)
    first_codebook = pattern[:, 0]
    start_positions = (first_codebook == -1).nonzero(as_tuple=False)
    first_start = (int(start_positions[:, 1].min()) if start_positions.numel() else sequence_length)
    flattened = pattern.reshape(batch_size * num_codebooks, -1)
    return flattened[..., :first_start], flattened


def shift_tokens_right(
    labels: Tensor,
    *,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> Tensor:
    """Shift time-major training labels using Parler's BOS token."""
    if not isinstance(labels, Tensor) or labels.ndim != 3:
        raise ValueError("Parler labels must have shape [batch, time, codebooks].")
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


def prepare_audio_code_labels(
    audio_codes: Tensor,
    *,
    bos_token_id: int,
    eos_token_id: int,
    audio_code_lengths: Tensor | Sequence[int] | None = None,
) -> Tensor:
    """Convert raw DAC codes to upstream delayed teacher-forcing labels."""
    if not isinstance(audio_codes, Tensor) or audio_codes.ndim != 3:
        raise ValueError("`audio_codes` must have shape [batch, codebooks, frames].")
    if (audio_codes.dtype == torch.bool or audio_codes.is_floating_point() or audio_codes.is_complex()):
        raise TypeError("`audio_codes` must use an integer dtype.")
    batch_size, num_codebooks, frames = audio_codes.shape
    if audio_code_lengths is not None:
        lengths = torch.as_tensor(audio_code_lengths)
        if lengths.ndim != 1 or lengths.shape[0] != batch_size:
            raise ValueError("`audio_code_lengths` must have one value per audio row.")
        if (lengths.dtype == torch.bool or lengths.is_floating_point() or lengths.is_complex()):
            raise TypeError("`audio_code_lengths` must use an integer dtype.")
        if (lengths < 1).any() or (lengths > frames).any():
            raise ValueError("`audio_code_lengths` values must be within encoded frames.")
        rows = tuple(
            prepare_audio_code_labels(
                audio_codes[index:index + 1, :, :int(length)],
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
            ).squeeze(0) for index, length in enumerate(lengths.tolist()))
        return nn.utils.rnn.pad_sequence(
            rows,
            batch_first=True,
            padding_value=-100,
        )
    bos = torch.full(
        (batch_size, num_codebooks, 1),
        bos_token_id,
        dtype=torch.long,
        device=audio_codes.device,
    )
    prefixed = torch.cat((bos, audio_codes.long()), dim=-1)
    _, pattern = build_delay_pattern_mask(
        prefixed,
        bos_token_id=bos_token_id,
        pad_token_id=eos_token_id,
        max_length=frames + 1 + num_codebooks,
        num_codebooks=num_codebooks,
    )
    delayed = torch.where(pattern == -1, eos_token_id, pattern)
    delayed = delayed.reshape(
        batch_size,
        num_codebooks,
        -1,
    ).transpose(1, 2)
    # Timestamp zero is all BOS. It is the decoder input, not a prediction.
    return delayed[:, 1:].contiguous()


class ParlerSinusoidalPositionalEmbedding(nn.Module):
    """Checkpoint-compatible, fixed tensor2tensor sinusoidal positions."""

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weights = nn.Parameter(
            self.get_embedding(num_positions, embedding_dim),
            requires_grad=False,
        )

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int) -> Tensor:
        half_dimension = embedding_dim // 2
        scale = math.log(10_000) / (half_dimension - 1)
        frequencies = torch.exp(torch.arange(half_dimension, dtype=torch.float32) * -scale)
        positions = torch.arange(num_embeddings, dtype=torch.float32)[:, None]
        values = positions * frequencies[None]
        embedding = torch.cat((torch.cos(values), torch.sin(values)), dim=1)
        if embedding_dim % 2:
            embedding = torch.cat(
                (embedding, torch.zeros(num_embeddings, 1)),
                dim=1,
            )
        return embedding.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, hidden_states: Tensor, offset: int = 0) -> Tensor:
        sequence_length = hidden_states.shape[1]
        stop = offset + sequence_length
        if stop > self.weights.shape[0]:
            raise ValueError(
                f"Decoder length {stop} exceeds the configured positional "
                f"limit {self.weights.shape[0]}.")
        return self.weights[offset:stop].detach()


def _repeat_key_values(hidden_states: Tensor, repeats: int) -> Tensor:
    if repeats == 1:
        return hidden_states
    batch, heads, sequence, dimension = hidden_states.shape
    expanded = hidden_states[:, :, None].expand(
        batch,
        heads,
        repeats,
        sequence,
        dimension,
    )
    return expanded.reshape(batch, heads * repeats, sequence, dimension)


class ParlerAttention(nn.Module):
    """Parler MHA/GQA with eager and native PyTorch SDPA execution."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        *,
        dropout: float,
        attention_implementation: str = "eager",
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("Attention hidden size must be divisible by heads.")
        if num_heads % num_key_value_heads:
            raise ValueError("Attention heads must be divisible by KV heads.")
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.scaling = self.head_dim**-0.5
        if attention_implementation not in {"eager", "sdpa"}:
            raise ValueError("Parler attention implementation must be 'eager' or 'sdpa'.")
        self.attention_implementation = attention_implementation
        self.k_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _shape_query(self, value: Tensor) -> Tensor:
        return value.view(
            value.shape[0],
            value.shape[1],
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _shape_key_value(self, value: Tensor) -> Tensor:
        return value.view(
            value.shape[0],
            value.shape[1],
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        key_value_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        query = self._shape_query(self.q_proj(hidden_states))
        source = hidden_states if key_value_states is None else key_value_states
        key = _repeat_key_values(
            self._shape_key_value(self.k_proj(source)),
            self.num_key_value_groups,
        )
        value = _repeat_key_values(
            self._shape_key_value(self.v_proj(source)),
            self.num_key_value_groups,
        )
        if (self.attention_implementation == "sdpa" and not output_attentions):
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            attended = attended.transpose(1, 2).reshape(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.embed_dim,
            )
            return self.out_proj(attended), None
        query = query * self.scaling
        weights = torch.matmul(query, key.transpose(2, 3))
        if attention_mask is not None:
            weights = weights + attention_mask[..., :key.shape[-2]]
        weights = F.softmax(weights, dim=-1)
        probabilities = F.dropout(
            weights,
            p=self.dropout,
            training=self.training,
        )
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.embed_dim,
        )
        return self.out_proj(attended), weights if output_attentions else None


def _activation(name: str, value: Tensor) -> Tensor:
    if name == "gelu":
        return F.gelu(value)
    if name == "gelu_new":
        coefficient = math.sqrt(2.0 / math.pi)
        return 0.5 * value * (1.0 + torch.tanh(coefficient * (value + 0.044715 * value.pow(3))))
    if name == "relu":
        return F.relu(value)
    if name == "silu":
        return F.silu(value)
    raise ValueError(f"Unsupported activation {name!r}.")


class ParlerDecoderLayer(nn.Module):

    def __init__(
        self,
        config: ParlerDecoderConfig,
        *,
        attention_implementation: str = "eager",
    ) -> None:
        super().__init__()
        self.config = config
        self.self_attn = ParlerAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            dropout=config.attention_dropout,
            attention_implementation=attention_implementation,
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.encoder_attn = ParlerAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_cross_attention_key_value_heads,
            dropout=config.attention_dropout,
            attention_implementation=attention_implementation,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.fc1 = nn.Linear(
            config.hidden_size,
            config.ffn_dim,
            bias=False,
        )
        self.fc2 = nn.Linear(
            config.ffn_dim,
            config.hidden_size,
            bias=False,
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor,
        encoder_hidden_states: Tensor | None,
        encoder_attention_mask: Tensor | None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        residual = hidden_states
        attended, self_weights = self.self_attn(
            self.self_attn_layer_norm(hidden_states),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + F.dropout(
            attended,
            p=self.config.dropout,
            training=self.training,
        )
        cross_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            attended, cross_weights = self.encoder_attn(
                self.encoder_attn_layer_norm(hidden_states),
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = residual + F.dropout(
                attended,
                p=self.config.dropout,
                training=self.training,
            )
        residual = hidden_states
        forwarded = _activation(
            self.config.activation_function,
            self.fc1(self.final_layer_norm(hidden_states)),
        )
        forwarded = F.dropout(
            forwarded,
            p=self.config.activation_dropout,
            training=self.training,
        )
        forwarded = self.fc2(forwarded)
        hidden_states = residual + F.dropout(
            forwarded,
            p=self.config.dropout,
            training=self.training,
        )
        return hidden_states, self_weights, cross_weights


@dataclass(frozen=True, slots=True)
class ParlerDecoderOutput:
    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] = ()
    attentions: tuple[Tensor, ...] = ()
    cross_attentions: tuple[Tensor, ...] = ()

    def __getitem__(self, index: int) -> Any:
        return (
            self.last_hidden_state,
            self.hidden_states,
            self.attentions,
            self.cross_attentions,
        )[index]


class ParlerDecoder(nn.Module):
    """Transformer decoder retaining the released checkpoint namespace."""

    def __init__(
        self,
        config: ParlerDecoderConfig,
        *,
        attention_implementation: str = "eager",
    ) -> None:
        super().__init__()
        if config.rope_embeddings:
            raise ValueError(
                "The pinned Mini v1 graph uses sinusoidal positions; RoPE "
                "checkpoints have not been parity-validated.")
        self.config = config
        self.num_codebooks = config.num_codebooks
        self.embed_scale = (math.sqrt(config.hidden_size) if config.scale_embedding else 1.0)
        self.embed_tokens = nn.ModuleList(
            nn.Embedding(config.vocab_size + 1, config.hidden_size) for _ in range(config.num_codebooks))
        self.embed_positions = ParlerSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            ParlerDecoderLayer(
                config,
                attention_implementation=attention_implementation,
            ) for _ in range(config.num_hidden_layers))
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    @staticmethod
    def _causal_mask(hidden_states: Tensor) -> Tensor:
        sequence = hidden_states.shape[1]
        minimum = torch.finfo(hidden_states.dtype).min
        mask = torch.full(
            (sequence, sequence),
            minimum,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return torch.triu(mask, diagonal=1)[None, None]

    @staticmethod
    def _encoder_mask(
        attention_mask: Tensor | None,
        hidden_states: Tensor,
        source_length: int,
    ) -> Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.ndim != 2 or attention_mask.shape[1] != source_length:
            raise ValueError("Encoder attention mask has an invalid shape.")
        minimum = torch.finfo(hidden_states.dtype).min
        return (~attention_mask.to(torch.bool))[:, None, None].to(hidden_states.dtype) * minimum

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        prompt_hidden_states: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> ParlerDecoderOutput:
        if not isinstance(input_ids, Tensor) or input_ids.ndim not in {2, 3}:
            raise ValueError(
                "Decoder IDs must have shape [batch * codebooks, time] or "
                "[batch, codebooks, time].")
        shaped = (
            input_ids.reshape(
                -1,
                self.num_codebooks,
                input_ids.shape[-1],
            ) if input_ids.ndim == 2 else input_ids)
        if shaped.shape[1] != self.num_codebooks:
            raise ValueError("Decoder IDs have the wrong codebook count.")
        embeddings = [self.embed_tokens[index](shaped[:, index]) for index in range(self.num_codebooks)]
        hidden_states = sum(embeddings) * self.embed_scale
        if prompt_hidden_states is not None:
            if prompt_hidden_states.shape[0] != hidden_states.shape[0]:
                raise ValueError("Prompt and decoder batches must match.")
            hidden_states = torch.cat((prompt_hidden_states, hidden_states), dim=1)
            if prompt_attention_mask is not None:
                generated_mask = (
                    attention_mask if attention_mask is not None else torch.ones(
                        shaped.shape[0],
                        shaped.shape[-1],
                        dtype=prompt_attention_mask.dtype,
                        device=prompt_attention_mask.device,
                    ))
                attention_mask = torch.cat(
                    (prompt_attention_mask, generated_mask),
                    dim=1,
                )
        hidden_states = hidden_states + self.embed_positions(hidden_states).to(
            device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = F.dropout(
            hidden_states,
            p=self.config.dropout,
            training=self.training,
        )
        causal_mask = self._causal_mask(hidden_states)
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError("Decoder attention mask has an invalid shape.")
            minimum = torch.finfo(hidden_states.dtype).min
            causal_mask = causal_mask + (~attention_mask.to(torch.bool))[:, None, None].to(
                hidden_states.dtype) * minimum
        cross_mask = (
            None if encoder_hidden_states is None else self._encoder_mask(
                encoder_attention_mask,
                hidden_states,
                encoder_hidden_states.shape[1],
            ))
        all_hidden: list[Tensor] = []
        all_self: list[Tensor] = []
        all_cross: list[Tensor] = []
        for layer in self.layers:
            if output_hidden_states:
                all_hidden.append(hidden_states)
            hidden_states, self_attention, cross_attention = layer(
                hidden_states,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=cross_mask,
                output_attentions=output_attentions,
            )
            if self_attention is not None:
                all_self.append(self_attention)
            if cross_attention is not None:
                all_cross.append(cross_attention)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden.append(hidden_states)
        return ParlerDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden),
            attentions=tuple(all_self),
            cross_attentions=tuple(all_cross),
        )


class ParlerTTSModel(nn.Module):

    def __init__(
        self,
        config: ParlerDecoderConfig,
        *,
        attention_implementation: str = "eager",
    ) -> None:
        super().__init__()
        self.decoder = ParlerDecoder(
            config,
            attention_implementation=attention_implementation,
        )

    def forward(self, *args: Any, **kwargs: Any) -> ParlerDecoderOutput:
        return self.decoder(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class ParlerCausalLMOutput:
    loss: Tensor | None
    logits: Tensor
    per_codebook_losses: tuple[Tensor, ...] | None
    hidden_states: tuple[Tensor, ...] = ()
    attentions: tuple[Tensor, ...] = ()
    cross_attentions: tuple[Tensor, ...] = ()

    def __getitem__(self, index: int) -> Any:
        values = (
            self.loss,
            self.logits,
            self.per_codebook_losses,
            self.hidden_states,
            self.attentions,
            self.cross_attentions,
        )
        return values[index]


class ParlerTTSForCausalLM(nn.Module):
    """Nine-head delayed codebook language model."""

    def __init__(
        self,
        config: ParlerDecoderConfig,
        *,
        attention_implementation: str = "eager",
    ) -> None:
        super().__init__()
        self.config = config
        self.model = ParlerTTSModel(
            config,
            attention_implementation=attention_implementation,
        )
        self.num_codebooks = config.num_codebooks
        self.vocab_size = config.vocab_size
        self.lm_heads = nn.ModuleList(
            nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_codebooks))

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        prompt_hidden_states: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        loss_reduction: str = "mean",
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> ParlerCausalLMOutput:
        decoded = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            prompt_hidden_states=prompt_hidden_states,
            prompt_attention_mask=prompt_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = decoded.last_hidden_state
        logits = torch.stack(
            tuple(head(hidden_states) for head in self.lm_heads),
            dim=1,
        )
        loss = None
        per_codebook_losses = None
        if labels is not None:
            if labels.ndim != 3 or labels.shape[-1] != self.num_codebooks:
                raise ValueError("Labels must have shape [batch, time, codebooks].")
            label_length = labels.shape[1]
            selected_logits = logits[:, :, -label_length:]
            prepared_labels = labels.masked_fill(
                labels == self.config.bos_token_id,
                -100,
            )
            shaped_input = input_ids.reshape(
                labels.shape[0],
                self.num_codebooks,
                -1,
            ).transpose(1, 2)
            shaped_input = shaped_input[:, -label_length:]
            mask = (shaped_input != self.config.eos_token_id) & (prepared_labels != -100)
            losses: list[Tensor] = []
            for codebook in range(self.num_codebooks):
                codebook_mask = mask[..., codebook].reshape(-1)
                codebook_logits = selected_logits[
                    :,
                    codebook,
                ].reshape(-1, self.vocab_size)
                codebook_labels = prepared_labels[
                    ...,
                    codebook,
                ].reshape(-1)
                if codebook_mask.any():
                    codebook_loss = F.cross_entropy(
                        codebook_logits[codebook_mask],
                        codebook_labels[codebook_mask],
                        reduction=loss_reduction,
                    )
                else:
                    codebook_loss = codebook_logits.sum() * 0.0
                losses.append(codebook_loss)
            per_codebook_losses = tuple(losses)
            if self.config.codebook_weights is None:
                loss = torch.stack(losses).mean()
            else:
                weights = logits.new_tensor(self.config.codebook_weights)
                loss = (torch.stack(losses) * weights).sum() / weights.sum()
        flattened = logits.reshape(-1, *logits.shape[2:])
        return ParlerCausalLMOutput(
            loss=loss,
            logits=flattened,
            per_codebook_losses=per_codebook_losses,
            hidden_states=decoded.hidden_states,
            attentions=decoded.attentions,
            cross_attentions=decoded.cross_attentions,
        )

    def build_delay_pattern_mask(
        self,
        input_ids: Tensor,
        *,
        bos_token_id: int,
        pad_token_id: int,
        max_length: int,
    ) -> tuple[Tensor, Tensor]:
        return build_delay_pattern_mask(
            input_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            max_length=max_length,
            num_codebooks=self.num_codebooks,
        )


class ParlerDacAudioEncoder(nn.Module):
    """Published DAC wrapper with the exact ``audio_encoder.model``
    namespace."""

    def __init__(self, config: ParlerTTSArchitectureConfig) -> None:
        super().__init__()
        self.config = config.audio_encoder
        self.model = DacModel(config.audio_encoder)

    def encode(self, audio_values: Tensor) -> Tensor:
        if audio_values.ndim == 2:
            audio_values = audio_values[:, None]
        if audio_values.ndim != 3:
            raise ValueError("Audio must have shape [batch, channels, samples].")
        preprocessed = self.model.preprocess(
            audio_values,
            self.config.sampling_rate,
        )
        return self.model.encode(preprocessed)[1]

    def decode(self, audio_codes: Tensor) -> Tensor:
        if audio_codes.ndim == 4:
            if audio_codes.shape[0] != 1:
                raise ValueError("Native Parler DAC supports one frame per batch.")
            audio_codes = audio_codes.squeeze(0)
        if audio_codes.ndim != 3:
            raise ValueError("Audio codes must have shape [batch, codebooks, frames].")
        quantized = self.model.quantizer.from_codes(audio_codes)[0]
        return self.model.decode(quantized)


@dataclass(frozen=True, slots=True)
class ParlerTTSOutput:
    loss: Tensor | None
    logits: Tensor
    per_codebook_losses: tuple[Tensor, ...] | None = None
    audio_values: Tensor | None = None
    audio_codes: Tensor | None = None

    def __getitem__(self, index: int) -> Any:
        return (
            self.loss,
            self.logits,
            self.per_codebook_losses,
            self.audio_values,
            self.audio_codes,
        )[index]


class ParlerTTSForConditionalGeneration(nn.Module):
    """Native composite Parler-TTS model with genuine token CE fine-tuning."""

    def __init__(
        self,
        config: ParlerTTSArchitectureConfig,
        *,
        attention_implementation: str = "eager",
    ) -> None:
        super().__init__()
        self.config = config
        self.text_encoder = NativeT5EncoderModel(
            config.text_encoder,
            attention_implementation=attention_implementation,
        )
        self.audio_encoder = ParlerDacAudioEncoder(config)
        self.decoder = ParlerTTSForCausalLM(
            config.decoder,
            attention_implementation=attention_implementation,
        )
        if config.text_encoder.d_model != config.decoder.hidden_size:
            self.enc_to_dec_proj = nn.Linear(
                config.text_encoder.d_model,
                config.decoder.hidden_size,
            )
        self.embed_prompts = nn.Embedding(
            config.vocab_size,
            config.decoder.hidden_size,
        )
        self.prompt_cross_attention = config.prompt_cross_attention
        if config.prompt_cross_attention:
            self.embed_positions = ParlerSinusoidalPositionalEmbedding(
                config.decoder.max_position_embeddings,
                config.decoder.hidden_size,
            )

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose the public acoustic-token boundary for the selected mode."""
        if mode == "training":
            attribute = "forward"
        elif mode == "inference":
            attribute = "generate"
        else:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (OptimizationCompileTarget(
            f"parlertts.{attribute}",
            self,
            attribute,
        ), )

    def freeze_encoders(self, freeze_text_encoder: bool = True) -> None:
        """Apply upstream encoder-freezing semantics without freezing
        decoder."""
        self.text_encoder.requires_grad_(not freeze_text_encoder)
        self.audio_encoder.requires_grad_(False)

    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        encoded = self.text_encoder(
            input_ids,
            attention_mask,
        ).last_hidden_state
        projection = getattr(self, "enc_to_dec_proj", None)
        if projection is not None:
            encoded = projection(encoded)
        if attention_mask is not None:
            encoded = encoded * attention_mask[..., None]
        return encoded

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        prompt_input_ids: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        input_values: Tensor | None = None,
        loss_reduction: str = "mean",
    ) -> ParlerTTSOutput:
        encoder_hidden_states = self.encode_text(input_ids, attention_mask)
        prompt_hidden_states = (None if prompt_input_ids is None else self.embed_prompts(prompt_input_ids))
        if prompt_hidden_states is not None and self.prompt_cross_attention:
            prompt_hidden_states = prompt_hidden_states + self.embed_positions(prompt_hidden_states).to(
                prompt_hidden_states)
            encoder_hidden_states = torch.cat(
                (encoder_hidden_states, prompt_hidden_states),
                dim=1,
            )
            if attention_mask is not None or prompt_attention_mask is not None:
                if attention_mask is None:
                    attention_mask = torch.ones(
                        encoder_hidden_states.shape[0],
                        encoder_hidden_states.shape[1] - prompt_hidden_states.shape[1],
                        dtype=prompt_attention_mask.dtype,
                        device=prompt_attention_mask.device,
                    )
                if prompt_attention_mask is None:
                    prompt_attention_mask = torch.ones(
                        prompt_hidden_states.shape[:2],
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                attention_mask = torch.cat(
                    (attention_mask, prompt_attention_mask),
                    dim=1,
                )
            prompt_hidden_states = None
            prompt_attention_mask = None
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels,
                pad_token_id=self.config.pad_token_id,
                decoder_start_token_id=self.config.decoder_start_token_id,
            ).transpose(1, 2)
        elif decoder_input_ids is None:
            if input_values is None:
                raise ValueError("Provide labels, decoder IDs, or waveform input.")
            decoder_input_ids = self.audio_encoder.encode(input_values)
        decoded = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            prompt_hidden_states=prompt_hidden_states,
            prompt_attention_mask=prompt_attention_mask,
            labels=labels,
            loss_reduction=loss_reduction,
        )
        return ParlerTTSOutput(
            loss=decoded.loss,
            logits=decoded.logits,
            per_codebook_losses=decoded.per_codebook_losses,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        prompt_input_ids: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        max_new_tokens: int | None = None,
        max_length: int = 2_580,
        min_new_tokens: int = 10,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
    ) -> ParlerTTSOutput:
        """Generate delayed DAC tokens and decode them to 44.1 kHz audio.

        This implementation deliberately recomputes the prefix instead
        of depending on a third-party KV-cache abstraction. The math and
        sampled token distribution are unchanged; optimization backends
        can add a cache through VoiceHub's execution-strategy boundary
        later.
        """
        if prompt_input_ids is None:
            raise ValueError("Parler-TTS generation requires prompt text IDs.")
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("`max_length` must be an integer.")
        if max_length < 2:
            raise ValueError("`max_length` must be at least two.")
        if max_new_tokens is None:
            max_new_tokens = max_length - 1
        elif isinstance(max_new_tokens, bool) or not isinstance(
                max_new_tokens,
                int,
        ):
            raise TypeError("`max_new_tokens` must be an integer or None.")
        if max_new_tokens < 1:
            raise ValueError("`max_new_tokens` must be positive.")
        if (isinstance(min_new_tokens, bool) or not isinstance(min_new_tokens, int)):
            raise TypeError("`min_new_tokens` must be an integer.")
        if not 0 <= min_new_tokens <= max_new_tokens:
            raise ValueError("`min_new_tokens` must be between zero and max_new_tokens.")
        batch_size = input_ids.shape[0]
        encoder_hidden_states = self.encode_text(input_ids, attention_mask)
        prompt_hidden_states = self.embed_prompts(prompt_input_ids)
        if self.prompt_cross_attention:
            prompt_hidden_states = prompt_hidden_states + self.embed_positions(prompt_hidden_states).to(
                prompt_hidden_states)
            encoder_hidden_states = torch.cat(
                (encoder_hidden_states, prompt_hidden_states),
                dim=1,
            )
            if attention_mask is not None or prompt_attention_mask is not None:
                if attention_mask is None:
                    attention_mask = torch.ones(
                        input_ids.shape,
                        dtype=prompt_attention_mask.dtype,
                        device=input_ids.device,
                    )
                if prompt_attention_mask is None:
                    prompt_attention_mask = torch.ones_like(prompt_input_ids)
                attention_mask = torch.cat(
                    (attention_mask, prompt_attention_mask),
                    dim=1,
                )
            prompt_hidden_states = None
            prompt_attention_mask = None
        initial = torch.full(
            (batch_size * self.decoder.num_codebooks, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        maximum_length = max_new_tokens + initial.shape[-1]
        generated, pattern = self.decoder.build_delay_pattern_mask(
            initial,
            bos_token_id=self.config.decoder.bos_token_id,
            pad_token_id=self.config.decoder.pad_token_id,
            max_length=maximum_length,
        )
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.config.decoder.eos_token_id,
            pad_token_id=self.config.decoder.pad_token_id,
            seed=seed,
            use_cache=False,
        )
        generator = create_generator(input_ids.device, seed)
        first_unfinished = (torch.arange(batch_size, device=input_ids.device) * self.decoder.num_codebooks)
        final_codebooks = (first_unfinished + self.decoder.num_codebooks - 1)
        codebook_rows = torch.arange(
            batch_size * self.decoder.num_codebooks,
            device=input_ids.device,
        )
        for step in range(max_new_tokens):
            constrained = apply_delay_pattern_mask(generated, pattern)
            output = self.decoder(
                constrained,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                prompt_hidden_states=prompt_hidden_states,
                prompt_attention_mask=prompt_attention_mask,
            )
            next_logits = output.logits[:, -1].clone()
            eos_seen = (constrained == self.config.decoder.eos_token_id).any(dim=1)
            first_unfinished = torch.where(
                eos_seen[first_unfinished]
                & (first_unfinished < final_codebooks),
                first_unfinished + 1,
                first_unfinished,
            )
            eos_forbidden = codebook_rows > first_unfinished.repeat_interleave(self.decoder.num_codebooks)
            next_logits[
                eos_forbidden,
                self.config.decoder.eos_token_id,
            ] = -math.inf
            if step < min_new_tokens:
                next_logits[:, self.config.decoder.eos_token_id] = -math.inf
            next_token = sample_next_token(
                next_logits,
                constrained.clamp_max(self.decoder.vocab_size - 1),
                generation_config,
                generator=generator,
            )
            next_token = torch.where(
                eos_seen,
                torch.full_like(next_token, self.config.decoder.pad_token_id),
                next_token,
            )
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            if generated.shape[1] >= maximum_length:
                break
            finished = eos_seen | (next_token == self.config.decoder.eos_token_id)
            if finished[final_codebooks].all():
                break
        delayed = apply_delay_pattern_mask(generated, pattern)
        _, final_pattern = self.decoder.build_delay_pattern_mask(
            initial,
            bos_token_id=self.config.decoder.bos_token_id,
            pad_token_id=self.config.decoder.pad_token_id,
            max_length=delayed.shape[1],
        )
        valid = ((final_pattern != self.config.decoder.bos_token_id)
                 & (final_pattern != self.config.decoder.pad_token_id))
        flat_codes = delayed[valid]
        if flat_codes.numel() % (batch_size * self.decoder.num_codebooks):
            raise RuntimeError("Generated delay pattern cannot form DAC frames.")
        codes = flat_codes.reshape(
            batch_size,
            self.decoder.num_codebooks,
            -1,
        )
        frame_valid = (codes < self.config.audio_encoder.codebook_size).all(dim=1)
        audio_rows = []
        for row, mask in zip(codes, frame_valid):
            selected = row[:, mask]
            if selected.shape[-1] == 0:
                audio_rows.append(torch.zeros(
                    1,
                    device=row.device,
                    dtype=next_logits.dtype,
                ))
            else:
                audio_rows.append(self.audio_encoder.decode(selected[None]).squeeze())
        audio_values = nn.utils.rnn.pad_sequence(
            audio_rows,
            batch_first=True,
        )
        return ParlerTTSOutput(
            loss=None,
            logits=next_logits,
            audio_values=audio_values,
            audio_codes=codes,
        )


__all__ = [
    "ParlerCausalLMOutput",
    "ParlerDecoder",
    "ParlerDecoderLayer",
    "ParlerDacAudioEncoder",
    "ParlerSinusoidalPositionalEmbedding",
    "ParlerTTSForCausalLM",
    "ParlerTTSForConditionalGeneration",
    "ParlerTTSOutput",
    "apply_delay_pattern_mask",
    "build_delay_pattern_mask",
    "prepare_audio_code_labels",
    "shift_tokens_right",
]
