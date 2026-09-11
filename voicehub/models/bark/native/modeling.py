"""VoiceHub-owned Bark transformer and generation runtime.

The tensor-bearing module names intentionally match the pinned Hugging
Face Bark graph for the semantic, coarse, and fine stages.  The embedded
Encodec uses VoiceHub's native implementation; checkpoint.py owns the
audited, bijective namespace translation for the provider checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicehub.components.audio.codecs.encodec import EncodecModel
from voicehub.optimization.protocols import OptimizationCompileTarget

from .configuration import (
    BarkArchitectureConfig,
    BarkCoarseConfig,
    BarkCoarseGenerationConfig,
    BarkFineConfig,
    BarkFineGenerationConfig,
    BarkGenerationConfig,
    BarkSemanticConfig,
    BarkSemanticGenerationConfig,
    BarkSubModelConfig,
)


@dataclass(slots=True)
class BarkCausalOutput:
    """Causal stage logits and optional key/value cache."""

    logits: Tensor
    past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass(slots=True)
class BarkFineOutput:
    """Fine-stage logits and optional diagnostics."""

    logits: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


class BarkSelfAttention(nn.Module):
    """Multi-head attention with Bark's published projection namespace."""

    def __init__(
        self,
        config: BarkSubModelConfig,
        *,
        is_causal: bool,
    ) -> None:
        super().__init__()
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.att_proj = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.bias,
        )
        self.out_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.bias,
        )
        self.is_causal = is_causal
        if is_causal:
            causal = torch.tril(torch.ones(
                config.block_size,
                config.block_size,
                dtype=torch.bool,
            )).view(1, 1, config.block_size, config.block_size)
            self.register_buffer("bias", causal)

    def _split_heads(self, value: Tensor) -> Tensor:
        shape = value.shape[:-1] + (self.num_heads, self.head_dim)
        return value.view(shape).permute(0, 2, 1, 3)

    def _merge_heads(self, value: Tensor) -> Tensor:
        value = value.transpose(1, 2).contiguous()
        return value.view(value.shape[:-2] + (self.embed_dim, ))

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[
            Tensor,
            tuple[Tensor, Tensor] | None,
            Tensor | None,
    ]:
        query, key, value = self.att_proj(hidden_states).split(
            self.embed_dim,
            dim=-1,
        )
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)
        present = (key, value) if use_cache else None

        weights = torch.matmul(query, key.transpose(-1, -2))
        weights = weights * (1.0 / math.sqrt(self.head_dim))
        if self.is_causal:
            query_length = query.shape[-2]
            key_length = key.shape[-2]
            if key_length > self.bias.shape[-1]:
                raise ValueError("Bark sequence length exceeds the configured block size.")
            causal = self.bias[
                :,
                :,
                key_length - query_length:key_length,
                :key_length,
            ]
            weights = weights.masked_fill(
                ~causal,
                torch.finfo(weights.dtype).min,
            )
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(value.dtype)
        weights = self.attn_dropout(weights)
        attended = torch.matmul(weights, value)
        attended = self._merge_heads(attended)
        attended = self.resid_dropout(self.out_proj(attended))
        return attended, present, weights if output_attentions else None


class BarkMLP(nn.Module):

    def __init__(self, config: BarkSubModelConfig) -> None:
        super().__init__()
        self.in_proj = nn.Linear(
            config.hidden_size,
            4 * config.hidden_size,
            bias=config.bias,
        )
        self.out_proj = nn.Linear(
            4 * config.hidden_size,
            config.hidden_size,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.dropout(self.out_proj(self.gelu(self.in_proj(hidden_states))))


class BarkBlock(nn.Module):

    def __init__(
        self,
        config: BarkSubModelConfig,
        *,
        is_causal: bool,
    ) -> None:
        super().__init__()
        layernorm_bias = config.bias if is_causal else True
        self.layernorm_1 = nn.LayerNorm(
            config.hidden_size,
            bias=layernorm_bias,
        )
        self.layernorm_2 = nn.LayerNorm(
            config.hidden_size,
            bias=layernorm_bias,
        )
        self.attn = BarkSelfAttention(config, is_causal=is_causal)
        self.mlp = BarkMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None, Tensor | None]:
        attended, present, weights = self.attn(
            self.layernorm_1(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + self.mlp(self.layernorm_2(hidden_states))
        return hidden_states, present, weights


def _additive_attention_mask(
    attention_mask: Tensor | None,
    *,
    dtype: torch.dtype,
    key_length: int,
) -> Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError("Bark attention masks must have shape [batch, tokens].")
    if attention_mask.shape[-1] != key_length:
        raise ValueError("Bark attention mask length must match the cached key length.")
    mask = attention_mask[:, None, None, :].to(dtype=dtype)
    return (1.0 - mask) * torch.finfo(dtype).min


class BarkCausalModel(nn.Module):
    """GPT-like Bark stage with exact published parameter names."""

    def __init__(self, config: BarkSubModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_embeds_layer = nn.Embedding(
            config.input_vocab_size,
            config.hidden_size,
        )
        self.position_embeds_layer = nn.Embedding(
            config.block_size,
            config.hidden_size,
        )
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([BarkBlock(config, is_causal=True) for _ in range(config.num_layers)])
        self.layernorm_final = nn.LayerNorm(
            config.hidden_size,
            bias=config.bias,
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.output_vocab_size,
            bias=False,
        )
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def device(self) -> torch.device:
        return self.input_embeds_layer.weight.device

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_: Any,
    ) -> BarkCausalOutput | tuple[Any, ...]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of `input_ids` or `inputs_embeds` to Bark.")
        if inputs_embeds is None:
            inputs_embeds = self.input_embeds_layer(input_ids)
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[1] == 0:
            raise ValueError("Bark embeddings must have shape [batch, tokens, hidden].")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if not isinstance(use_cache, bool):
            raise TypeError("Bark `use_cache` must be a boolean.")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("Bark cache depth does not match the transformer.")
        past_length = (0 if past_key_values is None else past_key_values[0][0].shape[-2])
        sequence_length = inputs_embeds.shape[1]
        total_length = past_length + sequence_length
        if total_length > self.config.block_size:
            raise ValueError(
                f"Bark sequence length {total_length} exceeds block size "
                f"{self.config.block_size}.")
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                total_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0)
        if position_ids.shape[-1] != sequence_length:
            raise ValueError("Bark position IDs must align with the input tokens.")
        hidden_states = self.drop(inputs_embeds + self.position_embeds_layer(position_ids))
        additive_mask = _additive_attention_mask(
            attention_mask,
            dtype=hidden_states.dtype,
            key_length=total_length,
        )
        hidden_history: list[Tensor] | None = [] if output_hidden_states else None
        attention_history: list[Tensor] | None = [] if output_attentions else None
        present_values: list[tuple[Tensor, Tensor]] | None = [] if use_cache else None
        for index, layer in enumerate(self.layers):
            if hidden_history is not None:
                hidden_history.append(hidden_states)
            layer_past = (None if past_key_values is None else past_key_values[index])
            hidden_states, present, weights = layer(
                hidden_states,
                attention_mask=additive_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            if present_values is not None and present is not None:
                present_values.append(present)
            if attention_history is not None and weights is not None:
                attention_history.append(weights)
        hidden_states = self.layernorm_final(hidden_states)
        if hidden_history is not None:
            hidden_history.append(hidden_states)
        output = BarkCausalOutput(
            logits=self.lm_head(hidden_states),
            past_key_values=(tuple(present_values) if present_values is not None else None),
            hidden_states=(tuple(hidden_history) if hidden_history is not None else None),
            attentions=(tuple(attention_history) if attention_history is not None else None),
        )
        if return_dict:
            return output
        return tuple(
            item for item in (
                output.logits,
                output.past_key_values,
                output.hidden_states,
                output.attentions,
            ) if item is not None)

    def _autoregressive_generate(
        self,
        prefix_ids: Tensor,
        *,
        inputs_embeds: Tensor | None = None,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_token_id: int | None = None,
        min_eos_p: float | None = None,
        allowed_token_range: tuple[int, int] | None = None,
        alternating_ranges: tuple[tuple[int, int], ...] | None = None,
    ) -> Tensor:
        if prefix_ids.ndim != 2 or prefix_ids.shape[1] == 0:
            raise ValueError("Bark generation prefix must be a non-empty token batch.")
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0):
            raise ValueError("Bark `max_new_tokens` must be non-negative.")
        if not 0 < temperature:
            raise ValueError("Bark temperature must be greater than zero.")
        if not 0 < top_p <= 1:
            raise ValueError("Bark top-p must be in (0, 1].")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("Bark top-k must be a non-negative integer.")

        result = prefix_ids
        cache = None
        finished = torch.zeros(
            prefix_ids.shape[0],
            dtype=torch.bool,
            device=prefix_ids.device,
        )
        for step in range(max_new_tokens):
            if cache is None:
                output = self(
                    prefix_ids if inputs_embeds is None else None,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                )
            else:
                output = self(
                    result[:, -1:],
                    past_key_values=cache,
                    use_cache=True,
                )
            cache = output.past_key_values
            logits = output.logits[:, -1, :].float()
            if allowed_token_range is not None:
                logits = _mask_outside(logits, *allowed_token_range)
            if alternating_ranges is not None:
                selected = alternating_ranges[step % len(alternating_ranges)]
                logits = _mask_outside(logits, *selected)
            if eos_token_id is not None and min_eos_p is not None:
                eos_probability = F.softmax(logits, dim=-1)[:, eos_token_id]
                prioritize = eos_probability >= min_eos_p
                if bool(prioritize.any()):
                    logits[prioritize] = torch.finfo(logits.dtype).min
                    logits[prioritize, eos_token_id] = 0
            next_token = _sample_token(
                logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )
                finished |= next_token == eos_token_id
            result = torch.cat((result, next_token[:, None]), dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return result


def _mask_outside(logits: Tensor, start: int, end: int) -> Tensor:
    if not 0 <= start < end <= logits.shape[-1]:
        raise ValueError("Bark generation token range is outside the vocabulary.")
    masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
    masked[:, start:end] = logits[:, start:end]
    return masked


def _sample_token(
    logits: Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Tensor:
    scaled = logits / temperature
    if top_k:
        threshold = torch.topk(
            scaled,
            k=min(top_k, scaled.shape[-1]),
            dim=-1,
        ).values[:, -1:]
        scaled = scaled.masked_fill(
            scaled < threshold,
            torch.finfo(scaled.dtype).min,
        )
    if top_p < 1:
        sorted_logits, sorted_indices = torch.sort(
            scaled,
            descending=True,
            dim=-1,
        )
        cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(
            remove,
            torch.finfo(sorted_logits.dtype).min,
        )
        filtered = torch.full_like(scaled, torch.finfo(scaled.dtype).min)
        filtered.scatter_(1, sorted_indices, sorted_logits)
        scaled = filtered
    if do_sample:
        probabilities = F.softmax(scaled, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(1)
    return scaled.argmax(dim=-1)


class BarkSemanticModel(BarkCausalModel):
    config: BarkSemanticConfig

    def generate(
        self,
        input_ids: Tensor,
        *,
        generation_config: BarkSemanticGenerationConfig,
        history_prompt: dict[str, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_eos_p: float | None = None,
    ) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] == 0:
            raise ValueError("Bark text IDs must have shape [batch, tokens].")
        maximum = generation_config.max_input_semantic_length
        input_ids = input_ids[:, :maximum].long()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask[:, :maximum]
        if input_ids.shape[1] < maximum:
            padding = maximum - input_ids.shape[1]
            input_ids = F.pad(
                input_ids,
                (0, padding),
                value=0,
            )
            attention_mask = F.pad(attention_mask, (0, padding), value=0)
        input_ids = input_ids + generation_config.text_encoding_offset
        input_ids = input_ids.masked_fill(
            ~attention_mask.bool(),
            generation_config.text_pad_token,
        )
        batch_size = input_ids.shape[0]
        if history_prompt is not None:
            history = history_prompt.get("semantic_prompt")
            if not isinstance(history, Tensor) or history.ndim != 1:
                raise ValueError("Bark semantic history must be a one-dimensional tensor.")
            history = history[-maximum:].to(
                device=self.device,
                dtype=torch.long,
            )
            history = F.pad(
                history,
                (0, maximum - history.shape[0]),
                value=generation_config.semantic_pad_token,
            )
        else:
            history = torch.full(
                (maximum, ),
                generation_config.semantic_pad_token,
                device=self.device,
                dtype=torch.long,
            )
        history = history.unsqueeze(0).expand(batch_size, -1)
        infer = torch.full(
            (batch_size, 1),
            generation_config.semantic_infer_token,
            device=self.device,
            dtype=torch.long,
        )
        embeddings = torch.cat(
            (
                self.input_embeds_layer(input_ids) + self.input_embeds_layer(history),
                self.input_embeds_layer(infer),
            ),
            dim=1,
        )
        prefix = torch.ones(
            batch_size,
            maximum + 1,
            device=self.device,
            dtype=torch.long,
        )
        output = self._autoregressive_generate(
            prefix,
            inputs_embeds=embeddings,
            max_new_tokens=(generation_config.max_new_tokens if max_new_tokens is None else max_new_tokens),
            do_sample=(generation_config.do_sample if do_sample is None else do_sample),
            temperature=(generation_config.temperature if temperature is None else temperature),
            top_k=generation_config.top_k if top_k is None else top_k,
            top_p=generation_config.top_p if top_p is None else top_p,
            eos_token_id=generation_config.eos_token_id,
            min_eos_p=(generation_config.min_eos_p if min_eos_p is None else min_eos_p),
            allowed_token_range=(
                0,
                generation_config.semantic_pad_token + 1,
            ),
        )
        return output[:, maximum + 1:]


class BarkCoarseModel(BarkCausalModel):
    config: BarkCoarseConfig

    def _histories(
        self,
        *,
        history_prompt: dict[str, Tensor] | None,
        max_coarse_history: int,
        semantic_to_coarse_ratio: float,
        batch_size: int,
        semantic_config: BarkSemanticGenerationConfig,
        codebook_size: int,
    ) -> tuple[Tensor, Tensor]:
        if history_prompt is None:
            empty = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=self.device,
            )
            return empty, empty
        semantic = history_prompt.get("semantic_prompt")
        coarse = history_prompt.get("coarse_prompt")
        if (not isinstance(semantic, Tensor) or semantic.ndim != 1 or not isinstance(coarse, Tensor) or
                coarse.ndim != 2 or coarse.shape[0] != 2):
            raise ValueError("Bark history requires semantic [tokens] and coarse "
                             "[2, frames] tensors.")
        semantic = semantic.to(device=self.device, dtype=torch.long)
        semantic = semantic.unsqueeze(0).expand(batch_size, -1)
        coarse = coarse.to(device=self.device, dtype=torch.long).clone()
        coarse[1] += codebook_size
        coarse = coarse.transpose(0, 1).reshape(-1)
        coarse = coarse + semantic_config.semantic_vocab_size
        coarse = coarse.unsqueeze(0).expand(batch_size, -1)
        max_semantic_history = math.floor(max_coarse_history / semantic_to_coarse_ratio)
        semantic_count = min(
            max_semantic_history,
            semantic.shape[1] - semantic.shape[1] % 2,
            math.floor(coarse.shape[1] / semantic_to_coarse_ratio),
        )
        coarse_count = round(semantic_count * semantic_to_coarse_ratio)
        semantic = semantic[:, -semantic_count:].long()
        coarse = coarse[:, -coarse_count:].long()
        return semantic, coarse[:, :-2]

    def generate(
        self,
        semantic_output: Tensor,
        *,
        semantic_config: BarkSemanticGenerationConfig,
        generation_config: BarkCoarseGenerationConfig,
        codebook_size: int = 1024,
        history_prompt: dict[str, Tensor] | None = None,
        return_output_lengths: bool = False,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if semantic_output.ndim != 2 or semantic_output.shape[1] == 0:
            raise ValueError("Bark semantic output must have shape [batch, tokens].")
        semantic_output = semantic_output.clone().long()
        semantic_output.masked_fill_(
            semantic_output == semantic_config.semantic_pad_token,
            generation_config.coarse_semantic_pad_token,
        )
        ratio = (
            generation_config.coarse_rate_hz / semantic_config.semantic_rate_hz *
            generation_config.n_coarse_codebooks)
        output_lengths = (semantic_output != generation_config.coarse_semantic_pad_token).sum(dim=1)
        output_lengths = torch.floor(output_lengths * ratio / generation_config.n_coarse_codebooks)
        output_lengths = torch.round(output_lengths * generation_config.n_coarse_codebooks).int()
        maximum_generated = int(output_lengths.max().item())
        batch_size = semantic_output.shape[0]
        semantic_history, coarse = self._histories(
            history_prompt=history_prompt,
            max_coarse_history=generation_config.max_coarse_history,
            semantic_to_coarse_ratio=ratio,
            batch_size=batch_size,
            semantic_config=semantic_config,
            codebook_size=codebook_size,
        )
        base_semantic_index = semantic_history.shape[1]
        semantic_output = torch.cat(
            (semantic_history, semantic_output),
            dim=1,
        )
        history_length = coarse.shape[1]
        generated_length = 0
        window_count = math.ceil(maximum_generated / generation_config.sliding_window_len)
        max_semantic_history = math.floor(generation_config.max_coarse_history / ratio)
        ranges = tuple((
            semantic_config.semantic_vocab_size + codebook * codebook_size,
            semantic_config.semantic_vocab_size + (codebook + 1) * codebook_size,
        ) for codebook in range(generation_config.n_coarse_codebooks))
        for _ in range(window_count):
            semantic_index = (base_semantic_index + round(generated_length / ratio))
            semantic_window = semantic_output[
                :,
                max(0, semantic_index - max_semantic_history):,
            ]
            semantic_window = semantic_window[
                :,
                :generation_config.max_coarse_input_length,
            ]
            semantic_window = F.pad(
                semantic_window,
                (
                    0,
                    generation_config.max_coarse_input_length - semantic_window.shape[1],
                ),
                value=generation_config.coarse_semantic_pad_token,
            )
            infer = torch.full(
                (batch_size, 1),
                generation_config.coarse_infer_token,
                device=self.device,
                dtype=torch.long,
            )
            prefix = torch.cat(
                (
                    semantic_window,
                    infer,
                    coarse[:, -generation_config.max_coarse_history:],
                ),
                dim=1,
            )
            requested = min(
                generation_config.sliding_window_len,
                maximum_generated - generated_length,
            )
            generated = self._autoregressive_generate(
                prefix,
                max_new_tokens=requested,
                do_sample=(generation_config.do_sample if do_sample is None else do_sample),
                temperature=(generation_config.temperature if temperature is None else temperature),
                top_k=(generation_config.top_k if top_k is None else top_k),
                top_p=(generation_config.top_p if top_p is None else top_p),
                alternating_ranges=ranges,
            )
            coarse = torch.cat(
                (coarse, generated[:, prefix.shape[1]:]),
                dim=1,
            )
            generated_length = coarse.shape[1] - history_length
        output = coarse[:, history_length:]
        if return_output_lengths:
            return output, output_lengths
        return output


class BarkFineModel(nn.Module):
    """Non-causal multi-codebook Bark refinement transformer."""

    def __init__(self, config: BarkFineConfig) -> None:
        super().__init__()
        self.config = config
        self.input_embeds_layers = nn.ModuleList(
            [nn.Embedding(config.input_vocab_size, config.hidden_size) for _ in range(config.n_codes_total)])
        self.position_embeds_layer = nn.Embedding(
            config.block_size,
            config.hidden_size,
        )
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([BarkBlock(config, is_causal=False) for _ in range(config.num_layers)])
        self.layernorm_final = nn.LayerNorm(config.hidden_size)
        self.lm_heads = nn.ModuleList([
            nn.Linear(
                config.hidden_size,
                config.output_vocab_size,
                bias=False,
            ) for _ in range(
                config.n_codes_given,
                config.n_codes_total,
            )
        ])
        self.apply(self._initialize)
        # The published graph ties each predicted codebook head to the
        # corresponding input embedding.
        for index, head in enumerate(self.lm_heads):
            head.weight = self.input_embeds_layers[index + config.n_codes_given].weight

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def device(self) -> torch.device:
        return self.position_embeds_layer.weight.device

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        codebook_idx: int,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_: Any,
    ) -> BarkFineOutput | tuple[Any, ...]:
        if (isinstance(codebook_idx, bool) or not isinstance(codebook_idx, int) or
                not self.config.n_codes_given <= codebook_idx < self.config.n_codes_total):
            raise ValueError("Bark `codebook_idx` must identify a fine codebook.")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of `input_ids` or `inputs_embeds` to Bark fine.")
        if input_ids is not None:
            if (input_ids.ndim != 3 or input_ids.shape[-1] != self.config.n_codes_total):
                raise ValueError("Bark fine IDs must have shape "
                                 "[batch, tokens, n_codes_total].")
            pieces = [
                embedding(input_ids[:, :, index]) for index, embedding in enumerate(self.input_embeds_layers)
                if index <= codebook_idx
            ]
            inputs_embeds = torch.stack(pieces, dim=-1).sum(dim=-1)
        sequence_length = inputs_embeds.shape[1]
        if sequence_length > self.config.block_size:
            raise ValueError("Bark fine sequence exceeds the configured block size.")
        if position_ids is None:
            position_ids = torch.arange(
                sequence_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0)
        hidden_states = self.drop(inputs_embeds + self.position_embeds_layer(position_ids))
        additive_mask = _additive_attention_mask(
            attention_mask,
            dtype=hidden_states.dtype,
            key_length=sequence_length,
        )
        hidden_history: list[Tensor] | None = [] if output_hidden_states else None
        attention_history: list[Tensor] | None = [] if output_attentions else None
        for layer in self.layers:
            if hidden_history is not None:
                hidden_history.append(hidden_states)
            hidden_states, _, weights = layer(
                hidden_states,
                attention_mask=additive_mask,
                output_attentions=output_attentions,
            )
            if attention_history is not None and weights is not None:
                attention_history.append(weights)
        hidden_states = self.layernorm_final(hidden_states)
        if hidden_history is not None:
            hidden_history.append(hidden_states)
        output = BarkFineOutput(
            logits=self.lm_heads[codebook_idx - self.config.n_codes_given](hidden_states),
            hidden_states=(tuple(hidden_history) if hidden_history is not None else None),
            attentions=(tuple(attention_history) if attention_history is not None else None),
        )
        if return_dict:
            return output
        return tuple(
            item for item in (
                output.logits,
                output.hidden_states,
                output.attentions,
            ) if item is not None)

    @torch.no_grad()
    def generate(
        self,
        coarse_output: Tensor,
        *,
        semantic_config: BarkSemanticGenerationConfig,
        coarse_config: BarkCoarseGenerationConfig,
        generation_config: BarkFineGenerationConfig,
        codebook_size: int = 1024,
        history_prompt: dict[str, Tensor] | None = None,
        temperature: float | None = None,
    ) -> Tensor:
        if (coarse_output.ndim != 2 or coarse_output.shape[1] % coarse_config.n_coarse_codebooks):
            raise ValueError("Bark coarse output must contain complete interleaved frames.")
        temperature = (generation_config.temperature if temperature is None else temperature)
        coarse = coarse_output.view(
            coarse_output.shape[0],
            -1,
            coarse_config.n_coarse_codebooks,
        )
        coarse = torch.remainder(
            coarse - semantic_config.semantic_vocab_size,
            codebook_size,
        )
        batch_size = coarse.shape[0]
        fine = F.pad(
            coarse,
            (
                0,
                generation_config.n_fine_codebooks - coarse_config.n_coarse_codebooks,
            ),
            value=codebook_size,
        )
        history = None
        if history_prompt is not None:
            history = history_prompt.get("fine_prompt")
            if (not isinstance(history, Tensor) or history.ndim != 2 or
                    history.shape[0] != generation_config.n_fine_codebooks):
                raise ValueError("Bark fine history must have shape [8, frames].")
            history = history.T.unsqueeze(0).expand(batch_size, -1, -1)
            history = history.to(device=self.device, dtype=torch.long)
            history = history[:, -generation_config.max_fine_history_length:]
            fine = torch.cat((history, fine), dim=1)
        history_length = 0 if history is None else history.shape[1]
        remove_end = 0
        if fine.shape[1] < generation_config.max_fine_input_length:
            remove_end = generation_config.max_fine_input_length - fine.shape[1]
            fine = F.pad(
                fine,
                (0, 0, 0, remove_end),
                value=codebook_size,
            )
        loops = (
            coarse.shape[1] - (generation_config.max_fine_input_length -
                               history_length)) / generation_config.max_fine_history_length
        loops = max(0, math.ceil(loops)) + 1
        n_coarse = coarse_config.n_coarse_codebooks
        for outer in range(loops):
            start = min(
                outer * generation_config.max_fine_history_length,
                fine.shape[1] - generation_config.max_fine_input_length,
            )
            fill = min(
                history_length + outer * generation_config.max_fine_history_length,
                fine.shape[1] - generation_config.max_fine_history_length,
            )
            relative_fill = fill - start
            buffer = fine[
                :,
                start:start + generation_config.max_fine_input_length,
                :,
            ].clone()
            for codebook in range(
                    n_coarse,
                    generation_config.n_fine_codebooks,
            ):
                logits = self(
                    buffer,
                    codebook_idx=codebook,
                ).logits
                relevant = logits[:, relative_fill:, :codebook_size]
                if temperature is None or temperature == 1:
                    prediction = relevant.argmax(dim=-1)
                else:
                    if temperature <= 0:
                        raise ValueError("Bark fine temperature must be greater than zero.")
                    probabilities = F.softmax(
                        relevant / temperature,
                        dim=-1,
                    )
                    prediction = torch.multinomial(
                        probabilities.reshape(-1, codebook_size),
                        num_samples=1,
                    ).view(batch_size, -1)
                buffer[:, relative_fill:, codebook] = prediction.to(dtype=buffer.dtype)
            length = (generation_config.max_fine_input_length - relative_fill)
            fine[:, fill:fill + length, n_coarse:] = buffer[
                :,
                relative_fill:,
                n_coarse:,
            ]
        fine = fine.transpose(1, 2)[:, :, history_length:]
        if remove_end:
            fine = fine[:, :, :-remove_end]
        if fine.shape[-1] != coarse.shape[1]:
            raise RuntimeError("Bark fine generation changed the acoustic frame count.")
        return fine


class BarkModel(nn.Module):
    """End-to-end native Bark graph with native Encodec decoding."""

    def __init__(
        self,
        config: BarkArchitectureConfig,
        *,
        generation_config: BarkGenerationConfig | None = None,
        codec_model: EncodecModel | nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.generation_config = generation_config or BarkGenerationConfig()
        if self.generation_config.sample_rate != config.codec.sample_rate:
            raise ValueError("Bark generation and codec sample rates must match.")
        if self.generation_config.codebook_size != config.codec.bins:
            raise ValueError("Bark generation and codec codebook sizes must match.")
        if (self.generation_config.fine.n_fine_codebooks != config.fine.n_codes_total):
            raise ValueError("Bark fine generation and model codebook counts must match.")
        if (self.generation_config.coarse.n_coarse_codebooks > self.generation_config.fine.n_fine_codebooks):
            raise ValueError("Bark coarse codebooks cannot exceed fine codebooks.")
        if (config.codec.resolved_n_q < self.generation_config.fine.n_fine_codebooks):
            raise ValueError("Bark codec has fewer quantizers than the fine stage.")
        semantic_generation = self.generation_config.semantic
        if (semantic_generation.semantic_infer_token >= config.semantic.input_vocab_size or
                semantic_generation.text_pad_token >= config.semantic.input_vocab_size or
                semantic_generation.semantic_pad_token >= config.semantic.output_vocab_size):
            raise ValueError("Bark semantic special tokens exceed the model vocabulary.")
        coarse_generation = self.generation_config.coarse
        if (coarse_generation.coarse_infer_token >= config.coarse.input_vocab_size or
                coarse_generation.coarse_semantic_pad_token >= config.coarse.input_vocab_size):
            raise ValueError("Bark coarse special tokens exceed the model vocabulary.")
        self.semantic = BarkSemanticModel(config.semantic)
        self.coarse_acoustics = BarkCoarseModel(config.coarse)
        self.fine_acoustics = BarkFineModel(config.fine)
        self.codec_model = (EncodecModel.from_config(config.codec) if codec_model is None else codec_model)

    @property
    def device(self) -> torch.device:
        return self.semantic.device

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Declare Bark's three executed transformer stage boundaries."""
        if mode not in {"inference", "training"}:
            raise ValueError("Bark compile targets require 'inference' or 'training' mode.")
        return (
            OptimizationCompileTarget(
                "semantic.forward",
                self.semantic,
                "forward",
            ),
            OptimizationCompileTarget(
                "coarse.forward",
                self.coarse_acoustics,
                "forward",
            ),
            OptimizationCompileTarget(
                "fine.forward",
                self.fine_acoustics,
                "forward",
            ),
        )

    def codec_decode(
        self,
        fine_output: Tensor,
        output_lengths: Tensor | None = None,
    ) -> Tensor | list[Tensor]:
        if not isinstance(self.codec_model, EncodecModel):
            decode_codes = getattr(self.codec_model, "decode_codes", None)
            if not callable(decode_codes):
                raise TypeError("Injected Bark codecs must implement `decode_codes`.")
            return decode_codes(fine_output, output_lengths=output_lengths)
        if fine_output.ndim != 3:
            raise ValueError("Bark fine codes must have shape [batch, codebooks, frames].")
        embeddings = self.codec_model.quantizer.decode(fine_output.transpose(0, 1))
        if output_lengths is None:
            return self.codec_model.decoder(embeddings).squeeze(1)
        audio: list[Tensor] = []
        for sample, length in zip(embeddings, output_lengths, strict=True):
            frames = int(length.item())
            audio.append(self.codec_model.decoder(sample[:, :frames].unsqueeze(0)).squeeze())
        return audio

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        history_prompt: dict[str, Tensor] | None = None,
        return_output_lengths: bool = False,
        **options: Any,
    ) -> Tensor | tuple[Tensor, list[int]]:
        semantic_options, coarse_options, fine_options = _split_options(options)
        semantic = self.semantic.generate(
            input_ids,
            generation_config=self.generation_config.semantic,
            history_prompt=history_prompt,
            attention_mask=attention_mask,
            **semantic_options,
        )
        coarse_result = self.coarse_acoustics.generate(
            semantic,
            semantic_config=self.generation_config.semantic,
            generation_config=self.generation_config.coarse,
            codebook_size=self.generation_config.codebook_size,
            history_prompt=history_prompt,
            return_output_lengths=return_output_lengths,
            **coarse_options,
        )
        output_lengths = None
        if return_output_lengths:
            coarse, output_lengths = coarse_result
            output_lengths = (output_lengths // self.generation_config.coarse.n_coarse_codebooks)
        else:
            coarse = coarse_result
        fine = self.fine_acoustics.generate(
            coarse,
            semantic_config=self.generation_config.semantic,
            coarse_config=self.generation_config.coarse,
            generation_config=self.generation_config.fine,
            codebook_size=self.generation_config.codebook_size,
            history_prompt=history_prompt,
            **fine_options,
        )
        audio = self.codec_decode(fine, output_lengths)
        if not return_output_lengths:
            return audio
        if not isinstance(audio, list):
            raise RuntimeError("Bark length-aware codec decoding must return a sample list.")
        lengths = [sample.numel() for sample in audio]
        return nn.utils.rnn.pad_sequence(audio, batch_first=True), lengths


def _split_options(options: dict[str, Any], ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    semantic: dict[str, Any] = {}
    coarse: dict[str, Any] = {}
    fine: dict[str, Any] = {}
    allowed = {
        "do_sample",
        "temperature",
        "top_k",
        "top_p",
    }
    semantic_only = {"max_new_tokens", "min_eos_p"}
    fine_allowed = {"temperature"}
    for name, value in options.items():
        if name.startswith("semantic_"):
            key = name[len("semantic_"):]
            if key not in allowed | semantic_only:
                raise ValueError(f"Unsupported Bark semantic option {key!r}.")
            semantic[key] = value
        elif name.startswith("coarse_"):
            key = name[len("coarse_"):]
            if key not in allowed:
                raise ValueError(f"Unsupported Bark coarse option {key!r}.")
            coarse[key] = value
        elif name.startswith("fine_"):
            key = name[len("fine_"):]
            if key not in fine_allowed:
                raise ValueError(f"Unsupported Bark fine option {key!r}.")
            fine[key] = value
        elif name in allowed:
            semantic.setdefault(name, value)
            coarse.setdefault(name, value)
            if name in fine_allowed:
                fine.setdefault(name, value)
        else:
            raise ValueError(f"Unsupported Bark generation option {name!r}.")
    return semantic, coarse, fine


__all__ = [
    "BarkCausalModel",
    "BarkCausalOutput",
    "BarkCoarseModel",
    "BarkFineModel",
    "BarkFineOutput",
    "BarkModel",
    "BarkSemanticModel",
]
