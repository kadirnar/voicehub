"""Native depth decoders used by MOSS-TTS Local and Realtime.

MOSS-TTS has two unrelated local-decoder families.  Local v1.5 uses a
GPT2-shaped block with even/odd rotary pairs.  The older Local release
and Realtime use Qwen3-shaped decoder blocks.  Keeping these
implementations separate makes their checkpoint namespaces and
positional semantics explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.causal_lm.native.configuration import Qwen3Config
from voicehub.models.causal_lm.native.modeling import CausalLMDecoderLayer
from voicehub.models.mosstts.native.configuration import MossGPT2Config
from voicehub.neural.cache import DynamicKVCache
from voicehub.neural.normalization import RMSNorm


@dataclass(frozen=True)
class LocalTransformerOutput:
    """Hidden states and an optional local KV cache."""

    last_hidden_state: Tensor
    past_key_values: tuple[tuple[Tensor, Tensor], ...] | DynamicKVCache | None = None


def _rotate_even_odd(value: Tensor) -> Tensor:
    even = value[..., ::2]
    odd = value[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(value)


def _local_rope(
    position_ids: Tensor,
    *,
    head_dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    inverse_frequency = 1.0 / (
        base**(torch.arange(
            0,
            head_dim,
            2,
            device=device,
            dtype=torch.float32,
        ) / head_dim))
    frequencies = torch.einsum(
        "bt,d->btd",
        position_ids.to(device=device, dtype=torch.float32),
        inverse_frequency,
    )
    cosine = frequencies.cos().repeat_interleave(2, dim=-1).unsqueeze(2)
    sine = frequencies.sin().repeat_interleave(2, dim=-1).unsqueeze(2)
    return cosine.to(dtype=dtype), sine.to(dtype=dtype)


class MossGPT2Attention(nn.Module):
    """GPT2-shaped causal attention with Local v1.5 RoPE."""

    def __init__(
        self,
        config: MossGPT2Config,
        layer_index: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.rope_base = config.rope_base
        self.attention_dropout = config.attention_dropout
        factory_kwargs = {"device": device, "dtype": dtype}
        self.c_attn = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=True,
            **factory_kwargs,
        )
        self.c_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            **factory_kwargs,
        )
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        layer_past: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        if hidden_states.ndim != 3:
            raise ValueError("Local hidden states must have shape [batch, sequence, hidden].")
        batch_size, query_length, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.hidden_size, dim=-1)
        query = query.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        )
        key = key.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        )
        value = value.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        )
        past_length = 0 if layer_past is None else int(layer_past[0].shape[1])
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + query_length,
                device=hidden_states.device,
                dtype=torch.long,
            ).unsqueeze(0)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[0] == 1 and batch_size != 1:
            position_ids = position_ids.expand(batch_size, -1)
        if tuple(position_ids.shape) != (batch_size, query_length):
            raise ValueError("Local `position_ids` must match batch and query length.")
        cosine, sine = _local_rope(
            position_ids,
            head_dim=self.head_dim,
            base=self.rope_base,
            device=hidden_states.device,
            dtype=query.dtype,
        )
        query = query * cosine + _rotate_even_odd(query) * sine
        key = key * cosine + _rotate_even_odd(key) * sine
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat(
                [past_key.to(device=key.device, dtype=key.dtype), key],
                dim=1,
            )
            value = torch.cat(
                [past_value.to(device=value.device, dtype=value.dtype), value],
                dim=1,
            )
        present = (key, value) if use_cache else None

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        key_length = key.shape[-2]
        query_positions = (past_length + torch.arange(query_length, device=hidden_states.device))
        key_positions = torch.arange(key_length, device=hidden_states.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        allowed = allowed.view(1, 1, query_length, key_length)
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != (
                    batch_size,
                    key_length,
            ):
                raise ValueError("Local attention mask must have shape [batch, cached + query].")
            allowed = allowed & attention_mask[:, None, None, :].to(torch.bool)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.head_dim**-0.5
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=query.dtype)
        probabilities = functional.dropout(
            probabilities,
            p=self.attention_dropout,
            training=self.training,
        )
        output = torch.matmul(probabilities, value)
        output = output.transpose(1, 2).reshape(
            batch_size,
            query_length,
            self.hidden_size,
        )
        return self.residual_dropout(self.c_proj(output)), present


class MossGPT2MLP(nn.Module):
    """SiLU feed-forward block under the official ``fc_in/fc_out`` names."""

    def __init__(
        self,
        config: MossGPT2Config,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fc_in = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            **factory_kwargs,
        )
        self.fc_out = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            **factory_kwargs,
        )
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.dropout(self.fc_out(functional.silu(self.fc_in(hidden_states))))


class MossGPT2Block(nn.Module):
    """One pre-norm Local v1.5 depth block."""

    def __init__(
        self,
        config: MossGPT2Config,
        layer_index: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ln_1 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            **factory_kwargs,
        )
        self.attn = MossGPT2Attention(
            config,
            layer_index,
            **factory_kwargs,
        )
        self.ln_2 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            **factory_kwargs,
        )
        self.mlp = MossGPT2MLP(config, **factory_kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
        layer_past: tuple[Tensor, Tensor] | None,
        use_cache: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        attention_output, present = self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, present


class MossGPT2Model(nn.Module):
    """Embedding-free Local v1.5 depth model."""

    def __init__(
        self,
        config: MossGPT2Config,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.h = nn.ModuleList([
            MossGPT2Block(
                config,
                layer_index,
                **factory_kwargs,
            ) for layer_index in range(config.num_hidden_layers)
        ])
        self.ln_f = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            **factory_kwargs,
        )
        self.gradient_checkpointing = False
        if initialize:
            self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
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

    def forward(
        self,
        inputs_embeds: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None,
        use_cache: bool = False,
    ) -> LocalTransformerOutput:
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "Local v1.5 inputs must have shape "
                f"[batch, sequence, {self.config.hidden_size}].")
        if (past_key_values is not None and len(past_key_values) != self.config.num_hidden_layers):
            raise ValueError("Local v1.5 cache layer count is incompatible.")
        hidden_states = inputs_embeds
        presents: list[tuple[Tensor, Tensor]] | None = [] if use_cache else None
        for index, block in enumerate(self.h):
            layer_past = None if past_key_values is None else past_key_values[index]
            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                layer_past=layer_past,
                use_cache=use_cache,
            )
            if presents is not None:
                if present is None:
                    raise RuntimeError("A cached local block returned no cache.")
                presents.append(present)
        return LocalTransformerOutput(
            last_hidden_state=self.ln_f(hidden_states),
            past_key_values=None if presents is None else tuple(presents),
        )


class MossQwenDepthModel(nn.Module):
    """Embedding-optional Qwen3 depth stack for Local and Realtime."""

    def __init__(
        self,
        config: Qwen3Config,
        *,
        audio_codebooks: int = 0,
        audio_vocab_size: int = 0,
        audio_pad_token_id: int | None = None,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        if audio_codebooks:
            self.embed_tokens: nn.Module | nn.ModuleList = nn.ModuleList([
                nn.Embedding(
                    audio_vocab_size,
                    config.hidden_size,
                    audio_pad_token_id,
                    **factory_kwargs,
                ) for _ in range(audio_codebooks)
            ])
        self.layers = nn.ModuleList([
            CausalLMDecoderLayer(
                config,
                layer_index,
                **factory_kwargs,
            ) for layer_index in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            **factory_kwargs,
        )
        self.gradient_checkpointing = False
        if initialize:
            self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
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
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def _embed_realtime(
        self,
        input_ids: Tensor,
        *,
        first_hidden_state: Tensor | None,
        codebook_index: int | None,
    ) -> Tensor:
        embeddings = getattr(self, "embed_tokens", None)
        if not isinstance(embeddings, nn.ModuleList):
            raise TypeError("This local depth model has no token embeddings.")
        if codebook_index is not None:
            if not 1 <= codebook_index <= len(embeddings):
                raise ValueError(f"`codebook_index` must be in [1, {len(embeddings)}].")
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(1)
            return embeddings[codebook_index - 1](input_ids[:, :1])
        if input_ids.ndim != 2:
            raise ValueError("Realtime local IDs must have shape [batch, depth].")
        if input_ids.shape[1] > len(embeddings) + 1:
            raise ValueError("Realtime local depth exceeds the embedding inventory.")
        hidden = torch.stack(
            [embeddings[max(index - 1, 0)](input_ids[:, index]) for index in range(input_ids.shape[1])],
            dim=1,
        )
        if first_hidden_state is not None:
            if first_hidden_state.ndim == 2:
                first_hidden_state = first_hidden_state.unsqueeze(1)
            if first_hidden_state.shape != hidden[:, :1].shape:
                raise ValueError("Realtime backbone hidden state has an invalid shape.")
            hidden[:, :1] = first_hidden_state
        return hidden

    def forward(
        self,
        *,
        inputs_embeds: Tensor | None = None,
        input_ids: Tensor | None = None,
        first_hidden_state: Tensor | None = None,
        codebook_index: int | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        use_cache: bool = False,
        apply_rope: bool = True,
    ) -> LocalTransformerOutput:
        if (inputs_embeds is None) == (input_ids is None):
            raise ValueError("Specify exactly one local input representation.")
        if inputs_embeds is None:
            inputs_embeds = self._embed_realtime(
                input_ids,
                first_hidden_state=first_hidden_state,
                codebook_index=codebook_index,
            )
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.hidden_size:
            raise ValueError("Local Qwen inputs have an invalid shape.")
        batch_size, query_length, _ = inputs_embeds.shape
        if past_key_values is not None and not isinstance(past_key_values, DynamicKVCache):
            raise TypeError("Local Qwen cache must be a DynamicKVCache.")
        if use_cache and past_key_values is None:
            past_key_values = DynamicKVCache()
        past_length = (0 if past_key_values is None else past_key_values.sequence_length())
        if position_ids is None:
            if apply_rope:
                position_ids = torch.arange(
                    past_length,
                    past_length + query_length,
                    device=inputs_embeds.device,
                ).unsqueeze(0)
            else:
                # Applying RoPE at position zero is the exact identity, which
                # represents the older Local graph's position-free attention.
                position_ids = torch.zeros(
                    1,
                    query_length,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
        if position_ids.shape[0] == 1 and batch_size != 1:
            position_ids = position_ids.expand(batch_size, -1)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states, _, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=past_key_values,
                use_cache=use_cache,
                output_attentions=False,
            )
        return LocalTransformerOutput(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=past_key_values if use_cache else None,
        )


__all__ = [
    "LocalTransformerOutput",
    "MossGPT2Model",
    "MossQwenDepthModel",
]
