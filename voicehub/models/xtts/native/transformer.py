"""GPT-2 blocks with the exact XTTS checkpoint namespace and tensor layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Conv1D(nn.Module):
    """Hugging Face GPT-2's historical transposed linear parameter layout."""

    def __init__(self, output_size: int, input_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_size, output_size))
        self.bias = nn.Parameter(torch.zeros(output_size))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, value: Tensor) -> Tensor:
        return torch.addmm(
            self.bias,
            value.reshape(-1, value.shape[-1]),
            self.weight,
        ).view(*value.shape[:-1], self.bias.numel())


class GPT2Attention(nn.Module):

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("Transformer width must divide evenly into heads.")
        self.num_heads = heads
        self.head_dim = width // heads
        self.c_attn = Conv1D(width * 3, width)
        self.c_proj = Conv1D(width, width)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        query, key, value = self.c_attn(hidden_states).chunk(3, dim=-1)
        query = self._split(query)
        key = self._split(key)
        value = self._split(value)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)
        present = (key, value) if use_cache else None
        scores = query @ key.transpose(-1, -2)
        scores = scores / self.head_dim**0.5
        query_count = query.shape[-2]
        key_count = key.shape[-2]
        past_count = key_count - query_count
        query_positions = torch.arange(
            past_count,
            past_count + query_count,
            device=query.device,
        )[:, None]
        key_positions = torch.arange(key_count, device=query.device)[None, :]
        scores = scores.masked_fill(
            key_positions > query_positions,
            torch.finfo(scores.dtype).min,
        )
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :key_count].to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        probabilities = self.attn_dropout(probabilities)
        attended = probabilities @ value
        attended = attended.transpose(1, 2).contiguous().view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            -1,
        )
        return self.resid_dropout(self.c_proj(attended)), present

    def _split(self, value: Tensor) -> Tensor:
        return value.view(*value.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)


class GPT2MLP(nn.Module):

    def __init__(self, width: int) -> None:
        super().__init__()
        self.c_fc = Conv1D(width * 4, width)
        self.c_proj = Conv1D(width, width * 4)
        self.dropout = nn.Dropout(0.1)

    def forward(self, value: Tensor) -> Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(value), approximate="tanh")))


class GPT2Block(nn.Module):

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width, eps=1e-5)
        self.attn = GPT2Attention(width, heads)
        self.ln_2 = nn.LayerNorm(width, eps=1e-5)
        self.mlp = GPT2MLP(width)

    def forward(
        self,
        value: Tensor,
        *,
        attention_mask: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        attended, present = self.attn(
            self.ln_1(value),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        value = value + attended
        return value + self.mlp(self.ln_2(value)), present


@dataclass(slots=True)
class TransformerOutput:
    last_hidden_state: Tensor
    past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None


class GPT2Model(nn.Module):
    """Embedding-free GPT-2 body; key names match ``gpt.gpt.h.*``."""

    def __init__(self, layers: int, width: int, heads: int) -> None:
        super().__init__()
        self.drop = nn.Dropout(0.1)
        self.h = nn.ModuleList([GPT2Block(width, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(width, eps=1e-5)

    def forward(
        self,
        inputs_embeds: Tensor,
        *,
        attention_mask: Tensor | None = None,
        past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None,
        use_cache: bool = False,
    ) -> TransformerOutput:
        value = self.drop(inputs_embeds)
        presents = []
        for index, block in enumerate(self.h):
            past = None if past_key_values is None else past_key_values[index]
            value, present = block(
                value,
                attention_mask=attention_mask,
                past_key_value=past,
                use_cache=use_cache,
            )
            if present is not None:
                presents.append(present)
        return TransformerOutput(
            last_hidden_state=self.ln_f(value),
            past_key_values=tuple(presents) if use_cache else None,
        )


__all__ = ["GPT2Model", "TransformerOutput"]
