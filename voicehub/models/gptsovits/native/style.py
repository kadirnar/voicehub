"""GPT-SoVITS reference-style and multi-reference text encoders."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.inflecttts.native.attentions import MultiHeadAttention as VITSMultiHeadAttention


class LinearNorm(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_channels, output_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.fc(inputs)


class Mish(nn.Module):

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * torch.tanh(functional.softplus(inputs))


class ConvNorm(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("GPT-SoVITS style kernels must be odd.")
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class Conv1dGLU(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.conv1 = ConvNorm(
            input_channels,
            output_channels * 2,
            kernel_size=kernel_size,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        first, second = self.conv1(inputs).split(self.output_channels, dim=1)
        return inputs + self.dropout(first * torch.sigmoid(second))


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature: float, dropout: float) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        attention = torch.bmm(query, key.transpose(1, 2)) / self.temperature
        if mask is not None:
            attention = attention.masked_fill(mask, -torch.inf)
        attention = functional.softmax(attention, dim=2)
        return torch.bmm(self.dropout(attention), value), attention


class StyleMultiHeadAttention(nn.Module):

    def __init__(
        self,
        heads: int,
        model_dim: int,
        key_dim: int,
        value_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_head = heads
        self.d_k = key_dim
        self.d_v = value_dim
        self.w_qs = nn.Linear(model_dim, heads * key_dim)
        self.w_ks = nn.Linear(model_dim, heads * key_dim)
        self.w_vs = nn.Linear(model_dim, heads * value_dim)
        self.attention = ScaledDotProductAttention(
            math.sqrt(model_dim),
            dropout,
        )
        self.fc = nn.Linear(heads * value_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, steps, _ = inputs.shape
        residual = inputs
        query = self.w_qs(inputs).view(batch, steps, self.n_head, self.d_k)
        key = self.w_ks(inputs).view(batch, steps, self.n_head, self.d_k)
        value = self.w_vs(inputs).view(batch, steps, self.n_head, self.d_v)
        query = query.permute(2, 0, 1, 3).reshape(-1, steps, self.d_k)
        key = key.permute(2, 0, 1, 3).reshape(-1, steps, self.d_k)
        value = value.permute(2, 0, 1, 3).reshape(-1, steps, self.d_v)
        repeated_mask = None if mask is None else mask.repeat(self.n_head, 1, 1)
        output, attention = self.attention(
            query,
            key,
            value,
            repeated_mask,
        )
        output = output.view(self.n_head, batch, steps, self.d_v)
        output = output.permute(1, 2, 0, 3).reshape(batch, steps, -1)
        return self.dropout(self.fc(output)) + residual, attention


class MelStyleEncoder(nn.Module):
    """Exact classic-S2 reference encoder with variant-specific bins."""

    def __init__(
        self,
        mel_channels: int = 704,
        *,
        style_hidden: int = 128,
        style_vector_dim: int = 512,
        style_kernel_size: int = 5,
        style_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spectral = nn.Sequential(
            LinearNorm(mel_channels, style_hidden),
            Mish(),
            nn.Dropout(dropout),
            LinearNorm(style_hidden, style_hidden),
            Mish(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.Sequential(
            Conv1dGLU(
                style_hidden,
                style_hidden,
                style_kernel_size,
                dropout,
            ),
            Conv1dGLU(
                style_hidden,
                style_hidden,
                style_kernel_size,
                dropout,
            ),
        )
        self.slf_attn = StyleMultiHeadAttention(
            style_heads,
            style_hidden,
            style_hidden // style_heads,
            style_hidden // style_heads,
            dropout,
        )
        self.fc = LinearNorm(style_hidden, style_vector_dim)

    def forward(
        self,
        spectrogram: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        hidden = spectrogram.transpose(1, 2)
        padding_mask = None
        if mask is not None:
            padding_mask = (mask.to(torch.int32) == 0).squeeze(1)
        steps = hidden.shape[1]
        attention_mask = (None if padding_mask is None else padding_mask.unsqueeze(1).expand(-1, steps, -1))
        hidden = self.spectral(hidden)
        hidden = self.temporal(hidden.transpose(1, 2)).transpose(1, 2)
        if padding_mask is not None:
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0)
        hidden, _ = self.slf_attn(hidden, mask=attention_mask)
        hidden = self.fc(hidden)
        if padding_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            lengths = (~padding_mask).sum(dim=1).unsqueeze(1)
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0)
            dtype = hidden.dtype
            pooled = (hidden.float() / lengths.unsqueeze(1)).sum(dim=1).to(dtype)
        return pooled.unsqueeze(-1)


class MRTE(nn.Module):
    """Multi-reference timbre encoder used by public classic-S2 graphs."""

    def __init__(self) -> None:
        super().__init__()
        self.cross_attention = VITSMultiHeadAttention(512, 512, 4)
        self.c_pre = nn.Conv1d(192, 512, 1)
        self.text_pre = nn.Conv1d(192, 512, 1)
        self.c_post = nn.Conv1d(512, 192, 1)

    def forward(
        self,
        ssl_hidden: Tensor,
        ssl_mask: Tensor,
        text_hidden: Tensor,
        text_mask: Tensor,
        style: Tensor | None,
    ) -> Tensor:
        condition: Tensor | int = 0 if style is None else style
        attention_mask = text_mask.unsqueeze(2) * ssl_mask.unsqueeze(-1)
        ssl_hidden = self.c_pre(ssl_hidden * ssl_mask)
        text_hidden = self.text_pre(text_hidden * text_mask)
        attended = self.cross_attention(
            ssl_hidden * ssl_mask,
            text_hidden * text_mask,
            attention_mask,
        )
        return self.c_post((attended + ssl_hidden + condition) * ssl_mask)


__all__ = ["MRTE", "MelStyleEncoder"]
