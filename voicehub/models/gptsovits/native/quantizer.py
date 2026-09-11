"""Native frozen residual vector quantizer for GPT-SoVITS classic S2."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional


class EuclideanCodebook(nn.Module):

    def __init__(self, dimension: int, codebook_size: int) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        embedding = torch.empty(codebook_size, dimension)
        nn.init.kaiming_uniform_(embedding)
        self.register_buffer("inited", torch.tensor([False], dtype=torch.float32))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embedding)
        self.register_buffer("embed_avg", embedding.clone())

    def _require_initialized(self) -> None:
        if not bool(self.inited.item()):
            raise RuntimeError(
                "The native GPT-SoVITS quantizer requires a loaded initialized "
                "codebook. Fresh k-means initialization is intentionally not "
                "reconstructed from an arbitrary batch.")

    def encode(self, inputs: Tensor) -> Tensor:
        self._require_initialized()
        flattened = inputs.reshape(-1, inputs.shape[-1])
        transposed = self.embed.t()
        distance = -(
            flattened.square().sum(1, keepdim=True) - 2 * flattened @ transposed +
            transposed.square().sum(0, keepdim=True))
        return distance.argmax(dim=-1).view(*inputs.shape[:-1])

    def decode(self, indices: Tensor) -> Tensor:
        self._require_initialized()
        return functional.embedding(indices, self.embed)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        indices = self.encode(inputs)
        return self.decode(indices), indices


class VectorQuantization(nn.Module):

    def __init__(self, dimension: int, codebook_size: int) -> None:
        super().__init__()
        self.project_in = nn.Identity()
        self.project_out = nn.Identity()
        self._codebook = EuclideanCodebook(dimension, codebook_size)

    def encode(self, inputs: Tensor) -> Tensor:
        return self._codebook.encode(inputs.transpose(1, 2))

    def decode(self, indices: Tensor) -> Tensor:
        return self._codebook.decode(indices).transpose(1, 2)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        channels_last = inputs.transpose(1, 2)
        quantized, indices = self._codebook(channels_last)
        if self.training:
            quantized = channels_last + (quantized - channels_last).detach()
            loss = functional.mse_loss(quantized.detach(), channels_last)
        else:
            loss = inputs.new_zeros(1)
        return self.project_out(quantized).transpose(1, 2), indices, loss


class ResidualVectorQuantization(nn.Module):

    def __init__(
        self,
        *,
        dimension: int,
        codebook_size: int,
        quantizers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(dimension, codebook_size) for _ in range(quantizers)])

    def forward(
        self,
        inputs: Tensor,
        *,
        quantizers: int,
        layers: list[int] | None,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        quantized_output: Tensor | float = 0.0
        residual = inputs
        losses = []
        indices = []
        selected = []
        for index, layer in enumerate(self.layers[:quantizers]):
            quantized, code_indices, loss = layer(residual)
            residual = residual - quantized
            quantized_output = quantized_output + quantized
            losses.append(loss)
            indices.append(code_indices)
            if layers and index in layers:
                selected.append(quantized)
        return (
            quantized_output,
            torch.stack(indices),
            torch.stack(losses),
            selected,
        )

    def decode(self, indices: Tensor, start: int = 0) -> Tensor:
        quantized: Tensor | float = 0.0
        for offset, layer_indices in enumerate(indices):
            quantized = quantized + self.layers[start + offset].decode(layer_indices)
        if not isinstance(quantized, Tensor):
            raise ValueError("At least one quantizer code tensor is required.")
        return quantized


class ResidualVectorQuantizer(nn.Module):
    """One-layer 768-dimensional, 1,024-entry public classic-S2 codebook."""

    def __init__(
        self,
        *,
        dimension: int = 768,
        quantizers: int = 1,
        bins: int = 1_024,
    ) -> None:
        super().__init__()
        self.n_q = quantizers
        self.dimension = dimension
        self.bins = bins
        self.vq = ResidualVectorQuantization(
            dimension=dimension,
            codebook_size=bins,
            quantizers=quantizers,
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        n_q: int | None = None,
        layers: list[int] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        quantizer_count = self.n_q if n_q is None else n_q
        if quantizer_count < 1 or quantizer_count > self.n_q:
            raise ValueError("Invalid GPT-SoVITS quantizer count.")
        quantized, codes, losses, selected = self.vq(
            inputs,
            quantizers=quantizer_count,
            layers=layers,
        )
        return quantized, codes, losses.mean(), selected

    def decode(self, codes: Any, start: int = 0) -> Tensor:
        return self.vq.decode(torch.as_tensor(codes), start=start)


__all__ = ["ResidualVectorQuantizer"]
