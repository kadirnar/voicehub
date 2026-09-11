"""Checkpoint-compatible Llama 3.2 decoder used by ConversationTTS.

The public ConversationTTS checkpoint was trained with TorchTune's Llama
3.2 components.  This module owns the exact subset of that graph
required by VoiceHub: pre-normalized decoder layers, grouped-query
attention, Llama 3 scaled RoPE, SwiGLU feed-forward blocks, and fixed-
size inference caches.

Parameter names deliberately match the released checkpoint (for example
``layers.0.attn.q_proj.weight``, ``layers.0.mlp.w1.weight``, and
``norm.scale``).  Cache tensors and RoPE tables are non-persistent
buffers, so serving state never leaks into a fine-tuning artifact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.kernels import KernelBackend, gated_silu
from voicehub.neural.backends import FlashAttention4Policy, flash_attention4_or_sdpa


class ConversationRMSNorm(nn.Module):
    """RMS normalization with the parameter namespace used by the
    checkpoint."""

    def __init__(self, dimension: int, *, epsilon: float = 1e-5) -> None:
        super().__init__()
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("`dimension` must be a positive integer.")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("`epsilon` must be finite and positive.")
        self.normalized_shape = (dimension, )
        self.epsilon = float(epsilon)
        self.scale = nn.Parameter(torch.ones(dimension))

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = functional.rms_norm(
            inputs.float(),
            normalized_shape=self.normalized_shape,
            weight=self.scale.float(),
            eps=self.epsilon,
        )
        return normalized.to(dtype=inputs.dtype)


class ConversationFeedForward(nn.Module):
    """Llama SwiGLU block with TorchTune-compatible state-dict names."""

    def __init__(self, dimension: int, intermediate_dimension: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dimension, intermediate_dimension, bias=False)
        self.w2 = nn.Linear(intermediate_dimension, dimension, bias=False)
        self.w3 = nn.Linear(dimension, intermediate_dimension, bias=False)
        self.kernel_backend = KernelBackend.TORCH

    def set_kernel_backend(self, backend: KernelBackend | str) -> None:
        """Select the SwiGLU implementation without changing parameters."""
        self.kernel_backend = KernelBackend.coerce(backend)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.w2(gated_silu(
            self.w1(inputs),
            self.w3(inputs),
            backend=self.kernel_backend,
        ))


class Llama3ScaledRotaryEmbedding(nn.Module):
    """Llama 3.1/3.2 rotary embedding with the released scaling constants."""

    def __init__(
        self,
        dimension: int,
        *,
        maximum_sequence_length: int,
        base: float = 500_000.0,
        scale_factor: float = 32.0,
        low_frequency_factor: float = 1.0,
        high_frequency_factor: float = 4.0,
        original_context_length: int = 8192,
    ) -> None:
        super().__init__()
        if dimension <= 0 or dimension % 2:
            raise ValueError("Rotary `dimension` must be a positive even integer.")
        if maximum_sequence_length <= 0:
            raise ValueError("`maximum_sequence_length` must be positive.")
        if base <= 0 or scale_factor <= 0:
            raise ValueError("Rotary base and scale factor must be positive.")
        if not 0 < low_frequency_factor < high_frequency_factor:
            raise ValueError(
                "`low_frequency_factor` must be positive and lower than "
                "`high_frequency_factor`.")
        if original_context_length <= 0:
            raise ValueError("`original_context_length` must be positive.")

        frequencies = 1.0 / (base**(torch.arange(0, dimension, 2, dtype=torch.float32) / dimension))
        low_wavelength = original_context_length / low_frequency_factor
        high_wavelength = original_context_length / high_frequency_factor
        wavelengths = (2.0 * math.pi) / frequencies
        scaled = frequencies / scale_factor
        smooth = (original_context_length / wavelengths -
                  low_frequency_factor) / (high_frequency_factor - low_frequency_factor)
        medium = (1.0 - smooth) * scaled + smooth * frequencies
        theta = torch.where(
            wavelengths < high_wavelength,
            frequencies,
            torch.where(wavelengths > low_wavelength, scaled, medium),
        )
        positions = torch.arange(maximum_sequence_length, dtype=torch.float32)
        angles = torch.outer(positions, theta)
        cache = torch.stack((angles.cos(), angles.sin()), dim=-1)
        self.dimension = dimension
        self.maximum_sequence_length = maximum_sequence_length
        self.register_buffer("theta", theta, persistent=False)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self,
        inputs: Tensor,
        *,
        input_positions: Tensor | None = None,
    ) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[-1] != self.dimension:
            raise ValueError("Rotary inputs must have shape [batch, time, heads, dimension].")
        sequence_length = inputs.shape[1]
        if input_positions is None:
            if sequence_length > self.maximum_sequence_length:
                raise ValueError("Input sequence exceeds the rotary cache length.")
            rotation = self.cache[:sequence_length].unsqueeze(0)
        else:
            if not isinstance(input_positions, Tensor):
                raise TypeError("`input_positions` must be a PyTorch tensor.")
            if (input_positions.dtype == torch.bool or input_positions.is_floating_point() or
                    input_positions.is_complex()):
                raise TypeError("`input_positions` must use an integer dtype.")
            if input_positions.ndim == 1:
                input_positions = input_positions.unsqueeze(0)
            if (input_positions.ndim != 2 or input_positions.shape[-1] != sequence_length or
                    input_positions.shape[0] not in (1, inputs.shape[0])):
                raise ValueError("`input_positions` must have shape [1|batch, time].")
            if input_positions.numel():
                minimum = int(input_positions.min().item())
                maximum = int(input_positions.max().item())
                if minimum < 0 or maximum >= self.maximum_sequence_length:
                    raise ValueError("`input_positions` exceed the rotary cache.")
            rotation = self.cache[input_positions.to(device=self.cache.device)]

        working = inputs.float().reshape(*inputs.shape[:-1], -1, 2)
        rotation = rotation.to(device=inputs.device).view(
            rotation.shape[0],
            sequence_length,
            1,
            working.shape[-2],
            2,
        )
        real = working[..., 0]
        imaginary = working[..., 1]
        cosine = rotation[..., 0]
        sine = rotation[..., 1]
        output = torch.stack(
            (
                real * cosine - imaginary * sine,
                imaginary * cosine + real * sine,
            ),
            dim=-1,
        )
        return output.flatten(-2).to(dtype=inputs.dtype)


class ConversationKVCache(nn.Module):
    """Fixed-size per-layer cache excluded from checkpoint state."""

    def __init__(
        self,
        *,
        batch_size: int,
        maximum_sequence_length: int,
        number_of_heads: int,
        head_dimension: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if min(
                batch_size,
                maximum_sequence_length,
                number_of_heads,
                head_dimension,
        ) <= 0:
            raise ValueError("Cache dimensions must be positive.")
        shape = (
            batch_size,
            number_of_heads,
            maximum_sequence_length,
            head_dimension,
        )
        self.register_buffer(
            "k_cache",
            torch.zeros(shape, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(shape, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "cache_pos",
            torch.arange(maximum_sequence_length, device=device),
            persistent=False,
        )

    @property
    def size(self) -> int:
        return int(self.cache_pos[0].item())

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos.sub_(self.size)

    def update(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise ValueError("Cache key/value inputs must be equal rank-four tensors.")
        if key.shape[0] > self.k_cache.shape[0]:
            raise ValueError("Cache input batch exceeds the configured batch size.")
        if key.shape[1:] != (
                self.k_cache.shape[1],
                key.shape[2],
                self.k_cache.shape[3],
        ):
            raise ValueError("Cache input head dimensions are incompatible.")
        sequence_length = key.shape[2]
        if self.size + sequence_length > self.k_cache.shape[2]:
            raise ValueError("Cache update exceeds the configured sequence length.")
        positions = self.cache_pos[:sequence_length]
        self.k_cache[:key.shape[0], :, positions] = key
        self.v_cache[:value.shape[0], :, positions] = value
        self.cache_pos.add_(sequence_length)
        return (
            self.k_cache[:key.shape[0]],
            self.v_cache[:value.shape[0]],
        )


class ConversationMultiHeadAttention(nn.Module):
    """Grouped-query self-attention with an optional fixed-size cache."""

    def __init__(
        self,
        *,
        embedding_dimension: int,
        number_of_heads: int,
        number_of_kv_heads: int,
        maximum_sequence_length: int,
        rotary_embedding: Llama3ScaledRotaryEmbedding,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dimension % number_of_heads:
            raise ValueError("Embedding dimension must be divisible by query heads.")
        if number_of_heads % number_of_kv_heads:
            raise ValueError("Query heads must be divisible by key/value heads.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("Attention dropout must be between zero and one.")
        self.embed_dim = embedding_dimension
        self.num_heads = number_of_heads
        self.num_kv_heads = number_of_kv_heads
        self.head_dim = embedding_dimension // number_of_heads
        self.max_seq_len = maximum_sequence_length
        self.attn_dropout = float(dropout)
        self.q_proj = nn.Linear(
            embedding_dimension,
            number_of_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            embedding_dimension,
            number_of_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            embedding_dimension,
            number_of_kv_heads * self.head_dim,
            bias=False,
        )
        self.output_proj = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False,
        )
        self.pos_embeddings = rotary_embedding
        self.kv_cache: ConversationKVCache | None = None
        self.cache_enabled = False
        self.flash_attention4_policy = FlashAttention4Policy.DISABLED

    def set_flash_attention4_policy(
        self,
        policy: FlashAttention4Policy | str,
    ) -> None:
        """Select FA4 or SDPA without altering checkpoint parameters."""
        self.flash_attention4_policy = FlashAttention4Policy.coerce(policy)

    def setup_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        maximum_sequence_length: int,
    ) -> None:
        if self.kv_cache is not None:
            raise RuntimeError("Attention cache is already initialized.")
        self.kv_cache = ConversationKVCache(
            batch_size=batch_size,
            maximum_sequence_length=maximum_sequence_length,
            number_of_heads=self.num_kv_heads,
            head_dimension=self.head_dim,
            dtype=dtype,
            device=self.q_proj.weight.device,
        )
        self.cache_enabled = True

    def reset_cache(self) -> None:
        if self.kv_cache is None:
            raise RuntimeError("Attention cache is not initialized.")
        self.kv_cache.reset()

    @staticmethod
    def _normalize_mask(
        mask: Tensor | None,
        *,
        batch_size: int,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> Tensor | None:
        if mask is None:
            return None
        if not isinstance(mask, Tensor):
            raise TypeError("Attention `mask` must be a PyTorch tensor.")
        mask = mask.to(device=device)
        if mask.ndim == 2:
            if tuple(mask.shape) != (query_length, key_length):
                raise ValueError("Rank-two attention mask has an invalid shape.")
            mask = mask.view(1, 1, query_length, key_length)
        elif mask.ndim == 3:
            if (mask.shape[0] not in (1, batch_size) or tuple(mask.shape[1:]) != (query_length, key_length)):
                raise ValueError("Rank-three attention mask has an invalid shape.")
            mask = mask.unsqueeze(1)
        elif mask.ndim == 4:
            if (mask.shape[0] not in (1, batch_size) or mask.shape[1] not in (1, ) or
                    tuple(mask.shape[2:]) != (query_length, key_length)):
                raise ValueError("Rank-four attention mask has an invalid shape.")
        else:
            raise ValueError("Attention mask must have rank two, three, or four.")
        return mask

    def forward(
        self,
        inputs: Tensor,
        values: Tensor | None = None,
        *,
        mask: Tensor | None = None,
        input_pos: Tensor | None = None,
    ) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.embed_dim:
            raise ValueError("Attention inputs must have shape [batch, time, embedding].")
        values = inputs if values is None else values
        if values.ndim != 3 or values.shape[0] != inputs.shape[0]:
            raise ValueError("Attention key/value inputs have an invalid shape.")
        batch_size, query_length, _ = inputs.shape
        value_length = values.shape[1]

        query = self.q_proj(inputs).view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        )
        key = self.k_proj(values).view(
            batch_size,
            value_length,
            self.num_kv_heads,
            self.head_dim,
        )
        value = self.v_proj(values).view(
            batch_size,
            value_length,
            self.num_kv_heads,
            self.head_dim,
        )
        query = self.pos_embeddings(
            query,
            input_positions=input_pos,
        ).transpose(1, 2)
        key = self.pos_embeddings(
            key,
            input_positions=input_pos,
        ).transpose(1, 2)
        value = value.transpose(1, 2)

        if self.kv_cache is not None and self.cache_enabled:
            key, value = self.kv_cache.update(key, value)
        key_length = key.shape[2]

        attention_mask = self._normalize_mask(
            mask,
            batch_size=batch_size,
            query_length=query_length,
            key_length=key_length,
            device=inputs.device,
        )
        attended = flash_attention4_or_sdpa(
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=(attention_mask is None and self.kv_cache is None),
            policy=self.flash_attention4_policy,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            query_length,
            self.embed_dim,
        )
        return self.output_proj(attended)


class ConversationDecoderLayer(nn.Module):
    """Pre-normalized residual decoder layer."""

    def __init__(
        self,
        attention: ConversationMultiHeadAttention,
        feed_forward: ConversationFeedForward,
        *,
        embedding_dimension: int,
        normalization_epsilon: float,
    ) -> None:
        super().__init__()
        self.attn = attention
        self.mlp = feed_forward
        self.sa_norm = ConversationRMSNorm(
            embedding_dimension,
            epsilon=normalization_epsilon,
        )
        self.mlp_norm = ConversationRMSNorm(
            embedding_dimension,
            epsilon=normalization_epsilon,
        )
        self.sa_scale = nn.Identity()
        self.mlp_scale = nn.Identity()

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        decoder_max_seq_len: int,
    ) -> None:
        self.attn.setup_cache(
            batch_size,
            dtype,
            decoder_max_seq_len,
        )

    def caches_are_setup(self) -> bool:
        return self.attn.kv_cache is not None

    def caches_are_enabled(self) -> bool:
        return self.attn.cache_enabled

    def reset_cache(self) -> None:
        self.attn.reset_cache()

    def forward(
        self,
        inputs: Tensor,
        *,
        mask: Tensor | None = None,
        input_pos: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        normalized = self.sa_norm(inputs)
        hidden = inputs + self.sa_scale(self.attn(
            normalized,
            normalized,
            mask=mask,
            input_pos=input_pos,
        ))
        return hidden + self.mlp_scale(self.mlp(self.mlp_norm(hidden)))


class ConversationDecoder(nn.Module):
    """Transformer decoder matching the released ConversationTTS namespace."""

    def __init__(
        self,
        *,
        token_embeddings: nn.Module,
        layers: Sequence[ConversationDecoderLayer],
        maximum_sequence_length: int,
        number_of_heads: int,
        head_dimension: int,
        normalization: nn.Module,
        output: nn.Module,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("Conversation decoder requires at least one layer.")
        self.tok_embeddings = token_embeddings
        self.layers = nn.ModuleList(layers)
        self.norm = normalization
        self.output = output
        self.max_seq_len = maximum_sequence_length
        self.num_heads = number_of_heads
        self.head_dim = head_dimension
        self.embedding_dimension = number_of_heads * head_dimension
        self.output_hidden_states: list[int] = []
        self.num_output_chunks = 0
        self.encoder_max_cache_seq_len = None
        self.decoder_max_cache_seq_len: int | None = None

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        decoder_max_seq_len: int | None = None,
        **_: Any,
    ) -> None:
        if self.caches_are_setup():
            raise RuntimeError("Decoder caches are already initialized.")
        maximum_length = (self.max_seq_len if decoder_max_seq_len is None else decoder_max_seq_len)
        if maximum_length <= 0 or maximum_length > self.max_seq_len:
            raise ValueError("Decoder cache length is outside the supported range.")
        self.decoder_max_cache_seq_len = maximum_length
        for layer in self.layers:
            layer.setup_caches(
                batch_size,
                dtype,
                decoder_max_seq_len=maximum_length,
            )

    def caches_are_setup(self) -> bool:
        return self.layers[0].caches_are_setup()

    def caches_are_enabled(self) -> bool:
        return self.layers[0].caches_are_enabled()

    def reset_caches(self) -> None:
        if not self.caches_are_enabled():
            raise RuntimeError("Decoder caches are not initialized.")
        for layer in self.layers:
            layer.reset_cache()

    def forward(
        self,
        inputs: Tensor,
        *,
        mask: Tensor | None = None,
        input_pos: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        if inputs.ndim < 2 or inputs.shape[1] > self.max_seq_len:
            raise ValueError("Decoder input sequence exceeds `max_seq_len`.")
        if self.caches_are_enabled() and (mask is None or input_pos is None):
            raise ValueError("Cached decoding requires both `mask` and `input_pos`.")
        hidden = self.tok_embeddings(inputs)
        for layer in self.layers:
            hidden = layer(
                hidden,
                mask=mask,
                input_pos=input_pos,
            )
        return self.output(self.norm(hidden)).float()


def build_llama32_decoder(
    *,
    vocabulary_size: int,
    number_of_layers: int,
    number_of_heads: int,
    number_of_kv_heads: int,
    embedding_dimension: int,
    maximum_sequence_length: int,
    intermediate_dimension: int,
    attention_dropout: float = 0.0,
    normalization_epsilon: float = 1e-5,
    rope_base: float = 500_000.0,
    rope_scale_factor: float = 32.0,
) -> ConversationDecoder:
    """Build the exact Llama 3.2 subset used by ConversationTTS."""
    integer_values = {
        "vocabulary_size": vocabulary_size,
        "number_of_layers": number_of_layers,
        "number_of_heads": number_of_heads,
        "number_of_kv_heads": number_of_kv_heads,
        "embedding_dimension": embedding_dimension,
        "maximum_sequence_length": maximum_sequence_length,
        "intermediate_dimension": intermediate_dimension,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"`{name}` must be a positive integer.")
    if embedding_dimension % number_of_heads:
        raise ValueError("Embedding dimension must be divisible by query heads.")
    rotary_embedding = Llama3ScaledRotaryEmbedding(
        embedding_dimension // number_of_heads,
        maximum_sequence_length=maximum_sequence_length,
        base=rope_base,
        scale_factor=rope_scale_factor,
    )
    layers = [
        ConversationDecoderLayer(
            ConversationMultiHeadAttention(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                number_of_kv_heads=number_of_kv_heads,
                maximum_sequence_length=maximum_sequence_length,
                rotary_embedding=rotary_embedding,
                dropout=attention_dropout,
            ),
            ConversationFeedForward(
                embedding_dimension,
                intermediate_dimension,
            ),
            embedding_dimension=embedding_dimension,
            normalization_epsilon=normalization_epsilon,
        ) for _ in range(number_of_layers)
    ]
    return ConversationDecoder(
        token_embeddings=nn.Embedding(vocabulary_size, embedding_dimension),
        layers=layers,
        maximum_sequence_length=maximum_sequence_length,
        number_of_heads=number_of_heads,
        head_dimension=embedding_dimension // number_of_heads,
        normalization=ConversationRMSNorm(
            embedding_dimension,
            epsilon=normalization_epsilon,
        ),
        output=nn.Linear(embedding_dimension, vocabulary_size, bias=False),
    )


__all__ = [
    "ConversationDecoder",
    "ConversationDecoderLayer",
    "ConversationFeedForward",
    "ConversationKVCache",
    "ConversationMultiHeadAttention",
    "ConversationRMSNorm",
    "Llama3ScaledRotaryEmbedding",
    "build_llama32_decoder",
]
