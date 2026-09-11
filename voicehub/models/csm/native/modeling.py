"""VoiceHub-owned PyTorch implementation of Sesame CSM.

The persistent tensor namespace matches the original ``sesame/csm-1b``
Safetensors release.  The decoder math follows the CSM source graph and
the Llama-3.2 components used by its pinned torchtune 0.4.0 dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.csm.native.configuration import CSMArchitectureConfig, CSMTransformerConfig
from voicehub.optimization.protocols import OptimizationCompileTarget


class CSMRMSNorm(nn.Module):
    """Source-compatible RMSNorm with an exact ``scale`` parameter name."""

    def __init__(
        self,
        dimension: int,
        *,
        epsilon: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(
            dimension,
            device=device,
            dtype=dtype,
        ), )

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return normalized.to(dtype=inputs.dtype) * self.scale


class CSMLlama3ScaledRoPE(nn.Module):
    """The Llama-3 scaling rule used by torchtune 0.4.0."""

    def __init__(
        self,
        config: CSMTransformerConfig,
        *,
        device=None,
    ) -> None:
        super().__init__()
        self.dimension = config.head_dim
        self.max_sequence_length = config.max_sequence_length
        resolved_device = torch.device("cpu" if device is None else device)
        if resolved_device.type == "meta":
            theta = torch.empty(
                self.dimension // 2,
                device=resolved_device,
            )
            cache = torch.empty(
                self.max_sequence_length,
                self.dimension // 2,
                2,
                device=resolved_device,
            )
        else:
            frequencies = 1.0 / (
                config.rope_theta**(
                    torch.arange(
                        0,
                        self.dimension,
                        2,
                        device=resolved_device,
                        dtype=torch.float32,
                    ) / self.dimension))
            theta = self._scale_frequencies(
                frequencies,
                scale_factor=config.rope_scale_factor,
                low_frequency_factor=config.rope_low_frequency_factor,
                high_frequency_factor=config.rope_high_frequency_factor,
                original_context_length=(config.rope_original_context_length),
            )
            positions = torch.arange(
                self.max_sequence_length,
                device=resolved_device,
                dtype=theta.dtype,
            )
            angles = torch.einsum("i,j->ij", positions, theta).float()
            cache = torch.stack(
                (angles.cos(), angles.sin()),
                dim=-1,
            )
        self.register_buffer("theta", theta, persistent=False)
        self.register_buffer("cache", cache, persistent=False)

    @staticmethod
    def _scale_frequencies(
        frequencies: Tensor,
        *,
        scale_factor: float,
        low_frequency_factor: float,
        high_frequency_factor: float,
        original_context_length: int,
    ) -> Tensor:
        low_wavelength = original_context_length / low_frequency_factor
        high_wavelength = original_context_length / high_frequency_factor
        values = []
        for frequency in frequencies:
            wavelength = 2.0 * math.pi / float(frequency)
            if wavelength < high_wavelength:
                value = frequency
            elif wavelength > low_wavelength:
                value = frequency / scale_factor
            else:
                smooth = (original_context_length / wavelength -
                          low_frequency_factor) / (high_frequency_factor - low_frequency_factor)
                value = ((1.0 - smooth) * frequency / scale_factor + smooth * frequency)
            values.append(value)
        return torch.stack(values).to(
            device=frequencies.device,
            dtype=frequencies.dtype,
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        input_positions: Tensor | None = None,
    ) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[-1] != self.dimension:
            raise ValueError("RoPE inputs must have shape [batch, time, heads, head_dim].")
        sequence_length = inputs.shape[1]
        if input_positions is None:
            positional = self.cache[:sequence_length]
        else:
            if (input_positions.ndim != 2 or
                    tuple(input_positions.shape) != (inputs.shape[0], sequence_length)):
                raise ValueError("`input_positions` must have shape [batch, time].")
            if input_positions.numel() and (int(input_positions.min()) < 0 or
                                            int(input_positions.max()) >= self.max_sequence_length):
                raise ValueError("CSM position IDs exceed the configured context.")
            positional = self.cache[input_positions]
        paired = inputs.float().reshape(*inputs.shape[:-1], -1, 2)
        positional = positional.reshape(
            -1,
            sequence_length,
            1,
            paired.shape[-2],
            2,
        )
        rotated = torch.stack(
            (
                paired[..., 0] * positional[..., 0] - paired[..., 1] * positional[..., 1],
                paired[..., 1] * positional[..., 0] + paired[..., 0] * positional[..., 1],
            ),
            dim=-1,
        )
        return rotated.flatten(3).to(dtype=inputs.dtype)


class CSMKVCache(nn.Module):
    """Fixed-size source-compatible cache, excluded from checkpoints."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_heads: int,
        max_sequence_length: int,
        head_dim: int,
        device,
        dtype,
    ) -> None:
        super().__init__()
        shape = (
            batch_size,
            num_heads,
            max_sequence_length,
            head_dim,
        )
        self.register_buffer(
            "keys",
            torch.zeros(shape, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "values",
            torch.zeros(shape, device=device, dtype=dtype),
            persistent=False,
        )
        self.position = 0

    def reset(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.position = 0

    def update(
        self,
        keys: Tensor,
        values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, _, length, _ = keys.shape
        if batch_size > self.keys.shape[0]:
            raise ValueError("CSM cache batch size was exceeded.")
        end = self.position + length
        if end > self.keys.shape[2]:
            raise ValueError("CSM cache sequence length was exceeded.")
        self.keys[:batch_size, :, self.position:end].copy_(keys)
        self.values[:batch_size, :, self.position:end].copy_(values)
        self.position = end
        return self.keys[:batch_size], self.values[:batch_size]


class CSMAttention(nn.Module):
    """Grouped-query self-attention with the original parameter names."""

    def __init__(
        self,
        config: CSMTransformerConfig,
        rope: CSMLlama3ScaledRoPE,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.max_sequence_length = config.max_sequence_length
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.output_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.rope = rope
        self.cache: CSMKVCache | None = None

    def setup_cache(
        self,
        batch_size: int,
        *,
        max_sequence_length: int,
        device,
        dtype,
    ) -> None:
        self.cache = CSMKVCache(
            batch_size=batch_size,
            num_heads=self.num_heads,
            max_sequence_length=max_sequence_length,
            head_dim=self.head_dim,
            device=device,
            dtype=dtype,
        )

    def reset_cache(self) -> None:
        if self.cache is None:
            raise RuntimeError("CSM attention cache has not been initialized.")
        self.cache.reset()

    def forward(
        self,
        inputs: Tensor,
        *,
        attention_mask: Tensor | None = None,
        input_positions: Tensor | None = None,
    ) -> Tensor:
        batch_size, sequence_length, _ = inputs.shape
        groups = self.num_heads // self.num_key_value_heads
        queries = self.q_proj(inputs).reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        keys = self.k_proj(inputs).reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        values = self.v_proj(inputs).reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        queries = self.rope(
            queries,
            input_positions=input_positions,
        ).transpose(1, 2)
        keys = self.rope(
            keys,
            input_positions=input_positions,
        )
        if groups != 1:
            keys = keys[:, :, :, None, :].expand(
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                groups,
                self.head_dim,
            ).reshape(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            values = values[:, :, :, None, :].expand(
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                groups,
                self.head_dim,
            ).reshape(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        if self.cache is not None:
            keys, values = self.cache.update(keys, values)
        normalized_mask = attention_mask
        if normalized_mask is not None:
            if normalized_mask.ndim == 2:
                normalized_mask = normalized_mask.unsqueeze(0)
            if normalized_mask.ndim == 3:
                normalized_mask = normalized_mask.unsqueeze(1)
            if normalized_mask.ndim != 4:
                raise ValueError("CSM attention masks must have rank two, three, or four.")
            normalized_mask = normalized_mask.to(
                device=inputs.device,
                dtype=torch.bool,
            )
        attended = functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=normalized_mask,
            dropout_p=(self.attention_dropout if self.training else 0.0),
            is_causal=self.cache is None and normalized_mask is None,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch_size,
            sequence_length,
            self.hidden_size,
        )
        return self.output_proj(attended)


class CSMFeedForward(nn.Module):
    """Source-compatible SwiGLU feed-forward network."""

    def __init__(
        self,
        config: CSMTransformerConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.w2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.w3 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.w2(functional.silu(self.w1(inputs)) * self.w3(inputs))


class CSMTransformerLayer(nn.Module):
    """Pre-normalized CSM decoder layer."""

    def __init__(
        self,
        config: CSMTransformerConfig,
        rope: CSMLlama3ScaledRoPE,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.attn = CSMAttention(
            config,
            rope,
            device=device,
            dtype=dtype,
        )
        self.mlp = CSMFeedForward(
            config,
            device=device,
            dtype=dtype,
        )
        self.sa_norm = CSMRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.mlp_norm = CSMRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        attention_mask: Tensor | None = None,
        input_positions: Tensor | None = None,
    ) -> Tensor:
        hidden = inputs + self.attn(
            self.sa_norm(inputs),
            attention_mask=attention_mask,
            input_positions=input_positions,
        )
        return hidden + self.mlp(self.mlp_norm(hidden))


class CSMTransformerDecoder(nn.Module):
    """Embedding-free transformer used by both levels of CSM."""

    def __init__(
        self,
        config: CSMTransformerConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_sequence_length
        rope = CSMLlama3ScaledRoPE(
            config,
            device=device,
        )
        self.layers = nn.ModuleList([
            CSMTransformerLayer(
                config,
                rope,
                device=device,
                dtype=dtype,
            ) for _ in range(config.num_hidden_layers)
        ])
        self.norm = CSMRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        decoder_max_seq_len: int | None = None,
    ) -> None:
        maximum = decoder_max_seq_len or self.max_seq_len
        device = next(self.parameters()).device
        for layer in self.layers:
            layer.attn.setup_cache(
                batch_size,
                max_sequence_length=maximum,
                device=device,
                dtype=dtype,
            )

    def caches_are_enabled(self) -> bool:
        return bool(self.layers and self.layers[0].attn.cache is not None)

    def reset_caches(self) -> None:
        if not self.caches_are_enabled():
            raise RuntimeError("CSM caches have not been initialized.")
        for layer in self.layers:
            layer.attn.reset_cache()

    def forward(
        self,
        inputs: Tensor,
        *,
        mask: Tensor | None = None,
        input_pos: Tensor | None = None,
    ) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("CSM transformer inputs must have shape [batch, time, hidden].")
        if inputs.shape[1] > self.max_seq_len:
            raise ValueError("CSM transformer context length was exceeded.")
        hidden = inputs
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                input_positions=input_pos,
            )
        return self.norm(hidden)


@dataclass(frozen=True)
class CSMOutput:
    """Differentiable outputs of the native CSM objective."""

    loss: Tensor | None
    backbone_loss: Tensor | None
    depth_decoder_loss: Tensor | None
    logits: Tensor
    depth_decoder_logits: Tensor | None
    hidden_states: Tensor


def _causal_mask(
    max_sequence_length: int,
    *,
    device,
) -> Tensor:
    return torch.tril(
        torch.ones(
            max_sequence_length,
            max_sequence_length,
            dtype=torch.bool,
            device=device,
        ), )


def _indexed_causal_mask(
    mask: Tensor,
    input_positions: Tensor,
) -> Tensor:
    return mask[input_positions]


def sample_top_k(
    logits: Tensor,
    *,
    top_k: int,
    temperature: float,
) -> Tensor:
    """Sample with the Gumbel/exponential method used by Sesame."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("`top_k` must be an integer.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("`temperature` must be a real number.")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("`temperature` must be finite and non-negative.")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True).to(dtype=torch.int)
    if top_k <= 0 or top_k > logits.shape[-1]:
        raise ValueError(f"`top_k` must be in [1, {logits.shape[-1]}] while sampling.")
    scaled = logits / temperature
    threshold = torch.topk(scaled, top_k, dim=-1).values[..., -1, None]
    scaled = scaled.masked_fill(scaled < threshold, -torch.inf)
    probabilities = functional.softmax(scaled, dim=-1)
    noise = torch.empty_like(probabilities).exponential_(1)
    return (probabilities / noise).argmax(
        dim=-1,
        keepdim=True,
    ).to(dtype=torch.int)


class CSMModel(nn.Module):
    """Native CSM language model with official checkpoint namespaces."""

    def __init__(
        self,
        config: CSMArchitectureConfig | None = None,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config or CSMArchitectureConfig()
        self.backbone = CSMTransformerDecoder(
            self.config.backbone,
            device=device,
            dtype=dtype,
        )
        self.decoder = CSMTransformerDecoder(
            self.config.depth_decoder,
            device=device,
            dtype=dtype,
        )
        self.text_embeddings = nn.Embedding(
            self.config.text_vocabulary_size,
            self.config.backbone.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.audio_embeddings = nn.Embedding(
            self.config.audio_vocabulary_size * self.config.num_audio_codebooks,
            self.config.backbone.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.projection = nn.Linear(
            self.config.backbone.hidden_size,
            self.config.depth_decoder.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.codebook0_head = nn.Linear(
            self.config.backbone.hidden_size,
            self.config.audio_vocabulary_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.audio_head = nn.Parameter(
            torch.empty(
                self.config.num_audio_codebooks - 1,
                self.config.depth_decoder.hidden_size,
                self.config.audio_vocabulary_size,
                device=device,
                dtype=dtype,
            ), )
        self.register_buffer(
            "backbone_causal_mask",
            torch.empty(0, dtype=torch.bool, device=device),
            persistent=False,
        )
        self.register_buffer(
            "decoder_causal_mask",
            torch.empty(0, dtype=torch.bool, device=device),
            persistent=False,
        )
        if torch.device("cpu" if device is None else device).type != "meta":
            self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.audio_head, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, CSMRMSNorm):
                nn.init.ones_(module.scale)

    def materialize_runtime_buffers(self, device=None) -> None:
        """Rebuild non-persistent RoPE buffers after meta-device loading."""
        target_device = (next(self.parameters()).device if device is None else torch.device(device))
        for decoder in (self.backbone, self.decoder):
            config = decoder.config
            replacement = CSMLlama3ScaledRoPE(
                config,
                device=target_device,
            )
            for layer in decoder.layers:
                layer.attn.rope = replacement
        self.backbone_causal_mask = torch.empty(
            0,
            dtype=torch.bool,
            device=target_device,
        )
        self.decoder_causal_mask = torch.empty(
            0,
            dtype=torch.bool,
            device=target_device,
        )

    def setup_caches(self, max_batch_size: int = 1) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size <= 0:
            raise ValueError("`max_batch_size` must be a positive integer.")
        parameter = next(self.parameters())
        self.backbone.setup_caches(
            max_batch_size,
            parameter.dtype,
        )
        self.decoder.setup_caches(
            max_batch_size,
            parameter.dtype,
            decoder_max_seq_len=self.config.num_audio_codebooks,
        )
        self.backbone_causal_mask = _causal_mask(
            self.config.backbone.max_sequence_length,
            device=parameter.device,
        )
        self.decoder_causal_mask = _causal_mask(
            self.config.num_audio_codebooks,
            device=parameter.device,
        )

    def reset_caches(self) -> None:
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose tensor boundaries shared by CSM generation and training."""
        if mode not in {"inference", "training"}:
            raise ValueError("CSM compile targets require 'inference' or 'training' mode.")
        return (
            OptimizationCompileTarget(
                "backbone.forward",
                self.backbone,
                "forward",
            ),
            OptimizationCompileTarget(
                "depth_decoder.forward",
                self.decoder,
                "forward",
            ),
        )

    def _embed_audio(
        self,
        codebook: int,
        tokens: Tensor,
    ) -> Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocabulary_size, )

    def embed_tokens(
        self,
        tokens: Tensor,
        tokens_mask: Tensor,
    ) -> Tensor:
        """Embed masked text/audio slots and sum them into frame vectors."""
        expected_width = self.config.num_audio_codebooks + 1
        if (tokens.ndim != 3 or tokens.shape[-1] != expected_width or
                tuple(tokens_mask.shape) != tuple(tokens.shape)):
            raise ValueError("CSM tokens and masks must have shape "
                             f"[batch, time, {expected_width}].")
        if tokens.dtype == torch.bool or tokens.is_floating_point():
            raise TypeError("CSM token IDs must use an integer dtype.")
        mask = tokens_mask.to(device=tokens.device, dtype=torch.bool)
        safe = torch.where(mask, tokens, torch.zeros_like(tokens))
        text_ids = safe[..., -1]
        active_text = mask[..., -1]
        if active_text.any() and (int(text_ids[active_text].min()) < 0 or
                                  int(text_ids[active_text].max()) >= self.config.text_vocabulary_size):
            raise ValueError("CSM text token ID is outside the vocabulary.")
        audio_ids = safe[..., :-1]
        active_audio = mask[..., :-1]
        if active_audio.any() and (int(audio_ids[active_audio].min()) < 0 or
                                   int(audio_ids[active_audio].max()) >= self.config.audio_vocabulary_size):
            raise ValueError("CSM audio token ID is outside the vocabulary.")
        text = self.text_embeddings(text_ids).unsqueeze(-2)
        offsets = (
            self.config.audio_vocabulary_size * torch.arange(
                self.config.num_audio_codebooks,
                device=tokens.device,
            ))
        audio = self.audio_embeddings(audio_ids + offsets, )
        embeddings = torch.cat((audio, text), dim=-2)
        return (embeddings * mask.unsqueeze(-1).to(dtype=embeddings.dtype)).sum(dim=-2)

    def forward_backbone(
        self,
        tokens: Tensor,
        tokens_mask: Tensor,
        *,
        attention_mask: Tensor | None = None,
        input_positions: Tensor | None = None,
    ) -> Tensor:
        if attention_mask is not None and attention_mask.ndim == 2:
            if tuple(attention_mask.shape) != tuple(tokens.shape[:2]):
                raise ValueError("Rank-two CSM attention masks must have shape "
                                 "[batch, time].")
            length = tokens.shape[1]
            causal = torch.tril(torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=tokens.device,
            ), )
            attention_mask = (
                causal.unsqueeze(0) & attention_mask.to(
                    device=tokens.device,
                    dtype=torch.bool,
                ).unsqueeze(1))
        return self.backbone(
            self.embed_tokens(tokens, tokens_mask),
            mask=attention_mask,
            input_pos=input_positions,
        )

    def _depth_forward(
        self,
        *,
        backbone_hidden: Tensor,
        codebook_inputs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (codebook_inputs.ndim != 2 or codebook_inputs.shape[1] != self.config.num_audio_codebooks - 1):
            raise ValueError("Depth inputs must contain codebooks 0 through K-2.")
        codebook_count = self.config.num_audio_codebooks - 1
        offsets = (
            self.config.audio_vocabulary_size * torch.arange(
                codebook_count,
                device=codebook_inputs.device,
            ))
        embedded = self.audio_embeddings(codebook_inputs + offsets)
        depth_inputs = torch.cat(
            (backbone_hidden.unsqueeze(1), embedded),
            dim=1,
        )
        hidden = self.decoder(self.projection(depth_inputs))
        logits = torch.einsum(
            "nkh,khv->nkv",
            hidden[:, 1:, :],
            self.audio_head,
        )
        return logits, hidden

    def forward(
        self,
        tokens: Tensor,
        tokens_mask: Tensor,
        *,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> CSMOutput:
        """Run inference logits or the published two-level CSM objective.

        ``labels`` is pre-encoded Mimi codebook data with shape
        ``[batch, time, num_codebooks]``.  Raw waveform encoding belongs
        to the separately frozen Mimi preprocessing boundary.
        """
        hidden = self.forward_backbone(
            tokens,
            tokens_mask,
            attention_mask=attention_mask,
        )
        backbone_logits = self.codebook0_head(hidden)
        if labels is None:
            return CSMOutput(
                loss=None,
                backbone_loss=None,
                depth_decoder_loss=None,
                logits=backbone_logits,
                depth_decoder_logits=None,
                hidden_states=hidden,
            )
        expected = (
            tokens.shape[0],
            tokens.shape[1],
            self.config.num_audio_codebooks,
        )
        if tuple(labels.shape) != expected:
            raise ValueError(f"CSM labels must have shape {expected!r}.")
        if labels.dtype == torch.bool or labels.is_floating_point():
            raise TypeError("CSM labels must use an integer dtype.")
        if labels.shape[1] < 2:
            raise ValueError("CSM training requires at least two frames.")
        supervised = labels != -100
        if bool(supervised.any()) and (int(labels[supervised].min()) < 0 or
                                       int(labels[supervised].max()) >= self.config.audio_vocabulary_size):
            raise ValueError("CSM label is outside the audio vocabulary.")
        backbone_targets = labels[:, 1:, 0].contiguous()
        if bool((backbone_targets != -100).any()):
            backbone_loss = functional.cross_entropy(
                backbone_logits[:, :-1, :].float().reshape(
                    -1,
                    self.config.audio_vocabulary_size,
                ),
                backbone_targets.reshape(-1),
                ignore_index=-100,
            )
        else:
            # PyTorch's mean-reduced CE returns NaN when every target is
            # ignored.  Keep fully masked examples numerically stable while
            # preserving a differentiable zero contribution.
            backbone_loss = backbone_logits.sum() * 0.0
        train_mask = ~(labels[..., 1:] == -100).all(dim=-1)
        train_mask[:, 0] = False
        if bool(train_mask.any()):
            frame_labels = labels[train_mask]
            depth_inputs = frame_labels[..., :-1]
            if bool((depth_inputs < 0).any()):
                raise ValueError("Selected CSM depth frames require valid codebooks 0 "
                                 "through K-2.")
            indices = train_mask.nonzero(as_tuple=True)
            preceding_hidden = hidden[indices[0], indices[1] - 1]
            depth_logits, _ = self._depth_forward(
                backbone_hidden=preceding_hidden,
                codebook_inputs=depth_inputs,
            )
            depth_targets = frame_labels[..., 1:]
            depth_loss = functional.cross_entropy(
                depth_logits.float().reshape(
                    -1,
                    self.config.audio_vocabulary_size,
                ),
                depth_targets.reshape(-1),
                ignore_index=-100,
            )
        else:
            depth_logits = backbone_logits.new_empty(
                0,
                self.config.num_audio_codebooks - 1,
                self.config.audio_vocabulary_size,
            )
            depth_loss = backbone_logits.sum() * 0.0
        return CSMOutput(
            loss=backbone_loss + depth_loss,
            backbone_loss=backbone_loss,
            depth_decoder_loss=depth_loss,
            logits=backbone_logits,
            depth_decoder_logits=depth_logits,
            hidden_states=hidden,
        )

    @torch.no_grad()
    def generate_frame(
        self,
        tokens: Tensor,
        tokens_mask: Tensor,
        input_positions: Tensor,
        *,
        temperature: float,
        top_k: int,
    ) -> Tensor:
        if not self.backbone.caches_are_enabled():
            raise RuntimeError("Call `setup_caches()` before CSM generation.")
        backbone_mask = _indexed_causal_mask(
            self.backbone_causal_mask,
            input_positions,
        )
        hidden = self.forward_backbone(
            tokens,
            tokens_mask,
            attention_mask=backbone_mask,
            input_positions=input_positions,
        )
        last_hidden = hidden[:, -1]
        first = sample_top_k(
            self.codebook0_head(last_hidden),
            top_k=top_k,
            temperature=temperature,
        )
        current_hidden = torch.cat(
            (
                last_hidden.unsqueeze(1),
                self._embed_audio(0, first),
            ),
            dim=1,
        )
        samples = first.clone()
        positions = torch.arange(
            current_hidden.shape[1],
            device=current_hidden.device,
        ).unsqueeze(0).expand(current_hidden.shape[0], -1)
        self.decoder.reset_caches()
        for codebook in range(1, self.config.num_audio_codebooks):
            decoder_mask = _indexed_causal_mask(
                self.decoder_causal_mask,
                positions,
            )
            decoder_hidden = self.decoder(
                self.projection(current_hidden),
                mask=decoder_mask,
                input_pos=positions,
            )
            logits = torch.matmul(
                decoder_hidden[:, -1],
                self.audio_head[codebook - 1],
            )
            sample = sample_top_k(
                logits,
                top_k=top_k,
                temperature=temperature,
            )
            samples = torch.cat((samples, sample), dim=1)
            current_hidden = self._embed_audio(codebook, sample)
            positions = positions[:, -1:] + 1
        return samples

    @torch.no_grad()
    def generate_audio_codes(
        self,
        prompt_tokens: Tensor,
        prompt_mask: Tensor,
        *,
        max_new_frames: int,
        temperature: float = 0.9,
        top_k: int = 50,
    ) -> Tensor:
        """Autoregressively generate Mimi codes from prepared prompt frames."""
        if (isinstance(max_new_frames, bool) or not isinstance(max_new_frames, int) or max_new_frames <= 0):
            raise ValueError("`max_new_frames` must be a positive integer.")
        if prompt_tokens.shape[0] != 1:
            raise ValueError("Native CSM generation currently requires batch size one.")
        if (prompt_tokens.shape[1] + max_new_frames >= self.config.backbone.max_sequence_length):
            raise ValueError("Prompt plus requested audio exceeds CSM context.")
        self.setup_caches(max_batch_size=1)
        self.backbone.reset_caches()
        samples = []
        current_tokens = prompt_tokens
        current_mask = prompt_mask
        positions = torch.arange(
            prompt_tokens.shape[1],
            device=prompt_tokens.device,
        ).unsqueeze(0)
        for _ in range(max_new_frames):
            sample = self.generate_frame(
                current_tokens,
                current_mask,
                positions,
                temperature=temperature,
                top_k=top_k,
            )
            if bool((sample == 0).all()):
                break
            samples.append(sample)
            current_tokens = torch.cat(
                (
                    sample,
                    torch.zeros(
                        1,
                        1,
                        device=sample.device,
                        dtype=sample.dtype,
                    ),
                ),
                dim=1,
            ).unsqueeze(1)
            current_mask = torch.cat(
                (
                    torch.ones_like(sample, dtype=torch.bool),
                    torch.zeros(
                        1,
                        1,
                        device=sample.device,
                        dtype=torch.bool,
                    ),
                ),
                dim=1,
            ).unsqueeze(1)
            positions = positions[:, -1:] + 1
        if not samples:
            raise RuntimeError("CSM generation reached EOS before producing an audio frame.")
        return torch.stack(samples, dim=-1)


__all__ = [
    "CSMAttention",
    "CSMFeedForward",
    "CSMLlama3ScaledRoPE",
    "CSMModel",
    "CSMOutput",
    "CSMRMSNorm",
    "CSMTransformerDecoder",
    "CSMTransformerLayer",
    "sample_top_k",
]
