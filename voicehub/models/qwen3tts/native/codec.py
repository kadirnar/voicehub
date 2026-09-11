"""Native decoder for the Qwen3-TTS 12 Hz speech tokenizer.

The checkpoint-exact Mimi-derived encoder lives in the adjacent
``encoder`` module so decoder-only use keeps this existing API and its
lightweight construction boundary.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.kernels.codecs import CodecSnakeBetaKernelOptimizable
from voicehub.models.qwen3tts.native.configuration import Qwen3TTSDecoderConfig
from voicehub.neural.normalization import RMSNorm
from voicehub.neural.rotary import RotaryEmbedding, apply_rotary_embedding
from voicehub.optimization.protocols import OptimizationCompileTarget


def _factory(
    *,
    initialize: bool,
    device: str | torch.device | None,
    dtype: torch.dtype | None,
) -> dict[str, Any]:
    return {
        "device": device if initialize else "meta",
        "dtype": dtype,
    }


class CausalConv1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            **factory_kwargs,
        )
        effective_kernel = (kernel_size - 1) * dilation + 1
        self.stride = stride
        self.kernel_size = effective_kernel
        self.padding = effective_kernel - stride

    def forward(self, hidden_states: Tensor) -> Tensor:
        length = hidden_states.shape[-1]
        frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal = ((math.ceil(frames) - 1) * self.stride + (self.kernel_size - self.padding))
        extra = ideal - length
        hidden_states = functional.pad(
            hidden_states,
            (self.padding, extra),
        )
        return self.conv(hidden_states).contiguous()


class CausalConvTranspose1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            **factory_kwargs,
        )
        self.right_pad = kernel_size - stride

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        if self.right_pad:
            hidden_states = hidden_states[..., :-self.right_pad]
        return hidden_states.contiguous()


class ConvNeXtBlock(nn.Module):

    def __init__(
        self,
        dimension: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.dwconv = CausalConv1d(
            dimension,
            dimension,
            7,
            groups=dimension,
            factory_kwargs=factory_kwargs,
        )
        self.norm = nn.LayerNorm(
            dimension,
            eps=1e-6,
            **factory_kwargs,
        )
        self.pwconv1 = nn.Linear(
            dimension,
            4 * dimension,
            **factory_kwargs,
        )
        self.pwconv2 = nn.Linear(
            4 * dimension,
            dimension,
            **factory_kwargs,
        )
        self.gamma = nn.Parameter(torch.full(
            (dimension, ),
            1e-6,
            **factory_kwargs,
        ))

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states).transpose(1, 2)
        hidden_states = self.pwconv2(functional.gelu(self.pwconv1(self.norm(hidden_states))))
        return residual + (self.gamma * hidden_states).transpose(1, 2)


def _expand_kv(hidden_states: Tensor, groups: int) -> Tensor:
    if groups == 1:
        return hidden_states
    batch, heads, time, dimension = hidden_states.shape
    return (
        hidden_states[:, :, None].expand(batch, heads, groups, time,
                                         dimension).reshape(batch, heads * groups, time, dimension))


def _sliding_causal_bias(
    *,
    batch: int,
    time: int,
    window: int,
    device: torch.device,
) -> Tensor:
    positions = torch.arange(time, device=device)
    allowed = ((positions[None, :] <= positions[:, None])
               & (positions[None, :] > positions[:, None] - window))
    bias = torch.zeros(
        (batch, 1, time, time),
        device=device,
        dtype=torch.float32,
    )
    return bias.masked_fill(
        ~allowed.view(1, 1, time, time),
        torch.finfo(torch.float32).min,
    )


class DecoderAttention(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSDecoderConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
            **factory_kwargs,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            **factory_kwargs,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            **factory_kwargs,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            **factory_kwargs,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cosine: Tensor,
        sine: Tensor,
        attention_bias: Tensor,
    ) -> Tensor:
        batch, time, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch,
            time,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch,
            time,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch,
            time,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)
        query, key = apply_rotary_embedding(query, key, cosine, sine)
        key = _expand_kv(key, self.groups)
        value = _expand_kv(value, self.groups)
        weights = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)).float() * self.scaling + attention_bias,
            dim=-1,
        ).to(dtype=query.dtype)
        weights = functional.dropout(
            weights,
            p=self.attention_dropout,
            training=self.training,
        )
        output = torch.matmul(weights, value)
        return self.o_proj(output.transpose(1, 2).reshape(
            batch,
            time,
            self.num_heads * self.head_dim,
        ))


class DecoderMLP(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSDecoderConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            **factory_kwargs,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DecoderLayerScale(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSDecoderConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(
            torch.full(
                (config.hidden_size, ),
                config.layer_scale_initial_scale,
                **factory_kwargs,
            ))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.scale * hidden_states


class DecoderTransformerLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSDecoderConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.self_attn = DecoderAttention(
            config,
            factory_kwargs=factory_kwargs,
        )
        self.mlp = DecoderMLP(
            config,
            factory_kwargs=factory_kwargs,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            **factory_kwargs,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            **factory_kwargs,
        )
        self.self_attn_layer_scale = DecoderLayerScale(
            config,
            factory_kwargs=factory_kwargs,
        )
        self.mlp_layer_scale = DecoderLayerScale(
            config,
            factory_kwargs=factory_kwargs,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cosine: Tensor,
        sine: Tensor,
        attention_bias: Tensor,
    ) -> Tensor:
        hidden_states = hidden_states + self.self_attn_layer_scale(
            self.self_attn(
                self.input_layernorm(hidden_states),
                cosine=cosine,
                sine=sine,
                attention_bias=attention_bias,
            ))
        return hidden_states + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(hidden_states)))


class DecoderTransformer(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSDecoderConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            DecoderTransformerLayer(
                config,
                factory_kwargs=factory_kwargs,
            ) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            **factory_kwargs,
        )
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            base=config.rope_theta,
            device=factory_kwargs["device"],
        )
        self.input_proj = nn.Linear(
            config.latent_dim,
            config.hidden_size,
            **factory_kwargs,
        )
        self.output_proj = nn.Linear(
            config.hidden_size,
            config.latent_dim,
            **factory_kwargs,
        )

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        hidden_states = self.input_proj(inputs_embeds)
        batch, time, _ = hidden_states.shape
        positions = torch.arange(
            time,
            device=hidden_states.device,
        ).unsqueeze(0).expand(batch, -1)
        cosine, sine = self.rotary_emb(
            positions,
            dtype=hidden_states.dtype,
        )
        bias = _sliding_causal_bias(
            batch=batch,
            time=time,
            window=self.config.sliding_window,
            device=hidden_states.device,
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cosine=cosine,
                sine=sine,
                attention_bias=bias,
            )
        return self.output_proj(self.norm(hidden_states))


class SnakeBeta(CodecSnakeBetaKernelOptimizable, nn.Module):

    def __init__(
        self,
        channels: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels, **factory_kwargs))
        self.beta = nn.Parameter(torch.zeros(channels, **factory_kwargs))
        self._initialize_codec_kernel_backend()

    def forward(self, hidden_states: Tensor) -> Tensor:
        alpha = self.alpha.exp()[None, :, None]
        beta = self.beta.exp()[None, :, None]
        return self._codec_snake_beta(hidden_states, alpha, beta)


class DecoderResidualUnit(nn.Module):

    def __init__(
        self,
        dimension: int,
        dilation: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.act1 = SnakeBeta(
            dimension,
            factory_kwargs=factory_kwargs,
        )
        self.conv1 = CausalConv1d(
            dimension,
            dimension,
            7,
            dilation=dilation,
            factory_kwargs=factory_kwargs,
        )
        self.act2 = SnakeBeta(
            dimension,
            factory_kwargs=factory_kwargs,
        )
        self.conv2 = CausalConv1d(
            dimension,
            dimension,
            1,
            factory_kwargs=factory_kwargs,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.conv1(self.act1(hidden_states))
        return residual + self.conv2(self.act2(hidden_states))


class DecoderBlock(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSDecoderConfig,
        layer_index: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        input_dimension = config.decoder_dim // 2**layer_index
        output_dimension = config.decoder_dim // 2**(layer_index + 1)
        rate = config.upsample_rates[layer_index]
        modules: list[nn.Module] = [
            SnakeBeta(
                input_dimension,
                factory_kwargs=factory_kwargs,
            ),
            CausalConvTranspose1d(
                input_dimension,
                output_dimension,
                2 * rate,
                rate,
                factory_kwargs=factory_kwargs,
            ),
        ]
        modules.extend(
            DecoderResidualUnit(
                output_dimension,
                dilation,
                factory_kwargs=factory_kwargs,
            ) for dilation in (1, 3, 9))
        self.block = nn.ModuleList(modules)

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.block:
            hidden_states = module(hidden_states)
        return hidden_states


class EuclideanCodebook(nn.Module):

    def __init__(
        self,
        dimension: int,
        size: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.cluster_usage = nn.Parameter(torch.ones(size, **factory_kwargs))
        self.embedding_sum = nn.Parameter(torch.zeros(size, dimension, **factory_kwargs))

    def decode(self, codes: Tensor) -> Tensor:
        embedding = self.embedding_sum / self.cluster_usage.clamp_min(1e-5)[:, None]
        return functional.embedding(codes, embedding)


class VectorQuantization(nn.Module):

    def __init__(
        self,
        dimension: int,
        size: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self._codebook = EuclideanCodebook(
            dimension,
            size,
            factory_kwargs=factory_kwargs,
        )

    def decode(self, codes: Tensor) -> Tensor:
        return self._codebook.decode(codes).transpose(1, 2)


class ResidualVectorQuantization(nn.Module):

    def __init__(
        self,
        count: int,
        dimension: int,
        size: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(
                dimension,
                size,
                factory_kwargs=factory_kwargs,
            ) for _ in range(count)])

    def decode(self, codes: Tensor) -> Tensor:
        result: Tensor | None = None
        for table, values in zip(self.layers, codes):
            decoded = table.decode(values)
            result = decoded if result is None else result + decoded
        if result is None:
            raise ValueError("At least one codebook is required.")
        return result


class ResidualVectorQuantizer(nn.Module):

    def __init__(
        self,
        *,
        count: int,
        dimension: int,
        input_dimension: int,
        output_dimension: int,
        size: int,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(
            input_dimension,
            dimension,
            1,
            bias=False,
            **factory_kwargs,
        )
        self.output_proj = nn.Conv1d(
            dimension,
            output_dimension,
            1,
            bias=False,
            **factory_kwargs,
        )
        self.vq = ResidualVectorQuantization(
            count,
            dimension,
            size,
            factory_kwargs=factory_kwargs,
        )

    def decode(self, codes: Tensor) -> Tensor:
        return self.output_proj(self.vq.decode(codes.transpose(0, 1)))


class SplitResidualVectorQuantizer(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSDecoderConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        dimension = config.codebook_dim // 2
        common = {
            "dimension": dimension,
            "input_dimension": config.codebook_dim,
            "output_dimension": config.codebook_dim,
            "size": config.codebook_size,
            "factory_kwargs": factory_kwargs,
        }
        self.rvq_first = ResidualVectorQuantizer(
            count=config.num_semantic_quantizers,
            **common,
        )
        self.rvq_rest = ResidualVectorQuantizer(
            count=config.num_quantizers - config.num_semantic_quantizers,
            **common,
        )
        self.semantic_count = config.num_semantic_quantizers

    def decode(self, codes: Tensor) -> Tensor:
        result = self.rvq_first.decode(codes[:, :self.semantic_count])
        if codes.shape[1] > self.semantic_count:
            result = result + self.rvq_rest.decode(codes[:, self.semantic_count:])
        return result


class Qwen3TTSSpeechDecoder(nn.Module):
    """Turn generated 16-codebook frames into 24 kHz waveform samples."""

    def __init__(
        self,
        config: Qwen3TTSDecoderConfig,
        *,
        initialize: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        factory_kwargs = _factory(
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.config = config
        self.total_upsample = config.total_upsample
        self.pre_transformer = DecoderTransformer(
            config,
            factory_kwargs=factory_kwargs,
        )
        self.quantizer = SplitResidualVectorQuantizer(
            config,
            factory_kwargs=factory_kwargs,
        )
        self.pre_conv = CausalConv1d(
            config.codebook_dim,
            config.latent_dim,
            3,
            factory_kwargs=factory_kwargs,
        )
        self.upsample = nn.ModuleList([
            nn.ModuleList([
                CausalConvTranspose1d(
                    config.latent_dim,
                    config.latent_dim,
                    factor,
                    factor,
                    factory_kwargs=factory_kwargs,
                ),
                ConvNeXtBlock(
                    config.latent_dim,
                    factory_kwargs=factory_kwargs,
                ),
            ]) for factor in config.upsampling_ratios
        ])
        modules: list[nn.Module] = [
            CausalConv1d(
                config.latent_dim,
                config.decoder_dim,
                7,
                factory_kwargs=factory_kwargs,
            )
        ]
        modules.extend(
            DecoderBlock(
                config,
                index,
                factory_kwargs=factory_kwargs,
            ) for index in range(len(config.upsample_rates)))
        output_dimension = config.decoder_dim // 2**len(config.upsample_rates)
        modules.extend([
            SnakeBeta(
                output_dimension,
                factory_kwargs=factory_kwargs,
            ),
            CausalConv1d(
                output_dimension,
                1,
                7,
                factory_kwargs=factory_kwargs,
            ),
        ])
        self.decoder = nn.ModuleList(modules)

    def _validate_codes(self, codes: Tensor) -> None:
        if (codes.ndim != 3 or codes.shape[1] != self.config.num_quantizers or codes.dtype == torch.bool or
                codes.is_floating_point() or codes.is_complex()):
            raise ValueError("Speech codes must be integer [batch, codebooks, frames].")
        if codes.numel() and (bool((codes < 0).any()) or bool((codes >= self.config.codebook_size).any())):
            raise ValueError("Speech code ID is outside the decoder codebook.")

    def _decode_codes_unchecked(self, codes: Tensor) -> Tensor:
        """Decode already validated code IDs without tensor-to-host sync."""
        hidden_states = self.quantizer.decode(codes)
        hidden_states = self.pre_conv(hidden_states).transpose(1, 2)
        hidden_states = self.pre_transformer(hidden_states).transpose(1, 2)
        for stage in self.upsample:
            for module in stage:
                hidden_states = module(hidden_states)
        for module in self.decoder:
            hidden_states = module(hidden_states)
        return hidden_states.clamp(-1, 1)

    def forward(self, codes: Tensor) -> Tensor:
        self._validate_codes(codes)
        return self._decode_codes_unchecked(codes)

    def codec_optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        if mode not in {"inference", "training"}:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (
            OptimizationCompileTarget(
                "codec.decode.qwen3_tts.decode_codes",
                self,
                "_decode_codes_unchecked",
                component="decode",
            ), )

    def chunked_decode(
        self,
        codes: Tensor,
        *,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> Tensor:
        if chunk_size <= 0 or left_context_size < 0:
            raise ValueError("Decoder chunk size must be positive and context non-negative.")
        chunks = []
        start = 0
        while start < codes.shape[-1]:
            end = min(start + chunk_size, codes.shape[-1])
            context = min(left_context_size, start)
            waveform = self(codes[..., start - context:end])
            chunks.append(waveform[..., context * self.total_upsample:])
            start = end
        if not chunks:
            return torch.empty(
                (codes.shape[0], 1, 0),
                device=codes.device,
                dtype=next(self.parameters()).dtype,
            )
        return torch.cat(chunks, dim=-1)


def materialize_qwen3_tts_decoder_buffers(
    decoder: nn.Module,
    *,
    device: str | torch.device,
) -> None:
    for module in decoder.modules():
        if isinstance(module, RotaryEmbedding):
            module.inverse_frequency = 1.0 / (
                module.base**(
                    torch.arange(
                        0,
                        module.dimension,
                        2,
                        dtype=torch.float32,
                        device=device,
                    ) / module.dimension))


__all__ = [
    "Qwen3TTSSpeechDecoder",
    "materialize_qwen3_tts_decoder_buffers",
]
