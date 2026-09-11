"""Native Qwen3-TTS 12 Hz speech-tokenizer encoder.

The graph is a checkpoint-exact PyTorch port of the encoder retained by
``Qwen3TTSTokenizerV2Encoder`` at the pinned Qwen3-TTS revision.  That
class subclasses the Apache-2.0 Transformers 4.57.3 Mimi model, then
removes Mimi's decoder.  Module names below intentionally mirror the
published ``encoder.*`` Safetensors namespace.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.qwen3tts.native.configuration import Qwen3TTSEncoderConfig
from voicehub.neural.rotary import RotaryEmbedding, apply_rotary_embedding


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


class EncoderCausalConv1d(nn.Module):
    """Mimi causal convolution with its exact asymmetric padding."""

    def __init__(
        self,
        config: Qwen3TTSEncoderConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        pad_mode: str | None = None,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            **factory_kwargs,
        )
        self.in_channels = in_channels
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding_total = self.kernel_size - stride
        self.pad_mode = config.pad_mode if pad_mode is None else pad_mode

    def output_length(self, input_length: Tensor) -> Tensor:
        frames = (input_length - self.kernel_size + self.padding_total) / self.stride + 1
        frames = torch.ceil(frames).to(torch.int64) - 1
        ideal = (frames * self.stride + self.kernel_size - self.padding_total)
        extra = ideal - input_length
        padded = input_length + self.padding_total + extra
        return (padded - self.conv.dilation[0] *
                (self.conv.kernel_size[0] - 1) - 1) // self.conv.stride[0] + 1

    def forward(self, hidden_states: Tensor) -> Tensor:
        length = hidden_states.shape[-1]
        frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        frames = math.ceil(frames) - 1
        ideal = (frames * self.stride + self.kernel_size - self.padding_total)
        extra = ideal - length
        hidden_states = functional.pad(
            hidden_states,
            (self.padding_total, extra),
            mode=self.pad_mode,
        )
        return self.conv(hidden_states)


class EncoderResnetBlock(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSEncoderConfig,
        channels: int,
        *,
        dilation: int,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        hidden_channels = channels // config.compress
        self.block = nn.ModuleList([
            nn.ELU(),
            EncoderCausalConv1d(
                config,
                channels,
                hidden_channels,
                config.residual_kernel_size,
                dilation=dilation,
                factory_kwargs=factory_kwargs,
            ),
            nn.ELU(),
            EncoderCausalConv1d(
                config,
                hidden_channels,
                channels,
                1,
                factory_kwargs=factory_kwargs,
            ),
        ])
        self.shortcut = (
            EncoderCausalConv1d(
                config,
                channels,
                channels,
                1,
                factory_kwargs=factory_kwargs,
            ) if config.use_conv_shortcut else nn.Identity())

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        for module in self.block:
            hidden_states = module(hidden_states)
        return self.shortcut(residual) + hidden_states


class EncoderSEANet(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSEncoderConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = [
            EncoderCausalConv1d(
                config,
                config.audio_channels,
                config.num_filters,
                config.kernel_size,
                factory_kwargs=factory_kwargs,
            )
        ]
        convolution_paths = ["layers.0"]
        scaling = 1
        for ratio in reversed(config.upsampling_ratios):
            channels = scaling * config.num_filters
            for residual_index in range(config.num_residual_layers):
                convolution_paths.extend([
                    f"layers.{len(modules)}.block.1",
                    f"layers.{len(modules)}.block.3",
                ])
                modules.append(
                    EncoderResnetBlock(
                        config,
                        channels,
                        dilation=config.dilation_growth_rate**residual_index,
                        factory_kwargs=factory_kwargs,
                    ))
            modules.append(nn.ELU())
            convolution_paths.append(f"layers.{len(modules)}")
            modules.append(
                EncoderCausalConv1d(
                    config,
                    channels,
                    channels * 2,
                    ratio * 2,
                    stride=ratio,
                    factory_kwargs=factory_kwargs,
                ))
            scaling *= 2
        modules.append(nn.ELU())
        convolution_paths.append(f"layers.{len(modules)}")
        modules.append(
            EncoderCausalConv1d(
                config,
                scaling * config.num_filters,
                config.hidden_size,
                config.last_kernel_size,
                factory_kwargs=factory_kwargs,
            ))
        self.layers = nn.ModuleList(modules)
        self.convolution_paths = tuple(convolution_paths)

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.layers:
            hidden_states = module(hidden_states)
        return hidden_states


def _expand_kv(hidden_states: Tensor, groups: int) -> Tensor:
    if groups == 1:
        return hidden_states
    batch, heads, time, dimension = hidden_states.shape
    return (
        hidden_states[:, :, None].expand(batch, heads, groups, time,
                                         dimension).reshape(batch, heads * groups, time, dimension))


def _causal_bias(
    *,
    time: int,
    device: torch.device,
) -> Tensor:
    positions = torch.arange(time, device=device)
    allowed = positions[None, :] <= positions[:, None]
    bias = torch.zeros(
        (1, 1, time, time),
        device=device,
        dtype=torch.float32,
    )
    return bias.masked_fill(
        ~allowed.view(1, 1, time, time),
        torch.finfo(torch.float32).min,
    )


class EncoderLayerScale(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSEncoderConfig,
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


class EncoderAttention(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSEncoderConfig,
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
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            base=config.rope_theta,
            device=factory_kwargs["device"],
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
        query, key = apply_rotary_embedding(
            query,
            key,
            cosine,
            sine,
        )
        key = _expand_kv(key, self.groups)
        value = _expand_kv(value, self.groups)
        output = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias.to(dtype=query.dtype),
            dropout_p=(self.attention_dropout if self.training else 0.0),
            is_causal=False,
            scale=self.scaling,
        )
        return self.o_proj(output.transpose(1, 2).reshape(
            batch,
            time,
            self.num_heads * self.head_dim,
        ))


class EncoderMLP(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSEncoderConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.fc2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            **factory_kwargs,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(functional.gelu(self.fc1(hidden_states)))


class EncoderTransformerLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSEncoderConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.self_attn = EncoderAttention(
            config,
            factory_kwargs=factory_kwargs,
        )
        self.mlp = EncoderMLP(
            config,
            factory_kwargs=factory_kwargs,
        )
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            **factory_kwargs,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            **factory_kwargs,
        )
        self.self_attn_layer_scale = EncoderLayerScale(
            config,
            factory_kwargs=factory_kwargs,
        )
        self.mlp_layer_scale = EncoderLayerScale(
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
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            cosine=cosine,
            sine=sine,
            attention_bias=attention_bias,
        )
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + self.mlp_layer_scale(hidden_states)


class EncoderTransformer(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSEncoderConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            EncoderTransformerLayer(
                config,
                factory_kwargs=factory_kwargs,
            ) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch, time, _ = hidden_states.shape
        positions = torch.arange(
            time,
            device=hidden_states.device,
        ).unsqueeze(0).expand(batch, -1)
        rotary = self.layers[0].self_attn.rotary_emb
        cosine, sine = rotary(
            positions,
            dtype=hidden_states.dtype,
        )
        # The pinned Transformers 4.57.3 Mimi encoder calls
        # ``create_causal_mask`` here.  Its ``sliding_window`` field is
        # consumed only by the optional FlashAttention path, not by the
        # published default SDPA path.
        attention_bias = _causal_bias(
            time=time,
            device=hidden_states.device,
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cosine=cosine,
                sine=sine,
                attention_bias=attention_bias,
            )
        return hidden_states


class EncoderEuclideanCodebook(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSEncoderConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.register_buffer(
            "initialized",
            torch.ones(1, **factory_kwargs),
        )
        self.register_buffer(
            "cluster_usage",
            torch.ones(config.codebook_size, **factory_kwargs),
        )
        self.register_buffer(
            "embed_sum",
            torch.zeros(
                config.codebook_size,
                config.codebook_dim,
                **factory_kwargs,
            ),
        )

    @property
    def embed(self) -> Tensor:
        return (self.embed_sum / self.cluster_usage.clamp_min(1e-5)[:, None])

    def encode(self, hidden_states: Tensor) -> Tensor:
        shape = hidden_states.shape
        flattened = hidden_states.reshape(-1, shape[-1])
        distances = torch.cdist(
            flattened[None].float(),
            self.embed[None].float(),
            p=2,
        )[0]
        return distances.argmin(dim=-1).view(*shape[:-1])

    def decode(self, indices: Tensor) -> Tensor:
        return functional.embedding(indices, self.embed)


class EncoderVectorQuantization(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSEncoderConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.codebook = EncoderEuclideanCodebook(
            config,
            factory_kwargs=factory_kwargs,
        )

    def encode(self, hidden_states: Tensor) -> Tensor:
        return self.codebook.encode(hidden_states.permute(0, 2, 1))

    def decode(self, indices: Tensor) -> Tensor:
        return self.codebook.decode(indices).permute(0, 2, 1)


class EncoderResidualVectorQuantizer(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSEncoderConfig,
        count: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderVectorQuantization(
                config,
                factory_kwargs=factory_kwargs,
            ) for _ in range(count)])
        self.input_proj = nn.Conv1d(
            config.hidden_size,
            config.vector_quantization_hidden_dimension,
            1,
            bias=False,
            **factory_kwargs,
        )
        self.output_proj = nn.Conv1d(
            config.vector_quantization_hidden_dimension,
            config.hidden_size,
            1,
            bias=False,
            **factory_kwargs,
        )

    def encode(
        self,
        embeddings: Tensor,
        *,
        num_quantizers: int | None = None,
    ) -> Tensor:
        embeddings = self.input_proj(embeddings)
        count = len(self.layers) if num_quantizers is None else num_quantizers
        if not 0 < count <= len(self.layers):
            raise ValueError("Requested encoder quantizer count is unavailable.")
        residual = embeddings
        indices = []
        for layer in self.layers[:count]:
            code = layer.encode(residual)
            residual = residual - layer.decode(code)
            indices.append(code)
        return torch.stack(indices)


class EncoderSplitResidualVectorQuantizer(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSEncoderConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.semantic_residual_vector_quantizer = (
            EncoderResidualVectorQuantizer(
                config,
                config.num_semantic_quantizers,
                factory_kwargs=factory_kwargs,
            ))
        self.acoustic_residual_vector_quantizer = (
            EncoderResidualVectorQuantizer(
                config,
                config.num_quantizers - config.num_semantic_quantizers,
                factory_kwargs=factory_kwargs,
            ))
        self.semantic_count = config.num_semantic_quantizers
        self.max_count = config.num_quantizers

    def encode(
        self,
        embeddings: Tensor,
        *,
        num_quantizers: int,
    ) -> Tensor:
        if not self.semantic_count <= num_quantizers <= self.max_count:
            raise ValueError("Requested encoder quantizer count is unavailable.")
        codes = self.semantic_residual_vector_quantizer.encode(embeddings, )
        if num_quantizers > self.semantic_count:
            acoustic = self.acoustic_residual_vector_quantizer.encode(
                embeddings,
                num_quantizers=num_quantizers - self.semantic_count,
            )
            codes = torch.cat((codes, acoustic), dim=0)
        return codes


class Qwen3TTSSpeechEncoder(nn.Module):
    """Encode 24 kHz mono waveforms into Qwen's 12.5 Hz RVQ frames."""

    def __init__(
        self,
        config: Qwen3TTSEncoderConfig,
        *,
        valid_num_quantizers: int,
        initialize: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        if not (config.num_semantic_quantizers <= valid_num_quantizers <= config.num_quantizers):
            raise ValueError("Valid encoder quantizer count is unavailable.")
        factory_kwargs = _factory(
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.config = config
        self.valid_num_quantizers = valid_num_quantizers
        self.encoder = EncoderSEANet(
            config,
            factory_kwargs=factory_kwargs,
        )
        self.encoder_transformer = EncoderTransformer(
            config,
            factory_kwargs=factory_kwargs,
        )
        if config.downsample_factor != 2:
            raise ValueError("The pinned Qwen3-TTS encoder requires its 2x Mimi downsampler.")
        self.downsample = EncoderCausalConv1d(
            config,
            config.hidden_size,
            config.hidden_size,
            2 * int(config.encodec_frame_rate / config.frame_rate),
            stride=2,
            bias=False,
            pad_mode="replicate",
            factory_kwargs=factory_kwargs,
        )
        self.quantizer = EncoderSplitResidualVectorQuantizer(
            config,
            factory_kwargs=factory_kwargs,
        )

    def forward(
        self,
        input_values: Tensor,
        *,
        num_quantizers: int | None = None,
    ) -> Tensor:
        if not isinstance(input_values, Tensor):
            raise TypeError("Qwen3-TTS encoder audio must be a tensor.")
        if input_values.ndim == 2:
            input_values = input_values.unsqueeze(1)
        if (input_values.ndim != 3 or input_values.shape[1] != self.config.audio_channels):
            raise ValueError("Qwen3-TTS encoder audio must have shape "
                             "[batch, 1, samples].")
        if input_values.shape[0] == 0 or input_values.shape[-1] == 0:
            raise ValueError("Qwen3-TTS encoder audio cannot be empty.")
        if not input_values.is_floating_point():
            raise TypeError("Qwen3-TTS encoder audio must be floating-point.")
        count = (self.valid_num_quantizers if num_quantizers is None else num_quantizers)
        hidden_states = self.encoder(input_values)
        hidden_states = self.encoder_transformer(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.downsample(hidden_states)
        return self.quantizer.encode(
            hidden_states,
            num_quantizers=count,
        ).transpose(0, 1)

    def encode(
        self,
        input_values: Tensor,
        padding_mask: Tensor | None = None,
    ) -> list[Tensor]:
        """Return one ``[frames, codebooks]`` tensor per waveform."""
        if input_values.ndim == 3:
            if input_values.shape[1] != 1:
                raise ValueError("Qwen3-TTS encoder expects mono audio.")
            unbatched_channels = input_values[:, 0]
        elif input_values.ndim == 2:
            unbatched_channels = input_values
        else:
            raise ValueError("Qwen3-TTS encoder input must be [batch, samples] "
                             "or [batch, 1, samples].")
        if padding_mask is None:
            padding_mask = torch.ones_like(
                unbatched_channels,
                dtype=torch.bool,
            )
        if not isinstance(padding_mask, Tensor):
            raise TypeError("Qwen3-TTS encoder padding mask must be a tensor.")
        if padding_mask.shape != unbatched_channels.shape:
            raise ValueError("Qwen3-TTS encoder padding mask shape is invalid.")
        if padding_mask.dtype != torch.bool and bool(((padding_mask != 0) & (padding_mask != 1)).any()):
            raise ValueError("Qwen3-TTS encoder padding mask must be binary.")
        mask = padding_mask.bool()
        if mask.shape[-1] > 1 and bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("Qwen3-TTS encoder requires right-padded audio.")
        lengths = mask.sum(dim=-1)
        if bool((lengths == 0).any()):
            raise ValueError("Qwen3-TTS encoder audio cannot be fully padded.")
        codes = self.forward(input_values)
        frame_lengths = (lengths + self.config.total_downsample - 1) // self.config.total_downsample
        return [code[:, :length].transpose(0, 1) for code, length in zip(codes, frame_lengths.tolist())]


def materialize_qwen3_tts_encoder_buffers(
    encoder: nn.Module,
    *,
    device: str | torch.device,
) -> None:
    """Materialize non-persistent RoPE buffers after meta checkpoint load."""
    for module in encoder.modules():
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
    "Qwen3TTSSpeechEncoder",
    "materialize_qwen3_tts_encoder_buffers",
]
