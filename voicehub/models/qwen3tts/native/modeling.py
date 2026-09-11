"""VoiceHub-owned Qwen3-TTS talker and speaker encoder.

This is a clean-room PyTorch implementation of the published 12 Hz graph
reviewed at the immutable upstream revision recorded in ``SOURCE.json``.
The module intentionally exposes the same persistent tensor namespace as
the official Safetensors checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.kernels import KernelBackend, gated_silu
from voicehub.models.qwen3tts.native.configuration import (
    Qwen3TTSArchitectureConfig,
    Qwen3TTSCodePredictorConfig,
    Qwen3TTSSpeakerEncoderConfig,
    Qwen3TTSTalkerConfig,
)
from voicehub.neural.backends import FlashAttention4Policy, flash_attention4_or_sdpa
from voicehub.neural.normalization import RMSNorm
from voicehub.neural.rotary import RotaryEmbedding, apply_rotary_embedding
from voicehub.objectives.sequence import sequence_cross_entropy


@dataclass(frozen=True, slots=True)
class Qwen3TTSTalkerOutput:
    logits: Tensor
    last_hidden_state: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[tuple[Tensor, ...], Tensor | None] | None = None


@dataclass(frozen=True, slots=True)
class Qwen3TTSCodePredictorOutput:
    logits: Tensor
    loss: Tensor | None = None


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


def _reflect_same_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    dilation: int = 1,
    factory_kwargs: dict[str, Any],
) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        dilation=dilation,
        padding="same",
        padding_mode="reflect",
        **factory_kwargs,
    )


class TimeDelayNetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.conv = _reflect_same_conv(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            factory_kwargs=factory_kwargs,
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.activation(self.conv(hidden_states))


class Res2NetBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        *,
        scale: int,
        kernel_size: int,
        dilation: int,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        split_channels = channels // scale
        self.blocks = nn.ModuleList([
            TimeDelayNetBlock(
                split_channels,
                split_channels,
                kernel_size,
                dilation,
                factory_kwargs=factory_kwargs,
            ) for _ in range(scale - 1)
        ])
        self.scale = scale

    def forward(self, hidden_states: Tensor) -> Tensor:
        outputs: list[Tensor] = []
        previous: Tensor | None = None
        for index, part in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if index == 0:
                output = part
            elif index == 1:
                output = self.blocks[index - 1](part)
            else:
                assert previous is not None
                output = self.blocks[index - 1](part + previous)
            outputs.append(output)
            previous = output
        return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        se_channels: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.conv1 = _reflect_same_conv(
            channels,
            se_channels,
            1,
            factory_kwargs=factory_kwargs,
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _reflect_same_conv(
            se_channels,
            channels,
            1,
            factory_kwargs=factory_kwargs,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        weights = hidden_states.mean(dim=2, keepdim=True)
        weights = torch.sigmoid(self.conv2(self.relu(self.conv1(weights))))
        return hidden_states * weights


class SqueezeExcitationRes2NetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        scale: int,
        se_channels: int,
        kernel_size: int,
        dilation: int,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.tdnn1 = TimeDelayNetBlock(
            in_channels,
            out_channels,
            1,
            1,
            factory_kwargs=factory_kwargs,
        )
        self.res2net_block = Res2NetBlock(
            out_channels,
            scale=scale,
            kernel_size=kernel_size,
            dilation=dilation,
            factory_kwargs=factory_kwargs,
        )
        self.tdnn2 = TimeDelayNetBlock(
            out_channels,
            out_channels,
            1,
            1,
            factory_kwargs=factory_kwargs,
        )
        self.se_block = SqueezeExcitationBlock(
            out_channels,
            se_channels,
            factory_kwargs=factory_kwargs,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.tdnn1(hidden_states)
        hidden_states = self.res2net_block(hidden_states)
        hidden_states = self.tdnn2(hidden_states)
        return self.se_block(hidden_states) + residual


class AttentiveStatisticsPooling(nn.Module):

    def __init__(
        self,
        channels: int,
        attention_channels: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.epsilon = 1e-12
        self.tdnn = TimeDelayNetBlock(
            channels * 3,
            attention_channels,
            1,
            1,
            factory_kwargs=factory_kwargs,
        )
        self.conv = _reflect_same_conv(
            attention_channels,
            channels,
            1,
            factory_kwargs=factory_kwargs,
        )

    def _statistics(self, inputs: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
        mean = (weights * inputs).sum(dim=2)
        variance = (weights * (inputs - mean.unsqueeze(2)).square()).sum(dim=2)
        return mean, torch.sqrt(variance.clamp_min(self.epsilon))

    def forward(self, hidden_states: Tensor) -> Tensor:
        time = hidden_states.shape[-1]
        uniform = torch.full(
            (hidden_states.shape[0], 1, time),
            1.0 / time,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        mean, std = self._statistics(hidden_states, uniform)
        context = torch.cat(
            (
                hidden_states,
                mean.unsqueeze(2).expand(-1, -1, time),
                std.unsqueeze(2).expand(-1, -1, time),
            ),
            dim=1,
        )
        attention = torch.softmax(
            self.conv(torch.tanh(self.tdnn(context))),
            dim=2,
        )
        mean, std = self._statistics(hidden_states, attention)
        return torch.cat((mean, std), dim=1).unsqueeze(2)


class Qwen3TTSSpeakerEncoder(nn.Module):
    """Exact ECAPA-TDNN speaker encoder used by Base checkpoints."""

    def __init__(
        self,
        config: Qwen3TTSSpeakerEncoderConfig,
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
        self.blocks = nn.ModuleList([
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
                factory_kwargs=factory_kwargs,
            )
        ])
        for index in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[index - 1],
                    config.enc_channels[index],
                    scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[index],
                    dilation=config.enc_dilations[index],
                    factory_kwargs=factory_kwargs,
                ))
        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
            factory_kwargs=factory_kwargs,
        )
        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            config.enc_attention_channels,
            factory_kwargs=factory_kwargs,
        )
        self.fc = _reflect_same_conv(
            config.enc_channels[-1] * 2,
            config.enc_dim,
            1,
            factory_kwargs=factory_kwargs,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.mel_dim:
            raise ValueError("Speaker mel features must have shape [batch, frames, mel_dim].")
        hidden_states = hidden_states.transpose(1, 2)
        intermediate = []
        for block in self.blocks:
            hidden_states = block(hidden_states)
            intermediate.append(hidden_states)
        hidden_states = self.mfa(torch.cat(intermediate[1:], dim=1))
        return self.fc(self.asp(hidden_states)).squeeze(-1)


def _expand_kv(hidden_states: Tensor, groups: int) -> Tensor:
    if groups == 1:
        return hidden_states
    batch, heads, time, dimension = hidden_states.shape
    return (
        hidden_states[:, :, None].expand(batch, heads, groups, time,
                                         dimension).reshape(batch, heads * groups, time, dimension))


_SDPA_SUPPORTS_GQA = "enable_gqa" in (functional.scaled_dot_product_attention.__doc__ or "")


def _scaled_dot_product_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        attention_bias: Tensor | None,
        scale: float,
        dropout_p: float,
        groups: int,
        is_causal: bool = False,
        flash_attention4_policy: FlashAttention4Policy | str = (FlashAttention4Policy.DISABLED),
) -> Tensor:
    """Run the selected exact-attention backend with a safe SDPA fallback.

    Native grouped-query attention avoids materializing repeated
    key/value heads on recent PyTorch CUDA and math backends. Older
    PyTorch releases do not expose ``enable_gqa``; MPS likewise needs
    the established eager head expansion. Both paths still use PyTorch
    SDPA for the attention calculation. FlashAttention-4 is an explicit
    opt-in (or capability-gated ``auto`` policy) and receives the same
    canonical Qwen scale.
    """
    policy = FlashAttention4Policy.coerce(flash_attention4_policy)
    mask = (None if attention_bias is None else attention_bias.to(device=query.device, dtype=query.dtype))
    if (policy is not FlashAttention4Policy.DISABLED or (is_causal and query.shape[-2] != key.shape[-2])):
        return flash_attention4_or_sdpa(
            query,
            key,
            value,
            attention_mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            policy=policy,
        )

    if groups == 1:
        return functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    if _SDPA_SUPPORTS_GQA and query.device.type != "mps":
        try:
            return functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=True,
            )
        except RuntimeError as error:
            # Some device/dtype combinations expose the keyword but have no
            # grouped-query kernel. Only compatibility failures fall back;
            # unrelated failures such as OOM must remain visible.
            message = str(error).lower()
            if not any(token in message for token in (
                    "grouped query",
                    "gqa",
                    "number of heads",
                    "num_heads",
                    "no available kernel",
            )):
                raise

    return functional.scaled_dot_product_attention(
        query,
        _expand_kv(key, groups),
        _expand_kv(value, groups),
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def _causal_bias(
    attention_mask: Tensor | None,
    *,
    batch: int,
    time: int,
    past_length: int = 0,
    sliding_window: int | None,
    device: torch.device,
) -> Tensor | None:
    if (attention_mask is None and sliding_window is None):
        return None
    key_length = past_length + time
    key_positions = torch.arange(key_length, device=device)
    query_positions = torch.arange(
        past_length,
        key_length,
        device=device,
    )
    allowed = key_positions[None, :] <= query_positions[:, None]
    if sliding_window is not None:
        allowed &= key_positions[None, :] > (query_positions[:, None] - sliding_window)
    allowed = allowed.view(1, 1, time, key_length).expand(
        batch,
        1,
        time,
        key_length,
    ).clone()
    if attention_mask is not None:
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (
                batch,
                key_length,
        ):
            raise ValueError("Attention mask must cover the complete cached sequence.")
        allowed &= attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
    bias = torch.zeros(
        (batch, 1, time, key_length),
        device=device,
        dtype=torch.float32,
    )
    return bias.masked_fill(~allowed, torch.finfo(torch.float32).min)


class Qwen3TTSMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.kernel_backend = KernelBackend.TORCH
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            **factory_kwargs,
        )

    def set_kernel_backend(self, backend: KernelBackend | str) -> None:
        """Select a registered fused SwiGLU implementation."""
        self.kernel_backend = KernelBackend.coerce(backend)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(
            gated_silu(
                self.gate_proj(hidden_states),
                self.up_proj(hidden_states),
                backend=self.kernel_backend,
            ))


class Qwen3TTSSelfAttention(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig | Qwen3TTSCodePredictorConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.flash_attention4_policy = FlashAttention4Policy.DISABLED
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
        self.q_norm = RMSNorm(
            self.head_dim,
            epsilon=config.rms_norm_eps,
            **factory_kwargs,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            epsilon=config.rms_norm_eps,
            **factory_kwargs,
        )

    def set_flash_attention4_policy(
        self,
        policy: FlashAttention4Policy | str,
    ) -> None:
        """Select disabled, automatic, or required FlashAttention-4."""
        self.flash_attention4_policy = FlashAttention4Policy.coerce(policy)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cosine: Tensor,
        sine: Tensor,
        attention_bias: Tensor | None,
    ) -> Tensor:
        output, _ = self.forward_with_cache(
            hidden_states,
            cosine=cosine,
            sine=sine,
            attention_bias=attention_bias,
        )
        return output

    def forward_with_cache(
        self,
        hidden_states: Tensor,
        *,
        cosine: Tensor,
        sine: Tensor,
        attention_bias: Tensor | None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, time, _ = hidden_states.shape
        query = self.q_norm(self.q_proj(hidden_states).view(
            batch,
            time,
            self.num_heads,
            self.head_dim,
        )).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(
            batch,
            time,
            self.num_kv_heads,
            self.head_dim,
        )).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch,
            time,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)
        query, key = apply_rotary_embedding(query, key, cosine, sine)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        present = (key, value)
        output = _scaled_dot_product_attention(
            query,
            key,
            value,
            attention_bias=attention_bias,
            scale=self.scale,
            dropout_p=self.dropout if self.training else 0.0,
            groups=self.groups,
            is_causal=attention_bias is None,
            flash_attention4_policy=self.flash_attention4_policy,
        )
        output = output.transpose(1, 2).reshape(
            batch,
            time,
            self.num_heads * self.head_dim,
        )
        return self.o_proj(output), present


class Qwen3TTSDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig | Qwen3TTSCodePredictorConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3TTSSelfAttention(
            config,
            factory_kwargs=factory_kwargs,
        )
        self.mlp = Qwen3TTSMLP(
            config.hidden_size,
            config.intermediate_size,
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

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cosine: Tensor,
        sine: Tensor,
        attention_bias: Tensor | None,
    ) -> Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            cosine=cosine,
            sine=sine,
            attention_bias=attention_bias,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

    def forward_with_cache(
        self,
        hidden_states: Tensor,
        *,
        cosine: Tensor,
        sine: Tensor,
        attention_bias: Tensor | None,
        past_key_value: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attention_output, present = self.self_attn.forward_with_cache(
            self.input_layernorm(hidden_states),
            cosine=cosine,
            sine=sine,
            attention_bias=attention_bias,
            past_key_value=past_key_value,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class Qwen3TTSDecoderBackbone(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig | Qwen3TTSCodePredictorConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Qwen3TTSDecoderLayer(config, factory_kwargs=factory_kwargs)
            for _ in range(config.num_hidden_layers)
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

    def _prepare_attention(
        self,
        inputs_embeds: Tensor,
        *,
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
        past_length: int,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if (inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.hidden_size):
            raise ValueError("Decoder embeddings must have shape [batch, time, hidden_size].")
        batch, time, _ = inputs_embeds.shape
        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(
                    past_length,
                    past_length + time,
                    device=inputs_embeds.device,
                ).unsqueeze(0).expand(batch, -1)
            else:
                position_ids = (attention_mask.to(dtype=torch.long).cumsum(-1).sub(1).clamp_min(0)[:, -time:])
        elif position_ids.ndim == 3:
            # The upstream mRoPE indices are identical for ordinary TTS text/
            # codec sequences. Select the temporal stream after validating it.
            if position_ids.shape[0] not in (3, 4):
                raise ValueError("mRoPE position IDs require three or four streams.")
            streams = position_ids[-3:]
            if not torch.equal(streams[0], streams[1]) or not torch.equal(
                    streams[0],
                    streams[2],
            ):
                raise ValueError(
                    "Native Qwen3-TTS currently accepts equal temporal, height, "
                    "and width mRoPE indices only.")
            position_ids = streams[0]
        if tuple(position_ids.shape) != (batch, time):
            raise ValueError("Position IDs must have shape [batch, time].")
        cosine, sine = self.rotary_emb(
            position_ids,
            dtype=inputs_embeds.dtype,
        )
        bias = _causal_bias(
            attention_mask,
            batch=batch,
            time=time,
            past_length=past_length,
            sliding_window=None,
            device=inputs_embeds.device,
        )
        return cosine, sine, bias

    def forward(
        self,
        inputs_embeds: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, ...] | None]:
        cosine, sine, bias = self._prepare_attention(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_length=0,
        )
        hidden_states = inputs_embeds
        history: list[Tensor] | None = [] if output_hidden_states else None
        if history is not None:
            history.append(hidden_states)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cosine=cosine,
                sine=sine,
                attention_bias=bias,
            )
            if history is not None:
                history.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        if history is not None:
            history[-1] = hidden_states
        return hidden_states, None if history is None else tuple(history)

    def forward_with_cache(
        self,
        inputs_embeds: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None,
    ) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]]:
        if (inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.hidden_size):
            raise ValueError("Decoder embeddings must have shape [batch, time, hidden_size].")
        batch = inputs_embeds.shape[0]
        if past_key_values is None:
            layer_pasts: tuple[tuple[Tensor, Tensor] | None, ...] = (None, ) * len(self.layers)
            past_length = 0
        else:
            if len(past_key_values) != len(self.layers):
                raise ValueError("The Qwen3-TTS cache must contain one entry per layer.")
            layer_pasts = past_key_values
            lengths: set[int] = set()
            for layer, (key, value) in zip(self.layers, past_key_values):
                expected_prefix = (
                    batch,
                    layer.self_attn.num_kv_heads,
                )
                if (key.ndim != 4 or value.shape != key.shape or key.shape[:2] != expected_prefix or
                        key.shape[-1] != layer.self_attn.head_dim):
                    raise ValueError("A Qwen3-TTS cache entry has an incompatible shape.")
                if (key.device != inputs_embeds.device or value.device != inputs_embeds.device or
                        key.dtype != inputs_embeds.dtype or value.dtype != inputs_embeds.dtype):
                    raise ValueError(
                        "Qwen3-TTS cache tensors must match the current "
                        "embedding device and dtype.")
                lengths.add(key.shape[2])
            if len(lengths) != 1:
                raise ValueError("Every Qwen3-TTS cache layer must have the same length.")
            past_length = lengths.pop()
        cosine, sine, bias = self._prepare_attention(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_length=past_length,
        )
        hidden_states = inputs_embeds
        present_key_values: list[tuple[Tensor, Tensor]] = []
        for layer, layer_past in zip(self.layers, layer_pasts):
            hidden_states, present = layer.forward_with_cache(
                hidden_states,
                cosine=cosine,
                sine=sine,
                attention_bias=bias,
                past_key_value=layer_past,
            )
            present_key_values.append(present)
        return self.norm(hidden_states), tuple(present_key_values)


class Qwen3TTSTalkerModel(Qwen3TTSDecoderBackbone):

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(config, factory_kwargs=factory_kwargs)
        self.codec_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            **factory_kwargs,
        )
        self.text_embedding = nn.Embedding(
            config.text_vocab_size,
            config.text_hidden_size,
            **factory_kwargs,
        )


class Qwen3TTSCodePredictorModel(Qwen3TTSDecoderBackbone):

    def __init__(
        self,
        config: Qwen3TTSCodePredictorConfig,
        embedding_dim: int,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(config, factory_kwargs=factory_kwargs)
        self.codec_embedding = nn.ModuleList([
            nn.Embedding(
                config.vocab_size,
                embedding_dim,
                **factory_kwargs,
            ) for _ in range(config.num_code_groups - 1)
        ])


class Qwen3TTSCodePredictor(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSCodePredictorConfig,
        *,
        talker_hidden_size: int,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3TTSCodePredictorModel(
            config,
            talker_hidden_size,
            factory_kwargs=factory_kwargs,
        )
        self.lm_head = nn.ModuleList([
            nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                **factory_kwargs,
            ) for _ in range(config.num_code_groups - 1)
        ])
        self.small_to_mtp_projection: nn.Module
        if talker_hidden_size == config.hidden_size:
            self.small_to_mtp_projection = nn.Identity()
        else:
            self.small_to_mtp_projection = nn.Linear(
                talker_hidden_size,
                config.hidden_size,
                bias=True,
                **factory_kwargs,
            )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.codec_embedding

    def forward_finetune(
        self,
        *,
        inputs_embeds: Tensor,
        labels: Tensor,
    ) -> Qwen3TTSCodePredictorOutput:
        projected = self.small_to_mtp_projection(inputs_embeds)
        hidden_states, _ = self.model(projected)
        if hidden_states.shape[1] != self.config.num_code_groups:
            raise ValueError("Code predictor fine-tuning expects one hidden state per codebook.")
        logits = torch.stack([head(hidden_states[:, index + 1]) for index, head in enumerate(self.lm_head)],
                             dim=1)
        if labels.shape != logits.shape[:-1]:
            raise ValueError("Code predictor labels must cover residual codebooks.")
        loss = sequence_cross_entropy(logits, labels)
        return Qwen3TTSCodePredictorOutput(logits=logits, loss=loss)


class Qwen3TTSTextProjection(nn.Module):

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig,
        *,
        factory_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(
            config.text_hidden_size,
            config.text_hidden_size,
            bias=True,
            **factory_kwargs,
        )
        self.linear_fc2 = nn.Linear(
            config.text_hidden_size,
            config.hidden_size,
            bias=True,
            **factory_kwargs,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.linear_fc2(functional.silu(self.linear_fc1(hidden_states)))


def _sample_token(
    logits: Tensor,
    *,
    do_sample: bool,
    top_k: int,
    top_p: float,
    temperature: float,
    generator: torch.Generator | None,
) -> Tensor:
    if not do_sample:
        return logits.argmax(dim=-1)
    scores = logits.float() / temperature
    if top_k > 0 and top_k < scores.shape[-1]:
        threshold = torch.topk(scores, top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(
            scores,
            descending=True,
            dim=-1,
        )
        probabilities = torch.softmax(sorted_scores, dim=-1)
        remove = probabilities.cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(
            -1,
            sorted_indices,
            sorted_scores,
        )
    probabilities = torch.softmax(scores, dim=-1)
    sampling_probabilities = probabilities
    if (generator is not None and generator.device.type != probabilities.device.type):
        sampling_probabilities = probabilities.cpu()
    sampled = torch.multinomial(
        sampling_probabilities,
        num_samples=1,
        generator=generator,
    ).squeeze(-1)
    return sampled.to(device=probabilities.device)


class Qwen3TTSTalker(nn.Module):
    """Autoregressive first-codebook talker plus residual code predictor."""

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig,
        *,
        tts_pad_token_id: int,
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
        if not 0 <= tts_pad_token_id < config.text_vocab_size:
            raise ValueError("The TTS pad token must fit the text vocabulary.")
        self.tts_pad_token_id = tts_pad_token_id
        self.model = Qwen3TTSTalkerModel(
            config,
            factory_kwargs=factory_kwargs,
        )
        self.text_projection = Qwen3TTSTextProjection(
            config,
            factory_kwargs=factory_kwargs,
        )
        self.codec_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            **factory_kwargs,
        )
        self.code_predictor = Qwen3TTSCodePredictor(
            config.code_predictor_config,
            talker_hidden_size=config.hidden_size,
            factory_kwargs=factory_kwargs,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.codec_embedding

    def get_text_embeddings(self) -> nn.Embedding:
        return self.model.text_embedding

    def forward(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_hidden_states: bool = False,
        position_ids: Tensor | None = None,
        **_: Any,
    ) -> Qwen3TTSTalkerOutput:
        hidden_states, history = self.model(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        logits = self.codec_head(hidden_states)
        loss = None
        if labels is not None:
            if labels.shape != logits.shape[:-1]:
                raise ValueError("Talker labels must have shape [batch, time].")
            loss = sequence_cross_entropy(
                logits[:, :-1],
                labels[:, 1:],
            )
        wrapped_history = (None if history is None else (history, None))
        return Qwen3TTSTalkerOutput(
            logits=logits,
            last_hidden_state=hidden_states,
            loss=loss,
            hidden_states=wrapped_history,
        )

    def forward_sub_talker_finetune(
        self,
        codec_ids: Tensor,
        talker_hidden_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (codec_ids.ndim != 2 or codec_ids.shape[1] != self.config.num_code_groups or
                talker_hidden_states.ndim != 2 or talker_hidden_states.shape[0] != codec_ids.shape[0] or
                talker_hidden_states.shape[1] != self.config.hidden_size):
            raise ValueError(
                "Sub-talker fine-tuning expects codes [frames, codebooks] "
                "and hidden states [frames, hidden_size].")
        embeddings = [
            talker_hidden_states.unsqueeze(1),
            self.get_input_embeddings()(codec_ids[:, :1]),
        ]
        for index, table in enumerate(
                self.code_predictor.get_input_embeddings()[:-1],
                start=1,
        ):
            embeddings.append(table(codec_ids[:, index:index + 1]))
        predictor = self.code_predictor.forward_finetune(
            inputs_embeds=torch.cat(embeddings, dim=1),
            labels=codec_ids[:, 1:],
        )
        assert predictor.loss is not None
        return predictor.logits, predictor.loss

    def _residual_codes(
        self,
        hidden: Tensor,
        first_code: Tensor,
        *,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        codes = [first_code]
        embeddings = [
            hidden.unsqueeze(1),
            self.get_input_embeddings()(first_code[:, None]),
        ]
        projected = self.code_predictor.small_to_mtp_projection(torch.cat(embeddings, dim=1))
        states, past_key_values = self.code_predictor.model.forward_with_cache(projected)
        for index, (table, head) in enumerate(zip(
                self.code_predictor.get_input_embeddings(),
                self.code_predictor.lm_head,
        )):
            code = _sample_token(
                head(states[:, -1]),
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                generator=generator,
            )
            codes.append(code)
            if index + 1 < len(self.code_predictor.lm_head):
                next_embedding = self.code_predictor.small_to_mtp_projection(table(code[:, None]))
                states, past_key_values = (
                    self.code_predictor.model.forward_with_cache(
                        next_embedding,
                        past_key_values=past_key_values,
                    ))
        return torch.stack(codes, dim=-1)

    @torch.no_grad()
    def generate_codes(
        self,
        *,
        prompt_embeds: Tensor,
        attention_mask: Tensor | None,
        trailing_text_hidden: Tensor,
        max_new_tokens: int,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        seed: int | None = None,
    ) -> Tensor:
        if prompt_embeds.shape[0] != 1:
            raise ValueError("Native Qwen3-TTS generation currently accepts batch size one.")
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        if not isinstance(do_sample, bool) or not isinstance(
                subtalker_dosample,
                bool,
        ):
            raise TypeError("Qwen3-TTS sampling flags must be boolean.")
        for name, value in (
            ("top_k", top_k),
            ("subtalker_top_k", subtalker_top_k),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"`{name}` must be a non-negative integer.")
        for name, value in (
            ("top_p", top_p),
            ("subtalker_top_p", subtalker_top_p),
        ):
            if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or
                    not 0 < value <= 1):
                raise ValueError(f"`{name}` must be in (0, 1].")
        continuous_values = (
            temperature,
            subtalker_temperature,
            repetition_penalty,
        )
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or
               value <= 0 for value in continuous_values):
            raise ValueError("Temperatures and repetition penalty must be finite and "
                             "positive.")
        generator = None
        if seed is not None:
            generator_device = (
                prompt_embeds.device if prompt_embeds.device.type in {"cpu", "cuda"} else torch.device("cpu"))
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(seed)
        mask = attention_mask
        generated: list[Tensor] = []
        states, past_key_values = self.model.forward_with_cache(
            prompt_embeds,
            attention_mask=mask,
        )
        hidden = states[:, -1]
        for step in range(max_new_tokens):
            logits = self.codec_head(hidden)
            if generated and repetition_penalty != 1:
                previous = torch.stack(generated, dim=1)[..., 0]
                selected = logits.gather(1, previous)
                adjusted = torch.where(
                    selected < 0,
                    selected * repetition_penalty,
                    selected / repetition_penalty,
                )
                logits.scatter_(1, previous, adjusted)
            suppress_start = self.config.vocab_size - 1024
            eos_score = logits[:, self.config.codec_eos_token_id].clone()
            logits[:, suppress_start:] = float("-inf")
            if step >= 2:
                logits[:, self.config.codec_eos_token_id] = eos_score
            first = _sample_token(
                logits,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                generator=generator,
            )
            codes = self._residual_codes(
                hidden,
                first,
                do_sample=subtalker_dosample,
                top_k=subtalker_top_k,
                top_p=subtalker_top_p,
                temperature=subtalker_temperature,
                generator=generator,
            )
            if bool((first == self.config.codec_eos_token_id).all()):
                break
            generated.append(codes)
            codec_embedding = self.get_input_embeddings()(codes[:, 0])
            for index, table in enumerate(
                    self.code_predictor.get_input_embeddings(),
                    start=1,
            ):
                codec_embedding = codec_embedding + table(codes[:, index])
            if step < trailing_text_hidden.shape[1]:
                codec_embedding = codec_embedding + trailing_text_hidden[:, step]
            else:
                codec_embedding = (
                    codec_embedding + self.text_projection(
                        self.get_text_embeddings()(
                            torch.full(
                                (1, ),
                                self.tts_pad_token_id,
                                device=prompt_embeds.device,
                                dtype=torch.long,
                            ))))
            if mask is not None:
                mask = functional.pad(mask, (0, 1), value=1)
            if step + 1 < max_new_tokens:
                states, past_key_values = self.model.forward_with_cache(
                    codec_embedding.unsqueeze(1),
                    attention_mask=mask,
                    past_key_values=past_key_values,
                )
                hidden = states[:, -1]
        if not generated:
            return torch.empty(
                (0, self.config.num_code_groups),
                dtype=torch.long,
                device=prompt_embeds.device,
            )
        return torch.cat(generated, dim=0)


class Qwen3TTSForConditionalGeneration(nn.Module):
    """Official talker graph with optional Base-checkpoint speaker encoder."""

    def __init__(
        self,
        config: Qwen3TTSArchitectureConfig,
        *,
        initialize: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.talker = Qwen3TTSTalker(
            config.talker_config,
            tts_pad_token_id=config.tts_pad_token_id,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.speaker_encoder = (
            Qwen3TTSSpeakerEncoder(
                config.speaker_encoder_config,
                initialize=initialize,
                device=device,
                dtype=dtype,
            ) if config.tts_model_type == "base" else None)
        self.tts_model_type = config.tts_model_type
        self.tts_model_size = config.tts_model_size
        self.tokenizer_type = config.tokenizer_type
        self._runtime_owner: Any | None = None
        self.speech_tokenizer: Any | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_supported_speakers(self) -> tuple[str, ...]:
        return tuple(sorted(self.config.talker_config.spk_id))

    def get_supported_languages(self) -> tuple[str, ...]:
        return ("auto", *sorted(self.config.talker_config.codec_language_id))

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        state_dict: dict[str, Tensor] | None = None,
        safe_serialization: bool = True,
    ) -> Any:
        """Delegate portable export to the owning native runtime."""
        if safe_serialization is not True:
            raise ValueError("Native Qwen3-TTS export is Safetensors-only.")
        if self._runtime_owner is None:
            raise RuntimeError("Qwen3-TTS portable export requires its loaded native runtime.")
        return self._runtime_owner.save_pretrained(
            directory,
            model_state_override=state_dict,
        )


def materialize_qwen3_tts_buffers(
    model: nn.Module,
    *,
    device: str | torch.device,
) -> None:
    """Move non-persistent RoPE buffers off ``meta`` after assign loading."""
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            inverse = 1.0 / (
                module.base**(
                    torch.arange(
                        0,
                        module.dimension,
                        2,
                        dtype=torch.float32,
                        device=device,
                    ) / module.dimension))
            module.inverse_frequency = inverse


__all__ = [
    "Qwen3TTSCodePredictorOutput",
    "Qwen3TTSForConditionalGeneration",
    "Qwen3TTSSpeakerEncoder",
    "Qwen3TTSTalker",
    "Qwen3TTSTalkerOutput",
    "materialize_qwen3_tts_buffers",
]
