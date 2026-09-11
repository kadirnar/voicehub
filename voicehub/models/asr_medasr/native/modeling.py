"""VoiceHub-owned PyTorch implementation of LASR CTC for MedASR."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from voicehub.models.asr_medasr.native.configuration import MedASRConfig
from voicehub.models.asr_medasr.native.frontend import lengths_to_mask, subsampled_lengths


def _validate_features(
    input_features: Tensor,
    attention_mask: Tensor | None,
    config: MedASRConfig,
) -> tuple[Tensor, Tensor]:
    if not isinstance(input_features, Tensor):
        raise TypeError("`input_features` must be a PyTorch tensor.")
    if input_features.ndim != 3:
        raise ValueError("`input_features` must have shape [batch, frames, mel_bins].")
    if input_features.shape[-1] != config.num_mel_bins:
        raise ValueError(
            f"MedASR expects {config.num_mel_bins} mel bins, found "
            f"{input_features.shape[-1]}.")
    if input_features.shape[0] < 1:
        raise ValueError("A MedASR batch cannot be empty.")
    if input_features.shape[1] < config.minimum_feature_frames:
        raise ValueError("MedASR requires at least "
                         f"{config.minimum_feature_frames} feature frames.")
    if not input_features.is_floating_point():
        raise TypeError("`input_features` must use a floating-point dtype.")
    if not torch.isfinite(input_features).all():
        raise ValueError("`input_features` cannot contain NaN or infinite values.")
    batch, frames, _ = input_features.shape
    if attention_mask is None:
        mask = torch.ones(
            (batch, frames),
            dtype=torch.bool,
            device=input_features.device,
        )
    else:
        if not isinstance(attention_mask, Tensor):
            raise TypeError("`attention_mask` must be a PyTorch tensor.")
        if tuple(attention_mask.shape) != (batch, frames):
            raise ValueError("`attention_mask` must match [batch, feature_frames].")
        if attention_mask.device != input_features.device:
            raise ValueError("`attention_mask` and `input_features` must share a device.")
        if attention_mask.is_complex():
            raise TypeError("`attention_mask` cannot use a complex dtype.")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("`attention_mask` must contain only zero and one.")
        mask = attention_mask.to(dtype=torch.bool)
    if frames > 1 and ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ValueError("`attention_mask` must describe right-padded audio.")
    lengths = mask.sum(dim=-1, dtype=torch.long)
    if (lengths < config.minimum_feature_frames).any():
        raise ValueError("Every utterance must contain enough valid frames for the LASR "
                         "subsampler.")
    return mask, lengths


class MedASRSubsampling(nn.Module):
    """Linear projection followed by the released two Conv1d subsampler."""

    def __init__(self, config: MedASRConfig) -> None:
        super().__init__()
        self.dense_0 = nn.Linear(config.num_mel_bins, config.hidden_size)
        self.conv_0 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
        )
        self.conv_1 = nn.Conv1d(
            config.hidden_size,
            config.subsampling_conv_channels,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
        )
        self.dense_1 = nn.Linear(
            config.subsampling_conv_channels,
            config.hidden_size,
        )

    def forward(self, input_features: Tensor) -> Tensor:
        hidden_states = functional.relu(self.dense_0(input_features))
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = functional.relu(self.conv_0(hidden_states))
        hidden_states = functional.relu(self.conv_1(hidden_states))
        return self.dense_1(hidden_states.transpose(1, 2))


def _rotary_embeddings(
    hidden_states: Tensor,
    config: MedASRConfig,
) -> tuple[Tensor, Tensor]:
    head_dimension = config.hidden_size // config.num_attention_heads
    inverse_frequency = 1.0 / (
        config.rope_theta**(
            torch.arange(
                0,
                head_dimension,
                2,
                dtype=torch.float32,
                device=hidden_states.device,
            ) / head_dimension))
    positions = torch.arange(
        hidden_states.shape[1],
        dtype=torch.float32,
        device=hidden_states.device,
    )
    frequencies = torch.outer(positions, inverse_frequency)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return (
        embedding.cos().unsqueeze(0).to(dtype=hidden_states.dtype),
        embedding.sin().unsqueeze(0).to(dtype=hidden_states.dtype),
    )


def _rotate_half(value: Tensor) -> Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    query: Tensor,
    key: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> tuple[Tensor, Tensor]:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


class MedASRAttention(nn.Module):
    """Bidirectional multi-head self-attention with default RoPE."""

    def __init__(self, config: MedASRConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dimension = config.hidden_size // self.heads
        self.scale = self.head_dimension**-0.5
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        key_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        batch, frames, _ = hidden_states.shape

        def split(value: Tensor) -> Tensor:
            return value.reshape(
                batch,
                frames,
                self.heads,
                self.head_dimension,
            ).transpose(1, 2)

        query = split(self.q_proj(hidden_states))
        key = split(self.k_proj(hidden_states))
        value = split(self.v_proj(hidden_states))
        query, key = _apply_rotary(
            query,
            key,
            *position_embeddings,
        )
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(
            ~key_mask[:, None, None, :],
            -torch.inf,
        )
        probabilities = torch.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        ).to(dtype=query.dtype)
        probabilities = functional.dropout(
            probabilities,
            p=self.dropout,
            training=self.training,
        )
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch,
            frames,
            self.heads * self.head_dimension,
        )
        return self.o_proj(attended)


class MedASRConvolutionModule(nn.Module):
    """Conformer GLU/depthwise-convolution branch."""

    def __init__(self, config: MedASRConfig) -> None:
        super().__init__()
        channels = config.hidden_size
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            bias=config.convolution_bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=config.conv_kernel_size,
            padding="same",
            groups=channels,
            bias=config.convolution_bias,
        )
        self.norm = nn.BatchNorm1d(
            channels,
            momentum=config.batch_norm_momentum,
        )
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            bias=config.convolution_bias,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        frame_mask: Tensor,
    ) -> Tensor:
        hidden_states = self.pointwise_conv1(hidden_states.transpose(1, 2), )
        hidden_states = functional.glu(hidden_states, dim=1)
        hidden_states = hidden_states.masked_fill(
            ~frame_mask.unsqueeze(1),
            0.0,
        )
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = functional.silu(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        return hidden_states.transpose(1, 2)


class MedASRFeedForward(nn.Module):

    def __init__(self, config: MedASRConfig) -> None:
        super().__init__()
        self.linear1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.attention_bias,
        )
        self.linear2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.dropout = config.activation_dropout

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = functional.silu(self.linear1(hidden_states))
        hidden_states = functional.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        return self.linear2(hidden_states)


class MedASREncoderBlock(nn.Module):
    """One checkpoint-compatible LASR Conformer block."""

    def __init__(self, config: MedASRConfig) -> None:
        super().__init__()
        self.feed_forward1 = MedASRFeedForward(config)
        self.self_attn = MedASRAttention(config)
        self.conv = MedASRConvolutionModule(config)
        self.feed_forward2 = MedASRFeedForward(config)
        self.norm_feed_forward1 = nn.LayerNorm(
            config.hidden_size,
            config.layer_norm_eps,
            bias=False,
        )
        self.norm_self_att = nn.LayerNorm(
            config.hidden_size,
            config.layer_norm_eps,
            bias=False,
        )
        self.norm_conv = nn.LayerNorm(
            config.hidden_size,
            config.layer_norm_eps,
            bias=False,
        )
        self.norm_feed_forward2 = nn.LayerNorm(
            config.hidden_size,
            config.layer_norm_eps,
            bias=False,
        )
        self.norm_out = nn.LayerNorm(
            config.hidden_size,
            config.layer_norm_eps,
            bias=False,
        )
        self.feed_forward_residual_weights = (config.feed_forward_residual_weights)
        self.conv_residual_weights = config.conv_residual_weights

    def forward(
        self,
        hidden_states: Tensor,
        frame_mask: Tensor,
        cosine: Tensor,
        sine: Tensor,
    ) -> Tensor:
        residual = hidden_states
        transformed = self.feed_forward1(self.norm_feed_forward1(hidden_states), )
        first, second = self.feed_forward_residual_weights
        hidden_states = first * residual + second * transformed
        hidden_states = hidden_states + self.self_attn(
            self.norm_self_att(hidden_states),
            key_mask=frame_mask,
            position_embeddings=(cosine, sine),
        )
        convolution = self.conv(
            self.norm_conv(hidden_states),
            frame_mask=frame_mask,
        )
        first, second = self.conv_residual_weights
        hidden_states = first * hidden_states + second * convolution
        residual = hidden_states
        transformed = self.feed_forward2(self.norm_feed_forward2(hidden_states), )
        first, second = self.feed_forward_residual_weights
        return self.norm_out(first * residual + second * transformed)


@dataclass(frozen=True, slots=True)
class MedASREncoderOutput:
    last_hidden_state: Tensor
    attention_mask: Tensor
    lengths: Tensor
    hidden_states: tuple[Tensor, ...] = ()


class MedASREncoder(nn.Module):
    """LASR encoder with exact output-length and padding semantics."""

    def __init__(self, config: MedASRConfig) -> None:
        super().__init__()
        self.config = config
        self.subsampler = MedASRSubsampling(config)
        self.layers = nn.ModuleList([MedASREncoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.out_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            bias=False,
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        input_features: Tensor,
        attention_mask: Tensor | None = None,
        *,
        output_hidden_states: bool = False,
    ) -> MedASREncoderOutput:
        _, input_lengths = _validate_features(
            input_features,
            attention_mask,
            self.config,
        )
        hidden_states = self.subsampler(input_features)
        output_lengths = subsampled_lengths(
            input_lengths,
            self.config,
        )
        output_mask = lengths_to_mask(
            output_lengths,
            hidden_states.shape[1],
        )
        cosine, sine = _rotary_embeddings(
            hidden_states,
            self.config,
        )
        hidden_states = functional.dropout(
            hidden_states,
            p=self.config.dropout,
            training=self.training,
        )
        cosine = functional.dropout(
            cosine,
            p=self.config.dropout_positions,
            training=self.training,
        )
        sine = functional.dropout(
            sine,
            p=self.config.dropout_positions,
            training=self.training,
        )
        captured: list[Tensor] = []
        for layer in self.layers:
            if (self.training and self.config.layerdrop and torch.rand(
                (),
                    device=hidden_states.device,
            ) < self.config.layerdrop):
                continue
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    output_mask,
                    cosine,
                    sine,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    output_mask,
                    cosine,
                    sine,
                )
            if output_hidden_states:
                captured.append(hidden_states)
        hidden_states = self.out_norm(hidden_states)
        return MedASREncoderOutput(
            last_hidden_state=hidden_states,
            attention_mask=output_mask,
            lengths=output_lengths,
            hidden_states=tuple(captured),
        )


@dataclass(frozen=True, slots=True)
class MedASRCTCOutput:
    loss: Tensor | None
    logits: Tensor
    encoded_lengths: Tensor
    hidden_states: tuple[Tensor, ...] = ()


class MedASRForCTC(nn.Module):
    """Checkpoint-compatible MedASR graph with native CTC fine-tuning."""

    def __init__(
        self,
        config: MedASRConfig | dict[str, Any] | None = None,
        *,
        initialize: bool = True,
    ) -> None:
        resolved = MedASRConfig.coerce(config)
        if not isinstance(initialize, bool):
            raise TypeError("`initialize` must be a boolean.")
        context = (torch.device("cpu") if initialize else torch.device("meta"))
        with context:
            super().__init__()
            self.config = resolved
            self.encoder = MedASREncoder(resolved)
            # Conv1d preserves the checkpoint namespace used by LASR.
            self.ctc_head = nn.Conv1d(
                resolved.hidden_size,
                resolved.vocab_size,
                kernel_size=1,
            )
        if initialize:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config.initializer_range,
                    a=-2 * self.config.initializer_range,
                    b=2 * self.config.initializer_range,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                module.running_mean.zero_()
                module.running_var.fill_(1)
                module.num_batches_tracked.zero_()

    @property
    def gradient_checkpointing(self) -> bool:
        return self.encoder.gradient_checkpointing

    def gradient_checkpointing_enable(self) -> None:
        self.encoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.encoder.gradient_checkpointing = False

    def forward(
        self,
        input_features: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        *,
        output_hidden_states: bool = False,
    ) -> MedASRCTCOutput:
        encoded = self.encoder(
            input_features,
            attention_mask,
            output_hidden_states=output_hidden_states,
        )
        logits = self.ctc_head(encoded.last_hidden_state.transpose(1, 2), ).transpose(1, 2)
        loss = None
        if labels is not None:
            if not isinstance(labels, Tensor):
                raise TypeError("`labels` must be a PyTorch tensor.")
            if labels.ndim != 2 or labels.shape[0] != logits.shape[0]:
                raise ValueError("`labels` must have shape [batch, target_tokens].")
            if (labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex()):
                raise TypeError("`labels` must use an integer dtype.")
            if labels.device != logits.device:
                raise ValueError("`labels` and model inputs must share a device.")
            if ((labels < 0) | (labels >= self.config.vocab_size)).any():
                raise ValueError("MedASR labels must be tokenizer IDs in "
                                 f"[0, {self.config.vocab_size}).")
            visible = labels != self.config.pad_token_id
            target_lengths = visible.sum(dim=-1, dtype=torch.long)
            if (target_lengths < 1).any():
                raise ValueError("Every MedASR transcript must contain at least one "
                                 "non-blank token.")
            targets = labels.masked_select(visible)
            log_probabilities = functional.log_softmax(
                logits,
                dim=-1,
                dtype=torch.float32,
            ).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss = functional.ctc_loss(
                    log_probabilities,
                    targets,
                    encoded.lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
        return MedASRCTCOutput(
            loss=loss,
            logits=logits,
            encoded_lengths=encoded.lengths,
            hidden_states=encoded.hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        input_features: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        outputs = self(
            input_features,
            attention_mask=attention_mask,
        )
        token_ids = outputs.logits.argmax(dim=-1)
        mask = lengths_to_mask(
            outputs.encoded_lengths,
            token_ids.shape[1],
        )
        return token_ids.masked_fill(
            ~mask,
            self.config.pad_token_id,
        )


__all__ = [
    "MedASRCTCOutput",
    "MedASREncoder",
    "MedASREncoderOutput",
    "MedASRForCTC",
]
