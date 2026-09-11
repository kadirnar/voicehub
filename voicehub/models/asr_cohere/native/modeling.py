"""Native PyTorch graph for Cohere Transcribe.

The module namespace intentionally matches the published
``CohereLabs/cohere-transcribe-03-2026`` Safetensors checkpoint.  The
FastConformer equations are the same Parakeet encoder family already
validated in VoiceHub, with Cohere's published 1280-wide, 48-layer
configuration and checkpoint naming retained exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from voicehub.models.asr_cohere.native.configuration import CohereAsrConfig
from voicehub.processing.audio import mel_filter_bank

LOG_ZERO_GUARD = 2**-24
NORMALIZATION_EPSILON = 1e-5


def _activation(name: str):
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported Cohere ASR activation {name!r}.")


@dataclass
class CohereEncoderOutput:
    """FastConformer hidden states and post-subsampling mask."""

    last_hidden_state: torch.Tensor
    attention_mask: torch.Tensor


@dataclass
class CohereAsrOutput:
    """Teacher-forced decoder output."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    encoder_last_hidden_state: torch.Tensor | None = None
    encoder_attention_mask: torch.Tensor | None = None


@dataclass
class CohereGenerateOutput:
    """Greedy autoregressive token sequences."""

    sequences: torch.Tensor


class FilterbankFeatures(nn.Module):
    """Persistent frontend buffers carried by the official checkpoint."""

    def __init__(self, config: CohereAsrConfig) -> None:
        super().__init__()
        window = torch.hann_window(
            config.win_length,
            periodic=False,
            dtype=torch.float32,
        )
        filters = mel_filter_bank(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.encoder_config.num_mel_bins,
            dtype=torch.float64,
            device=window.device,
        ).to(torch.float32)
        self.register_buffer("window", window)
        self.register_buffer("fb", filters.unsqueeze(0))


class CoherePreprocessor(nn.Module):
    """Checkpoint-compatible owner for the frontend buffers."""

    def __init__(self, config: CohereAsrConfig) -> None:
        super().__init__()
        self.featurizer = FilterbankFeatures(config)


class MaskedConvSequential(nn.Sequential):
    """Apply time masks between strided convolution stages."""

    @staticmethod
    def _mask(
        tensor: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        valid = (torch.arange(tensor.shape[2], device=tensor.device)[None, :] < lengths[:, None])
        return valid[:, None, :, None]

    @staticmethod
    def _output_lengths(
        lengths: torch.Tensor,
        layer: nn.Conv2d,
    ) -> torch.Tensor:
        return (lengths + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_lengths = lengths.to(torch.long)
        for layer in self:
            hidden_states = hidden_states * self._mask(
                hidden_states,
                current_lengths,
            )
            hidden_states = layer(hidden_states)
            if isinstance(layer, nn.Conv2d) and layer.stride[0] > 1:
                current_lengths = self._output_lengths(
                    current_lengths,
                    layer,
                )
        hidden_states = hidden_states * self._mask(
            hidden_states,
            current_lengths,
        )
        return hidden_states, current_lengths


class ConvSubsampling(nn.Module):
    """Published three-stage depthwise-separable 8x subsampler."""

    def __init__(self, config: CohereAsrConfig) -> None:
        super().__init__()
        encoder = config.encoder_config
        channels = encoder.subsampling_conv_channels
        self.conv = MaskedConvSequential(
            nn.Conv2d(
                1,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=channels,
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=channels,
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(),
        )
        output_frequency = (encoder.num_mel_bins // encoder.subsampling_factor)
        self.out = nn.Linear(
            channels * output_frequency,
            encoder.hidden_size,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = input_features.unsqueeze(1)
        hidden_states, lengths = self.conv(hidden_states, lengths)
        batch, channels, frames, frequency = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch,
            frames,
            channels * frequency,
        )
        return self.out(hidden_states), lengths


class RelPositionalEncoding(nn.Module):
    """Transformer-XL relative sinusoidal positions."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            10_000.0**(torch.arange(0, hidden_size, 2, dtype=torch.float32) / hidden_size))
        self.register_buffer(
            "_inverse_frequency",
            inverse_frequency,
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        length = hidden_states.shape[1]
        positions = torch.arange(
            length - 1,
            -length,
            -1,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        frequencies = torch.outer(
            positions,
            self._inverse_frequency.to(hidden_states.device),
        )
        positional = torch.stack(
            (frequencies.sin(), frequencies.cos()),
            dim=-1,
        ).reshape(2 * length - 1, -1)
        return positional.unsqueeze(0).to(dtype=hidden_states.dtype)


class ConformerFeedForward(nn.Module):
    """Half-step Conformer feed-forward branch."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = float(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.silu(self.linear1(hidden_states))
        hidden_states = F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        return self.linear2(hidden_states)


class ConformerConvolution(nn.Module):
    """GLU, depthwise convolution, BatchNorm, and SiLU branch."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(
            hidden_size,
            hidden_size * 2,
            kernel_size=1,
        )
        self.depthwise_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=(kernel_size - 1) // 2,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.pointwise_conv2 = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.glu(
            self.pointwise_conv1(hidden_states),
            dim=1,
        )
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(
                padding_mask[:, None, :],
                0.0,
            )
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = F.silu(hidden_states)
        return self.pointwise_conv2(hidden_states).transpose(1, 2)


class RelPositionMultiHeadAttention(nn.Module):
    """Checkpoint-compatible relative self-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = float(dropout)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_pos = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.pos_bias_u = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.zeros(num_heads, self.head_dim))

    @staticmethod
    def _relative_shift(scores: torch.Tensor) -> torch.Tensor:
        batch, heads, query_length, position_length = scores.shape
        scores = F.pad(scores, (1, 0))
        scores = scores.view(batch, heads, -1, query_length)
        return scores[:, :, 1:].view(
            batch,
            heads,
            query_length,
            position_length,
        )

    def _reshape(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(
            value.shape[0],
            value.shape[1],
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        invalid_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = hidden_states.shape[0]
        query = self._reshape(self.linear_q(hidden_states))
        key = self._reshape(self.linear_k(hidden_states))
        value = self._reshape(self.linear_v(hidden_states))
        if position_embeddings.shape[0] == 1 and batch > 1:
            position_embeddings = position_embeddings.expand(
                batch,
                -1,
                -1,
            )
        position = self._reshape(self.linear_pos(position_embeddings))
        content_scores = torch.matmul(
            query + self.pos_bias_u[None, :, None, :],
            key.transpose(-1, -2),
        )
        position_scores = torch.matmul(
            query + self.pos_bias_v[None, :, None, :],
            position.transpose(-1, -2),
        )
        position_scores = self._relative_shift(position_scores)
        position_scores = position_scores[..., :content_scores.shape[-1]]
        scores = (content_scores + position_scores) * self.scaling
        expanded_mask = None
        if invalid_attention_mask is not None:
            expanded_mask = invalid_attention_mask[:, None, :, :]
            scores = scores.masked_fill(expanded_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        if expanded_mask is not None:
            weights = weights.masked_fill(expanded_mask, 0.0)
        weights = F.dropout(
            weights,
            p=self.dropout,
            training=self.training,
        )
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            -1,
        )
        return self.linear_out(output)


class ConformerLayer(nn.Module):
    """One pre-norm FastConformer block."""

    def __init__(
        self,
        config: CohereAsrConfig,
    ) -> None:
        super().__init__()
        encoder = config.encoder_config
        self.norm_feed_forward1 = nn.LayerNorm(encoder.hidden_size)
        self.feed_forward1 = ConformerFeedForward(
            encoder.hidden_size,
            encoder.intermediate_size,
            encoder.activation_dropout,
        )
        self.norm_self_att = nn.LayerNorm(encoder.hidden_size)
        self.self_attn = RelPositionMultiHeadAttention(
            encoder.hidden_size,
            encoder.num_attention_heads,
            encoder.attention_dropout,
        )
        self.norm_conv = nn.LayerNorm(encoder.hidden_size)
        self.conv = ConformerConvolution(
            encoder.hidden_size,
            encoder.conv_kernel_size,
        )
        self.norm_feed_forward2 = nn.LayerNorm(encoder.hidden_size)
        self.feed_forward2 = ConformerFeedForward(
            encoder.hidden_size,
            encoder.intermediate_size,
            encoder.activation_dropout,
        )
        self.norm_out = nn.LayerNorm(encoder.hidden_size)
        self.dropout = float(encoder.dropout)

    def _drop(self, value: torch.Tensor) -> torch.Tensor:
        return F.dropout(
            value,
            p=self.dropout,
            training=self.training,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        invalid_attention_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + 0.5 * self._drop(
            self.feed_forward1(self.norm_feed_forward1(hidden_states)))
        hidden_states = hidden_states + self._drop(
            self.self_attn(
                self.norm_self_att(hidden_states),
                position_embeddings,
                invalid_attention_mask,
            ))
        hidden_states = hidden_states + self._drop(self.conv(
            self.norm_conv(hidden_states),
            padding_mask,
        ))
        hidden_states = hidden_states + 0.5 * self._drop(
            self.feed_forward2(self.norm_feed_forward2(hidden_states)))
        return self.norm_out(hidden_states)


class ConformerEncoder(nn.Module):
    """Cohere's Parakeet-compatible FastConformer encoder."""

    def __init__(self, config: CohereAsrConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_encode = ConvSubsampling(config)
        self.pos_enc = RelPositionalEncoding(config.encoder_config.hidden_size)
        self.layers = nn.ModuleList(
            ConformerLayer(config) for _ in range(config.encoder_config.num_hidden_layers))
        self.gradient_checkpointing = False

    @staticmethod
    def _masks(
        lengths: torch.Tensor,
        maximum: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = (torch.arange(maximum, device=lengths.device)[None, :] < lengths[:, None])
        invalid_padding = ~valid
        invalid_attention = ~(valid[:, :, None] & valid[:, None, :])
        return invalid_padding, invalid_attention

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> CohereEncoderOutput:
        if input_features.ndim != 3:
            raise ValueError("Cohere ASR features must have shape [batch, frames, mel].")
        expected_mels = self.config.encoder_config.num_mel_bins
        if input_features.shape[-1] != expected_mels:
            raise ValueError(
                f"Cohere ASR expects {expected_mels} mel bins; received "
                f"{input_features.shape[-1]}.")
        if attention_mask is None:
            lengths = torch.full(
                (input_features.shape[0], ),
                input_features.shape[1],
                device=input_features.device,
                dtype=torch.long,
            )
        else:
            if attention_mask.shape != input_features.shape[:2]:
                raise ValueError(
                    "Cohere ASR attention mask must match feature batch and "
                    "frame dimensions.")
            if torch.any((attention_mask != 0) & (attention_mask != 1)):
                raise ValueError("Cohere ASR attention mask must contain zeros and ones.")
            lengths = attention_mask.sum(-1).to(torch.long)
        if torch.any(lengths < 1):
            raise ValueError("Every Cohere ASR sample needs a valid frame.")
        hidden_states, lengths = self.pre_encode(
            input_features,
            lengths,
        )
        position_embeddings = self.pos_enc(hidden_states)
        padding_mask, attention = self._masks(
            lengths,
            hidden_states.shape[1],
        )
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    position_embeddings,
                    attention,
                    padding_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    position_embeddings,
                    attention,
                    padding_mask,
                )
        return CohereEncoderOutput(
            last_hidden_state=hidden_states,
            attention_mask=~padding_mask,
        )


class FixedPositionalEncoding(nn.Module):
    """Published fixed sinusoidal decoder positions."""

    def __init__(
        self,
        hidden_size: int,
        maximum_length: int,
    ) -> None:
        super().__init__()
        position = torch.arange(
            maximum_length,
            dtype=torch.float32,
        ).unsqueeze(1)
        frequency = torch.exp(
            (-math.log(10_000.0) / hidden_size) * torch.arange(0, hidden_size, 2, dtype=torch.float32))
        encoding = torch.zeros(
            maximum_length,
            hidden_size,
            dtype=torch.float32,
        )
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        encoding.div_(math.sqrt(hidden_size))
        self.register_buffer("pos_enc", encoding)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        if (position_ids.ndim != 2 or torch.any(position_ids < 0) or
                torch.any(position_ids >= self.pos_enc.shape[0])):
            raise ValueError("Cohere ASR decoder positions exceed the configured range.")
        flattened = position_ids.reshape(-1)
        return torch.index_select(
            self.pos_enc,
            0,
            flattened,
        ).reshape(*position_ids.shape, -1)


class DecoderAttention(nn.Module):
    """Decoder self- or cross-attention with official parameter names."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.query_net = nn.Linear(hidden_size, hidden_size)
        self.key_net = nn.Linear(hidden_size, hidden_size)
        self.value_net = nn.Linear(hidden_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

    def _reshape(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(
            value.shape[0],
            value.shape[1],
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float,
    ) -> torch.Tensor:
        source = hidden_states if context_states is None else context_states
        query = self._reshape(self.query_net(hidden_states))
        key = self._reshape(self.key_net(source))
        value = self._reshape(self.value_net(source))
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout if self.training else 0.0,
            scale=self.scale,
        )
        output = output.transpose(1, 2).contiguous().view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.hidden_size,
        )
        return self.out_projection(output)


class DecoderFeedForward(nn.Module):
    """Decoder MLP with the checkpoint's ``dense_in/out`` namespace."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, intermediate_size)
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
        self.activation = _activation(activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense_out(self.activation(self.dense_in(hidden_states)))


class TransformerDecoderLayer(nn.Module):
    """Pre-norm causal self-attention, cross-attention, and MLP."""

    def __init__(self, config: CohereAsrConfig) -> None:
        super().__init__()
        hidden_size = config.decoder_hidden_size
        heads = config.decoder_num_attention_heads
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.first_sub_layer = DecoderAttention(hidden_size, heads)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.second_sub_layer = DecoderAttention(hidden_size, heads)
        self.layer_norm_3 = nn.LayerNorm(hidden_size)
        self.third_sub_layer = DecoderFeedForward(
            hidden_size,
            config.decoder_intermediate_size,
            config.decoder_hidden_act,
        )
        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.first_sub_layer(
            self.layer_norm_1(hidden_states),
            None,
            self_attention_mask,
            dropout=self.attention_dropout,
        )
        hidden_states = hidden_states + self.second_sub_layer(
            self.layer_norm_2(hidden_states),
            encoder_hidden_states,
            cross_attention_mask,
            dropout=self.attention_dropout,
        )
        return hidden_states + self.third_sub_layer(self.layer_norm_3(hidden_states))


class TransformerDecoderEmbedding(nn.Module):
    """Token, fixed position, and LayerNorm embedding stage."""

    def __init__(self, config: CohereAsrConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.decoder_hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = FixedPositionalEncoding(
            config.decoder_hidden_size,
            config.decoder_max_position_embeddings,
        )
        self.layer_norm = nn.LayerNorm(config.decoder_hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.layer_norm(self.token_embedding(input_ids) + self.position_embedding(positions))


class TransformerDecoderCore(nn.Module):
    """Eight-layer decoder core for the official model."""

    def __init__(self, config: CohereAsrConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerDecoderLayer(config) for _ in range(config.decoder_num_hidden_layers))
        self.final_layer_norm = nn.LayerNorm(config.decoder_hidden_size)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    encoder_hidden_states,
                    self_attention_mask,
                    cross_attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    encoder_hidden_states,
                    self_attention_mask,
                    cross_attention_mask,
                )
        return self.final_layer_norm(hidden_states)


class TransformerDecoderWrapper(nn.Module):
    """Checkpoint-compatible decoder owner."""

    def __init__(self, config: CohereAsrConfig) -> None:
        super().__init__()
        self._embedding = TransformerDecoderEmbedding(config)
        self._decoder = TransformerDecoderCore(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = self._embedding(input_ids, positions)
        return self._decoder(
            hidden_states,
            encoder_hidden_states,
            self_attention_mask,
            cross_attention_mask,
        )


class TokenClassifierHead(nn.Module):
    """Single projection layer retaining Cohere's exported namespace."""

    def __init__(self, config: CohereAsrConfig) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.layer0 = nn.Linear(
            config.decoder_hidden_size,
            config.vocab_size,
        )
        self.use_log_softmax = config.log_softmax

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.mlp.layer0(hidden_states)
        return (torch.log_softmax(logits, dim=-1) if self.use_log_softmax else logits)


def shift_tokens_right(
    labels: torch.Tensor,
    *,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> torch.Tensor:
    """Shift teacher-forcing labels and replace ignored entries by padding."""
    if labels.ndim != 2:
        raise ValueError("Cohere ASR labels must have shape [batch, tokens].")
    shifted = labels.new_full(labels.shape, pad_token_id)
    shifted[:, 0] = decoder_start_token_id
    shifted[:, 1:] = labels[:, :-1]
    return shifted.masked_fill(shifted == -100, pad_token_id)


class CohereAsrForConditionalGeneration(nn.Module):
    """Complete native Cohere Transcribe graph."""

    def __init__(
        self,
        config: CohereAsrConfig | dict[str, Any],
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        del initialize
        self.config = CohereAsrConfig.coerce(config)
        self.encoder = ConformerEncoder(self.config)
        self.transf_decoder = TransformerDecoderWrapper(self.config)
        self.encoder_decoder_proj = nn.Linear(
            self.config.encoder_config.hidden_size,
            self.config.decoder_hidden_size,
        )
        self.log_softmax = TokenClassifierHead(self.config)
        self.preprocessor = CoherePreprocessor(self.config)
        self.tie_weights()

    def tie_weights(self) -> None:
        """Tie decoder input/output embeddings as in the official graph."""
        self.log_softmax.mlp.layer0.weight = (self.transf_decoder._embedding.token_embedding.weight)

    @property
    def gradient_checkpointing(self) -> bool:
        return (self.encoder.gradient_checkpointing and self.transf_decoder._decoder.gradient_checkpointing)

    def gradient_checkpointing_enable(self) -> None:
        self.encoder.gradient_checkpointing = True
        self.transf_decoder._decoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.encoder.gradient_checkpointing = False
        self.transf_decoder._decoder.gradient_checkpointing = False

    @staticmethod
    def _causal_mask(
        decoder_attention_mask: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch, length = decoder_attention_mask.shape
        blocked = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=decoder_attention_mask.device,
            ),
            diagonal=1,
        )
        mask = torch.zeros(
            batch,
            1,
            length,
            length,
            dtype=dtype,
            device=decoder_attention_mask.device,
        )
        mask.masked_fill_(blocked[None, None], float("-inf"))
        invalid_keys = ~decoder_attention_mask.bool()
        return mask.masked_fill(
            invalid_keys[:, None, None, :],
            -1e9,
        )

    @staticmethod
    def _cross_mask(
        encoder_attention_mask: torch.Tensor,
        *,
        target_length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del target_length
        invalid = ~encoder_attention_mask.bool()
        mask = torch.zeros(
            encoder_attention_mask.shape[0],
            1,
            1,
            encoder_attention_mask.shape[1],
            dtype=dtype,
            device=encoder_attention_mask.device,
        )
        return mask.masked_fill(invalid[:, None, None, :], -1e9)

    @staticmethod
    def _validate_labels(
        labels: torch.Tensor,
        *,
        vocab_size: int,
    ) -> None:
        if labels.ndim != 2:
            raise ValueError("Cohere ASR labels must have shape [batch, tokens].")
        if labels.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
        }:
            raise TypeError("Cohere ASR labels must contain integer token IDs.")
        invalid = (labels != -100) & ((labels < 0) | (labels >= vocab_size))
        if torch.any(invalid):
            raise ValueError("Cohere ASR labels contain an out-of-vocabulary token ID.")
        if not torch.any(labels != -100):
            raise ValueError("Cohere ASR labels contain no supervised positions.")

    def encode(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> CohereEncoderOutput:
        return self.encoder(input_features, attention_mask)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_output: CohereEncoderOutput,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if decoder_input_ids.ndim != 2:
            raise ValueError("Cohere ASR decoder IDs must have shape [batch, tokens].")
        if decoder_input_ids.shape[0] != (encoder_output.last_hidden_state.shape[0]):
            raise ValueError("Cohere ASR encoder and decoder batch dimensions disagree.")
        if torch.any((decoder_input_ids < 0) | (decoder_input_ids >= self.config.vocab_size)):
            raise ValueError("Cohere ASR decoder IDs contain an invalid token.")
        if decoder_attention_mask is None:
            decoder_attention_mask = (decoder_input_ids != self.config.pad_token_id)
        elif decoder_attention_mask.shape != decoder_input_ids.shape:
            raise ValueError("Cohere ASR decoder attention mask must match decoder IDs.")
        positions = torch.arange(
            decoder_input_ids.shape[1],
            device=decoder_input_ids.device,
        )[None, :].expand(decoder_input_ids.shape[0], -1)
        encoder_hidden_states = self.encoder_decoder_proj(encoder_output.last_hidden_state)
        decoder_dtype = encoder_hidden_states.dtype
        self_mask = self._causal_mask(
            decoder_attention_mask,
            dtype=decoder_dtype,
        )
        cross_mask = self._cross_mask(
            encoder_output.attention_mask,
            target_length=decoder_input_ids.shape[1],
            dtype=decoder_dtype,
        )
        hidden_states = self.transf_decoder(
            decoder_input_ids,
            positions,
            encoder_hidden_states,
            self_mask,
            cross_mask,
        )
        return self.log_softmax(hidden_states)

    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        encoder_outputs: CohereEncoderOutput | None = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> CohereAsrOutput:
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError("Native Cohere ASR received unsupported forward options: "
                             f"{unsupported}.")
        if encoder_outputs is None:
            if input_features is None:
                raise ValueError("Cohere ASR requires input features or encoder outputs.")
            encoder_outputs = self.encode(input_features, attention_mask)
        elif not isinstance(encoder_outputs, CohereEncoderOutput):
            raise TypeError("`encoder_outputs` must be CohereEncoderOutput.")
        if labels is not None:
            self._validate_labels(
                labels,
                vocab_size=self.config.vocab_size,
            )
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels,
                    pad_token_id=self.config.pad_token_id,
                    decoder_start_token_id=self.config.decoder_start_token_id,
                )
        if decoder_input_ids is None:
            raise ValueError("Cohere ASR requires decoder input IDs.")
        logits = self.decode(
            decoder_input_ids,
            encoder_outputs,
            decoder_attention_mask,
        )
        loss = None
        if labels is not None:
            if labels.shape != logits.shape[:2]:
                raise ValueError(
                    "Cohere ASR labels and decoder logits must have matching "
                    "batch/token dimensions.")
            if reduction not in {"mean", "sum", "none"}:
                raise ValueError("Cohere ASR loss reduction must be 'mean', 'sum', or "
                                 "'none'.")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction=reduction,
            )
            if reduction == "none":
                loss = loss.reshape(labels.shape)
        return CohereAsrOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_outputs.attention_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        *,
        decoder_attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        num_beams: int = 1,
        use_cache: bool = False,
    ) -> CohereGenerateOutput:
        """Greedy decoding; unsupported search modes fail closed."""
        if do_sample:
            raise ValueError("Native Cohere ASR does not implement sampling.")
        if num_beams != 1:
            raise ValueError("Native Cohere ASR implements greedy decoding only.")
        if use_cache:
            raise ValueError("Native Cohere ASR does not yet expose a validated KV cache.")
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        if (decoder_input_ids.shape[1] + max_new_tokens > self.config.decoder_max_position_embeddings):
            raise ValueError("Requested Cohere ASR generation exceeds decoder positions.")
        encoder_output = self.encode(input_features, attention_mask)
        sequences = decoder_input_ids.clone()
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(
                sequences,
                dtype=torch.bool,
            )
        else:
            decoder_attention_mask = decoder_attention_mask.bool().clone()
        finished = torch.zeros(
            sequences.shape[0],
            dtype=torch.bool,
            device=sequences.device,
        )
        for _ in range(max_new_tokens):
            logits = self.decode(
                sequences,
                encoder_output,
                decoder_attention_mask,
            )
            next_token = logits[:, -1].argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.config.pad_token_id),
                next_token,
            )
            sequences = torch.cat(
                (sequences, next_token[:, None]),
                dim=1,
            )
            decoder_attention_mask = torch.cat(
                (
                    decoder_attention_mask,
                    (~finished)[:, None],
                ),
                dim=1,
            )
            finished |= next_token == self.config.eos_token_id
            if torch.all(finished):
                break
        return CohereGenerateOutput(sequences=sequences)


__all__ = [
    "CohereAsrForConditionalGeneration",
    "CohereAsrOutput",
    "CohereEncoderOutput",
    "CohereGenerateOutput",
    "FilterbankFeatures",
    "shift_tokens_right",
]
