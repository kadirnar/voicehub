"""VoiceHub-owned WeNet GigaSpeech U2++ Conformer implementation.

The module layout intentionally matches WeNet revision
``a50d4208f13bbf3a0746e606ac29176cd2e87e6b`` so the audited checkpoint can
be converted without renaming tensors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.rnn import pad_sequence

from voicehub.models.asr_wenet.native.configuration import WeNetU2PPConfig
from voicehub.processing.kaldi import KaldiFbank, KaldiFbankConfig


def make_pad_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    if lengths.ndim != 1:
        raise ValueError("Lengths must have shape [batch].")
    if max_length is None:
        max_length = int(lengths.max().item())
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) >= lengths.unsqueeze(1)


def subsequent_mask(size: int, *, device: torch.device) -> Tensor:
    return torch.ones(size, size, dtype=torch.bool, device=device).tril()


def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int,
    *,
    device: torch.device,
) -> Tensor:
    result = torch.zeros(size, size, dtype=torch.bool, device=device)
    for index in range(size):
        start = (
            0 if num_left_chunks < 0 else max(
                (index // chunk_size - num_left_chunks) * chunk_size,
                0,
            ))
        end = min((index // chunk_size + 1) * chunk_size, size)
        result[index, start:end] = True
    return result


def add_optional_chunk_mask(
    xs: Tensor,
    masks: Tensor,
    *,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
) -> Tensor:
    chunk_size = 0
    num_left_chunks = -1
    if use_dynamic_chunk:
        maximum_length = xs.size(1)
        if decoding_chunk_size < 0 or maximum_length <= 1:
            chunk_size = maximum_length
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            chunk_size = int(torch.randint(
                1,
                maximum_length,
                (1, ),
                device=xs.device,
            ).item())
            if chunk_size > maximum_length // 2:
                chunk_size = maximum_length
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    maximum_left = (maximum_length - 1) // chunk_size
                    if maximum_left > 0:
                        num_left_chunks = int(
                            torch.randint(
                                0,
                                maximum_left,
                                (1, ),
                                device=xs.device,
                            ).item())
    elif static_chunk_size > 0:
        chunk_size = static_chunk_size
        num_left_chunks = num_decoding_left_chunks
    if chunk_size <= 0:
        return masks
    chunks = subsequent_chunk_mask(
        xs.size(1),
        chunk_size,
        num_left_chunks,
        device=xs.device,
    ).unsqueeze(0)
    return masks & chunks


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        dimension: int,
        dropout_rate: float,
        maximum_length: int = 5_000,
        *,
        relative: bool = False,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.scale = math.sqrt(dimension)
        self.dropout = nn.Dropout(dropout_rate)
        self.maximum_length = maximum_length
        self.relative = relative
        positions = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32) * -(math.log(10_000.0) / dimension))
        values = torch.zeros(maximum_length, dimension)
        values[:, 0::2] = torch.sin(positions * frequencies)
        values[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("pe", values.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> tuple[Tensor, Tensor]:
        if offset < 0 or offset + x.size(1) >= self.maximum_length:
            raise ValueError("Position range exceeds the configured maximum.")
        position = self.pe[:, offset:offset + x.size(1)].to(
            device=x.device,
            dtype=x.dtype,
        )
        encoded = x * self.scale
        if not self.relative:
            encoded = encoded + position
        return self.dropout(encoded), self.dropout(position)

    def position_encoding(self, offset: int, size: int) -> Tensor:
        if offset < 0 or offset + size >= self.maximum_length:
            raise ValueError("Position range exceeds the configured maximum.")
        return self.dropout(self.pe[:, offset:offset + size])


class Conv2dSubsampling6(nn.Module):

    subsampling_rate = 6
    right_context = 10

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, output_dimension, 3, 2),
            nn.ReLU(),
            nn.Conv2d(output_dimension, output_dimension, 5, 3),
            nn.ReLU(),
        )
        flattened = output_dimension * (((input_dimension - 1) // 2 - 2) // 3)
        self.linear = nn.Linear(flattened, output_dimension)
        self.pos_enc = PositionalEncoding(
            output_dimension,
            dropout_rate,
            relative=True,
        )

    def forward(
        self,
        x: Tensor,
        x_mask: Tensor,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        x = self.conv(x.unsqueeze(1))
        batch, channels, frames, features = x.shape
        x = self.linear(x.transpose(1, 2).contiguous().view(
            batch,
            frames,
            channels * features,
        ))
        x, position = self.pos_enc(x, offset)
        mask = x_mask[:, :, :-2:2][:, :, :-4:3]
        return x, position, mask

    def position_encoding(self, offset: int, size: int) -> Tensor:
        return self.pos_enc.position_encoding(offset, size)


class GlobalCMVN(nn.Module):

    def __init__(self, feature_dimension: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(feature_dimension))
        self.register_buffer("istd", torch.ones(feature_dimension))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) * self.istd


class MultiHeadedAttention(nn.Module):

    def __init__(self, heads: int, dimension: int, dropout_rate: float) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("Attention dimension must be divisible by heads.")
        self.d_k = dimension // heads
        self.h = heads
        self.linear_q = nn.Linear(dimension, dimension)
        self.linear_k = nn.Linear(dimension, dimension)
        self.linear_v = nn.Linear(dimension, dimension)
        self.linear_out = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def _qkv(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = query.size(0)
        q = self.linear_q(query).view(batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(batch, -1, self.h, self.d_k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def _attention(
        self,
        value: Tensor,
        scores: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        batch = value.size(0)
        if mask is not None:
            blocked = mask.unsqueeze(1).eq(0)
            scores = scores.masked_fill(blocked, -float("inf"))
            attention = torch.softmax(scores, dim=-1).masked_fill(blocked, 0.0)
        else:
            attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(self.dropout(attention), value)
        output = output.transpose(1, 2).contiguous().view(
            batch,
            -1,
            self.h * self.d_k,
        )
        return self.linear_out(output)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
        position: Tensor | None = None,
    ) -> Tensor:
        del position
        q, k, v = self._qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self._attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):

    def __init__(self, heads: int, dimension: int, dropout_rate: float) -> None:
        super().__init__(heads, dimension, dropout_rate)
        self.linear_pos = nn.Linear(dimension, dimension, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
        position: Tensor | None = None,
    ) -> Tensor:
        if position is None:
            raise ValueError("Relative attention requires positional encoding.")
        q, k, v = self._qkv(query, key, value)
        q = q.transpose(1, 2)
        batch = position.size(0)
        p = self.linear_pos(position).view(batch, -1, self.h, self.d_k)
        p = p.transpose(1, 2)
        q_u = (q + self.pos_bias_u).transpose(1, 2)
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        scores = (torch.matmul(q_u, k.transpose(-2, -1)) +
                  torch.matmul(q_v, p.transpose(-2, -1))) / math.sqrt(self.d_k)
        return self._attention(v, scores, mask)


class PositionwiseFeedForward(nn.Module):

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        dropout_rate: float,
        *,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.w_1 = nn.Linear(input_dimension, hidden_dimension)
        self.activation = nn.ReLU() if activation is None else activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_dimension, input_dimension)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class ConvolutionModule(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        causal: bool,
    ) -> None:
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, 1)
        self.lorder = kernel_size - 1 if causal else 0
        padding = 0 if causal else (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1)
        self.activation = nn.SiLU()

    def forward(
        self,
        x: Tensor,
        mask_pad: Tensor | None = None,
        cache: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        x = x.transpose(1, 2)
        if mask_pad is not None:
            x = x.masked_fill(~mask_pad, 0.0)
        if self.lorder:
            if cache is None:
                x = functional.pad(x, (self.lorder, 0))
            else:
                x = torch.cat((cache, x), dim=2)
            new_cache = x[:, :, -self.lorder:]
        else:
            new_cache = x.new_zeros(1)
        x = functional.glu(self.pointwise_conv1(x), dim=1)
        x = self.depthwise_conv(x).transpose(1, 2)
        x = self.activation(self.norm(x)).transpose(1, 2)
        x = self.pointwise_conv2(x)
        if mask_pad is not None:
            x = x.masked_fill(~mask_pad, 0.0)
        return x.transpose(1, 2), new_cache


class ConformerEncoderLayer(nn.Module):

    def __init__(self, config: WeNetU2PPConfig) -> None:
        super().__init__()
        dimension = config.encoder_dim
        self.self_attn = RelPositionMultiHeadedAttention(
            config.encoder_heads,
            dimension,
            config.attention_dropout,
        )
        self.feed_forward = PositionwiseFeedForward(
            dimension,
            config.encoder_linear_units,
            config.dropout,
            activation=nn.SiLU(),
        )
        self.feed_forward_macaron = PositionwiseFeedForward(
            dimension,
            config.encoder_linear_units,
            config.dropout,
            activation=nn.SiLU(),
        )
        self.conv_module = ConvolutionModule(
            dimension,
            config.convolution_kernel_size,
            causal=config.causal_convolution,
        )
        self.norm_ff = nn.LayerNorm(dimension, eps=1e-12)
        self.norm_mha = nn.LayerNorm(dimension, eps=1e-12)
        self.norm_ff_macaron = nn.LayerNorm(dimension, eps=1e-12)
        self.norm_conv = nn.LayerNorm(dimension, eps=1e-12)
        self.norm_final = nn.LayerNorm(dimension, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        self.size = dimension
        self.concat_linear = nn.Linear(2 * dimension, dimension)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        position: Tensor,
        mask_pad: Tensor | None = None,
        output_cache: Tensor | None = None,
        cnn_cache: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual = x
        x = residual + 0.5 * self.dropout(self.feed_forward_macaron(self.norm_ff_macaron(x)))

        residual = x
        normalized = self.norm_mha(x)
        query = normalized
        if output_cache is not None:
            chunk = normalized.size(1) - output_cache.size(1)
            query = normalized[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]
        x = residual + self.dropout(self.self_attn(query, normalized, normalized, mask, position))

        residual = x
        convolved, new_cache = self.conv_module(
            self.norm_conv(x),
            mask_pad,
            cnn_cache,
        )
        x = residual + self.dropout(convolved)

        residual = x
        x = residual + 0.5 * self.dropout(self.feed_forward(self.norm_ff(x)))
        x = self.norm_final(x)
        if output_cache is not None:
            x = torch.cat((output_cache, x), dim=1)
        return x, mask, new_cache


class ConformerEncoder(nn.Module):

    def __init__(self, config: WeNetU2PPConfig) -> None:
        super().__init__()
        self.config = config
        self.global_cmvn = GlobalCMVN(config.input_dim)
        self.embed = Conv2dSubsampling6(
            config.input_dim,
            config.encoder_dim,
            config.dropout,
        )
        self.after_norm = nn.LayerNorm(config.encoder_dim, eps=1e-12)
        self.encoders = nn.ModuleList(ConformerEncoderLayer(config) for _ in range(config.encoder_layers))

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
        *,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> tuple[Tensor, Tensor]:
        masks = ~make_pad_mask(
            feature_lengths,
            features.size(1),
        ).unsqueeze(1)
        x = self.global_cmvn(features)
        x, position, masks = self.embed(x, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(
            x,
            masks,
            use_dynamic_chunk=self.config.use_dynamic_chunk,
            use_dynamic_left_chunk=self.config.use_dynamic_left_chunk,
            decoding_chunk_size=decoding_chunk_size,
            static_chunk_size=self.config.static_chunk_size,
            num_decoding_left_chunks=num_decoding_left_chunks,
        )
        for layer in self.encoders:
            x, chunk_masks, _ = layer(
                x,
                chunk_masks,
                position,
                mask_pad,
            )
        return self.after_norm(x), masks


class DecoderLayer(nn.Module):

    def __init__(self, config: WeNetU2PPConfig) -> None:
        super().__init__()
        dimension = config.encoder_dim
        self.size = dimension
        self.self_attn = MultiHeadedAttention(
            config.decoder_heads,
            dimension,
            config.decoder_self_attention_dropout,
        )
        self.src_attn = MultiHeadedAttention(
            config.decoder_heads,
            dimension,
            config.decoder_source_attention_dropout,
        )
        self.feed_forward = PositionwiseFeedForward(
            dimension,
            config.decoder_linear_units,
            config.dropout,
        )
        self.norm1 = nn.LayerNorm(dimension, eps=1e-12)
        self.norm2 = nn.LayerNorm(dimension, eps=1e-12)
        self.norm3 = nn.LayerNorm(dimension, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        self.concat_linear1 = nn.Linear(2 * dimension, dimension)
        self.concat_linear2 = nn.Linear(2 * dimension, dimension)

    def forward(
        self,
        target: Tensor,
        target_mask: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        cache: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        residual = target
        normalized = self.norm1(target)
        query = normalized
        query_mask = target_mask
        if cache is not None:
            expected = (target.shape[0], target.shape[1] - 1, self.size)
            if tuple(cache.shape) != expected:
                raise ValueError(f"Decoder cache has shape {tuple(cache.shape)}, "
                                 f"expected {expected}.")
            query = normalized[:, -1:, :]
            residual = residual[:, -1:, :]
            query_mask = target_mask[:, -1:, :]
        x = residual + self.dropout(self.self_attn(
            query,
            normalized,
            normalized,
            query_mask,
        ))
        residual = x
        x = residual + self.dropout(self.src_attn(
            self.norm2(x),
            memory,
            memory,
            memory_mask,
        ))
        residual = x
        x = residual + self.dropout(self.feed_forward(self.norm3(x)))
        if cache is not None:
            x = torch.cat((cache, x), dim=1)
        return x, target_mask, memory, memory_mask


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        config: WeNetU2PPConfig,
        *,
        number_of_layers: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Embedding(config.vocab_size, config.encoder_dim),
            PositionalEncoding(
                config.encoder_dim,
                config.positional_dropout,
            ),
        )
        self.after_norm = nn.LayerNorm(config.encoder_dim, eps=1e-12)
        self.output_layer = nn.Linear(config.encoder_dim, config.vocab_size)
        self.decoders = nn.ModuleList(DecoderLayer(config) for _ in range(number_of_layers))

    def forward(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        input_ids: Tensor,
        input_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        valid = ~make_pad_mask(
            input_lengths,
            input_ids.size(1),
        ).unsqueeze(1)
        autoregressive = subsequent_mask(
            valid.size(-1),
            device=valid.device,
        ).unsqueeze(0)
        target_mask = valid & autoregressive
        x, _ = self.embed(input_ids)
        for layer in self.decoders:
            x, target_mask, memory, memory_mask = layer(
                x,
                target_mask,
                memory,
                memory_mask,
            )
        x = self.output_layer(self.after_norm(x))
        return x, target_mask.sum(1)

    def forward_one_step(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        input_ids: Tensor,
        target_mask: Tensor,
        cache: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        x, _ = self.embed(input_ids)
        new_cache: list[Tensor] = []
        for index, layer in enumerate(self.decoders):
            previous = None if cache is None else cache[index]
            x, target_mask, memory, memory_mask = layer(
                x,
                target_mask,
                memory,
                memory_mask,
                previous,
            )
            new_cache.append(x)
        logits = self.output_layer(self.after_norm(x[:, -1]))
        return functional.log_softmax(logits, dim=-1), new_cache


class BiTransformerDecoder(nn.Module):

    def __init__(self, config: WeNetU2PPConfig) -> None:
        super().__init__()
        self.left_decoder = TransformerDecoder(
            config,
            number_of_layers=config.decoder_layers,
        )
        self.right_decoder = TransformerDecoder(
            config,
            number_of_layers=config.reverse_decoder_layers,
        )

    def forward(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        input_ids: Tensor,
        input_lengths: Tensor,
        reverse_input_ids: Tensor,
        reverse_weight: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output, lengths = self.left_decoder(
            memory,
            memory_mask,
            input_ids,
            input_lengths,
        )
        if reverse_weight > 0.0:
            reverse_output, lengths = self.right_decoder(
                memory,
                memory_mask,
                reverse_input_ids,
                input_lengths,
            )
        else:
            reverse_output = output.new_zeros(())
        return output, reverse_output, lengths

    def forward_one_step(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        input_ids: Tensor,
        target_mask: Tensor,
        cache: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        return self.left_decoder.forward_one_step(
            memory,
            memory_mask,
            input_ids,
            target_mask,
            cache,
        )


class CTC(nn.Module):

    def __init__(self, config: WeNetU2PPConfig) -> None:
        super().__init__()
        self.ctc_lo = nn.Linear(config.encoder_dim, config.vocab_size)
        self.loss = nn.CTCLoss(
            blank=config.blank_token_id,
            reduction="sum",
            zero_infinity=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        hidden_lengths: Tensor,
        labels: Tensor,
        label_lengths: Tensor,
    ) -> Tensor:
        log_probabilities = functional.log_softmax(
            self.ctc_lo(hidden_states),
            dim=-1,
        ).transpose(0, 1)
        return (
            self.loss(
                log_probabilities,
                labels,
                hidden_lengths,
                label_lengths,
            ) / log_probabilities.size(1))

    def log_softmax(self, hidden_states: Tensor) -> Tensor:
        return functional.log_softmax(self.ctc_lo(hidden_states), dim=-1)


class LabelSmoothingLoss(nn.Module):

    def __init__(
        self,
        vocabulary_size: int,
        padding_id: int,
        smoothing: float,
        *,
        normalize_length: bool,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.padding_id = padding_id
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.normalize_length = normalize_length

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        batch_size = logits.size(0)
        flattened = logits.reshape(-1, self.vocabulary_size)
        targets = targets.reshape(-1)
        distribution = torch.zeros_like(flattened)
        distribution.fill_(self.smoothing / (self.vocabulary_size - 1))
        ignored = targets == self.padding_id
        total = targets.numel() - int(ignored.sum().item())
        safe_targets = targets.masked_fill(ignored, 0)
        distribution.scatter_(1, safe_targets.unsqueeze(1), self.confidence)
        divergence = functional.kl_div(
            functional.log_softmax(flattened, dim=1),
            distribution,
            reduction="none",
        )
        denominator = total if self.normalize_length else batch_size
        return divergence.masked_fill(ignored.unsqueeze(1), 0).sum() / max(
            denominator,
            1,
        )


def _add_sos_eos(
    labels: Tensor,
    *,
    sos: int,
    eos: int,
    ignore_id: int,
) -> tuple[Tensor, Tensor]:
    sos_tensor = labels.new_tensor([sos])
    eos_tensor = labels.new_tensor([eos])
    sequences = [row[row != ignore_id] for row in labels]
    inputs = [torch.cat((sos_tensor, row)) for row in sequences]
    targets = [torch.cat((row, eos_tensor)) for row in sequences]
    return (
        pad_sequence(inputs, batch_first=True, padding_value=eos),
        pad_sequence(targets, batch_first=True, padding_value=ignore_id),
    )


def _reverse_labels(
    labels: Tensor,
    lengths: Tensor,
    *,
    padding_value: int,
) -> Tensor:
    values = [torch.flip(row[:int(length.item())], dims=(0, )) for row, length in zip(labels, lengths)]
    return pad_sequence(
        values,
        batch_first=True,
        padding_value=padding_value,
    )


class WeNetSpecAugment(nn.Module):

    def __init__(self, config: WeNetU2PPConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        if not self.training or not self.config.spec_augment:
            return features
        result = features.clone()
        for batch_index, length in enumerate(lengths.tolist()):
            valid_frames = int(length)
            for _ in range(self.config.spec_time_masks):
                if valid_frames == 0 or self.config.spec_max_time == 0:
                    continue
                start = int(torch.randint(
                    0,
                    valid_frames,
                    (1, ),
                    device=features.device,
                ).item())
                width = int(
                    torch.randint(
                        1,
                        self.config.spec_max_time + 1,
                        (1, ),
                        device=features.device,
                    ).item())
                result[
                    batch_index,
                    start:min(valid_frames, start + width),
                ] = 0.0
            for _ in range(self.config.spec_frequency_masks):
                if self.config.spec_max_frequency == 0:
                    continue
                start = int(torch.randint(
                    0,
                    features.size(2),
                    (1, ),
                    device=features.device,
                ).item())
                width = int(
                    torch.randint(
                        1,
                        self.config.spec_max_frequency + 1,
                        (1, ),
                        device=features.device,
                    ).item())
                result[
                    batch_index,
                    :valid_frames,
                    start:min(features.size(2), start + width),
                ] = 0.0
        return result


@dataclass
class WeNetU2PPOutput:
    loss: Tensor | None
    attention_loss: Tensor | None
    ctc_loss: Tensor | None
    encoder_output: Tensor
    encoder_mask: Tensor
    log_probabilities: Tensor
    encoded_lengths: Tensor


def wenet_u2pp_hybrid_loss(
    *,
    attention_loss: Tensor | None,
    ctc_loss: Tensor | None,
    ctc_weight: float,
) -> Tensor | None:
    """Combine WeNet's native attention and CTC objectives."""
    if attention_loss is None:
        return ctc_loss
    if ctc_loss is None:
        return attention_loss
    return ctc_weight * ctc_loss + (1.0 - ctc_weight) * attention_loss


class WeNetU2PPForASR(nn.Module):
    """Exact trainable hybrid CTC/attention U2++ graph."""

    def __init__(
        self,
        config: WeNetU2PPConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = WeNetU2PPConfig.coerce(config)
        self.sos = self.config.sos_eos_token_id
        self.eos = self.config.sos_eos_token_id
        self.ignore_id = self.config.ignore_token_id
        self.encoder = ConformerEncoder(self.config)
        self.decoder = BiTransformerDecoder(self.config)
        self.ctc = CTC(self.config)
        self.criterion_att = LabelSmoothingLoss(
            self.config.vocab_size,
            self.ignore_id,
            self.config.label_smoothing,
            normalize_length=self.config.length_normalized_loss,
        )
        base_frontend = {
            "sample_frequency": float(self.config.sampling_rate),
            "num_mel_bins": self.config.input_dim,
            "frame_length": self.config.frame_length_ms,
            "frame_shift": self.config.frame_shift_ms,
            "energy_floor": 0.0,
        }
        self.inference_frontend = KaldiFbank(
            KaldiFbankConfig(
                **base_frontend,
                dither=self.config.inference_dither,
            ),
            waveform_scale=float(1 << 15),
        )
        self.training_frontend = KaldiFbank(
            KaldiFbankConfig(
                **base_frontend,
                dither=self.config.training_dither,
            ),
            waveform_scale=float(1 << 15),
        )
        self.spec_augment = WeNetSpecAugment(self.config)

    def extract_features(
        self,
        input_signal: Tensor,
        input_signal_length: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if input_signal.ndim == 1:
            input_signal = input_signal.unsqueeze(0)
        if input_signal.ndim != 2:
            raise ValueError("Raw WeNet audio must have shape [batch, samples].")
        if input_signal_length is None:
            input_signal_length = torch.full(
                (input_signal.size(0), ),
                input_signal.size(1),
                dtype=torch.long,
                device=input_signal.device,
            )
        window_samples = int(self.config.sampling_rate * self.config.frame_length_ms / 1000.0)
        shift_samples = int(self.config.sampling_rate * self.config.frame_shift_ms / 1000.0)
        minimum_samples = (window_samples + shift_samples * self.config.right_context)
        if input_signal.size(1) < minimum_samples:
            input_signal = functional.pad(
                input_signal,
                (0, minimum_samples - input_signal.size(1)),
            )
        input_signal_length = input_signal_length.clamp_min(minimum_samples)
        frontend = (self.training_frontend if self.training else self.inference_frontend)
        features, lengths = frontend(input_signal, input_signal_length)
        return self.spec_augment(features, lengths), lengths

    def encode(
        self,
        features: Tensor,
        feature_lengths: Tensor,
        *,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden, mask = self.encoder(
            features,
            feature_lengths,
            decoding_chunk_size=decoding_chunk_size,
            num_decoding_left_chunks=num_decoding_left_chunks,
        )
        lengths = mask.squeeze(1).sum(1).to(dtype=torch.long)
        return hidden, mask, lengths

    def _attention_loss(
        self,
        hidden: Tensor,
        mask: Tensor,
        labels: Tensor,
        label_lengths: Tensor,
    ) -> Tensor:
        inputs, targets = _add_sos_eos(
            labels,
            sos=self.sos,
            eos=self.eos,
            ignore_id=self.ignore_id,
        )
        reverse_labels = _reverse_labels(
            labels,
            label_lengths,
            padding_value=self.ignore_id,
        )
        reverse_inputs, reverse_targets = _add_sos_eos(
            reverse_labels,
            sos=self.sos,
            eos=self.eos,
            ignore_id=self.ignore_id,
        )
        lengths = label_lengths + 1
        logits, reverse_logits, _ = self.decoder(
            hidden,
            mask,
            inputs,
            lengths,
            reverse_inputs,
            self.config.reverse_weight,
        )
        forward_loss = self.criterion_att(logits, targets)
        reverse_loss = forward_loss.new_zeros(())
        if self.config.reverse_weight > 0.0:
            reverse_loss = self.criterion_att(reverse_logits, reverse_targets)
        return ((1.0 - self.config.reverse_weight) * forward_loss + self.config.reverse_weight * reverse_loss)

    def forward(
        self,
        *,
        input_signal: Tensor | None = None,
        input_signal_length: Tensor | None = None,
        features: Tensor | None = None,
        feature_lengths: Tensor | None = None,
        labels: Tensor | None = None,
        label_lengths: Tensor | None = None,
        decoding_chunk_size: int | None = None,
        num_decoding_left_chunks: int = -1,
    ) -> WeNetU2PPOutput:
        if (input_signal is None) == (features is None):
            raise ValueError("Pass exactly one of raw `input_signal` or precomputed "
                             "`features`.")
        if input_signal is not None:
            features, feature_lengths = self.extract_features(
                input_signal,
                input_signal_length,
            )
        if features is None or feature_lengths is None:
            raise ValueError("Feature lengths are required.")
        if decoding_chunk_size is None:
            decoding_chunk_size = 0 if self.training else -1
        hidden, mask, encoded_lengths = self.encode(
            features,
            feature_lengths,
            decoding_chunk_size=decoding_chunk_size,
            num_decoding_left_chunks=num_decoding_left_chunks,
        )
        log_probabilities = self.ctc.log_softmax(hidden)
        attention_loss = None
        ctc_loss = None
        loss = None
        if labels is not None:
            if label_lengths is None:
                label_lengths = (labels != self.ignore_id).sum(1)
            if self.config.ctc_weight < 1.0:
                attention_loss = self._attention_loss(
                    hidden,
                    mask,
                    labels,
                    label_lengths,
                )
            if self.config.ctc_weight > 0.0:
                ctc_loss = self.ctc(
                    hidden,
                    encoded_lengths,
                    labels,
                    label_lengths,
                )
            loss = wenet_u2pp_hybrid_loss(
                attention_loss=attention_loss,
                ctc_loss=ctc_loss,
                ctc_weight=self.config.ctc_weight,
            )
        return WeNetU2PPOutput(
            loss=loss,
            attention_loss=attention_loss,
            ctc_loss=ctc_loss,
            encoder_output=hidden,
            encoder_mask=mask,
            log_probabilities=log_probabilities,
            encoded_lengths=encoded_lengths,
        )

    def forward_attention_decoder(
        self,
        hypotheses: Tensor,
        hypothesis_lengths: Tensor,
        encoder_output: Tensor,
        reverse_weight: float,
    ) -> tuple[Tensor, Tensor]:
        if encoder_output.size(0) != 1:
            raise ValueError("Attention rescoring expects one encoder output.")
        count = hypotheses.size(0)
        repeated = encoder_output.repeat(count, 1, 1)
        memory_mask = torch.ones(
            count,
            1,
            repeated.size(1),
            dtype=torch.bool,
            device=repeated.device,
        )
        reverse_lengths = hypothesis_lengths - 1
        reversed_hypotheses = _reverse_labels(
            hypotheses[:, 1:],
            reverse_lengths,
            padding_value=self.ignore_id,
        )
        reverse_inputs, _ = _add_sos_eos(
            reversed_hypotheses,
            sos=self.sos,
            eos=self.eos,
            ignore_id=self.ignore_id,
        )
        forward, reverse, _ = self.decoder(
            repeated,
            memory_mask,
            hypotheses,
            hypothesis_lengths,
            reverse_inputs,
            reverse_weight,
        )
        return (
            functional.log_softmax(forward, dim=-1),
            functional.log_softmax(reverse, dim=-1),
        )


__all__ = [
    "WeNetU2PPConfig",
    "WeNetU2PPForASR",
    "WeNetU2PPOutput",
    "WeNetSpecAugment",
    "make_pad_mask",
    "subsequent_mask",
    "wenet_u2pp_hybrid_loss",
]
