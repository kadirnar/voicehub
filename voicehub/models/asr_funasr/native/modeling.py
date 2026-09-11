"""VoiceHub-owned PyTorch implementation of SenseVoiceSmall."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.asr_funasr.native.configuration import SenseVoiceSmallConfig
from voicehub.outputs import SpeechTrainingOutput

LANGUAGE_QUERY_IDS = {
    "auto": 0,
    "zh": 3,
    "en": 4,
    "yue": 7,
    "ja": 11,
    "ko": 12,
    "nospeech": 13,
}
LANGUAGE_TOKEN_TO_QUERY = {
    24_884: 3,
    24_885: 4,
    24_888: 7,
    24_892: 11,
    24_896: 12,
    24_992: 13,
}
TEXT_NORMALIZATION_QUERY_IDS = {
    "withitn": 14,
    "woitn": 15,
}
TEXT_NORMALIZATION_TOKEN_TO_QUERY = {
    25_016: 14,
    25_017: 15,
}
EMOTION_TOKEN_IDS = {
    "happy": 25_001,
    "sad": 25_002,
    "angry": 25_003,
    "neutral": 25_004,
    "fearful": 25_005,
    "disgusted": 25_006,
    "surprised": 25_007,
    "other": 25_008,
    "unknown": 25_009,
}


def sequence_mask(
    lengths: Tensor,
    maximum_length: int,
) -> Tensor:
    if lengths.ndim != 1:
        raise ValueError("Sequence lengths must have shape [batch].")
    positions = torch.arange(maximum_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


class SenseVoiceLayerNorm(nn.LayerNorm):
    """The source graph computes layer normalization in float32."""

    def forward(self, value: Tensor) -> Tensor:
        output = functional.layer_norm(
            value.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        )
        return output.to(dtype=value.dtype)


class SinusoidalPositionEncoder(nn.Module):
    """Position encoding used by the released SenseVoice graph."""

    @staticmethod
    def encode(
        positions: Tensor,
        depth: int,
        dtype: torch.dtype,
    ) -> Tensor:
        if depth < 4 or depth % 2:
            raise ValueError("Position-encoding depth must be even and >= 4.")
        positions = positions.to(dtype=dtype)
        increment = torch.log(torch.tensor([10_000.0], dtype=dtype,
                                           device=positions.device)) / (depth / 2 - 1)
        inverse_scales = torch.exp(
            torch.arange(
                depth // 2,
                device=positions.device,
                dtype=dtype,
            ) * -increment)
        scaled = positions.reshape(1, -1, 1) * inverse_scales.reshape(
            1,
            1,
            -1,
        )
        return torch.cat((scaled.sin(), scaled.cos()), dim=2)

    def forward(self, value: Tensor) -> Tensor:
        positions = torch.arange(
            1,
            value.shape[1] + 1,
            device=value.device,
        ).unsqueeze(0)
        return value + self.encode(
            positions,
            value.shape[-1],
            value.dtype,
        )


class PositionwiseFeedForward(nn.Module):

    def __init__(
        self,
        dimension: int,
        hidden_units: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.w_1 = nn.Linear(dimension, hidden_units)
        self.w_2 = nn.Linear(hidden_units, dimension)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, value: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(value))))


class MultiHeadedAttentionSANM(nn.Module):
    """Memory-equipped self-attention with the released fused QKV layout."""

    def __init__(
        self,
        heads: int,
        input_dimension: int,
        output_dimension: int,
        dropout: float,
        kernel_size: int,
        memory_shift: int = 0,
    ) -> None:
        super().__init__()
        if output_dimension % heads:
            raise ValueError("SANM output dimension must be divisible by heads.")
        self.d_k = output_dimension // heads
        self.h = heads
        self.linear_out = nn.Linear(output_dimension, output_dimension)
        self.linear_q_k_v = nn.Linear(
            input_dimension,
            output_dimension * 3,
        )
        self.fsmn_block = nn.Conv1d(
            output_dimension,
            output_dimension,
            kernel_size,
            stride=1,
            padding=0,
            groups=output_dimension,
            bias=False,
        )
        left = (kernel_size - 1) // 2 + memory_shift
        right = kernel_size - 1 - left
        self.pad_fn = nn.ConstantPad1d((left, right), 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward_qkv(
        self,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, frames, _ = value.shape
        query, key, projected_value = self.linear_q_k_v(value).chunk(3, dim=-1)

        def split_heads(item: Tensor) -> Tensor:
            return item.reshape(
                batch,
                frames,
                self.h,
                self.d_k,
            ).transpose(1, 2)

        return (
            split_heads(query),
            split_heads(key),
            split_heads(projected_value),
            projected_value,
        )

    def forward_fsmn(
        self,
        value: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        if mask is not None:
            visible = mask.reshape(value.shape[0], -1, 1).to(value.dtype)
            value = value * visible
        else:
            visible = None
        memory = self.fsmn_block(self.pad_fn(value.transpose(1, 2)))
        memory = self.dropout(memory.transpose(1, 2) + value)
        return memory if visible is None else memory * visible

    def forward_attention(
        self,
        value: Tensor,
        scores: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        if mask is not None:
            blocked = ~mask.bool().unsqueeze(1)
            scores = scores.masked_fill(blocked, -float("inf"))
            probabilities = torch.softmax(scores, dim=-1)
            probabilities = probabilities.masked_fill(blocked, 0.0)
        else:
            probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(self.dropout(probabilities), value)
        attended = attended.transpose(1, 2).contiguous().reshape(
            value.shape[0],
            -1,
            self.h * self.d_k,
        )
        return self.linear_out(attended)

    def forward(
        self,
        value: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        query, key, projected_value_heads, projected_value = self.forward_qkv(value)
        memory = self.forward_fsmn(projected_value, mask)
        scores = torch.matmul(
            query * self.d_k**-0.5,
            key.transpose(-2, -1),
        )
        attention = self.forward_attention(
            projected_value_heads,
            scores,
            mask,
        )
        return attention + memory


class EncoderLayerSANM(nn.Module):

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        self_attention: MultiHeadedAttentionSANM,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = self_attention
        self.feed_forward = feed_forward
        self.norm1 = SenseVoiceLayerNorm(input_dimension)
        self.norm2 = SenseVoiceLayerNorm(output_dimension)
        self.dropout = nn.Dropout(dropout)
        self.in_size = input_dimension
        self.size = output_dimension

    def forward(
        self,
        value: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residual = value
        attended = self.self_attn(self.norm1(value), mask)
        value = (residual + self.dropout(attended) if self.in_size == self.size else self.dropout(attended))
        value = value + self.dropout(self.feed_forward(self.norm2(value)))
        return value, mask


class SenseVoiceEncoderSmall(nn.Module):

    def __init__(
        self,
        config: SenseVoiceSmallConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = SenseVoiceSmallConfig.coerce(config)
        self.embed = SinusoidalPositionEncoder()
        self.encoders0 = nn.ModuleList(
            [self._layer(
                self.config.input_dimension,
                self.config.encoder_dimension,
            )])
        self.encoders = nn.ModuleList([
            self._layer(
                self.config.encoder_dimension,
                self.config.encoder_dimension,
            ) for _ in range(self.config.encoder_blocks - 1)
        ])
        self.tp_encoders = nn.ModuleList([
            self._layer(
                self.config.encoder_dimension,
                self.config.encoder_dimension,
            ) for _ in range(self.config.temporal_blocks)
        ])
        self.after_norm = SenseVoiceLayerNorm(self.config.encoder_dimension)
        self.tp_norm = SenseVoiceLayerNorm(self.config.encoder_dimension)
        self.gradient_checkpointing = False

    def _layer(
        self,
        input_dimension: int,
        output_dimension: int,
    ) -> EncoderLayerSANM:
        return EncoderLayerSANM(
            input_dimension,
            output_dimension,
            MultiHeadedAttentionSANM(
                self.config.attention_heads,
                input_dimension,
                output_dimension,
                self.config.attention_dropout,
                self.config.memory_kernel_size,
                self.config.memory_shift,
            ),
            PositionwiseFeedForward(
                output_dimension,
                self.config.linear_units,
                self.config.dropout,
            ),
            self.config.dropout,
        )

    def _run_layer(
        self,
        layer: EncoderLayerSANM,
        value: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                lambda hidden: layer(hidden, mask)[0],
                value,
                use_reentrant=False,
            )
        return layer(value, mask)[0]

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if features.ndim != 3:
            raise ValueError("SenseVoice features must have shape [batch, frames, bins].")
        if features.shape[-1] != self.config.input_dimension:
            raise ValueError(
                "SenseVoice expected feature dimension "
                f"{self.config.input_dimension}, found {features.shape[-1]}.")
        lengths = torch.as_tensor(
            feature_lengths,
            dtype=torch.long,
            device=features.device,
        )
        if tuple(lengths.shape) != (features.shape[0], ):
            raise ValueError("Feature lengths must have shape [batch].")
        if torch.any(lengths <= 0) or torch.any(lengths > features.shape[1]):
            raise ValueError("Feature lengths are outside the padded batch.")
        mask = sequence_mask(lengths, features.shape[1]).unsqueeze(1)
        value = self.embed(features * math.sqrt(self.config.encoder_dimension))
        for layer in self.encoders0:
            value = self._run_layer(layer, value, mask)
        for layer in self.encoders:
            value = self._run_layer(layer, value, mask)
        value = self.after_norm(value)
        output_lengths = mask.squeeze(1).sum(dim=1).to(dtype=torch.long)
        for layer in self.tp_encoders:
            value = self._run_layer(layer, value, mask)
        return self.tp_norm(value), output_lengths


class SenseVoiceCTC(nn.Module):

    def __init__(self, config: SenseVoiceSmallConfig) -> None:
        super().__init__()
        self.config = config
        self.ctc_lo = nn.Linear(
            config.encoder_dimension,
            config.vocabulary_size,
        )

    def logits(self, hidden_states: Tensor) -> Tensor:
        return self.ctc_lo(hidden_states)

    def log_softmax(self, hidden_states: Tensor) -> Tensor:
        return functional.log_softmax(self.logits(hidden_states), dim=-1)

    def forward(
        self,
        hidden_states: Tensor,
        hidden_lengths: Tensor,
        labels: Tensor,
        label_lengths: Tensor,
    ) -> Tensor:
        logits = functional.log_softmax(
            self.logits(hidden_states),
            dim=-1,
        ).transpose(0, 1).float()
        targets = torch.cat(
            [labels[index, :int(length)] for index, length in enumerate(label_lengths.tolist())])
        losses = functional.ctc_loss(
            logits,
            targets,
            hidden_lengths,
            label_lengths,
            blank=self.config.blank_token_id,
            reduction="none",
            zero_infinity=False,
        )
        return losses.sum() / hidden_states.shape[0]


@dataclass
class SenseVoiceInferenceOutput:
    logits: Tensor
    log_probabilities: Tensor
    encoded_lengths: Tensor
    hidden_states: Tensor


class SenseVoiceSmallForCTC(nn.Module):
    """Exact trainable SANM encoder, query heads, and CTC projection."""

    def __init__(
        self,
        config: SenseVoiceSmallConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = SenseVoiceSmallConfig.coerce(config)
        self.encoder = SenseVoiceEncoderSmall(self.config)
        self.ctc = SenseVoiceCTC(self.config)
        self.embed = nn.Embedding(
            self.config.query_embedding_size,
            self.config.input_dimension,
        )

    def gradient_checkpointing_enable(self) -> None:
        self.encoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.encoder.gradient_checkpointing = False

    def _query_features(
        self,
        features: Tensor,
        feature_lengths: Tensor,
        *,
        language_query_ids: Tensor,
        text_normalization_query_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        language = self.embed(language_query_ids[:, None])
        event_emotion = self.embed(torch.tensor(
            [1, 2],
            dtype=torch.long,
            device=features.device,
        )).unsqueeze(0).expand(features.shape[0], -1, -1)
        text_normalization = self.embed(text_normalization_query_ids[:, None])
        queries = torch.cat(
            (language, event_emotion, text_normalization),
            dim=1,
        )
        return (
            torch.cat((queries, features), dim=1),
            feature_lengths + 4,
        )

    def infer(
        self,
        features: Tensor,
        feature_lengths: Tensor,
        *,
        language: str = "auto",
        use_itn: bool = False,
        ban_unknown_emotion: bool = False,
    ) -> SenseVoiceInferenceOutput:
        if language not in LANGUAGE_QUERY_IDS:
            choices = ", ".join(LANGUAGE_QUERY_IDS)
            raise ValueError(f"Unsupported SenseVoice language {language!r}; use {choices}.")
        batch = features.shape[0]
        device = features.device
        language_ids = torch.full(
            (batch, ),
            LANGUAGE_QUERY_IDS[language],
            dtype=torch.long,
            device=device,
        )
        style = "withitn" if use_itn else "woitn"
        style_ids = torch.full(
            (batch, ),
            TEXT_NORMALIZATION_QUERY_IDS[style],
            dtype=torch.long,
            device=device,
        )
        queried, queried_lengths = self._query_features(
            features,
            feature_lengths,
            language_query_ids=language_ids,
            text_normalization_query_ids=style_ids,
        )
        hidden_states, encoded_lengths = self.encoder(
            queried,
            queried_lengths,
        )
        logits = self.ctc.logits(hidden_states)
        if ban_unknown_emotion and self.config.vocabulary_size > 25_009:
            logits = logits.clone()
            logits[..., EMOTION_TOKEN_IDS["unknown"]] = -float("inf")
        return SenseVoiceInferenceOutput(
            logits=logits,
            log_probabilities=functional.log_softmax(logits, dim=-1),
            encoded_lengths=encoded_lengths,
            hidden_states=hidden_states,
        )

    def _training_query_ids(
        self,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        language_ids = torch.tensor(
            [LANGUAGE_TOKEN_TO_QUERY.get(int(token), 0) for token in labels[:, 0]],
            dtype=torch.long,
            device=labels.device,
        )
        if self.training and self.config.language_dropout:
            keep = torch.rand(
                language_ids.shape,
                device=labels.device,
            ) > self.config.language_dropout
            language_ids = torch.where(
                keep,
                language_ids,
                torch.zeros_like(language_ids),
            )
        style_ids = torch.tensor(
            [TEXT_NORMALIZATION_TOKEN_TO_QUERY.get(int(token), -1) for token in labels[:, 3]],
            dtype=torch.long,
            device=labels.device,
        )
        if torch.any(style_ids < 0):
            raise ValueError("SenseVoice label position 3 must be <|withitn|> or "
                             "<|woitn|>.")
        return language_ids, style_ids

    def _rich_loss(
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        flattened = logits.reshape(-1, self.config.vocabulary_size)
        targets = labels.reshape(-1)
        ignored = targets == self.config.ignore_token_id
        valid_count = int((~ignored).sum().item())
        safe_targets = targets.masked_fill(ignored, 0)
        with torch.no_grad():
            distribution = torch.full_like(
                flattened,
                self.config.label_smoothing / (self.config.vocabulary_size - 1),
            )
            distribution.scatter_(
                1,
                safe_targets.unsqueeze(1),
                1.0 - self.config.label_smoothing,
            )
        losses = functional.kl_div(
            functional.log_softmax(flattened, dim=-1),
            distribution,
            reduction="none",
        ).masked_fill(ignored.unsqueeze(1), 0.0)
        denominator = (valid_count if self.config.length_normalized_loss else labels.shape[0])
        if denominator < 1:
            raise ValueError("SenseVoice rich labels contain no valid targets.")
        return losses.sum() / denominator

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
        labels: Tensor | None = None,
        label_lengths: Tensor | None = None,
        **_: Any,
    ) -> SpeechTrainingOutput:
        if labels is None:
            raise ValueError(
                "SenseVoice training requires labels with four rich-control "
                "tokens followed by transcript tokens.")
        labels = torch.as_tensor(
            labels,
            dtype=torch.long,
            device=features.device,
        )
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if labels.ndim != 2 or labels.shape[0] != features.shape[0]:
            raise ValueError("SenseVoice labels must have shape [batch, tokens].")
        if labels.shape[1] < 4:
            raise ValueError("SenseVoice labels require four rich-control tokens.")
        if label_lengths is None:
            label_lengths = labels.ne(self.config.ignore_token_id).sum(dim=1)
        label_lengths = torch.as_tensor(
            label_lengths,
            dtype=torch.long,
            device=features.device,
        )
        if (tuple(label_lengths.shape) != (features.shape[0], ) or torch.any(label_lengths < 4) or
                torch.any(label_lengths > labels.shape[1])):
            raise ValueError("SenseVoice label lengths are invalid.")
        language_ids, style_ids = self._training_query_ids(labels)
        queried, queried_lengths = self._query_features(
            features,
            feature_lengths,
            language_query_ids=language_ids,
            text_normalization_query_ids=style_ids,
        )
        hidden_states, encoded_lengths = self.encoder(
            queried,
            queried_lengths,
        )
        transcript_lengths = label_lengths - 4
        ctc_loss = self.ctc(
            hidden_states[:, 4:],
            encoded_lengths - 4,
            labels[:, 4:],
            transcript_lengths,
        )
        rich_logits = self.ctc.logits(hidden_states[:, :4])
        rich_loss = self._rich_loss(rich_logits, labels[:, :4])
        loss = ctc_loss + rich_loss
        return SpeechTrainingOutput(
            loss=loss,
            logits=self.ctc.logits(hidden_states),
            hidden_states=hidden_states,
            losses={
                "ctc": ctc_loss,
                "rich": rich_loss,
            },
            metadata={
                "encoded_lengths": encoded_lengths,
                "objective": "sensevoice-ctc-plus-rich-control-ce",
            },
        )


__all__ = [
    "EMOTION_TOKEN_IDS",
    "LANGUAGE_QUERY_IDS",
    "LANGUAGE_TOKEN_TO_QUERY",
    "TEXT_NORMALIZATION_QUERY_IDS",
    "TEXT_NORMALIZATION_TOKEN_TO_QUERY",
    "EncoderLayerSANM",
    "MultiHeadedAttentionSANM",
    "PositionwiseFeedForward",
    "SenseVoiceEncoderSmall",
    "SenseVoiceInferenceOutput",
    "SenseVoiceLayerNorm",
    "SenseVoiceSmallForCTC",
    "SinusoidalPositionEncoder",
    "sequence_mask",
]
