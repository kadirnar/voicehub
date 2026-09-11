"""VoiceHub-owned ESPnet 0.8 Transformer/CTC architecture."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.asr_espnet.native.configuration import ESPnetLibriSpeechTransformerConfig
from voicehub.models.asr_espnet.native.frontend import (
    ESPnetDefaultFrontend,
    ESPnetGlobalMVN,
    ESPnetSpecAugment,
    make_pad_mask,
)


class ESPnetPositionalEncoding(nn.Module):
    """Original sinusoidal encoding with the ESPnet input scale."""

    def __init__(
        self,
        dimension: int,
        dropout_rate: float,
        maximum_length: int = 5_000,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.scale = math.sqrt(dimension)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer(
            "_pe",
            self._build(maximum_length),
            persistent=False,
        )

    def _build(
        self,
        length: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        positions = torch.arange(
            length,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(
                0,
                self.dimension,
                2,
                device=device,
                dtype=torch.float32,
            ) * -(math.log(10_000.0) / self.dimension))
        values = torch.zeros(
            length,
            self.dimension,
            device=device,
            dtype=torch.float32,
        )
        values[:, 0::2] = torch.sin(positions * frequencies)
        values[:, 1::2] = torch.cos(positions * frequencies)
        return values.unsqueeze(0).to(dtype=dtype)

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[1] > self._pe.shape[1]:
            self._pe = self._build(
                values.shape[1],
                device=values.device,
                dtype=values.dtype,
            )
        positions = self._pe[:, :values.shape[1]].to(
            device=values.device,
            dtype=values.dtype,
        )
        return self.dropout(values * self.scale + positions)


class ESPnetMultiHeadedAttention(nn.Module):
    """Scaled dot-product multi-head attention with source tensor names."""

    def __init__(
        self,
        heads: int,
        dimension: int,
        dropout_rate: float,
    ) -> None:
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
        self.attention: Tensor | None = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        batch = query.shape[0]
        query = self.linear_q(query).view(batch, -1, self.h, self.d_k)
        key = self.linear_k(key).view(batch, -1, self.h, self.d_k)
        value = self.linear_v(value).view(batch, -1, self.h, self.d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_k)
        if mask is not None:
            blocked = ~mask.unsqueeze(1).bool()
            scores = scores.masked_fill(
                blocked,
                torch.finfo(scores.dtype).min,
            )
            attention = scores.softmax(dim=-1).masked_fill(blocked, 0.0)
        else:
            attention = scores.softmax(dim=-1)
        self.attention = attention
        attended = self.dropout(attention) @ value
        attended = attended.transpose(1, 2).contiguous().view(
            batch,
            -1,
            self.h * self.d_k,
        )
        return self.linear_out(attended)


class ESPnetPositionwiseFeedForward(nn.Module):
    """Two-layer ReLU feed-forward network."""

    def __init__(
        self,
        dimension: int,
        hidden_dimension: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.w_1 = nn.Linear(dimension, hidden_dimension)
        self.w_2 = nn.Linear(hidden_dimension, dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, values: Tensor) -> Tensor:
        return self.w_2(self.dropout(functional.relu(self.w_1(values))))


class ESPnetConv2dSubsampling6(nn.Module):
    """Two convolutional stages with the exact 2x/3x time reduction."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, output_dimension, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                output_dimension,
                output_dimension,
                kernel_size=5,
                stride=3,
            ),
            nn.ReLU(),
        )
        flattened = (output_dimension * (((input_dimension - 1) // 2 - 1) // 3))
        self.out = nn.Sequential(
            nn.Linear(flattened, output_dimension),
            ESPnetPositionalEncoding(output_dimension, dropout_rate),
        )

    def forward(
        self,
        values: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        values = self.conv(values.unsqueeze(1))
        batch, channels, time, features = values.shape
        values = self.out(values.transpose(1, 2).contiguous().view(
            batch,
            time,
            channels * features,
        ))
        if mask is not None:
            mask = mask[:, :, :-2:2][:, :, :-4:3]
        return values, mask


class ESPnetEncoderLayer(nn.Module):
    """One pre-norm Transformer encoder layer."""

    def __init__(
        self,
        config: ESPnetLibriSpeechTransformerConfig,
    ) -> None:
        super().__init__()
        dimension = config.encoder_dimension
        self.self_attn = ESPnetMultiHeadedAttention(
            config.encoder_attention_heads,
            dimension,
            config.attention_dropout_rate,
        )
        self.feed_forward = ESPnetPositionwiseFeedForward(
            dimension,
            config.encoder_linear_units,
            config.dropout_rate,
        )
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.normalize_before = config.normalize_before

    def forward(
        self,
        values: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        residual = values
        if self.normalize_before:
            values = self.norm1(values)
        values = residual + self.dropout(self.self_attn(values, values, values, mask))
        if not self.normalize_before:
            values = self.norm1(values)
        residual = values
        if self.normalize_before:
            values = self.norm2(values)
        values = residual + self.dropout(self.feed_forward(values))
        if not self.normalize_before:
            values = self.norm2(values)
        return values, mask


class ESPnetTransformerEncoder(nn.Module):
    """18-layer source-compatible acoustic encoder."""

    def __init__(
        self,
        config: ESPnetLibriSpeechTransformerConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed = ESPnetConv2dSubsampling6(
            config.n_mels,
            config.encoder_dimension,
            config.positional_dropout_rate,
        )
        self.encoders = nn.ModuleList(ESPnetEncoderLayer(config) for _ in range(config.encoder_blocks))
        self.after_norm = nn.LayerNorm(config.encoder_dimension)

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (features.ndim != 3 or features.shape[-1] != self.config.n_mels):
            raise ValueError(
                "ESPnet encoder features must have shape "
                f"[batch, frames, {self.config.n_mels}].")
        lengths = torch.as_tensor(
            feature_lengths,
            dtype=torch.long,
            device=features.device,
        )
        if (lengths.ndim != 1 or lengths.shape[0] != features.shape[0] or
                torch.any(lengths > features.shape[1])):
            raise ValueError("ESPnet feature lengths must describe the padded batch.")
        if (features.shape[1] < self.config.minimum_feature_frames or
                torch.any(lengths < self.config.minimum_feature_frames)):
            raise ValueError(
                "ESPnet conv2d6 requires at least "
                f"{self.config.minimum_feature_frames} feature frames.")
        mask = ~make_pad_mask(
            lengths,
            features.shape[1],
        ).unsqueeze(1)
        values, mask = self.embed(features, mask)
        for layer in self.encoders:
            values, mask = layer(values, mask)
        if self.config.normalize_before:
            values = self.after_norm(values)
        if mask is None:
            lengths = torch.full(
                (values.shape[0], ),
                values.shape[1],
                dtype=torch.long,
                device=values.device,
            )
        else:
            lengths = mask.squeeze(1).sum(dim=1).long()
        return values, lengths


def subsequent_mask(size: int, *, device: torch.device) -> Tensor:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("Subsequent-mask size must be a positive integer.")
    return torch.ones(size, size, dtype=torch.bool, device=device).tril()


class ESPnetDecoderLayer(nn.Module):
    """One pre-norm causal decoder layer."""

    def __init__(
        self,
        config: ESPnetLibriSpeechTransformerConfig,
    ) -> None:
        super().__init__()
        dimension = config.encoder_dimension
        self.self_attn = ESPnetMultiHeadedAttention(
            config.decoder_attention_heads,
            dimension,
            config.attention_dropout_rate,
        )
        self.src_attn = ESPnetMultiHeadedAttention(
            config.decoder_attention_heads,
            dimension,
            config.attention_dropout_rate,
        )
        self.feed_forward = ESPnetPositionwiseFeedForward(
            dimension,
            config.decoder_linear_units,
            config.dropout_rate,
        )
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.norm3 = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.normalize_before = config.normalize_before

    def forward(
        self,
        target: Tensor,
        target_mask: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        residual = target
        if self.normalize_before:
            target = self.norm1(target)
        values = residual + self.dropout(self.self_attn(target, target, target, target_mask))
        if not self.normalize_before:
            values = self.norm1(values)
        residual = values
        if self.normalize_before:
            values = self.norm2(values)
        values = residual + self.dropout(self.src_attn(values, memory, memory, memory_mask))
        if not self.normalize_before:
            values = self.norm2(values)
        residual = values
        if self.normalize_before:
            values = self.norm3(values)
        values = residual + self.dropout(self.feed_forward(values))
        if not self.normalize_before:
            values = self.norm3(values)
        return values, target_mask, memory, memory_mask


class ESPnetTransformerDecoder(nn.Module):
    """Six-layer autoregressive decoder."""

    def __init__(
        self,
        config: ESPnetLibriSpeechTransformerConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Sequential(
            nn.Embedding(
                config.vocabulary_size,
                config.encoder_dimension,
            ),
            ESPnetPositionalEncoding(
                config.encoder_dimension,
                config.positional_dropout_rate,
            ),
        )
        self.decoders = nn.ModuleList(ESPnetDecoderLayer(config) for _ in range(config.decoder_blocks))
        self.after_norm = nn.LayerNorm(config.encoder_dimension)
        self.output_layer = nn.Linear(
            config.encoder_dimension,
            config.vocabulary_size,
        )

    def forward(
        self,
        memory: Tensor,
        memory_lengths: Tensor,
        tokens: Tensor,
        token_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        target_mask = ~make_pad_mask(
            token_lengths.to(tokens.device),
            tokens.shape[1],
        ).unsqueeze(1)
        target_mask = target_mask & subsequent_mask(
            tokens.shape[1],
            device=tokens.device,
        ).unsqueeze(0)
        memory_mask = ~make_pad_mask(
            memory_lengths.to(memory.device),
            memory.shape[1],
        ).unsqueeze(1)
        values = self.embed(tokens)
        for layer in self.decoders:
            values, target_mask, memory, memory_mask = layer(
                values,
                target_mask,
                memory,
                memory_mask,
            )
        if self.config.normalize_before:
            values = self.after_norm(values)
        return self.output_layer(values), target_mask.sum(dim=1).long()

    def score(
        self,
        prefix: Tensor,
        memory: Tensor,
    ) -> Tensor:
        if prefix.ndim != 1 or prefix.numel() == 0:
            raise ValueError("Decoder prefix must be a non-empty token vector.")
        logits, _ = self(
            memory.unsqueeze(0),
            torch.tensor(
                [memory.shape[0]],
                dtype=torch.long,
                device=memory.device,
            ),
            prefix.unsqueeze(0),
            torch.tensor(
                [prefix.shape[0]],
                dtype=torch.long,
                device=prefix.device,
            ),
        )
        return logits[0, -1].log_softmax(dim=-1)


class ESPnetCTC(nn.Module):
    """Built-in CTC projection and batch-averaged objective."""

    def __init__(
        self,
        config: ESPnetLibriSpeechTransformerConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.ctc_lo = nn.Linear(
            config.encoder_dimension,
            config.vocabulary_size,
        )
        self.loss = nn.CTCLoss(
            blank=config.blank_token_id,
            reduction="sum",
            zero_infinity=False,
        )

    def logits(self, encoder_states: Tensor) -> Tensor:
        return self.ctc_lo(
            functional.dropout(
                encoder_states,
                p=self.config.ctc_dropout_rate,
                training=self.training,
            ))

    def forward(
        self,
        encoder_states: Tensor,
        encoder_lengths: Tensor,
        labels: Tensor,
        label_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = self.logits(encoder_states)
        targets = torch.cat(
            [labels[index, :int(length.item())] for index, length in enumerate(label_lengths)])
        loss = self.loss(
            logits.log_softmax(dim=-1).transpose(0, 1),
            targets,
            encoder_lengths,
            label_lengths,
        )
        return loss / encoder_states.shape[0], logits


class ESPnetSequentialRNNLanguageModel(nn.Module):
    """Released four-layer batch-first LSTM language model."""

    def __init__(
        self,
        config: ESPnetLibriSpeechTransformerConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Embedding(
            config.vocabulary_size,
            config.language_model_units,
            padding_idx=config.blank_token_id,
        )
        self.rnn = nn.LSTM(
            config.language_model_units,
            config.language_model_units,
            config.language_model_layers,
            dropout=config.language_model_dropout,
            batch_first=True,
        )
        self.decoder = nn.Linear(
            config.language_model_units,
            config.vocabulary_size,
        )
        self.dropout = nn.Dropout(config.language_model_dropout)

    def forward(
        self,
        tokens: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        embedded = self.dropout(self.encoder(tokens))
        output, next_state = self.rnn(embedded, state)
        return self.decoder(self.dropout(output)), next_state

    def score(
        self,
        last_token: Tensor,
        state: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if last_token.ndim == 0:
            last_token = last_token.view(1)
        logits, next_state = self(last_token.reshape(-1, 1), state)
        return logits[:, -1].log_softmax(dim=-1), next_state


@dataclass(slots=True)
class ESPnetASRModelOutput:
    """Native hybrid ASR forward result."""

    loss: Tensor | None
    logits: Tensor | None
    ctc_logits: Tensor
    encoder_states: Tensor
    encoder_lengths: Tensor
    losses: dict[str, Tensor]


def espnet_label_smoothed_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    padding_id: int,
    smoothing: float,
    normalize_length: bool,
) -> Tensor:
    """Exact KL label smoothing used by the pinned ESPnet source."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("Attention logits/targets have incompatible shapes.")
    batch_size, _, vocabulary_size = logits.shape
    flattened = logits.reshape(-1, vocabulary_size)
    target = targets.reshape(-1)
    with torch.no_grad():
        distribution = flattened.new_full(
            flattened.shape,
            smoothing / (vocabulary_size - 1),
        )
        ignore = target == padding_id
        valid_target = target.masked_fill(ignore, 0)
        distribution.scatter_(
            1,
            valid_target.unsqueeze(1),
            1.0 - smoothing,
        )
    values = functional.kl_div(
        flattened.log_softmax(dim=-1),
        distribution,
        reduction="none",
    )
    denominator = (int((~ignore).sum().item()) if normalize_length else batch_size)
    return values.masked_fill(ignore.unsqueeze(1), 0.0).sum() / max(
        denominator,
        1,
    )


def _initialize_like_espnet(module: nn.Module) -> None:
    """Apply the recipe's ``init: xavier_uniform`` policy."""
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        for parameter in module.parameters():
            if parameter.ndim == 1:
                parameter.zero_()
        for child in module.modules():
            if isinstance(child, (nn.Embedding, nn.LayerNorm)):
                child.reset_parameters()


class ESPnetLibriSpeechTransformerForASR(nn.Module):
    """Raw waveform to hybrid Transformer/CTC training graph."""

    def __init__(
        self,
        config: ESPnetLibriSpeechTransformerConfig,
    ) -> None:
        super().__init__()
        self.config = ESPnetLibriSpeechTransformerConfig.coerce(config)
        self.frontend = ESPnetDefaultFrontend(self.config)
        self.specaug = ESPnetSpecAugment(self.config)
        self.normalize = ESPnetGlobalMVN(self.config)
        self.encoder = ESPnetTransformerEncoder(self.config)
        self.decoder = ESPnetTransformerDecoder(self.config)
        self.ctc = ESPnetCTC(self.config)
        _initialize_like_espnet(self)

    def encode_features(
        self,
        features: Tensor,
        feature_lengths: Tensor,
        *,
        apply_augmentation: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        augmented = (
            self.training and self.config.apply_spec_augment
            if apply_augmentation is None else apply_augmentation)
        if augmented:
            features, feature_lengths = self.specaug(
                features,
                feature_lengths,
            )
        features, feature_lengths = self.normalize(
            features,
            feature_lengths,
        )
        return self.encoder(features, feature_lengths)

    def encode(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor | None = None,
        *,
        apply_augmentation: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        features, feature_lengths = self.frontend(
            waveforms,
            waveform_lengths,
        )
        return self.encode_features(
            features,
            feature_lengths,
            apply_augmentation=apply_augmentation,
        )

    def _attention_targets(
        self,
        labels: Tensor,
        label_lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = labels.shape[0]
        maximum = int(label_lengths.max().item()) + 1
        decoder_inputs = labels.new_full(
            (batch, maximum),
            self.config.sos_eos_token_id,
        )
        decoder_targets = labels.new_full(
            (batch, maximum),
            self.config.ignore_token_id,
        )
        for index, length in enumerate(label_lengths):
            size = int(length.item())
            decoder_inputs[index, 1:size + 1] = labels[index, :size]
            decoder_targets[index, :size] = labels[index, :size]
            decoder_targets[index, size] = self.config.sos_eos_token_id
        return decoder_inputs, decoder_targets, label_lengths + 1

    def forward(
        self,
        waveforms: Tensor | None = None,
        waveform_lengths: Tensor | None = None,
        labels: Tensor | None = None,
        label_lengths: Tensor | None = None,
        *,
        features: Tensor | None = None,
        feature_lengths: Tensor | None = None,
        apply_augmentation: bool | None = None,
    ) -> ESPnetASRModelOutput:
        if (waveforms is None) == (features is None):
            raise ValueError("Provide exactly one of `waveforms` or `features`.")
        if features is None:
            encoder_states, encoder_lengths = self.encode(
                waveforms,
                waveform_lengths,
                apply_augmentation=apply_augmentation,
            )
        else:
            if feature_lengths is None:
                feature_lengths = torch.full(
                    (features.shape[0], ),
                    features.shape[1],
                    dtype=torch.long,
                    device=features.device,
                )
            encoder_states, encoder_lengths = self.encode_features(
                features,
                feature_lengths,
                apply_augmentation=apply_augmentation,
            )
        ctc_logits = self.ctc.logits(encoder_states)
        if labels is None:
            return ESPnetASRModelOutput(
                loss=None,
                logits=None,
                ctc_logits=ctc_logits,
                encoder_states=encoder_states,
                encoder_lengths=encoder_lengths,
                losses={},
            )
        labels = torch.as_tensor(
            labels,
            dtype=torch.long,
            device=encoder_states.device,
        )
        if labels.ndim != 2 or labels.shape[0] != encoder_states.shape[0]:
            raise ValueError("Labels must have shape [batch, tokens].")
        if label_lengths is None:
            label_lengths = (labels != self.config.ignore_token_id).sum(dim=1)
        else:
            label_lengths = torch.as_tensor(
                label_lengths,
                dtype=torch.long,
                device=labels.device,
            )
        if (label_lengths.ndim != 1 or label_lengths.shape[0] != labels.shape[0] or
                torch.any(label_lengths <= 0) or torch.any(label_lengths > labels.shape[1])):
            raise ValueError("Label lengths must describe non-empty padded rows.")
        for index, length in enumerate(label_lengths):
            row = labels[index, :int(length.item())]
            if torch.any(row < 0) or torch.any(row >= self.config.vocabulary_size):
                raise ValueError("Valid ESPnet labels must be vocabulary IDs.")
            if torch.any(row == self.config.blank_token_id):
                raise ValueError("Transcript labels cannot contain the CTC blank ID.")
            if torch.any(row == self.config.sos_eos_token_id):
                raise ValueError("Transcript labels must not include SOS/EOS.")
        if torch.any(label_lengths > encoder_lengths):
            raise ValueError("CTC transcript length exceeds the subsampled acoustic sequence.")
        ctc_loss, ctc_logits = self.ctc(
            encoder_states,
            encoder_lengths,
            labels,
            label_lengths,
        )
        decoder_inputs, decoder_targets, decoder_lengths = (self._attention_targets(labels, label_lengths))
        logits, _ = self.decoder(
            encoder_states,
            encoder_lengths,
            decoder_inputs,
            decoder_lengths,
        )
        attention_loss = espnet_label_smoothed_loss(
            logits,
            decoder_targets,
            padding_id=self.config.ignore_token_id,
            smoothing=self.config.label_smoothing,
            normalize_length=self.config.length_normalized_loss,
        )
        loss = (self.config.ctc_weight * ctc_loss + (1.0 - self.config.ctc_weight) * attention_loss)
        return ESPnetASRModelOutput(
            loss=loss,
            logits=logits,
            ctc_logits=ctc_logits,
            encoder_states=encoder_states,
            encoder_lengths=encoder_lengths,
            losses={
                "loss": loss,
                "ctc_loss": ctc_loss,
                "attention_loss": attention_loss,
            },
        )


__all__ = [
    "ESPnetASRModelOutput",
    "ESPnetCTC",
    "ESPnetConv2dSubsampling6",
    "ESPnetLibriSpeechTransformerForASR",
    "ESPnetMultiHeadedAttention",
    "ESPnetPositionalEncoding",
    "ESPnetSequentialRNNLanguageModel",
    "ESPnetTransformerDecoder",
    "ESPnetTransformerEncoder",
    "espnet_label_smoothed_loss",
    "subsequent_mask",
]
