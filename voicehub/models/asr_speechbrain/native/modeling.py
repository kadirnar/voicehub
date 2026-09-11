"""VoiceHub-owned SpeechBrain CRDNN, attention decoder, and RNNLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.asr_speechbrain.native.configuration import SpeechBrainCRDNNASRConfig
from voicehub.models.asr_speechbrain.native.frontend import SpeechBrainASRFrontend


def _same_reflect_convolution(
    inputs: Tensor,
    convolution: nn.Conv2d,
) -> Tensor:
    kernel_frequency, kernel_time = convolution.kernel_size
    dilation_frequency, dilation_time = convolution.dilation
    frequency_padding = dilation_frequency * (kernel_frequency - 1) // 2
    time_padding = dilation_time * (kernel_time - 1) // 2
    values = functional.pad(
        inputs,
        (
            time_padding,
            time_padding,
            frequency_padding,
            frequency_padding,
        ),
        mode="reflect",
    )
    return convolution(values)


class SpeechBrainCNNBlock(nn.Module):
    """One VGG-like SpeechBrain block in ``[B,T,F,C]`` layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        input_frequency: int,
        kernel_size: tuple[int, int],
        pooling_size: int,
        dropout: float,
        negative_slope: float,
    ) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
        )
        self.norm_1 = nn.LayerNorm((input_frequency, out_channels))
        self.conv_2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
        )
        self.norm_2 = nn.LayerNorm((input_frequency, out_channels))
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.pooling = nn.MaxPool2d(
            kernel_size=(1, pooling_size),
            stride=(1, pooling_size),
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        values = inputs.permute(0, 3, 2, 1)
        values = _same_reflect_convolution(values, self.conv_1)
        values = values.permute(0, 3, 2, 1)
        values = self.activation(self.norm_1(values))
        values = values.permute(0, 3, 2, 1)
        values = _same_reflect_convolution(values, self.conv_2)
        values = values.permute(0, 3, 2, 1)
        values = self.activation(self.norm_2(values))
        # SpeechBrain's Pooling1d(pool_axis=2, input_dims=4) transposes the
        # frequency axis to the end and then applies a (1, pooling_size)
        # MaxPool2d.  The time dimension therefore remains unchanged.
        values = values.transpose(-1, 2)
        values = self.pooling(values)
        values = values.transpose(-1, 2)
        # The pinned Dropout2d wrapper maps [B,T,F,C] to [B,F,C,T], so the
        # PyTorch channel mask is shared by every time step and convolution
        # channel for one frequency bin.
        values = self.dropout(values.permute(0, 2, 3, 1))
        return values.permute(0, 3, 1, 2)


class SpeechBrainDNNBlock(nn.Module):
    """Linear, temporal BatchNorm, LeakyReLU, and dropout."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        dropout: float,
        negative_slope: float,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.norm = nn.BatchNorm1d(output_size)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        values = self.linear(inputs)
        values = self.norm(values.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.activation(values))


class SpeechBrainCRDNNEncoder(nn.Module):
    """Exact two-CNN, four-BiLSTM, two-DNN released encoder."""

    def __init__(self, config: SpeechBrainCRDNNASRConfig) -> None:
        super().__init__()
        frequency = config.n_mels
        in_channels = 1
        cnn_blocks = []
        for out_channels, pooling_size in zip(
                config.cnn_channels,
                config.inter_layer_pooling_size,
        ):
            cnn_blocks.append(
                SpeechBrainCNNBlock(
                    in_channels,
                    out_channels,
                    input_frequency=frequency,
                    kernel_size=config.cnn_kernel_size,
                    pooling_size=pooling_size,
                    dropout=config.dropout,
                    negative_slope=config.negative_slope,
                ))
            frequency //= pooling_size
            in_channels = out_channels
        self.cnn_blocks = nn.ModuleList(cnn_blocks)
        self.time_pooling = nn.MaxPool2d(
            kernel_size=(1, config.time_pooling_size),
            stride=(1, config.time_pooling_size),
        )
        self.rnn = nn.LSTM(
            input_size=config.encoder_rnn_input_size,
            hidden_size=config.rnn_neurons,
            num_layers=config.rnn_layers,
            dropout=config.dropout,
            bidirectional=config.rnn_bidirectional,
            batch_first=True,
        )
        dnn_blocks = []
        input_size = config.encoder_rnn_output_size
        for _ in range(config.dnn_blocks):
            dnn_blocks.append(
                SpeechBrainDNNBlock(
                    input_size,
                    config.dnn_neurons,
                    dropout=config.dropout,
                    negative_slope=config.negative_slope,
                ))
            input_size = config.dnn_neurons
        self.dnn_blocks = nn.ModuleList(dnn_blocks)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("CRDNN features must have shape [batch, frames, mel].")
        values = features.unsqueeze(-1)
        for block in self.cnn_blocks:
            values = block(values)
        values = values.permute(0, 3, 2, 1)
        values = self.time_pooling(values)
        values = values.permute(0, 3, 2, 1)
        values = values.flatten(start_dim=2)
        self.rnn.flatten_parameters()
        values, _ = self.rnn(values)
        for block in self.dnn_blocks:
            values = block(values)
        return values


@dataclass(slots=True)
class LocationAttentionState:
    """Per-request location-attention cache."""

    encoded_projection: Tensor
    mask: Tensor
    previous_attention: Tensor

    def index_select(self, indexes: Tensor) -> LocationAttentionState:
        return LocationAttentionState(
            encoded_projection=self.encoded_projection.index_select(0, indexes),
            mask=self.mask.index_select(0, indexes),
            previous_attention=self.previous_attention.index_select(0, indexes),
        )


class SpeechBrainLocationAttention(nn.Module):
    """Location-aware attention from the pinned SpeechBrain decoder."""

    def __init__(self, config: SpeechBrainCRDNNASRConfig) -> None:
        super().__init__()
        self.encoder_projection = nn.Linear(
            config.dnn_neurons,
            config.attention_dim,
        )
        self.decoder_projection = nn.Linear(
            config.decoder_neurons,
            config.attention_dim,
        )
        self.location_convolution = nn.Conv1d(
            1,
            config.attention_channels,
            kernel_size=2 * config.attention_kernel_size + 1,
            padding=config.attention_kernel_size,
            bias=False,
        )
        self.location_projection = nn.Linear(
            config.attention_channels,
            config.attention_dim,
        )
        self.score_projection = nn.Linear(
            config.attention_dim,
            1,
            bias=False,
        )
        self.output_projection = nn.Linear(
            config.dnn_neurons,
            config.attention_dim,
        )

    def initialize(
        self,
        encoder_states: Tensor,
        encoder_lengths: Tensor,
    ) -> LocationAttentionState:
        lengths = torch.as_tensor(
            encoder_lengths,
            dtype=torch.long,
            device=encoder_states.device,
        )
        positions = torch.arange(
            encoder_states.shape[1],
            device=encoder_states.device,
        )
        mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        previous = mask.to(encoder_states.dtype) / lengths.clamp_min(1).to(
            encoder_states.dtype, ).unsqueeze(1)
        return LocationAttentionState(
            encoded_projection=self.encoder_projection(encoder_states),
            mask=mask,
            previous_attention=previous,
        )

    def forward(
        self,
        encoder_states: Tensor,
        decoder_state: Tensor,
        state: LocationAttentionState,
    ) -> tuple[Tensor, Tensor, LocationAttentionState]:
        location = self.location_convolution(state.previous_attention.unsqueeze(1), ).transpose(1, 2)
        location = self.location_projection(location)
        decoder = self.decoder_projection(decoder_state.unsqueeze(1))
        scores = self.score_projection(torch.tanh(state.encoded_projection + decoder +
                                                  location, )).squeeze(-1)
        scores = scores.masked_fill(~state.mask, float("-inf"))
        attention = scores.softmax(dim=-1)
        context = torch.bmm(
            attention.unsqueeze(1),
            encoder_states,
        ).squeeze(1)
        context = self.output_projection(context)
        return (
            context,
            attention,
            LocationAttentionState(
                encoded_projection=state.encoded_projection,
                mask=state.mask,
                previous_attention=attention.detach(),
            ),
        )


class SpeechBrainAttentionalGRUDecoder(nn.Module):
    """Autoregressive GRUCell decoder with location-aware attention."""

    def __init__(self, config: SpeechBrainCRDNNASRConfig) -> None:
        super().__init__()
        self.config = config
        self.attention = SpeechBrainLocationAttention(config)
        cells = [nn.GRUCell(
            config.embedding_size + config.attention_dim,
            config.decoder_neurons,
        )]
        cells.extend(nn.GRUCell(config.decoder_neurons, config.decoder_neurons) for _ in range(1, 1))
        self.rnn_cells = nn.ModuleList(cells)
        self.dropout_layers = nn.ModuleList()
        self.input_dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(
            config.decoder_neurons + config.attention_dim,
            config.decoder_neurons,
        )

    def forward_step(
        self,
        embedded_input: Tensor,
        hidden: Tensor | None,
        context: Tensor,
        encoder_states: Tensor,
        attention_state: LocationAttentionState,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, LocationAttentionState]:
        cell_input = self.input_dropout(torch.cat((embedded_input, context), dim=-1), )
        if hidden is None:
            hidden = cell_input.new_zeros(
                len(self.rnn_cells),
                cell_input.shape[0],
                self.config.decoder_neurons,
            )
        current = self.rnn_cells[0](cell_input, hidden[0])
        hidden_rows = [current]
        for index, cell in enumerate(self.rnn_cells[1:], start=1):
            current = cell(
                self.dropout_layers[index - 1](current),
                hidden[index],
            )
            hidden_rows.append(current)
        next_hidden = torch.stack(hidden_rows)
        next_context, attention, attention_state = self.attention(
            encoder_states,
            current,
            attention_state,
        )
        output = self.output_projection(torch.cat((next_context, current), dim=-1), )
        return (
            output,
            next_hidden,
            next_context,
            attention,
            attention_state,
        )

    def forward(
        self,
        embedded_inputs: Tensor,
        encoder_states: Tensor,
        relative_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        encoder_lengths = torch.round(relative_lengths * encoder_states.shape[1], ).long()
        attention_state = self.attention.initialize(
            encoder_states,
            encoder_lengths,
        )
        context = encoder_states.new_zeros(
            encoder_states.shape[0],
            self.config.attention_dim,
        )
        hidden = None
        outputs = []
        weights = []
        for step in range(embedded_inputs.shape[1]):
            (
                output,
                hidden,
                context,
                attention,
                attention_state,
            ) = self.forward_step(
                embedded_inputs[:, step],
                hidden,
                context,
                encoder_states,
                attention_state,
            )
            outputs.append(output)
            weights.append(attention)
        if not outputs:
            raise ValueError("Decoder inputs must contain at least one token.")
        return torch.stack(outputs, dim=1), torch.stack(weights, dim=1)


class SpeechBrainRNNLanguageModel(nn.Module):
    """Released two-layer LSTM language model used for shallow fusion."""

    def __init__(self, config: SpeechBrainCRDNNASRConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            config.output_neurons,
            config.embedding_size,
        )
        self.dropout = nn.Dropout(config.lm_dropout)
        self.rnn = nn.LSTM(
            config.embedding_size,
            config.lm_rnn_neurons,
            num_layers=config.lm_rnn_layers,
            dropout=(config.lm_dropout if config.lm_rnn_layers > 1 else 0.0),
            batch_first=True,
        )
        blocks = []
        input_size = config.lm_rnn_neurons
        for _ in range(config.lm_dnn_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Linear(input_size, config.lm_dnn_neurons),
                    nn.LayerNorm(config.lm_dnn_neurons),
                    nn.LeakyReLU(config.negative_slope),
                    nn.Dropout(config.lm_dropout),
                ))
            input_size = config.lm_dnn_neurons
        self.dnn_blocks = nn.ModuleList(blocks)
        self.output = nn.Linear(input_size, config.output_neurons)

    def forward(
        self,
        token_ids: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        values = self.dropout(self.embedding(token_ids))
        squeeze = values.ndim == 2
        if squeeze:
            values = values.unsqueeze(1)
        self.rnn.flatten_parameters()
        values, hidden = self.rnn(values, hidden)
        for block in self.dnn_blocks:
            values = block(values)
        logits = self.output(values)
        return (logits.squeeze(1) if squeeze else logits), hidden


@dataclass(slots=True)
class SpeechBrainASROutput:
    """Differentiable outputs shared by training and decoding."""

    loss: Tensor | None
    seq2seq_loss: Tensor | None
    ctc_loss: Tensor | None
    sequence_logits: Tensor | None
    ctc_logits: Tensor | None
    encoder_states: Tensor
    encoder_lengths: Tensor
    relative_lengths: Tensor
    attention: Tensor | None


def speechbrain_sequence_loss(
    logits: Tensor,
    targets: Tensor,
    target_lengths: Tensor,
    *,
    label_smoothing: float,
) -> Tensor:
    log_probabilities = logits.log_softmax(dim=-1)
    target_steps = min(logits.shape[1], targets.shape[1])
    log_probabilities = log_probabilities[:, :target_steps]
    targets = targets[:, :target_steps]
    positions = torch.arange(target_steps, device=logits.device)
    mask = positions.unsqueeze(0) < target_lengths.unsqueeze(1)
    selected = log_probabilities.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    nll = -(selected * mask).sum() / mask.sum().clamp_min(1)
    if label_smoothing == 0.0:
        return nll
    regularizer = (log_probabilities.mean(dim=-1) * mask).sum() / mask.sum().clamp_min(1)
    return ((1.0 - label_smoothing) * nll - label_smoothing * regularizer)


class SpeechBrainCRDNNForASR(nn.Module):
    """Raw-waveform inference and exact combined CTC/seq2seq fine-tuning."""

    def __init__(
        self,
        config: SpeechBrainCRDNNASRConfig | dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = SpeechBrainCRDNNASRConfig.coerce(config)
        self.frontend = SpeechBrainASRFrontend(self.config)
        self.encoder = SpeechBrainCRDNNEncoder(self.config)
        self.embedding = nn.Embedding(
            self.config.output_neurons,
            self.config.embedding_size,
        )
        self.decoder = SpeechBrainAttentionalGRUDecoder(self.config)
        self.ctc_linear = nn.Linear(
            self.config.dnn_neurons,
            self.config.output_neurons,
        )
        self.sequence_linear = nn.Linear(
            self.config.decoder_neurons,
            self.config.output_neurons,
        )
        self.language_model = SpeechBrainRNNLanguageModel(self.config)
        if self.config.freeze_language_model:
            self.language_model.requires_grad_(False)

    def encode(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor | None = None,
        *,
        epoch: int = 0,
        update_normalization: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        features, relative_lengths, _ = self.frontend(
            waveforms,
            waveform_lengths,
            epoch=epoch,
            update_statistics=update_normalization,
        )
        states = self.encoder(features)
        lengths = torch.round(relative_lengths * states.shape[1], ).long()
        return states, lengths, relative_lengths

    def forward(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor | None = None,
        *,
        tokens_bos: Tensor | None = None,
        tokens_eos: Tensor | None = None,
        token_lengths: Tensor | None = None,
        ctc_tokens: Tensor | None = None,
        ctc_token_lengths: Tensor | None = None,
        epoch: int = 1,
        update_normalization: bool | None = None,
    ) -> SpeechBrainASROutput:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError("`epoch` must be a positive integer.")
        if update_normalization is None:
            update_normalization = self.training
        states, encoder_lengths, relative_lengths = self.encode(
            waveforms,
            waveform_lengths,
            epoch=epoch,
            update_normalization=update_normalization,
        )
        sequence_logits = None
        attention = None
        seq_loss = None
        if tokens_bos is not None:
            tokens_bos = torch.as_tensor(
                tokens_bos,
                dtype=torch.long,
                device=states.device,
            )
            if (tokens_bos.ndim != 2 or tokens_bos.shape[0] != states.shape[0]):
                raise ValueError("`tokens_bos` must have shape [batch, tokens].")
            decoded, attention = self.decoder(
                self.embedding(tokens_bos),
                states,
                relative_lengths,
            )
            sequence_logits = self.sequence_linear(decoded)
            if tokens_eos is not None:
                targets = torch.as_tensor(
                    tokens_eos,
                    dtype=torch.long,
                    device=states.device,
                )
                if token_lengths is None:
                    token_lengths = (targets >= 0).sum(dim=1)
                else:
                    token_lengths = torch.as_tensor(
                        token_lengths,
                        dtype=torch.long,
                        device=states.device,
                    )
                safe_targets = targets.clamp_min(0)
                seq_loss = speechbrain_sequence_loss(
                    sequence_logits,
                    safe_targets,
                    token_lengths,
                    label_smoothing=self.config.label_smoothing,
                )

        ctc_logits = None
        ctc_objective = None
        ctc_active = epoch <= self.config.number_of_ctc_epochs
        if ctc_active:
            ctc_logits = self.ctc_linear(states)
        if ctc_tokens is not None and ctc_active:
            targets = torch.as_tensor(
                ctc_tokens,
                dtype=torch.long,
                device=states.device,
            )
            if ctc_token_lengths is None:
                ctc_token_lengths = (targets >= 0).sum(dim=1)
            else:
                ctc_token_lengths = torch.as_tensor(
                    ctc_token_lengths,
                    dtype=torch.long,
                    device=states.device,
                )
            safe_targets = targets.clamp_min(0)
            ctc_input_lengths = torch.floor(relative_lengths * states.shape[1], ).long().clamp_min(1)
            ctc_objective = functional.ctc_loss(
                ctc_logits.log_softmax(dim=-1).transpose(0, 1),
                safe_targets,
                ctc_input_lengths,
                ctc_token_lengths,
                blank=self.config.blank_token_id,
                reduction="mean",
                zero_infinity=True,
            )
        if seq_loss is None:
            loss = ctc_objective
        elif ctc_objective is not None and ctc_active:
            loss = (self.config.ctc_weight * ctc_objective + (1.0 - self.config.ctc_weight) * seq_loss)
        else:
            loss = seq_loss
        return SpeechBrainASROutput(
            loss=loss,
            seq2seq_loss=seq_loss,
            ctc_loss=ctc_objective,
            sequence_logits=sequence_logits,
            ctc_logits=ctc_logits,
            encoder_states=states,
            encoder_lengths=encoder_lengths,
            relative_lengths=relative_lengths,
            attention=attention,
        )


__all__ = [
    "LocationAttentionState",
    "SpeechBrainASROutput",
    "SpeechBrainAttentionalGRUDecoder",
    "SpeechBrainCNNBlock",
    "SpeechBrainCRDNNEncoder",
    "SpeechBrainCRDNNForASR",
    "SpeechBrainDNNBlock",
    "SpeechBrainLocationAttention",
    "SpeechBrainRNNLanguageModel",
    "speechbrain_sequence_loss",
]
