"""VoiceHub-owned PyTorch implementation of SpeechBrain CRDNN VAD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.vad_speechbrain.native.configuration import SpeechBrainCRDNNVADConfig
from voicehub.models.vad_speechbrain.native.frontend import SpeechBrainVADFrontend
from voicehub.models.vad_speechbrain.native.objective import speechbrain_vad_binary_cross_entropy


class SpeechBrainCNNBlock(nn.Module):
    """Legacy SpeechBrain CNN block using time-major public tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        frequency_bins: int,
        config: SpeechBrainCRDNNVADConfig,
    ) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels,
            out_channels,
            config.cnn_kernel_size,
        )
        self.norm_1 = nn.LayerNorm((frequency_bins, out_channels))
        self.conv_2 = nn.Conv2d(
            out_channels,
            out_channels,
            config.cnn_kernel_size,
        )
        self.norm_2 = nn.LayerNorm((frequency_bins, out_channels))
        self.pool_size = config.cnn_pool_size
        self.slope = config.leaky_relu_slope
        self.dropout = nn.Dropout2d(config.dropout)

    @staticmethod
    def _convolve(inputs: Tensor, convolution: nn.Conv2d) -> Tensor:
        channels_first = inputs.transpose(1, -1)
        padded = functional.pad(channels_first, (1, 1, 1, 1), mode="reflect")
        return convolution(padded).transpose(1, -1)

    def forward(self, inputs: Tensor) -> Tensor:
        values = self._convolve(inputs, self.conv_1)
        values = functional.leaky_relu(
            self.norm_1(values),
            negative_slope=self.slope,
        )
        values = self._convolve(values, self.conv_2)
        values = functional.leaky_relu(
            self.norm_2(values),
            negative_slope=self.slope,
        )
        values = values.transpose(-1, 2)
        values = functional.max_pool2d(
            values,
            kernel_size=(1, self.pool_size),
            stride=(1, self.pool_size),
        )
        values = values.transpose(-1, 2)
        # SpeechBrain's historical Dropout2d wrapper uses this exact axis
        # order.  It is unusual, but changing it would alter fine-tuning.
        dropout_layout = values.transpose(1, 2).transpose(2, -1)
        dropout_layout = self.dropout(dropout_layout)
        return dropout_layout.transpose(-1, 1).transpose(2, -1)


class SpeechBrainDNNBlock(nn.Module):
    """Linear, channel-first BatchNorm, LeakyReLU, and dropout."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: SpeechBrainCRDNNVADConfig,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.norm = nn.BatchNorm1d(output_size)
        self.slope = config.leaky_relu_slope
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        values = self.linear(inputs)
        values = self.norm(values.transpose(1, 2)).transpose(1, 2)
        values = functional.leaky_relu(values, negative_slope=self.slope)
        return self.dropout(values)


@dataclass(slots=True)
class SpeechBrainCRDNNVADOutput:
    """Differentiable CRDNN frame output."""

    logits: Tensor
    speech_probabilities: Tensor
    frame_lengths: Tensor
    loss: Tensor | None = None
    frame_mask: Tensor | None = None


class SpeechBrainCRDNNVADModel(nn.Module):
    """Exact released frontend, CNN, bidirectional GRU, and DNN graph."""

    def __init__(
        self,
        config: SpeechBrainCRDNNVADConfig | dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = SpeechBrainCRDNNVADConfig.coerce(config)
        self.frontend = SpeechBrainVADFrontend(self.config)
        self.initial_norm = nn.LayerNorm(self.config.n_mels)
        blocks = []
        frequency = self.config.n_mels
        in_channels = 1
        for channels in self.config.cnn_channels:
            blocks.append(
                SpeechBrainCNNBlock(
                    in_channels,
                    channels,
                    frequency_bins=frequency,
                    config=self.config,
                ))
            in_channels = channels
            frequency //= self.config.cnn_pool_size
        self.cnn_blocks = nn.ModuleList(blocks)
        self.rnn = nn.GRU(
            input_size=self.config.rnn_input_size,
            hidden_size=self.config.rnn_hidden_size,
            num_layers=self.config.rnn_num_layers,
            dropout=self.config.dropout,
            bidirectional=self.config.rnn_bidirectional,
            batch_first=True,
        )
        rnn_output = self.config.rnn_hidden_size * 2
        dnn = []
        input_size = rnn_output
        for _ in range(self.config.dnn_num_layers):
            dnn.append(SpeechBrainDNNBlock(
                input_size,
                self.config.dnn_hidden_size,
                config=self.config,
            ))
            input_size = self.config.dnn_hidden_size
        self.dnn_blocks = nn.ModuleList(dnn)
        self.output = nn.Linear(input_size, 1, bias=False)

    def frame_count(self, sample_count: int) -> int:
        return self.frontend.frame_count(sample_count)

    def forward(
        self,
        waveforms: Tensor | None = None,
        *,
        input_values: Tensor | None = None,
        waveform_lengths: Tensor | None = None,
        features: Tensor | None = None,
        feature_lengths: Tensor | None = None,
        labels: Tensor | None = None,
        label_mask: Tensor | None = None,
        positive_weight: float | Tensor | None = None,
    ) -> SpeechBrainCRDNNVADOutput:
        if waveforms is not None and input_values is not None:
            raise TypeError("Pass `waveforms` or `input_values`, not both.")
        if waveforms is None:
            waveforms = input_values
        if features is None:
            if not isinstance(waveforms, Tensor):
                raise TypeError("Raw-audio execution requires a waveform tensor.")
            features, frame_lengths = self.frontend(
                waveforms,
                waveform_lengths,
            )
        elif waveforms is not None:
            raise TypeError("Pass raw waveforms or precomputed features, not both.")
        else:
            if not isinstance(features, Tensor) or features.ndim != 3:
                raise ValueError("`features` must have shape [batch, frames, 40].")
            if features.shape[-1] != self.config.n_mels:
                raise ValueError(f"Expected {self.config.n_mels} feature bins.")
            if feature_lengths is None:
                frame_lengths = torch.full(
                    (features.shape[0], ),
                    features.shape[1],
                    dtype=torch.long,
                    device=features.device,
                )
            else:
                frame_lengths = torch.as_tensor(
                    feature_lengths,
                    dtype=torch.long,
                    device=features.device,
                )
        values = self.initial_norm(features).unsqueeze(-1)
        for block in self.cnn_blocks:
            values = block(values)
        values = values.reshape(
            values.shape[0],
            values.shape[1],
            values.shape[2] * values.shape[3],
        )
        values, _ = self.rnn(values)
        for block in self.dnn_blocks:
            values = block(values)
        logits = self.output(values).squeeze(-1)
        probabilities = logits.sigmoid()
        if labels is None:
            target_frames = logits.shape[1]
        else:
            target_values = torch.as_tensor(labels)
            if target_values.ndim not in {2, 3}:
                raise ValueError("`labels` must have shape [batch, frames] or [batch, frames, 1].")
            target_frames = target_values.shape[1]
        target_frames = min(target_frames, logits.shape[1])
        positions = torch.arange(target_frames, device=logits.device)
        effective_mask = positions.unsqueeze(0) < frame_lengths.unsqueeze(1)
        if label_mask is not None:
            supplied = torch.as_tensor(
                label_mask,
                dtype=torch.bool,
                device=logits.device,
            )
            if supplied.shape != effective_mask.shape:
                raise ValueError("`label_mask` must match the label frame shape.")
            effective_mask = effective_mask & supplied
        loss = (
            None if labels is None else speechbrain_vad_binary_cross_entropy(
                logits,
                labels,
                label_mask=effective_mask,
                positive_weight=positive_weight,
            ))
        return SpeechBrainCRDNNVADOutput(
            logits=logits,
            speech_probabilities=probabilities,
            frame_lengths=frame_lengths,
            loss=loss,
            frame_mask=effective_mask,
        )


__all__ = [
    "SpeechBrainCNNBlock",
    "SpeechBrainCRDNNVADModel",
    "SpeechBrainCRDNNVADOutput",
    "SpeechBrainDNNBlock",
]
