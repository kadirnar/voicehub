"""VoiceHub-owned PyTorch implementation of multilingual MarbleNet Frame-
VAD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.models.vad_nemo.native.configuration import MarbleNetVADConfig
from voicehub.models.vad_nemo.native.frontend import MarbleNetAudioPreprocessor, MarbleNetSpecAugment
from voicehub.models.vad_nemo.native.objective import marblenet_vad_loss


class MaskedConv1d(nn.Module):
    """Length-aware Conv1d with NeMo-compatible parameter names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

    def output_lengths(self, lengths: Tensor) -> Tensor:
        convolution = self.conv
        if (convolution.stride[0] == 1 and
                2 * convolution.padding[0] == convolution.dilation[0] * (convolution.kernel_size[0] - 1)):
            return lengths
        return (
            torch.div(
                lengths + 2 * convolution.padding[0] - convolution.dilation[0] *
                (convolution.kernel_size[0] - 1) - 1,
                convolution.stride[0],
                rounding_mode="trunc",
            ) + 1)

    def forward(self, inputs: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        positions = torch.arange(inputs.shape[-1], device=inputs.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        values = inputs * valid.unsqueeze(1).to(dtype=inputs.dtype)
        return self.conv(values), self.output_lengths(lengths)


def _conv_norm(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int,
    dilation: int,
    separable: bool,
) -> list[nn.Module]:
    padding = dilation * (kernel_size - 1) // 2
    if separable:
        layers: list[nn.Module] = [
            MaskedConv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                groups=in_channels,
            ),
            MaskedConv1d(
                in_channels,
                out_channels,
                1,
            ),
        ]
    else:
        layers = [
            MaskedConv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            )
        ]
    layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))
    return layers


class MarbleNetBlock(nn.Module):
    """One depthwise-separable Jasper block from the released graph."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        repeat: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
        residual: bool,
        separable: bool,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current_channels = in_channels
        for repeat_index in range(repeat):
            modules.extend(
                _conv_norm(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    separable=separable,
                ))
            if repeat_index < repeat - 1:
                modules.extend((nn.ReLU(inplace=True), nn.Dropout(dropout)))
            current_channels = out_channels
        self.mconv = nn.ModuleList(modules)
        if residual:
            self.res: nn.ModuleList | None = nn.ModuleList([
                nn.ModuleList(
                    _conv_norm(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        dilation=1,
                        separable=False,
                    ))
            ])
        else:
            self.res = None
        self.mout = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(dropout))

    @staticmethod
    def _run(
        modules: nn.ModuleList,
        values: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        current_lengths = lengths
        for module in modules:
            if isinstance(module, MaskedConv1d):
                values, current_lengths = module(values, current_lengths)
            else:
                values = module(values)
        return values, current_lengths

    def forward(self, inputs: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        output, output_lengths = self._run(self.mconv, inputs, lengths)
        if self.res is not None:
            residual, _ = self._run(self.res[0], inputs, lengths)
            output = output + residual
        return self.mout(output), output_lengths


class MarbleNetEncoder(nn.Module):
    """The exact 80→128→64×3→128×2 released encoder."""

    _LAYERS = (
        (128, 1, 11, 2, 1, False, True),
        (64, 2, 13, 1, 1, True, True),
        (64, 2, 15, 1, 1, True, True),
        (64, 2, 17, 1, 1, True, True),
        (128, 1, 29, 1, 2, False, True),
        (128, 1, 1, 1, 1, False, False),
    )

    def __init__(self, config: MarbleNetVADConfig) -> None:
        super().__init__()
        current_channels = config.num_mel_bins
        blocks = []
        for (
                out_channels,
                repeat,
                kernel_size,
                stride,
                dilation,
                residual,
                separable,
        ) in self._LAYERS:
            blocks.append(
                MarbleNetBlock(
                    current_channels,
                    out_channels,
                    repeat=repeat,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    dropout=config.dropout,
                    residual=residual,
                    separable=separable,
                ))
            current_channels = out_channels
        self.encoder = nn.ModuleList(blocks)

    def forward(self, audio_signal: Tensor, length: Tensor) -> tuple[Tensor, Tensor]:
        values = audio_signal
        lengths = length
        for block in self.encoder:
            values, lengths = block(values, lengths)
        return values, lengths


class MarbleNetFrameDecoder(nn.Module):
    """Per-frame linear decoder with the official `decoder.layer0`
    namespace."""

    def __init__(self) -> None:
        super().__init__()
        self.layer0 = nn.Linear(128, 2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.layer0(hidden_states)


@dataclass(slots=True)
class MarbleNetVADOutput:
    """Differentiable frame-classification result."""

    logits: Tensor
    probabilities: Tensor
    speech_probabilities: Tensor
    frame_lengths: Tensor
    loss: Tensor | None = None
    frame_mask: Tensor | None = None


class MarbleNetVADModel(nn.Module):
    """Native frontend, MarbleNet encoder, decoder, and fine-tuning loss."""

    def __init__(self, config: MarbleNetVADConfig | dict[str, Any]) -> None:
        super().__init__()
        self.config = MarbleNetVADConfig.coerce(config)
        self.preprocessor = MarbleNetAudioPreprocessor(self.config)
        self.spec_augmentation = MarbleNetSpecAugment(self.config)
        self.encoder = MarbleNetEncoder(self.config)
        self.decoder = MarbleNetFrameDecoder()

    def forward(
        self,
        waveforms: Tensor | None = None,
        *,
        input_values: Tensor | None = None,
        waveform_lengths: Tensor | None = None,
        processed_features: Tensor | None = None,
        feature_lengths: Tensor | None = None,
        labels: Tensor | None = None,
        label_mask: Tensor | None = None,
    ) -> MarbleNetVADOutput:
        if waveforms is not None and input_values is not None:
            raise TypeError("Pass `waveforms` or `input_values`, not both.")
        if waveforms is None:
            waveforms = input_values
        if processed_features is None:
            if not isinstance(waveforms, Tensor):
                raise TypeError("Raw-audio execution requires a `waveforms` tensor.")
            if waveforms.ndim == 1:
                waveforms = waveforms.unsqueeze(0)
            if waveform_lengths is None:
                waveform_lengths = torch.full(
                    (waveforms.shape[0], ),
                    waveforms.shape[-1],
                    dtype=torch.long,
                    device=waveforms.device,
                )
            processed_features, feature_lengths = self.preprocessor(
                waveforms,
                waveform_lengths,
            )
        elif waveforms is not None:
            raise TypeError("Pass raw waveforms or precomputed features, not both.")
        if not isinstance(processed_features, Tensor) or processed_features.ndim != 3:
            raise ValueError("`processed_features` must have shape [batch, 80, frames].")
        if processed_features.shape[1] != self.config.num_mel_bins:
            raise ValueError(
                f"Expected {self.config.num_mel_bins} mel bins, found "
                f"{processed_features.shape[1]}.")
        if feature_lengths is None:
            feature_lengths = torch.full(
                (processed_features.shape[0], ),
                processed_features.shape[-1],
                dtype=torch.long,
                device=processed_features.device,
            )
        else:
            feature_lengths = torch.as_tensor(
                feature_lengths,
                dtype=torch.long,
                device=processed_features.device,
            )
        if self.training:
            processed_features = self.spec_augmentation(
                processed_features,
                feature_lengths,
            )
        encoded, frame_lengths = self.encoder(processed_features, feature_lengths)
        logits = self.decoder(encoded.transpose(1, 2))
        probabilities = logits.softmax(dim=-1)
        speech = probabilities[..., self.config.speech_class_id]
        positions = torch.arange(logits.shape[1], device=logits.device)
        length_mask = positions.unsqueeze(0) < frame_lengths.unsqueeze(1)
        effective_mask = length_mask
        if label_mask is not None:
            supplied_mask = torch.as_tensor(
                label_mask,
                dtype=torch.bool,
                device=logits.device,
            )
            if supplied_mask.shape != length_mask.shape:
                raise ValueError("`label_mask` must have shape [batch, frames].")
            effective_mask = length_mask & supplied_mask
        loss = (None if labels is None else marblenet_vad_loss(
            logits,
            labels,
            label_mask=effective_mask,
        ))
        return MarbleNetVADOutput(
            logits=logits,
            probabilities=probabilities,
            speech_probabilities=speech,
            frame_lengths=frame_lengths,
            loss=loss,
            frame_mask=effective_mask,
        )


__all__ = [
    "MarbleNetBlock",
    "MarbleNetEncoder",
    "MarbleNetFrameDecoder",
    "MarbleNetVADModel",
    "MarbleNetVADOutput",
    "MaskedConv1d",
]
