"""VoiceHub-owned QuartzNet/Jasper CTC architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.models.asr_nemo.native.configuration import JasperBlockConfig, NeMoQuartzNetCTCConfig
from voicehub.models.asr_nemo.native.frontend import NeMoAudioPreprocessor, NeMoSpecCutout
from voicehub.objectives.ctc import ctc_loss


class MaskedConv1d(nn.Module):
    """Length-aware convolution with NeMo-compatible parameter names."""

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

    def forward(
        self,
        inputs: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        positions = torch.arange(inputs.shape[-1], device=inputs.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        values = inputs * valid.unsqueeze(1).to(dtype=inputs.dtype)
        return self.conv(values), self.output_lengths(lengths)


def _convolution_and_normalization(
    in_channels: int,
    out_channels: int,
    block: JasperBlockConfig,
) -> list[nn.Module]:
    padding = block.dilation * (block.kernel_size - 1) // 2
    if block.separable:
        layers: list[nn.Module] = [
            MaskedConv1d(
                in_channels,
                in_channels,
                block.kernel_size,
                stride=block.stride,
                dilation=block.dilation,
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
                block.kernel_size,
                stride=block.stride,
                dilation=block.dilation,
                padding=padding,
            )
        ]
    layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))
    return layers


def _residual_projection(
    in_channels: int,
    out_channels: int,
) -> list[nn.Module]:
    return [
        MaskedConv1d(
            in_channels,
            out_channels,
            1,
        ),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1),
    ]


class JasperBlock(nn.Module):
    """One additive-residual Jasper block used by QuartzNet15x5."""

    def __init__(
        self,
        in_channels: int,
        config: JasperBlockConfig,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current_channels = in_channels
        for repeat_index in range(config.repeat):
            modules.extend(_convolution_and_normalization(
                current_channels,
                config.filters,
                config,
            ))
            if repeat_index < config.repeat - 1:
                modules.extend((
                    nn.ReLU(inplace=True),
                    nn.Dropout(config.dropout),
                ))
            current_channels = config.filters
        self.mconv = nn.ModuleList(modules)
        if config.residual:
            self.res: nn.ModuleList | None = nn.ModuleList(
                [nn.ModuleList(_residual_projection(
                    in_channels,
                    config.filters,
                ))])
        else:
            self.res = None
        self.mout = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

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

    def forward(
        self,
        inputs: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        output, output_lengths = self._run(self.mconv, inputs, lengths)
        if self.res is not None:
            residual, _ = self._run(self.res[0], inputs, lengths)
            output = output + residual
        return self.mout(output), output_lengths


class QuartzNetEncoder(nn.Module):
    """Configurable Jasper stack with the released ``encoder.encoder``
    namespace."""

    def __init__(self, config: NeMoQuartzNetCTCConfig) -> None:
        super().__init__()
        current_channels = config.num_mel_bins
        blocks = []
        for block_config in config.encoder_blocks:
            blocks.append(JasperBlock(
                current_channels,
                block_config,
            ))
            current_channels = block_config.filters
        self.encoder = nn.ModuleList(blocks)

    def forward(
        self,
        audio_signal: Tensor,
        length: Tensor,
    ) -> tuple[Tensor, Tensor]:
        values = audio_signal
        lengths = length
        for block in self.encoder:
            values, lengths = block(values, lengths)
        return values, lengths


class QuartzNetCTCDecoder(nn.Module):
    """Pointwise character classifier with a trailing CTC blank class."""

    def __init__(self, config: NeMoQuartzNetCTCConfig) -> None:
        super().__init__()
        self.decoder_layers = nn.Sequential(
            nn.Conv1d(
                config.encoder_output_size,
                config.num_classes,
                kernel_size=1,
                bias=True,
            ))

    def raw_logits(self, encoder_output: Tensor) -> Tensor:
        return self.decoder_layers(encoder_output).transpose(1, 2)

    def forward(self, encoder_output: Tensor) -> Tensor:
        return self.raw_logits(encoder_output).log_softmax(dim=-1)


@dataclass(slots=True)
class NeMoCTCOutput:
    """Differentiable QuartzNet output used by inference and VoiceHub
    Trainer."""

    loss: Tensor | None
    logits: Tensor
    log_probabilities: Tensor
    predictions: Tensor
    encoded_lengths: Tensor


class NeMoQuartzNetForCTC(nn.Module):
    """Raw-audio QuartzNet graph with exact NeMo checkpoint namespaces."""

    def __init__(
        self,
        config: NeMoQuartzNetCTCConfig | dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = NeMoQuartzNetCTCConfig.coerce(config)
        self.preprocessor = NeMoAudioPreprocessor(self.config)
        self.spec_augmentation = NeMoSpecCutout(self.config)
        self.encoder = QuartzNetEncoder(self.config)
        self.decoder = QuartzNetCTCDecoder(self.config)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                module.reset_running_stats()
                if module.affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def _objective(
        self,
        logits: Tensor,
        encoded_lengths: Tensor,
        labels: Tensor,
        label_lengths: Tensor | None,
    ) -> Tensor:
        targets = torch.as_tensor(
            labels,
            dtype=torch.long,
            device=logits.device,
        )
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        if targets.ndim != 2 or targets.shape[0] != logits.shape[0]:
            raise ValueError("`labels` must have shape [batch, target_steps].")
        if label_lengths is None:
            label_lengths = (targets >= 0).sum(dim=1)
        else:
            label_lengths = torch.as_tensor(
                label_lengths,
                dtype=torch.long,
                device=logits.device,
            )
        if label_lengths.ndim != 1 or label_lengths.shape[0] != logits.shape[0]:
            raise ValueError("`label_lengths` must have shape [batch].")
        safe_targets = targets.masked_fill(targets < 0, 0)
        losses = ctc_loss(
            logits,
            safe_targets,
            encoded_lengths,
            label_lengths,
            blank=self.config.blank_id,
            reduction="none",
            zero_infinity=True,
        )
        if self.config.ctc_reduction == "none":
            return losses
        if self.config.ctc_reduction == "sum":
            return losses.sum()
        if self.config.ctc_reduction == "mean":
            denominator = label_lengths.clamp_min(1).to(dtype=losses.dtype)
            return (losses / denominator).mean()
        if self.config.ctc_reduction == "mean_volume":
            return losses.sum() / label_lengths.sum().clamp_min(1)
        return losses.mean()

    def forward(
        self,
        input_signal: Tensor | None = None,
        input_signal_length: Tensor | None = None,
        *,
        waveforms: Tensor | None = None,
        waveform_lengths: Tensor | None = None,
        processed_signal: Tensor | None = None,
        processed_signal_length: Tensor | None = None,
        labels: Tensor | None = None,
        label_lengths: Tensor | None = None,
    ) -> NeMoCTCOutput:
        if waveforms is not None:
            if input_signal is not None:
                raise TypeError("Pass `input_signal` or `waveforms`, not both.")
            input_signal = waveforms
        if waveform_lengths is not None:
            if input_signal_length is not None:
                raise TypeError("Pass `input_signal_length` or `waveform_lengths`, not both.")
            input_signal_length = waveform_lengths
        has_raw_audio = input_signal is not None
        has_features = processed_signal is not None
        if has_raw_audio == has_features:
            raise ValueError("Pass exactly one of raw `input_signal` or `processed_signal`.")

        if has_raw_audio:
            if not isinstance(input_signal, Tensor):
                raise TypeError("`input_signal` must be a PyTorch tensor.")
            if input_signal.ndim == 1:
                input_signal = input_signal.unsqueeze(0)
            if input_signal.ndim != 2:
                raise ValueError("`input_signal` must have shape [batch, samples].")
            if input_signal_length is None:
                input_signal_length = torch.full(
                    (input_signal.shape[0], ),
                    input_signal.shape[-1],
                    dtype=torch.long,
                    device=input_signal.device,
                )
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal,
                input_signal_length,
            )
        else:
            if not isinstance(processed_signal, Tensor) or processed_signal.ndim != 3:
                raise ValueError("`processed_signal` must have shape [batch, mel_bins, frames].")
            if processed_signal.shape[1] != self.config.num_mel_bins:
                raise ValueError(
                    f"Expected {self.config.num_mel_bins} mel bins, found "
                    f"{processed_signal.shape[1]}.")
            if processed_signal_length is None:
                processed_signal_length = torch.full(
                    (processed_signal.shape[0], ),
                    processed_signal.shape[-1],
                    dtype=torch.long,
                    device=processed_signal.device,
                )

        processed_signal_length = torch.as_tensor(
            processed_signal_length,
            dtype=torch.long,
            device=processed_signal.device,
        )
        if (processed_signal_length.ndim != 1 or
                processed_signal_length.shape[0] != processed_signal.shape[0]):
            raise ValueError("Processed-signal lengths must have shape [batch].")
        if self.training:
            processed_signal = self.spec_augmentation(
                processed_signal,
                processed_signal_length,
            )
        encoded, encoded_lengths = self.encoder(
            processed_signal,
            processed_signal_length,
        )
        logits = self.decoder.raw_logits(encoded)
        log_probabilities = logits.log_softmax(dim=-1)
        predictions = log_probabilities.argmax(dim=-1)
        loss = (
            None if labels is None else self._objective(
                logits,
                encoded_lengths,
                labels,
                label_lengths,
            ))
        return NeMoCTCOutput(
            loss=loss,
            logits=logits,
            log_probabilities=log_probabilities,
            predictions=predictions,
            encoded_lengths=encoded_lengths,
        )


__all__ = [
    "JasperBlock",
    "MaskedConv1d",
    "NeMoCTCOutput",
    "NeMoQuartzNetForCTC",
    "QuartzNetCTCDecoder",
    "QuartzNetEncoder",
]
