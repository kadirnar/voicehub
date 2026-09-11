"""VoiceHub-owned PyTorch implementation of FunASR's FSMN VAD graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.vad_funasr.native.configuration import FSMNVADConfig
from voicehub.models.vad_funasr.native.frontend import FSMNVADFrontend
from voicehub.models.vad_funasr.native.objective import fsmn_vad_loss


class LinearTransform(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(inputs)


class AffineTransform(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(inputs)


class RectifiedLinear(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.relu(inputs)


class FSMNMemoryBlock(nn.Module):
    """Depthwise tapped-delay memory with explicit request-local cache."""

    def __init__(
        self,
        dimension: int,
        *,
        left_order: int,
        right_order: int,
        left_stride: int,
        right_stride: int,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.left_order = left_order
        self.right_order = right_order
        self.left_stride = left_stride
        self.right_stride = right_stride
        self.conv_left = nn.Conv2d(
            dimension,
            dimension,
            (left_order, 1),
            dilation=(left_stride, 1),
            groups=dimension,
            bias=False,
        )
        self.conv_right = (
            nn.Conv2d(
                dimension,
                dimension,
                (right_order, 1),
                dilation=(right_stride, 1),
                groups=dimension,
                bias=False,
            ) if right_order > 0 else None)

    @property
    def cache_frames(self) -> int:
        return (self.left_order - 1) * self.left_stride

    def forward(
        self,
        inputs: Tensor,
        cache: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        values = inputs.unsqueeze(1).permute(0, 3, 2, 1)
        if cache is None:
            left = functional.pad(
                values,
                (0, 0, self.cache_frames, 0),
            )
            next_cache = None
        else:
            if (cache.ndim != 4 or cache.shape[0] != values.shape[0] or cache.shape[1] != values.shape[1] or
                    cache.shape[2] != self.cache_frames or cache.shape[3] != 1):
                raise ValueError(
                    "FSMN cache must have shape "
                    f"[batch, {self.dimension}, {self.cache_frames}, 1].")
            left = torch.cat(
                (cache.to(device=values.device, dtype=values.dtype), values),
                dim=2,
            )
            next_cache = left[:, :, -self.cache_frames:].detach()
        output = values + self.conv_left(left)
        if self.conv_right is not None:
            right = functional.pad(
                values,
                (0, 0, 0, self.right_order * self.right_stride),
            )
            right = right[:, :, self.right_stride:]
            output = output + self.conv_right(right)
        return output.permute(0, 3, 2, 1).squeeze(1), next_cache


class FSMNLayer(nn.Module):

    def __init__(
        self,
        config: FSMNVADConfig,
        *,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.linear = LinearTransform(
            config.linear_dim,
            config.projection_dim,
        )
        self.fsmn_block = FSMNMemoryBlock(
            config.projection_dim,
            left_order=config.left_order,
            right_order=config.right_order,
            left_stride=config.left_stride,
            right_stride=config.right_stride,
        )
        self.affine = AffineTransform(
            config.projection_dim,
            config.linear_dim,
        )
        self.relu = RectifiedLinear()

    @property
    def cache_name(self) -> str:
        return f"cache_layer_{self.layer_index}"

    def forward(
        self,
        inputs: Tensor,
        cache: dict[str, Tensor] | None = None,
    ) -> Tensor:
        projected = self.linear(inputs)
        layer_cache = None
        if cache is not None:
            layer_cache = cache.get(self.cache_name)
            if layer_cache is None:
                layer_cache = projected.new_zeros(
                    projected.shape[0],
                    projected.shape[-1],
                    self.fsmn_block.cache_frames,
                    1,
                )
        memory, next_cache = self.fsmn_block(projected, layer_cache)
        if cache is not None and next_cache is not None:
            cache[self.cache_name] = next_cache
        return self.relu(self.affine(memory))


class FSMNEncoder(nn.Module):
    """The exact 400→140→250→4×FSMN→140→248 network."""

    def __init__(self, config: FSMNVADConfig) -> None:
        super().__init__()
        self.config = FSMNVADConfig.coerce(config)
        self.in_linear1 = AffineTransform(
            self.config.input_dim,
            self.config.input_affine_dim,
        )
        self.in_linear2 = AffineTransform(
            self.config.input_affine_dim,
            self.config.linear_dim,
        )
        self.relu = RectifiedLinear()
        self.fsmn = nn.ModuleList(
            [FSMNLayer(self.config, layer_index=index) for index in range(self.config.fsmn_layers)])
        self.out_linear1 = AffineTransform(
            self.config.linear_dim,
            self.config.output_affine_dim,
        )
        self.out_linear2 = AffineTransform(
            self.config.output_affine_dim,
            self.config.output_dim,
        )

    def forward(
        self,
        inputs: Tensor,
        cache: dict[str, Tensor] | None = None,
    ) -> Tensor:
        values = self.relu(self.in_linear2(self.in_linear1(inputs)))
        for layer in self.fsmn:
            values = layer(values, cache)
        return self.out_linear2(self.out_linear1(values))


@dataclass(slots=True)
class FSMNVADOutput:
    """Differentiable frame output."""

    logits: Tensor
    probabilities: Tensor
    speech_probabilities: Tensor
    loss: Tensor | None = None
    objective: str | None = None
    frame_mask: Tensor | None = None


class FSMNVADModel(nn.Module):
    """Native frontend, FSMN encoder, and model-owned fine-tuning loss."""

    def __init__(
        self,
        config: FSMNVADConfig | dict[str, Any],
        *,
        cmvn_shift: Tensor | None = None,
        cmvn_scale: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = FSMNVADConfig.coerce(config)
        self.frontend = FSMNVADFrontend(
            self.config,
            cmvn_shift=cmvn_shift,
            cmvn_scale=cmvn_scale,
        )
        self.encoder = FSMNEncoder(self.config)

    def frame_count(self, sample_count: int) -> int:
        return self.frontend.frame_count(sample_count)

    def forward(
        self,
        waveforms: Tensor | None = None,
        *,
        input_values: Tensor | None = None,
        features: Tensor | None = None,
        waveform_lengths: Tensor | None = None,
        labels: Tensor | None = None,
        label_mask: Tensor | None = None,
        target_kind: str = "auto",
        cache: dict[str, Tensor] | None = None,
        final: bool = True,
    ) -> FSMNVADOutput:
        if waveforms is not None and input_values is not None:
            raise TypeError("Pass `waveforms` or `input_values`, not both.")
        if waveforms is None:
            waveforms = input_values
        if features is None:
            if not isinstance(waveforms, Tensor):
                raise TypeError("Raw-audio execution requires a `waveforms` tensor.")
            if waveforms.ndim == 1:
                waveforms = waveforms.unsqueeze(0)
            features = self.frontend(waveforms, final=final)
        elif waveforms is not None:
            raise TypeError("Pass raw waveforms or precomputed features, not both.")
        if not isinstance(features, Tensor) or features.ndim != 3:
            raise ValueError("`features` must have shape [batch, frames, 400].")
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.config.input_dim}, "
                f"found {features.shape[-1]}.")
        logits = self.encoder(features, cache=cache)
        probabilities = logits.softmax(dim=-1)
        silence = probabilities[..., list(self.config.silence_pdf_ids)].sum(dim=-1)
        speech = 1.0 - silence

        frame_mask = label_mask
        if waveform_lengths is not None:
            if not isinstance(waveform_lengths, Tensor):
                waveform_lengths = torch.as_tensor(
                    waveform_lengths,
                    device=logits.device,
                )
            waveform_lengths = waveform_lengths.to(device=logits.device)
            if waveform_lengths.ndim != 1 or waveform_lengths.shape[0] != logits.shape[0]:
                raise ValueError("`waveform_lengths` must have shape [batch].")
            lengths = torch.div(
                (waveform_lengths - self.config.frame_length_samples).clamp_min(0),
                self.config.frame_shift_samples,
                rounding_mode="floor",
            ) + (waveform_lengths >= self.config.frame_length_samples)
            indices = torch.arange(
                logits.shape[1],
                device=logits.device,
            ).unsqueeze(0)
            length_mask = indices < lengths.unsqueeze(1)
            frame_mask = (
                length_mask if frame_mask is None else length_mask & frame_mask.to(
                    device=logits.device,
                    dtype=torch.bool,
                ))

        loss = None
        objective = None
        if labels is not None:
            loss, objective = fsmn_vad_loss(
                logits,
                labels,
                silence_pdf_ids=self.config.silence_pdf_ids,
                label_mask=frame_mask,
                target_kind=target_kind,
            )
        return FSMNVADOutput(
            logits=logits,
            probabilities=probabilities,
            speech_probabilities=speech,
            loss=loss,
            objective=objective,
            frame_mask=frame_mask,
        )


__all__ = [
    "AffineTransform",
    "FSMNEncoder",
    "FSMNLayer",
    "FSMNMemoryBlock",
    "FSMNVADModel",
    "FSMNVADOutput",
    "LinearTransform",
    "RectifiedLinear",
]
