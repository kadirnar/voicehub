"""VoiceHub-owned differentiable implementation of the TEN VAD graph."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.vad_ten.native.configuration import TENVADConfig
from voicehub.models.vad_ten.native.frontend import TENVADFrontend, TENVADFrontendState
from voicehub.models.vad_ten.native.objective import ten_vad_binary_cross_entropy


@dataclass(frozen=True, slots=True)
class TENVADRecurrentState:
    """Hidden and cell tensors for the two released LSTM layers."""

    hidden_1: Tensor
    cell_1: Tensor
    hidden_2: Tensor
    cell_2: Tensor

    def detached(self, *, clone: bool = False) -> TENVADRecurrentState:

        def convert(value: Tensor) -> Tensor:
            result = value.detach()
            return result.clone() if clone else result

        return TENVADRecurrentState(
            hidden_1=convert(self.hidden_1),
            cell_1=convert(self.cell_1),
            hidden_2=convert(self.hidden_2),
            cell_2=convert(self.cell_2),
        )


@dataclass(frozen=True, slots=True)
class TENVADState:
    """All request-local acoustic and recurrent streaming state."""

    frontend: TENVADFrontendState
    recurrent: TENVADRecurrentState

    def detached(self, *, clone: bool = False) -> TENVADState:
        return TENVADState(
            frontend=self.frontend.detached(clone=clone),
            recurrent=self.recurrent.detached(clone=clone),
        )


@dataclass(slots=True)
class TENVADFrameOutput:
    """One context window's speech score and next recurrent state."""

    logits: Tensor
    speech_probabilities: Tensor
    state: TENVADRecurrentState


@dataclass(slots=True)
class TENVADOutput:
    """Differentiable sequence output for inference or fine-tuning."""

    logits: Tensor
    speech_probabilities: Tensor
    state: TENVADState
    frame_mask: Tensor
    loss: Tensor | None = None


class _ONNXLSTMCell(nn.Module):
    """Single-step ONNX LSTM using the specification's IOFC gate order."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(hidden_size * 4, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size * 4, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(hidden_size * 4))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size * 4))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use PyTorch's standard recurrent-cell initialization."""
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(
        self,
        inputs: Tensor,
        state: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        hidden, cell = state
        gates = (
            functional.linear(inputs, self.weight_ih, self.bias_ih) +
            functional.linear(hidden, self.weight_hh, self.bias_hh))
        input_gate, output_gate, forget_gate, candidate = gates.chunk(4, dim=-1)
        input_gate = input_gate.sigmoid()
        output_gate = output_gate.sigmoid()
        forget_gate = forget_gate.sigmoid()
        candidate = candidate.tanh()
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * cell.tanh()
        return hidden, cell


class TENVADModel(nn.Module):
    """Released separable-convolution, two-LSTM, and dense TEN graph.

    The model contains the reviewed Sherpa acoustic frontend as
    registered buffers, so native Safetensors artifacts are complete and
    do not depend on ONNX metadata at inference or training time.
    """

    def __init__(self, config: TENVADConfig | dict[str, Any]) -> None:
        super().__init__()
        self.config = TENVADConfig.coerce(config)
        channels = self.config.convolution_channels
        self.frontend = TENVADFrontend(self.config)
        self.spatial_depthwise = nn.Conv2d(
            1,
            1,
            kernel_size=(3, 3),
            bias=False,
        )
        self.spatial_pointwise = nn.Conv2d(
            1,
            channels,
            kernel_size=1,
        )
        self.temporal_depthwise_1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 3),
            stride=(2, 2),
            padding=(0, 1),
            groups=channels,
            bias=False,
        )
        self.temporal_pointwise_1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )
        self.temporal_depthwise_2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 3),
            stride=(2, 2),
            groups=channels,
            bias=False,
        )
        self.temporal_pointwise_2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )
        self.lstm_1 = _ONNXLSTMCell(80, self.config.recurrent_size)
        self.lstm_2 = _ONNXLSTMCell(
            self.config.recurrent_size,
            self.config.recurrent_size,
        )
        self.dense = nn.Linear(
            self.config.recurrent_size * 2,
            self.config.dense_size,
        )
        self.output = nn.Linear(self.config.dense_size, 1)

    def initial_recurrent_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> TENVADRecurrentState:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("`batch_size` must be an integer.")
        if batch_size < 1:
            raise ValueError("`batch_size` must be positive.")
        target = (next(self.parameters()).device if device is None else torch.device(device))
        zero = torch.zeros(
            batch_size,
            self.config.recurrent_size,
            dtype=torch.float32,
            device=target,
        )
        return TENVADRecurrentState(
            hidden_1=zero,
            cell_1=torch.zeros_like(zero),
            hidden_2=torch.zeros_like(zero),
            cell_2=torch.zeros_like(zero),
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> TENVADState:
        recurrent = self.initial_recurrent_state(batch_size, device=device)
        return TENVADState(
            frontend=self.frontend.initial_state(
                batch_size,
                device=recurrent.hidden_1.device,
            ),
            recurrent=recurrent,
        )

    def _validated_recurrent(
        self,
        state: TENVADRecurrentState | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> TENVADRecurrentState:
        if state is None:
            return self.initial_recurrent_state(batch_size, device=device)
        if not isinstance(state, TENVADRecurrentState):
            raise TypeError("`state` must be a TENVADRecurrentState or None.")
        expected = (batch_size, self.config.recurrent_size)
        for name in ("hidden_1", "cell_1", "hidden_2", "cell_2"):
            value = getattr(state, name)
            if tuple(value.shape) != expected:
                raise ValueError(f"TEN recurrent {name} must have shape {expected}.")
            if value.dtype != torch.float32 or value.device != device:
                raise ValueError("TEN recurrent state must be float32 and colocated with features.")
        return state

    def score_context(
        self,
        context: Tensor,
        state: TENVADRecurrentState | None = None,
    ) -> TENVADFrameOutput:
        """Score normalized features shaped ``[batch, 3, 41]``."""
        if not isinstance(context, Tensor):
            raise TypeError("`context` must be a PyTorch tensor.")
        expected_tail = (
            self.config.context_frames,
            self.config.feature_size,
        )
        if context.ndim != 3 or tuple(context.shape[1:]) != expected_tail:
            raise ValueError(
                f"`context` must have shape [batch, {expected_tail[0]}, "
                f"{expected_tail[1]}].")
        if context.dtype != torch.float32:
            raise TypeError("TEN VAD context features must use float32.")
        if not torch.isfinite(context).all():
            raise ValueError("TEN VAD context cannot contain NaN or infinite values.")
        if context.device != next(self.parameters()).device:
            raise ValueError("TEN model parameters and features must be colocated.")
        state = self._validated_recurrent(
            state,
            batch_size=context.shape[0],
            device=context.device,
        )

        values = context.reshape(context.shape[0], 1, 3, 41)
        values = self.spatial_depthwise(values)
        values = functional.relu(self.spatial_pointwise(values))
        values = functional.max_pool2d(
            values,
            kernel_size=(1, 3),
            stride=(1, 2),
        )
        values = self.temporal_depthwise_1(values)
        values = functional.relu(self.temporal_pointwise_1(values))
        values = functional.pad(values, (0, 1, 0, 0))
        values = self.temporal_depthwise_2(values)
        values = functional.relu(self.temporal_pointwise_2(values))
        if tuple(values.shape[1:]) != (16, 1, 5):
            raise RuntimeError(f"TEN convolution stack produced unexpected shape {tuple(values.shape)}.")
        values = values.squeeze(2).transpose(1, 2).reshape(
            context.shape[0],
            80,
        )
        hidden_1, cell_1 = self.lstm_1(
            values,
            (state.hidden_1, state.cell_1),
        )
        hidden_2, cell_2 = self.lstm_2(
            hidden_1,
            (state.hidden_2, state.cell_2),
        )
        joined = torch.cat((hidden_2, hidden_1), dim=-1)
        logits = self.output(functional.relu(self.dense(joined))).squeeze(-1)
        return TENVADFrameOutput(
            logits=logits,
            speech_probabilities=logits.sigmoid(),
            state=TENVADRecurrentState(
                hidden_1=hidden_1,
                cell_1=cell_1,
                hidden_2=hidden_2,
                cell_2=cell_2,
            ),
        )

    def score_audio_frame(
        self,
        frame: Tensor,
        state: TENVADState | None = None,
    ) -> tuple[TENVADFrameOutput, TENVADState]:
        if frame.ndim == 1:
            frame = frame.unsqueeze(0)
        if frame.ndim != 2:
            raise ValueError("TEN audio frames must have shape [batch, samples].")
        if state is None:
            state = self.initial_state(frame.shape[0], device=frame.device)
        if not isinstance(state, TENVADState):
            raise TypeError("`state` must be a TENVADState or None.")
        frontend = self.frontend(frame, state.frontend)
        output = self.score_context(frontend.context, state.recurrent)
        return output, TENVADState(
            frontend=frontend.state,
            recurrent=output.state,
        )

    def forward(
        self,
        waveforms: Tensor | None = None,
        *,
        input_values: Tensor | None = None,
        waveform_lengths: Tensor | None = None,
        features: Tensor | None = None,
        state: TENVADState | None = None,
        labels: Tensor | None = None,
        label_mask: Tensor | None = None,
        positive_weight: float | Tensor | None = None,
        pad_final_frame: bool = True,
        detach_state: bool = False,
    ) -> TENVADOutput:
        """Score a raw waveform batch or normalized per-frame features."""
        if waveforms is not None and input_values is not None:
            raise TypeError("Pass `waveforms` or `input_values`, not both.")
        if waveforms is None:
            waveforms = input_values
        if (waveforms is None) == (features is None):
            raise TypeError("Pass exactly one of raw waveforms or normalized features.")
        if not isinstance(pad_final_frame, bool) or not isinstance(detach_state, bool):
            raise TypeError("Frame-padding and state-detachment flags must be booleans.")

        contexts: list[Tensor] = []
        frame_mask: Tensor
        if features is not None:
            if waveform_lengths is not None:
                raise TypeError("`waveform_lengths` cannot be used with precomputed features.")
            if (not isinstance(features, Tensor) or features.ndim != 3 or
                    features.shape[-1] != self.config.feature_size):
                raise ValueError("`features` must have shape [batch, frames, 41].")
            if features.dtype != torch.float32:
                raise TypeError("TEN VAD features must use float32.")
            if not torch.isfinite(features).all():
                raise ValueError("TEN VAD features cannot contain NaN or infinite values.")
            if state is None:
                state = self.initial_state(features.shape[0], device=features.device)
            elif not isinstance(state, TENVADState):
                raise TypeError("`state` must be a TENVADState or None.")
            history = state.frontend.history
            for index in range(features.shape[1]):
                current = features[:, index:index + 1]
                context = torch.cat((history, current), dim=1)
                contexts.append(context)
                history = context[:, 1:]
            state = TENVADState(
                frontend=TENVADFrontendState(
                    previous_sample=state.frontend.previous_sample,
                    history=history,
                ),
                recurrent=state.recurrent,
            )
            frame_mask = torch.ones(
                features.shape[:2],
                dtype=torch.bool,
                device=features.device,
            )
        else:
            if not isinstance(waveforms, Tensor):
                raise TypeError("Raw TEN VAD execution requires a waveform tensor.")
            if waveforms.ndim == 1:
                waveforms = waveforms.unsqueeze(0)
            if waveforms.ndim != 2 or waveforms.shape[1] < 1:
                raise ValueError("`waveforms` must have shape [batch, samples].")
            if waveforms.dtype != torch.float32:
                raise TypeError("TEN VAD waveforms must use float32.")
            if state is None:
                state = self.initial_state(waveforms.shape[0], device=waveforms.device)
            elif not isinstance(state, TENVADState):
                raise TypeError("`state` must be a TENVADState or None.")
            frame_size = self.config.window_size
            complete = ((waveforms.shape[1] + frame_size - 1) //
                        frame_size if pad_final_frame else waveforms.shape[1] // frame_size)
            if complete < 1:
                raise ValueError("Waveform does not contain one complete TEN frame.")
            for index in range(complete):
                frame = waveforms[:, index * frame_size:(index + 1) * frame_size]
                if frame.shape[1] < frame_size:
                    frame = functional.pad(frame, (0, frame_size - frame.shape[1]))
                frontend = self.frontend(frame, state.frontend)
                contexts.append(frontend.context)
                state = TENVADState(
                    frontend=frontend.state,
                    recurrent=state.recurrent,
                )
            if waveform_lengths is None:
                frame_mask = torch.ones(
                    waveforms.shape[0],
                    complete,
                    dtype=torch.bool,
                    device=waveforms.device,
                )
            else:
                lengths = torch.as_tensor(
                    waveform_lengths,
                    dtype=torch.long,
                    device=waveforms.device,
                )
                if tuple(lengths.shape) != (waveforms.shape[0], ):
                    raise ValueError("`waveform_lengths` must contain one value per waveform.")
                if torch.any(lengths < 1) or torch.any(lengths > waveforms.shape[1]):
                    raise ValueError("Waveform lengths are outside the padded batch.")
                frame_counts = (
                    torch.div(lengths + frame_size - 1, frame_size, rounding_mode="floor")
                    if pad_final_frame else torch.div(lengths, frame_size, rounding_mode="floor"))
                indexes = torch.arange(complete, device=waveforms.device)
                frame_mask = indexes.unsqueeze(0) < frame_counts.unsqueeze(1)

        logits = []
        probabilities = []
        recurrent = state.recurrent
        for context in contexts:
            frame_output = self.score_context(context, recurrent)
            logits.append(frame_output.logits)
            probabilities.append(frame_output.speech_probabilities)
            recurrent = frame_output.state
        state = TENVADState(frontend=state.frontend, recurrent=recurrent)
        if detach_state:
            state = state.detached()
        logits_tensor = torch.stack(logits, dim=1)
        probabilities_tensor = torch.stack(probabilities, dim=1)
        loss = None
        if labels is not None:
            labels = torch.as_tensor(
                labels,
                dtype=logits_tensor.dtype,
                device=logits_tensor.device,
            )
            if label_mask is None:
                mask = frame_mask
            else:
                label_mask = torch.as_tensor(
                    label_mask,
                    dtype=torch.bool,
                    device=frame_mask.device,
                )
                if label_mask.shape != logits_tensor.shape:
                    raise ValueError("TEN VAD label mask must match the logits shape.")
                mask = frame_mask & label_mask
            loss = ten_vad_binary_cross_entropy(
                logits_tensor,
                labels,
                mask=mask,
                positive_weight=positive_weight,
            )
        elif label_mask is not None:
            raise ValueError("`label_mask` requires `labels`.")
        return TENVADOutput(
            logits=logits_tensor,
            speech_probabilities=probabilities_tensor,
            state=state,
            frame_mask=frame_mask,
            loss=loss,
        )


__all__ = [
    "TENVADFrameOutput",
    "TENVADModel",
    "TENVADOutput",
    "TENVADRecurrentState",
    "TENVADState",
]
