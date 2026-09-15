"""PyTorch-only Silero VAD graph with explicit streaming state.

The convolution, LSTM, and state flow mirror the official v6.2.1 source and
TorchScript graph at revision
``7e30209a3e901f9842f81b225f3e93d8199902b1``.  VoiceHub executes this module
directly; it never imports ``silero_vad`` or executes an upstream scripted
module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.architectures.silero_vad.configuration import SileroVADConfig


def _floating_audio(
    input_values: Tensor,
    *,
    frame_size: int | None,
) -> Tensor:
    if not isinstance(input_values, Tensor):
        raise TypeError("`input_values` must be a PyTorch tensor.")
    if input_values.ndim == 1:
        input_values = input_values.unsqueeze(0)
    if input_values.ndim != 2:
        raise ValueError("`input_values` must have shape [batch, samples].")
    if input_values.shape[0] < 1 or input_values.shape[1] < 1:
        raise ValueError("`input_values` must contain non-empty waveforms.")
    if frame_size is not None and input_values.shape[1] != frame_size:
        raise ValueError(
            f"A Silero VAD frame must contain exactly {frame_size} samples; "
            f"found {input_values.shape[1]}.")
    if input_values.dtype != torch.float32:
        raise TypeError("The released Silero VAD graph requires float32 audio.")
    if not torch.isfinite(input_values).all():
        raise ValueError("`input_values` cannot contain NaN or infinite values.")
    return input_values


@dataclass(frozen=True, slots=True)
class SileroVADState:
    """One stream's recurrent and waveform context.

    State is passed explicitly instead of being stored on the model.
    Multiple callers can therefore share one model without resetting or
    contaminating each other's streams.
    """

    hidden: Tensor
    cell: Tensor
    context: Tensor

    def __post_init__(self) -> None:
        for name in ("hidden", "cell", "context"):
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise TypeError(f"`{name}` must be a PyTorch tensor.")
            if value.ndim != 2:
                raise ValueError(f"`{name}` must have shape [batch, width].")
            if value.dtype != torch.float32:
                raise TypeError(f"`{name}` must use float32.")
        if self.hidden.shape != self.cell.shape:
            raise ValueError("Hidden and cell state shapes must match.")
        if self.hidden.shape[0] != self.context.shape[0]:
            raise ValueError("Recurrent state and context batch sizes must match.")
        if not (self.hidden.device == self.cell.device == self.context.device):
            raise ValueError("Every state tensor must be on the same device.")

    @property
    def batch_size(self) -> int:
        return self.hidden.shape[0]

    @property
    def recurrent(self) -> Tensor:
        """Return the upstream ``[2, batch, hidden]`` representation."""
        return torch.stack((self.hidden, self.cell), dim=0)

    def detached(self, *, clone: bool = False) -> SileroVADState:
        """Detach state at an inference or truncated-BPTT boundary."""

        def convert(value: Tensor) -> Tensor:
            result = value.detach()
            return result.clone() if clone else result

        return SileroVADState(
            hidden=convert(self.hidden),
            cell=convert(self.cell),
            context=convert(self.context),
        )


@dataclass(frozen=True, slots=True)
class SileroVADFrameOutput:
    """Speech probability and next state for one fixed-size frame."""

    probabilities: Tensor
    logits: Tensor
    state: SileroVADState


@dataclass(frozen=True, slots=True)
class SileroVADAudioOutput:
    """Frame-aligned output for an arbitrary-length waveform batch."""

    probabilities: Tensor
    logits: Tensor
    state: SileroVADState
    valid_samples: int


class SileroVADModel(nn.Module):
    """Native execution graph for the official 8 kHz or 16 kHz branch."""

    def __init__(
        self,
        config: SileroVADConfig | Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.config = SileroVADConfig.coerce(config or {})
        spectrum_bins = self.config.spectrum_bins

        self.stft_conv = nn.Conv1d(
            1,
            spectrum_bins * 2,
            kernel_size=self.config.filter_length,
            stride=self.config.hop_length,
            bias=False,
        )
        self.stft_conv.weight.requires_grad_(False)
        self.conv1 = nn.Conv1d(
            spectrum_bins,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            128,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv3 = nn.Conv1d(
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv4 = nn.Conv1d(
            64,
            self.config.recurrent_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.lstm_cell = nn.LSTMCell(
            self.config.recurrent_size,
            self.config.recurrent_size,
        )
        self.decoder_dropout = nn.Dropout(self.config.decoder_dropout)
        self.final_conv = nn.Conv1d(
            self.config.recurrent_size,
            1,
            kernel_size=1,
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> SileroVADState:
        """Create a fresh, independent state on the model device."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("`batch_size` must be an integer.")
        if batch_size < 1:
            raise ValueError("`batch_size` must be positive.")
        parameter = next(self.parameters())
        target_device = parameter.device if device is None else torch.device(device)
        hidden = torch.zeros(
            batch_size,
            self.config.recurrent_size,
            dtype=torch.float32,
            device=target_device,
        )
        return SileroVADState(
            hidden=hidden,
            cell=torch.zeros_like(hidden),
            context=torch.zeros(
                batch_size,
                self.config.context_size,
                dtype=torch.float32,
                device=target_device,
            ),
        )

    def _validated_state(
        self,
        state: SileroVADState | None,
        *,
        input_values: Tensor,
    ) -> SileroVADState:
        if state is None:
            return self.initial_state(
                input_values.shape[0],
                device=input_values.device,
            )
        if not isinstance(state, SileroVADState):
            raise TypeError("`state` must be a SileroVADState or None.")
        expected_recurrent = (
            input_values.shape[0],
            self.config.recurrent_size,
        )
        expected_context = (
            input_values.shape[0],
            self.config.context_size,
        )
        if tuple(state.hidden.shape) != expected_recurrent:
            raise ValueError(
                "Recurrent state must have shape "
                f"{expected_recurrent}; found {tuple(state.hidden.shape)}.")
        if tuple(state.context.shape) != expected_context:
            raise ValueError(
                f"Context must have shape {expected_context}; "
                f"found {tuple(state.context.shape)}.")
        if state.hidden.device != input_values.device:
            raise ValueError("State and audio must be on the same device.")
        return state

    def _analysis_transform(self, values: Tensor) -> Tensor:
        padded = functional.pad(
            values,
            (0, self.config.reflection_padding),
            mode="reflect",
        ).unsqueeze(1)
        transformed = self.stft_conv(padded)
        cutoff = self.config.spectrum_bins
        real = transformed[:, :cutoff].float()
        imaginary = transformed[:, cutoff:].float()
        return torch.sqrt(real.square() + imaginary.square())

    def forward_with_context(
        self,
        values: Tensor,
        recurrent_state: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Execute the released differentiable graph on context + frame."""
        expected_samples = self.config.context_size + self.config.frame_size
        if values.ndim != 2 or values.shape[1] != expected_samples:
            raise ValueError(
                "`values` must contain one context-prefixed frame with "
                f"{expected_samples} samples.")
        hidden, cell = recurrent_state
        features = self._analysis_transform(values)
        features = functional.relu(self.conv1(features))
        features = functional.relu(self.conv2(features))
        features = functional.relu(self.conv3(features))
        features = functional.relu(self.conv4(features))
        if features.shape[-1] != 1:
            raise RuntimeError("The fixed Silero convolution stack must produce one frame.")
        features = features.squeeze(-1)
        hidden, cell = self.lstm_cell(features, (hidden, cell))
        decoded = self.decoder_dropout(hidden.unsqueeze(-1)).float()
        decoded = functional.relu(decoded)
        logits = self.final_conv(decoded).squeeze(1).mean(dim=1, keepdim=True)
        probabilities = torch.sigmoid(logits)
        return probabilities, logits, (hidden, cell)

    def forward(
        self,
        input_values: Tensor,
        state: SileroVADState | None = None,
    ) -> SileroVADFrameOutput:
        """Score one fixed-size frame and return its stream-local next
        state."""
        input_values = _floating_audio(
            input_values,
            frame_size=self.config.frame_size,
        )
        self._validate_model_input(input_values)
        state = self._validated_state(state, input_values=input_values)
        combined = torch.cat((state.context, input_values), dim=1)
        probabilities, logits, (hidden, cell) = self.forward_with_context(
            combined,
            (state.hidden, state.cell),
        )
        next_state = SileroVADState(
            hidden=hidden,
            cell=cell,
            context=combined[:, -self.config.context_size:],
        )
        return SileroVADFrameOutput(
            probabilities=probabilities,
            logits=logits,
            state=next_state,
        )

    def _validate_model_input(self, input_values: Tensor) -> None:
        parameter = next(self.parameters())
        if parameter.device != input_values.device:
            raise ValueError("Model parameters and audio must be on the same device.")
        if parameter.dtype != torch.float32:
            raise TypeError(
                "The released Silero VAD graph and checkpoint require "
                "float32 model parameters.")

    def frame_probabilities(
        self,
        input_values: Tensor,
        *,
        state: SileroVADState | None = None,
        pad_final_frame: bool = True,
    ) -> SileroVADAudioOutput:
        """Score arbitrary-length audio without hiding recurrent state."""
        if not isinstance(pad_final_frame, bool):
            raise TypeError("`pad_final_frame` must be a boolean.")
        input_values = _floating_audio(input_values, frame_size=None)
        self._validate_model_input(input_values)
        current_state = self._validated_state(state, input_values=input_values)
        valid_samples = input_values.shape[1]
        remainder = valid_samples % self.config.frame_size
        if remainder:
            if not pad_final_frame:
                raise ValueError(
                    "Audio length must be divisible by the Silero frame size "
                    "when `pad_final_frame` is False.")
            input_values = functional.pad(
                input_values,
                (0, self.config.frame_size - remainder),
            )

        probabilities: list[Tensor] = []
        logits: list[Tensor] = []
        hidden, cell, context = current_state.hidden, current_state.cell, current_state.context
        for offset in range(0, input_values.shape[1], self.config.frame_size):
            # The complete waveform and initial state are validated above.
            # Repeating finite scans, parameter traversal, and state dataclass
            # construction for each 32 ms frame adds avoidable inference cost.
            combined = torch.cat((context, input_values[:, offset:offset + self.config.frame_size]), dim=1)
            probability, logit, (hidden, cell) = self.forward_with_context(
                combined,
                (hidden, cell),
            )
            probabilities.append(probability)
            logits.append(logit)
            context = combined[:, -self.config.context_size:]
        current_state = SileroVADState(hidden=hidden, cell=cell, context=context)
        return SileroVADAudioOutput(
            probabilities=torch.cat(probabilities, dim=1),
            logits=torch.cat(logits, dim=1),
            state=current_state,
            valid_samples=valid_samples,
        )

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        """Yield the parameters used by the official supervised tuning
        recipe."""
        yield from self.lstm_cell.parameters()
        yield from self.final_conv.parameters()

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the convolutional encoder.

        The Fourier basis remains fixed, matching its buffer semantics
        in the official PyTorch checkpoint.
        """
        if not isinstance(trainable, bool):
            raise TypeError("`trainable` must be a boolean.")
        for module in (self.conv1, self.conv2, self.conv3, self.conv4):
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)


class SileroVADStream:
    """An inference session that owns exactly one model state."""

    def __init__(self, model: SileroVADModel) -> None:
        if not isinstance(model, SileroVADModel):
            raise TypeError("`model` must be a SileroVADModel.")
        self.model = model
        self._state: SileroVADState | None = None
        self._lock = RLock()

    @property
    def initialized(self) -> bool:
        with self._lock:
            return self._state is not None

    @property
    def state(self) -> SileroVADState | None:
        """Return a defensive snapshot of this stream's state."""
        with self._lock:
            if self._state is None:
                return None
            return self._state.detached(clone=True)

    def reset(self) -> None:
        """Discard recurrent and waveform context for this stream only."""
        with self._lock:
            self._state = None

    def process(self, input_values: Tensor) -> Tensor:
        """Return frame probabilities and advance this stream atomically."""
        with self._lock, torch.no_grad():
            output = self.model(input_values, state=self._state)
            self._state = output.state.detached()
            return output.probabilities


__all__ = [
    "SileroVADAudioOutput",
    "SileroVADFrameOutput",
    "SileroVADModel",
    "SileroVADState",
    "SileroVADStream",
]
