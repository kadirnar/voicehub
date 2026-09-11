"""Sherpa-compatible TEN VAD acoustic frontend in native PyTorch."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.vad_ten.native.configuration import TENVADConfig

_FEATURE_MEAN = (
    -8.198236465454102,
    -6.265716552734375,
    -5.483818531036377,
    -4.758691310882568,
    -4.417088985443115,
    -4.142892837524414,
    -3.9128503799438477,
    -3.8459279537200928,
    -3.657090425491333,
    -3.723418712615967,
    -3.876134157180786,
    -3.843890905380249,
    -3.6904051303863525,
    -3.756065845489502,
    -3.6986961364746094,
    -3.650463104248047,
    -3.7004687786102295,
    -3.567321300506592,
    -3.4989001750946045,
    -3.47780704498291,
    -3.4588160514831543,
    -3.4449238777160645,
    -3.4013285636901855,
    -3.3062613010406494,
    -3.2785568237304688,
    -3.233250856399536,
    -3.1986160278320312,
    -3.204526424407959,
    -3.20879864692688,
    -3.257838010787964,
    -3.3813767433166504,
    -3.5340213775634766,
    -3.6408679485321045,
    -3.7268588542938232,
    -3.773730993270874,
    -3.8046672344207764,
    -3.8329010009765625,
    -3.8711204528808594,
    -3.9905929565429688,
    -4.480289459228516,
    92.35690307617188,
)
_FEATURE_INV_STDDEV = (
    0.19357097148895264,
    0.20091579854488373,
    0.21281595528125763,
    0.2159537374973297,
    0.2157800942659378,
    0.2154635488986969,
    0.2154858261346817,
    0.21429947018623352,
    0.2150290459394455,
    0.21551626920700073,
    0.21563807129859924,
    0.21644558012485504,
    0.21756553649902344,
    0.21917064487934113,
    0.21956980228424072,
    0.21896639466285706,
    0.21917855739593506,
    0.21918226778507233,
    0.2180882692337036,
    0.21738281846046448,
    0.21772992610931396,
    0.21805863082408905,
    0.2181740552186966,
    0.21616514027118683,
    0.2161247283220291,
    0.21615596115589142,
    0.21565639972686768,
    0.21353760361671448,
    0.211558997631073,
    0.21122492849826813,
    0.2103833705186844,
    0.2061973512172699,
    0.20536264777183533,
    0.20472995936870575,
    0.20319722592830658,
    0.2016449272632599,
    0.20013532042503357,
    0.19822297990322113,
    0.19715245068073273,
    0.19621542096138,
    0.008679524064064026,
)


def _mel_scale_slaney(frequency: float) -> float:
    if frequency <= 1_000.0:
        return frequency * 3.0 / 200.0
    return 15.0 + 14.545078505785561 * math.log(frequency / 1_000.0)


def _inverse_mel_scale_slaney(mel: float) -> float:
    if mel <= 15.0:
        return 200.0 / 3.0 * mel
    return 1_000.0 * math.exp((mel - 15.0) * 0.06875177742094911)


def sherpa_ten_vad_mel_filterbank(
    *,
    sampling_rate: int = 16_000,
    fft_size: int = 1_024,
    mel_bins: int = 40,
) -> Tensor:
    """Reproduce kaldi-native-fbank 1.22.3's reviewed Librosa path."""
    if sampling_rate != 16_000 or fft_size != 1_024 or mel_bins != 40:
        raise ValueError("The released TEN frontend uses 16 kHz, FFT-1024, and 40 mel bins.")
    low_mel = _mel_scale_slaney(0.0)
    high_mel = _mel_scale_slaney(sampling_rate / 2)
    delta = (high_mel - low_mel) / (mel_bins + 1)
    filters = torch.zeros(mel_bins, fft_size // 2 + 1, dtype=torch.float32)
    scale = (fft_size + 1.0) / sampling_rate
    for index in range(mel_bins):
        left = int(_inverse_mel_scale_slaney(low_mel + index * delta) * scale)
        center = int(_inverse_mel_scale_slaney(low_mel + (index + 1) * delta) * scale)
        right = int(_inverse_mel_scale_slaney(low_mel + (index + 2) * delta) * scale)
        if not left < center < right:
            raise RuntimeError("TEN mel construction produced an empty filter.")
        for bin_index in range(left + 1, center + 1):
            filters[index, bin_index] = ((bin_index - left) / (center - left))
        for bin_index in range(center + 1, right):
            filters[index, bin_index] = ((right - bin_index) / (right - center))
    return filters


def sherpa_ten_vad_window() -> Tensor:
    """Return the exact symmetric 768-sample metadata window."""
    return torch.tensor(
        [0.5 - 0.5 * math.cos(2.0 * math.pi * index / 767.0) for index in range(768)],
        dtype=torch.float32,
    )


@dataclass(frozen=True, slots=True)
class TENVADFrontendState:
    """Per-stream pre-emphasis and two-frame feature history."""

    previous_sample: Tensor
    history: Tensor

    def detached(self, *, clone: bool = False) -> TENVADFrontendState:

        def convert(value: Tensor) -> Tensor:
            result = value.detach()
            return result.clone() if clone else result

        return TENVADFrontendState(
            previous_sample=convert(self.previous_sample),
            history=convert(self.history),
        )


@dataclass(frozen=True, slots=True)
class TENVADFrontendOutput:
    features: Tensor
    context: Tensor
    state: TENVADFrontendState


class TENVADFrontend(nn.Module):
    """Scale, pre-emphasize, FFT, log-mel, normalize, and stack context."""

    def __init__(self, config: TENVADConfig) -> None:
        super().__init__()
        self.config = TENVADConfig.coerce(config)
        self.register_buffer(
            "mean",
            torch.tensor(_FEATURE_MEAN, dtype=torch.float32),
        )
        self.register_buffer(
            "inv_stddev",
            torch.tensor(_FEATURE_INV_STDDEV, dtype=torch.float32),
        )
        self.register_buffer("window", sherpa_ten_vad_window())
        self.register_buffer(
            "mel_filterbank",
            sherpa_ten_vad_mel_filterbank(),
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> TENVADFrontendState:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("`batch_size` must be an integer.")
        if batch_size < 1:
            raise ValueError("`batch_size` must be positive.")
        target = self.mean.device if device is None else torch.device(device)
        return TENVADFrontendState(
            previous_sample=torch.zeros(
                batch_size,
                dtype=torch.float32,
                device=target,
            ),
            history=torch.zeros(
                batch_size,
                self.config.context_frames - 1,
                self.config.feature_size,
                dtype=torch.float32,
                device=target,
            ),
        )

    def _validate_state(
        self,
        state: TENVADFrontendState | None,
        frame: Tensor,
    ) -> TENVADFrontendState:
        if state is None:
            return self.initial_state(frame.shape[0], device=frame.device)
        if not isinstance(state, TENVADFrontendState):
            raise TypeError("`state` must be a TENVADFrontendState or None.")
        if tuple(state.previous_sample.shape) != (frame.shape[0], ):
            raise ValueError("TEN frontend previous-sample state has the wrong shape.")
        expected = (
            frame.shape[0],
            self.config.context_frames - 1,
            self.config.feature_size,
        )
        if tuple(state.history.shape) != expected:
            raise ValueError(f"TEN frontend history must have shape {expected}.")
        if (state.previous_sample.device != frame.device or state.history.device != frame.device):
            raise ValueError("TEN frontend state and audio must use the same device.")
        return state

    def forward(
        self,
        frame: Tensor,
        state: TENVADFrontendState | None = None,
    ) -> TENVADFrontendOutput:
        if not isinstance(frame, Tensor):
            raise TypeError("`frame` must be a PyTorch tensor.")
        if frame.ndim == 1:
            frame = frame.unsqueeze(0)
        if frame.ndim != 2:
            raise ValueError("`frame` must have shape [batch, samples].")
        if frame.shape[1] < 1 or frame.shape[1] > self.config.analysis_window_size:
            raise ValueError("TEN frames must contain between 1 and 768 samples.")
        if frame.dtype != torch.float32:
            raise TypeError("TEN VAD requires float32 audio.")
        if not torch.isfinite(frame).all():
            raise ValueError("TEN VAD audio cannot contain NaN or infinite values.")
        if frame.device != self.mean.device:
            raise ValueError("TEN frontend buffers and audio must use the same device.")
        state = self._validate_state(state, frame)

        scaled = frame * self.config.input_scale
        first = scaled[:, :1] - self.config.preemphasis * state.previous_sample[:, None]
        remainder = (
            scaled[:, 1:] -
            self.config.preemphasis * scaled[:, :-1] if scaled.shape[1] > 1 else scaled[:, :0])
        emphasized = torch.cat((first, remainder), dim=1)
        windowed = emphasized * self.window[:frame.shape[1]]
        padded = functional.pad(
            windowed,
            (0, self.config.fft_size - frame.shape[1]),
        )
        spectrum = torch.fft.rfft(padded, n=self.config.fft_size)
        power = spectrum.real.square() + spectrum.imag.square()
        mel = functional.linear(power, self.mel_filterbank)
        log_mel = torch.log(mel + self.config.log_floor) - math.log(
            self.config.input_scale * self.config.input_scale)
        pitch = torch.zeros(
            frame.shape[0],
            1,
            dtype=torch.float32,
            device=frame.device,
        )
        features = torch.cat((log_mel, pitch), dim=1)
        features = (features - self.mean) * self.inv_stddev
        context = torch.cat((state.history, features.unsqueeze(1)), dim=1)
        next_state = TENVADFrontendState(
            previous_sample=scaled[:, -1],
            history=context[:, 1:],
        )
        return TENVADFrontendOutput(
            features=features,
            context=context,
            state=next_state,
        )


__all__ = [
    "TENVADFrontend",
    "TENVADFrontendOutput",
    "TENVADFrontendState",
    "sherpa_ten_vad_mel_filterbank",
    "sherpa_ten_vad_window",
]
