"""Validated configuration for VoiceHub's native SpeechBrain CRDNN VAD."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < 1:
        raise ValueError(f"`{name}` must be positive.")
    return value


@dataclass(frozen=True, slots=True)
class SpeechBrainCRDNNVADConfig:
    """Complete graph description for ``vad-crdnn-libriparty``.

    Defaults reproduce the published immutable artifact.  Configuration
    is explicit so future CRDNN checkpoints can use the same
    implementation without accepting an ambiguous tensor namespace.
    """

    sampling_rate: int = 16_000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 40
    f_min: float = 0.0
    f_max: float = 8_000.0
    top_db: float = 80.0
    cnn_channels: tuple[int, int] = (16, 32)
    cnn_kernel_size: tuple[int, int] = (3, 3)
    cnn_pool_size: int = 2
    rnn_hidden_size: int = 32
    rnn_num_layers: int = 2
    rnn_bidirectional: bool = True
    dnn_hidden_size: int = 16
    dnn_num_layers: int = 2
    dropout: float = 0.15
    leaky_relu_slope: float = 0.01
    normalization_epsilon: float = 1e-10
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "sampling_rate",
                "n_fft",
                "win_length",
                "hop_length",
                "n_mels",
                "cnn_pool_size",
                "rnn_hidden_size",
                "rnn_num_layers",
                "dnn_hidden_size",
                "dnn_num_layers",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(name, getattr(self, name)),
            )
        if self.sampling_rate != 16_000:
            raise ValueError("The published SpeechBrain VAD checkpoint requires 16 kHz audio.")
        if self.n_fft != 400 or self.win_length != 400 or self.hop_length != 160:
            raise ValueError(
                "The published SpeechBrain VAD checkpoint requires a "
                "400-sample FFT/window and 160-sample hop.")
        if self.n_mels != 40:
            raise ValueError("The published SpeechBrain VAD checkpoint requires 40 mel bins.")
        if self.win_length > self.n_fft:
            raise ValueError("`win_length` cannot exceed `n_fft`.")
        for name in ("cnn_channels", "cnn_kernel_size"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError(f"`{name}` must be a two-item tuple.")
            object.__setattr__(
                self,
                name,
                tuple(_positive_integer(f"{name} item", item) for item in value),
            )
        if any(kernel % 2 == 0 for kernel in self.cnn_kernel_size):
            raise ValueError("SpeechBrain same-padding requires odd CNN kernels.")
        if self.cnn_channels != (16, 32):
            raise ValueError("The published checkpoint requires CNN channels (16, 32).")
        if self.cnn_kernel_size != (3, 3) or self.cnn_pool_size != 2:
            raise ValueError("The published checkpoint requires 3x3 CNNs and 2x frequency pooling.")
        if (self.rnn_hidden_size != 32 or self.rnn_num_layers != 2 or self.rnn_bidirectional is not True):
            raise ValueError("The published checkpoint requires a two-layer bidirectional 32-unit GRU.")
        if self.dnn_hidden_size != 16 or self.dnn_num_layers != 2:
            raise ValueError("The published checkpoint requires two 16-unit DNN blocks.")
        for name in (
                "f_min",
                "f_max",
                "top_db",
                "dropout",
                "leaky_relu_slope",
                "normalization_epsilon",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a real number.")
            object.__setattr__(self, name, float(value))
        if not 0.0 <= self.f_min < self.f_max <= self.sampling_rate / 2:
            raise ValueError("Mel frequencies must satisfy 0 <= f_min < f_max <= Nyquist.")
        if self.top_db <= 0.0:
            raise ValueError("`top_db` must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("`dropout` must be in [0, 1).")
        if self.leaky_relu_slope < 0.0 or self.normalization_epsilon <= 0.0:
            raise ValueError("Activation slope must be non-negative and epsilon positive.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @property
    def time_resolution(self) -> float:
        return self.hop_length / self.sampling_rate

    @property
    def rnn_input_size(self) -> int:
        frequency = self.n_mels
        for _ in self.cnn_channels:
            frequency //= self.cnn_pool_size
        return frequency * self.cnn_channels[-1]

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> SpeechBrainCRDNNVADConfig:
        if not isinstance(values, Mapping):
            raise TypeError("SpeechBrain CRDNN VAD configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        for name in ("cnn_channels", "cnn_kernel_size"):
            if name in resolved:
                resolved[name] = tuple(resolved[name])
        extras = {name: value for name, value in source.items() if name not in canonical | {"extra_config"}}
        supplied = source.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        return cls(**resolved, extra_config=extras)

    @classmethod
    def coerce(
        cls,
        value: SpeechBrainCRDNNVADConfig | Mapping[str, Any],
    ) -> SpeechBrainCRDNNVADConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            result[item.name] = list(value) if isinstance(value, tuple) else value
        result.setdefault("model_type", "vad_speechbrain")
        result.setdefault("architecture", "speechbrain-crdnn-vad")
        result.setdefault('architectures', ["SpeechBrainCRDNNVADModel"])
        return result


__all__ = ["SpeechBrainCRDNNVADConfig"]
