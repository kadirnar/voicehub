"""Typed configuration for the VoiceHub-native OpenVoice V2 converter."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _positive_tuple(name: str, value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence.")
    result = tuple(value)
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    for item in result:
        _positive_integer(name, item)
    return result


def _nested_positive_tuple(
    name: str,
    value: Any,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a nested sequence.")
    result = tuple(_positive_tuple(f"{name}[{index}]", item) for index, item in enumerate(value))
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    return result


@dataclass(frozen=True, slots=True)
class OpenVoiceConverterConfig:
    """Complete released OpenVoice V2 tone-color converter topology."""

    version: str = "v2"
    sample_rate: int = 22_050
    n_fft: int = 1_024
    hop_length: int = 256
    win_length: int = 1_024
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    upsample_rates: tuple[int, ...] = (8, 8, 2, 2)
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 4, 4)
    speaker_embedding_size: int = 256
    zero_generator_conditioning: bool = True

    def __post_init__(self) -> None:
        if self.version != "v2":
            raise ValueError("The native OpenVoice converter supports V2 only.")
        for name in (
                "sample_rate",
                "n_fft",
                "hop_length",
                "win_length",
                "inter_channels",
                "hidden_channels",
                "filter_channels",
                "n_heads",
                "n_layers",
                "kernel_size",
                "upsample_initial_channel",
                "speaker_embedding_size",
        ):
            _positive_integer(name, getattr(self, name))
        if not self.hop_length <= self.win_length <= self.n_fft:
            raise ValueError("OpenVoice requires hop_length <= win_length <= n_fft.")
        if self.hidden_channels % self.n_heads:
            raise ValueError("`hidden_channels` must be divisible by `n_heads`.")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or
                not math.isfinite(float(self.dropout)) or not 0.0 <= float(self.dropout) < 1.0):
            raise ValueError("`dropout` must be finite and in [0, 1).")
        if self.resblock not in {"1", "2"}:
            raise ValueError("`resblock` must be '1' or '2'.")
        kernels = _positive_tuple(
            "resblock_kernel_sizes",
            self.resblock_kernel_sizes,
        )
        dilations = _nested_positive_tuple(
            "resblock_dilation_sizes",
            self.resblock_dilation_sizes,
        )
        rates = _positive_tuple("upsample_rates", self.upsample_rates)
        upsample_kernels = _positive_tuple(
            "upsample_kernel_sizes",
            self.upsample_kernel_sizes,
        )
        if len(kernels) != len(dilations):
            raise ValueError("Residual kernels and dilation groups must align.")
        expected_dilations = 3 if self.resblock == "1" else 2
        if any(len(group) != expected_dilations for group in dilations):
            raise ValueError(
                f"OpenVoice resblock {self.resblock} requires "
                f"{expected_dilations} dilations per kernel.")
        if len(rates) != len(upsample_kernels):
            raise ValueError("Upsample rates and kernels must align.")
        if math.prod(rates) != self.hop_length:
            raise ValueError("OpenVoice upsample rates must multiply to `hop_length`.")
        if any(kernel < rate or (kernel - rate) % 2 for rate, kernel in zip(rates, upsample_kernels)):
            raise ValueError("Upsample kernels must cover their rates with matching parity.")
        if not isinstance(self.zero_generator_conditioning, bool):
            raise TypeError("`zero_generator_conditioning` must be a boolean.")
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "resblock_kernel_sizes", kernels)
        object.__setattr__(self, "resblock_dilation_sizes", dilations)
        object.__setattr__(self, "upsample_rates", rates)
        object.__setattr__(self, "upsample_kernel_sizes", upsample_kernels)

    @property
    def spectrogram_channels(self) -> int:
        return self.n_fft // 2 + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> OpenVoiceConverterConfig:
        """Parse the official nested configuration or native flat schema."""
        if not isinstance(values, Mapping):
            raise TypeError("OpenVoice configuration must be a mapping.")
        document = dict(values)
        if "data" in document or "model" in document:
            data = document.get("data")
            model = document.get("model")
            if not isinstance(data, Mapping) or not isinstance(model, Mapping):
                raise ValueError("Official OpenVoice config requires `data` and `model` objects.")
            if int(data.get("n_speakers", -1)) != 0:
                raise ValueError("The V2 converter must use its reference encoder "
                                 "(`n_speakers=0`).")
            document = {
                "version": values.get("_version_", "v2"),
                "sample_rate": data.get("sampling_rate", 22_050),
                "n_fft": data.get("filter_length", 1_024),
                "hop_length": data.get("hop_length", 256),
                "win_length": data.get("win_length", 1_024),
                "inter_channels": model.get("inter_channels", 192),
                "hidden_channels": model.get("hidden_channels", 192),
                "filter_channels": model.get("filter_channels", 768),
                "n_heads": model.get("n_heads", 2),
                "n_layers": model.get("n_layers", 6),
                "kernel_size": model.get("kernel_size", 3),
                "dropout": model.get("p_dropout", 0.1),
                "resblock": model.get("resblock", "1"),
                "resblock_kernel_sizes": model.get(
                    "resblock_kernel_sizes",
                    (3, 7, 11),
                ),
                "resblock_dilation_sizes": model.get(
                    "resblock_dilation_sizes",
                    ((1, 3, 5), ) * 3,
                ),
                "upsample_rates": model.get("upsample_rates", (8, 8, 2, 2)),
                "upsample_initial_channel": model.get(
                    "upsample_initial_channel",
                    512,
                ),
                "upsample_kernel_sizes": model.get(
                    "upsample_kernel_sizes",
                    (16, 16, 4, 4),
                ),
                "speaker_embedding_size": model.get("gin_channels", 256),
                "zero_generator_conditioning": model.get("zero_g", True),
            }
        return cls(**document)


__all__ = ["OpenVoiceConverterConfig"]
