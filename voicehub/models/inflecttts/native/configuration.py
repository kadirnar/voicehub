"""Validated configuration for Inflect Micro/Nano v2.

The schema mirrors the immutable ``config.json`` files published with
the Inflect v2 releases.  It intentionally keeps the release's
``inference_only`` flag separate from VoiceHub's training warm-start
mode: the public checkpoint does not contain the posterior encoder or
discriminators.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be positive.")
    return value


def _probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    normalized = float(value)
    if not 0.0 <= normalized < 1.0:
        raise ValueError(f"`{name}` must be in [0, 1).")
    return normalized


def _integer_tuple(name: str, value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence of integers.")
    result = tuple(_positive_integer(name, item) for item in value)
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    return result


def _nested_integer_tuple(
    name: str,
    value: object,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a nested sequence of integers.")
    result = tuple(_integer_tuple(f"{name}[{index}]", item) for index, item in enumerate(value))
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    return result


@dataclass(frozen=True, slots=True)
class InflectV2Config:
    """Executable Inflect v2 graph configuration."""

    format: str = "inflect_v2_inference_config_v1"
    vocabulary_size: int = 178
    segment_size: int = 16_384
    sample_rate: int = 24_000
    filter_length: int = 1_024
    hop_length: int = 256
    win_length: int = 1_024
    mel_channels: int = 80
    mel_min_frequency: float = 0.0
    mel_max_frequency: float = 12_000.0
    add_blank: bool = True
    inter_channels: int = 192
    hidden_channels: int = 96
    filter_channels: int = 768
    attention_heads: int = 2
    attention_layers: int = 3
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
    upsample_initial_channel: int = 320
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 4, 4)
    posterior_layers_declared: int = 3
    use_spectral_norm: bool = False
    use_stochastic_duration_predictor: bool = False
    inference_only: bool = True
    decoder_alias_free: bool = False
    decoder_alias_free_start_stage: int = 2
    decoder_snake_logscale: bool = True

    def __post_init__(self) -> None:
        if self.format != "inflect_v2_inference_config_v1":
            raise ValueError(f"Unsupported Inflect config format {self.format!r}.")
        for name in (
                "vocabulary_size",
                "segment_size",
                "sample_rate",
                "filter_length",
                "hop_length",
                "win_length",
                "mel_channels",
                "inter_channels",
                "hidden_channels",
                "filter_channels",
                "attention_heads",
                "attention_layers",
                "kernel_size",
                "upsample_initial_channel",
                "posterior_layers_declared",
        ):
            _positive_integer(name, getattr(self, name))
        if self.hidden_channels % self.attention_heads:
            raise ValueError("`hidden_channels` must be divisible by `attention_heads`.")
        if self.filter_length % 2:
            raise ValueError("`filter_length` must be even.")
        if self.segment_size % self.hop_length:
            raise ValueError("`segment_size` must be divisible by `hop_length`.")
        if self.mel_min_frequency < 0:
            raise ValueError("`mel_min_frequency` cannot be negative.")
        if self.mel_max_frequency <= self.mel_min_frequency:
            raise ValueError("`mel_max_frequency` must exceed `mel_min_frequency`.")
        object.__setattr__(self, "dropout", _probability("dropout", self.dropout))
        for name in (
                "add_blank",
                "use_spectral_norm",
                "use_stochastic_duration_predictor",
                "inference_only",
                "decoder_alias_free",
                "decoder_snake_logscale",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.resblock not in {"1", "2"}:
            raise ValueError("`resblock` must be '1' or '2'.")
        for name in (
                "resblock_kernel_sizes",
                "upsample_rates",
                "upsample_kernel_sizes",
        ):
            object.__setattr__(
                self,
                name,
                _integer_tuple(name, getattr(self, name)),
            )
        object.__setattr__(
            self,
            "resblock_dilation_sizes",
            _nested_integer_tuple(
                "resblock_dilation_sizes",
                self.resblock_dilation_sizes,
            ),
        )
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilation_sizes):
            raise ValueError("Each residual kernel requires one dilation sequence.")
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("Each upsample rate requires one upsample kernel.")
        for rate, kernel in zip(
                self.upsample_rates,
                self.upsample_kernel_sizes,
        ):
            if kernel < rate or (kernel - rate) % 2:
                raise ValueError(
                    "Inflect upsample kernels must be no smaller than their "
                    "rates and have even kernel-rate differences.")
        if (isinstance(self.decoder_alias_free_start_stage, bool) or
                not isinstance(self.decoder_alias_free_start_stage, int) or
                self.decoder_alias_free_start_stage < 0):
            raise ValueError("`decoder_alias_free_start_stage` must be a non-negative "
                             "integer.")

    @property
    def spectrogram_channels(self) -> int:
        return self.filter_length // 2 + 1

    @property
    def segment_frames(self) -> int:
        return self.segment_size // self.hop_length

    def for_training(self) -> InflectV2Config:
        """Return the same graph with its fresh posterior encoder enabled."""
        return replace(self, inference_only=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InflectV2Config:
        if not isinstance(payload, Mapping):
            raise TypeError("Inflect configuration must be a mapping.")
        unknown = set(payload) - {"format", "train", "data", "model"}
        if unknown:
            raise ValueError("Unknown top-level Inflect config fields: " + ", ".join(sorted(unknown)))
        train = payload.get("train")
        data = payload.get("data")
        model = payload.get("model")
        if not all(isinstance(item, Mapping) for item in (train, data, model)):
            raise ValueError("Inflect config requires mapping-valued train, data, and "
                             "model sections.")
        allowed_train = {"segment_size"}
        allowed_data = {
            "text_cleaners",
            "max_wav_value",
            "sampling_rate",
            "filter_length",
            "hop_length",
            "win_length",
            "n_mel_channels",
            "mel_fmin",
            "mel_fmax",
            "add_blank",
            "n_speakers",
            "cleaned_text",
        }
        allowed_model = {
            "inter_channels",
            "hidden_channels",
            "filter_channels",
            "n_heads",
            "n_layers",
            "kernel_size",
            "p_dropout",
            "resblock",
            "resblock_kernel_sizes",
            "resblock_dilation_sizes",
            "upsample_rates",
            "upsample_initial_channel",
            "upsample_kernel_sizes",
            "n_layers_q",
            "use_spectral_norm",
            "use_sdp",
            "inference_only",
            "decoder_alias_free",
            "decoder_alias_free_start_stage",
            "decoder_snake_logscale",
        }
        for section, allowed, name in (
            (train, allowed_train, "train"),
            (data, allowed_data, "data"),
            (model, allowed_model, "model"),
        ):
            section_unknown = set(section) - allowed
            if section_unknown:
                raise ValueError(f"Unknown Inflect {name} fields: " + ", ".join(sorted(section_unknown)))
        if data.get("text_cleaners", []) != []:
            raise ValueError("Inflect v2 expects pre-cleaned phoneme input.")
        if data.get("n_speakers", 0) != 0:
            raise ValueError("Published Inflect v2 is a fixed single voice.")
        if data.get("cleaned_text", True) is not True:
            raise ValueError("Inflect v2 requires cleaned text tokens.")
        if float(data.get("max_wav_value", 32768.0)) != 32768.0:
            raise ValueError("Unsupported Inflect `max_wav_value`.")
        return cls(
            format=payload.get("format", "inflect_v2_inference_config_v1"),
            segment_size=train["segment_size"],
            sample_rate=data["sampling_rate"],
            filter_length=data["filter_length"],
            hop_length=data["hop_length"],
            win_length=data["win_length"],
            mel_channels=data["n_mel_channels"],
            mel_min_frequency=float(data["mel_fmin"]),
            mel_max_frequency=float(data["mel_fmax"]),
            add_blank=data["add_blank"],
            inter_channels=model["inter_channels"],
            hidden_channels=model["hidden_channels"],
            filter_channels=model["filter_channels"],
            attention_heads=model["n_heads"],
            attention_layers=model["n_layers"],
            kernel_size=model["kernel_size"],
            dropout=model["p_dropout"],
            resblock=model["resblock"],
            resblock_kernel_sizes=tuple(model["resblock_kernel_sizes"]),
            resblock_dilation_sizes=tuple(tuple(item) for item in model["resblock_dilation_sizes"]),
            upsample_rates=tuple(model["upsample_rates"]),
            upsample_initial_channel=model["upsample_initial_channel"],
            upsample_kernel_sizes=tuple(model["upsample_kernel_sizes"]),
            posterior_layers_declared=model.get("n_layers_q", 3),
            use_spectral_norm=model.get("use_spectral_norm", False),
            use_stochastic_duration_predictor=model.get("use_sdp", False),
            inference_only=model.get("inference_only", True),
            decoder_alias_free=model.get("decoder_alias_free", False),
            decoder_alias_free_start_stage=model.get(
                "decoder_alias_free_start_stage",
                2,
            ),
            decoder_snake_logscale=model.get("decoder_snake_logscale", True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "train": {
                "segment_size": self.segment_size
            },
            "data": {
                "text_cleaners": [],
                "max_wav_value": 32768.0,
                "sampling_rate": self.sample_rate,
                "filter_length": self.filter_length,
                "hop_length": self.hop_length,
                "win_length": self.win_length,
                "n_mel_channels": self.mel_channels,
                "mel_fmin": self.mel_min_frequency,
                "mel_fmax": self.mel_max_frequency,
                "add_blank": self.add_blank,
                "n_speakers": 0,
                "cleaned_text": True,
            },
            "model": {
                "inter_channels":
                self.inter_channels,
                "hidden_channels":
                self.hidden_channels,
                "filter_channels":
                self.filter_channels,
                "n_heads":
                self.attention_heads,
                "n_layers":
                self.attention_layers,
                "kernel_size":
                self.kernel_size,
                "p_dropout":
                self.dropout,
                "resblock":
                self.resblock,
                "resblock_kernel_sizes":
                list(self.resblock_kernel_sizes),
                "resblock_dilation_sizes": [list(item) for item in self.resblock_dilation_sizes],
                "upsample_rates":
                list(self.upsample_rates),
                "upsample_initial_channel":
                self.upsample_initial_channel,
                "upsample_kernel_sizes":
                list(self.upsample_kernel_sizes),
                "n_layers_q":
                self.posterior_layers_declared,
                "use_spectral_norm":
                self.use_spectral_norm,
                "use_sdp":
                self.use_stochastic_duration_predictor,
                "inference_only":
                self.inference_only,
                **({
                    "decoder_alias_free": True,
                    "decoder_alias_free_start_stage": self.decoder_alias_free_start_stage,
                    "decoder_snake_logscale": self.decoder_snake_logscale,
                } if self.decoder_alias_free else {}),
            },
        }

    def model_kwargs(self) -> dict[str, Any]:
        """Return the keyword layout expected by ``SynthesizerTrn``."""
        return {
            "inter_channels": self.inter_channels,
            "hidden_channels": self.hidden_channels,
            "filter_channels": self.filter_channels,
            "n_heads": self.attention_heads,
            "n_layers": self.attention_layers,
            "kernel_size": self.kernel_size,
            "p_dropout": self.dropout,
            "resblock": self.resblock,
            "resblock_kernel_sizes": copy.deepcopy(list(self.resblock_kernel_sizes)),
            "resblock_dilation_sizes": [list(item) for item in self.resblock_dilation_sizes],
            "upsample_rates": list(self.upsample_rates),
            "upsample_initial_channel": self.upsample_initial_channel,
            "upsample_kernel_sizes": list(self.upsample_kernel_sizes),
            "n_speakers": 0,
            "use_sdp": self.use_stochastic_duration_predictor,
            "inference_only": self.inference_only,
            "decoder_alias_free": self.decoder_alias_free,
            "decoder_alias_free_start_stage": self.decoder_alias_free_start_stage,
            "decoder_snake_logscale": self.decoder_snake_logscale,
        }


INFLECT_MICRO_V2_CONFIG = InflectV2Config()
INFLECT_NANO_V2_CONFIG = InflectV2Config(
    inter_channels=128,
    hidden_channels=72,
    filter_channels=384,
    upsample_initial_channel=192,
    posterior_layers_declared=2,
)

__all__ = [
    "INFLECT_MICRO_V2_CONFIG",
    "INFLECT_NANO_V2_CONFIG",
    "InflectV2Config",
]
