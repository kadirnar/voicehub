"""Typed, dependency-free configuration for the MeloTTS VITS graph."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _nonnegative_number(name: str, value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or
            float(value) < 0.0):
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return float(value)


def _probability(name: str, value: Any) -> float:
    result = _nonnegative_number(name, value)
    if result >= 1.0:
        raise ValueError(f"`{name}` must be in [0, 1).")
    return result


def _integer_tuple(name: str, value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence of integers.")
    result = tuple(value)
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    for item in result:
        _positive_integer(name, item)
    return result


def _nested_integer_tuple(
    name: str,
    value: Any,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a nested integer sequence.")
    result = tuple(_integer_tuple(f"{name}[{index}]", item) for index, item in enumerate(value))
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    return result


def _speaker_items(value: Any) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        items = tuple(value.items())
    elif not isinstance(value, (str, bytes)) and isinstance(value, Sequence):
        items = tuple(value)
    else:
        raise TypeError("MeloTTS speakers must be a mapping or pair sequence.")
    result: list[tuple[str, int]] = []
    seen_names: set[str] = set()
    seen_ids: set[int] = set()
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("MeloTTS speaker entries must be name/ID pairs.")
        raw_name, raw_id = item
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("MeloTTS speaker names must be non-empty strings.")
        name = raw_name.strip()
        if name in seen_names:
            raise ValueError("MeloTTS speaker names must be unique.")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id < 0:
            raise ValueError("MeloTTS speaker IDs must be non-negative integers.")
        if raw_id in seen_ids:
            raise ValueError("MeloTTS speaker IDs must be unique.")
        result.append((name, raw_id))
        seen_names.add(name)
        seen_ids.add(raw_id)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class MeloTTSDataConfig:
    sample_rate: int = 44_100
    n_fft: int = 2_048
    hop_length: int = 512
    win_length: int = 2_048
    n_mels: int = 128
    mel_fmin: float = 0.0
    mel_fmax: float | None = None
    n_speakers: int = 256
    speaker_ids: tuple[tuple[str, int], ...] = ()
    add_blank: bool = True

    def __post_init__(self) -> None:
        for name in (
                "sample_rate",
                "n_fft",
                "hop_length",
                "win_length",
                "n_mels",
                "n_speakers",
        ):
            _positive_integer(name, getattr(self, name))
        if self.hop_length > self.win_length or self.win_length > self.n_fft:
            raise ValueError("MeloTTS audio dimensions must satisfy "
                             "hop_length <= win_length <= n_fft.")
        _nonnegative_number("mel_fmin", self.mel_fmin)
        if self.mel_fmax is not None:
            _nonnegative_number("mel_fmax", self.mel_fmax)
        maximum = self.sample_rate / 2 if self.mel_fmax is None else self.mel_fmax
        if not self.mel_fmin < maximum <= self.sample_rate / 2:
            raise ValueError("MeloTTS mel limits must lie below Nyquist.")
        if not isinstance(self.add_blank, bool):
            raise TypeError("`add_blank` must be a boolean.")
        normalized = _speaker_items(self.speaker_ids)
        object.__setattr__(self, "speaker_ids", normalized)
        if any(speaker_id >= self.n_speakers for _, speaker_id in normalized):
            raise ValueError("MeloTTS speaker IDs must be smaller than `n_speakers`.")

    @property
    def speakers(self) -> dict[str, int]:
        return dict(self.speaker_ids)


@dataclass(frozen=True, slots=True)
class MeloTTSModelConfig:
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    n_layers_trans_flow: int = 3
    n_flow_layer: int = 4
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    upsample_rates: tuple[int, ...] = (8, 8, 2, 2, 2)
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 8, 2, 2)
    gin_channels: int = 256
    use_spk_conditioned_encoder: bool = True
    use_noise_scaled_mas: bool = True
    use_duration_discriminator: bool = True
    use_spectral_norm: bool = False
    use_transformer_flow: bool = True
    flow_share_parameter: bool = False
    use_vc: bool = False
    mas_noise_scale_initial: float = 0.01
    noise_scale_delta: float = 0.000002

    def __post_init__(self) -> None:
        for name in (
                "inter_channels",
                "hidden_channels",
                "filter_channels",
                "n_heads",
                "n_layers",
                "n_layers_trans_flow",
                "n_flow_layer",
                "kernel_size",
                "upsample_initial_channel",
                "gin_channels",
        ):
            _positive_integer(name, getattr(self, name))
        _probability("p_dropout", self.p_dropout)
        if self.resblock not in {"1", "2"}:
            raise ValueError("MeloTTS `resblock` must be '1' or '2'.")
        object.__setattr__(
            self,
            "resblock_kernel_sizes",
            _integer_tuple(
                "resblock_kernel_sizes",
                self.resblock_kernel_sizes,
            ),
        )
        object.__setattr__(
            self,
            "resblock_dilation_sizes",
            _nested_integer_tuple(
                "resblock_dilation_sizes",
                self.resblock_dilation_sizes,
            ),
        )
        object.__setattr__(
            self,
            "upsample_rates",
            _integer_tuple("upsample_rates", self.upsample_rates),
        )
        object.__setattr__(
            self,
            "upsample_kernel_sizes",
            _integer_tuple(
                "upsample_kernel_sizes",
                self.upsample_kernel_sizes,
            ),
        )
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilation_sizes):
            raise ValueError("MeloTTS residual kernels and dilations must align.")
        expected_dilations = 3 if self.resblock == "1" else 2
        if any(len(group) != expected_dilations for group in self.resblock_dilation_sizes):
            raise ValueError(
                f"MeloTTS resblock {self.resblock} requires exactly "
                f"{expected_dilations} dilations per kernel.")
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("MeloTTS upsample rates and kernels must align.")
        if any(kernel < rate or (kernel - rate) % 2 for rate, kernel in zip(
                self.upsample_rates,
                self.upsample_kernel_sizes,
        )):
            raise ValueError(
                "MeloTTS upsample kernels must be at least their rates and "
                "have matching parity.")
        channel_divisor = 2**len(self.upsample_rates)
        if (self.upsample_initial_channel < channel_divisor or
                self.upsample_initial_channel % channel_divisor):
            raise ValueError("MeloTTS upsample channels must be divisible by two at "
                             "every decoder stage.")
        if self.inter_channels % 2:
            raise ValueError("MeloTTS flow channels must be even.")
        if self.hidden_channels % self.n_heads:
            raise ValueError("MeloTTS hidden channels must be divisible by heads.")
        if self.kernel_size % 2 == 0:
            raise ValueError("MeloTTS text kernel size must be odd.")
        if self.use_spk_conditioned_encoder and self.n_layers < 3:
            raise ValueError(
                "Speaker-conditioned MeloTTS text encoders require at least "
                "three attention layers.")
        for name in (
                "use_spk_conditioned_encoder",
                "use_noise_scaled_mas",
                "use_duration_discriminator",
                "use_spectral_norm",
                "use_transformer_flow",
                "flow_share_parameter",
                "use_vc",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.use_transformer_flow and self.n_layers_trans_flow != 3:
            raise ValueError(
                "The pinned MeloTTS transformer coupling layer requires "
                "`n_layers_trans_flow=3`.")
        _nonnegative_number(
            "mas_noise_scale_initial",
            self.mas_noise_scale_initial,
        )
        _nonnegative_number("noise_scale_delta", self.noise_scale_delta)


@dataclass(frozen=True, slots=True)
class MeloTTSArchitectureConfig:
    """Everything required to reconstruct one checkpoint-compatible graph."""

    symbols: tuple[str, ...]
    num_tones: int
    num_languages: int
    segment_size: int = 16_384
    data: MeloTTSDataConfig = MeloTTSDataConfig()
    model: MeloTTSModelConfig = MeloTTSModelConfig()

    def __post_init__(self) -> None:
        if isinstance(self.symbols, (str, bytes)):
            raise TypeError("MeloTTS `symbols` must be a sequence.")
        symbols = tuple(self.symbols)
        if not symbols or any(not isinstance(symbol, str) or not symbol for symbol in symbols):
            raise ValueError("MeloTTS symbols must be non-empty strings.")
        if len(set(symbols)) != len(symbols):
            raise ValueError("MeloTTS symbols must be unique.")
        object.__setattr__(self, "symbols", symbols)
        _positive_integer("num_tones", self.num_tones)
        _positive_integer("num_languages", self.num_languages)
        _positive_integer("segment_size", self.segment_size)
        if not isinstance(self.data, MeloTTSDataConfig):
            raise TypeError("`data` must be a MeloTTSDataConfig.")
        if not isinstance(self.model, MeloTTSModelConfig):
            raise TypeError("`model` must be a MeloTTSModelConfig.")
        if self.segment_size % self.data.hop_length:
            raise ValueError("MeloTTS `segment_size` must be divisible by `hop_length`.")
        upsample_factor = math.prod(self.model.upsample_rates)
        if upsample_factor != self.data.hop_length:
            raise ValueError("MeloTTS decoder upsample factor must equal the audio hop length.")

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)

    @property
    def segment_frames(self) -> int:
        return self.segment_size // self.data.hop_length

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> MeloTTSArchitectureConfig:
        if not isinstance(values, Mapping):
            raise TypeError("MeloTTS configuration must be a mapping.")
        root = dict(values)
        architecture = root.get("architecture")
        if architecture is not None and architecture != "melotts":
            raise ValueError("Native MeloTTS configuration declares another architecture.")
        format_version = root.get("format_version")
        if format_version is not None and format_version != 1:
            raise ValueError("Unsupported native MeloTTS config format version.")
        data_values = root.get("data", {})
        model_values = root.get("model", {})
        train_values = root.get("train", {})
        if not isinstance(data_values, Mapping):
            raise TypeError("MeloTTS `data` must be a mapping.")
        if not isinstance(model_values, Mapping):
            raise TypeError("MeloTTS `model` must be a mapping.")
        if not isinstance(train_values, Mapping):
            raise TypeError("MeloTTS `train` must be a mapping.")
        data_root = dict(data_values)
        model_root = dict(model_values)
        speakers = data_root.pop(
            "speaker_ids",
            data_root.pop("spk2id", {}),
        )
        data = MeloTTSDataConfig(
            sample_rate=data_root.pop(
                "sample_rate",
                data_root.pop("sampling_rate", 44_100),
            ),
            n_fft=data_root.pop(
                "n_fft",
                data_root.pop("filter_length", 2_048),
            ),
            hop_length=data_root.pop("hop_length", 512),
            win_length=data_root.pop("win_length", 2_048),
            n_mels=data_root.pop(
                "n_mels",
                data_root.pop("n_mel_channels", 128),
            ),
            mel_fmin=data_root.pop("mel_fmin", 0.0),
            mel_fmax=data_root.pop("mel_fmax", None),
            n_speakers=data_root.pop("n_speakers", 256),
            speaker_ids=_speaker_items(speakers),
            add_blank=data_root.pop("add_blank", True),
        )
        # These upstream data-only flags do not alter the executable graph.
        for key in (
                "cleaned_text",
                "disable_bert",
                "max_wav_value",
                "training_files",
                "validation_files",
        ):
            data_root.pop(key, None)
        if data_root:
            unknown = ", ".join(sorted(data_root))
            raise ValueError(f"Unsupported MeloTTS data config keys: {unknown}.")

        use_mel_posterior = model_root.pop(
            "use_mel_posterior_encoder",
            False,
        )
        if use_mel_posterior is not False:
            raise ValueError(
                "The pinned MeloTTS graph does not implement the optional "
                "mel posterior encoder.")
        model_root.pop("n_layers_q", None)
        model = MeloTTSModelConfig(**model_root)
        segment_size = root.get(
            "segment_size",
            train_values.get("segment_size", 16_384),
        )
        return cls(
            symbols=tuple(root.get("symbols", ())),
            num_tones=root.get("num_tones"),
            num_languages=root.get("num_languages"),
            segment_size=segment_size,
            data=data,
            model=model,
        )

    def to_dict(self) -> dict[str, Any]:
        model = {name: getattr(self.model, name) for name in self.model.__slots__}
        for name in (
                "resblock_kernel_sizes",
                "upsample_rates",
                "upsample_kernel_sizes",
        ):
            model[name] = list(model[name])
        model["resblock_dilation_sizes"] = [list(group) for group in self.model.resblock_dilation_sizes]
        data = {
            "sample_rate": self.data.sample_rate,
            "n_fft": self.data.n_fft,
            "hop_length": self.data.hop_length,
            "win_length": self.data.win_length,
            "n_mels": self.data.n_mels,
            "mel_fmin": self.data.mel_fmin,
            "mel_fmax": self.data.mel_fmax,
            "n_speakers": self.data.n_speakers,
            "speaker_ids": self.data.speakers,
            "add_blank": self.data.add_blank,
        }
        return {
            "architecture": "melotts",
            "format_version": 1,
            "symbols": list(self.symbols),
            "num_tones": self.num_tones,
            "num_languages": self.num_languages,
            "segment_size": self.segment_size,
            "data": data,
            "model": model,
        }


def load_melotts_config(path: str | Path, ) -> MeloTTSArchitectureConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MeloTTS configuration was not found: {source}.")
    if source.suffix.lower() != ".json":
        raise ValueError("MeloTTS native configuration must be JSON.")
    return MeloTTSArchitectureConfig.from_dict(read_json_file(source))


__all__ = [
    "MeloTTSArchitectureConfig",
    "MeloTTSDataConfig",
    "MeloTTSModelConfig",
    "load_melotts_config",
]
