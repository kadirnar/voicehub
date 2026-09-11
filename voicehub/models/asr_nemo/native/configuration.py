"""Validated configuration for VoiceHub-native NeMo QuartzNet CTC graphs."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

QUARTZNET15X5_VOCABULARY = (
    " ",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "'",
)


def _integer(
    name: str,
    value: int,
    *,
    minimum: int = 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}.")
    return value


def _probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    normalized = float(value)
    if not 0.0 <= normalized < 1.0:
        raise ValueError(f"`{name}` must be in [0, 1).")
    return normalized


@dataclass(frozen=True, slots=True)
class JasperBlockConfig:
    """One length-aware Jasper/QuartzNet convolutional block."""

    filters: int
    repeat: int
    kernel_size: int
    stride: int = 1
    dilation: int = 1
    dropout: float = 0.0
    residual: bool = False
    separable: bool = True

    def __post_init__(self) -> None:
        for name in ("filters", "repeat", "kernel_size", "stride", "dilation"):
            object.__setattr__(
                self,
                name,
                _integer(name, getattr(self, name)),
            )
        if self.kernel_size % 2 == 0:
            raise ValueError("QuartzNet kernels must have an odd width.")
        if self.stride > 1 and self.dilation > 1:
            raise ValueError("A Jasper block cannot combine stride and dilation.")
        object.__setattr__(
            self,
            "dropout",
            _probability("dropout", self.dropout),
        )
        for name in ("residual", "separable"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> JasperBlockConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Jasper block configuration must be a mapping.")
        source = dict(values)
        if "kernel" in source and "kernel_size" not in source:
            kernel = source.pop("kernel")
            source["kernel_size"] = kernel[0] if isinstance(kernel, Sequence) else kernel
        for name in ("stride", "dilation"):
            value = source.get(name)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) != 1:
                    raise ValueError(f"`{name}` must contain exactly one value.")
                source[name] = value[0]
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(source) - allowed)
        if unknown:
            raise ValueError(f"Unsupported Jasper block option(s): {', '.join(unknown)}.")
        return cls(**source)

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def quartznet15x5_blocks() -> tuple[JasperBlockConfig, ...]:
    """Return the exact block sequence in NVIDIA's released checkpoint."""
    values = [
        (256, 1, 33, 2, 1, False, True),
        (256, 5, 33, 1, 1, True, True),
        (256, 5, 33, 1, 1, True, True),
        (256, 5, 33, 1, 1, True, True),
        (256, 5, 39, 1, 1, True, True),
        (256, 5, 39, 1, 1, True, True),
        (256, 5, 39, 1, 1, True, True),
        (512, 5, 51, 1, 1, True, True),
        (512, 5, 51, 1, 1, True, True),
        (512, 5, 51, 1, 1, True, True),
        (512, 5, 63, 1, 1, True, True),
        (512, 5, 63, 1, 1, True, True),
        (512, 5, 63, 1, 1, True, True),
        (512, 5, 75, 1, 1, True, True),
        (512, 5, 75, 1, 1, True, True),
        (512, 5, 75, 1, 1, True, True),
        (512, 1, 87, 1, 2, False, True),
        (1024, 1, 1, 1, 1, False, False),
    ]
    return tuple(
        JasperBlockConfig(
            filters=filters,
            repeat=repeat,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            residual=residual,
            separable=separable,
        ) for (
            filters,
            repeat,
            kernel_size,
            stride,
            dilation,
            residual,
            separable,
        ) in values)


@dataclass(frozen=True, slots=True)
class NeMoQuartzNetCTCConfig:
    """Complete native graph for NVIDIA's English QuartzNet15x5 release.

    ``variant="quartznet15x5"`` is intentionally locked to the audited
    checkpoint. ``variant="custom"`` enables shape-reduced research
    graphs and VoiceHub-native checkpoints without claiming
    compatibility with an arbitrary NeMo archive.
    """

    variant: str = "quartznet15x5"
    sampling_rate: int = 16_000
    window_length: int = 320
    hop_length: int = 160
    n_fft: int = 512
    num_mel_bins: int = 64
    preemphasis: float = 0.97
    log_guard: float = 2**-24
    dither: float = 1e-5
    pad_to: int = 16
    frontend_gradients: bool = False
    vocabulary: tuple[str, ...] = QUARTZNET15X5_VOCABULARY
    encoder_blocks: tuple[JasperBlockConfig, ...] = field(default_factory=quartznet15x5_blocks, )
    spec_cutout_masks: int = 5
    spec_cutout_time: int = 120
    spec_cutout_frequency: int = 50
    ctc_reduction: str = "mean_batch"
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ValueError("`variant` must be a non-empty string.")
        variant = self.variant.strip().lower().replace("_", "-")
        if variant not in {"quartznet15x5", "custom"}:
            raise ValueError(
                "Native NeMo CTC supports `quartznet15x5` or an explicitly "
                "VoiceHub-owned `custom` graph.")
        object.__setattr__(self, "variant", variant)
        for name in (
                "sampling_rate",
                "window_length",
                "hop_length",
                "n_fft",
                "num_mel_bins",
                "pad_to",
        ):
            object.__setattr__(
                self,
                name,
                _integer(name, getattr(self, name)),
            )
        if self.n_fft < self.window_length:
            raise ValueError("`n_fft` cannot be smaller than `window_length`.")
        for name in ("preemphasis", "log_guard", "dither"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a real number.")
            object.__setattr__(self, name, float(value))
        if not 0.0 <= self.preemphasis < 1.0:
            raise ValueError("`preemphasis` must be in [0, 1).")
        if self.log_guard <= 0.0:
            raise ValueError("`log_guard` must be positive.")
        if self.dither < 0.0:
            raise ValueError("`dither` cannot be negative.")
        if not isinstance(self.frontend_gradients, bool):
            raise TypeError("`frontend_gradients` must be a boolean.")

        vocabulary = tuple(self.vocabulary)
        if len(vocabulary) < 2:
            raise ValueError("`vocabulary` must contain at least two labels.")
        if any(not isinstance(token, str) or len(token) != 1 for token in vocabulary):
            raise ValueError("QuartzNet CTC vocabulary entries must be single characters.")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("`vocabulary` cannot contain duplicate characters.")
        object.__setattr__(self, "vocabulary", vocabulary)

        blocks = tuple(
            value if isinstance(value, JasperBlockConfig) else JasperBlockConfig.from_dict(value)
            for value in self.encoder_blocks)
        if not blocks:
            raise ValueError("`encoder_blocks` cannot be empty.")
        object.__setattr__(self, "encoder_blocks", blocks)
        for name in (
                "spec_cutout_masks",
                "spec_cutout_time",
                "spec_cutout_frequency",
        ):
            object.__setattr__(
                self,
                name,
                _integer(name, getattr(self, name), minimum=0),
            )
        if self.ctc_reduction not in {"mean_batch", "mean_volume", "mean", "sum", "none"}:
            raise ValueError("`ctc_reduction` must be mean_batch, mean_volume, mean, sum, or none.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )
        if self.variant == "quartznet15x5":
            self._validate_released_graph()

    def _validate_released_graph(self) -> None:
        expected = {
            "sampling_rate": 16_000,
            "window_length": 320,
            "hop_length": 160,
            "n_fft": 512,
            "num_mel_bins": 64,
            "preemphasis": 0.97,
            "log_guard": 2**-24,
            "dither": 1e-5,
            "pad_to": 16,
            "vocabulary": QUARTZNET15X5_VOCABULARY,
            "encoder_blocks": quartznet15x5_blocks(),
            "spec_cutout_masks": 5,
            "spec_cutout_time": 120,
            "spec_cutout_frequency": 50,
        }
        changed = [name for name, value in expected.items() if getattr(self, name) != value]
        if changed:
            names = ", ".join(changed)
            raise ValueError(
                "The audited `quartznet15x5` variant has immutable graph "
                f"settings; changed: {names}. Use `variant='custom'` for a "
                "VoiceHub-native graph.")

    @property
    def blank_id(self) -> int:
        return len(self.vocabulary)

    @property
    def num_classes(self) -> int:
        return len(self.vocabulary) + 1

    @property
    def encoder_output_size(self) -> int:
        return self.encoder_blocks[-1].filters

    @property
    def subsampling_factor(self) -> int:
        factor = 1
        for block in self.encoder_blocks:
            factor *= block.stride**block.repeat
        return factor

    @property
    def output_frame_hop_samples(self) -> int:
        return self.hop_length * self.subsampling_factor

    @property
    def minimum_input_samples(self) -> int:
        return self.hop_length * 2

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> NeMoQuartzNetCTCConfig:
        if not isinstance(values, Mapping):
            raise TypeError("NeMo QuartzNet configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        if "sample_rate" in source and "sampling_rate" not in source:
            source["sampling_rate"] = source["sample_rate"]
        if "labels" in source and "vocabulary" not in source:
            source["vocabulary"] = source["labels"]
        canonical = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical if name in source}
        extras = {name: value for name, value in source.items() if name not in canonical | {"extra_config"}}
        supplied_extras = source.get("extra_config")
        if supplied_extras is not None:
            if not isinstance(supplied_extras, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied_extras)))
        return cls(**resolved, extra_config=extras)

    @classmethod
    def coerce(
        cls,
        value: NeMoQuartzNetCTCConfig | Mapping[str, Any],
    ) -> NeMoQuartzNetCTCConfig:
        if isinstance(value, cls):
            return value
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            if item.name == "encoder_blocks":
                value = [block.to_dict() for block in value]
            elif item.name == "vocabulary":
                value = list(value)
            result[item.name] = value
        result.setdefault("model_type", "nemo-quartznet-ctc")
        result.setdefault('architectures', ["NeMoQuartzNetForCTC"])
        return result


__all__ = [
    "JasperBlockConfig",
    "NeMoQuartzNetCTCConfig",
    "QUARTZNET15X5_VOCABULARY",
    "quartznet15x5_blocks",
]
