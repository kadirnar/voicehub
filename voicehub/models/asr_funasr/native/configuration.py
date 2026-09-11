"""Validated configuration for VoiceHub's native SenseVoiceSmall graph."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}.")
    return value


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not 0.0 <= result < 1.0:
        raise ValueError(f"`{name}` must be in [0, 1).")
    return result


@dataclass(frozen=True, slots=True)
class SenseVoiceSmallConfig:
    """The published 234M SenseVoiceSmall SANM-CTC architecture.

    ``variant="sensevoice-small"`` locks every graph-defining field to
    the audited public checkpoint. ``variant="custom"`` is intentionally
    limited to VoiceHub-owned, shape-reduced artifacts and unit tests;
    it is not a promise that unrelated FunASR checkpoints share this
    graph.
    """

    variant: str = "sensevoice-small"
    sampling_rate: int = 16_000
    num_mel_bins: int = 80
    lfr_window: int = 7
    lfr_stride: int = 6
    input_dimension: int = 560
    vocabulary_size: int = 25_055
    encoder_dimension: int = 512
    attention_heads: int = 4
    linear_units: int = 2_048
    encoder_blocks: int = 50
    temporal_blocks: int = 20
    memory_kernel_size: int = 11
    memory_shift: int = 0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    label_smoothing: float = 0.0
    length_normalized_loss: bool = True
    blank_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    ignore_token_id: int = -1
    query_embedding_size: int = 16
    waveform_scale: float = 32_768.0
    inference_dither: float = 0.0
    training_dither: float = 1.0
    language_dropout: float = 0.2
    optimizer: str = "adamw"
    learning_rate: float = 0.00002
    warmup_steps: int = 25_000
    gradient_clip_norm: float = 5.0
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ValueError("`variant` must be a non-empty string.")
        variant = self.variant.strip().lower().replace("_", "-")
        if variant not in {"sensevoice-small", "custom"}:
            raise ValueError(
                "Native SenseVoice supports the audited `sensevoice-small` "
                "graph or a VoiceHub-owned `custom` graph.")
        object.__setattr__(self, "variant", variant)
        for name in (
                "sampling_rate",
                "num_mel_bins",
                "lfr_window",
                "lfr_stride",
                "input_dimension",
                "vocabulary_size",
                "encoder_dimension",
                "attention_heads",
                "linear_units",
                "encoder_blocks",
                "query_embedding_size",
                "memory_kernel_size",
                "warmup_steps",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name, minimum=1),
            )
        for name in (
                "temporal_blocks",
                "memory_shift",
                "blank_token_id",
                "bos_token_id",
                "eos_token_id",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name, minimum=0),
            )
        if (isinstance(self.ignore_token_id, bool) or not isinstance(self.ignore_token_id, int)):
            raise TypeError("`ignore_token_id` must be an integer.")
        if self.input_dimension != self.num_mel_bins * self.lfr_window:
            raise ValueError("`input_dimension` must equal `num_mel_bins * lfr_window`.")
        if self.encoder_dimension % self.attention_heads:
            raise ValueError("`encoder_dimension` must be divisible by `attention_heads`.")
        if self.memory_kernel_size % 2 == 0:
            raise ValueError("`memory_kernel_size` must be odd.")
        if ((self.memory_kernel_size - 1) // 2 + self.memory_shift > self.memory_kernel_size - 1):
            raise ValueError("`memory_shift` exceeds the SANM kernel context.")
        for name in (
                "dropout",
                "attention_dropout",
                "label_smoothing",
                "language_dropout",
        ):
            object.__setattr__(
                self,
                name,
                _probability(getattr(self, name), name=name),
            )
        for name in (
                "waveform_scale",
                "inference_dither",
                "training_dither",
                "learning_rate",
                "gradient_clip_norm",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a real number.")
            value = float(value)
            minimum = 0.0 if name in {
                "inference_dither",
                "training_dither",
            } else 0.0
            if value < minimum or (name in {
                    "waveform_scale",
                    "learning_rate",
                    "gradient_clip_norm",
            } and value == 0.0):
                raise ValueError(f"`{name}` must be positive.")
            object.__setattr__(self, name, value)
        if not isinstance(self.length_normalized_loss, bool):
            raise TypeError("`length_normalized_loss` must be a boolean.")
        if self.optimizer != "adamw":
            raise ValueError("The audited SenseVoice recipe uses AdamW.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )
        special_ids = (
            self.blank_token_id,
            self.bos_token_id,
            self.eos_token_id,
        )
        if len(set(special_ids)) != len(special_ids):
            raise ValueError("Blank, BOS, and EOS token IDs must differ.")
        if any(token_id >= self.vocabulary_size for token_id in special_ids):
            raise ValueError("Special token IDs must be inside the vocabulary.")
        if variant == "sensevoice-small":
            self._validate_released_graph()

    def _validate_released_graph(self) -> None:
        expected = type(self)(variant="custom")
        changed = [
            item.name for item in fields(self) if item.name not in {"variant", "extra_config"} and
            getattr(self, item.name) != getattr(expected, item.name)
        ]
        if changed:
            raise ValueError(
                "The released SenseVoiceSmall graph is immutable; changed "
                f"field(s): {', '.join(changed)}. Use `variant='custom'` for "
                "VoiceHub-owned checkpoints.")

    def to_dict(self) -> dict[str, Any]:
        return {
            item.name: (
                copy.deepcopy(dict(self.extra_config)) if item.name == "extra_config" else copy.deepcopy(
                    getattr(self, item.name)))
            for item in fields(self)
        }

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> SenseVoiceSmallConfig:
        if not isinstance(values, Mapping):
            raise TypeError("SenseVoice configuration must be a mapping.")
        known = {item.name for item in fields(cls)}
        aliases = {
            "sample_rate": "sampling_rate",
            "vocab_size": "vocabulary_size",
        }
        normalized: dict[str, Any] = {}
        extra = dict(values.get("extra_config", {}))
        for name, value in values.items():
            target = aliases.get(name, name)
            if target in known:
                normalized[target] = value
            elif name not in {
                    'architectures',
                    "checkpoint_format",
                    "model_type",
                    "name_or_path",
                    "source_artifact_revision",
                    "source_training_revision",
                    "voicehub_provider",
            }:
                extra[name] = value
        normalized["extra_config"] = extra
        return cls(**normalized)

    @classmethod
    def coerce(
        cls,
        value: SenseVoiceSmallConfig | Mapping[str, Any] | None,
    ) -> SenseVoiceSmallConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls.from_dict(value)


__all__ = ["SenseVoiceSmallConfig"]
