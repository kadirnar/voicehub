"""Validated configuration for VoiceHub's native WeNet U2++ graph."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any


def _integer(value: int, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}.")
    return value


def _probability(value: float, *, name: str, inclusive_one: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    upper_valid = result <= 1.0 if inclusive_one else result < 1.0
    if result < 0.0 or not upper_valid:
        closing = "]" if inclusive_one else ")"
        raise ValueError(f"`{name}` must be in [0, 1{closing}.")
    return result


@dataclass(frozen=True, slots=True)
class WeNetU2PPConfig:
    """Exact GigaSpeech XL U2++ Conformer graph.

    The released variant is immutable so a compatible filename cannot
    silently select a different graph. ``variant="custom"`` is reserved
    for VoiceHub-owned, shape-reduced checkpoints and tests.
    """

    variant: str = "gigaspeech-u2pp-conformer"
    sampling_rate: int = 16_000
    input_dim: int = 80
    vocab_size: int = 4_999
    encoder_dim: int = 512
    encoder_heads: int = 8
    encoder_linear_units: int = 2_048
    encoder_layers: int = 12
    decoder_heads: int = 8
    decoder_linear_units: int = 2_048
    decoder_layers: int = 3
    reverse_decoder_layers: int = 3
    convolution_kernel_size: int = 31
    causal_convolution: bool = True
    dropout: float = 0.1
    positional_dropout: float = 0.1
    attention_dropout: float = 0.0
    decoder_self_attention_dropout: float = 0.0
    decoder_source_attention_dropout: float = 0.0
    use_dynamic_chunk: bool = True
    use_dynamic_left_chunk: bool = False
    static_chunk_size: int = 0
    ctc_weight: float = 0.3
    reverse_weight: float = 0.3
    label_smoothing: float = 0.1
    length_normalized_loss: bool = False
    blank_token_id: int = 0
    unknown_token_id: int = 1
    sos_eos_token_id: int = 4_998
    ignore_token_id: int = -1
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    inference_dither: float = 0.0
    training_dither: float = 1.0
    spec_augment: bool = True
    spec_time_masks: int = 3
    spec_frequency_masks: int = 2
    spec_max_time: int = 50
    spec_max_frequency: int = 10
    optimizer: str = "adam"
    learning_rate: float = 0.001
    warmup_steps: int = 80_000
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
        if variant not in {"gigaspeech-u2pp-conformer", "custom"}:
            raise ValueError(
                "Native WeNet supports the audited `gigaspeech-u2pp-conformer` "
                "graph or a VoiceHub-owned `custom` graph.")
        object.__setattr__(self, "variant", variant)
        for name in (
                "sampling_rate",
                "input_dim",
                "vocab_size",
                "encoder_dim",
                "encoder_heads",
                "encoder_linear_units",
                "encoder_layers",
                "decoder_heads",
                "decoder_linear_units",
                "decoder_layers",
                "reverse_decoder_layers",
                "convolution_kernel_size",
                "warmup_steps",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name),
            )
        for name in (
                "static_chunk_size",
                "blank_token_id",
                "unknown_token_id",
                "sos_eos_token_id",
                "spec_time_masks",
                "spec_frequency_masks",
                "spec_max_time",
                "spec_max_frequency",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name, minimum=0),
            )
        if (isinstance(self.ignore_token_id, bool) or not isinstance(self.ignore_token_id, int)):
            raise TypeError("`ignore_token_id` must be an integer.")
        if self.encoder_dim % self.encoder_heads:
            raise ValueError("`encoder_dim` must be divisible by `encoder_heads`.")
        if self.encoder_dim % self.decoder_heads:
            raise ValueError("`encoder_dim` must be divisible by `decoder_heads`.")
        if self.convolution_kernel_size % 2 == 0:
            raise ValueError("`convolution_kernel_size` must be odd.")
        if self.vocab_size < 3:
            raise ValueError("`vocab_size` must contain at least three tokens.")
        special_ids = {
            "blank_token_id": self.blank_token_id,
            "unknown_token_id": self.unknown_token_id,
            "sos_eos_token_id": self.sos_eos_token_id,
        }
        outside_vocabulary = [name for name, token_id in special_ids.items() if token_id >= self.vocab_size]
        if outside_vocabulary:
            raise ValueError(
                "Special token IDs must be inside the vocabulary: " + ", ".join(outside_vocabulary) + ".")
        if len(set(special_ids.values())) != len(special_ids):
            raise ValueError("Blank, unknown, and SOS/EOS token IDs must differ.")
        for name in (
                "dropout",
                "positional_dropout",
                "attention_dropout",
                "decoder_self_attention_dropout",
                "decoder_source_attention_dropout",
                "label_smoothing",
        ):
            object.__setattr__(
                self,
                name,
                _probability(getattr(self, name), name=name),
            )
        for name in ("ctc_weight", "reverse_weight"):
            object.__setattr__(
                self,
                name,
                _probability(
                    getattr(self, name),
                    name=name,
                    inclusive_one=True,
                ),
            )
        for name in (
                "causal_convolution",
                "use_dynamic_chunk",
                "use_dynamic_left_chunk",
                "length_normalized_loss",
                "spec_augment",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        for name in (
                "frame_length_ms",
                "frame_shift_ms",
                "inference_dither",
                "training_dither",
                "learning_rate",
                "gradient_clip_norm",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a real number.")
            value = float(value)
            if value < 0.0 or (name in {
                    "frame_length_ms",
                    "frame_shift_ms",
                    "learning_rate",
                    "gradient_clip_norm",
            } and value == 0.0):
                raise ValueError(f"`{name}` must be positive.")
            object.__setattr__(self, name, value)
        if self.optimizer != "adam":
            raise ValueError("The audited training recipe uses the Adam optimizer.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )
        if variant == "gigaspeech-u2pp-conformer":
            self._validate_released_graph()

    def _validate_released_graph(self) -> None:
        expected = type(self)(variant="custom")
        changed = []
        for item in fields(self):
            if item.name in {"variant", "extra_config"}:
                continue
            if getattr(self, item.name) != getattr(expected, item.name):
                changed.append(item.name)
        if changed:
            raise ValueError(
                "The audited GigaSpeech U2++ graph is immutable; use "
                "`variant='custom'` for changed field(s): " + ", ".join(changed) + ".")

    @property
    def subsampling_rate(self) -> int:
        return 6

    @property
    def right_context(self) -> int:
        return 10

    @classmethod
    def coerce(
        cls,
        value: WeNetU2PPConfig | Mapping[str, Any] | None,
    ) -> WeNetU2PPConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise TypeError("WeNet U2++ configuration must be a mapping or config.")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> WeNetU2PPConfig:
        if not isinstance(values, Mapping):
            raise TypeError("WeNet U2++ configuration must be a mapping.")
        source = dict(values)
        allowed = {item.name for item in fields(cls)}
        known_metadata = {
            "_name_or_path",
            'architectures',
            "checkpoint_format",
            "model_type",
            "source_checkpoint_name",
            "source_checkpoint_sha256",
            "source_tensor_fingerprint",
            "voicehub_provider",
        }
        unknown = sorted(set(source) - allowed - known_metadata)
        if unknown:
            raise ValueError("Unsupported WeNet U2++ configuration field(s): " + ", ".join(unknown) + ".")
        extra = {name: source.pop(name) for name in tuple(source) if name in known_metadata}
        source.setdefault("extra_config", extra)
        return cls(**source)

    def to_dict(self) -> dict[str, Any]:
        values = {
            item.name: copy.deepcopy(getattr(self, item.name))
            for item in fields(self) if item.name != "extra_config"
        }
        values["extra_config"] = copy.deepcopy(dict(self.extra_config))
        return values


__all__ = ["WeNetU2PPConfig"]
