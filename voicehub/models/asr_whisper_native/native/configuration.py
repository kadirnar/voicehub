"""Validated configuration for VoiceHub's native Whisper architecture.

The field semantics were checked against OpenAI Whisper at revision
``04f449b8a437f1bbd3dba5c9f826aca972e7709a`` and Hugging Face Transformers
Whisper at revision ``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``.  This
module is an independent VoiceHub implementation and imports neither runtime.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

_OPENAI_ALIASES = {
    "n_mels": "num_mel_bins",
    "n_audio_ctx": "max_source_positions",
    "n_audio_state": "d_model",
    "n_audio_head": "encoder_attention_heads",
    "n_audio_layer": "encoder_layers",
    "n_vocab": "vocab_size",
    "n_text_ctx": "max_target_positions",
    "n_text_state": "d_model",
    "n_text_head": "decoder_attention_heads",
    "n_text_layer": "decoder_layers",
}


def _require_integer(name: str, value: int, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}; found {value}.")


def _require_probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"`{name}` must be finite and in the interval [0, 1).")


@dataclass(frozen=True, slots=True)
class WhisperConfig:
    """Complete executable configuration for the native Whisper graph.

    The public fields use Hugging Face's descriptive names.  Use
    :meth:`from_dict` to read either those fields or OpenAI's ``n_*``
    dimension keys.  Unknown metadata is retained in
    :attr:`extra_config`, so round trips do not discard tokenizer or
    generation metadata.
    """

    vocab_size: int = 51_865
    num_mel_bins: int = 80
    d_model: int = 384
    encoder_layers: int = 4
    encoder_attention_heads: int = 6
    encoder_ffn_dim: int | None = None
    decoder_layers: int = 4
    decoder_attention_heads: int = 6
    decoder_ffn_dim: int | None = None
    max_source_positions: int = 1_500
    max_target_positions: int = 448
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    encoder_layerdrop: float = 0.0
    decoder_layerdrop: float = 0.0
    activation_function: str = "gelu"
    init_std: float = 0.02
    layer_norm_eps: float = 1e-5
    scale_embedding: bool = False
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    decoder_start_token_id: int = 1
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        integer_fields = (
            "vocab_size",
            "num_mel_bins",
            "d_model",
            "encoder_layers",
            "encoder_attention_heads",
            "decoder_layers",
            "decoder_attention_heads",
            "max_source_positions",
            "max_target_positions",
        )
        for name in integer_fields:
            _require_integer(name, getattr(self, name), minimum=1)

        for name in ("encoder_ffn_dim", "decoder_ffn_dim"):
            value = getattr(self, name)
            if value is None:
                value = self.d_model * 4
                object.__setattr__(self, name, value)
            _require_integer(name, value, minimum=1)

        if self.d_model % 2:
            raise ValueError("`d_model` must be even for Whisper sinusoidal positions.")
        for name in ("encoder_attention_heads", "decoder_attention_heads"):
            heads = getattr(self, name)
            if self.d_model % heads:
                raise ValueError(f"`d_model` must be divisible by `{name}`.")

        for name in (
                "dropout",
                "attention_dropout",
                "activation_dropout",
                "encoder_layerdrop",
                "decoder_layerdrop",
        ):
            _require_probability(name, getattr(self, name))

        for name in ("init_std", "layer_norm_eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a real number.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"`{name}` must be finite and greater than zero.")

        supported_activations = {"gelu", "gelu_new", "relu", "silu"}
        if not isinstance(self.activation_function, str):
            raise TypeError("`activation_function` must be a string.")
        if self.activation_function not in supported_activations:
            choices = ", ".join(sorted(supported_activations))
            raise ValueError(
                f"`activation_function` must be one of {choices}; "
                f"found {self.activation_function!r}.")

        for name in ("scale_embedding", "use_cache"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")

        for name in (
                "pad_token_id",
                "bos_token_id",
                "eos_token_id",
                "decoder_start_token_id",
        ):
            value = getattr(self, name)
            _require_integer(name, value)
            if value >= self.vocab_size:
                raise ValueError(f"`{name}` must be smaller than `vocab_size`; found {value}.")

        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        detached = copy.deepcopy(dict(self.extra_config))
        object.__setattr__(self, "extra_config", MappingProxyType(detached))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> WhisperConfig:
        """Parse OpenAI dimensions, Hugging Face config fields, or a mixture.

        When a canonical field and one of its OpenAI aliases are both
        present, they must agree.  OpenAI's audio and text state
        dimensions must also be equal because Whisper cross-attention
        projects both in one model width.
        """
        if not isinstance(values, Mapping):
            raise TypeError("Whisper configuration values must be a mapping.")

        source = copy.deepcopy(dict(values))
        canonical_names = {item.name for item in fields(cls) if item.name != "extra_config"}
        resolved = {name: source[name] for name in canonical_names if name in source}

        for alias, target in _OPENAI_ALIASES.items():
            if alias not in source:
                continue
            value = source[alias]
            if target in resolved and resolved[target] != value:
                raise ValueError(
                    f"Conflicting Whisper dimensions: `{alias}` is {value!r}, "
                    f"but `{target}` is {resolved[target]!r}.")
            resolved[target] = value

        audio_width = source.get("n_audio_state")
        text_width = source.get("n_text_state")
        if audio_width is not None and text_width is not None:
            if audio_width != text_width:
                raise ValueError(
                    "Whisper requires `n_audio_state` and `n_text_state` to "
                    "match for cross-attention.")

        if "encoder_ffn_dim" not in resolved and "d_model" in resolved:
            resolved["encoder_ffn_dim"] = resolved["d_model"] * 4
        if "decoder_ffn_dim" not in resolved and "d_model" in resolved:
            resolved["decoder_ffn_dim"] = resolved["d_model"] * 4

        consumed = canonical_names | set(_OPENAI_ALIASES) | {"extra_config"}
        extras = {name: value for name, value in source.items() if name not in consumed}
        supplied_extras = source.get("extra_config")
        if supplied_extras is not None:
            if not isinstance(supplied_extras, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied_extras)))
        return cls(**resolved, extra_config=extras)

    @classmethod
    def from_openai_dimensions(
        cls,
        dimensions: Mapping[str, Any],
    ) -> WhisperConfig:
        """Construct a configuration from OpenAI checkpoint dimensions."""
        missing = tuple(name for name in _OPENAI_ALIASES if name not in dimensions)
        if missing:
            raise ValueError("OpenAI Whisper dimensions are incomplete; missing "
                             f"{', '.join(missing)}.")
        return cls.from_dict(dimensions)

    @classmethod
    def coerce(
        cls,
        value: WhisperConfig | Mapping[str, Any],
    ) -> WhisperConfig:
        """Return ``value`` as a validated :class:`WhisperConfig`."""
        if isinstance(value, cls):
            return value
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached Hugging Face-compatible configuration mapping."""
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name != "extra_config":
                result[item.name] = getattr(self, item.name)
        result.setdefault("model_type", "whisper")
        result.setdefault("is_encoder_decoder", True)
        return result

    def to_openai_dimensions(self) -> dict[str, int]:
        """Return the exact dimension mapping stored in OpenAI checkpoints."""
        return {
            "n_mels": self.num_mel_bins,
            "n_audio_ctx": self.max_source_positions,
            "n_audio_state": self.d_model,
            "n_audio_head": self.encoder_attention_heads,
            "n_audio_layer": self.encoder_layers,
            "n_vocab": self.vocab_size,
            "n_text_ctx": self.max_target_positions,
            "n_text_state": self.d_model,
            "n_text_head": self.decoder_attention_heads,
            "n_text_layer": self.decoder_layers,
        }

    @property
    def expected_input_frames(self) -> int:
        """Number of log-mel frames in a full Whisper context window."""
        return self.max_source_positions * 2


__all__ = ["WhisperConfig"]
