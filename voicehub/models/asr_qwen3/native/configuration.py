"""Validated configuration for VoiceHub's native Qwen3-ASR graph.

The public checkpoint schema is a composite ``thinker_config``
containing an audio encoder and a dense Qwen3 decoder.  This module
keeps that schema intact while translating only the executable decoder
fields into VoiceHub's shared causal-LM configuration.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from voicehub.models.causal_lm.native.configuration import Qwen3Config


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _nonnegative_probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"`{name}` must be finite and in [0, 1).")
    return result


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("Configuration extras must be a mapping.")
    return MappingProxyType(copy.deepcopy(dict(value)))


@dataclass(frozen=True, slots=True)
class Qwen3ASRAudioConfig:
    """Dimensions and numerical controls for the Qwen3-ASR audio tower."""

    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5_120
    d_model: int = 1_280
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_function: str = "gelu"
    activation_dropout: float = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    max_source_positions: int = 1_500
    n_window: int = 50
    output_dim: int = 3_584
    n_window_infer: int = 800
    conv_chunksize: int = 500
    downsample_hidden_size: int = 480
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "num_mel_bins",
                "encoder_layers",
                "encoder_attention_heads",
                "encoder_ffn_dim",
                "d_model",
                "max_source_positions",
                "n_window",
                "output_dim",
                "n_window_infer",
                "conv_chunksize",
                "downsample_hidden_size",
        ):
            _positive_integer(name, getattr(self, name))
        if self.d_model % self.encoder_attention_heads:
            raise ValueError("`d_model` must be divisible by `encoder_attention_heads`.")
        if self.d_model < 4 or self.d_model % 2:
            raise ValueError("`d_model` must be even and at least four.")
        if self.n_window != 50:
            raise ValueError(
                "Published Qwen3-ASR checkpoints require `n_window=50`; "
                "their output-length protocol is defined in 100-frame "
                "convolution chunks.")
        if self.n_window_infer % (self.n_window * 2):
            raise ValueError("`n_window_infer` must be divisible by `2 * n_window`.")
        for name in ("dropout", "attention_dropout", "activation_dropout"):
            object.__setattr__(
                self,
                name,
                _nonnegative_probability(name, getattr(self, name)),
            )
        if self.activation_function != "gelu":
            raise ValueError("Published Qwen3-ASR checkpoints require "
                             "`activation_function='gelu'`.")
        if not isinstance(self.scale_embedding, bool):
            raise TypeError("`scale_embedding` must be a boolean.")
        if (isinstance(self.initializer_range, bool) or not isinstance(self.initializer_range,
                                                                       (int, float)) or
                not math.isfinite(float(self.initializer_range)) or self.initializer_range <= 0):
            raise ValueError("`initializer_range` must be finite and positive.")
        object.__setattr__(
            self,
            "initializer_range",
            float(self.initializer_range),
        )
        object.__setattr__(
            self,
            "extra_config",
            _frozen_mapping(self.extra_config),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Qwen3ASRAudioConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Qwen3-ASR audio configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        names = {
            "num_mel_bins",
            "encoder_layers",
            "encoder_attention_heads",
            "encoder_ffn_dim",
            "d_model",
            "dropout",
            "attention_dropout",
            "activation_function",
            "activation_dropout",
            "scale_embedding",
            "initializer_range",
            "max_source_positions",
            "n_window",
            "output_dim",
            "n_window_infer",
            "conv_chunksize",
            "downsample_hidden_size",
        }
        resolved = {name: source.pop(name) for name in tuple(source) if name in names}
        supplied_extras = source.pop("extra_config", None)
        if supplied_extras is not None:
            if not isinstance(supplied_extras, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            source.update(copy.deepcopy(dict(supplied_extras)))
        return cls(**resolved, extra_config=source)

    def to_dict(self) -> dict[str, Any]:
        values = copy.deepcopy(dict(self.extra_config))
        for name in (
                "num_mel_bins",
                "encoder_layers",
                "encoder_attention_heads",
                "encoder_ffn_dim",
                "d_model",
                "dropout",
                "attention_dropout",
                "activation_function",
                "activation_dropout",
                "scale_embedding",
                "initializer_range",
                "max_source_positions",
                "n_window",
                "output_dim",
                "n_window_infer",
                "conv_chunksize",
                "downsample_hidden_size",
        ):
            values[name] = getattr(self, name)
        values["model_type"] = "qwen3_asr_audio_encoder"
        values["num_hidden_layers"] = self.encoder_layers
        return values


@dataclass(frozen=True, slots=True)
class Qwen3ASRArchitectureConfig:
    """Complete executable Qwen3-ASR configuration."""

    audio_config: Qwen3ASRAudioConfig
    text_config: Qwen3Config
    audio_token_id: int = 151_676
    audio_start_token_id: int = 151_669
    audio_end_token_id: int = 151_670
    support_languages: tuple[str, ...] = ()
    initializer_range: float = 0.02
    source_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.audio_config, Qwen3ASRAudioConfig):
            raise TypeError("`audio_config` must be Qwen3ASRAudioConfig.")
        if not isinstance(self.text_config, Qwen3Config):
            raise TypeError("`text_config` must be Qwen3Config.")
        if self.audio_config.output_dim != self.text_config.hidden_size:
            raise ValueError("The audio projector output must equal the decoder hidden "
                             "size.")
        if not self.text_config.tie_word_embeddings:
            raise ValueError("Qwen3-ASR checkpoints require tied decoder input/output "
                             "embeddings.")
        if (isinstance(self.initializer_range, bool) or not isinstance(self.initializer_range,
                                                                       (int, float)) or
                not math.isfinite(float(self.initializer_range)) or self.initializer_range <= 0):
            raise ValueError("`initializer_range` must be finite and positive.")
        object.__setattr__(
            self,
            "initializer_range",
            float(self.initializer_range),
        )
        for name in (
                "audio_token_id",
                "audio_start_token_id",
                "audio_end_token_id",
        ):
            value = getattr(self, name)
            _positive_integer(name, value)
            if value >= self.text_config.vocab_size:
                raise ValueError(f"`{name}` must be smaller than the decoder vocabulary.")
        languages = tuple(self.support_languages)
        if any(not isinstance(value, str) or not value.strip() for value in languages):
            raise ValueError("`support_languages` must contain non-empty strings.")
        if len(set(languages)) != len(languages):
            raise ValueError("`support_languages` cannot contain duplicates.")
        object.__setattr__(
            self,
            "support_languages",
            tuple(value.strip() for value in languages),
        )
        object.__setattr__(
            self,
            "source_config",
            _frozen_mapping(self.source_config),
        )

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> Qwen3ASRArchitectureConfig:
        """Parse either an official or a VoiceHub-native public config."""
        if not isinstance(values, Mapping):
            raise TypeError("Qwen3-ASR configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        model_type = str(source.get("model_type", "")).lower().replace("-", "_")
        if model_type not in {"qwen3_asr", "asr_qwen3"}:
            raise ValueError(
                "Qwen3-ASR requires `model_type='qwen3_asr'`; received "
                f"{model_type or 'unknown'!r}.")
        thinker = source.get("thinker_config")
        if thinker is None:
            raise ValueError("Qwen3-ASR config must contain `thinker_config`.")
        if not isinstance(thinker, Mapping):
            raise TypeError("Qwen3-ASR `thinker_config` must be a mapping.")
        thinker = copy.deepcopy(dict(thinker))
        audio_values = thinker.get("audio_config")
        text_values = thinker.get("text_config")
        if audio_values is None or text_values is None:
            raise ValueError("`thinker_config` must contain audio and text configurations.")
        if not isinstance(audio_values, Mapping) or not isinstance(
                text_values,
                Mapping,
        ):
            raise TypeError("Qwen3-ASR audio and text configurations must be mappings.")

        text_source = copy.deepcopy(dict(text_values))
        rope_scaling = text_source.get("rope_scaling")
        if rope_scaling is not None:
            if not isinstance(rope_scaling, Mapping):
                raise TypeError("Qwen3-ASR `rope_scaling` must be a mapping.")
            rope_values = dict(rope_scaling)
            rope_type = rope_values.get(
                "rope_type",
                rope_values.get("type", "default"),
            )
            if rope_type not in (None, "default"):
                raise ValueError("Published Qwen3-ASR supports only default multimodal "
                                 "RoPE.")
            section = rope_values.get("mrope_section")
            if section is not None and (isinstance(section,
                                                   (str, bytes)) or not isinstance(section, Sequence) or
                                        sum(int(value) for value in section)
                                        != int(text_source.get("head_dim", 128)) // 2):
                raise ValueError(
                    "Qwen3-ASR `mrope_section` must partition half of the "
                    "attention head dimension.")
            # ASR supplies identical temporal/height/width positions.  The
            # interleaving therefore reduces exactly to ordinary default
            # temporal RoPE in the shared decoder.
            text_source["rope_scaling"] = {
                "rope_type": "default",
                "type": "default",
            }
        text_source["model_type"] = "qwen3"
        text_config = Qwen3Config.from_dict(text_source)
        audio_config = Qwen3ASRAudioConfig.from_dict(audio_values)
        languages = source.get("support_languages", ())
        if languages is None:
            languages = ()
        if isinstance(languages, (str, bytes)) or not isinstance(
                languages,
                Sequence,
        ):
            raise TypeError("`support_languages` must be a sequence.")
        return cls(
            audio_config=audio_config,
            text_config=text_config,
            audio_token_id=int(thinker.get("audio_token_id", 151_676)),
            audio_start_token_id=int(thinker.get("audio_start_token_id", 151_669)),
            audio_end_token_id=int(thinker.get("audio_end_token_id", 151_670)),
            support_languages=tuple(languages),
            initializer_range=float(thinker.get("initializer_range", 0.02)),
            source_config=source,
        )

    @classmethod
    def coerce(
        cls,
        value: Qwen3ASRArchitectureConfig | Mapping[str, Any],
    ) -> Qwen3ASRArchitectureConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        """Return an official-schema configuration for portable exports."""
        root = copy.deepcopy(dict(self.source_config))
        text_values = self.text_config.to_dict()
        original_thinker = root.get("thinker_config")
        original_text = (
            original_thinker.get("text_config") if isinstance(original_thinker, Mapping) else None)
        if isinstance(original_text, Mapping) and "rope_scaling" in original_text:
            text_values["rope_scaling"] = copy.deepcopy(original_text["rope_scaling"])
        thinker = {
            **(copy.deepcopy(dict(original_thinker)) if isinstance(original_thinker, Mapping) else {}),
            "model_type":
            "qwen3_asr",
            "audio_config":
            self.audio_config.to_dict(),
            "text_config":
            text_values,
            "audio_token_id":
            self.audio_token_id,
            "audio_start_token_id":
            self.audio_start_token_id,
            "audio_end_token_id":
            self.audio_end_token_id,
            "initializer_range":
            self.initializer_range,
        }
        root.update({
            'architectures': ["Qwen3ASRForConditionalGeneration"],
            "model_type": "qwen3_asr",
            "support_languages": list(self.support_languages),
            "thinker_config": thinker,
        })
        return root


__all__ = [
    "Qwen3ASRArchitectureConfig",
    "Qwen3ASRAudioConfig",
]
