"""Validated native configurations for NeuTTS and NeuCodec.

The executable fields mirror the immutable public checkpoints listed in
``metadata.py``.  Unknown JSON metadata is retained for round-tripping, while
values that change graph mathematics are validated eagerly.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

from voicehub.models.causal_lm.native.configuration import CausalLMConfig
from voicehub.models.llasa.xcodec2 import Wav2Vec2BertSemanticConfig


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return value


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"`{name}` must be finite and in [0, 1).")
    return result


@dataclass(frozen=True, slots=True)
class NeuCodecConfig:
    """Architecture values for the official self-contained NeuCodec release."""

    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    encoder_hidden_size: int = 48
    downsampling_ratios: tuple[int, ...] = (2, 2, 4, 4, 5)
    input_sampling_rate: int = 16_000
    output_sampling_rate: int = 24_000
    activation_dropout: float = 0.1
    quantization_dim: int = 2048
    quantization_levels: tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4, 4)
    rope_theta: float = 10_000.0
    semantic_model_config: Wav2Vec2BertSemanticConfig = field(default_factory=Wav2Vec2BertSemanticConfig)
    extra_config: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
                "encoder_hidden_size",
                "input_sampling_rate",
                "output_sampling_rate",
                "quantization_dim",
        ):
            _positive_integer(getattr(self, name), name=name)
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError("The released NeuCodec decoder uses one key/value head per "
                             "attention head.")
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("`num_attention_heads * head_dim` must equal `hidden_size`.")
        if self.hidden_size % 32:
            raise ValueError("NeuCodec hidden size must be divisible by 32 for GroupNorm.")
        if self.hidden_act != "silu":
            raise ValueError("Published NeuCodec checkpoints require SiLU.")
        if self.attention_bias:
            raise ValueError("Published NeuCodec attention projections are bias-free.")
        object.__setattr__(
            self,
            "attention_dropout",
            _probability(self.attention_dropout, name="attention_dropout"),
        )
        object.__setattr__(
            self,
            "activation_dropout",
            _probability(self.activation_dropout, name="activation_dropout"),
        )
        ratios = tuple(self.downsampling_ratios)
        if not ratios:
            raise ValueError("`downsampling_ratios` cannot be empty.")
        for value in ratios:
            _positive_integer(value, name="downsampling_ratios")
        object.__setattr__(self, "downsampling_ratios", ratios)
        levels = tuple(self.quantization_levels)
        if not levels or any(isinstance(value, bool) or not isinstance(value, int) or value < 2
                             for value in levels):
            raise ValueError("`quantization_levels` must contain integers of at least two.")
        if math.prod(levels) != 65_536:
            raise ValueError("NeuTTS expects NeuCodec's 65,536-entry scalar codebook.")
        object.__setattr__(self, "quantization_levels", levels)
        semantic = self.semantic_model_config
        if isinstance(semantic, dict):
            semantic = Wav2Vec2BertSemanticConfig.from_dict(semantic)
            object.__setattr__(self, "semantic_model_config", semantic)
        if not isinstance(semantic, Wav2Vec2BertSemanticConfig):
            raise TypeError("`semantic_model_config` must be a mapping or semantic config.")
        if self.quantization_dim != self.hidden_size + semantic.hidden_size:
            raise ValueError(
                "`quantization_dim` must equal the acoustic and semantic "
                "hidden sizes combined.")
        if semantic.feature_projection_input_dim != 160:
            raise ValueError("The NeuCodec frontend emits paired 80-bin filter-bank frames.")
        if self.encoder_frame_rate != self.decoder_frame_rate:
            raise ValueError("NeuCodec encoder and decoder frame rates must match.")
        if (isinstance(self.rope_theta, bool) or not isinstance(self.rope_theta, (int, float)) or
                not math.isfinite(self.rope_theta) or self.rope_theta <= 1.0):
            raise ValueError("`rope_theta` must be finite and greater than one.")
        if not isinstance(self.extra_config, dict):
            raise TypeError("`extra_config` must be a dictionary.")

    @property
    def sampling_rate(self) -> int:
        """Compatibility alias used by the shared neural building blocks."""
        return self.input_sampling_rate

    @property
    def encoder_hop_length(self) -> int:
        return math.prod(self.downsampling_ratios)

    @property
    def hop_length(self) -> int:
        return round(self.encoder_hop_length * self.output_sampling_rate / self.input_sampling_rate)

    @property
    def n_fft(self) -> int:
        return self.hop_length * 4

    @property
    def encoder_frame_rate(self) -> int:
        return self.input_sampling_rate // self.encoder_hop_length

    @property
    def decoder_frame_rate(self) -> int:
        return self.output_sampling_rate // self.hop_length

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> NeuCodecConfig:
        if not isinstance(values, Mapping):
            raise TypeError("NeuCodec configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        if str(source.pop("model_type", "neucodec")).lower() != "neucodec":
            raise ValueError("NeuCodec config must declare `model_type=neucodec`.")
        # The first public conversion called this field `sampling_rate`.
        input_rate = source.pop(
            "input_sampling_rate",
            source.pop("sampling_rate", None),
        )
        if input_rate is not None:
            source["input_sampling_rate"] = input_rate
        rope = source.pop("rope_parameters", None)
        if rope is not None:
            if not isinstance(rope, Mapping):
                raise TypeError("`rope_parameters` must be a mapping.")
            if rope.get("rope_type", "default") != "default":
                raise ValueError("NeuCodec supports default RoPE only.")
            source["rope_theta"] = rope.get("rope_theta", 10_000.0)
        semantic = source.pop("semantic_model_config", None)
        known = {item.name for item in fields(cls)} - {"extra_config"}
        selected = {name: source.pop(name) for name in tuple(source) if name in known}
        if semantic is not None:
            selected["semantic_model_config"] = (Wav2Vec2BertSemanticConfig.from_dict(dict(semantic)))
        selected["extra_config"] = source
        return cls(**selected)

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        extra = output.pop("extra_config")
        output.update(extra)
        output["model_type"] = "neucodec"
        output['architectures'] = ["NeuCodecModel"]
        output["sampling_rate"] = output.pop("input_sampling_rate")
        output["downsampling_ratios"] = list(self.downsampling_ratios)
        output["quantization_levels"] = list(self.quantization_levels)
        output["semantic_model_config"] = self.semantic_model_config.to_dict()
        output["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": self.rope_theta,
        }
        output.pop("rope_theta")
        return output


@dataclass(frozen=True, slots=True)
class NeuTTSBackboneConfig:
    """Native LM configuration plus NeuTTS prompt semantics."""

    causal_lm: CausalLMConfig
    input_format: str
    supported_languages: tuple[str, ...] = ()
    supported_emotions: tuple[str, ...] = ()
    source_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.input_format not in {"BPE", "phonemes"}:
            raise ValueError("NeuTTS `input_format` must be 'BPE' or 'phonemes'.")
        object.__setattr__(
            self,
            "supported_languages",
            tuple(str(value) for value in self.supported_languages),
        )
        object.__setattr__(
            self,
            "supported_emotions",
            tuple(str(value).lower() for value in self.supported_emotions),
        )
        if not isinstance(self.source_config, Mapping):
            raise TypeError("`source_config` must be a mapping.")
        object.__setattr__(
            self,
            "source_config",
            copy.deepcopy(dict(self.source_config)),
        )

    @property
    def linear_rope_factor(self) -> float | None:
        rope = self.source_config.get("rope_scaling")
        if not isinstance(rope, Mapping):
            return None
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "linear":
            return None
        factor = float(rope.get("factor", 0.0))
        if not math.isfinite(factor) or factor < 1.0:
            raise ValueError("NeuTTS linear RoPE requires a finite factor of at least one.")
        return factor

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> NeuTTSBackboneConfig:
        if not isinstance(values, Mapping):
            raise TypeError("NeuTTS backbone configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        neuphonic = source.get("neuphonic", {})
        if neuphonic is None:
            neuphonic = {}
        if not isinstance(neuphonic, Mapping):
            raise TypeError("NeuTTS `neuphonic` metadata must be a mapping.")
        input_format = str(
            neuphonic.get(
                "input_format",
                "BPE" if source.get("model_type") == "qwen3" else "phonemes",
            ))
        if input_format.lower() == "bpe":
            input_format = "BPE"
        else:
            input_format = input_format.lower()

        # Linear scaling is a published NeuTTS-Nano specialization.  The
        # shared causal-LM config intentionally rejects it, so parse a graph-
        # equivalent base config and install the local rotary module later.
        causal_source = copy.deepcopy(source)
        rope = causal_source.get("rope_scaling")
        if isinstance(rope, Mapping):
            rope_type = rope.get("rope_type", rope.get("type", "default"))
            if rope_type == "linear":
                causal_source["rope_scaling"] = None
        causal = CausalLMConfig.from_dict(causal_source)
        result = cls(
            causal_lm=causal,
            input_format=input_format,
            supported_languages=tuple(neuphonic.get("supported_langs", ())),
            supported_emotions=tuple(neuphonic.get("supported_emotions", ())),
            source_config=source,
        )
        # Force validation now rather than during the first forward pass.
        result.linear_rope_factor
        return result

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.source_config))


__all__ = [
    "NeuCodecConfig",
    "NeuTTSBackboneConfig",
]
