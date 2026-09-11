"""Validated configuration for VoiceHub's native SeamlessM4T-v2 S2T graph."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

_PUBLISHED_VARIANT = "seamless-m4t-v2-large"
_VARIANTS = frozenset({_PUBLISHED_VARIANT, "custom"})


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}.")
    return value


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"`{name}` must be finite and positive.")
    return result


def _probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"`{name}` must be finite and in [0, 1).")
    return result


@dataclass(frozen=True, slots=True)
class SeamlessM4Tv2S2TConfig:
    """Executable speech-to-text subset of the published unified model.

    ``variant="seamless-m4t-v2-large"`` locks all graph and frontend
    dimensions to the audited Facebook checkpoint. ``variant="custom"``
    permits reduced VoiceHub-owned graphs for tests and derived
    artifacts; it never implies compatibility with a different public
    checkpoint.
    """

    variant: str = _PUBLISHED_VARIANT
    vocab_size: int = 256_102
    hidden_size: int = 1_024
    feature_projection_input_dim: int = 160
    speech_encoder_layers: int = 24
    speech_encoder_attention_heads: int = 16
    speech_encoder_intermediate_size: int = 4_096
    speech_encoder_hidden_act: str = "swish"
    speech_encoder_dropout: float = 0.0
    speech_encoder_layerdrop: float = 0.1
    speech_encoder_chunk_size: int = 20_000
    speech_encoder_left_chunk_num: int = 128
    conv_depthwise_kernel_size: int = 31
    position_embeddings_type: str = "relative_key"
    left_max_position_embeddings: int = 64
    right_max_position_embeddings: int = 8
    add_adapter: bool = True
    num_adapter_layers: int = 1
    adaptor_kernel_size: int = 8
    adaptor_stride: int = 8
    adaptor_dropout: float = 0.1
    decoder_layers: int = 24
    decoder_attention_heads: int = 16
    decoder_ffn_dim: int = 8_192
    decoder_layerdrop: float = 0.05
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    activation_function: str = "relu"
    max_position_embeddings: int = 4_096
    scale_embedding: bool = True
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    decoder_start_token_id: int = 3
    sampling_rate: int = 16_000
    num_mel_bins: int = 80
    feature_stride: int = 2
    feature_window_length: int = 400
    feature_hop_length: int = 160
    feature_fft_size: int = 512
    feature_preemphasis: float = 0.97
    feature_mel_floor: float = 1.192092955078125e-7
    max_new_tokens: int = 256
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ValueError("`variant` must be a non-empty string.")
        variant = self.variant.strip().lower().replace("_", "-")
        if variant not in _VARIANTS:
            raise ValueError("`variant` must be 'seamless-m4t-v2-large' or 'custom'.")
        object.__setattr__(self, "variant", variant)
        for name in (
                "vocab_size",
                "hidden_size",
                "feature_projection_input_dim",
                "speech_encoder_layers",
                "speech_encoder_attention_heads",
                "speech_encoder_intermediate_size",
                "speech_encoder_chunk_size",
                "conv_depthwise_kernel_size",
                "left_max_position_embeddings",
                "right_max_position_embeddings",
                "num_adapter_layers",
                "adaptor_kernel_size",
                "adaptor_stride",
                "decoder_layers",
                "decoder_attention_heads",
                "decoder_ffn_dim",
                "max_position_embeddings",
                "sampling_rate",
                "num_mel_bins",
                "feature_stride",
                "feature_window_length",
                "feature_hop_length",
                "feature_fft_size",
                "max_new_tokens",
        ):
            object.__setattr__(
                self,
                name,
                _integer(name, getattr(self, name), minimum=1),
            )
        object.__setattr__(
            self,
            "speech_encoder_left_chunk_num",
            _integer(
                "speech_encoder_left_chunk_num",
                self.speech_encoder_left_chunk_num,
                minimum=0,
            ),
        )
        for name in (
                "pad_token_id",
                "bos_token_id",
                "eos_token_id",
                "decoder_start_token_id",
        ):
            object.__setattr__(
                self,
                name,
                _integer(name, getattr(self, name), minimum=0),
            )
            if getattr(self, name) >= self.vocab_size:
                raise ValueError(f"`{name}` must be smaller than `vocab_size`.")
        if self.hidden_size % self.speech_encoder_attention_heads:
            raise ValueError("`hidden_size` must be divisible by "
                             "`speech_encoder_attention_heads`.")
        if self.hidden_size % self.decoder_attention_heads:
            raise ValueError("`hidden_size` must be divisible by "
                             "`decoder_attention_heads`.")
        if self.conv_depthwise_kernel_size % 2 != 1:
            raise ValueError("`conv_depthwise_kernel_size` must be odd.")
        if self.feature_fft_size < self.feature_window_length:
            raise ValueError("`feature_fft_size` must be at least `feature_window_length`.")
        if self.feature_projection_input_dim != self.num_mel_bins * self.feature_stride:
            raise ValueError("`feature_projection_input_dim` must equal "
                             "`num_mel_bins * feature_stride`.")
        if self.position_embeddings_type != "relative_key":
            raise ValueError("Native SeamlessM4T-v2 implements relative-key speech attention.")
        if self.speech_encoder_hidden_act not in {"swish", "silu"}:
            raise ValueError("Speech encoder activation must be swish/SiLU.")
        if self.activation_function != "relu":
            raise ValueError("The native S2T decoder implements ReLU.")
        for name in (
                "speech_encoder_dropout",
                "speech_encoder_layerdrop",
                "adaptor_dropout",
                "decoder_layerdrop",
                "dropout",
                "attention_dropout",
                "activation_dropout",
        ):
            object.__setattr__(
                self,
                name,
                _probability(name, getattr(self, name)),
            )
        for name in (
                "initializer_range",
                "layer_norm_eps",
                "feature_preemphasis",
                "feature_mel_floor",
        ):
            object.__setattr__(
                self,
                name,
                _positive(name, getattr(self, name)),
            )
        if self.feature_preemphasis >= 1.0:
            raise ValueError("`feature_preemphasis` must be smaller than one.")
        for name in ("add_adapter", "scale_embedding"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.add_adapter is False and self.num_adapter_layers != 1:
            raise ValueError(
                "A custom adapter-free graph must retain the declarative "
                "`num_adapter_layers=1` default.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )
        if variant == _PUBLISHED_VARIANT:
            self._validate_published_graph()

    def _validate_published_graph(self) -> None:
        expected = {
            "vocab_size": 256_102,
            "hidden_size": 1_024,
            "feature_projection_input_dim": 160,
            "speech_encoder_layers": 24,
            "speech_encoder_attention_heads": 16,
            "speech_encoder_intermediate_size": 4_096,
            "speech_encoder_hidden_act": "swish",
            "speech_encoder_dropout": 0.0,
            "speech_encoder_layerdrop": 0.1,
            "speech_encoder_chunk_size": 20_000,
            "speech_encoder_left_chunk_num": 128,
            "conv_depthwise_kernel_size": 31,
            "position_embeddings_type": "relative_key",
            "left_max_position_embeddings": 64,
            "right_max_position_embeddings": 8,
            "add_adapter": True,
            "num_adapter_layers": 1,
            "adaptor_kernel_size": 8,
            "adaptor_stride": 8,
            "adaptor_dropout": 0.1,
            "decoder_layers": 24,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": 8_192,
            "decoder_layerdrop": 0.05,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "activation_function": "relu",
            "max_position_embeddings": 4_096,
            "scale_embedding": True,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-5,
            "pad_token_id": 0,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "decoder_start_token_id": 3,
            "sampling_rate": 16_000,
            "num_mel_bins": 80,
            "feature_stride": 2,
            "feature_window_length": 400,
            "feature_hop_length": 160,
            "feature_fft_size": 512,
            "feature_preemphasis": 0.97,
            "feature_mel_floor": 1.192092955078125e-7,
            "max_new_tokens": 256,
        }
        changed = [name for name, value in expected.items() if getattr(self, name) != value]
        if changed:
            raise ValueError(
                "The published SeamlessM4T-v2-large S2T graph is immutable; "
                f"changed field(s): {', '.join(changed)}. Use "
                "`variant='custom'` only for VoiceHub-owned artifacts.")

    @property
    def speech_head_dimension(self) -> int:
        return self.hidden_size // self.speech_encoder_attention_heads

    @property
    def decoder_head_dimension(self) -> int:
        return self.hidden_size // self.decoder_attention_heads

    def adapter_output_lengths(self, lengths):
        """Return adapter lengths using the released Conv1d geometry."""
        result = lengths
        padding = self.adaptor_stride // 2
        for _ in range(self.num_adapter_layers if self.add_adapter else 0):
            result = (result + 2 * padding - self.adaptor_kernel_size) // self.adaptor_stride + 1
        return result

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> SeamlessM4Tv2S2TConfig:
        if not isinstance(values, Mapping):
            raise TypeError("SeamlessM4T-v2 configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        known = {item.name for item in fields(cls)}
        resolved = {name: source[name] for name in known - {"extra_config"} if name in source}
        extras = {name: value for name, value in source.items() if name not in known}
        supplied = source.get("extra_config")
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("`extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied)))
        # Official config files predate VoiceHub's explicit variant marker.
        if "variant" not in resolved:
            marker = source.get("voicehub_checkpoint_format")
            resolved["variant"] = (
                "custom" if marker and marker != "native-seamless-m4t-v2-s2t-v1" else _PUBLISHED_VARIANT)
        return cls(**resolved, extra_config=extras)

    @classmethod
    def coerce(
        cls,
        value: SeamlessM4Tv2S2TConfig | Mapping[str, Any],
    ) -> SeamlessM4Tv2S2TConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name != "extra_config":
                result[item.name] = getattr(self, item.name)
        result.update({
            'architectures': ["SeamlessM4Tv2ForSpeechToText"],
            "is_encoder_decoder": True,
            "model_type": "seamless_m4t_v2",
            "torch_dtype": "float32",
            "use_cache": False,
            "voicehub_checkpoint_format": "native-seamless-m4t-v2-s2t-v1",
        })
        return result


__all__ = ["SeamlessM4Tv2S2TConfig"]
