"""Validated configuration for VoiceHub-native CosyVoice 3.

The default values reproduce the executable graph declared by
``Fun-CosyVoice3-0.5B-2512/cosyvoice3.yaml`` at immutable revision
``29e01c4e8d000f4bcd70751be16fa94bf3d85a18``.  The generation field is
explicit, and rejected unless it is 3, so future family implementations cannot
silently load a different graph through this configuration.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from voicehub.models.causal_lm.native.configuration import Qwen2Config


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be positive.")
    return value


def _positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"`{name}` must be finite and positive.")
    return value


@dataclass(frozen=True, slots=True)
class CosyVoiceLanguageConfig:
    """Qwen2 speech-token LM used by the audited CosyVoice 3 checkpoint."""

    text_vocab_size: int = 151_936
    speech_vocab_size: int = 6_561
    control_token_count: int = 200
    hidden_size: int = 896
    intermediate_size: int = 4_864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32_768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    mix_ratio: tuple[int, int] = (5, 15)
    label_smoothing: float = 0.0
    length_normalized_loss: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mix_ratio", tuple(self.mix_ratio))
        for name in (
                "text_vocab_size",
                "speech_vocab_size",
                "control_token_count",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "max_position_embeddings",
        ):
            _positive_integer(name, getattr(self, name))
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("`num_key_value_heads` must divide `num_attention_heads`.")
        if len(self.mix_ratio) != 2 or any(value <= 0 for value in self.mix_ratio):
            raise ValueError("`mix_ratio` must contain two positive integers.")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("`label_smoothing` must be in [0, 1).")
        _positive_float("rope_theta", self.rope_theta)
        _positive_float("rms_norm_eps", self.rms_norm_eps)

    @property
    def output_vocab_size(self) -> int:
        return self.speech_vocab_size + self.control_token_count

    @property
    def sos_token_id(self) -> int:
        return self.speech_vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.speech_vocab_size + 1

    @property
    def task_token_id(self) -> int:
        return self.speech_vocab_size + 2

    @property
    def fill_token_id(self) -> int:
        return self.speech_vocab_size + 3

    def qwen_config(self) -> Qwen2Config:
        return Qwen2Config(
            vocab_size=self.text_vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
            tie_word_embeddings=True,
            use_cache=True,
            use_sliding_window=False,
            sliding_window=None,
            max_window_layers=self.num_hidden_layers,
            bos_token_id=151_643 if self.text_vocab_size > 151_645 else None,
            eos_token_id=151_645 if self.text_vocab_size > 151_645 else None,
            pad_token_id=151_643 if self.text_vocab_size > 151_645 else None,
        )


@dataclass(frozen=True, slots=True)
class CosyVoiceFlowConfig:
    """Causal conditional-flow matcher and DiT estimator."""

    mel_channels: int = 80
    speaker_embedding_dim: int = 192
    speech_vocab_size: int = 6_561
    token_frame_rate: int = 25
    token_mel_ratio: int = 2
    lookahead_frames: int = 3
    lookahead_hidden_size: int = 1_024
    model_dim: int = 1_024
    depth: int = 22
    heads: int = 16
    head_dim: int = 64
    feed_forward_multiplier: int = 2
    static_chunk_size: int = 50
    num_decoding_left_chunks: int = -1
    sigma_min: float = 1e-6
    training_cfg_rate: float = 0.2
    inference_cfg_rate: float = 0.7

    def __post_init__(self) -> None:
        for name in (
                "mel_channels",
                "speaker_embedding_dim",
                "speech_vocab_size",
                "token_frame_rate",
                "token_mel_ratio",
                "lookahead_frames",
                "lookahead_hidden_size",
                "model_dim",
                "depth",
                "heads",
                "head_dim",
                "feed_forward_multiplier",
                "static_chunk_size",
        ):
            _positive_integer(name, getattr(self, name))
        if self.model_dim != self.heads * self.head_dim:
            raise ValueError("Flow `heads * head_dim` must equal `model_dim`.")
        for name in ("sigma_min", ):
            _positive_float(name, getattr(self, name))
        for name in ("training_cfg_rate", "inference_cfg_rate"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"`{name}` must be in [0, 1].")


@dataclass(frozen=True, slots=True)
class CosyVoiceHiFTConfig:
    """Causal HiFT waveform generator configuration."""

    mel_channels: int = 80
    base_channels: int = 512
    harmonics: int = 8
    sample_rate: int = 24_000
    upsample_rates: tuple[int, ...] = (8, 5, 3)
    upsample_kernel_sizes: tuple[int, ...] = (16, 11, 7)
    istft_n_fft: int = 16
    istft_hop_length: int = 4
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilations: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    source_resblock_kernel_sizes: tuple[int, ...] = (7, 7, 11)
    source_resblock_dilations: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    f0_hidden_size: int = 512
    conv_pre_look_right: int = 4
    audio_limit: float = 0.99

    def __post_init__(self) -> None:
        for name in (
                "upsample_rates",
                "upsample_kernel_sizes",
                "resblock_kernel_sizes",
                "source_resblock_kernel_sizes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("resblock_dilations", "source_resblock_dilations"):
            object.__setattr__(
                self,
                name,
                tuple(tuple(values) for values in getattr(self, name)),
            )
        for name in (
                "mel_channels",
                "base_channels",
                "harmonics",
                "sample_rate",
                "istft_n_fft",
                "istft_hop_length",
                "f0_hidden_size",
                "conv_pre_look_right",
        ):
            _positive_integer(name, getattr(self, name))
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("HiFT upsample rate/kernel lists must have equal length.")
        if len(self.source_resblock_kernel_sizes) != len(self.upsample_rates):
            raise ValueError("HiFT requires one source residual block per upsample stage.")
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilations):
            raise ValueError("HiFT residual kernel/dilation lists must have equal length.")
        if not 0 < self.audio_limit <= 1:
            raise ValueError("`audio_limit` must be in (0, 1].")

    @property
    def samples_per_mel_frame(self) -> int:
        result = self.istft_hop_length
        for rate in self.upsample_rates:
            result *= rate
        return result


@dataclass(frozen=True, slots=True)
class CosyVoiceArchitectureConfig:
    """Complete trainable CosyVoice graph."""

    generation: int = 3
    sample_rate: int = 24_000
    language: CosyVoiceLanguageConfig = field(default_factory=CosyVoiceLanguageConfig)
    flow: CosyVoiceFlowConfig = field(default_factory=CosyVoiceFlowConfig)
    hift: CosyVoiceHiFTConfig = field(default_factory=CosyVoiceHiFTConfig)
    extra_config: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.generation != 3:
            raise ValueError(
                "This native graph supports CosyVoice 3 only; CosyVoice 1 and "
                "2 have different executable checkpoint graphs.")
        _positive_integer("sample_rate", self.sample_rate)
        if self.sample_rate != self.hift.sample_rate:
            raise ValueError("Top-level and HiFT sample rates must match.")
        if self.language.speech_vocab_size != self.flow.speech_vocab_size:
            raise ValueError("LM and flow speech vocabularies must match.")
        if self.flow.mel_channels != self.hift.mel_channels:
            raise ValueError("Flow and HiFT mel channel counts must match.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(self, "extra_config", copy.deepcopy(dict(self.extra_config)))

    @classmethod
    def tiny(cls) -> CosyVoiceArchitectureConfig:
        language = CosyVoiceLanguageConfig(
            text_vocab_size=320,
            speech_vocab_size=32,
            control_token_count=8,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
        flow = CosyVoiceFlowConfig(
            mel_channels=8,
            speaker_embedding_dim=4,
            speech_vocab_size=32,
            lookahead_hidden_size=16,
            model_dim=32,
            depth=2,
            heads=4,
            head_dim=8,
            static_chunk_size=8,
        )
        hift = CosyVoiceHiFTConfig(
            mel_channels=8,
            base_channels=32,
            harmonics=2,
            upsample_rates=(2, 2),
            upsample_kernel_sizes=(4, 4),
            resblock_kernel_sizes=(3, ),
            resblock_dilations=((1, 2), ),
            source_resblock_kernel_sizes=(3, 3),
            source_resblock_dilations=((1, 2), (1, 2)),
            f0_hidden_size=16,
            conv_pre_look_right=2,
        )
        return cls(language=language, flow=flow, hift=hift)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> CosyVoiceArchitectureConfig:
        if not isinstance(values, Mapping):
            raise TypeError("CosyVoice configuration must be a mapping.")
        values = copy.deepcopy(dict(values))
        model_type = values.pop("model_type", "cosyvoice")
        architecture = values.pop("architecture", "cosyvoice_native")
        if model_type != "cosyvoice":
            raise ValueError(f"Expected CosyVoice `model_type`, found {model_type!r}.")
        if architecture != "cosyvoice_native":
            raise ValueError("Expected VoiceHub-native CosyVoice architecture, found "
                             f"{architecture!r}.")
        for name, target in (
            ("language", CosyVoiceLanguageConfig),
            ("flow", CosyVoiceFlowConfig),
            ("hift", CosyVoiceHiFTConfig),
        ):
            if name in values and not isinstance(values[name], target):
                values[name] = target(**values[name])
        known = {"generation", "sample_rate", "language", "flow", "hift", "extra_config"}
        extras = {name: values.pop(name) for name in tuple(values) if name not in known}
        extras.update(values.pop("extra_config", {}))
        return cls(**values, extra_config=extras)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["extra_config"] = copy.deepcopy(dict(self.extra_config))
        result.update({
            "architecture": "cosyvoice_native",
            "model_type": "cosyvoice",
        })
        return result


__all__ = [
    "CosyVoiceArchitectureConfig",
    "CosyVoiceFlowConfig",
    "CosyVoiceHiFTConfig",
    "CosyVoiceLanguageConfig",
]
