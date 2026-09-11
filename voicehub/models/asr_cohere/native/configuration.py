"""Validated configuration for VoiceHub's native Cohere Transcribe graph.

The published checkpoint uses a 48-layer Parakeet-compatible
FastConformer encoder and an eight-layer autoregressive cross-attention
decoder.  This module accepts both the original Cohere repository schema
and VoiceHub's normalized schema without importing Transformers or NeMo.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

from voicehub.models.asr_parakeet_tdt.native.configuration import ParakeetEncoderConfig

SUPPORTED_LANGUAGES = (
    "ar",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "vi",
    "zh",
)


def _integer(name: str, value: int, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}; found {value}.")


def _finite(name: str, value: float, *, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        qualifier = "finite" if minimum is None else f"finite and at least {minimum}"
        raise ValueError(f"`{name}` must be {qualifier}.")


def _legacy_encoder(values: Mapping[str, Any]) -> ParakeetEncoderConfig:
    """Translate Cohere's exported NeMo-style encoder dictionary."""
    hidden_size = int(values.get("d_model", 1280))
    dropout = float(values.get("dropout", 0.0))
    return ParakeetEncoderConfig(
        hidden_size=hidden_size,
        num_hidden_layers=int(values.get("n_layers", 48)),
        num_attention_heads=int(values.get("n_heads", 8)),
        num_key_value_heads=int(values.get("n_heads", 8)),
        intermediate_size=hidden_size * int(values.get("ff_expansion_factor", 4)),
        hidden_act="silu",
        attention_bias=True,
        convolution_bias=True,
        conv_kernel_size=int(values.get("conv_kernel_size", 9)),
        subsampling_factor=int(values.get("subsampling_factor", 8)),
        subsampling_conv_channels=int(values.get("subsampling_conv_channels", 256)),
        num_mel_bins=int(values.get("feat_in", 128)),
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=dropout,
        dropout_positions=float(values.get("dropout_emb", 0.0)),
        layerdrop=0.0,
        activation_dropout=dropout,
        attention_dropout=float(values.get("dropout_att", dropout)),
        max_position_embeddings=int(values.get("pos_emb_max_len", 5000)),
        scale_input=bool(values.get("xscaling", False)),
        initializer_range=0.02,
    )


def _official_encoder() -> ParakeetEncoderConfig:
    return ParakeetEncoderConfig(
        hidden_size=1280,
        num_hidden_layers=48,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=5120,
        hidden_act="silu",
        attention_bias=True,
        convolution_bias=True,
        conv_kernel_size=9,
        subsampling_factor=8,
        subsampling_conv_channels=256,
        num_mel_bins=128,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=5000,
        scale_input=False,
        initializer_range=0.02,
    )


@dataclass(frozen=True, slots=True)
class CohereAsrConfig:
    """Complete native Cohere Transcribe configuration."""

    encoder_config: ParakeetEncoderConfig | Mapping[str, Any] = field(default_factory=_official_encoder)
    vocab_size: int = 16_384
    decoder_hidden_size: int = 1024
    decoder_num_hidden_layers: int = 8
    decoder_num_attention_heads: int = 8
    decoder_intermediate_size: int = 4096
    decoder_hidden_act: str = "relu"
    decoder_max_position_embeddings: int = 1024
    attention_dropout: float = 0.0
    pad_token_id: int = 2
    eos_token_id: int = 3
    bos_token_id: int = 4
    decoder_start_token_id: int = 13_764
    sample_rate: int = 16_000
    hop_length: int = 160
    n_fft: int = 512
    win_length: int = 400
    preemphasis: float = 0.97
    dither: float = 1e-5
    max_audio_clip_s: float = 35.0
    overlap_chunk_second: float = 5.0
    min_energy_window_samples: int = 1600
    log_softmax: bool = True
    mask_prompt_loss: bool = False
    is_encoder_decoder: bool = True
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        encoder = ParakeetEncoderConfig.coerce(self.encoder_config)
        object.__setattr__(self, "encoder_config", encoder)
        for name in (
                "vocab_size",
                "decoder_hidden_size",
                "decoder_num_hidden_layers",
                "decoder_num_attention_heads",
                "decoder_intermediate_size",
                "decoder_max_position_embeddings",
                "sample_rate",
                "hop_length",
                "n_fft",
                "win_length",
                "min_energy_window_samples",
        ):
            _integer(name, getattr(self, name), minimum=1)
        for name in (
                "pad_token_id",
                "eos_token_id",
                "bos_token_id",
                "decoder_start_token_id",
        ):
            value = getattr(self, name)
            _integer(name, value)
            if value >= self.vocab_size:
                raise ValueError(f"`{name}` must be smaller than `vocab_size`.")
        if len({
                self.pad_token_id,
                self.eos_token_id,
                self.bos_token_id,
        }) != 3:
            raise ValueError("Cohere ASR pad, EOS, and BOS token IDs must differ.")
        if self.decoder_hidden_size % self.decoder_num_attention_heads:
            raise ValueError("`decoder_hidden_size` must be divisible by "
                             "`decoder_num_attention_heads`.")
        if self.decoder_hidden_act not in {"relu", "silu"}:
            raise ValueError("Cohere ASR decoder activation must be 'relu' or 'silu'.")
        if self.sample_rate != 16_000:
            raise ValueError("Cohere Transcribe requires 16 kHz audio.")
        if self.win_length > self.n_fft:
            raise ValueError("`win_length` cannot exceed `n_fft`.")
        for name, minimum in (
            ("preemphasis", 0.0),
            ("dither", 0.0),
            ("max_audio_clip_s", 0.0),
            ("overlap_chunk_second", 0.0),
            ("attention_dropout", 0.0),
        ):
            _finite(name, getattr(self, name), minimum=minimum)
        if not 0.0 <= self.preemphasis < 1.0:
            raise ValueError("`preemphasis` must be in [0, 1).")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("`attention_dropout` must be in [0, 1).")
        if self.max_audio_clip_s <= 0.0:
            raise ValueError("`max_audio_clip_s` must be positive.")
        if self.overlap_chunk_second >= self.max_audio_clip_s:
            raise ValueError("`overlap_chunk_second` must be smaller than "
                             "`max_audio_clip_s`.")
        for name in (
                "log_softmax",
                "mask_prompt_loss",
                "is_encoder_decoder",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if not self.is_encoder_decoder:
            raise ValueError("Cohere ASR must be configured as encoder-decoder.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @property
    def decoder_head_dim(self) -> int:
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> CohereAsrConfig:
        """Parse the official remote-code or normalized VoiceHub schema."""
        if not isinstance(values, Mapping):
            raise TypeError("Cohere ASR configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        if str(source.get("model_type", "cohere_asr")).lower() != "cohere_asr":
            raise ValueError("Native Cohere ASR requires `model_type='cohere_asr'`.")
        architectures = source.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        allowed_architectures = {
            "CohereAsrForConditionalGeneration",
            "CohereForSpeechRecognition",
        }
        if (architectures and
            (not isinstance(architectures, Sequence) or not any(str(value) in allowed_architectures
                                                                for value in architectures))):
            raise ValueError("Native Cohere ASR requires a Cohere conditional-generation "
                             "checkpoint.")

        normalized: dict[str, Any] = {}
        raw_encoder = source.get("encoder_config")
        if raw_encoder is None:
            raw_encoder = source.get("encoder")
            if isinstance(raw_encoder, Mapping):
                normalized["encoder_config"] = _legacy_encoder(raw_encoder)
        elif isinstance(raw_encoder, Mapping):
            normalized["encoder_config"] = ParakeetEncoderConfig.from_dict(raw_encoder)
        elif isinstance(raw_encoder, ParakeetEncoderConfig):
            normalized["encoder_config"] = raw_encoder
        else:
            raise TypeError("Cohere ASR encoder configuration must be a mapping.")

        decoder = source.get("transf_decoder")
        decoder_values: Mapping[str, Any] = {}
        if isinstance(decoder, Mapping):
            candidate = decoder.get("config_dict", decoder)
            if isinstance(candidate, Mapping):
                decoder_values = candidate
        mapping = {
            "decoder_hidden_size": ("hidden_size", 1024),
            "decoder_num_hidden_layers": ("num_layers", 8),
            "decoder_num_attention_heads": ("num_attention_heads", 8),
            "decoder_intermediate_size": ("inner_size", 4096),
            "decoder_hidden_act": ("hidden_act", "relu"),
            "decoder_max_position_embeddings": ("max_sequence_length", 1024),
        }
        for target, (legacy, default) in mapping.items():
            normalized[target] = source.get(
                target,
                decoder_values.get(legacy, default),
            )

        head = source.get("head")
        head_values = head if isinstance(head, Mapping) else {}
        preprocessor = source.get("preprocessor")
        preprocessor_values = (preprocessor if isinstance(preprocessor, Mapping) else {})
        normalized.update({
            "vocab_size":
            source.get("vocab_size", head_values.get("num_classes", 16_384)),
            "attention_dropout":
            source.get("attention_dropout", 0.0),
            "pad_token_id":
            source.get("pad_token_id", 2),
            "eos_token_id":
            source.get("eos_token_id", 3),
            "bos_token_id":
            source.get("bos_token_id", 4),
            "decoder_start_token_id":
            source.get("decoder_start_token_id", 13_764),
            "sample_rate":
            source.get("sample_rate", preprocessor_values.get("sample_rate", 16_000)),
            "hop_length":
            source.get(
                "hop_length",
                int(
                    float(preprocessor_values.get("window_stride", 0.01)) *
                    int(preprocessor_values.get("sample_rate", 16_000))),
            ),
            "n_fft":
            source.get("n_fft", preprocessor_values.get("n_fft", 512)),
            "win_length":
            source.get(
                "win_length",
                int(
                    float(preprocessor_values.get("window_size", 0.025)) *
                    int(preprocessor_values.get("sample_rate", 16_000))),
            ),
            "preemphasis":
            source.get("preemphasis", 0.97),
            "dither":
            source.get("dither", preprocessor_values.get("dither", 1e-5)),
            "max_audio_clip_s":
            source.get("max_audio_clip_s", 35.0),
            "overlap_chunk_second":
            source.get("overlap_chunk_second", 5.0),
            "min_energy_window_samples":
            source.get("min_energy_window_samples", 1600),
            "log_softmax":
            source.get("log_softmax", head_values.get("log_softmax", True)),
            "mask_prompt_loss":
            source.get(
                "mask_prompt_loss",
                source.get("use_loss_mask_for_prompt", False),
            ),
            "is_encoder_decoder":
            source.get("is_encoder_decoder", True),
        })
        field_names = {item.name for item in fields(cls) if item.name != "extra_config"}
        for name in field_names:
            if name in source and name not in normalized:
                normalized[name] = source[name]
        consumed = {
            'architectures',
            "auto_map",
            "encoder",
            "encoder_config",
            "head",
            "model_type",
            "preprocessor",
            "transf_decoder",
            "extra_config",
        } | field_names
        extras = {name: value for name, value in source.items() if name not in consumed}
        supplied_extras = source.get("extra_config")
        if supplied_extras is not None:
            if not isinstance(supplied_extras, Mapping):
                raise TypeError("Cohere ASR `extra_config` must be a mapping.")
            extras.update(copy.deepcopy(dict(supplied_extras)))
        normalized["extra_config"] = extras
        return cls(**normalized)

    @classmethod
    def coerce(
        cls,
        value: CohereAsrConfig | Mapping[str, Any],
    ) -> CohereAsrConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        for item in fields(self):
            if item.name == "extra_config":
                continue
            value = getattr(self, item.name)
            result[item.name] = (value.to_dict() if isinstance(value, ParakeetEncoderConfig) else value)
        result['architectures'] = ["CohereAsrForConditionalGeneration"]
        result["model_type"] = "cohere_asr"
        return result


__all__ = ["CohereAsrConfig", "SUPPORTED_LANGUAGES"]
