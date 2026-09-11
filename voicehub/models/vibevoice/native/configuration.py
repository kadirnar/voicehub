"""Validated configurations for the published Microsoft VibeVoice family.

The classes in this module parse the public ``config.json`` files
directly. They intentionally keep the ASR, non-streaming TTS, and
realtime TTS graphs separate: those releases use different speech
tokenizers and different decoder topologies despite sharing a model-
family name.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from voicehub.models.causal_lm.native import Qwen2Config


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a mapping.")
    return copy.deepcopy(dict(value))


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _nonnegative_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return result


def _positive_number(value: Any, *, name: str) -> float:
    result = _nonnegative_number(value, name=name)
    if result == 0.0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return result


def _integer_sequence(
    value: Any,
    *,
    name: str,
    expected_length: int | None = None,
) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            values = tuple(int(item) for item in value.split("-"))
        except ValueError as error:
            raise ValueError(f"`{name}` is not a dash-separated integer list.") from error
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
    else:
        raise TypeError(f"`{name}` must be a sequence of integers.")
    if not values or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in values):
        raise ValueError(f"`{name}` must contain positive integers.")
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(f"`{name}` must contain {expected_length} values; found {len(values)}.")
    return values


@dataclass(frozen=True, slots=True)
class VibeVoiceASRTokenizerConfig:
    """Continuous causal speech encoder used by the ASR-HF checkpoint."""

    channels: int = 1
    hidden_size: int = 64
    kernel_size: int = 7
    rms_norm_eps: float = 1e-5
    layer_scale_init_value: float = 1e-6
    initializer_range: float = 1e-2
    num_filters: int = 32
    downsampling_ratios: tuple[int, ...] = (2, 2, 4, 5, 5, 8)
    depths: tuple[int, ...] = (3, 3, 3, 3, 3, 3, 8)
    hidden_act: str = "gelu"
    ffn_expansion: int = 4
    vae_std: float = 0.625
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in (
                "channels",
                "hidden_size",
                "kernel_size",
                "num_filters",
                "ffn_expansion",
        ):
            _positive_integer(getattr(self, name), name=name)
        ratios = _integer_sequence(
            self.downsampling_ratios,
            name="downsampling_ratios",
        )
        depths = _integer_sequence(
            self.depths,
            name="depths",
            expected_length=len(ratios) + 1,
        )
        object.__setattr__(self, "downsampling_ratios", ratios)
        object.__setattr__(self, "depths", depths)
        _positive_number(self.rms_norm_eps, name="rms_norm_eps")
        _nonnegative_number(
            self.layer_scale_init_value,
            name="layer_scale_init_value",
        )
        _positive_number(self.initializer_range, name="initializer_range")
        _nonnegative_number(self.vae_std, name="vae_std")
        if self.hidden_act != "gelu":
            raise ValueError("Published VibeVoice speech encoders require GELU.")
        if not isinstance(self.extra, Mapping):
            raise TypeError("`extra` must be a mapping.")
        object.__setattr__(self, "extra", copy.deepcopy(dict(self.extra)))

    @property
    def hop_length(self) -> int:
        return math.prod(self.downsampling_ratios)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> VibeVoiceASRTokenizerConfig:
        source = _mapping(values, name="speech tokenizer configuration")
        known_names = set(cls.__dataclass_fields__) - {"extra"}
        known = {name: source.pop(name) for name in tuple(source) if name in known_names}
        for name in ("downsampling_ratios", "depths"):
            if name in known:
                known[name] = tuple(known[name])
        return cls(**known, extra=source)

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        extra = output.pop("extra")
        output["downsampling_ratios"] = list(output["downsampling_ratios"])
        output["depths"] = list(output["depths"])
        output.update(extra)
        return output


@dataclass(frozen=True, slots=True)
class VibeVoiceLegacyTokenizerConfig:
    """Speech codec schema used by the 1.5B and realtime TTS releases."""

    channels: int = 1
    corpus_normalize: float = 0.0
    causal: bool = True
    vae_dim: int = 64
    fix_std: float = 0.5
    std_dist_type: str = "gaussian"
    mixer_layer: str = "depthwise_conv"
    conv_norm: str = "none"
    pad_mode: str = "constant"
    disable_last_norm: bool = True
    layernorm: str = "RMSNorm"
    layernorm_eps: float = 1e-5
    layernorm_elementwise_affine: bool = True
    conv_bias: bool = True
    layer_scale_init_value: float = 1e-6
    weight_init_value: float = 1e-2
    encoder_n_filters: int = 32
    encoder_ratios: tuple[int, ...] = (8, 5, 5, 4, 2, 2)
    encoder_depths: tuple[int, ...] = (3, 3, 3, 3, 3, 3, 8)
    decoder_n_filters: int = 32
    decoder_ratios: tuple[int, ...] | None = None
    decoder_depths: tuple[int, ...] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in (
                "channels",
                "vae_dim",
                "encoder_n_filters",
                "decoder_n_filters",
        ):
            _positive_integer(getattr(self, name), name=name)
        ratios = _integer_sequence(self.encoder_ratios, name="encoder_ratios")
        depths = _integer_sequence(
            self.encoder_depths,
            name="encoder_depths",
            expected_length=len(ratios) + 1,
        )
        decoder_ratios = (
            ratios if self.decoder_ratios is None else _integer_sequence(
                self.decoder_ratios,
                name="decoder_ratios",
                expected_length=len(ratios),
            ))
        decoder_depths = (
            tuple(reversed(depths)) if self.decoder_depths is None else _integer_sequence(
                self.decoder_depths,
                name="decoder_depths",
                expected_length=len(depths),
            ))
        object.__setattr__(self, "encoder_ratios", ratios)
        object.__setattr__(self, "encoder_depths", depths)
        object.__setattr__(self, "decoder_ratios", decoder_ratios)
        object.__setattr__(self, "decoder_depths", decoder_depths)
        for name in (
                "causal",
                "disable_last_norm",
                "layernorm_elementwise_affine",
                "conv_bias",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if not self.causal:
            raise ValueError("Published VibeVoice TTS codecs are causal.")
        if self.mixer_layer != "depthwise_conv":
            raise ValueError("Published VibeVoice TTS codecs use depthwise convolutions.")
        if self.conv_norm != "none":
            raise ValueError("Published VibeVoice TTS checkpoints use unnormalized convolutions.")
        if self.pad_mode != "constant":
            raise ValueError("Published VibeVoice TTS checkpoints use constant padding.")
        if self.layernorm != "RMSNorm":
            raise ValueError("Published VibeVoice TTS checkpoints require RMSNorm.")
        if self.std_dist_type not in {"gaussian", "none"}:
            raise ValueError("`std_dist_type` must be 'gaussian' or 'none'.")
        _nonnegative_number(self.corpus_normalize, name="corpus_normalize")
        _nonnegative_number(self.fix_std, name="fix_std")
        _positive_number(self.layernorm_eps, name="layernorm_eps")
        _nonnegative_number(
            self.layer_scale_init_value,
            name="layer_scale_init_value",
        )
        _positive_number(self.weight_init_value, name="weight_init_value")
        if not isinstance(self.extra, Mapping):
            raise TypeError("`extra` must be a mapping.")
        object.__setattr__(self, "extra", copy.deepcopy(dict(self.extra)))

    @property
    def hop_length(self) -> int:
        return math.prod(self.encoder_ratios)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> VibeVoiceLegacyTokenizerConfig:
        source = _mapping(values, name="legacy speech tokenizer configuration")
        known_names = set(cls.__dataclass_fields__) - {"extra"}
        known = {name: source.pop(name) for name in tuple(source) if name in known_names}
        for name in (
                "encoder_ratios",
                "decoder_ratios",
                "encoder_depths",
                "decoder_depths",
        ):
            value = known.get(name)
            if value is not None:
                known[name] = _integer_sequence(value, name=name)
        return cls(**known, extra=source)

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        extra = output.pop("extra")
        output["encoder_ratios"] = list(output["encoder_ratios"])
        output["encoder_depths"] = "-".join(str(item) for item in output["encoder_depths"])
        output["decoder_ratios"] = list(output["decoder_ratios"])
        # ``None`` in published configs means reversed encoder depths.
        output["decoder_depths"] = "-".join(str(item) for item in output["decoder_depths"])
        output.update(extra)
        return output


@dataclass(frozen=True, slots=True)
class VibeVoiceDiffusionConfig:
    hidden_size: int = 768
    head_layers: int = 4
    head_ffn_ratio: float = 3.0
    rms_norm_eps: float = 1e-5
    latent_size: int = 64
    speech_vae_dim: int | None = None
    prediction_type: str = "v_prediction"
    diffusion_type: str = "ddpm"
    ddpm_num_steps: int = 1_000
    ddpm_num_inference_steps: int = 20
    ddpm_beta_schedule: str = "cosine"
    ddpm_batch_mul: int = 4
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in (
                "hidden_size",
                "head_layers",
                "latent_size",
                "ddpm_num_steps",
                "ddpm_num_inference_steps",
                "ddpm_batch_mul",
        ):
            _positive_integer(getattr(self, name), name=name)
        if self.speech_vae_dim is not None:
            _positive_integer(self.speech_vae_dim, name="speech_vae_dim")
        _positive_number(self.head_ffn_ratio, name="head_ffn_ratio")
        _positive_number(self.rms_norm_eps, name="rms_norm_eps")
        if self.prediction_type not in {"epsilon", "v_prediction"}:
            raise ValueError("VibeVoice diffusion predicts epsilon or velocity.")
        if self.diffusion_type != "ddpm" or self.ddpm_beta_schedule != "cosine":
            raise ValueError("Published VibeVoice checkpoints require cosine DDPM diffusion.")
        if not isinstance(self.extra, Mapping):
            raise TypeError("`extra` must be a mapping.")
        object.__setattr__(self, "extra", copy.deepcopy(dict(self.extra)))

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> VibeVoiceDiffusionConfig:
        source = _mapping(values, name="diffusion head configuration")
        known_names = set(cls.__dataclass_fields__) - {"extra"}
        known = {name: source.pop(name) for name in tuple(source) if name in known_names}
        return cls(**known, extra=source)

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        extra = output.pop("extra")
        output.update(extra)
        return output


@dataclass(frozen=True, slots=True)
class VibeVoiceASRConfig:
    acoustic_tokenizer_encoder_config: VibeVoiceASRTokenizerConfig
    semantic_tokenizer_encoder_config: VibeVoiceASRTokenizerConfig
    text_config: Qwen2Config
    acoustic_tokenizer_chunk_size: int = 1_440_000
    audio_bos_token_id: int = 151_646
    audio_eos_token_id: int = 151_647
    audio_token_id: int = 151_648
    model_type: str = "vibevoice_asr"
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "vibevoice_asr":
            raise ValueError("ASR configuration requires `model_type='vibevoice_asr'`.")
        if self.acoustic_tokenizer_encoder_config.hop_length != 3_200:
            raise ValueError("Published VibeVoice ASR uses a 3,200-sample hop.")
        if (self.semantic_tokenizer_encoder_config.hop_length
                != self.acoustic_tokenizer_encoder_config.hop_length):
            raise ValueError("ASR acoustic and semantic encoders must share one hop.")
        _positive_integer(
            self.acoustic_tokenizer_chunk_size,
            name="acoustic_tokenizer_chunk_size",
        )
        if (self.acoustic_tokenizer_chunk_size % self.acoustic_tokenizer_encoder_config.hop_length):
            raise ValueError("ASR chunk size must be an exact tokenizer-hop multiple.")
        for name in (
                "audio_bos_token_id",
                "audio_eos_token_id",
                "audio_token_id",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"`{name}` must be an integer.")
            if not 0 <= value < self.text_config.vocab_size:
                raise ValueError(f"`{name}` is outside the text vocabulary.")
        if (len({
                self.audio_bos_token_id,
                self.audio_eos_token_id,
                self.audio_token_id,
        }) != 3):
            raise ValueError("ASR audio control IDs must be distinct.")
        if not isinstance(self.extra, Mapping):
            raise TypeError("`extra` must be a mapping.")
        object.__setattr__(self, "extra", copy.deepcopy(dict(self.extra)))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> VibeVoiceASRConfig:
        source = _mapping(values, name="VibeVoice ASR configuration")
        acoustic = VibeVoiceASRTokenizerConfig.from_dict(source.pop("acoustic_tokenizer_encoder_config"))
        semantic = VibeVoiceASRTokenizerConfig.from_dict(source.pop("semantic_tokenizer_encoder_config"))
        text = Qwen2Config.from_dict(source.pop("text_config"))
        known_names = set(cls.__dataclass_fields__) - {
            "acoustic_tokenizer_encoder_config",
            "semantic_tokenizer_encoder_config",
            "text_config",
            "extra",
        }
        known = {name: source.pop(name) for name in tuple(source) if name in known_names}
        return cls(
            acoustic_tokenizer_encoder_config=acoustic,
            semantic_tokenizer_encoder_config=semantic,
            text_config=text,
            **known,
            extra=source,
        )

    def to_dict(self) -> dict[str, Any]:
        output = copy.deepcopy(dict(self.extra))
        output.update({
            "acoustic_tokenizer_chunk_size": self.acoustic_tokenizer_chunk_size,
            "acoustic_tokenizer_encoder_config": self.acoustic_tokenizer_encoder_config.to_dict(),
            "audio_bos_token_id": self.audio_bos_token_id,
            "audio_eos_token_id": self.audio_eos_token_id,
            "audio_token_id": self.audio_token_id,
            "model_type": self.model_type,
            "semantic_tokenizer_encoder_config": self.semantic_tokenizer_encoder_config.to_dict(),
            "text_config": self.text_config.to_dict(),
        })
        return output


@dataclass(frozen=True, slots=True)
class VibeVoiceTTSConfig:
    acoustic_tokenizer_config: VibeVoiceLegacyTokenizerConfig
    semantic_tokenizer_config: VibeVoiceLegacyTokenizerConfig | None
    decoder_config: Qwen2Config
    diffusion_head_config: VibeVoiceDiffusionConfig
    acoustic_vae_dim: int = 64
    semantic_vae_dim: int | None = 128
    tts_backbone_num_hidden_layers: int | None = None
    model_type: str = "vibevoice"
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type not in {"vibevoice", "vibevoice_streaming"}:
            raise ValueError("Unsupported VibeVoice TTS model type.")
        _positive_integer(self.acoustic_vae_dim, name="acoustic_vae_dim")
        if self.acoustic_vae_dim != self.acoustic_tokenizer_config.vae_dim:
            raise ValueError("Acoustic VAE dimensions disagree.")
        if self.diffusion_head_config.latent_size != self.acoustic_vae_dim:
            raise ValueError("Diffusion and acoustic latent dimensions disagree.")
        streaming = self.model_type == "vibevoice_streaming"
        if streaming:
            if self.semantic_tokenizer_config is not None:
                raise ValueError("Realtime VibeVoice does not contain a semantic tokenizer.")
            if self.semantic_vae_dim is not None:
                raise ValueError("Realtime VibeVoice has no semantic latent size.")
            if self.tts_backbone_num_hidden_layers is None:
                raise ValueError("Realtime VibeVoice requires its upper-layer count.")
            _positive_integer(
                self.tts_backbone_num_hidden_layers,
                name="tts_backbone_num_hidden_layers",
            )
            if (self.tts_backbone_num_hidden_layers >= self.decoder_config.num_hidden_layers):
                raise ValueError("Realtime TTS upper layers must leave at least one text layer.")
        else:
            if self.semantic_tokenizer_config is None:
                raise ValueError("VibeVoice 1.5B requires a semantic tokenizer.")
            if self.semantic_vae_dim is None:
                raise ValueError("VibeVoice 1.5B requires a semantic latent size.")
            _positive_integer(self.semantic_vae_dim, name="semantic_vae_dim")
            if self.semantic_vae_dim != self.semantic_tokenizer_config.vae_dim:
                raise ValueError("Semantic VAE dimensions disagree.")
            if self.tts_backbone_num_hidden_layers is not None:
                raise ValueError("Non-streaming VibeVoice does not split its decoder.")
        if not isinstance(self.extra, Mapping):
            raise TypeError("`extra` must be a mapping.")
        object.__setattr__(self, "extra", copy.deepcopy(dict(self.extra)))

    @property
    def is_streaming(self) -> bool:
        return self.model_type == "vibevoice_streaming"

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> VibeVoiceTTSConfig:
        source = _mapping(values, name="VibeVoice TTS configuration")
        model_type = str(source.get("model_type", ""))
        acoustic = VibeVoiceLegacyTokenizerConfig.from_dict(source.pop("acoustic_tokenizer_config"))
        raw_semantic = source.pop("semantic_tokenizer_config", None)
        semantic = (None if raw_semantic is None else VibeVoiceLegacyTokenizerConfig.from_dict(raw_semantic))
        decoder = Qwen2Config.from_dict(source.pop("decoder_config"))
        diffusion = VibeVoiceDiffusionConfig.from_dict(source.pop("diffusion_head_config"))
        source.pop("model_type", None)
        known_names = set(cls.__dataclass_fields__) - {
            "acoustic_tokenizer_config",
            "semantic_tokenizer_config",
            "decoder_config",
            "diffusion_head_config",
            "model_type",
            "extra",
        }
        known = {name: source.pop(name) for name in tuple(source) if name in known_names}
        if model_type == "vibevoice_streaming":
            known.setdefault("semantic_vae_dim", None)
        return cls(
            acoustic_tokenizer_config=acoustic,
            semantic_tokenizer_config=semantic,
            decoder_config=decoder,
            diffusion_head_config=diffusion,
            model_type=model_type,
            **known,
            extra=source,
        )

    def to_dict(self) -> dict[str, Any]:
        output = copy.deepcopy(dict(self.extra))
        output.update({
            "acoustic_tokenizer_config": self.acoustic_tokenizer_config.to_dict(),
            "acoustic_vae_dim": self.acoustic_vae_dim,
            "decoder_config": self.decoder_config.to_dict(),
            "diffusion_head_config": self.diffusion_head_config.to_dict(),
            "model_type": self.model_type,
        })
        if self.semantic_tokenizer_config is not None:
            output["semantic_tokenizer_config"] = (self.semantic_tokenizer_config.to_dict())
            output["semantic_vae_dim"] = self.semantic_vae_dim
        if self.tts_backbone_num_hidden_layers is not None:
            output["tts_backbone_num_hidden_layers"] = (self.tts_backbone_num_hidden_layers)
        return output


def parse_vibevoice_config(values: Mapping[str, Any]) -> VibeVoiceASRConfig | VibeVoiceTTSConfig:
    """Parse a published VibeVoice config without guessing its graph."""
    source = _mapping(values, name="VibeVoice configuration")
    model_type = str(source.get("model_type", "")).strip()
    if model_type == "vibevoice_asr":
        return VibeVoiceASRConfig.from_dict(source)
    if model_type in {"vibevoice", "vibevoice_streaming"}:
        return VibeVoiceTTSConfig.from_dict(source)
    raise ValueError(f"Unsupported VibeVoice `model_type` {model_type!r}.")


__all__ = [
    "VibeVoiceASRConfig",
    "VibeVoiceASRTokenizerConfig",
    "VibeVoiceDiffusionConfig",
    "VibeVoiceLegacyTokenizerConfig",
    "VibeVoiceTTSConfig",
    "parse_vibevoice_config",
]
