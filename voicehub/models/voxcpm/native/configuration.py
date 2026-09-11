"""Validated configuration for VoiceHub's native VoxCPM2 graph."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import Any


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return value


def _positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"`{name}` must be finite and greater than zero.")
    return result


@dataclass(frozen=True, slots=True)
class VoxCPMRopeScalingConfig:
    """MiniCPM-4 long-RoPE factors persisted by VoxCPM2."""

    long_factor: tuple[float, ...]
    short_factor: tuple[float, ...]
    original_max_position_embeddings: int
    type: str = "longrope"

    def __post_init__(self) -> None:
        if self.type != "longrope":
            raise ValueError("VoxCPM2 only supports the published `longrope` scaling.")
        _positive_integer(
            "original_max_position_embeddings",
            self.original_max_position_embeddings,
        )
        for name in ("long_factor", "short_factor"):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"`{name}` must contain finite positive factors.")
            object.__setattr__(self, name, values)
        if len(self.long_factor) != len(self.short_factor):
            raise ValueError("Long- and short-context RoPE factors must have equal lengths.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> VoxCPMRopeScalingConfig:
        if not isinstance(values, Mapping):
            raise TypeError("VoxCPM RoPE configuration must be a mapping.")
        return cls(
            type=str(values.get("type", "longrope")),
            long_factor=tuple(values["long_factor"]),
            short_factor=tuple(values["short_factor"]),
            original_max_position_embeddings=int(values["original_max_position_embeddings"]),
        )


def _default_rope_factors() -> tuple[float, ...]:
    # Production resolves the immutable published vector from config.json.
    # Neutral factors keep direct construction valid without inventing a
    # different tensor layout; they do not claim reference-checkpoint parity.
    return tuple(1.0 for _ in range(64))


@dataclass(frozen=True, slots=True)
class VoxCPMTransformerConfig:
    """One MiniCPM-4 decoder used by VoxCPM2."""

    hidden_size: int = 2_048
    intermediate_size: int = 6_144
    max_position_embeddings: int = 32_768
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    vocab_size: int = 73_448
    use_mup: bool = False
    scale_emb: float = 12.0
    dim_model_base: int = 256
    scale_depth: float = 1.4
    kv_channels: int | None = 128
    no_rope: bool = False
    rope_scaling: VoxCPMRopeScalingConfig = field(
        default_factory=lambda: VoxCPMRopeScalingConfig(
            long_factor=_default_rope_factors(),
            short_factor=_default_rope_factors(),
            original_max_position_embeddings=32_768,
        ))
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        for name in (
                "hidden_size",
                "intermediate_size",
                "max_position_embeddings",
                "num_attention_heads",
                "num_hidden_layers",
                "num_key_value_heads",
                "dim_model_base",
        ):
            _positive_integer(name, getattr(self, name))
        if isinstance(self.vocab_size, bool) or not isinstance(self.vocab_size, int):
            raise TypeError("`vocab_size` must be an integer.")
        if self.vocab_size < 0:
            raise ValueError("`vocab_size` cannot be negative.")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("`num_key_value_heads` must divide `num_attention_heads`.")
        head_dim = self.head_dim
        if head_dim % 2:
            raise ValueError("VoxCPM attention head dimensions must be even.")
        if not isinstance(self.use_mup, bool) or not isinstance(self.no_rope, bool):
            raise TypeError("`use_mup` and `no_rope` must be booleans.")
        for name in ("rms_norm_eps", "rope_theta", "scale_emb", "scale_depth"):
            object.__setattr__(self, name, _positive_float(name, getattr(self, name)))
        if self.kv_channels is not None:
            _positive_integer("kv_channels", self.kv_channels)
        if not isinstance(self.rope_scaling, VoxCPMRopeScalingConfig):
            object.__setattr__(
                self,
                "rope_scaling",
                VoxCPMRopeScalingConfig.from_mapping(self.rope_scaling),
            )
        if not self.no_rope and len(self.rope_scaling.long_factor) != head_dim // 2:
            raise ValueError("VoxCPM RoPE factor count must equal half the attention head dimension.")

    @property
    def head_dim(self) -> int:
        return (
            self.hidden_size // self.num_attention_heads if self.kv_channels is None else self.kv_channels)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> VoxCPMTransformerConfig:
        if not isinstance(values, Mapping):
            raise TypeError("VoxCPM transformer configuration must be a mapping.")
        known = {item.name for item in fields(cls)}
        normalized = {key: value for key, value in values.items() if key in known}
        if "rope_scaling" in normalized:
            normalized["rope_scaling"] = VoxCPMRopeScalingConfig.from_mapping(normalized["rope_scaling"])
        return cls(**normalized)


@dataclass(frozen=True, slots=True)
class VoxCPMLocalConfig:
    hidden_dim: int = 1_024
    ffn_dim: int = 4_096
    num_heads: int = 16
    num_layers: int = 12
    kv_channels: int | None = 128

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "ffn_dim", "num_heads", "num_layers"):
            _positive_integer(name, getattr(self, name))
        if self.hidden_dim % self.num_heads:
            raise ValueError("Local hidden dimension must be divisible by its head count.")
        if self.kv_channels is not None:
            _positive_integer("kv_channels", self.kv_channels)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> VoxCPMLocalConfig:
        if not isinstance(values, Mapping):
            raise TypeError("VoxCPM local configuration must be a mapping.")
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass(frozen=True, slots=True)
class VoxCPMCFMConfig:
    sigma_min: float = 1e-6
    solver: str = "euler"
    t_scheduler: str = "log-norm"
    training_cfg_rate: float = 0.1
    inference_cfg_rate: float = 2.0
    reg_loss_type: str = "l1"
    ratio_r_neq_t_range: tuple[float, float] = (0.25, 0.75)
    noise_cond_prob_range: tuple[float, float] = (0.0, 0.0)
    noise_cond_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.solver != "euler":
            raise ValueError("The native VoxCPM2 runtime implements the source Euler solver.")
        if self.t_scheduler not in {"log-norm", "uniform"}:
            raise ValueError("Unsupported VoxCPM CFM time scheduler.")
        object.__setattr__(self, "sigma_min", _positive_float("sigma_min", self.sigma_min))
        for name in ("training_cfg_rate", "inference_cfg_rate", "noise_cond_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"`{name}` must be finite and non-negative.")
            object.__setattr__(self, name, value)
        for name in ("ratio_r_neq_t_range", "noise_cond_prob_range"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 2 or not 0 <= values[0] <= values[1] <= 1:
                raise ValueError(f"`{name}` must be an ordered probability pair.")
            object.__setattr__(self, name, values)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> VoxCPMCFMConfig:
        if not isinstance(values, Mapping):
            raise TypeError("VoxCPM CFM configuration must be a mapping.")
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass(frozen=True, slots=True)
class VoxCPMDiTConfig(VoxCPMLocalConfig):
    dit_mean_mode: bool = False
    cfm_config: VoxCPMCFMConfig = field(default_factory=VoxCPMCFMConfig)

    def __post_init__(self) -> None:
        VoxCPMLocalConfig.__post_init__(self)
        if not isinstance(self.dit_mean_mode, bool):
            raise TypeError("`dit_mean_mode` must be a boolean.")
        if not isinstance(self.cfm_config, VoxCPMCFMConfig):
            object.__setattr__(
                self,
                "cfm_config",
                VoxCPMCFMConfig.from_mapping(self.cfm_config),
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> VoxCPMDiTConfig:
        if not isinstance(values, Mapping):
            raise TypeError("VoxCPM DiT configuration must be a mapping.")
        known = {item.name for item in fields(cls)}
        normalized = {key: value for key, value in values.items() if key in known}
        # The published config used `mean_mode`; source used `dit_mean_mode`.
        normalized["dit_mean_mode"] = values.get(
            "dit_mean_mode",
            values.get("mean_mode", False),
        )
        if "cfm_config" in normalized:
            normalized["cfm_config"] = VoxCPMCFMConfig.from_mapping(normalized["cfm_config"])
        return cls(**normalized)


@dataclass(frozen=True, slots=True)
class VoxCPMAudioVAEConfig:
    encoder_dim: int = 128
    encoder_rates: tuple[int, ...] = (2, 5, 8, 8)
    latent_dim: int = 64
    decoder_dim: int = 2_048
    decoder_rates: tuple[int, ...] = (8, 6, 5, 2, 2, 2)
    depthwise: bool = True
    sample_rate: int = 16_000
    out_sample_rate: int = 48_000
    use_noise_block: bool = False
    sr_bin_boundaries: tuple[int, ...] | None = (20_000, 30_000, 40_000)
    cond_type: str = "scale_bias"
    cond_dim: int = 128
    cond_out_layer: bool = False

    def __post_init__(self) -> None:
        for name in (
                "encoder_dim",
                "latent_dim",
                "decoder_dim",
                "sample_rate",
                "out_sample_rate",
                "cond_dim",
        ):
            _positive_integer(name, getattr(self, name))
        for name in ("encoder_rates", "decoder_rates"):
            rates = tuple(_positive_integer(name, value) for value in getattr(self, name))
            if not rates:
                raise ValueError(f"`{name}` cannot be empty.")
            object.__setattr__(self, name, rates)
        for name in ("depthwise", "use_noise_block", "cond_out_layer"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.cond_type != "scale_bias":
            raise ValueError("The official VoxCPM2 AudioVAE uses scale/bias conditioning.")
        if self.sr_bin_boundaries is not None:
            boundaries = tuple(int(value) for value in self.sr_bin_boundaries)
            if tuple(sorted(boundaries)) != boundaries or any(value <= 0 for value in boundaries):
                raise ValueError("AudioVAE sampling-rate boundaries must be increasing.")
            object.__setattr__(self, "sr_bin_boundaries", boundaries)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> VoxCPMAudioVAEConfig:
        if not isinstance(values, Mapping):
            raise TypeError("VoxCPM AudioVAE configuration must be a mapping.")
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass(frozen=True, slots=True)
class VoxCPM2ArchitectureConfig:
    """Complete source/checkpoint layout for VoxCPM2."""

    lm_config: VoxCPMTransformerConfig = field(default_factory=VoxCPMTransformerConfig)
    patch_size: int = 4
    feat_dim: int = 64
    residual_lm_num_layers: int = 8
    residual_lm_no_rope: bool = True
    scalar_quantization_latent_dim: int = 512
    scalar_quantization_scale: int = 9
    encoder_config: VoxCPMLocalConfig = field(default_factory=VoxCPMLocalConfig)
    dit_config: VoxCPMDiTConfig = field(default_factory=VoxCPMDiTConfig)
    audio_vae_config: VoxCPMAudioVAEConfig = field(default_factory=VoxCPMAudioVAEConfig)
    max_length: int = 8_192

    def __post_init__(self) -> None:
        for name in (
                "patch_size",
                "feat_dim",
                "residual_lm_num_layers",
                "scalar_quantization_latent_dim",
                "scalar_quantization_scale",
                "max_length",
        ):
            _positive_integer(name, getattr(self, name))
        if not isinstance(self.residual_lm_no_rope, bool):
            raise TypeError("`residual_lm_no_rope` must be a boolean.")
        converters = (
            ("lm_config", VoxCPMTransformerConfig, VoxCPMTransformerConfig.from_mapping),
            ("encoder_config", VoxCPMLocalConfig, VoxCPMLocalConfig.from_mapping),
            ("dit_config", VoxCPMDiTConfig, VoxCPMDiTConfig.from_mapping),
            ("audio_vae_config", VoxCPMAudioVAEConfig, VoxCPMAudioVAEConfig.from_mapping),
        )
        for name, expected, converter in converters:
            if not isinstance(getattr(self, name), expected):
                object.__setattr__(self, name, converter(getattr(self, name)))
        if self.feat_dim != self.audio_vae_config.latent_dim:
            raise ValueError("VoxCPM feature dimension must match the AudioVAE latent dimension.")

    @property
    def sample_rate(self) -> int:
        return self.audio_vae_config.out_sample_rate

    @classmethod
    def tiny(
        cls,
        *,
        vocab_size: int = 128,
        hidden_size: int = 32,
        feat_dim: int = 8,
    ) -> VoxCPM2ArchitectureConfig:
        factors = tuple(1.0 for _ in range(hidden_size // 4 // 2))
        rope = VoxCPMRopeScalingConfig(
            long_factor=factors,
            short_factor=factors,
            original_max_position_embeddings=128,
        )
        lm = VoxCPMTransformerConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            max_position_embeddings=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            num_key_value_heads=2,
            vocab_size=vocab_size,
            scale_emb=1.0,
            dim_model_base=hidden_size,
            scale_depth=1.0,
            kv_channels=hidden_size // 4,
            rope_scaling=rope,
        )
        return cls(
            lm_config=lm,
            patch_size=2,
            feat_dim=feat_dim,
            residual_lm_num_layers=1,
            scalar_quantization_latent_dim=8,
            scalar_quantization_scale=3,
            encoder_config=VoxCPMLocalConfig(
                hidden_dim=hidden_size,
                ffn_dim=hidden_size * 2,
                num_heads=4,
                num_layers=1,
                kv_channels=hidden_size // 4,
            ),
            dit_config=VoxCPMDiTConfig(
                hidden_dim=hidden_size,
                ffn_dim=hidden_size * 2,
                num_heads=4,
                num_layers=1,
                kv_channels=hidden_size // 4,
                cfm_config=VoxCPMCFMConfig(),
            ),
            audio_vae_config=VoxCPMAudioVAEConfig(
                encoder_dim=4,
                encoder_rates=(2, 2),
                latent_dim=feat_dim,
                decoder_dim=32,
                decoder_rates=(2, 2, 2),
                sample_rate=16_000,
                out_sample_rate=16_000,
                sr_bin_boundaries=None,
            ),
            max_length=128,
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.update({"architecture": "voxcpm2", "format_version": 1})
        return values

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> VoxCPM2ArchitectureConfig:
        if not isinstance(values, Mapping):
            raise TypeError("VoxCPM2 configuration must be a mapping.")
        architecture = values.get("architecture")
        if architecture not in (None, "voxcpm2"):
            raise ValueError(f"Expected a VoxCPM2 config, received {architecture!r}.")
        known = {item.name for item in fields(cls)}
        normalized = {key: value for key, value in values.items() if key in known}
        if "lm_config" in normalized:
            normalized["lm_config"] = VoxCPMTransformerConfig.from_mapping(normalized["lm_config"])
        if "encoder_config" in normalized:
            normalized["encoder_config"] = VoxCPMLocalConfig.from_mapping(normalized["encoder_config"])
        if "dit_config" in normalized:
            normalized["dit_config"] = VoxCPMDiTConfig.from_mapping(normalized["dit_config"])
        if "audio_vae_config" in normalized:
            normalized["audio_vae_config"] = VoxCPMAudioVAEConfig.from_mapping(normalized["audio_vae_config"])
        return cls(**normalized)


__all__ = [
    "VoxCPM2ArchitectureConfig",
    "VoxCPMAudioVAEConfig",
    "VoxCPMCFMConfig",
    "VoxCPMDiTConfig",
    "VoxCPMLocalConfig",
    "VoxCPMRopeScalingConfig",
    "VoxCPMTransformerConfig",
]
