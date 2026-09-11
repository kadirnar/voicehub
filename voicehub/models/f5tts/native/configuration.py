"""Validated graph configuration for native F5-TTS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _probability(name: str, value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0):
        raise ValueError(f"`{name}` must be in [0, 1].")
    return float(value)


@dataclass(frozen=True, slots=True)
class F5TTSArchitectureConfig:
    """Dimensions and objective settings shared by training and inference.

    Defaults are the released ``F5TTS_v1_Base`` graph.  Small values may
    be supplied for tests and research variants without changing runtime
    code.
    """

    model_name: str = "F5TTS_v1_Base"
    mel_dim: int = 100
    dim: int = 1_024
    depth: int = 22
    heads: int = 16
    dim_head: int = 64
    ff_mult: float = 2.0
    text_dim: int = 512
    text_num_embeds: int = 2_545
    text_mask_padding: bool = True
    text_embedding_average_upsampling: bool = False
    conv_layers: int = 4
    conv_mult: int = 2
    dropout: float = 0.1
    qk_norm: str | None = None
    pe_attn_head: int | None = None
    attn_mask_enabled: bool = False
    long_skip_connection: bool = False
    checkpoint_activations: bool = False
    sample_rate: int = 24_000
    n_fft: int = 1_024
    hop_length: int = 256
    win_length: int = 1_024
    audio_drop_prob: float = 0.3
    cond_drop_prob: float = 0.2
    mask_fraction_min: float = 0.7
    mask_fraction_max: float = 1.0
    sigma: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("`model_name` must be a non-empty string.")
        for name in (
                "mel_dim",
                "dim",
                "depth",
                "heads",
                "dim_head",
                "text_dim",
                "text_num_embeds",
                "conv_mult",
                "sample_rate",
                "n_fft",
                "hop_length",
                "win_length",
        ):
            _positive_integer(name, getattr(self, name))
        if isinstance(self.conv_layers, bool) or not isinstance(self.conv_layers, int):
            raise TypeError("`conv_layers` must be an integer.")
        if self.conv_layers < 0:
            raise ValueError("`conv_layers` cannot be negative.")
        if self.dim != self.heads * self.dim_head:
            raise ValueError("`dim` must equal `heads * dim_head`.")
        if (isinstance(self.ff_mult, bool) or not isinstance(self.ff_mult, (int, float)) or
                float(self.ff_mult) <= 0):
            raise ValueError("`ff_mult` must be positive.")
        for name in ("dropout", "audio_drop_prob", "cond_drop_prob"):
            _probability(name, getattr(self, name))
        low = _probability("mask_fraction_min", self.mask_fraction_min)
        high = _probability("mask_fraction_max", self.mask_fraction_max)
        if low <= 0 or low > high:
            raise ValueError(
                "`mask_fraction_min` must be positive and no greater than "
                "`mask_fraction_max`.")
        if self.qk_norm not in {None, "rms_norm"}:
            raise ValueError("`qk_norm` must be None or 'rms_norm'.")
        if self.pe_attn_head is not None:
            _positive_integer("pe_attn_head", self.pe_attn_head)
            if self.pe_attn_head > self.heads:
                raise ValueError("`pe_attn_head` cannot exceed `heads`.")
        if self.hop_length > self.win_length or self.win_length > self.n_fft:
            raise ValueError("Expected `hop_length <= win_length <= n_fft`.")
        for name in (
                "text_mask_padding",
                "text_embedding_average_upsampling",
                "attn_mask_enabled",
                "long_skip_connection",
                "checkpoint_activations",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if (isinstance(self.sigma, bool) or not isinstance(self.sigma, (int, float)) or
                not 0.0 <= float(self.sigma) < 1.0):
            raise ValueError("`sigma` must be in [0, 1).")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> F5TTSArchitectureConfig:
        if values is None:
            source: dict[str, Any] = {}
        elif not isinstance(values, Mapping):
            raise TypeError("F5-TTS architecture configuration must be a mapping.")
        else:
            source = dict(values)
        source.update(overrides)
        return cls(**source)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def f5tts_architecture_config(model_name: str) -> F5TTSArchitectureConfig:
    """Return a pinned released graph configuration.

    E2-TTS uses the separate UNetT graph and is rejected instead of
    being silently interpreted as F5-TTS DiT.
    """
    normalized = model_name.strip().lower().replace("-", "_")
    if normalized == "f5tts_v1_base":
        return F5TTSArchitectureConfig(model_name="F5TTS_v1_Base")
    if normalized == "f5tts_base":
        return F5TTSArchitectureConfig(
            model_name="F5TTS_Base",
            text_mask_padding=False,
            pe_attn_head=1,
        )
    if normalized == "f5tts_v1_small":
        return F5TTSArchitectureConfig(
            model_name="F5TTS_v1_Small",
            dim=768,
            depth=18,
            heads=12,
        )
    if normalized == "f5tts_small":
        return F5TTSArchitectureConfig(
            model_name="F5TTS_Small",
            dim=768,
            depth=18,
            heads=12,
            text_mask_padding=False,
            pe_attn_head=1,
        )
    if normalized.startswith("e2tts"):
        raise ValueError(
            "E2-TTS checkpoints use the UNetT architecture and cannot be "
            "loaded by the native F5-TTS DiT runtime.")
    raise ValueError(
        f"Unknown released F5-TTS graph {model_name!r}; pass an explicit "
        "`architecture` mapping for custom checkpoints.")


__all__ = ["F5TTSArchitectureConfig", "f5tts_architecture_config"]
