"""Validated configuration for the published ZONOS2 MoE architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil, isfinite
from typing import Any, Mapping


def _integer(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"`{name}` must be {qualifier}.")
    return value


def _finite_positive(value: Any, *, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0):
        raise ValueError(f"`{name}` must be a finite positive number.")
    return float(value)


@dataclass(slots=True)
class Zonos2ArchitectureConfig:
    """Architecture values serialized in the official ``params.json``.

    Unknown keys are preserved in :attr:`extra`, allowing a future
    compatible checkpoint to round-trip without silently discarding
    metadata.
    """

    model_type: str = "zonos2"
    dtype: str = "bfloat16"
    n_layers: int = 28
    dim: int = 2048
    head_dim: int = 128
    n_heads: int | None = None
    n_kv_heads: int | None = 4
    ffn_dim_multiplier: float = 1.5
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    max_seqlen: int = 6_144
    n_codebooks: int = 9
    codebook_size: int = 1_024
    eoa_id: int = 1_024
    audio_pad_id: int = 1_025
    text_vocab: int = 519
    loss_softcap: float = 15.0
    speaker_enabled: bool = True
    speaker_embedding_dim: int = 2_048
    speaker_lda_dim: int | None = 1_024
    speaker_background_token_enabled: bool = True
    accurate_mode_token_enabled: bool = True
    speaking_rate_num_buckets: int = 8
    speaking_rate_buckets: tuple[str, ...] = (
        "0-8",
        "8-11",
        "11-14",
        "14-17",
        "17-21",
        "21-28",
        "28-40",
        "40+",
    )
    quality_num_buckets: int = 60
    quality_features: tuple[str, ...] = (
        "lufs",
        "estimated_snr",
        "max_pause",
        "estimated_bandlimit_hz",
        "leading_silence_s",
        "trailing_silence_s",
    )
    quality_buckets: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "lufs": (
                "-1000--50",
                "-50--45.5",
                "-45.5--41",
                "-41--36.5",
                "-36.5--32",
                "-32--27.5",
                "-27.5--23",
                "-23--18.5",
                "-18.5--14",
                "-14--9.5",
                "-9.5--5",
                "-5+",
            ),
            "estimated_snr": (
                "-1000-0",
                "0-6",
                "6-12",
                "12-18",
                "18-24",
                "24-30",
                "30-36",
                "36-42",
                "42-48",
                "48-54",
                "54-60",
                "60+",
            ),
            "max_pause": (
                "0-0.5",
                "0.5-1",
                "1-1.5",
                "1.5-2",
                "2-2.5",
                "2.5-3",
                "3-3.5",
                "3.5-4",
                "4-4.5",
                "4.5-5",
                "5-5.5",
                "5.5-6",
            ),
            "estimated_bandlimit_hz": (
                "495.3-3433",
                "3433-6371",
                "6371-9310",
                "9310-12248",
                "12248-15186",
                "15186-18124",
                "18124-21062",
                "21062-24000",
            ),
            "leading_silence_s": (
                "0-0.05",
                "0.05-0.1",
                "0.1-0.25",
                "0.25-0.5",
                "0.5-1",
                "1-2",
                "2-4",
                "4+",
            ),
            "trailing_silence_s": (
                "0-0.05",
                "0.05-0.1",
                "0.1-0.25",
                "0.25-0.5",
                "0.5-1",
                "1-2",
                "2-4",
                "4+",
            ),
        })
    quality_dropout: dict[str, float] = field(
        default_factory=lambda: {
            "lufs": 0.25,
            "estimated_snr": 0.25,
            "max_pause": 0.25,
            "estimated_bandlimit_hz": 0.25,
            "leading_silence_s": 0.25,
            "trailing_silence_s": 0.25,
        })
    moe_impl: str = "sonic"
    moe_n_experts: int = 16
    moe_router_topk: int = 1
    special_topk_layers: dict[int, int] = field(default_factory=lambda: {26: 2})
    moe_router_dim: int = 128
    moe_start_from_layer: int = 3
    moe_end_from_layer: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.speaking_rate_buckets = tuple(str(item) for item in self.speaking_rate_buckets)
        self.quality_features = tuple(str(item) for item in self.quality_features)
        self.quality_buckets = {
            str(name): tuple(str(item) for item in values)
            for name, values in self.quality_buckets.items()
        }
        self.quality_dropout = {str(name): float(value) for name, value in self.quality_dropout.items()}
        self.special_topk_layers = {
            int(layer): int(value)
            for layer, value in self.special_topk_layers.items()
        }
        self.validate()

    @property
    def num_attention_heads(self) -> int:
        return self.dim // self.head_dim if self.n_heads is None else self.n_heads

    @property
    def num_key_value_heads(self) -> int:
        return (self.num_attention_heads if self.n_kv_heads is None else self.n_kv_heads)

    @property
    def intermediate_size(self) -> int:
        width = ceil(self.ffn_dim_multiplier * self.dim)
        return self.multiple_of * ceil(width / self.multiple_of)

    @property
    def audio_vocab_size(self) -> int:
        return self.codebook_size + 2

    @property
    def frame_width(self) -> int:
        return self.n_codebooks + 1

    @property
    def quality_bucket_counts(self) -> tuple[int, ...]:
        return tuple(len(self.quality_buckets[name]) for name in self.quality_features)

    def is_moe_layer(self, layer_index: int) -> bool:
        return (
            self.moe_n_experts > 1 and layer_index >= self.moe_start_from_layer and
            self.n_layers - layer_index > self.moe_end_from_layer)

    def top_k_for_layer(self, layer_index: int) -> int:
        return self.special_topk_layers.get(layer_index, self.moe_router_topk)

    def validate(self) -> None:
        if self.model_type != "zonos2":
            raise ValueError("ZONOS2 configuration requires model_type='zonos2'.")
        for name in (
                "n_layers",
                "dim",
                "head_dim",
                "multiple_of",
                "max_seqlen",
                "n_codebooks",
                "codebook_size",
                "text_vocab",
                "speaker_embedding_dim",
                "moe_n_experts",
                "moe_router_topk",
                "moe_router_dim",
        ):
            _integer(getattr(self, name), name=name)
        for name in ("moe_start_from_layer", "moe_end_from_layer"):
            _integer(getattr(self, name), name=name, minimum=0)
        for name in ("ffn_dim_multiplier", "norm_eps", "rope_theta"):
            _finite_positive(getattr(self, name), name=name)
        if self.loss_softcap < 0 or not isfinite(self.loss_softcap):
            raise ValueError("`loss_softcap` must be finite and non-negative.")
        if self.dim % self.head_dim:
            raise ValueError("`dim` must be divisible by `head_dim`.")
        if self.head_dim % 2:
            raise ValueError("Interleaved RoPE requires an even `head_dim`.")
        if self.num_attention_heads * self.head_dim != self.dim:
            raise ValueError("`n_heads * head_dim` must equal `dim`.")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("`n_heads` must be divisible by `n_kv_heads`.")
        if not 0 <= self.eoa_id < self.audio_vocab_size:
            raise ValueError("`eoa_id` is outside the audio vocabulary.")
        if not 0 <= self.audio_pad_id < self.audio_vocab_size:
            raise ValueError("`audio_pad_id` is outside the audio vocabulary.")
        if self.eoa_id == self.audio_pad_id:
            raise ValueError("End-of-audio and padding token IDs must differ.")
        if self.speaker_lda_dim is not None:
            _integer(self.speaker_lda_dim, name="speaker_lda_dim")
        if self.speaking_rate_num_buckets != len(self.speaking_rate_buckets):
            raise ValueError("`speaking_rate_num_buckets` does not match its bucket list.")
        if set(self.quality_features) != set(self.quality_buckets):
            raise ValueError("`quality_buckets` must define exactly every quality feature.")
        if self.quality_num_buckets != sum(self.quality_bucket_counts):
            raise ValueError("`quality_num_buckets` does not match the bucket inventory.")
        unknown_dropout = set(self.quality_dropout) - set(self.quality_features)
        if unknown_dropout:
            raise ValueError("Quality dropout contains unknown features: "
                             f"{sorted(unknown_dropout)!r}.")
        if any(not 0.0 <= value <= 1.0 for value in self.quality_dropout.values()):
            raise ValueError("Quality dropout probabilities must be in [0, 1].")
        if self.moe_impl != "sonic" and self.moe_n_experts > 1:
            raise ValueError("Native ZONOS2 currently supports `moe_impl='sonic'`.")
        if self.moe_router_topk > self.moe_n_experts:
            raise ValueError("`moe_router_topk` cannot exceed expert count.")
        for layer, top_k in self.special_topk_layers.items():
            if not 0 <= layer < self.n_layers:
                raise ValueError(f"Special top-k layer {layer} is out of range.")
            if not 1 <= top_k <= self.moe_n_experts:
                raise ValueError(f"Invalid top-k {top_k} for layer {layer}.")

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> Zonos2ArchitectureConfig:
        if not isinstance(values, Mapping):
            raise TypeError("ZONOS2 architecture configuration must be a mapping.")
        source = dict(values)
        known = {
            name: source.pop(name)
            for name in tuple(source) if name in cls.__dataclass_fields__ and name != "extra"
        }
        existing_extra = source.pop("extra", {})
        if existing_extra is None:
            existing_extra = {}
        if not isinstance(existing_extra, Mapping):
            raise TypeError("ZONOS2 `extra` configuration must be a mapping.")
        source = {**dict(existing_extra), **source}
        return cls(**known, extra=source)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        extra = values.pop("extra")
        values["speaking_rate_buckets"] = list(self.speaking_rate_buckets)
        values["quality_features"] = list(self.quality_features)
        values["quality_buckets"] = {name: list(items) for name, items in self.quality_buckets.items()}
        values["special_topk_layers"] = {
            str(layer): top_k
            for layer, top_k in self.special_topk_layers.items()
        }
        values.update(extra)
        return values


__all__ = ["Zonos2ArchitectureConfig"]
