"""Configuration for the VoiceHub-owned Irodori-TTS graph."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

_PUBLISHED_VARIANTS = {
    "v3": {
        "use_speaker_condition": True,
        "use_caption_condition": False,
        "use_duration_predictor": True,
        "duration_architecture": "token_sum_adarn_zero_no_aux",
    },
    "v3-voice-design": {
        "use_speaker_condition": True,
        "use_caption_condition": True,
        "use_duration_predictor": True,
        "duration_architecture": "token_sum_dual_adarn_zero_no_aux",
    },
    "v2": {
        "use_speaker_condition": True,
        "use_caption_condition": False,
        "use_duration_predictor": False,
        "duration_architecture": "token_sum_adarn_zero_no_aux",
    },
    "v2-voice-design": {
        "use_speaker_condition": False,
        "use_caption_condition": True,
        "use_duration_predictor": False,
        "duration_architecture": "token_sum_adarn_zero_no_aux",
    },
}


@dataclass
class IrodoriModelConfig:
    """Exact graph configuration used by the published v2 and v3 families.

    The defaults describe ``Aratako/Irodori-TTS-500M-v3``.  Custom,
    smaller configurations are intentionally supported for testing and
    future architecture-compatible checkpoints.
    """

    variant: str = "v3"
    latent_dim: int = 32
    latent_patch_size: int = 1
    model_dim: int = 1280
    num_layers: int = 12
    num_heads: int = 20
    mlp_ratio: float = 2.875
    text_mlp_ratio: float | None = 2.6
    speaker_mlp_ratio: float | None = 2.6
    dropout: float = 0.0
    text_vocab_size: int = 99_574
    text_tokenizer_repo: str = "llm-jp/llm-jp-3-150m"
    text_add_bos: bool = True
    text_dim: int = 512
    text_layers: int = 10
    text_heads: int = 8
    use_caption_condition: bool = False
    use_speaker_condition: bool | None = True
    caption_vocab_size: int | None = None
    caption_tokenizer_repo: str | None = None
    caption_add_bos: bool | None = None
    caption_dim: int | None = None
    caption_layers: int | None = None
    caption_heads: int | None = None
    caption_mlp_ratio: float | None = None
    speaker_dim: int = 768
    speaker_layers: int = 8
    speaker_heads: int = 12
    speaker_patch_size: int = 1
    timestep_embed_dim: int = 512
    adaln_rank: int = 192
    norm_eps: float = 1e-5
    use_duration_predictor: bool = True
    duration_aux_dim: int = 14
    duration_hidden_dim: int = 1024
    duration_layers: int = 3
    duration_dropout: float = 0.1
    duration_attention_heads: int = 8
    duration_architecture: str = "token_sum_adarn_zero_no_aux"
    duration_token_init_frames: float = 9.0
    duration_speaker_fusion: str = "adarn_zero"
    duration_caption_fusion: str = "adarn_zero"
    duration_caption_pooling: str = "masked_mean"

    def __post_init__(self) -> None:
        self.variant = str(self.variant).strip().lower().replace("_", "-")
        self.validate()

    @classmethod
    def for_variant(
        cls,
        variant: str,
        **overrides: Any,
    ) -> IrodoriModelConfig:
        normalized = str(variant).strip().lower().replace("_", "-")
        if normalized not in _PUBLISHED_VARIANTS:
            choices = ", ".join(sorted(_PUBLISHED_VARIANTS))
            raise ValueError(f"Unsupported Irodori variant {variant!r}; choose {choices}.")
        values = dict(_PUBLISHED_VARIANTS[normalized])
        values.update(overrides)
        return cls(variant=normalized, **values)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> IrodoriModelConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Irodori model configuration must be a mapping.")
        accepted = {field.name for field in fields(cls)}
        return cls(**{name: value for name, value in values.items() if name in accepted})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def patched_latent_dim(self) -> int:
        return self.latent_dim * self.latent_patch_size

    @property
    def speaker_patched_latent_dim(self) -> int:
        return self.patched_latent_dim * self.speaker_patch_size

    @property
    def use_speaker_condition_resolved(self) -> bool:
        if self.use_speaker_condition is None:
            return not self.use_caption_condition
        return bool(self.use_speaker_condition)

    @property
    def text_mlp_ratio_resolved(self) -> float:
        return self.mlp_ratio if self.text_mlp_ratio is None else float(self.text_mlp_ratio)

    @property
    def caption_vocab_size_resolved(self) -> int:
        return self.text_vocab_size if self.caption_vocab_size is None else self.caption_vocab_size

    @property
    def caption_tokenizer_repo_resolved(self) -> str:
        return (
            self.text_tokenizer_repo if self.caption_tokenizer_repo is None else self.caption_tokenizer_repo)

    @property
    def caption_add_bos_resolved(self) -> bool:
        return self.text_add_bos if self.caption_add_bos is None else self.caption_add_bos

    @property
    def caption_dim_resolved(self) -> int:
        return self.text_dim if self.caption_dim is None else self.caption_dim

    @property
    def caption_layers_resolved(self) -> int:
        return self.text_layers if self.caption_layers is None else self.caption_layers

    @property
    def caption_heads_resolved(self) -> int:
        return self.text_heads if self.caption_heads is None else self.caption_heads

    @property
    def caption_mlp_ratio_resolved(self) -> float:
        return (
            self.text_mlp_ratio_resolved if self.caption_mlp_ratio is None else float(self.caption_mlp_ratio))

    @property
    def speaker_mlp_ratio_resolved(self) -> float:
        return self.mlp_ratio if self.speaker_mlp_ratio is None else float(self.speaker_mlp_ratio)

    def validate(self) -> None:
        if self.variant not in {*_PUBLISHED_VARIANTS, "custom"}:
            choices = ", ".join((*sorted(_PUBLISHED_VARIANTS), "custom"))
            raise ValueError(f"Unsupported Irodori variant {self.variant!r}; choose {choices}.")
        for name in (
                "text_add_bos",
                "use_caption_condition",
                "use_duration_predictor",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.use_speaker_condition is not None and not isinstance(
                self.use_speaker_condition,
                bool,
        ):
            raise TypeError("`use_speaker_condition` must be a boolean or None.")
        if self.caption_add_bos is not None and not isinstance(
                self.caption_add_bos,
                bool,
        ):
            raise TypeError("`caption_add_bos` must be a boolean or None.")
        for name in ("text_tokenizer_repo", "caption_tokenizer_repo"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"`{name}` must be a non-empty string or None.")
        positive_integers = (
            "adaln_rank",
            "duration_attention_heads",
            "duration_aux_dim",
            "duration_hidden_dim",
            "duration_layers",
            "latent_dim",
            "latent_patch_size",
            "model_dim",
            "num_heads",
            "num_layers",
            "speaker_dim",
            "speaker_heads",
            "speaker_layers",
            "speaker_patch_size",
            "text_dim",
            "text_heads",
            "text_layers",
            "text_vocab_size",
            "timestep_embed_dim",
        )
        for name in positive_integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")
        optional_positive_integers = (
            "caption_dim",
            "caption_heads",
            "caption_layers",
            "caption_vocab_size",
        )
        for name in optional_positive_integers:
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"`{name}` must be a positive integer or None.")
        if self.timestep_embed_dim % 2:
            raise ValueError("`timestep_embed_dim` must be even.")
        for dimension, heads in (
            (self.model_dim, self.num_heads),
            (self.text_dim, self.text_heads),
            (self.speaker_dim, self.speaker_heads),
        ):
            if dimension % heads:
                raise ValueError("Every attention dimension must be divisible by its head count.")
            if (dimension // heads) % 2:
                raise ValueError("Irodori rotary attention requires an even head dimension.")
        if self.use_caption_condition:
            if self.caption_dim_resolved % self.caption_heads_resolved:
                raise ValueError("Caption dimension must be divisible by caption heads.")
            if (self.caption_dim_resolved // self.caption_heads_resolved) % 2:
                raise ValueError("Caption rotary attention requires an even head dimension.")
        for name in ("dropout", "duration_dropout"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0):
                raise ValueError(f"`{name}` must be finite and in [0, 1).")
        for name, value in (
            ("mlp_ratio", self.mlp_ratio),
            ("text_mlp_ratio", self.text_mlp_ratio_resolved),
            ("speaker_mlp_ratio", self.speaker_mlp_ratio_resolved),
            ("norm_eps", self.norm_eps),
            ("duration_token_init_frames", self.duration_token_init_frames),
        ):
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or float(value) <= 0.0):
                raise ValueError(f"`{name}` must be finite and positive.")
        if self.use_caption_condition:
            caption_mlp_ratio = self.caption_mlp_ratio_resolved
            if (isinstance(caption_mlp_ratio, bool) or not isinstance(caption_mlp_ratio, (int, float)) or
                    not math.isfinite(float(caption_mlp_ratio)) or float(caption_mlp_ratio) <= 0.0):
                raise ValueError("`caption_mlp_ratio` must be finite and positive.")
        if self.duration_architecture not in {
                "pooled",
                "token_sum_adarn_zero_no_aux",
                "token_sum_dual_adarn_zero_no_aux",
        }:
            raise ValueError("Unsupported Irodori duration architecture.")
        if self.duration_speaker_fusion not in {
                "concat",
                "adarn",
                "adarn_zero",
                "speaker_cross_attn",
                "text_cross_attn",
        }:
            raise ValueError("Unsupported Irodori duration speaker fusion.")
        if self.duration_caption_fusion != "adarn_zero":
            raise ValueError("Unsupported Irodori duration caption fusion.")
        if self.duration_caption_pooling != "masked_mean":
            raise ValueError("Unsupported Irodori duration caption pooling.")
        if (self.use_duration_predictor and self.duration_architecture in {
                "token_sum_adarn_zero_no_aux",
                "token_sum_dual_adarn_zero_no_aux",
        } and not self.use_speaker_condition_resolved):
            raise ValueError("The token-sum duration architecture requires speaker conditioning.")
        if (self.use_duration_predictor and
                self.duration_architecture == "token_sum_dual_adarn_zero_no_aux" and
                not self.use_caption_condition):
            raise ValueError("The dual token-sum duration architecture requires caption conditioning.")
        if (self.use_duration_predictor and self.duration_architecture in {
                "token_sum_adarn_zero_no_aux",
                "token_sum_dual_adarn_zero_no_aux",
        } and self.duration_speaker_fusion != "adarn_zero"):
            raise ValueError(
                "The token-sum duration architecture requires "
                "`duration_speaker_fusion='adarn_zero'`.")
        if (self.use_duration_predictor and
                self.duration_architecture == "token_sum_dual_adarn_zero_no_aux" and
                self.duration_caption_fusion != "adarn_zero"):
            raise ValueError(
                "The dual token-sum duration architecture requires "
                "`duration_caption_fusion='adarn_zero'`.")
        if (self.use_duration_predictor and self.duration_architecture == "pooled" and
                not self.use_speaker_condition_resolved and self.duration_speaker_fusion != "concat"):
            raise ValueError(
                "A pooled duration predictor without speaker conditioning "
                "requires `duration_speaker_fusion='concat'`.")
        if (self.use_duration_predictor and self.duration_architecture == "pooled" and
                self.text_dim % self.duration_attention_heads):
            raise ValueError(
                "Pooled duration attention requires `text_dim` to be divisible "
                "by `duration_attention_heads`.")


ModelConfig = IrodoriModelConfig

__all__ = ["IrodoriModelConfig", "ModelConfig"]
