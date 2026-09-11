"""Typed configuration for the VoiceHub-native Bark runtime.

The defaults mirror ``suno/bark-small`` at the immutable revision
recorded in ``metadata.py``.  These objects deliberately contain no
Hugging Face runtime types; config JSON remains an interchange format,
not executable code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from voicehub.components.audio.codecs.encodec import EncodecConfig, encodec_24khz_config


@dataclass(slots=True)
class BarkSubModelConfig:
    """Shared GPT graph parameters for one Bark token stage."""

    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    num_layers: int = 12
    num_heads: int = 12
    hidden_size: int = 768
    dropout: float = 0.0
    bias: bool = True
    initializer_range: float = 0.02
    use_cache: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            "block_size",
            "input_vocab_size",
            "output_vocab_size",
            "num_layers",
            "num_heads",
            "hidden_size",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Bark `{name}` must be a positive integer.")
        if self.hidden_size % self.num_heads:
            raise ValueError("Bark hidden size must be divisible by the head count.")
        if not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError("Bark dropout must be in [0, 1).")
        self.dropout = float(self.dropout)
        if (isinstance(self.initializer_range, bool) or
                not isinstance(self.initializer_range, (int, float)) or self.initializer_range <= 0):
            raise ValueError("Bark initializer range must be positive.")
        self.initializer_range = float(self.initializer_range)
        if not isinstance(self.bias, bool) or not isinstance(self.use_cache, bool):
            raise TypeError("Bark `bias` and `use_cache` must be booleans.")


@dataclass(slots=True)
class BarkSemanticConfig(BarkSubModelConfig):
    input_vocab_size: int = 129_600
    bias: bool = False


@dataclass(slots=True)
class BarkCoarseConfig(BarkSubModelConfig):
    input_vocab_size: int = 12_096
    output_vocab_size: int = 12_096
    bias: bool = False


@dataclass(slots=True)
class BarkFineConfig(BarkSubModelConfig):
    input_vocab_size: int = 1_056
    output_vocab_size: int = 1_056
    bias: bool = False
    n_codes_total: int = 8
    n_codes_given: int = 1

    def __post_init__(self) -> None:
        BarkSubModelConfig.__post_init__(self)
        if (isinstance(self.n_codes_total, bool) or not isinstance(self.n_codes_total, int) or
                self.n_codes_total <= 1):
            raise ValueError("Bark fine `n_codes_total` must exceed one.")
        if (isinstance(self.n_codes_given, bool) or not isinstance(self.n_codes_given, int) or
                not 0 < self.n_codes_given < self.n_codes_total):
            raise ValueError("Bark fine `n_codes_given` must be between zero and "
                             "`n_codes_total`.")


@dataclass(slots=True)
class BarkSemanticGenerationConfig:
    max_input_semantic_length: int = 256
    max_new_tokens: int = 768
    semantic_infer_token: int = 129_599
    semantic_pad_token: int = 10_000
    semantic_rate_hz: float = 49.9
    semantic_vocab_size: int = 10_000
    text_encoding_offset: int = 10_048
    text_pad_token: int = 129_595
    eos_token_id: int = 10_000
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 1.0
    min_eos_p: float | None = None

    def __post_init__(self) -> None:
        _validate_sampling(
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        for name in (
                "max_input_semantic_length",
                "max_new_tokens",
                "semantic_infer_token",
                "semantic_pad_token",
                "semantic_vocab_size",
                "text_encoding_offset",
                "text_pad_token",
                "eos_token_id",
        ):
            _nonnegative_integer(getattr(self, name), name=name)
        if self.max_input_semantic_length == 0 or self.max_new_tokens == 0:
            raise ValueError("Bark semantic lengths must be positive.")
        _positive_float(self.semantic_rate_hz, name="semantic_rate_hz")
        if self.min_eos_p is not None and not 0 <= self.min_eos_p <= 1:
            raise ValueError("Bark `min_eos_p` must be in [0, 1] or None.")


@dataclass(slots=True)
class BarkCoarseGenerationConfig:
    coarse_infer_token: int = 12_050
    coarse_rate_hz: float = 75.0
    coarse_semantic_pad_token: int = 12_048
    max_coarse_history: int = 630
    max_coarse_input_length: int = 256
    n_coarse_codebooks: int = 2
    sliding_window_len: int = 60
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 1.0

    def __post_init__(self) -> None:
        _validate_sampling(
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        for name in (
                "coarse_infer_token",
                "coarse_semantic_pad_token",
                "max_coarse_history",
                "max_coarse_input_length",
                "n_coarse_codebooks",
                "sliding_window_len",
        ):
            _nonnegative_integer(getattr(self, name), name=name)
        if min(
                self.max_coarse_history,
                self.max_coarse_input_length,
                self.n_coarse_codebooks,
                self.sliding_window_len,
        ) <= 0:
            raise ValueError("Bark coarse sizes must be positive.")
        _positive_float(self.coarse_rate_hz, name="coarse_rate_hz")


@dataclass(slots=True)
class BarkFineGenerationConfig:
    max_fine_history_length: int = 512
    max_fine_input_length: int = 1024
    n_fine_codebooks: int = 8
    temperature: float | None = 0.5

    def __post_init__(self) -> None:
        for name in (
                "max_fine_history_length",
                "max_fine_input_length",
                "n_fine_codebooks",
        ):
            _nonnegative_integer(getattr(self, name), name=name)
        if min(
                self.max_fine_history_length,
                self.max_fine_input_length,
                self.n_fine_codebooks,
        ) <= 0:
            raise ValueError("Bark fine sizes must be positive.")
        if (self.max_fine_input_length < self.max_fine_history_length):
            raise ValueError("Bark fine input length cannot be shorter than its history.")
        if self.temperature is not None:
            _positive_float(self.temperature, name="temperature")


@dataclass(slots=True)
class BarkGenerationConfig:
    """Generation constants published with the pinned Bark checkpoint."""

    sample_rate: int = 24_000
    codebook_size: int = 1024
    semantic: BarkSemanticGenerationConfig = field(default_factory=BarkSemanticGenerationConfig)
    coarse: BarkCoarseGenerationConfig = field(default_factory=BarkCoarseGenerationConfig)
    fine: BarkFineGenerationConfig = field(default_factory=BarkFineGenerationConfig)

    def __post_init__(self) -> None:
        _nonnegative_integer(self.sample_rate, name="sample_rate")
        _nonnegative_integer(self.codebook_size, name="codebook_size")
        if self.sample_rate == 0 or self.codebook_size == 0:
            raise ValueError("Bark sample rate and codebook size must be positive.")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> BarkGenerationConfig:
        if not isinstance(values, dict):
            raise TypeError("Bark generation configuration must be an object.")
        return cls(
            sample_rate=int(values.get("sample_rate", 24_000)),
            codebook_size=int(values.get("codebook_size", 1024)),
            semantic=BarkSemanticGenerationConfig(
                **_known(
                    values.get("semantic_config", values.get("semantic", {})),
                    BarkSemanticGenerationConfig,
                )),
            coarse=BarkCoarseGenerationConfig(
                **_known(
                    values.get(
                        "coarse_acoustics_config",
                        values.get("coarse", {}),
                    ),
                    BarkCoarseGenerationConfig,
                )),
            fine=BarkFineGenerationConfig(
                **_known(
                    values.get(
                        "fine_acoustics_config",
                        values.get("fine", {}),
                    ),
                    BarkFineGenerationConfig,
                )),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "bark",
            "sample_rate": self.sample_rate,
            "codebook_size": self.codebook_size,
            "semantic_config": asdict(self.semantic),
            "coarse_acoustics_config": asdict(self.coarse),
            "fine_acoustics_config": asdict(self.fine),
        }


@dataclass(slots=True)
class BarkArchitectureConfig:
    """Complete Bark transformer and native Encodec graph."""

    semantic: BarkSemanticConfig = field(default_factory=BarkSemanticConfig)
    coarse: BarkCoarseConfig = field(default_factory=BarkCoarseConfig)
    fine: BarkFineConfig = field(default_factory=BarkFineConfig)
    codec: EncodecConfig = field(default_factory=encodec_24khz_config)
    initializer_range: float = 0.02

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> BarkArchitectureConfig:
        if not isinstance(values, dict):
            raise TypeError("Bark model configuration must be an object.")
        semantic = values.get("semantic_config", values.get("semantic", {}))
        coarse = values.get(
            "coarse_acoustics_config",
            values.get("coarse", {}),
        )
        fine = values.get("fine_acoustics_config", values.get("fine", {}))
        codec = values.get("codec_config", values.get("codec", {}))
        if not all(isinstance(item, dict) for item in (semantic, coarse, fine, codec)):
            raise TypeError("Every Bark sub-configuration must be an object.")
        return cls(
            semantic=BarkSemanticConfig(**_known(semantic, BarkSemanticConfig)),
            coarse=BarkCoarseConfig(**_known(coarse, BarkCoarseConfig)),
            fine=BarkFineConfig(**_known(fine, BarkFineConfig)),
            codec=_codec_from_hugging_face(codec),
            initializer_range=float(values.get("initializer_range", 0.02)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "bark",
            'architectures': ["BarkModel"],
            "initializer_range": self.initializer_range,
            "semantic_config": {
                **asdict(self.semantic),
                "model_type": "semantic",
            },
            "coarse_acoustics_config": {
                **asdict(self.coarse),
                "model_type": "coarse_acoustics",
            },
            "fine_acoustics_config": {
                **asdict(self.fine),
                "model_type": "fine_acoustics",
            },
            "codec_config": {
                **self.codec.to_dict(),
                "model_type": "encodec",
            },
        }


def _known(values: Any, target: type) -> dict[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise TypeError(f"{target.__name__} configuration must be an object.")
    names = target.__dataclass_fields__
    return {name: value for name, value in values.items() if name in names}


def _nonnegative_integer(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Bark `{name}` must be a non-negative integer.")


def _positive_float(value: Any, *, name: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not 0 < float(value) < float("inf")):
        raise ValueError(f"Bark `{name}` must be finite and positive.")


def _validate_sampling(
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> None:
    if not isinstance(do_sample, bool):
        raise TypeError("Bark `do_sample` must be a boolean.")
    _positive_float(temperature, name="temperature")
    _nonnegative_integer(top_k, name="top_k")
    if (isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < float(top_p) <= 1):
        raise ValueError("Bark `top_p` must be in (0, 1].")


def _codec_from_hugging_face(values: dict[str, Any]) -> EncodecConfig:
    """Translate declarative HF Encodec names to the native codec config."""
    if not values:
        return BarkArchitectureConfig().codec
    if "sample_rate" in values and "ratios" in values:
        native_fields = set(EncodecConfig.__dataclass_fields__)
        return EncodecConfig.from_dict({
            name: value
            for name, value in values.items() if name in native_fields
        })
    return EncodecConfig(
        sample_rate=int(values.get("sampling_rate", values.get("sample_rate", 24_000))),
        channels=int(values.get("audio_channels", values.get("channels", 1))),
        dimension=int(values.get("codebook_dim", values.get("dimension", 128))),
        n_filters=int(values.get("num_filters", values.get("n_filters", 32))),
        n_residual_layers=int(values.get("num_residual_layers", values.get("n_residual_layers", 1))),
        ratios=tuple(values.get(
            "upsampling_ratios",
            values.get("ratios", (8, 5, 4, 2)),
        )),
        model_norm=str(values.get("norm_type", values.get("model_norm", "weight_norm"))),
        kernel_size=int(values.get("kernel_size", 7)),
        last_kernel_size=int(values.get("last_kernel_size", 7)),
        residual_kernel_size=int(values.get("residual_kernel_size", 3)),
        dilation_base=int(values.get("dilation_growth_rate", values.get("dilation_base", 2))),
        causal=bool(values.get("use_causal_conv", values.get("causal", True))),
        # HF's flag means "materialize the shortcut convolution", whereas
        # native Encodec's `true_skip` means the inverse.
        true_skip=(
            bool(values["true_skip"])
            if "true_skip" in values else not bool(values.get("use_conv_shortcut", True))),
        compress=int(values.get("compress", 2)),
        lstm=int(values.get("num_lstm_layers", values.get("lstm", 2))),
        bins=int(values.get("codebook_size", values.get("bins", 1024))),
        target_bandwidths=tuple(
            float(item) for item in values.get(
                "target_bandwidths",
                (1.5, 3.0, 6.0, 12.0, 24.0),
            )),
        normalize=bool(values.get("normalize", False)),
        segment=values.get("chunk_length_s", values.get("segment")),
        overlap=float(values.get("overlap", 0.01) or 0.01),
        trim_right_ratio=float(values.get("trim_right_ratio", 1.0)),
        name="encodec_24khz",
    )


__all__ = [
    "BarkArchitectureConfig",
    "BarkCoarseConfig",
    "BarkCoarseGenerationConfig",
    "BarkFineConfig",
    "BarkFineGenerationConfig",
    "BarkGenerationConfig",
    "BarkSemanticConfig",
    "BarkSemanticGenerationConfig",
    "BarkSubModelConfig",
]
