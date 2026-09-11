"""Audited GPT-SoVITS S1 and classic-S2 configuration contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SUPPORTED_GPT_SOVITS_VARIANTS = (
    "v1",
    "v2",
    "v2Pro",
    "v2ProPlus",
)
UNSUPPORTED_GPT_SOVITS_VARIANTS = ("v3", "v4", "LoRA")

_VARIANT_ALIASES = {
    "v1": "v1",
    "v2": "v2",
    "v2pro": "v2Pro",
    "v2proplus": "v2ProPlus",
}

_S1_PHONEME_VOCABULARY = {
    "v1": 512,
    "v2": 732,
}

_S2_VARIANT_TOPOLOGY = {
    "v1": {
        "phoneme_vocabulary_size": 322,
        "style_channels": 1_025,
        "dropout": 0.1,
        "upsample_initial_channels": 512,
        "upsample_kernel_sizes": (16, 16, 8, 2, 2),
        "gin_channels": 512,
        "speaker_embedding_dim": None,
        "discriminator_periods": (2, 3, 5, 7, 11),
    },
    "v2": {
        "phoneme_vocabulary_size": 732,
        "style_channels": 704,
        "dropout": 0.1,
        "upsample_initial_channels": 512,
        "upsample_kernel_sizes": (16, 16, 8, 2, 2),
        "gin_channels": 512,
        "speaker_embedding_dim": None,
        "discriminator_periods": (2, 3, 5, 7, 11),
    },
    "v2Pro": {
        "phoneme_vocabulary_size": 732,
        "style_channels": 704,
        "dropout": 0.0,
        "upsample_initial_channels": 512,
        "upsample_kernel_sizes": (16, 16, 8, 2, 2),
        "gin_channels": 1_024,
        "speaker_embedding_dim": 20_480,
        "discriminator_periods": (2, 3, 5, 7, 11, 17, 23),
    },
    "v2ProPlus": {
        "phoneme_vocabulary_size": 732,
        "style_channels": 704,
        "dropout": 0.0,
        "upsample_initial_channels": 768,
        "upsample_kernel_sizes": (20, 16, 8, 2, 2),
        "gin_channels": 1_024,
        "speaker_embedding_dim": 20_480,
        "discriminator_periods": (2, 3, 5, 7, 11, 17, 23),
    },
}


def normalize_gptsovits_variant(variant: object) -> str:
    """Return one canonical classic-S2 variant or fail closed."""
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError("GPT-SoVITS `variant` must be a non-empty string.")
    normalized = variant.strip()
    try:
        return _VARIANT_ALIASES[normalized.lower()]
    except KeyError as error:
        if normalized.lower() in {"v3", "v4"} or "lora" in normalized.lower():
            raise ValueError(
                "GPT-SoVITS V3/V4 and LoRA checkpoints use the separate "
                "flow-matching/vocoder graph and are not accepted by the "
                "classic native S2 runtime.") from error
        raise ValueError(
            "Unsupported GPT-SoVITS variant. Native classic-S2 support is "
            f"limited to {SUPPORTED_GPT_SOVITS_VARIANTS}.") from error


def s1_variant_for_s2(variant: object) -> str:
    """Map an S2 release family to its checkpoint-compatible S1 family."""
    return "v1" if normalize_gptsovits_variant(variant) == "v1" else "v2"


def _positive(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


@dataclass(frozen=True, slots=True)
class GPTSoVITSS1Config:
    """Exact public autoregressive semantic-model topology."""

    version: str = "v2"
    vocabulary_size: int = 1_025
    phoneme_vocabulary_size: int = 732
    embedding_dim: int = 512
    hidden_dim: int = 512
    attention_heads: int = 16
    layers: int = 24
    dropout: float = 0.0
    eos_token_id: int = 1_024
    bert_feature_dim: int = 1_024
    maximum_generated_tokens: int = 1_500

    def __post_init__(self) -> None:
        if self.version not in _S1_PHONEME_VOCABULARY:
            raise ValueError("GPT-SoVITS S1 `version` must be v1 or v2.")
        for name in (
                "vocabulary_size",
                "phoneme_vocabulary_size",
                "embedding_dim",
                "hidden_dim",
                "attention_heads",
                "layers",
                "bert_feature_dim",
                "maximum_generated_tokens",
        ):
            _positive(name, getattr(self, name))
        expected_phonemes = _S1_PHONEME_VOCABULARY[self.version]
        if self.phoneme_vocabulary_size != expected_phonemes:
            raise ValueError(
                f"GPT-SoVITS {self.version} S1 requires "
                f"{expected_phonemes} phoneme embeddings.")
        if self.embedding_dim != self.hidden_dim:
            raise ValueError("The audited S1 checkpoints require equal embedding and hidden dimensions.")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("S1 hidden size must be divisible by its attention heads.")
        if self.eos_token_id != self.vocabulary_size - 1:
            raise ValueError("S1 EOS must be the final semantic vocabulary ID.")
        if self.dropout != 0.0:
            raise ValueError("The public S1 configurations declare dropout=0.")

    @classmethod
    def for_variant(cls, variant: object) -> GPTSoVITSS1Config:
        version = s1_variant_for_s2(variant)
        return cls(
            version=version,
            phoneme_vocabulary_size=_S1_PHONEME_VOCABULARY[version],
        )

    @classmethod
    def from_upstream(
        cls,
        payload: dict[str, Any],
        *,
        variant: object = "v2",
    ) -> GPTSoVITSS1Config:
        version = s1_variant_for_s2(variant)
        model = payload.get("model")
        if not isinstance(model, dict):
            raise ValueError("GPT-SoVITS S1 config is missing its `model` mapping.")
        expected = {
            "vocab_size": 1_025,
            "phoneme_vocab_size": _S1_PHONEME_VOCABULARY[version],
            "embedding_dim": 512,
            "hidden_dim": 512,
            "head": 16,
            "n_layer": 24,
            "dropout": 0,
            "EOS": 1_024,
        }
        mismatches = {
            key: (model.get(key), value)
            for key, value in expected.items() if model.get(key) != value
        }
        if mismatches:
            raise ValueError(f"GPT-SoVITS {version} S1 topology mismatch: {mismatches}.")
        return cls.for_variant(version)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GPTSoVITSS1Config:
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class GPTSoVITSS2Config:
    """Exact public V1/V2/V2Pro/V2ProPlus classic SoVITS topology."""

    version: str = "v2"
    spectrogram_channels: int = 1_025
    style_channels: int = 704
    segment_size: int = 20_480
    segment_frames: int = 32
    sample_rate: int = 32_000
    filter_length: int = 2_048
    hop_length: int = 640
    window_length: int = 2_048
    mel_channels: int = 128
    mel_min_frequency: float = 0.0
    mel_max_frequency: float | None = None
    phoneme_vocabulary_size: int = 732
    ssl_channels: int = 768
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    attention_heads: int = 2
    layers: int = 6
    kernel_size: int = 3
    dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    upsample_rates: tuple[int, ...] = (10, 8, 2, 2, 2)
    upsample_initial_channels: int = 512
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 8, 2, 2)
    posterior_layers: int = 16
    gin_channels: int = 512
    semantic_frame_rate: str = "25hz"
    freeze_quantizer: bool = True
    use_spectral_norm: bool = False
    speaker_embedding_dim: int | None = None
    discriminator_periods: tuple[int, ...] = (2, 3, 5, 7, 11)

    def __post_init__(self) -> None:
        canonical = normalize_gptsovits_variant(self.version)
        if canonical != self.version:
            raise ValueError(f"Use canonical GPT-SoVITS variant name {canonical!r}.")
        for name in (
                "spectrogram_channels",
                "style_channels",
                "segment_size",
                "segment_frames",
                "sample_rate",
                "filter_length",
                "hop_length",
                "window_length",
                "mel_channels",
                "phoneme_vocabulary_size",
                "ssl_channels",
                "inter_channels",
                "hidden_channels",
                "filter_channels",
                "attention_heads",
                "layers",
                "kernel_size",
                "upsample_initial_channels",
                "posterior_layers",
                "gin_channels",
        ):
            _positive(name, getattr(self, name))
        if self.speaker_embedding_dim is not None:
            _positive("speaker_embedding_dim", self.speaker_embedding_dim)
        if self.spectrogram_channels != self.filter_length // 2 + 1:
            raise ValueError("S2 spectrogram channels must equal filter_length // 2 + 1.")
        if self.segment_frames != self.segment_size // self.hop_length:
            raise ValueError("S2 segment frames must equal segment_size // hop_length.")
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("S2 upsample rate/kernel lists must have equal length.")
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilation_sizes):
            raise ValueError("S2 residual kernel/dilation lists must have equal length.")
        if self.semantic_frame_rate != "25hz" or not self.freeze_quantizer:
            raise ValueError("The supported public S2 generators use a frozen 25 Hz quantizer.")
        expected = _S2_VARIANT_TOPOLOGY[self.version]
        mismatches = {
            name: (getattr(self, name), value)
            for name, value in expected.items() if getattr(self, name) != value
        }
        if mismatches:
            raise ValueError(f"GPT-SoVITS {self.version} S2 topology mismatch: {mismatches}.")

    @property
    def requires_speaker_embedding(self) -> bool:
        return self.speaker_embedding_dim is not None

    @classmethod
    def for_variant(cls, variant: object) -> GPTSoVITSS2Config:
        canonical = normalize_gptsovits_variant(variant)
        return cls(version=canonical, **_S2_VARIANT_TOPOLOGY[canonical])

    @classmethod
    def from_upstream(
        cls,
        payload: dict[str, Any],
        *,
        variant: object = "v2",
    ) -> GPTSoVITSS2Config:
        canonical = normalize_gptsovits_variant(variant)
        train = payload.get("train")
        data = payload.get("data")
        model = payload.get("model")
        if not all(isinstance(item, dict) for item in (train, data, model)):
            raise ValueError("GPT-SoVITS S2 config must contain train/data/model mappings.")
        topology = _S2_VARIANT_TOPOLOGY[canonical]
        expected = {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": topology["dropout"],
            "resblock": "1",
            "upsample_initial_channel": topology["upsample_initial_channels"],
            "gin_channels": topology["gin_channels"],
            "semantic_frame_rate": "25hz",
            "freeze_quantizer": True,
        }
        mismatches = {
            key: (model.get(key), value)
            for key, value in expected.items() if model.get(key) != value
        }
        declared_version = model.get("version")
        if declared_version is not None:
            try:
                declared_version = normalize_gptsovits_variant(declared_version)
            except ValueError:
                mismatches["version"] = (model.get("version"), canonical)
            else:
                if declared_version != canonical:
                    mismatches["version"] = (declared_version, canonical)
        data_expected = {
            "sampling_rate": 32_000,
            "filter_length": 2_048,
            "hop_length": 640,
            "win_length": 2_048,
            "n_mel_channels": 128,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "n_speakers": 300,
        }
        mismatches.update({
            f"data.{key}": (data.get(key), value)
            for key, value in data_expected.items() if data.get(key) != value
        })
        if train.get("segment_size") != 20_480:
            mismatches["train.segment_size"] = (train.get("segment_size"), 20_480)
        for name, expected_value in (("c_mel", 45), ("c_kl", 1.0)):
            if train.get(name) != expected_value:
                mismatches[f"train.{name}"] = (train.get(name), expected_value)
        expected_kernels = topology["upsample_kernel_sizes"]
        structural = {
            "resblock_kernel_sizes": (3, 7, 11),
            "resblock_dilation_sizes": ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            "upsample_rates": (10, 8, 2, 2, 2),
            "upsample_kernel_sizes": expected_kernels,
        }
        for name, expected_value in structural.items():
            actual = tuple(
                tuple(item) if isinstance(item, (list, tuple)) else item for item in model.get(name, ()))
            if actual != expected_value:
                mismatches[name] = (actual, expected_value)
        if model.get("use_spectral_norm") is not False:
            mismatches["use_spectral_norm"] = (
                model.get("use_spectral_norm"),
                False,
            )
        if mismatches:
            raise ValueError(f"GPT-SoVITS {canonical} S2 topology mismatch: {mismatches}.")
        return cls.for_variant(canonical)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GPTSoVITSS2Config:
        normalized = dict(payload)
        for name in (
                "resblock_kernel_sizes",
                "upsample_rates",
                "upsample_kernel_sizes",
                "discriminator_periods",
        ):
            if name in normalized:
                normalized[name] = tuple(normalized[name])
        if "resblock_dilation_sizes" in normalized:
            normalized["resblock_dilation_sizes"] = tuple(
                tuple(item) for item in normalized["resblock_dilation_sizes"])
        return cls(**normalized)


__all__ = [
    "GPTSoVITSS1Config",
    "GPTSoVITSS2Config",
    "SUPPORTED_GPT_SOVITS_VARIANTS",
    "UNSUPPORTED_GPT_SOVITS_VARIANTS",
    "normalize_gptsovits_variant",
    "s1_variant_for_s2",
]
