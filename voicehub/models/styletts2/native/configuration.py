"""Typed, dependency-free configuration for StyleTTS 2."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voicehub.models.kokoro.native.configuration import KokoroAlbertConfig
from voicehub.models.styletts2.native.metadata import (
    STYLETTS2_LEGACY_CONFIG_SHA256,
    STYLETTS2_SINGLE_SPEAKER_CONFIG_SHA256,
)


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _probability(name: str, value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or
            not 0.0 <= float(value) < 1.0):
        raise ValueError(f"`{name}` must be finite and in [0, 1).")
    return float(value)


def _integer_tuple(name: str, value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence.")
    result = tuple(value)
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    for item in result:
        _positive_integer(name, item)
    return result


def _nested_integer_tuple(
    name: str,
    value: Any,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a nested sequence.")
    result = tuple(_integer_tuple(f"{name}[{index}]", item) for index, item in enumerate(value))
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    return result


@dataclass(frozen=True, slots=True)
class StyleTTS2DecoderConfig:
    """Released LibriTTS HiFi-GAN decoder dimensions."""

    type: str = "hifigan"
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    upsample_rates: tuple[int, ...] = (10, 5, 3, 2)
    upsample_initial_channel: int = 512
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    upsample_kernel_sizes: tuple[int, ...] = (20, 10, 6, 4)
    gen_istft_n_fft: int | None = None
    gen_istft_hop_size: int | None = None

    def __post_init__(self) -> None:
        if self.type not in {"hifigan", "istftnet"}:
            raise ValueError("StyleTTS 2 decoder type is not recognized.")
        object.__setattr__(
            self,
            "resblock_kernel_sizes",
            _integer_tuple(
                "resblock_kernel_sizes",
                self.resblock_kernel_sizes,
            ),
        )
        object.__setattr__(
            self,
            "upsample_rates",
            _integer_tuple("upsample_rates", self.upsample_rates),
        )
        object.__setattr__(
            self,
            "resblock_dilation_sizes",
            _nested_integer_tuple(
                "resblock_dilation_sizes",
                self.resblock_dilation_sizes,
            ),
        )
        object.__setattr__(
            self,
            "upsample_kernel_sizes",
            _integer_tuple(
                "upsample_kernel_sizes",
                self.upsample_kernel_sizes,
            ),
        )
        _positive_integer(
            "upsample_initial_channel",
            self.upsample_initial_channel,
        )
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("Decoder upsample rates and kernels must have equal lengths.")
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilation_sizes):
            raise ValueError("Decoder residual kernels and dilation groups must align.")
        if self.type == "hifigan":
            if (self.gen_istft_n_fft is not None or self.gen_istft_hop_size is not None):
                raise ValueError("HiFi-GAN config cannot declare iSTFT dimensions.")
        else:
            _positive_integer("gen_istft_n_fft", self.gen_istft_n_fft)
            _positive_integer(
                "gen_istft_hop_size",
                self.gen_istft_hop_size,
            )
            if self.gen_istft_hop_size > self.gen_istft_n_fft:
                raise ValueError("iSTFT hop size cannot exceed its FFT size.")

    @classmethod
    def released_istftnet(cls) -> StyleTTS2DecoderConfig:
        return cls(
            type="istftnet",
            upsample_rates=(10, 6),
            upsample_kernel_sizes=(20, 12),
            gen_istft_n_fft=20,
            gen_istft_hop_size=5,
        )


@dataclass(frozen=True, slots=True)
class StyleTTS2TransformerConfig:
    num_layers: int = 3
    num_heads: int = 8
    head_features: int = 64
    multiplier: int = 2

    def __post_init__(self) -> None:
        for name in (
                "num_layers",
                "num_heads",
                "head_features",
                "multiplier",
        ):
            _positive_integer(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class StyleTTS2DistributionConfig:
    sigma_data: float = 0.2
    estimate_sigma_data: bool = True
    mean: float = -3.0
    std: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.estimate_sigma_data, bool):
            raise TypeError("`estimate_sigma_data` must be a boolean.")
        for name in ("sigma_data", "mean", "std"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value))):
                raise ValueError(f"`{name}` must be finite.")
        if self.sigma_data <= 0 or self.std <= 0:
            raise ValueError("Diffusion sigma data/std must be positive.")


@dataclass(frozen=True, slots=True)
class StyleTTS2DiffusionConfig:
    embedding_mask_proba: float = 0.1
    transformer: StyleTTS2TransformerConfig = StyleTTS2TransformerConfig()
    dist: StyleTTS2DistributionConfig = StyleTTS2DistributionConfig()

    def __post_init__(self) -> None:
        _probability("embedding_mask_proba", self.embedding_mask_proba)
        if not isinstance(self.transformer, StyleTTS2TransformerConfig):
            raise TypeError("`transformer` must be a typed configuration.")
        if not isinstance(self.dist, StyleTTS2DistributionConfig):
            raise TypeError("`dist` must be a typed configuration.")


@dataclass(frozen=True, slots=True)
class StyleTTS2SLMConfig:
    hidden: int = 768
    nlayers: int = 13
    initial_channel: int = 64

    def __post_init__(self) -> None:
        for name in ("hidden", "nlayers", "initial_channel"):
            _positive_integer(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class StyleTTS2ArchitectureConfig:
    """All dimensions needed to reconstruct the released deployable graph."""

    sample_rate: int = 24_000
    n_fft: int = 2_048
    win_length: int = 1_200
    hop_length: int = 300
    multispeaker: bool = True
    dim_in: int = 64
    hidden_dim: int = 512
    max_conv_dim: int = 512
    n_layer: int = 3
    n_mels: int = 80
    n_token: int = 178
    max_dur: int = 50
    style_dim: int = 128
    dropout: float = 0.2
    decoder: StyleTTS2DecoderConfig = StyleTTS2DecoderConfig()
    diffusion: StyleTTS2DiffusionConfig = StyleTTS2DiffusionConfig()
    slm: StyleTTS2SLMConfig = StyleTTS2SLMConfig()
    plbert: KokoroAlbertConfig = KokoroAlbertConfig(source_dropout=0.1)

    def __post_init__(self) -> None:
        for name in (
                "sample_rate",
                "n_fft",
                "win_length",
                "hop_length",
                "dim_in",
                "hidden_dim",
                "max_conv_dim",
                "n_layer",
                "n_mels",
                "n_token",
                "max_dur",
                "style_dim",
        ):
            _positive_integer(name, getattr(self, name))
        if not isinstance(self.multispeaker, bool):
            raise TypeError("`multispeaker` must be a boolean.")
        _probability("dropout", self.dropout)
        if self.win_length > self.n_fft:
            raise ValueError("`win_length` cannot exceed `n_fft`.")
        if self.hop_length > self.win_length:
            raise ValueError("`hop_length` cannot exceed `win_length`.")
        if self.n_token != self.plbert.vocab_size:
            raise ValueError("Text encoder and PL-BERT vocabularies must match.")

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> StyleTTS2ArchitectureConfig:
        if not isinstance(values, Mapping):
            raise TypeError("StyleTTS 2 configuration must be a mapping.")
        root = dict(values)
        legacy_layout = "model_params" in root
        model = dict(root["model_params"] if legacy_layout else root)
        preprocess = root.get("preprocess_params", {})
        if not isinstance(preprocess, Mapping):
            raise TypeError("`preprocess_params` must be a mapping.")
        spect = preprocess.get("spect_params", {})
        if not isinstance(spect, Mapping):
            raise TypeError("`spect_params` must be a mapping.")

        decoder_value = model.pop("decoder", {})
        diffusion_value = model.pop("diffusion", {})
        slm_value = model.pop("slm", {})
        plbert_value = root.get("plbert", {})
        if not legacy_layout:
            for key in (
                    "sample_rate",
                    "n_fft",
                    "win_length",
                    "hop_length",
                    "plbert",
            ):
                model.pop(key, None)
        if not all(isinstance(item, Mapping) for item in (
                decoder_value,
                diffusion_value,
                slm_value,
                plbert_value,
        )):
            raise TypeError("Nested StyleTTS 2 configs must be mappings.")
        diffusion_value = dict(diffusion_value)
        transformer_value = diffusion_value.pop("transformer", {})
        distribution_value = diffusion_value.pop("dist", {})
        if not isinstance(transformer_value, Mapping) or not isinstance(
                distribution_value,
                Mapping,
        ):
            raise TypeError("Diffusion transformer/dist must be mappings.")

        known_model = {
            "multispeaker",
            "dim_in",
            "hidden_dim",
            "max_conv_dim",
            "n_layer",
            "n_mels",
            "n_token",
            "max_dur",
            "style_dim",
            "dropout",
        }
        unknown = set(model) - known_model
        if unknown:
            raise ValueError(
                "Unknown StyleTTS 2 model configuration key(s): " + ", ".join(sorted(unknown)) + ".")
        known_diffusion = {"embedding_mask_proba"}
        unknown_diffusion = set(diffusion_value) - known_diffusion
        if unknown_diffusion:
            raise ValueError(
                "Unknown StyleTTS 2 diffusion key(s): " + ", ".join(sorted(unknown_diffusion)) + ".")
        return cls(
            sample_rate=int(preprocess.get("sr", root.get("sample_rate", 24_000))),
            n_fft=int(spect.get("n_fft", root.get("n_fft", 2_048))),
            win_length=int(spect.get("win_length", root.get("win_length", 1_200))),
            hop_length=int(spect.get("hop_length", root.get("hop_length", 300))),
            decoder=StyleTTS2DecoderConfig(**dict(decoder_value)),
            diffusion=StyleTTS2DiffusionConfig(
                transformer=StyleTTS2TransformerConfig(**dict(transformer_value)),
                dist=StyleTTS2DistributionConfig(**dict(distribution_value)),
                **diffusion_value,
            ),
            slm=StyleTTS2SLMConfig(
                **{
                    key: value
                    for key, value in slm_value.items() if key in {"hidden", "nlayers", "initial_channel"}
                }),
            plbert=KokoroAlbertConfig.from_dict(
                plbert_value,
                vocab_size=int(model.get("n_token", 178)),
            ),
            **model,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "multispeaker": self.multispeaker,
            "dim_in": self.dim_in,
            "hidden_dim": self.hidden_dim,
            "max_conv_dim": self.max_conv_dim,
            "n_layer": self.n_layer,
            "n_mels": self.n_mels,
            "n_token": self.n_token,
            "max_dur": self.max_dur,
            "style_dim": self.style_dim,
            "dropout": self.dropout,
            "decoder": {
                "type": self.decoder.type,
                "resblock_kernel_sizes": list(self.decoder.resblock_kernel_sizes),
                "upsample_rates": list(self.decoder.upsample_rates),
                "upsample_initial_channel": (self.decoder.upsample_initial_channel),
                "resblock_dilation_sizes": [list(group) for group in self.decoder.resblock_dilation_sizes],
                "upsample_kernel_sizes": list(self.decoder.upsample_kernel_sizes),
                "gen_istft_n_fft": self.decoder.gen_istft_n_fft,
                "gen_istft_hop_size": self.decoder.gen_istft_hop_size,
            },
            "diffusion": {
                "embedding_mask_proba": (self.diffusion.embedding_mask_proba),
                "transformer": {
                    "num_layers": self.diffusion.transformer.num_layers,
                    "num_heads": self.diffusion.transformer.num_heads,
                    "head_features": (self.diffusion.transformer.head_features),
                    "multiplier": self.diffusion.transformer.multiplier,
                },
                "dist": {
                    "sigma_data": self.diffusion.dist.sigma_data,
                    "estimate_sigma_data": (self.diffusion.dist.estimate_sigma_data),
                    "mean": self.diffusion.dist.mean,
                    "std": self.diffusion.dist.std,
                },
            },
            "slm": {
                "hidden": self.slm.hidden,
                "nlayers": self.slm.nlayers,
                "initial_channel": self.slm.initial_channel,
            },
            "plbert": self.plbert.to_dict(),
        }


def load_styletts2_config(path: str | Path | None, ) -> StyleTTS2ArchitectureConfig:
    """Load typed JSON or recognize a pinned upstream YAML profile."""
    if path is None:
        return StyleTTS2ArchitectureConfig()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"StyleTTS 2 configuration was not found: {source}.")
    if source.suffix.lower() == ".json":
        try:
            values = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"StyleTTS 2 JSON is invalid: {error}.") from error
        return StyleTTS2ArchitectureConfig.from_dict(values)
    if source.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("StyleTTS 2 config must be JSON or pinned YAML.")
    # Git may materialize tracked text files with CRLF on Windows. Pin the
    # configuration content rather than the checkout's newline convention.
    canonical_bytes = source.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    if digest not in STYLETTS2_LEGACY_CONFIG_SHA256:
        raise ValueError(
            "Unpinned YAML cannot be interpreted without a YAML runtime. "
            "Convert the configuration to typed VoiceHub JSON explicitly.")
    if digest == STYLETTS2_SINGLE_SPEAKER_CONFIG_SHA256:
        return StyleTTS2ArchitectureConfig(
            multispeaker=False,
            decoder=StyleTTS2DecoderConfig.released_istftnet(),
        )
    return StyleTTS2ArchitectureConfig()


__all__ = [
    "StyleTTS2ArchitectureConfig",
    "StyleTTS2DecoderConfig",
    "StyleTTS2DiffusionConfig",
    "StyleTTS2DistributionConfig",
    "StyleTTS2SLMConfig",
    "StyleTTS2TransformerConfig",
    "load_styletts2_config",
]
