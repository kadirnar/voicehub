"""Validated architecture values extracted from Supertonic ``tts.json``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Supertonic config section {name!r} is missing.")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Supertonic config {name!r} must be positive.")
    return value


@dataclass(frozen=True, slots=True)
class SupertonicArchitectureConfig:
    """Runtime dimensions required outside the imported exact graphs."""

    version: str = "v1.7.3"
    sample_rate: int = 44_100
    base_chunk_size: int = 512
    autoencoder_compression: int = 1
    latent_dimension: int = 24
    text_to_latent_compression: int = 6

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Supertonic `version` must be non-empty.")
        for name in (
                "sample_rate",
                "base_chunk_size",
                "autoencoder_compression",
                "latent_dimension",
                "text_to_latent_compression",
        ):
            _positive_integer(getattr(self, name), name=name)

    @property
    def latent_channels(self) -> int:
        return self.latent_dimension * self.text_to_latent_compression

    @property
    def latent_hop_length(self) -> int:
        return (self.base_chunk_size * self.autoencoder_compression * self.text_to_latent_compression)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> SupertonicArchitectureConfig:
        if not isinstance(value, Mapping):
            raise TypeError("Supertonic architecture config must be a mapping.")
        autoencoder = _mapping(value.get("ae"), name="ae")
        text_to_latent = _mapping(value.get("ttl"), name="ttl")
        return cls(
            version=str(value.get("tts_version", "")),
            sample_rate=_positive_integer(
                autoencoder.get("sample_rate"),
                name="ae.sample_rate",
            ),
            base_chunk_size=_positive_integer(
                autoencoder.get("base_chunk_size"),
                name="ae.base_chunk_size",
            ),
            autoencoder_compression=_positive_integer(
                autoencoder.get("chunk_compress_factor"),
                name="ae.chunk_compress_factor",
            ),
            latent_dimension=_positive_integer(
                text_to_latent.get("latent_dim"),
                name="ttl.latent_dim",
            ),
            text_to_latent_compression=_positive_integer(
                text_to_latent.get("chunk_compress_factor"),
                name="ttl.chunk_compress_factor",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["SupertonicArchitectureConfig"]
