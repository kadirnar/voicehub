"""Pinned, safe artifact resolution for native Zonos v0.1."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voicehub.checkpointing.errors import CheckpointIntegrityError
from voicehub.hub import resolve_pretrained_file
from voicehub.models.zonos.native.metadata import (
    ZONOS_HYBRID_REPOSITORY,
    ZONOS_TRANSFORMER_CHECKPOINT_SHA256,
    ZONOS_TRANSFORMER_CHECKPOINT_SIZE,
    ZONOS_TRANSFORMER_CONFIG_SHA256,
    ZONOS_TRANSFORMER_REPOSITORY,
    ZONOS_TRANSFORMER_REVISION,
)


@dataclass(frozen=True, slots=True)
class ZonosArtifacts:
    """Resolved immutable inputs for one native runtime."""

    config: Path
    checkpoint: Path
    source: str
    revision: str | None


def verify_zonos_file(
    path: str | Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    source = Path(path).expanduser().resolve()
    if expected_size is not None and source.stat().st_size != expected_size:
        raise CheckpointIntegrityError(
            f"{source.name} has size {source.stat().st_size}; expected "
            f"{expected_size}.")
    if expected_sha256 is not None:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while block := stream.read(chunk_size):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise CheckpointIntegrityError(
                f"{source.name} has SHA-256 {actual}; expected "
                f"{expected_sha256}.")


def resolve_zonos_artifacts(
    pretrained_model_name_or_path: str | Path = ZONOS_TRANSFORMER_REPOSITORY,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = False,
) -> ZonosArtifacts:
    """Resolve ``config.json`` plus an exact Safetensors checkpoint."""
    source_path = Path(pretrained_model_name_or_path).expanduser()
    if source_path.is_dir():
        config = source_path / "config.json"
        checkpoint = source_path / "model.safetensors"
        missing = [path.name for path in (config, checkpoint) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Zonos directory {source_path} is missing {missing!r}.")
        return ZonosArtifacts(
            config=config.resolve(),
            checkpoint=checkpoint.resolve(),
            source=str(source_path),
            revision=None,
        )
    if source_path.is_file():
        if source_path.suffix.lower() != ".safetensors":
            raise ValueError("Native Zonos loads Safetensors only; received "
                             f"{source_path.name!r}.")
        config = source_path.with_name("config.json")
        if not config.is_file():
            raise FileNotFoundError(f"Zonos configuration was not found beside {source_path}.")
        return ZonosArtifacts(
            config=config.resolve(),
            checkpoint=source_path.resolve(),
            source=str(source_path),
            revision=None,
        )

    source = str(pretrained_model_name_or_path)
    if source == ZONOS_HYBRID_REPOSITORY:
        raise NotImplementedError(
            "Native Zonos v0.1 currently supports the dense Transformer "
            "checkpoint only. The hybrid repository requires the published "
            "Mamba-2 state-space graph and is intentionally not loaded.")
    resolved_revision = (
        ZONOS_TRANSFORMER_REVISION
        if source == ZONOS_TRANSFORMER_REPOSITORY and revision is None else revision)
    common = {
        "cache_dir": cache_dir,
        "revision": resolved_revision,
        "token": token,
        "local_files_only": local_files_only,
    }
    config = resolve_pretrained_file(
        source,
        "config.json",
        **common,
    )
    checkpoint = resolve_pretrained_file(
        source,
        "model.safetensors",
        **common,
    )
    if (verify_integrity and source == ZONOS_TRANSFORMER_REPOSITORY and
            resolved_revision == ZONOS_TRANSFORMER_REVISION):
        verify_zonos_file(
            config,
            expected_sha256=ZONOS_TRANSFORMER_CONFIG_SHA256,
        )
        verify_zonos_file(
            checkpoint,
            expected_size=ZONOS_TRANSFORMER_CHECKPOINT_SIZE,
            expected_sha256=ZONOS_TRANSFORMER_CHECKPOINT_SHA256,
        )
    return ZonosArtifacts(
        config=config,
        checkpoint=checkpoint,
        source=source,
        revision=resolved_revision,
    )


__all__ = [
    "ZonosArtifacts",
    "resolve_zonos_artifacts",
    "verify_zonos_file",
]
