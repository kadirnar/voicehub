"""Immutable local/Hub artifact resolution for native VoxCPM2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.voxcpm.native.metadata import (
    VOXCPM2_CHECKPOINT_FILE,
    VOXCPM2_CHECKPOINT_REPOSITORY,
    VOXCPM2_CHECKPOINT_REVISION,
    VOXCPM2_CHECKPOINT_SHA256,
    VOXCPM2_CHECKPOINT_SIZE,
    VOXCPM2_CODEC_LEGACY_FILE,
    VOXCPM2_CODEC_LEGACY_SHA256,
    VOXCPM2_CODEC_LEGACY_SIZE,
    VOXCPM2_CODEC_NATIVE_FILE,
    VOXCPM2_CONFIG_FILE,
    VOXCPM2_CONFIG_SHA256,
    VOXCPM2_CONFIG_SIZE,
    VOXCPM2_TOKENIZER_FILE,
    VOXCPM2_TOKENIZER_SHA256,
    VOXCPM2_TOKENIZER_SIZE,
)
from voicehub.path_utils import is_explicit_local_path


@dataclass(frozen=True, slots=True)
class VoxCPM2Artifacts:
    source: str
    revision: str | None
    checkpoint: Path
    config: Path
    tokenizer: Path
    codec_checkpoint: Path
    legacy_codec: bool
    official: bool


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native VoxCPM2 requires {filename!r} in {root}.")
    return path.resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify(
    path: Path,
    *,
    size: int,
    sha256: str,
) -> None:
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ValueError(f"VoxCPM artifact {path.name!r} has size {actual_size}; expected {size}.")
    digest = _file_sha256(path)
    if digest != sha256:
        raise ValueError(f"VoxCPM artifact {path.name!r} has SHA-256 {digest}; expected {sha256}.")


def resolve_voxcpm2_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    codec_path: str | Path | None = None,
    allow_legacy_codec: bool = False,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = False,
    verify_checkpoint_integrity: bool = False,
) -> VoxCPM2Artifacts:
    """Resolve a complete runtime without importing a model framework."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("VoxCPM source must be a non-empty path or Hub ID.")
    if not isinstance(allow_legacy_codec, bool):
        raise TypeError("`allow_legacy_codec` must be a boolean.")
    local = Path(source).expanduser()
    if local.exists():
        checkpoint = (local.resolve() if local.is_file() else _required(local, VOXCPM2_CHECKPOINT_FILE))
        if checkpoint.suffix.lower() != ".safetensors":
            raise ValueError("Native VoxCPM model checkpoints must use Safetensors.")
        root = checkpoint.parent
        config = _required(root, VOXCPM2_CONFIG_FILE)
        tokenizer_path = _required(root, VOXCPM2_TOKENIZER_FILE)
        official = False
        resolved_revision = None
        source_name = str(local.resolve())
        if codec_path is None:
            native_codec = root / VOXCPM2_CODEC_NATIVE_FILE
            legacy_codec_path = root / VOXCPM2_CODEC_LEGACY_FILE
            if native_codec.is_file():
                codec = native_codec.resolve()
                legacy_codec = False
            elif legacy_codec_path.is_file() and allow_legacy_codec:
                codec = legacy_codec_path.resolve()
                legacy_codec = True
            elif legacy_codec_path.is_file():
                raise PermissionError(
                    "This VoxCPM directory contains only the upstream pickle "
                    "AudioVAE. Convert it explicitly or enable the reviewed "
                    "legacy conversion boundary.")
            else:
                raise FileNotFoundError(f"Native VoxCPM2 requires {VOXCPM2_CODEC_NATIVE_FILE!r}.")
    else:
        if is_explicit_local_path(source):
            raise FileNotFoundError(f"VoxCPM model path was not found: {local}.")
        source_name = str(source)
        resolved_revision = (
            revision or
            (VOXCPM2_CHECKPOINT_REVISION if source_name == VOXCPM2_CHECKPOINT_REPOSITORY else None))
        options = {
            "revision": resolved_revision,
            "cache_dir": cache_dir,
            "token": token,
            "local_files_only": local_files_only,
        }
        checkpoint = resolve_pretrained_file(
            source_name,
            VOXCPM2_CHECKPOINT_FILE,
            **options,
        )
        config = resolve_pretrained_file(
            source_name,
            VOXCPM2_CONFIG_FILE,
            **options,
        )
        tokenizer_path = resolve_pretrained_file(
            source_name,
            VOXCPM2_TOKENIZER_FILE,
            **options,
        )
        official = (
            source_name == VOXCPM2_CHECKPOINT_REPOSITORY and resolved_revision == VOXCPM2_CHECKPOINT_REVISION)
        if codec_path is None:
            if not allow_legacy_codec:
                raise PermissionError(
                    "The official VoxCPM2 release publishes AudioVAE as a "
                    "pickle archive. Pass an already converted "
                    "`audiovae.safetensors`, or explicitly permit its "
                    "digest-pinned one-time conversion.")
            codec = resolve_pretrained_file(
                source_name,
                VOXCPM2_CODEC_LEGACY_FILE,
                **options,
            )
            legacy_codec = True
    if codec_path is not None:
        codec = Path(codec_path).expanduser()
        if not codec.is_file():
            raise FileNotFoundError(f"VoxCPM AudioVAE was not found: {codec}.")
        codec = codec.resolve()
        legacy_codec = codec.suffix.lower() != ".safetensors"
        if legacy_codec and not allow_legacy_codec:
            raise PermissionError("Legacy VoxCPM AudioVAE input requires explicit opt-in.")
    if not legacy_codec and codec.suffix.lower() != ".safetensors":
        raise ValueError("Native VoxCPM AudioVAE checkpoints must use Safetensors.")
    if verify_integrity and official:
        _verify(
            config,
            size=VOXCPM2_CONFIG_SIZE,
            sha256=VOXCPM2_CONFIG_SHA256,
        )
        _verify(
            tokenizer_path,
            size=VOXCPM2_TOKENIZER_SIZE,
            sha256=VOXCPM2_TOKENIZER_SHA256,
        )
        if legacy_codec:
            _verify(
                codec,
                size=VOXCPM2_CODEC_LEGACY_SIZE,
                sha256=VOXCPM2_CODEC_LEGACY_SHA256,
            )
    if verify_checkpoint_integrity and official:
        _verify(
            checkpoint,
            size=VOXCPM2_CHECKPOINT_SIZE,
            sha256=VOXCPM2_CHECKPOINT_SHA256,
        )
    return VoxCPM2Artifacts(
        source=source_name,
        revision=resolved_revision,
        checkpoint=checkpoint.absolute(),
        config=config.resolve(),
        tokenizer=tokenizer_path.resolve(),
        codec_checkpoint=codec.absolute(),
        legacy_codec=legacy_codec,
        official=official,
    )


__all__ = ["VoxCPM2Artifacts", "resolve_voxcpm2_artifacts"]
