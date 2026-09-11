"""Coherent local and Hub artifact resolution for native Bark."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.path_utils import is_explicit_local_path

from .metadata import (
    BARK_CHECKPOINT,
    BARK_CHECKPOINT_FILE,
    BARK_CHECKPOINT_REVISION,
    BARK_CONFIG_SHA256,
    BARK_GENERATION_CONFIG_SHA256,
    BARK_SPEAKER_INDEX_SHA256,
    BARK_TOKENIZER_CONFIG_SHA256,
    BARK_TOKENIZER_SHA256,
    BARK_VOCAB_SHA256,
)

_CONFIG = "config.json"
_GENERATION_CONFIG = "generation_config.json"
_TOKENIZER = "tokenizer.json"
_TOKENIZER_CONFIG = "tokenizer_config.json"
_VOCAB = "vocab.txt"
_SPEAKER_INDEX = "speaker_embeddings_path.json"
_NATIVE_CHECKPOINT = "model.safetensors"


@dataclass(frozen=True, slots=True)
class BarkArtifacts:
    """Files required for one Bark runtime."""

    source: str
    revision: str | None
    config: Path
    generation_config: Path
    tokenizer: Path
    tokenizer_config: Path
    vocabulary: Path
    speaker_index: Path
    checkpoint: Path | None
    official_snapshot: bool
    legacy_checkpoint: bool


def _required(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Native Bark requires {name!r} in {root}.")
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_bark_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    allow_legacy_checkpoint: bool = False,
    verify_integrity: bool = True,
) -> BarkArtifacts:
    """Resolve one internally consistent Bark artifact set.

    The official release has no Safetensors file. Its legacy checkpoint
    is resolved only when the caller explicitly opts into the conversion
    path.
    """
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Bark source must be a non-empty path or Hub ID.")
    local = Path(source).expanduser()
    if local.exists():
        if local.is_file():
            if local.suffix.lower() not in {".safetensors", ".bin"}:
                raise ValueError("Bark checkpoint files must use `.safetensors` or `.bin`.")
            if local.suffix.lower() == ".bin" and not allow_legacy_checkpoint:
                raise PermissionError(
                    "Loading a Bark `.bin` checkpoint requires explicit legacy "
                    "conversion trust.")
            root = local.resolve().parent
            checkpoint = local.resolve()
        else:
            root = local.resolve()
            native = root / _NATIVE_CHECKPOINT
            legacy = root / BARK_CHECKPOINT_FILE
            if native.is_file():
                checkpoint = native.resolve()
            elif legacy.is_file() and allow_legacy_checkpoint:
                checkpoint = legacy.resolve()
            elif legacy.is_file():
                checkpoint = None
            else:
                raise FileNotFoundError(
                    f"Native Bark requires {_NATIVE_CHECKPOINT!r} in {root}; "
                    "the pinned legacy artifact needs explicit conversion trust.")
        artifacts = BarkArtifacts(
            source=str(root),
            revision=None,
            config=_required(root, _CONFIG),
            generation_config=_required(root, _GENERATION_CONFIG),
            tokenizer=_required(root, _TOKENIZER),
            tokenizer_config=_required(root, _TOKENIZER_CONFIG),
            vocabulary=_required(root, _VOCAB),
            speaker_index=_required(root, _SPEAKER_INDEX),
            checkpoint=checkpoint,
            official_snapshot=False,
            legacy_checkpoint=(checkpoint is not None and checkpoint.suffix.lower() == ".bin"),
        )
    else:
        if is_explicit_local_path(source):
            raise FileNotFoundError(f"Bark model path was not found: {local}.")
        repo_id = str(source)
        pinned_revision = revision or (BARK_CHECKPOINT_REVISION if repo_id == BARK_CHECKPOINT else None)

        def resolve(name: str) -> Path:
            return resolve_pretrained_file(
                repo_id,
                name,
                revision=pinned_revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )

        official = (repo_id == BARK_CHECKPOINT and pinned_revision == BARK_CHECKPOINT_REVISION)
        checkpoint = None
        legacy = False
        if allow_legacy_checkpoint:
            checkpoint = resolve(BARK_CHECKPOINT_FILE)
            legacy = True
        else:
            try:
                checkpoint = resolve(_NATIVE_CHECKPOINT)
            except (FileNotFoundError, OSError):
                if not official:
                    raise
        artifacts = BarkArtifacts(
            source=repo_id,
            revision=pinned_revision,
            config=resolve(_CONFIG),
            generation_config=resolve(_GENERATION_CONFIG),
            tokenizer=resolve(_TOKENIZER),
            tokenizer_config=resolve(_TOKENIZER_CONFIG),
            vocabulary=resolve(_VOCAB),
            speaker_index=resolve(_SPEAKER_INDEX),
            checkpoint=checkpoint,
            official_snapshot=official,
            legacy_checkpoint=legacy,
        )
    if verify_integrity and artifacts.official_snapshot:
        expected = {
            artifacts.config: BARK_CONFIG_SHA256,
            artifacts.generation_config: BARK_GENERATION_CONFIG_SHA256,
            artifacts.tokenizer: BARK_TOKENIZER_SHA256,
            artifacts.tokenizer_config: BARK_TOKENIZER_CONFIG_SHA256,
            artifacts.vocabulary: BARK_VOCAB_SHA256,
            artifacts.speaker_index: BARK_SPEAKER_INDEX_SHA256,
        }
        for path, digest in expected.items():
            actual = _sha256(path)
            if actual != digest:
                raise ValueError(f"Bark artifact {path.name!r} has SHA-256 {actual}, "
                                 f"expected {digest}.")
    return artifacts


__all__ = ["BarkArtifacts", "resolve_bark_artifacts"]
