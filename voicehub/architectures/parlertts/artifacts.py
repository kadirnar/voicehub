"""Coherent local and Hub artifact resolution for native Parler-TTS."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voicehub.architectures.parlertts.metadata import (
    PARLER_TTS_CHECKPOINT,
    PARLER_TTS_CHECKPOINT_FILE,
    PARLER_TTS_CHECKPOINT_REVISION,
    PARLER_TTS_CHECKPOINT_SHA256,
    PARLER_TTS_CONFIG_SHA256,
    PARLER_TTS_GENERATION_CONFIG_SHA256,
    PARLER_TTS_SENTENCEPIECE_SHA256,
)
from voicehub.hub import resolve_pretrained_file
from voicehub.path_utils import is_explicit_local_path

_CONFIG_FILE = "config.json"
_GENERATION_CONFIG_FILE = "generation_config.json"
_TOKENIZER_MODEL_FILE = "spiece.model"


@dataclass(frozen=True, slots=True)
class ParlerTTSArtifacts:
    """Immutable files that form one loadable Parler-TTS runtime."""

    source: str
    revision: str | None
    config: Path
    generation_config: Path
    tokenizer_model: Path
    checkpoint: Path
    official_snapshot: bool


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native Parler-TTS requires {filename!r} in {root}.")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify(
    artifacts: ParlerTTSArtifacts,
    *,
    include_checkpoint: bool,
) -> None:
    expected = {
        artifacts.config: PARLER_TTS_CONFIG_SHA256,
        artifacts.generation_config: PARLER_TTS_GENERATION_CONFIG_SHA256,
        artifacts.tokenizer_model: PARLER_TTS_SENTENCEPIECE_SHA256,
    }
    if include_checkpoint:
        expected[artifacts.checkpoint] = PARLER_TTS_CHECKPOINT_SHA256
    for path, digest in expected.items():
        actual = _sha256(path)
        if actual != digest:
            raise ValueError(
                f"Parler-TTS artifact {path.name!r} has SHA-256 {actual}, "
                f"expected {digest}.")


def resolve_parlertts_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = False,
    verify_checkpoint_integrity: bool = False,
) -> ParlerTTSArtifacts:
    """Resolve a complete artifact snapshot at one immutable Hub revision."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Parler-TTS source must be a non-empty path or Hub ID.")
    local = Path(source).expanduser()
    if local.exists():
        checkpoint = local if local.is_file() else _required(
            local,
            PARLER_TTS_CHECKPOINT_FILE,
        )
        if checkpoint.suffix.lower() != ".safetensors":
            raise ValueError("Native Parler-TTS checkpoints use Safetensors.")
        root = checkpoint.parent
        artifacts = ParlerTTSArtifacts(
            source=str(local.absolute()),
            revision=None,
            config=_required(root, _CONFIG_FILE),
            generation_config=_required(root, _GENERATION_CONFIG_FILE),
            tokenizer_model=_required(root, _TOKENIZER_MODEL_FILE),
            checkpoint=checkpoint.absolute(),
            official_snapshot=False,
        )
    else:
        if is_explicit_local_path(source):
            raise FileNotFoundError(f"Parler-TTS model path was not found: {local}.")
        repo_id = str(source)
        pinned_revision = revision or (
            PARLER_TTS_CHECKPOINT_REVISION if repo_id == PARLER_TTS_CHECKPOINT else None)

        def resolve(filename: str) -> Path:
            return resolve_pretrained_file(
                repo_id,
                filename,
                revision=pinned_revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )

        artifacts = ParlerTTSArtifacts(
            source=repo_id,
            revision=pinned_revision,
            config=resolve(_CONFIG_FILE),
            generation_config=resolve(_GENERATION_CONFIG_FILE),
            tokenizer_model=resolve(_TOKENIZER_MODEL_FILE),
            checkpoint=resolve(PARLER_TTS_CHECKPOINT_FILE),
            official_snapshot=(
                repo_id == PARLER_TTS_CHECKPOINT and pinned_revision == PARLER_TTS_CHECKPOINT_REVISION),
        )
    if (artifacts.official_snapshot and (verify_integrity or verify_checkpoint_integrity)):
        _verify(
            artifacts,
            include_checkpoint=verify_checkpoint_integrity,
        )
    return artifacts


__all__ = ["ParlerTTSArtifacts", "resolve_parlertts_artifacts"]
