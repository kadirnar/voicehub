"""Coherent local and Hub artifact resolution for native CSM."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.csm.native.metadata import (
    CSM_CHECKPOINT_FILE,
    CSM_CHECKPOINT_REPOSITORY,
    CSM_CHECKPOINT_REVISION,
    CSM_CHECKPOINT_SHA256,
    CSM_CHECKPOINT_SIZE,
    CSM_TOKENIZER_FILE,
    CSM_TOKENIZER_SHA256,
    CSM_TOKENIZER_SIZE,
    MIMI_CHECKPOINT_FILE,
    MIMI_CHECKPOINT_REPOSITORY,
    MIMI_CHECKPOINT_REVISION,
    MIMI_CHECKPOINT_SHA256,
    MIMI_CHECKPOINT_SIZE,
)
from voicehub.path_utils import is_explicit_local_path


@dataclass(frozen=True, slots=True)
class CSMArtifacts:
    """Every immutable file required by a selected native runtime."""

    source: str
    revision: str | None
    checkpoint: Path
    tokenizer: Path
    config: Path | None
    codec_checkpoint: Path | None
    official_model: bool
    official_codec: bool


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native CSM requires {filename!r} in {root}.")
    return path.resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    size = path.stat().st_size
    if size != expected_size:
        raise ValueError(f"CSM artifact {path.name!r} has size {size}; expected "
                         f"{expected_size}.")
    actual = _file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"CSM artifact {path.name!r} has SHA-256 {actual}; expected "
                         f"{expected_sha256}.")


def resolve_csm_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    codec_path: str | Path | None = None,
    include_codec: bool = True,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = False,
    verify_checkpoint_integrity: bool = False,
) -> CSMArtifacts:
    """Resolve model, tokenizer, and optional Mimi at pinned revisions."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("CSM source must be a non-empty path or Hub ID.")
    if not isinstance(include_codec, bool):
        raise TypeError("`include_codec` must be a boolean.")
    local = Path(source).expanduser()
    if local.exists():
        checkpoint = (local.resolve() if local.is_file() else _required(local, CSM_CHECKPOINT_FILE))
        if checkpoint.suffix.lower() != ".safetensors":
            raise ValueError("Native CSM checkpoints must use Safetensors.")
        root = checkpoint.parent
        tokenizer = _required(root, CSM_TOKENIZER_FILE)
        candidate_config = root / "config.json"
        config = candidate_config.resolve() if candidate_config.is_file() else None
        official_model = False
        resolved_revision = None
        source_name = str(local.resolve())
    else:
        if is_explicit_local_path(source):
            raise FileNotFoundError(f"CSM model path was not found: {local}.")
        source_name = str(source)
        resolved_revision = (
            revision or (CSM_CHECKPOINT_REVISION if source_name == CSM_CHECKPOINT_REPOSITORY else None))
        checkpoint = resolve_pretrained_file(
            source_name,
            CSM_CHECKPOINT_FILE,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        tokenizer = resolve_pretrained_file(
            source_name,
            CSM_TOKENIZER_FILE,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        config = None
        official_model = (
            source_name == CSM_CHECKPOINT_REPOSITORY and resolved_revision == CSM_CHECKPOINT_REVISION)

    codec_checkpoint = None
    official_codec = False
    if codec_path is not None:
        codec_checkpoint = Path(codec_path).expanduser()
        if not codec_checkpoint.is_file():
            raise FileNotFoundError(f"CSM Mimi checkpoint was not found: {codec_checkpoint}.")
        if codec_checkpoint.suffix.lower() != ".safetensors":
            raise ValueError("Native CSM Mimi checkpoints use Safetensors.")
        codec_checkpoint = codec_checkpoint.resolve()
    elif include_codec:
        local_codec = checkpoint.parent / "mimi.safetensors"
        if local.exists() and local_codec.is_file():
            codec_checkpoint = local_codec.resolve()
        elif local.exists():
            raise FileNotFoundError(
                "A local CSM runtime requires `mimi.safetensors`, an "
                "explicit `codec_path`, or an injected codec.")
        else:
            codec_checkpoint = resolve_pretrained_file(
                MIMI_CHECKPOINT_REPOSITORY,
                MIMI_CHECKPOINT_FILE,
                revision=MIMI_CHECKPOINT_REVISION,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
            official_codec = True

    if verify_integrity and official_model:
        _verify_file(
            tokenizer,
            expected_size=CSM_TOKENIZER_SIZE,
            expected_sha256=CSM_TOKENIZER_SHA256,
        )
    if verify_checkpoint_integrity and official_model:
        _verify_file(
            checkpoint,
            expected_size=CSM_CHECKPOINT_SIZE,
            expected_sha256=CSM_CHECKPOINT_SHA256,
        )
    if verify_integrity and official_codec and codec_checkpoint is not None:
        _verify_file(
            codec_checkpoint,
            expected_size=MIMI_CHECKPOINT_SIZE,
            expected_sha256=MIMI_CHECKPOINT_SHA256,
        )
    return CSMArtifacts(
        source=source_name,
        revision=resolved_revision,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        config=config,
        codec_checkpoint=codec_checkpoint,
        official_model=official_model,
        official_codec=official_codec,
    )


__all__ = ["CSMArtifacts", "resolve_csm_artifacts"]
