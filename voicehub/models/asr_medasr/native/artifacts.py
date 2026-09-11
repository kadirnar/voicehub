"""Coherent local or immutable Hub artifact resolution for MedASR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.asr_medasr.native.metadata import MEDASR_MODEL_ID, MEDASR_MODEL_REVISION
from voicehub.path_utils import is_explicit_local_path

_REQUIRED = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_OPTIONAL = (
    "added_tokens.json",
    "processor_config.json",
    "spiece.model",
)
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{7,64}$")


@dataclass(frozen=True, slots=True)
class MedASRArtifacts:
    source: str
    revision: str | None
    config: Path
    checkpoint: Path
    preprocessor_config: Path
    tokenizer: Path
    tokenizer_config: Path
    added_tokens: Path | None = None
    processor_config: Path | None = None
    sentencepiece_model: Path | None = None


def _plain_filename(value: str | None) -> str:
    filename = "model.safetensors" if value is None else value
    if (not isinstance(filename, str) or not filename or Path(filename).name != filename or
            not filename.endswith(".safetensors")):
        raise ValueError("MedASR `checkpoint_filename` must be a plain Safetensors "
                         "filename.")
    return filename


def _required(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Native MedASR requires {name!r} in {root}.")
    return path


def _optional(root: Path, name: str) -> Path | None:
    path = root / name
    return path if path.is_file() else None


def _local(
    source: Path,
    *,
    checkpoint_filename: str,
) -> MedASRArtifacts:
    if source.is_file():
        if source.suffix != ".safetensors":
            raise ValueError("A direct MedASR checkpoint must use Safetensors.")
        root = source.parent
        checkpoint = source
    else:
        root = source
        checkpoint = _required(root, checkpoint_filename)
    required = {name: _required(root, name) for name in _REQUIRED}
    return MedASRArtifacts(
        source=str(source),
        revision=None,
        config=required["config.json"],
        checkpoint=checkpoint,
        preprocessor_config=required["preprocessor_config.json"],
        tokenizer=required["tokenizer.json"],
        tokenizer_config=required["tokenizer_config.json"],
        added_tokens=_optional(root, "added_tokens.json"),
        processor_config=_optional(root, "processor_config.json"),
        sentencepiece_model=_optional(root, "spiece.model"),
    )


def _remote_optional(
    repo_id: str,
    filename: str,
    *,
    revision: str,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> Path | None:
    try:
        return resolve_pretrained_file(
            repo_id,
            filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    except FileNotFoundError:
        return None


def _require_coherent_snapshot(
    root: Path,
    *artifacts: Path | None,
) -> None:
    mismatched = tuple(
        artifact for artifact in artifacts
        if artifact is not None and artifact.parent.resolve() != root.resolve())
    if mismatched:
        raise RuntimeError(
            "MedASR artifacts did not resolve from one immutable snapshot: " +
            ", ".join(str(path) for path in mismatched) + ".")


def resolve_medasr_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> MedASRArtifacts:
    """Resolve every runtime asset from one root and one commit."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("MedASR source must be a non-empty path or Hub ID.")
    filename = _plain_filename(checkpoint_filename)
    local = Path(source).expanduser()
    if local.exists():
        return _local(
            local.resolve(),
            checkpoint_filename=filename,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"MedASR model path was not found: {local}.")

    repo_id = str(source)
    requested_revision = (revision or (MEDASR_MODEL_REVISION if repo_id == MEDASR_MODEL_ID else "main"))
    config = resolve_pretrained_file(
        repo_id,
        "config.json",
        revision=requested_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    cached_revision = get_cached_hugging_face_commit(
        repo_id,
        "config.json",
        revision=requested_revision,
        cache_dir=cache_dir,
    )
    resolved_revision = cached_revision or (
        requested_revision.lower() if _IMMUTABLE_REVISION.fullmatch(requested_revision) else None)
    if (resolved_revision is None or not _IMMUTABLE_REVISION.fullmatch(resolved_revision)):
        raise RuntimeError(
            "VoiceHub could not prove an immutable MedASR Hub revision after "
            "resolving `config.json`. Retry online or pass an explicit commit.")
    required = {
        name: (
            config if name == "config.json" else resolve_pretrained_file(
                repo_id,
                name,
                revision=resolved_revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            ))
        for name in _REQUIRED
    }
    checkpoint = resolve_pretrained_file(
        repo_id,
        filename,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    optional = {
        name:
        _remote_optional(
            repo_id,
            name,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        for name in _OPTIONAL
    }
    _require_coherent_snapshot(
        config.parent,
        *required.values(),
        checkpoint,
        *optional.values(),
    )
    return MedASRArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=required["config.json"],
        checkpoint=checkpoint,
        preprocessor_config=required["preprocessor_config.json"],
        tokenizer=required["tokenizer.json"],
        tokenizer_config=required["tokenizer_config.json"],
        added_tokens=optional["added_tokens.json"],
        processor_config=optional["processor_config.json"],
        sentencepiece_model=optional["spiece.model"],
    )


__all__ = [
    "MedASRArtifacts",
    "resolve_medasr_artifacts",
]
