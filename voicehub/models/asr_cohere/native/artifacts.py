"""Coherent artifact resolution for native Cohere Transcribe."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.asr_cohere.native.metadata import COHERE_ASR_CHECKPOINTS
from voicehub.path_utils import is_explicit_local_path

_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_SINGLE_CHECKPOINT = "model.safetensors"
_SHARDED_CHECKPOINT = "model.safetensors.index.json"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{7,64}$")


@dataclass(frozen=True, slots=True)
class CohereAsrArtifacts:
    """Files resolved from one local root or immutable Hub snapshot."""

    source: str
    revision: str | None
    config: Path
    generation_config: Path
    preprocessor_config: Path
    processor_config: Path
    tokenizer: Path
    tokenizer_config: Path
    checkpoint: Path

    @property
    def is_sharded(self) -> bool:
        return self.checkpoint.name.endswith(".safetensors.index.json")


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native Cohere ASR requires {filename!r} in {root}.")
    return path


def _optional(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path if path.is_file() else None


def _validate_checkpoint(path: Path) -> None:
    if (path.suffix != ".safetensors" and not path.name.endswith(".safetensors.index.json")):
        raise ValueError("Native Cohere ASR accepts Safetensors only.")


def _shards(index_path: Path) -> tuple[str, ...]:
    document = read_json_file(index_path)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Cohere ASR Safetensors index requires a non-empty weight map.")
    names = set()
    for tensor_name, filename in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("Cohere ASR index contains an invalid tensor name.")
        if not isinstance(filename, str) or not filename:
            raise ValueError("Cohere ASR index contains an invalid shard name.")
        normalized = PurePosixPath(filename.replace("\\", "/"))
        if (normalized.is_absolute() or len(normalized.parts) != 1 or ".." in normalized.parts or
                not filename.endswith(".safetensors")):
            raise ValueError(f"Unsafe Cohere ASR shard path {filename!r}.")
        names.add(filename)
    return tuple(sorted(names))


def _from_root(
    root: Path,
    *,
    source: str,
    revision: str | None,
    checkpoint: Path | None = None,
) -> CohereAsrArtifacts:
    single = _optional(root, _SINGLE_CHECKPOINT)
    index = _optional(root, _SHARDED_CHECKPOINT)
    if checkpoint is None:
        if single is not None and index is not None:
            raise ValueError("Cohere ASR directory contains both single-file and sharded "
                             "checkpoints.")
        checkpoint = single or _required(root, _SHARDED_CHECKPOINT)
    _validate_checkpoint(checkpoint)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for filename in _shards(checkpoint):
            _required(root, filename)
    resolved = {name: _required(root, name) for name in _REQUIRED_FILES}
    return CohereAsrArtifacts(
        source=source,
        revision=revision,
        config=resolved["config.json"],
        generation_config=resolved["generation_config.json"],
        preprocessor_config=resolved["preprocessor_config.json"],
        processor_config=resolved["processor_config.json"],
        tokenizer=resolved["tokenizer.json"],
        tokenizer_config=resolved["tokenizer_config.json"],
        checkpoint=checkpoint,
    )


def _resolve_local(source: Path) -> CohereAsrArtifacts:
    if source.is_file():
        _validate_checkpoint(source)
        return _from_root(
            source.parent,
            source=str(source),
            revision=None,
            checkpoint=source,
        )
    return _from_root(
        source,
        source=str(source),
        revision=None,
    )


def _resolve_optional_remote(
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
    *paths: Path,
) -> None:
    expected = root.resolve()
    mismatched = [path for path in paths if path.parent.resolve() != expected]
    if mismatched:
        raise RuntimeError(
            "Cohere ASR artifacts did not resolve from one immutable "
            "snapshot: " + ", ".join(str(path) for path in mismatched) + ".")


def resolve_cohere_asr_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> CohereAsrArtifacts:
    """Resolve a strict local artifact or one proven Hub commit."""
    if (not isinstance(source, (str, Path)) or not str(source).strip()):
        raise ValueError("Cohere ASR source must be a non-empty path or Hub ID.")
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _resolve_local(source_path.resolve())
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Cohere ASR model path was not found: {source_path}.")

    repo_id = str(source)
    known = COHERE_ASR_CHECKPOINTS.get(repo_id)
    requested = revision or (str(known["revision"]) if known is not None else "main")
    config = resolve_pretrained_file(
        repo_id,
        "config.json",
        revision=requested,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    cached_commit = get_cached_hugging_face_commit(
        repo_id,
        "config.json",
        revision=requested,
        cache_dir=cache_dir,
    )
    pinned = cached_commit or (requested.lower() if _IMMUTABLE_REVISION.fullmatch(requested) else None)
    if pinned is None or not _IMMUTABLE_REVISION.fullmatch(pinned):
        raise RuntimeError(
            "VoiceHub could not prove an immutable Cohere ASR revision. "
            "Retry online or pass an explicit commit hash.")
    root = config.parent
    resolved = {"config.json": config}
    for filename in _REQUIRED_FILES[1:]:
        resolved[filename] = resolve_pretrained_file(
            repo_id,
            filename,
            revision=pinned,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    single = _resolve_optional_remote(
        repo_id,
        _SINGLE_CHECKPOINT,
        revision=pinned,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    index = _resolve_optional_remote(
        repo_id,
        _SHARDED_CHECKPOINT,
        revision=pinned,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if single is not None and index is not None:
        raise ValueError("Cohere ASR Hub snapshot contains both single-file and sharded "
                         "checkpoints.")
    checkpoint = single or index
    if checkpoint is None:
        raise FileNotFoundError(
            "Native Cohere ASR could not find model.safetensors or its "
            "Safetensors index.")
    shards = []
    if checkpoint.name.endswith(".safetensors.index.json"):
        for filename in _shards(checkpoint):
            shards.append(
                resolve_pretrained_file(
                    repo_id,
                    filename,
                    revision=pinned,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                ))
    _require_coherent_snapshot(
        root,
        *resolved.values(),
        checkpoint,
        *shards,
    )
    return CohereAsrArtifacts(
        source=repo_id,
        revision=pinned,
        config=resolved["config.json"],
        generation_config=resolved["generation_config.json"],
        preprocessor_config=resolved["preprocessor_config.json"],
        processor_config=resolved["processor_config.json"],
        tokenizer=resolved["tokenizer.json"],
        tokenizer_config=resolved["tokenizer_config.json"],
        checkpoint=checkpoint,
    )


__all__ = [
    "CohereAsrArtifacts",
    "resolve_cohere_asr_artifacts",
]
