"""Coherent local and Hugging Face artifact resolution for Parakeet TDT."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.asr_parakeet_tdt.native.metadata import PARAKEET_TDT_CHECKPOINTS
from voicehub.path_utils import is_explicit_local_path

_REQUIRED_FILES = (
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_OPTIONAL_FILES = ("generation_config.json", )
_SINGLE_CHECKPOINT = "model.safetensors"
_SHARDED_CHECKPOINT = "model.safetensors.index.json"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{7,64}$")


@dataclass(frozen=True, slots=True)
class ParakeetTDTArtifacts:
    """Immutable files from one coherent model revision."""

    source: str
    revision: str | None
    config: Path
    processor_config: Path
    tokenizer: Path
    tokenizer_config: Path
    checkpoint: Path
    generation_config: Path | None = None

    @property
    def is_sharded(self) -> bool:
        return self.checkpoint.name.endswith(".safetensors.index.json")


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native Parakeet TDT requires {filename!r} in {root}.")
    return path


def _optional(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path if path.is_file() else None


def _validate_checkpoint(path: Path) -> None:
    if (path.suffix != ".safetensors" and not path.name.endswith(".safetensors.index.json")):
        raise ValueError("Native Parakeet TDT accepts Safetensors only.")


def _shards(index_path: Path) -> tuple[str, ...]:
    values = read_json_file(index_path)
    weight_map = values.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Parakeet TDT Safetensors index requires a non-empty `weight_map`.")
    names = set()
    for tensor_name, filename in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("Invalid tensor name in Parakeet TDT index.")
        if not isinstance(filename, str) or not filename:
            raise ValueError("Invalid shard name in Parakeet TDT index.")
        normalized = PurePosixPath(filename.replace("\\", "/"))
        if (normalized.is_absolute() or len(normalized.parts) != 1 or ".." in normalized.parts or
                not filename.endswith(".safetensors")):
            raise ValueError(f"Unsafe Parakeet TDT shard path {filename!r}.")
        names.add(filename)
    return tuple(sorted(names))


def _resolve_local(source: Path) -> ParakeetTDTArtifacts:
    if source.is_file():
        _validate_checkpoint(source)
        root = source.parent
        checkpoint = source
    else:
        root = source
        single = _optional(root, _SINGLE_CHECKPOINT)
        index = _optional(root, _SHARDED_CHECKPOINT)
        if single is not None and index is not None:
            raise ValueError(
                "Parakeet TDT directory contains both single-file and "
                "sharded model artifacts.")
        checkpoint = single or _required(root, _SHARDED_CHECKPOINT)
    _validate_checkpoint(checkpoint)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _shards(checkpoint):
            _required(root, shard)
    return ParakeetTDTArtifacts(
        source=str(source),
        revision=None,
        config=_required(root, "config.json"),
        processor_config=_required(root, "processor_config.json"),
        tokenizer=_required(root, "tokenizer.json"),
        tokenizer_config=_required(root, "tokenizer_config.json"),
        checkpoint=checkpoint,
        generation_config=_optional(root, "generation_config.json"),
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
    *artifacts: Path | None,
) -> None:
    expected = root.resolve()
    mismatched = [path for path in artifacts if path is not None and path.parent.resolve() != expected]
    if mismatched:
        raise RuntimeError(
            "Parakeet TDT artifacts did not resolve from one immutable "
            "snapshot: " + ", ".join(str(path) for path in mismatched) + ".")


def resolve_parakeet_tdt_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> ParakeetTDTArtifacts:
    """Resolve all runtime files from one local root or immutable Hub
    commit."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Parakeet TDT source must be a non-empty path or Hub ID.")
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _resolve_local(source_path.resolve())
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Parakeet TDT model path was not found: {source_path}.")

    repo_id = str(source)
    known = PARAKEET_TDT_CHECKPOINTS.get(repo_id)
    requested_revision = (revision or (str(known["revision"]) if known is not None else "main"))
    config = resolve_pretrained_file(
        repo_id,
        "config.json",
        revision=requested_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    pinned = get_cached_hugging_face_commit(
        repo_id,
        "config.json",
        revision=requested_revision,
        cache_dir=cache_dir,
    )
    resolved_revision = pinned or (
        requested_revision.lower() if _IMMUTABLE_REVISION.fullmatch(requested_revision) else None)
    if (resolved_revision is None or not _IMMUTABLE_REVISION.fullmatch(resolved_revision)):
        raise RuntimeError(
            "VoiceHub could not prove an immutable Parakeet TDT Hub "
            "revision after resolving `config.json`. Retry online or pass "
            "an explicit commit.")
    root = config.parent
    resolved = {"config.json": config}
    for filename in _REQUIRED_FILES[1:]:
        resolved[filename] = resolve_pretrained_file(
            repo_id,
            filename,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    single_checkpoint = _resolve_optional_remote(
        repo_id,
        _SINGLE_CHECKPOINT,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    index_checkpoint = _resolve_optional_remote(
        repo_id,
        _SHARDED_CHECKPOINT,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if single_checkpoint is not None and index_checkpoint is not None:
        raise ValueError(
            "Parakeet TDT Hub snapshot contains both single-file and sharded "
            "model artifacts.")
    checkpoint = single_checkpoint
    if checkpoint is None:
        checkpoint = index_checkpoint
    if checkpoint is None:
        raise FileNotFoundError(
            "Native Parakeet TDT could not find model.safetensors or a "
            "Safetensors index.")
    shard_paths = []
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _shards(checkpoint):
            shard_paths.append(
                resolve_pretrained_file(
                    repo_id,
                    shard,
                    revision=resolved_revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                ))
    generation = _resolve_optional_remote(
        repo_id,
        _OPTIONAL_FILES[0],
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    _require_coherent_snapshot(
        root,
        *resolved.values(),
        checkpoint,
        *shard_paths,
        generation,
    )
    return ParakeetTDTArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=resolved["config.json"],
        processor_config=resolved["processor_config.json"],
        tokenizer=resolved["tokenizer.json"],
        tokenizer_config=resolved["tokenizer_config.json"],
        checkpoint=checkpoint,
        generation_config=generation,
    )


__all__ = [
    "ParakeetTDTArtifacts",
    "resolve_parakeet_tdt_artifacts",
]
