"""Coherent immutable artifacts for native SeamlessM4T-v2 speech-to-text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.asr_seamless_m4t_v2.native.metadata import SEAMLESS_M4T_V2_CHECKPOINTS
from voicehub.path_utils import is_explicit_local_path

_REQUIRED = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
)
_SINGLE = "model.safetensors"
_INDEX = "model.safetensors.index.json"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{7,64}$")


@dataclass(frozen=True, slots=True)
class SeamlessM4Tv2S2TArtifacts:
    """Every file needed to execute one S2T graph."""

    source: str
    revision: str | None
    root: Path
    config: Path
    generation_config: Path
    preprocessor_config: Path
    tokenizer_model: Path
    tokenizer_config: Path
    added_tokens: Path
    special_tokens_map: Path
    checkpoint: Path

    @property
    def is_sharded(self) -> bool:
        return self.checkpoint.name.endswith(".safetensors.index.json")


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native SeamlessM4T-v2 requires {filename!r} in {root}.")
    return path


def _optional(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path if path.is_file() else None


def _validate_checkpoint(path: Path) -> None:
    if (path.suffix != ".safetensors" and not path.name.endswith(".safetensors.index.json")):
        raise ValueError("Native SeamlessM4T-v2 accepts Safetensors only.")


def _shard_names(index: Path) -> tuple[str, ...]:
    document = read_json_file(index)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("SeamlessM4T-v2 Safetensors index requires a weight map.")
    names = set()
    for tensor_name, filename in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("Checkpoint index contains an invalid tensor name.")
        if not isinstance(filename, str) or not filename:
            raise ValueError("Checkpoint index contains an invalid shard name.")
        normalized = PurePosixPath(filename.replace("\\", "/"))
        if (normalized.is_absolute() or len(normalized.parts) != 1 or ".." in normalized.parts or
                not filename.endswith(".safetensors")):
            raise ValueError(f"Unsafe checkpoint shard path {filename!r}.")
        names.add(filename)
    return tuple(sorted(names))


def _construct(
    root: Path,
    *,
    source: str,
    revision: str | None,
    checkpoint: Path | None = None,
) -> SeamlessM4Tv2S2TArtifacts:
    single = _optional(root, _SINGLE)
    index = _optional(root, _INDEX)
    if checkpoint is None:
        if single is not None and index is not None:
            raise ValueError(
                "SeamlessM4T-v2 directory contains both single-file and "
                "sharded checkpoints.")
        checkpoint = single or _required(root, _INDEX)
    _validate_checkpoint(checkpoint)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for filename in _shard_names(checkpoint):
            _required(root, filename)
    required = {name: _required(root, name) for name in _REQUIRED}
    return SeamlessM4Tv2S2TArtifacts(
        source=source,
        revision=revision,
        root=root,
        config=required["config.json"],
        generation_config=required["generation_config.json"],
        preprocessor_config=required["preprocessor_config.json"],
        tokenizer_model=required["tokenizer.model"],
        tokenizer_config=required["tokenizer_config.json"],
        added_tokens=required["added_tokens.json"],
        special_tokens_map=required["special_tokens_map.json"],
        checkpoint=checkpoint,
    )


def _resolve_local(source: Path) -> SeamlessM4Tv2S2TArtifacts:
    if source.is_file():
        _validate_checkpoint(source)
        return _construct(
            source.parent,
            source=str(source),
            revision=None,
            checkpoint=source,
        )
    return _construct(
        source,
        source=str(source),
        revision=None,
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


def _coherent(root: Path, *paths: Path) -> None:
    expected = root.resolve()
    mismatched = [path for path in paths if path.parent.resolve() != expected]
    if mismatched:
        raise RuntimeError(
            "SeamlessM4T-v2 artifacts did not resolve from one immutable "
            "snapshot: " + ", ".join(str(path) for path in mismatched) + ".")


def resolve_seamless_m4t_v2_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> SeamlessM4Tv2S2TArtifacts:
    """Resolve a local portable export or a proven immutable Hub snapshot."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("SeamlessM4T-v2 source must be a non-empty path or Hub ID.")
    local = Path(source).expanduser()
    if local.exists():
        if revision is not None:
            raise ValueError("`revision` cannot be applied to a local artifact.")
        return _resolve_local(local.resolve())
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"SeamlessM4T-v2 model path was not found: {local}.")

    repo_id = str(source)
    known = SEAMLESS_M4T_V2_CHECKPOINTS.get(repo_id)
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
            "VoiceHub could not prove an immutable SeamlessM4T-v2 Hub "
            "revision. Retry online or pass an explicit commit hash.")
    root = config.parent
    required = {"config.json": config}
    for filename in _REQUIRED[1:]:
        required[filename] = resolve_pretrained_file(
            repo_id,
            filename,
            revision=pinned,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    single = _remote_optional(
        repo_id,
        _SINGLE,
        revision=pinned,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    index = _remote_optional(
        repo_id,
        _INDEX,
        revision=pinned,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if single is not None and index is not None:
        raise ValueError("SeamlessM4T-v2 snapshot contains both single-file and sharded "
                         "checkpoints.")
    checkpoint = single or index
    if checkpoint is None:
        raise FileNotFoundError("No SeamlessM4T-v2 Safetensors checkpoint was found.")
    shards = []
    if checkpoint.name.endswith(".safetensors.index.json"):
        for filename in _shard_names(checkpoint):
            shards.append(
                resolve_pretrained_file(
                    repo_id,
                    filename,
                    revision=pinned,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                ))
    _coherent(
        root,
        *required.values(),
        checkpoint,
        *shards,
    )
    if known is not None and pinned == known["revision"] and index is not None:
        expected_shards = known["shards"]
        actual_names = {path.name for path in shards}
        if actual_names != set(expected_shards):
            raise RuntimeError("Published SeamlessM4T-v2 shard set is incomplete.")
        for shard in shards:
            expected_size = expected_shards[shard.name]["size"]
            if shard.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Published shard {shard.name!r} has size "
                    f"{shard.stat().st_size}; expected {expected_size}.")
    return SeamlessM4Tv2S2TArtifacts(
        source=repo_id,
        revision=pinned,
        root=root,
        config=required["config.json"],
        generation_config=required["generation_config.json"],
        preprocessor_config=required["preprocessor_config.json"],
        tokenizer_model=required["tokenizer.model"],
        tokenizer_config=required["tokenizer_config.json"],
        added_tokens=required["added_tokens.json"],
        special_tokens_map=required["special_tokens_map.json"],
        checkpoint=checkpoint,
    )


__all__ = [
    "SeamlessM4Tv2S2TArtifacts",
    "resolve_seamless_m4t_v2_artifacts",
]
