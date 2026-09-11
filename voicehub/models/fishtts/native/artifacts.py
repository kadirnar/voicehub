"""Coherent immutable artifact resolution for native Fish S2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.fishtts.native.metadata import (
    FISH_S2_CHECKPOINT,
    FISH_S2_CHECKPOINT_REVISION,
    FISH_S2_LEGACY_CODEC_FILENAME,
)
from voicehub.path_utils import is_explicit_local_path

_CONFIG = "config.json"
_TOKENIZER = "tokenizer.json"
_TOKENIZER_CONFIG = "tokenizer_config.json"
_SINGLE_CHECKPOINT = "model.safetensors"
_SHARDED_CHECKPOINT = "model.safetensors.index.json"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{7,64}$")


@dataclass(frozen=True, slots=True)
class FishSemanticArtifacts:
    source: str
    revision: str | None
    root: Path
    config: Path
    checkpoint: Path
    tokenizer: Path
    tokenizer_config: Path | None

    @property
    def is_sharded(self) -> bool:
        return self.checkpoint.name.endswith(".safetensors.index.json")


@dataclass(frozen=True, slots=True)
class FishCodecArtifacts:
    source: str
    revision: str | None
    config: Path
    checkpoint: Path


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native Fish S2 requires {filename!r} in {root}.")
    return path


def _optional(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path if path.is_file() else None


def _safe_shards(index: Path) -> tuple[str, ...]:
    document = read_json_file(index)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Fish Safetensors index requires a non-empty `weight_map`.")
    shards: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("Fish index contains an invalid tensor name.")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError("Fish index contains an invalid shard name.")
        path = PurePosixPath(shard_name)
        if ("\\" in shard_name or path.is_absolute() or len(path.parts) != 1 or ".." in path.parts or
                not shard_name.endswith(".safetensors")):
            raise ValueError(f"Unsafe Fish shard path {shard_name!r}.")
        shards.add(shard_name)
    return tuple(sorted(shards))


def _local_semantic(source: Path) -> FishSemanticArtifacts:
    checkpoint_override = None
    if source.is_file():
        if (source.suffix != ".safetensors" and not source.name.endswith(".safetensors.index.json")):
            raise ValueError(
                "Native Fish S2 loads Safetensors only. GGUF, pickle, "
                "and PyTorch binary semantic checkpoints are rejected.")
        checkpoint_override = source
        root = source.parent
    else:
        root = source
    config = _required(root, _CONFIG)
    tokenizer = _required(root, _TOKENIZER)
    single = _optional(root, _SINGLE_CHECKPOINT)
    index = _optional(root, _SHARDED_CHECKPOINT)
    if checkpoint_override is None and single is not None and index is not None:
        raise ValueError("Fish artifact directory contains both single and sharded "
                         "semantic checkpoints.")
    checkpoint = checkpoint_override or single or index
    if checkpoint is None:
        raise FileNotFoundError(f"Native Fish S2 found no Safetensors checkpoint in {root}.")
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _safe_shards(checkpoint):
            _required(root, shard)
    return FishSemanticArtifacts(
        source=str(source),
        revision=None,
        root=root.resolve(),
        config=config.resolve(),
        checkpoint=checkpoint.resolve(),
        tokenizer=tokenizer.resolve(),
        tokenizer_config=(
            _optional(root, _TOKENIZER_CONFIG).resolve()
            if _optional(root, _TOKENIZER_CONFIG) is not None else None),
    )


def _remote_optional(
    source: str,
    filename: str,
    *,
    revision: str,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> Path | None:
    try:
        return resolve_pretrained_file(
            source,
            filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    except FileNotFoundError:
        return None


def _resolved_revision(
    source: str,
    filename: str,
    *,
    requested: str,
    cache_dir: str | None,
) -> str:
    pinned = get_cached_hugging_face_commit(
        source,
        filename,
        revision=requested,
        cache_dir=cache_dir,
    )
    if pinned is not None:
        return pinned
    if _IMMUTABLE_REVISION.fullmatch(requested):
        return requested.lower()
    raise RuntimeError(
        "VoiceHub could not prove an immutable Fish Hub revision after "
        f"resolving {filename!r}. Retry online or pass an explicit commit.")


def _require_coherent_snapshot(
    root: Path,
    *artifacts: Path | None,
) -> None:
    mismatched = [path for path in artifacts if path is not None and path.parent.resolve() != root.resolve()]
    if mismatched:
        raise RuntimeError(
            "Fish artifacts did not resolve from one immutable snapshot: " +
            ", ".join(str(path) for path in mismatched) + ".")


def resolve_fish_semantic_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> FishSemanticArtifacts:
    """Resolve config, tokenizer, index, and every shard at one commit."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Fish semantic source must be a non-empty path or ID.")
    local = Path(source).expanduser()
    if local.exists():
        return _local_semantic(local.resolve())
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Fish model path was not found: {local}.")
    repo_id = str(source)
    requested = (
        FISH_S2_CHECKPOINT_REVISION
        if repo_id == FISH_S2_CHECKPOINT and revision is None else revision or "main")
    config = resolve_pretrained_file(
        repo_id,
        _CONFIG,
        revision=requested,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    resolved_revision = _resolved_revision(
        repo_id,
        _CONFIG,
        requested=requested,
        cache_dir=cache_dir,
    )
    tokenizer = resolve_pretrained_file(
        repo_id,
        _TOKENIZER,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    checkpoint = _remote_optional(
        repo_id,
        _SINGLE_CHECKPOINT,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    checkpoint = checkpoint or _remote_optional(
        repo_id,
        _SHARDED_CHECKPOINT,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if checkpoint is None:
        raise FileNotFoundError("Native Fish S2 found no Safetensors semantic checkpoint.")
    shard_paths: list[Path] = []
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _safe_shards(checkpoint):
            shard_paths.append(
                resolve_pretrained_file(
                    repo_id,
                    shard,
                    revision=resolved_revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                ))
    tokenizer_config = _remote_optional(
        repo_id,
        _TOKENIZER_CONFIG,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    _require_coherent_snapshot(
        config.parent,
        tokenizer,
        checkpoint,
        tokenizer_config,
        *shard_paths,
    )
    return FishSemanticArtifacts(
        source=repo_id,
        revision=resolved_revision,
        root=config.parent.resolve(),
        config=config,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
    )


def _local_codec(source: Path) -> FishCodecArtifacts:
    if source.is_file():
        if source.suffix != ".safetensors":
            raise ValueError(
                "Native Fish ModifiedDAC loads Safetensors only. Convert a "
                "trusted legacy `codec.pth` explicitly.")
        root = source.parent
        checkpoint = source
    else:
        root = source
        checkpoint = _required(root, _SINGLE_CHECKPOINT)
    return FishCodecArtifacts(
        source=str(source),
        revision=None,
        config=_required(root, _CONFIG).resolve(),
        checkpoint=checkpoint.resolve(),
    )


def resolve_fish_codec_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> FishCodecArtifacts:
    """Resolve a previously converted, steady-state safe codec."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Fish codec source must be a non-empty path or ID.")
    local = Path(source).expanduser()
    if local.exists():
        return _local_codec(local.resolve())
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Fish codec path was not found: {local}.")
    repo_id = str(source)
    requested = revision or "main"
    config = resolve_pretrained_file(
        repo_id,
        _CONFIG,
        revision=requested,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    resolved_revision = _resolved_revision(
        repo_id,
        _CONFIG,
        requested=requested,
        cache_dir=cache_dir,
    )
    checkpoint = resolve_pretrained_file(
        repo_id,
        _SINGLE_CHECKPOINT,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    _require_coherent_snapshot(config.parent, checkpoint)
    return FishCodecArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=config,
        checkpoint=checkpoint,
    )


def resolve_official_fish_legacy_codec(
    *,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve only the immutable official pickle for explicit conversion."""
    return resolve_pretrained_file(
        FISH_S2_CHECKPOINT,
        FISH_S2_LEGACY_CODEC_FILENAME,
        revision=FISH_S2_CHECKPOINT_REVISION,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )


__all__ = [
    "FishCodecArtifacts",
    "FishSemanticArtifacts",
    "resolve_fish_codec_artifacts",
    "resolve_fish_semantic_artifacts",
    "resolve_official_fish_legacy_codec",
]
