"""Coherent local and Hub artifact resolution for native Dia."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.path_utils import is_explicit_local_path

_CONFIG = "config.json"
_GENERATION_CONFIG = "generation_config.json"
_PREPROCESSOR_CONFIG = "preprocessor_config.json"
_TOKENIZER_CONFIG = "tokenizer_config.json"
_AUDIO_TOKENIZER_CONFIG = "audio_tokenizer_config.json"
_SINGLE_CHECKPOINT = "model.safetensors"
_SHARDED_CHECKPOINT = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class DiaArtifacts:
    source: str
    revision: str | None
    config: Path
    checkpoint: Path
    generation_config: Path | None = None
    preprocessor_config: Path | None = None
    tokenizer_config: Path | None = None
    audio_tokenizer_config: Path | None = None

    @property
    def is_sharded(self) -> bool:
        return self.checkpoint.name.endswith(".safetensors.index.json")


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native Dia requires {filename!r} in {root}.")
    return path


def _optional(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path if path.is_file() else None


def _validate_checkpoint(path: Path) -> None:
    if (path.suffix != ".safetensors" and not path.name.endswith(".safetensors.index.json")):
        raise ValueError(
            "Native Dia loads Safetensors only. The original Dia-1.6B "
            "`dia-v1.pth`/`pytorch_model.bin` artifacts execute a different "
            "legacy layout; migrate to `nari-labs/Dia-1.6B-0626`.")


def _safe_shards(index: Path) -> tuple[str, ...]:
    document = read_json_file(index)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Dia Safetensors index requires a non-empty `weight_map`.")
    shards = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("Dia index contains an invalid tensor name.")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError("Dia index contains an invalid shard name.")
        path = PurePosixPath(shard_name)
        if ("\\" in shard_name or path.is_absolute() or len(path.parts) != 1 or ".." in path.parts or
                not shard_name.endswith(".safetensors")):
            raise ValueError(f"Unsafe Dia shard path {shard_name!r}.")
        shards.add(shard_name)
    return tuple(sorted(shards))


def _optional_remote(
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


def _resolve_local(source: Path) -> DiaArtifacts:
    checkpoint_override = None
    if source.is_file():
        _validate_checkpoint(source)
        checkpoint_override = source
        root = source.parent
    else:
        root = source
    config = _required(root, _CONFIG)
    document = read_json_file(config)
    if "model" in document and "data" in document:
        raise ValueError(
            "This is the original Dia-1.6B checkpoint layout. Native "
            "fine-tuning and inference require the converted Safetensors "
            "artifact `nari-labs/Dia-1.6B-0626`.")
    checkpoint = checkpoint_override or _optional(root, _SINGLE_CHECKPOINT)
    checkpoint = checkpoint or _required(root, _SHARDED_CHECKPOINT)
    _validate_checkpoint(checkpoint)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _safe_shards(checkpoint):
            _required(root, shard)
    return DiaArtifacts(
        source=str(source),
        revision=None,
        config=config,
        checkpoint=checkpoint,
        generation_config=_optional(root, _GENERATION_CONFIG),
        preprocessor_config=_optional(root, _PREPROCESSOR_CONFIG),
        tokenizer_config=_optional(root, _TOKENIZER_CONFIG),
        audio_tokenizer_config=_optional(root, _AUDIO_TOKENIZER_CONFIG),
    )


def resolve_dia_artifacts(
    source: str | Path,
    *,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> DiaArtifacts:
    """Resolve one safe checkpoint set at a single immutable revision."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Dia `source` must be a non-empty path or Hub ID.")
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _resolve_local(source_path.resolve())
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Dia model path was not found: {source_path}.")
    if str(source).rstrip("/").lower() == "nari-labs/dia-1.6b":
        raise ValueError(
            "The original `nari-labs/Dia-1.6B` pickle layout is not loaded "
            "by the native runtime. Use `nari-labs/Dia-1.6B-0626`, whose "
            "strict Safetensors layout supports inference and fine-tuning.")

    repo_id = str(source)
    requested_revision = revision or "main"
    config = resolve_pretrained_file(
        repo_id,
        _CONFIG,
        revision=requested_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    pinned = get_cached_hugging_face_commit(
        repo_id,
        _CONFIG,
        revision=requested_revision,
        cache_dir=cache_dir,
    )
    resolved_revision = pinned or requested_revision
    checkpoint = _optional_remote(
        repo_id,
        _SINGLE_CHECKPOINT,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    checkpoint = checkpoint or _optional_remote(
        repo_id,
        _SHARDED_CHECKPOINT,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if checkpoint is None:
        raise FileNotFoundError(
            "Native Dia found no `model.safetensors` or "
            "`model.safetensors.index.json`.")
    _validate_checkpoint(checkpoint)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _safe_shards(checkpoint):
            resolve_pretrained_file(
                repo_id,
                shard,
                revision=resolved_revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )

    def optional(filename: str) -> Path | None:
        return _optional_remote(
            repo_id,
            filename,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )

    return DiaArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=config,
        checkpoint=checkpoint,
        generation_config=optional(_GENERATION_CONFIG),
        preprocessor_config=optional(_PREPROCESSOR_CONFIG),
        tokenizer_config=optional(_TOKENIZER_CONFIG),
        audio_tokenizer_config=optional(_AUDIO_TOKENIZER_CONFIG),
    )


__all__ = ["DiaArtifacts", "resolve_dia_artifacts"]
