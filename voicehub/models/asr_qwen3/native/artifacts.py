"""Immutable local/Hub artifact resolution for native Qwen3-ASR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.asr_qwen3.native.metadata import QWEN3_ASR_CHECKPOINTS
from voicehub.path_utils import is_explicit_local_path

_REQUIRED_ASSETS = (
    "config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
)
_OPTIONAL_ASSETS = (
    "generation_config.json",
    "preprocessor_config.json",
    "chat_template.json",
)
_SINGLE_CHECKPOINT = "model.safetensors"
_SHARDED_CHECKPOINT = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class Qwen3ASRArtifacts:
    """Coherent immutable paths needed by one Qwen3-ASR runtime."""

    source: str
    revision: str | None
    config: Path
    checkpoint: Path
    vocab: Path
    merges: Path
    tokenizer_config: Path
    generation_config: Path | None = None
    preprocessor_config: Path | None = None
    chat_template: Path | None = None

    @property
    def is_sharded(self) -> bool:
        return self.checkpoint.name.endswith(".safetensors.index.json")


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native Qwen3-ASR requires {filename!r} in {root}.")
    return path


def _optional(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path if path.is_file() else None


def _safe_shards(index_path: Path) -> tuple[str, ...]:
    values = read_json_file(index_path)
    weight_map = values.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Qwen3-ASR Safetensors index must contain a non-empty "
                         "`weight_map`.")
    names: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if (not isinstance(tensor_name, str) or not tensor_name or not isinstance(shard_name, str) or
                not shard_name):
            raise ValueError("Qwen3-ASR Safetensors index contains an invalid record.")
        path = PurePosixPath(shard_name)
        if (path.is_absolute() or len(path.parts) != 1 or ".." in path.parts or
                not shard_name.endswith(".safetensors")):
            raise ValueError(f"Unsafe Qwen3-ASR shard path {shard_name!r}.")
        names.add(shard_name)
    return tuple(sorted(names))


def _resolve_local(source: Path) -> Qwen3ASRArtifacts:
    checkpoint_override = None
    if source.is_file():
        if (source.suffix != ".safetensors" and not source.name.endswith(".safetensors.index.json")):
            raise ValueError("A direct Qwen3-ASR file must be Safetensors or its index.")
        checkpoint_override = source
        root = source.parent
    else:
        root = source
    paths = {name: _required(root, name) for name in _REQUIRED_ASSETS}
    checkpoint = checkpoint_override or _optional(root, _SINGLE_CHECKPOINT)
    checkpoint = checkpoint or _required(root, _SHARDED_CHECKPOINT)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _safe_shards(checkpoint):
            _required(root, shard)
    return Qwen3ASRArtifacts(
        source=str(source),
        revision=None,
        config=paths["config.json"],
        checkpoint=checkpoint,
        vocab=paths["vocab.json"],
        merges=paths["merges.txt"],
        tokenizer_config=paths["tokenizer_config.json"],
        generation_config=_optional(root, "generation_config.json"),
        preprocessor_config=_optional(root, "preprocessor_config.json"),
        chat_template=_optional(root, "chat_template.json"),
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


def resolve_qwen3_asr_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Qwen3ASRArtifacts:
    """Resolve every file from one immutable checkpoint snapshot."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Qwen3-ASR source must be a non-empty path or Hub ID.")
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _resolve_local(source_path.resolve())
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Qwen3-ASR model path was not found: {source_path}.")

    repo_id = str(source)
    known = QWEN3_ASR_CHECKPOINTS.get(repo_id)
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
    resolved_revision = pinned or requested_revision
    paths = {
        name:
        resolve_pretrained_file(
            repo_id,
            name,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        for name in _REQUIRED_ASSETS[1:]
    }
    checkpoint = _resolve_optional_remote(
        repo_id,
        _SINGLE_CHECKPOINT,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if checkpoint is None:
        checkpoint = resolve_pretrained_file(
            repo_id,
            _SHARDED_CHECKPOINT,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
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
    optional = {
        name:
        _resolve_optional_remote(
            repo_id,
            name,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        for name in _OPTIONAL_ASSETS
    }
    return Qwen3ASRArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=config,
        checkpoint=checkpoint,
        vocab=paths["vocab.json"],
        merges=paths["merges.txt"],
        tokenizer_config=paths["tokenizer_config.json"],
        generation_config=optional["generation_config.json"],
        preprocessor_config=optional["preprocessor_config.json"],
        chat_template=optional["chat_template.json"],
    )


__all__ = [
    "Qwen3ASRArtifacts",
    "resolve_qwen3_asr_artifacts",
]
