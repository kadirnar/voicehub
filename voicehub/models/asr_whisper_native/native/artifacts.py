"""Artifact resolution for the native Whisper runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.path_utils import is_explicit_local_path

_CONFIG_NAME = "config.json"
_TOKENIZER_NAME = "tokenizer.json"
_GENERATION_CONFIG_NAME = "generation_config.json"
_PREPROCESSOR_CONFIG_NAME = "preprocessor_config.json"
_SINGLE_CHECKPOINT_NAME = "model.safetensors"
_SHARDED_CHECKPOINT_NAME = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class WhisperArtifacts:
    """Immutable paths required by one native Whisper runtime."""

    source: str
    revision: str | None
    config: Path
    tokenizer: Path
    checkpoint: Path
    generation_config: Path | None = None
    preprocessor_config: Path | None = None

    @property
    def is_sharded(self) -> bool:
        """Whether ``checkpoint`` is a Safetensors index."""
        return self.checkpoint.name.endswith(".safetensors.index.json")


def _optional_local(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path if path.is_file() else None


def _required_local(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native Whisper requires {filename!r} in {root}.")
    return path


def _resolve_optional_remote(
    repo_id: str,
    filename: str,
    *,
    cache_dir: str | None,
    revision: str,
    token: str | bool | None,
    local_files_only: bool,
) -> Path | None:
    try:
        return resolve_pretrained_file(
            repo_id,
            filename,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
    except FileNotFoundError:
        return None


def _safe_shard_names(index_path: Path) -> tuple[str, ...]:
    document = read_json_file(index_path)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Whisper Safetensors index must contain a non-empty `weight_map`.")
    names = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("Whisper Safetensors index contains an invalid tensor name.")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError("Whisper Safetensors index contains an invalid shard name.")
        path = PurePosixPath(shard_name)
        if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
            raise ValueError(f"Unsafe Whisper checkpoint shard path {shard_name!r}.")
        if not shard_name.endswith(".safetensors"):
            raise ValueError(f"Whisper checkpoint shard is not Safetensors: {shard_name!r}.")
        names.add(shard_name)
    return tuple(sorted(names))


def _resolve_local_artifacts(
    source: Path,
    *,
    checkpoint_filename: str | None,
    tokenizer_filename: str,
) -> WhisperArtifacts:
    checkpoint_override: Path | None = None
    if source.is_file():
        if source.name in {_CONFIG_NAME, _TOKENIZER_NAME}:
            raise ValueError("A direct native Whisper file must be a Safetensors "
                             "checkpoint or index.")
        checkpoint_override = source
        root = source.parent
    else:
        root = source

    config = _required_local(root, _CONFIG_NAME)
    tokenizer = _required_local(root, tokenizer_filename)
    checkpoint = checkpoint_override
    if checkpoint is None and checkpoint_filename is not None:
        checkpoint = _required_local(root, checkpoint_filename)
    if checkpoint is None:
        checkpoint = _optional_local(root, _SINGLE_CHECKPOINT_NAME)
    if checkpoint is None:
        checkpoint = _required_local(root, _SHARDED_CHECKPOINT_NAME)
    if (checkpoint.suffix != ".safetensors" and not checkpoint.name.endswith(".safetensors.index.json")):
        raise ValueError("Native Whisper accepts Safetensors checkpoints or a "
                         "Safetensors index.")
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _safe_shard_names(checkpoint):
            _required_local(root, shard)
    return WhisperArtifacts(
        source=str(source),
        revision=None,
        config=config,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        generation_config=_optional_local(root, _GENERATION_CONFIG_NAME),
        preprocessor_config=_optional_local(root, _PREPROCESSOR_CONFIG_NAME),
    )


def resolve_whisper_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str | None = None,
    tokenizer_filename: str = _TOKENIZER_NAME,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> WhisperArtifacts:
    """Resolve a coherent local or immutable Hub Whisper artifact set."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Whisper `source` must be a non-empty path or Hub ID.")
    if checkpoint_filename is not None and (not isinstance(checkpoint_filename, str) or
                                            not checkpoint_filename.strip()):
        raise ValueError("`checkpoint_filename` must be a non-empty string or None.")
    if not isinstance(tokenizer_filename, str) or not tokenizer_filename.strip():
        raise ValueError("`tokenizer_filename` must be a non-empty string.")
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _resolve_local_artifacts(
            source_path.resolve(),
            checkpoint_filename=checkpoint_filename,
            tokenizer_filename=tokenizer_filename,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Whisper model path was not found: {source_path}.")

    repo_id = str(source)
    requested_revision = revision or "main"
    config = resolve_pretrained_file(
        repo_id,
        _CONFIG_NAME,
        cache_dir=cache_dir,
        revision=requested_revision,
        token=token,
        local_files_only=local_files_only,
    )
    pinned_revision = get_cached_hugging_face_commit(
        repo_id,
        _CONFIG_NAME,
        cache_dir=cache_dir,
        revision=requested_revision,
    )
    resolved_revision = pinned_revision or requested_revision
    tokenizer = resolve_pretrained_file(
        repo_id,
        tokenizer_filename,
        cache_dir=cache_dir,
        revision=resolved_revision,
        token=token,
        local_files_only=local_files_only,
    )

    candidates = ((checkpoint_filename, ) if checkpoint_filename is not None else
                  (_SINGLE_CHECKPOINT_NAME, _SHARDED_CHECKPOINT_NAME))
    checkpoint = None
    for candidate in candidates:
        checkpoint = _resolve_optional_remote(
            repo_id,
            candidate,
            cache_dir=cache_dir,
            revision=resolved_revision,
            token=token,
            local_files_only=local_files_only,
        )
        if checkpoint is not None:
            break
    if checkpoint is None:
        names = ", ".join(repr(name) for name in candidates)
        raise FileNotFoundError(f"Native Whisper could not find a checkpoint among: {names}.")
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _safe_shard_names(checkpoint):
            resolve_pretrained_file(
                repo_id,
                shard,
                cache_dir=cache_dir,
                revision=resolved_revision,
                token=token,
                local_files_only=local_files_only,
            )

    return WhisperArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=config,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        generation_config=_resolve_optional_remote(
            repo_id,
            _GENERATION_CONFIG_NAME,
            cache_dir=cache_dir,
            revision=resolved_revision,
            token=token,
            local_files_only=local_files_only,
        ),
        preprocessor_config=_resolve_optional_remote(
            repo_id,
            _PREPROCESSOR_CONFIG_NAME,
            cache_dir=cache_dir,
            revision=resolved_revision,
            token=token,
            local_files_only=local_files_only,
        ),
    )


__all__ = ["WhisperArtifacts", "resolve_whisper_artifacts"]
