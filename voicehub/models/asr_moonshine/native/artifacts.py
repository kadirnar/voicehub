"""Coherent local and Hub artifact resolution for native Moonshine ASR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.path_utils import is_explicit_local_path

_CONFIG_NAME = "config.json"
_GENERATION_CONFIG_NAME = "generation_config.json"
_PREPROCESSOR_CONFIG_NAME = "preprocessor_config.json"
_TOKENIZER_NAME = "tokenizer.json"
_SINGLE_CHECKPOINT_NAME = "model.safetensors"
_SHARDED_CHECKPOINT_NAME = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class MoonshineArtifacts:
    """Immutable files forming one safe native Moonshine runtime."""

    source: str
    revision: str | None
    config: Path
    generation_config: Path
    preprocessor_config: Path
    tokenizer: Path
    checkpoint: Path

    @property
    def is_sharded(self) -> bool:
        return self.checkpoint.name.endswith(".safetensors.index.json")


def _required_local(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native Moonshine requires {filename!r} in {root}.")
    return path


def _validate_filename(
    value: str | None,
    *,
    name: str,
    optional: bool,
) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        suffix = " or None" if optional else ""
        raise ValueError(f"`{name}` must be a non-empty string{suffix}.")
    path = PurePosixPath(value)
    if ("\\" in value or path.is_absolute() or len(path.parts) != 1 or ".." in path.parts):
        raise ValueError(f"`{name}` must be one safe checkpoint-root filename.")


def _safe_shard_names(index_path: Path) -> tuple[str, ...]:
    document = read_json_file(index_path)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Moonshine Safetensors index must contain a non-empty `weight_map`.")
    names: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("Moonshine Safetensors index contains an invalid tensor name.")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError("Moonshine Safetensors index contains an invalid shard name.")
        shard_path = PurePosixPath(shard_name)
        if ("\\" in shard_name or shard_path.is_absolute() or len(shard_path.parts) != 1 or
                ".." in shard_path.parts):
            raise ValueError(f"Unsafe Moonshine checkpoint shard path {shard_name!r}.")
        if not shard_name.endswith(".safetensors"):
            raise ValueError("Moonshine checkpoint shard is not Safetensors: "
                             f"{shard_name!r}.")
        names.add(shard_name)
    return tuple(sorted(names))


def _validate_checkpoint(path: Path) -> None:
    if (path.suffix != ".safetensors" and not path.name.endswith(".safetensors.index.json")):
        raise ValueError(
            "Native Moonshine accepts a Safetensors checkpoint or index only. "
            "Pickle, ONNX, GGUF, and remote-code checkpoints are unsupported.")


def _resolve_local(
    source: Path,
    *,
    checkpoint_filename: str | None,
    tokenizer_filename: str,
) -> MoonshineArtifacts:
    checkpoint_override = None
    if source.is_file():
        _validate_checkpoint(source)
        checkpoint_override = source
        root = source.parent
    else:
        root = source
    checkpoint = checkpoint_override
    if checkpoint is None and checkpoint_filename is not None:
        checkpoint = _required_local(root, checkpoint_filename)
    if checkpoint is None:
        candidate = root / _SINGLE_CHECKPOINT_NAME
        checkpoint = (candidate if candidate.is_file() else _required_local(root, _SHARDED_CHECKPOINT_NAME))
    _validate_checkpoint(checkpoint)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard_name in _safe_shard_names(checkpoint):
            _required_local(root, shard_name)
    return MoonshineArtifacts(
        source=str(source),
        revision=None,
        config=_required_local(root, _CONFIG_NAME),
        generation_config=_required_local(root, _GENERATION_CONFIG_NAME),
        preprocessor_config=_required_local(root, _PREPROCESSOR_CONFIG_NAME),
        tokenizer=_required_local(root, tokenizer_filename),
        checkpoint=checkpoint,
    )


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


def resolve_moonshine_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str | None = None,
    tokenizer_filename: str = _TOKENIZER_NAME,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> MoonshineArtifacts:
    """Resolve one local root or one immutable Hub revision atomically."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Moonshine `source` must be a non-empty path or Hub ID.")
    _validate_filename(
        checkpoint_filename,
        name="checkpoint_filename",
        optional=True,
    )
    _validate_filename(
        tokenizer_filename,
        name="tokenizer_filename",
        optional=False,
    )
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _resolve_local(
            source_path.resolve(),
            checkpoint_filename=checkpoint_filename,
            tokenizer_filename=tokenizer_filename,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Moonshine model path was not found: {source_path}.")

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
    resolved = {
        name:
        resolve_pretrained_file(
            repo_id,
            name,
            cache_dir=cache_dir,
            revision=resolved_revision,
            token=token,
            local_files_only=local_files_only,
        )
        for name in (
            _GENERATION_CONFIG_NAME,
            _PREPROCESSOR_CONFIG_NAME,
            tokenizer_filename,
        )
    }
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
        raise FileNotFoundError(
            "Native Moonshine could not find a safe checkpoint among: "
            f"{names}. Legacy pickle and executable model formats are "
            "intentionally rejected.")
    _validate_checkpoint(checkpoint)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard_name in _safe_shard_names(checkpoint):
            resolve_pretrained_file(
                repo_id,
                shard_name,
                cache_dir=cache_dir,
                revision=resolved_revision,
                token=token,
                local_files_only=local_files_only,
            )
    return MoonshineArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=config,
        generation_config=resolved[_GENERATION_CONFIG_NAME],
        preprocessor_config=resolved[_PREPROCESSOR_CONFIG_NAME],
        tokenizer=resolved[tokenizer_filename],
        checkpoint=checkpoint,
    )


__all__ = [
    "MoonshineArtifacts",
    "resolve_moonshine_artifacts",
]
