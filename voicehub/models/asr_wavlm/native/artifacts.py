"""Coherent local and Hugging Face artifact resolution for WavLM CTC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.path_utils import is_explicit_local_path

_CONFIG_NAME = "config.json"
_VOCABULARY_NAME = "vocab.json"
_ADDED_TOKENS_NAME = "added_tokens.json"
_TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
_SPECIAL_TOKENS_NAME = "special_tokens_map.json"
_PREPROCESSOR_CONFIG_NAME = "preprocessor_config.json"
_SINGLE_CHECKPOINT_NAME = "model.safetensors"
_SHARDED_CHECKPOINT_NAME = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class WavLMArtifacts:
    """Immutable files forming one native WavLM CTC runtime."""

    source: str
    revision: str | None
    config: Path
    vocabulary: Path
    checkpoint: Path
    added_tokens: Path | None = None
    tokenizer_config: Path | None = None
    special_tokens_map: Path | None = None
    preprocessor_config: Path | None = None

    @property
    def is_sharded(self) -> bool:
        """Whether ``checkpoint`` is a Safetensors index."""
        return self.checkpoint.name.endswith(".safetensors.index.json")


def _required_local(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native WavLM requires {filename!r} in {root}.")
    return path


def _optional_local(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path if path.is_file() else None


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
        raise ValueError("WavLM Safetensors index must contain a non-empty `weight_map`.")
    names: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("WavLM Safetensors index contains an invalid tensor name.")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError("WavLM Safetensors index contains an invalid shard name.")
        shard_path = PurePosixPath(shard_name)
        if ("\\" in shard_name or shard_path.is_absolute() or len(shard_path.parts) != 1 or
                ".." in shard_path.parts):
            raise ValueError(f"Unsafe WavLM checkpoint shard path {shard_name!r}.")
        if not shard_name.endswith(".safetensors"):
            raise ValueError("WavLM checkpoint shard is not Safetensors: "
                             f"{shard_name!r}.")
        names.add(shard_name)
    return tuple(sorted(names))


def _validate_checkpoint(path: Path) -> None:
    if (path.suffix != ".safetensors" and not path.name.endswith(".safetensors.index.json")):
        raise ValueError("Native WavLM accepts a Safetensors checkpoint or index.")


def _resolve_local(
    source: Path,
    *,
    checkpoint_filename: str | None,
    vocabulary_filename: str,
) -> WavLMArtifacts:
    checkpoint_override: Path | None = None
    if source.is_file():
        _validate_checkpoint(source)
        checkpoint_override = source
        root = source.parent
    else:
        root = source

    config = _required_local(root, _CONFIG_NAME)
    vocabulary = _required_local(root, vocabulary_filename)
    checkpoint = checkpoint_override
    if checkpoint is None and checkpoint_filename is not None:
        checkpoint = _required_local(root, checkpoint_filename)
    if checkpoint is None:
        checkpoint = _optional_local(root, _SINGLE_CHECKPOINT_NAME)
    if checkpoint is None:
        checkpoint = _required_local(root, _SHARDED_CHECKPOINT_NAME)
    _validate_checkpoint(checkpoint)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _safe_shard_names(checkpoint):
            _required_local(root, shard)

    return WavLMArtifacts(
        source=str(source),
        revision=None,
        config=config,
        vocabulary=vocabulary,
        checkpoint=checkpoint,
        added_tokens=_optional_local(root, _ADDED_TOKENS_NAME),
        tokenizer_config=_optional_local(root, _TOKENIZER_CONFIG_NAME),
        special_tokens_map=_optional_local(root, _SPECIAL_TOKENS_NAME),
        preprocessor_config=_optional_local(
            root,
            _PREPROCESSOR_CONFIG_NAME,
        ),
    )


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


def resolve_wavlm_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str | None = None,
    vocabulary_filename: str = _VOCABULARY_NAME,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> WavLMArtifacts:
    """Resolve a local directory/file or one immutable Hub artifact set."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("WavLM `source` must be a non-empty path or Hub ID.")
    _validate_filename(
        checkpoint_filename,
        name="checkpoint_filename",
        optional=True,
    )
    _validate_filename(
        vocabulary_filename,
        name="vocabulary_filename",
        optional=False,
    )

    source_path = Path(source).expanduser()
    if source_path.exists():
        return _resolve_local(
            source_path.resolve(),
            checkpoint_filename=checkpoint_filename,
            vocabulary_filename=vocabulary_filename,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"WavLM model path was not found: {source_path}.")

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
    vocabulary = resolve_pretrained_file(
        repo_id,
        vocabulary_filename,
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
        raise FileNotFoundError("Native WavLM could not find a checkpoint among: "
                                f"{names}.")
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

    return WavLMArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=config,
        vocabulary=vocabulary,
        checkpoint=checkpoint,
        added_tokens=_resolve_optional_remote(
            repo_id,
            _ADDED_TOKENS_NAME,
            cache_dir=cache_dir,
            revision=resolved_revision,
            token=token,
            local_files_only=local_files_only,
        ),
        tokenizer_config=_resolve_optional_remote(
            repo_id,
            _TOKENIZER_CONFIG_NAME,
            cache_dir=cache_dir,
            revision=resolved_revision,
            token=token,
            local_files_only=local_files_only,
        ),
        special_tokens_map=_resolve_optional_remote(
            repo_id,
            _SPECIAL_TOKENS_NAME,
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


__all__ = ["WavLMArtifacts", "resolve_wavlm_artifacts"]
