"""Coherent local and Hugging Face artifact resolution for Wav2Vec2 CTC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.path_utils import is_explicit_local_path

_CONFIG_NAME = "config.json"
_VOCABULARY_NAME = "vocab.json"
_TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
_SPECIAL_TOKENS_NAME = "special_tokens_map.json"
_PREPROCESSOR_CONFIG_NAME = "preprocessor_config.json"
_SINGLE_CHECKPOINT_NAME = "model.safetensors"
_SHARDED_CHECKPOINT_NAME = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class Wav2Vec2Artifacts:
    """Immutable files forming one native Wav2Vec2 CTC runtime."""

    source: str
    revision: str | None
    config: Path
    vocabulary: Path
    checkpoint: Path
    tokenizer_config: Path | None = None
    special_tokens_map: Path | None = None
    preprocessor_config: Path | None = None

    @property
    def is_sharded(self) -> bool:
        """Whether ``checkpoint`` is a Safetensors index."""
        return self.checkpoint.name.endswith(".safetensors.index.json")


@dataclass(frozen=True, slots=True)
class Wav2Vec2ClassificationArtifacts:
    """Files required by a native Wav2Vec2 classification runtime."""

    source: str
    revision: str | None
    config: Path
    checkpoint: Path
    preprocessor_config: Path | None = None

    @property
    def is_sharded(self) -> bool:
        """Whether ``checkpoint`` is a Safetensors index."""
        return self.checkpoint.name.endswith(".safetensors.index.json")


def _required_local(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native Wav2Vec2 requires {filename!r} in {root}.")
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
        raise ValueError("Wav2Vec2 Safetensors index must contain a non-empty "
                         "`weight_map`.")
    names = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("Wav2Vec2 Safetensors index contains an invalid tensor name.")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError("Wav2Vec2 Safetensors index contains an invalid shard name.")
        shard_path = PurePosixPath(shard_name.replace("\\", "/"))
        if (shard_path.is_absolute() or len(shard_path.parts) != 1 or ".." in shard_path.parts):
            raise ValueError(f"Unsafe Wav2Vec2 checkpoint shard path {shard_name!r}.")
        if not shard_name.endswith(".safetensors"):
            raise ValueError("Wav2Vec2 checkpoint shard is not Safetensors: "
                             f"{shard_name!r}.")
        names.add(shard_name)
    return tuple(sorted(names))


def _validate_checkpoint(path: Path) -> None:
    if (path.suffix != ".safetensors" and not path.name.endswith(".safetensors.index.json")):
        raise ValueError("Native Wav2Vec2 accepts a Safetensors checkpoint or index.")


def _resolve_local(
    source: Path,
    *,
    checkpoint_filename: str | None,
    vocabulary_filename: str,
) -> Wav2Vec2Artifacts:
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

    return Wav2Vec2Artifacts(
        source=str(source),
        revision=None,
        config=config,
        vocabulary=vocabulary,
        checkpoint=checkpoint,
        tokenizer_config=_optional_local(root, _TOKENIZER_CONFIG_NAME),
        special_tokens_map=_optional_local(root, _SPECIAL_TOKENS_NAME),
        preprocessor_config=_optional_local(
            root,
            _PREPROCESSOR_CONFIG_NAME,
        ),
    )


def resolve_wav2vec2_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str | None = None,
    vocabulary_filename: str = _VOCABULARY_NAME,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Wav2Vec2Artifacts:
    """Resolve a local directory/file or immutable Hub artifact set."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Wav2Vec2 `source` must be a non-empty path or Hub ID.")
    for name, value, optional in (
        ("checkpoint_filename", checkpoint_filename, True),
        ("vocabulary_filename", vocabulary_filename, False),
    ):
        if value is None and optional:
            continue
        if not isinstance(value, str) or not value.strip():
            suffix = " or None" if optional else ""
            raise ValueError(f"`{name}` must be a non-empty string{suffix}.")

    source_path = Path(source).expanduser()
    if source_path.exists():
        return _resolve_local(
            source_path.resolve(),
            checkpoint_filename=checkpoint_filename,
            vocabulary_filename=vocabulary_filename,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Wav2Vec2 model path was not found: {source_path}.")

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
        raise FileNotFoundError("Native Wav2Vec2 could not find a checkpoint among: "
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

    return Wav2Vec2Artifacts(
        source=repo_id,
        revision=resolved_revision,
        config=config,
        vocabulary=vocabulary,
        checkpoint=checkpoint,
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


def resolve_wav2vec2_classification_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Wav2Vec2ClassificationArtifacts:
    """Resolve one coherent native Wav2Vec2 classifier artifact set."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Wav2Vec2 `source` must be a non-empty path or Hub ID.")
    if checkpoint_filename is not None:
        if not isinstance(checkpoint_filename, str) or not checkpoint_filename.strip():
            raise ValueError("`checkpoint_filename` must be a non-empty string or None.")

    source_path = Path(source).expanduser()
    if source_path.exists():
        resolved_source = source_path.resolve()
        if resolved_source.is_file():
            _validate_checkpoint(resolved_source)
            root = resolved_source.parent
            checkpoint = resolved_source
        else:
            root = resolved_source
            checkpoint = None
        config = _required_local(root, _CONFIG_NAME)
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
        return Wav2Vec2ClassificationArtifacts(
            source=str(resolved_source),
            revision=None,
            config=config,
            checkpoint=checkpoint,
            preprocessor_config=_optional_local(
                root,
                _PREPROCESSOR_CONFIG_NAME,
            ),
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Wav2Vec2 model path was not found: {source_path}.")

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
        raise FileNotFoundError("Native Wav2Vec2 could not find a checkpoint among: "
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
    return Wav2Vec2ClassificationArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=config,
        checkpoint=checkpoint,
        preprocessor_config=_resolve_optional_remote(
            repo_id,
            _PREPROCESSOR_CONFIG_NAME,
            cache_dir=cache_dir,
            revision=resolved_revision,
            token=token,
            local_files_only=local_files_only,
        ),
    )


__all__ = [
    "Wav2Vec2Artifacts",
    "Wav2Vec2ClassificationArtifacts",
    "resolve_wav2vec2_artifacts",
    "resolve_wav2vec2_classification_artifacts",
]
