"""Coherent, immutable artifact resolution for native LLaSA and XCodec2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.path_utils import is_explicit_local_path

LLASA_MULTILINGUAL_REVISION = "7f094cb62b0a9779b334c60d039a61c5a6e04456"
XCODEC2_HF_REVISION = "64bd034d12d441299cdd535b15c33efd6ccdf252"
LLASA_MULTILINGUAL_REPOSITORY = "HKUSTAudio/Llasa-1B-Multilingual"
XCODEC2_HF_REPOSITORY = "HKUSTAudio/xcodec2-hf"
_CONFIG = "config.json"
_TOKENIZER = "tokenizer.json"
_TOKENIZER_CONFIG = "tokenizer_config.json"
_PREPROCESSOR_CONFIG = "preprocessor_config.json"
_CHECKPOINT = "model.safetensors"
_CHECKPOINT_INDEX = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class LlasaArtifacts:
    """Language-model, tokenizer, and checkpoint from one revision."""

    source: str
    revision: str | None
    config: Path
    tokenizer: Path
    checkpoint: Path | None
    tokenizer_config: Path | None = None

    @property
    def root(self) -> Path:
        return self.config.parent


@dataclass(frozen=True, slots=True)
class XCodec2Artifacts:
    """Native XCodec2 config/checkpoint/frontend metadata."""

    source: str
    revision: str | None
    config: Path
    checkpoint: Path
    preprocessor_config: Path | None = None

    @property
    def root(self) -> Path:
        return self.config.parent


def _safe_filename(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a non-empty filename or None.")
    normalized = value.strip()
    path = PurePosixPath(normalized)
    if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
        raise ValueError(f"`{name}` must be one safe artifact-root filename.")
    return normalized


def _required(root: Path, filename: str, *, owner: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native {owner} requires {filename!r} in {root}.")
    # HF snapshot names are symlinks to extensionless cache blobs. Retain the
    # artifact name for format dispatch and index-relative shard resolution.
    return path.absolute()


def _optional(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path.absolute() if path.is_file() else None


def _validate_checkpoint(path: Path, *, owner: str) -> None:
    if not (path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json")):
        raise ValueError(f"Native {owner} accepts Safetensors checkpoints only.")


def _shard_names(index: Path, *, owner: str) -> tuple[str, ...]:
    document = read_json_file(index)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{owner} Safetensors index requires a non-empty `weight_map`.")
    names = set()
    for name in weight_map.values():
        if not isinstance(name, str):
            raise ValueError(f"{owner} checkpoint shard names must be strings.")
        path = PurePosixPath(name)
        if (path.is_absolute() or len(path.parts) != 1 or ".." in path.parts or
                not name.endswith(".safetensors")):
            raise ValueError(f"Unsafe {owner} checkpoint shard {name!r}.")
        names.add(name)
    return tuple(sorted(names))


def _local_checkpoint(
    root: Path,
    *,
    filename: str | None,
    owner: str,
) -> Path:
    if filename is not None:
        checkpoint = _required(root, filename, owner=owner)
    else:
        checkpoint = _optional(root, _CHECKPOINT)
        if checkpoint is None:
            checkpoint = _required(root, _CHECKPOINT_INDEX, owner=owner)
    _validate_checkpoint(checkpoint, owner=owner)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _shard_names(checkpoint, owner=owner):
            _required(root, shard, owner=owner)
    return checkpoint


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


def _remote_checkpoint(
    source: str,
    *,
    filename: str | None,
    revision: str,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
    owner: str,
) -> Path:
    names = (filename, ) if filename is not None else (
        _CHECKPOINT,
        _CHECKPOINT_INDEX,
    )
    checkpoint = None
    for candidate in names:
        checkpoint = _remote_optional(
            source,
            candidate,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        if checkpoint is not None:
            break
    if checkpoint is None:
        raise FileNotFoundError(
            f"Native {owner} found none of: " + ", ".join(repr(name) for name in names) + ".")
    _validate_checkpoint(checkpoint, owner=owner)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _shard_names(checkpoint, owner=owner):
            resolve_pretrained_file(
                source,
                shard,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
    return checkpoint


def _resolved_revision(
    source: str,
    config: Path,
    requested: str,
    *,
    cache_dir: str | None,
) -> str:
    return (
        get_cached_hugging_face_commit(
            source,
            config.name,
            cache_dir=cache_dir,
            revision=requested,
        ) or requested)


def resolve_llasa_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    include_checkpoint: bool = True,
) -> LlasaArtifacts:
    """Resolve one LLaSA LM/tokenizer snapshot without a Hub SDK."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("LLaSA `source` must be a non-empty path or Hub ID.")
    checkpoint_filename = _safe_filename(
        checkpoint_filename,
        name="checkpoint_filename",
    )
    if not isinstance(include_checkpoint, bool):
        raise TypeError("`include_checkpoint` must be a boolean.")
    source_path = Path(source).expanduser()
    if source_path.exists():
        if not source_path.is_dir():
            raise NotADirectoryError("A local LLaSA source must be an artifact directory.")
        root = source_path.resolve()
        return LlasaArtifacts(
            source=str(root),
            revision=None,
            config=_required(root, _CONFIG, owner="LLaSA"),
            tokenizer=_required(root, _TOKENIZER, owner="LLaSA"),
            tokenizer_config=_optional(root, _TOKENIZER_CONFIG),
            checkpoint=(
                _local_checkpoint(
                    root,
                    filename=checkpoint_filename,
                    owner="LLaSA",
                ) if include_checkpoint else None),
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"LLaSA model path was not found: {source_path}.")

    repo_id = str(source)
    requested = (
        revision or (LLASA_MULTILINGUAL_REVISION if repo_id == LLASA_MULTILINGUAL_REPOSITORY else "main"))
    config = resolve_pretrained_file(
        repo_id,
        _CONFIG,
        revision=requested,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    resolved = _resolved_revision(
        repo_id,
        config,
        requested,
        cache_dir=cache_dir,
    )
    tokenizer = resolve_pretrained_file(
        repo_id,
        _TOKENIZER,
        revision=resolved,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    return LlasaArtifacts(
        source=repo_id,
        revision=resolved,
        config=config,
        tokenizer=tokenizer,
        tokenizer_config=_remote_optional(
            repo_id,
            _TOKENIZER_CONFIG,
            revision=resolved,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        ),
        checkpoint=(
            _remote_checkpoint(
                repo_id,
                filename=checkpoint_filename,
                revision=resolved,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                owner="LLaSA",
            ) if include_checkpoint else None),
    )


def resolve_xcodec2_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> XCodec2Artifacts:
    """Resolve the self-contained official XCodec2 conversion."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("XCodec2 `source` must be a non-empty path or Hub ID.")
    checkpoint_filename = _safe_filename(
        checkpoint_filename,
        name="codec_checkpoint_filename",
    )
    source_path = Path(source).expanduser()
    if source_path.exists():
        if not source_path.is_dir():
            raise NotADirectoryError("A local XCodec2 source must be an artifact directory.")
        root = source_path.resolve()
        return XCodec2Artifacts(
            source=str(root),
            revision=None,
            config=_required(root, _CONFIG, owner="XCodec2"),
            checkpoint=_local_checkpoint(
                root,
                filename=checkpoint_filename,
                owner="XCodec2",
            ),
            preprocessor_config=_optional(root, _PREPROCESSOR_CONFIG),
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"XCodec2 model path was not found: {source_path}.")

    repo_id = str(source)
    if repo_id in {"HKUSTAudio/xcodec2", "HKUST-Audio/xcodec2"}:
        raise ValueError(
            "The legacy XCodec2 repository uses an executable remote-code "
            "layout. Native VoiceHub requires the authors' self-contained "
            f"conversion {XCODEC2_HF_REPOSITORY!r}.")
    requested = (revision or (XCODEC2_HF_REVISION if repo_id == XCODEC2_HF_REPOSITORY else "main"))
    config = resolve_pretrained_file(
        repo_id,
        _CONFIG,
        revision=requested,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    resolved = _resolved_revision(
        repo_id,
        config,
        requested,
        cache_dir=cache_dir,
    )
    return XCodec2Artifacts(
        source=repo_id,
        revision=resolved,
        config=config,
        checkpoint=_remote_checkpoint(
            repo_id,
            filename=checkpoint_filename,
            revision=resolved,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            owner="XCodec2",
        ),
        preprocessor_config=_remote_optional(
            repo_id,
            _PREPROCESSOR_CONFIG,
            revision=resolved,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        ),
    )


__all__ = [
    "LLASA_MULTILINGUAL_REPOSITORY",
    "LLASA_MULTILINGUAL_REVISION",
    "XCODEC2_HF_REPOSITORY",
    "XCODEC2_HF_REVISION",
    "LlasaArtifacts",
    "XCodec2Artifacts",
    "resolve_llasa_artifacts",
    "resolve_xcodec2_artifacts",
]
