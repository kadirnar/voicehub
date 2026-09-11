"""Coherent, revision-pinned artifact resolution for native NeuTTS."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.neutts.native.metadata import NEUCODEC_REFERENCE, NEUTTS_VARIANTS
from voicehub.path_utils import is_explicit_local_path

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")
_CONFIG = "config.json"
_TOKENIZER = "tokenizer.json"
_TOKENIZER_CONFIG = "tokenizer_config.json"
_GENERATION_CONFIG = "generation_config.json"
_PREPROCESSOR_CONFIG = "preprocessor_config.json"
_CHECKPOINT = "model.safetensors"
_CHECKPOINT_INDEX = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class NeuTTSArtifacts:
    source: str
    revision: str | None
    config: Path
    tokenizer: Path
    checkpoint: Path
    tokenizer_config: Path | None = None
    generation_config: Path | None = None

    @property
    def root(self) -> Path:
        return self.config.parent


@dataclass(frozen=True, slots=True)
class NeuCodecArtifacts:
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
    return path.resolve()


def _optional(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path.resolve() if path.is_file() else None


def _validate_checkpoint(path: Path, *, owner: str) -> None:
    if not (path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json")):
        raise ValueError(
            f"Native {owner} accepts Safetensors checkpoints only; GGUF, "
            "ONNX, and pickle checkpoints require external or executable "
            "runtimes and are intentionally rejected.")


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
        logical_checkpoint = root / filename
        checkpoint = _required(root, filename, owner=owner)
    else:
        logical_checkpoint = root / _CHECKPOINT
        checkpoint = _optional(root, _CHECKPOINT)
        if checkpoint is None:
            logical_checkpoint = root / _CHECKPOINT_INDEX
            checkpoint = _required(root, _CHECKPOINT_INDEX, owner=owner)
    # Hugging Face snapshot directories use logical artifact-name symlinks
    # whose resolved blob names are content hashes without an extension.
    # Validate and classify the trusted logical name, and retain that name for
    # downstream readers that distinguish a single checkpoint from an index
    # by suffix. The OS still follows the symlink for I/O.
    _validate_checkpoint(logical_checkpoint, owner=owner)
    if logical_checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _shard_names(checkpoint, owner=owner):
            _required(root, shard, owner=owner)
    return logical_checkpoint.absolute()


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


def _requested_revision(
    source: str,
    revision: str | None,
    *,
    owner: str,
) -> str:
    reference = (
        NEUTTS_VARIANTS.get(source) if owner == "NeuTTS" else
        (NEUCODEC_REFERENCE if source == NEUCODEC_REFERENCE["model_id"] else None))
    requested = revision or (str(reference["revision"]) if reference is not None else None)
    if requested is None:
        raise ValueError(
            f"Remote custom {owner} sources require an explicit immutable "
            "`revision`; floating `main` artifacts are not accepted.")
    if not _IMMUTABLE_REVISION.fullmatch(requested):
        raise ValueError(f"Native {owner} requires a 40-64 character hexadecimal revision.")
    return requested.lower()


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


def _reject_external_format_source(source: str | Path, *, owner: str) -> None:
    lowered = str(source).strip().lower()
    if lowered.endswith((".gguf", ".onnx", ".bin", ".pt", ".pth")) or ("-gguf" in lowered):
        raise ValueError(
            f"Native {owner} accepts safe, differentiable Safetensors "
            "artifacts only. GGUF/ONNX are external inference formats and "
            "pickle checkpoints are not loaded.")


def resolve_neutts_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> NeuTTSArtifacts:
    """Resolve one coherent NeuTTS LM/tokenizer snapshot."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("NeuTTS `source` must be a non-empty path or Hub ID.")
    _reject_external_format_source(source, owner="NeuTTS")
    checkpoint_filename = _safe_filename(
        checkpoint_filename,
        name="checkpoint_filename",
    )
    source_path = Path(source).expanduser()
    if source_path.exists():
        if not source_path.is_dir():
            raise NotADirectoryError("A local NeuTTS source must be an artifact directory.")
        # Preserve the caller's logical snapshot path. On platforms where a
        # temporary-directory prefix is itself a symlink (for example
        # /var -> /private/var on macOS), resolving the root would silently
        # change the public artifact identity before we preserve the logical
        # model.safetensors symlink below.
        root = source_path.absolute()
        return NeuTTSArtifacts(
            source=str(root),
            revision=None,
            config=_required(root, _CONFIG, owner="NeuTTS"),
            tokenizer=_required(root, _TOKENIZER, owner="NeuTTS"),
            checkpoint=_local_checkpoint(
                root,
                filename=checkpoint_filename,
                owner="NeuTTS",
            ),
            tokenizer_config=_optional(root, _TOKENIZER_CONFIG),
            generation_config=_optional(root, _GENERATION_CONFIG),
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"NeuTTS model path was not found: {source_path}.")

    repo_id = str(source)
    requested = _requested_revision(
        repo_id,
        revision,
        owner="NeuTTS",
    )
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
    return NeuTTSArtifacts(
        source=repo_id,
        revision=resolved,
        config=config,
        tokenizer=tokenizer,
        checkpoint=_remote_checkpoint(
            repo_id,
            filename=checkpoint_filename,
            revision=resolved,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            owner="NeuTTS",
        ),
        tokenizer_config=_remote_optional(
            repo_id,
            _TOKENIZER_CONFIG,
            revision=resolved,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        ),
        generation_config=_remote_optional(
            repo_id,
            _GENERATION_CONFIG,
            revision=resolved,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        ),
    )


def resolve_neucodec_artifacts(
    source: str | Path = str(NEUCODEC_REFERENCE["model_id"]),
    *,
    checkpoint_filename: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> NeuCodecArtifacts:
    """Resolve the safe self-contained NeuCodec conversion."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("NeuCodec `source` must be a non-empty path or Hub ID.")
    _reject_external_format_source(source, owner="NeuCodec")
    checkpoint_filename = _safe_filename(
        checkpoint_filename,
        name="codec_checkpoint_filename",
    )
    source_path = Path(source).expanduser()
    if source_path.exists():
        if not source_path.is_dir():
            raise NotADirectoryError("A local NeuCodec source must be an artifact directory.")
        root = source_path.resolve()
        return NeuCodecArtifacts(
            source=str(root),
            revision=None,
            config=_required(root, _CONFIG, owner="NeuCodec"),
            checkpoint=_local_checkpoint(
                root,
                filename=checkpoint_filename,
                owner="NeuCodec",
            ),
            preprocessor_config=_optional(root, _PREPROCESSOR_CONFIG),
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"NeuCodec model path was not found: {source_path}.")
    repo_id = str(source)
    if "distill" in repo_id.lower():
        raise ValueError(
            "The public distilled NeuCodec repository has no audited "
            "self-contained Safetensors artifact. Native VoiceHub fails "
            "closed instead of loading its legacy pickle checkpoint.")
    requested = _requested_revision(
        repo_id,
        revision,
        owner="NeuCodec",
    )
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
    return NeuCodecArtifacts(
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
            owner="NeuCodec",
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
    "NeuCodecArtifacts",
    "NeuTTSArtifacts",
    "resolve_neucodec_artifacts",
    "resolve_neutts_artifacts",
]
