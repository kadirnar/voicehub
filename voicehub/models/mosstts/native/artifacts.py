"""Coherent, immutable artifact resolution for native MOSS-TTS.

The loader intentionally accepts only Safetensors checkpoints and
resolves the model configuration, byte-BPE assets, and every shard from
one snapshot. This keeps inference and fine-tuning on exactly the same
artifact contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.mosstts.native.metadata import MOSS_CODEC_REVISIONS, MOSS_TTS_REVISIONS
from voicehub.path_utils import is_explicit_local_path

_CONFIG = "config.json"
_GENERATION_CONFIG = "generation_config.json"
_MERGES = "merges.txt"
_TOKENIZER_CONFIG = "tokenizer_config.json"
_VOCABULARY = "vocab.json"
_SINGLE_CHECKPOINT = "model.safetensors"
_SHARDED_CHECKPOINT = "model.safetensors.index.json"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{7,64}$")


@dataclass(frozen=True, slots=True)
class MossTTSArtifacts:
    """One complete native MOSS-TTS snapshot."""

    source: str
    revision: str | None
    root: Path
    config: Path
    checkpoint: Path
    vocabulary: Path
    merges: Path
    tokenizer_config: Path
    generation_config: Path | None = None

    @property
    def is_sharded(self) -> bool:
        return self.checkpoint.name.endswith(".safetensors.index.json")


@dataclass(frozen=True, slots=True)
class MossCodecArtifacts:
    """One complete native MOSS Audio Tokenizer snapshot."""

    source: str
    revision: str | None
    root: Path
    config: Path
    checkpoint: Path

    @property
    def is_sharded(self) -> bool:
        return self.checkpoint.name.endswith(".safetensors.index.json")


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native MOSS-TTS requires {filename!r} in {root}.")
    return path.resolve()


def _optional(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path.resolve() if path.is_file() else None


def _safe_shards(index: Path) -> tuple[str, ...]:
    values = read_json_file(index)
    weight_map = values.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("MOSS-TTS Safetensors index requires a non-empty `weight_map`.")
    shards: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("MOSS-TTS index contains an invalid tensor name.")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError("MOSS-TTS index contains an invalid shard name.")
        path = PurePosixPath(shard_name)
        if ("\\" in shard_name or path.is_absolute() or len(path.parts) != 1 or ".." in path.parts or
                not shard_name.endswith(".safetensors")):
            raise ValueError(f"Unsafe MOSS-TTS shard path {shard_name!r}.")
        shards.add(shard_name)
    return tuple(sorted(shards))


def _validate_checkpoint(path: Path) -> None:
    if (path.suffix != ".safetensors" and not path.name.endswith(".safetensors.index.json")):
        raise ValueError(
            "Native MOSS-TTS accepts only Safetensors checkpoints. "
            "GGUF, pickle, and PyTorch binary files are rejected.")


def _resolve_local(source: Path) -> MossTTSArtifacts:
    checkpoint_override: Path | None = None
    if source.is_file():
        _validate_checkpoint(source)
        checkpoint_override = source.resolve()
        root = source.parent.resolve()
    else:
        root = source.resolve()

    config = _required(root, _CONFIG)
    single = _optional(root, _SINGLE_CHECKPOINT)
    index = _optional(root, _SHARDED_CHECKPOINT)
    if checkpoint_override is None and single is not None and index is not None:
        raise ValueError(
            "MOSS-TTS artifact directory contains both single and sharded "
            "checkpoints; pass the intended checkpoint path explicitly.")
    checkpoint = checkpoint_override or single or index
    if checkpoint is None:
        raise FileNotFoundError(f"Native MOSS-TTS found no Safetensors checkpoint in {root}.")
    _validate_checkpoint(checkpoint)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _safe_shards(checkpoint):
            _required(root, shard)

    return MossTTSArtifacts(
        source=str(source),
        revision=None,
        root=root,
        config=config,
        checkpoint=checkpoint,
        vocabulary=_required(root, _VOCABULARY),
        merges=_required(root, _MERGES),
        tokenizer_config=_required(root, _TOKENIZER_CONFIG),
        generation_config=_optional(root, _GENERATION_CONFIG),
    )


def _resolve_codec_local(source: Path) -> MossCodecArtifacts:
    checkpoint_override: Path | None = None
    if source.is_file():
        _validate_checkpoint(source)
        checkpoint_override = source.resolve()
        root = source.parent.resolve()
    else:
        root = source.resolve()

    config = _required(root, _CONFIG)
    single = _optional(root, _SINGLE_CHECKPOINT)
    index = _optional(root, _SHARDED_CHECKPOINT)
    if checkpoint_override is None and single is not None and index is not None:
        raise ValueError(
            "MOSS codec artifact directory contains both single and sharded "
            "checkpoints; pass the intended checkpoint path explicitly.")
    checkpoint = checkpoint_override or single or index
    if checkpoint is None:
        raise FileNotFoundError(f"Native MOSS codec found no Safetensors checkpoint in {root}.")
    _validate_checkpoint(checkpoint)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _safe_shards(checkpoint):
            _required(root, shard)
    return MossCodecArtifacts(
        source=str(source),
        revision=None,
        root=root,
        config=config,
        checkpoint=checkpoint,
    )


def _remote_optional(
    repository: str,
    filename: str,
    *,
    revision: str,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> Path | None:
    try:
        return resolve_pretrained_file(
            repository,
            filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        ).absolute()
    except FileNotFoundError:
        return None


def _resolved_revision(
    repository: str,
    filename: str,
    *,
    requested: str,
    cache_dir: str | None,
) -> str:
    pinned = get_cached_hugging_face_commit(
        repository,
        filename,
        revision=requested,
        cache_dir=cache_dir,
    )
    if pinned is not None:
        return pinned
    if _IMMUTABLE_REVISION.fullmatch(requested):
        return requested.lower()
    raise RuntimeError(
        "VoiceHub could not prove an immutable MOSS-TTS Hub revision after "
        f"resolving {filename!r}. Retry online or pass an explicit commit.")


def _coherent(root: Path, *paths: Path | None) -> None:
    mismatched = [path for path in paths if path is not None and path.parent.absolute() != root.absolute()]
    if mismatched:
        raise RuntimeError(
            "MOSS-TTS artifacts did not resolve from one immutable snapshot: " +
            ", ".join(str(path) for path in mismatched) + ".")


def resolve_mosstts_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> MossTTSArtifacts:
    """Resolve one strict model/tokenizer checkpoint set.

    Official repository IDs default to their audited commit.  Other Hub
    repositories are accepted for fine-tuned exports only after VoiceHub
    can pin the requested revision to an immutable commit.
    """
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("MOSS-TTS `source` must be a non-empty path or Hub ID.")
    local = Path(source).expanduser()
    if local.exists():
        return _resolve_local(local)
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"MOSS-TTS model path was not found: {local}.")

    repository = str(source)
    requested = revision or MOSS_TTS_REVISIONS.get(repository, "main")
    config = resolve_pretrained_file(
        repository,
        _CONFIG,
        revision=requested,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    ).absolute()
    pinned = _resolved_revision(
        repository,
        _CONFIG,
        requested=requested,
        cache_dir=cache_dir,
    )
    vocabulary = resolve_pretrained_file(
        repository,
        _VOCABULARY,
        revision=pinned,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    ).absolute()
    merges = resolve_pretrained_file(
        repository,
        _MERGES,
        revision=pinned,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    ).absolute()
    tokenizer_config = resolve_pretrained_file(
        repository,
        _TOKENIZER_CONFIG,
        revision=pinned,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    ).absolute()

    single = _remote_optional(
        repository,
        _SINGLE_CHECKPOINT,
        revision=pinned,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    index = _remote_optional(
        repository,
        _SHARDED_CHECKPOINT,
        revision=pinned,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if single is not None and index is not None:
        raise ValueError("MOSS-TTS Hub snapshot contains both single and sharded "
                         "checkpoints.")
    checkpoint = single or index
    if checkpoint is None:
        raise FileNotFoundError(
            "Native MOSS-TTS found no `model.safetensors` or "
            "`model.safetensors.index.json`.")
    shard_paths: list[Path] = []
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _safe_shards(checkpoint):
            shard_paths.append(
                resolve_pretrained_file(
                    repository,
                    shard,
                    revision=pinned,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                ).absolute())
    generation_config = _remote_optional(
        repository,
        _GENERATION_CONFIG,
        revision=pinned,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    root = config.parent.absolute()
    _coherent(
        root,
        vocabulary,
        merges,
        tokenizer_config,
        checkpoint,
        generation_config,
        *shard_paths,
    )
    return MossTTSArtifacts(
        source=repository,
        revision=pinned,
        root=root,
        config=config,
        checkpoint=checkpoint,
        vocabulary=vocabulary,
        merges=merges,
        tokenizer_config=tokenizer_config,
        generation_config=generation_config,
    )


def resolve_moss_codec_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> MossCodecArtifacts:
    """Resolve one strict codec configuration/checkpoint snapshot."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("MOSS codec `source` must be a non-empty path or Hub ID.")
    local = Path(source).expanduser()
    if local.exists():
        return _resolve_codec_local(local)
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"MOSS codec path was not found: {local}.")

    repository = str(source)
    requested = revision or MOSS_CODEC_REVISIONS.get(repository, "main")
    config = resolve_pretrained_file(
        repository,
        _CONFIG,
        revision=requested,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    ).absolute()
    pinned = _resolved_revision(
        repository,
        _CONFIG,
        requested=requested,
        cache_dir=cache_dir,
    )
    single = _remote_optional(
        repository,
        _SINGLE_CHECKPOINT,
        revision=pinned,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    index = _remote_optional(
        repository,
        _SHARDED_CHECKPOINT,
        revision=pinned,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if single is not None and index is not None:
        raise ValueError("MOSS codec Hub snapshot contains both single and sharded checkpoints.")
    checkpoint = single or index
    if checkpoint is None:
        raise FileNotFoundError(
            "Native MOSS codec found no `model.safetensors` or "
            "`model.safetensors.index.json`.")
    shard_paths = [
        resolve_pretrained_file(
            repository,
            shard,
            revision=pinned,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        ).absolute() for shard in (
            _safe_shards(checkpoint) if checkpoint.name.endswith(".safetensors.index.json") else ())
    ]
    root = config.parent.absolute()
    _coherent(root, checkpoint, *shard_paths)
    return MossCodecArtifacts(
        source=repository,
        revision=pinned,
        root=root,
        config=config,
        checkpoint=checkpoint,
    )


__all__ = [
    "MossCodecArtifacts",
    "MossTTSArtifacts",
    "resolve_moss_codec_artifacts",
    "resolve_mosstts_artifacts",
]
