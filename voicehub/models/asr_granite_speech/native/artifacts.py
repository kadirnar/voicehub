"""Coherent immutable artifact resolution for native Granite Speech."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.asr_granite_speech.native.metadata import GRANITE_SPEECH_CHECKPOINTS
from voicehub.path_utils import is_explicit_local_path

_REQUIRED_ASSETS = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_OPTIONAL_ASSETS = (
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
)
_SINGLE_CHECKPOINT = "model.safetensors"
_SHARDED_CHECKPOINT = "model.safetensors.index.json"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{7,64}$")


@dataclass(frozen=True, slots=True)
class GraniteSpeechArtifacts:
    """Files resolved from one local directory or immutable Hub snapshot."""

    source: str
    revision: str | None
    config: Path
    checkpoint: Path
    preprocessor_config: Path
    processor_config: Path
    tokenizer: Path
    tokenizer_config: Path
    special_tokens_map: Path | None = None
    added_tokens: Path | None = None
    chat_template: Path | None = None
    generation_config: Path | None = None

    @property
    def is_sharded(self) -> bool:
        return self.checkpoint.name.endswith(".safetensors.index.json", )


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native Granite Speech requires {filename!r} in {root}.")
    return path


def _optional(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path if path.is_file() else None


def _safe_shards(index_path: Path) -> tuple[str, ...]:
    document = read_json_file(index_path)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Granite Speech Safetensors index must contain a non-empty "
                         "`weight_map`.")
    shards: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if (not isinstance(tensor_name, str) or not tensor_name or not isinstance(shard_name, str) or
                not shard_name):
            raise ValueError("Granite Speech Safetensors index contains an invalid record.")
        relative = PurePosixPath(shard_name)
        if ("\\" in shard_name or relative.is_absolute() or len(relative.parts) != 1 or
                ".." in relative.parts or not shard_name.endswith(".safetensors")):
            raise ValueError(f"Unsafe Granite Speech shard path {shard_name!r}.")
        shards.add(shard_name)
    return tuple(sorted(shards))


def _construct(
    root: Path,
    *,
    source: str,
    revision: str | None,
    checkpoint: Path,
) -> GraniteSpeechArtifacts:
    required = {name: _required(root, name) for name in _REQUIRED_ASSETS}
    return GraniteSpeechArtifacts(
        source=source,
        revision=revision,
        config=required["config.json"],
        checkpoint=checkpoint,
        preprocessor_config=required["preprocessor_config.json"],
        processor_config=required["processor_config.json"],
        tokenizer=required["tokenizer.json"],
        tokenizer_config=required["tokenizer_config.json"],
        special_tokens_map=_optional(
            root,
            "special_tokens_map.json",
        ),
        added_tokens=_optional(root, "added_tokens.json"),
        chat_template=_optional(root, "chat_template.jinja"),
        generation_config=_optional(root, "generation_config.json"),
    )


def _resolve_local(source: Path) -> GraniteSpeechArtifacts:
    if source.is_file():
        if (source.suffix != ".safetensors" and not source.name.endswith(".safetensors.index.json")):
            raise ValueError("A direct Granite Speech checkpoint must be Safetensors "
                             "or an index JSON.")
        root = source.parent
        checkpoint = source
    else:
        root = source
        single = _optional(root, _SINGLE_CHECKPOINT)
        index = _optional(root, _SHARDED_CHECKPOINT)
        if single is not None and index is not None:
            raise ValueError(
                "Granite Speech directory contains both single-file and "
                "sharded model artifacts.")
        checkpoint = single or _required(root, _SHARDED_CHECKPOINT)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard in _safe_shards(checkpoint):
            _required(root, shard)
    return _construct(
        root,
        source=str(source),
        revision=None,
        checkpoint=checkpoint,
    )


def _remote_optional(
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


def _require_coherent_snapshot(
    root: Path,
    *artifacts: Path | None,
) -> None:
    mismatched = [
        path for path in artifacts if (path is not None and path.parent.resolve() != root.resolve())
    ]
    if mismatched:
        raise RuntimeError(
            "Granite Speech artifacts did not resolve from one immutable "
            "snapshot: " + ", ".join(str(path) for path in mismatched) + ".")


def resolve_granite_speech_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> GraniteSpeechArtifacts:
    """Resolve every runtime file from one revision and one source."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Granite Speech source must be a non-empty path or Hub ID.")
    local = Path(source).expanduser()
    if local.exists():
        return _resolve_local(local.resolve())
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Granite Speech model path was not found: {local}.")

    repo_id = str(source)
    known = GRANITE_SPEECH_CHECKPOINTS.get(repo_id)
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
    resolved_revision = pinned or (
        requested_revision.lower() if _IMMUTABLE_REVISION.fullmatch(requested_revision) else None)
    if (resolved_revision is None or not _IMMUTABLE_REVISION.fullmatch(resolved_revision)):
        raise RuntimeError(
            "VoiceHub could not prove an immutable Granite Speech Hub "
            "revision after resolving `config.json`. Retry online or pass "
            "an explicit commit.")
    root = config.parent
    required = {
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
    checkpoint = _remote_optional(
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
    shard_paths = []
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
    optional = {
        name:
        _remote_optional(
            repo_id,
            name,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        for name in _OPTIONAL_ASSETS
    }
    _require_coherent_snapshot(
        root,
        *required.values(),
        checkpoint,
        *shard_paths,
        *optional.values(),
    )
    return GraniteSpeechArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=config,
        checkpoint=checkpoint,
        preprocessor_config=required["preprocessor_config.json"],
        processor_config=required["processor_config.json"],
        tokenizer=required["tokenizer.json"],
        tokenizer_config=required["tokenizer_config.json"],
        special_tokens_map=optional["special_tokens_map.json"],
        added_tokens=optional["added_tokens.json"],
        chat_template=optional["chat_template.jinja"],
        generation_config=optional["generation_config.json"],
    )


__all__ = [
    "GraniteSpeechArtifacts",
    "resolve_granite_speech_artifacts",
]
