"""Coherent artifact resolution for native VibeVoice runtimes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.vibevoice.native.metadata import (
    QWEN_0_5B_TOKENIZER_REPOSITORY,
    QWEN_0_5B_TOKENIZER_REVISION,
    QWEN_1_5B_TOKENIZER_REPOSITORY,
    QWEN_1_5B_TOKENIZER_REVISION,
    VIBEVOICE_CHECKPOINTS,
)
from voicehub.path_utils import is_explicit_local_path

_COMMIT = re.compile(r"^[0-9a-fA-F]{7,64}$")
_UNSAFE_SUFFIXES = frozenset({
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
})


@dataclass(frozen=True, slots=True)
class VibeVoiceArtifacts:
    """One model snapshot and its explicitly versioned tokenizer snapshot."""

    source: str
    revision: str | None
    root: Path
    model_type: str
    config: Path
    checkpoint: Path
    shards: tuple[Path, ...]
    processor_config: Path
    tokenizer_source: str
    tokenizer_revision: str | None
    tokenizer: Path
    tokenizer_config: Path
    generation_config: Path | None = None
    chat_template: Path | None = None

    @property
    def is_sharded(self) -> bool:
        return self.checkpoint.name == "model.safetensors.index.json"


def _read_model_type(config: Path) -> str:
    value = read_json_file(config)
    model_type = str(value.get("model_type", "")).strip()
    if model_type not in {
            "vibevoice",
            "vibevoice_asr",
            "vibevoice_streaming",
    }:
        raise ValueError(f"Unsupported VibeVoice model type {model_type!r} in {config}.")
    return model_type


def _index_shard_names(index_path: Path) -> tuple[str, ...]:
    try:
        value = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not parse VibeVoice Safetensors index {index_path}: {error}.") from error
    weight_map = value.get("weight_map") if isinstance(value, dict) else None
    if (not isinstance(weight_map, dict) or not weight_map or
            any(not isinstance(name, str) or not name or not isinstance(shard, str) or not shard
                for name, shard in weight_map.items())):
        raise ValueError("VibeVoice index has an invalid `weight_map`.")
    shards = tuple(sorted(set(weight_map.values())))
    for shard in shards:
        path = PurePosixPath(shard)
        if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
            raise ValueError(f"Unsafe VibeVoice shard path {shard!r}.")
        if not shard.endswith(".safetensors"):
            raise ValueError(f"VibeVoice shard is not Safetensors: {shard!r}.")
    return shards


def _weight_artifact(root: Path) -> tuple[Path, tuple[Path, ...]]:
    single = root / "model.safetensors"
    index = root / "model.safetensors.index.json"
    if single.is_file() and index.is_file():
        raise ValueError(
            "VibeVoice artifact is ambiguous: both single and sharded "
            "Safetensors checkpoints are present.")
    unsafe = sorted(
        path.name for path in root.iterdir() if path.is_file() and path.suffix.lower() in _UNSAFE_SUFFIXES)
    if unsafe:
        raise ValueError(f"Native VibeVoice accepts Safetensors only; found {unsafe!r}.")
    if single.is_file():
        other = sorted(path.name for path in root.glob("*.safetensors") if path != single)
        if other:
            raise ValueError(
                "A single-file VibeVoice artifact contains unexpected "
                f"Safetensors files: {other!r}.")
        return single, ()
    if not index.is_file():
        raise FileNotFoundError(f"VibeVoice requires model.safetensors or its index in {root}.")
    names = _index_shard_names(index)
    shards = tuple(root / name for name in names)
    missing = [path for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "VibeVoice shard(s) were not found: " + ", ".join(str(path) for path in missing) + ".")
    indexed = {path.name for path in shards}
    extras = sorted(path.name for path in root.glob("*.safetensors") if path.name not in indexed)
    if extras:
        raise ValueError(f"VibeVoice artifact contains unindexed shards: {extras!r}.")
    return index, shards


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native VibeVoice requires {filename!r} in {root}.")
    return path


def _local_tokenizer(root: Path) -> tuple[str, None, Path, Path] | None:
    for candidate in (root, root / "tokenizer"):
        tokenizer = candidate / "tokenizer.json"
        config = candidate / "tokenizer_config.json"
        if tokenizer.is_file() and config.is_file():
            return str(candidate), None, tokenizer, config
    return None


def _tokenizer_reference(model_type: str) -> tuple[str, str]:
    if model_type == "vibevoice":
        return (
            QWEN_1_5B_TOKENIZER_REPOSITORY,
            QWEN_1_5B_TOKENIZER_REVISION,
        )
    if model_type == "vibevoice_streaming":
        return (
            QWEN_0_5B_TOKENIZER_REPOSITORY,
            QWEN_0_5B_TOKENIZER_REVISION,
        )
    raise ValueError("ASR tokenizer is stored with its model snapshot.")


def _resolve_tokenizer(
    root: Path,
    *,
    model_type: str,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> tuple[str, str | None, Path, Path]:
    local = _local_tokenizer(root)
    if local is not None:
        return local
    repository, revision = _tokenizer_reference(model_type)
    tokenizer = resolve_pretrained_file(
        repository,
        "tokenizer.json",
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    tokenizer_config = resolve_pretrained_file(
        repository,
        "tokenizer_config.json",
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if tokenizer.parent.resolve() != tokenizer_config.parent.resolve():
        raise RuntimeError("VibeVoice tokenizer files did not resolve from one snapshot.")
    return repository, revision, tokenizer, tokenizer_config


def _construct_local(
    source: Path,
    *,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> VibeVoiceArtifacts:
    root = source.parent if source.is_file() else source
    root = root.resolve()
    config = _required(root, "config.json")
    model_type = _read_model_type(config)
    checkpoint, shards = _weight_artifact(root)
    if source.is_file() and source.resolve() != checkpoint.resolve():
        raise ValueError(
            "Direct VibeVoice source must be model.safetensors or "
            "model.safetensors.index.json.")
    if model_type == "vibevoice_asr":
        tokenizer = _required(root, "tokenizer.json")
        tokenizer_config = _required(root, "tokenizer_config.json")
        processor = _required(root, "processor_config.json")
        generation = _required(root, "generation_config.json")
        chat_template = _required(root, "chat_template.jinja")
        tokenizer_source = str(root)
        tokenizer_revision = None
    else:
        processor = _required(root, "preprocessor_config.json")
        generation = None
        chat_template = None
        (
            tokenizer_source,
            tokenizer_revision,
            tokenizer,
            tokenizer_config,
        ) = _resolve_tokenizer(
            root,
            model_type=model_type,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    return VibeVoiceArtifacts(
        source=str(source),
        revision=None,
        root=root,
        model_type=model_type,
        config=config,
        checkpoint=checkpoint,
        shards=shards,
        processor_config=processor,
        tokenizer_source=tokenizer_source,
        tokenizer_revision=tokenizer_revision,
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        generation_config=generation,
        chat_template=chat_template,
    )


def _resolved_revision(
    repo_id: str,
    config: Path,
    *,
    requested: str,
    cache_dir: str | None,
) -> str:
    cached = get_cached_hugging_face_commit(
        repo_id,
        "config.json",
        revision=requested,
        cache_dir=cache_dir,
    )
    result = cached or (requested.lower() if _COMMIT.fullmatch(requested) else None)
    if result is None or _COMMIT.fullmatch(result) is None:
        raise RuntimeError(
            "VoiceHub could not prove an immutable VibeVoice snapshot after "
            f"resolving {config}.")
    return result


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


def _resolve_remote(
    repo_id: str,
    *,
    revision: str | None,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> VibeVoiceArtifacts:
    known = VIBEVOICE_CHECKPOINTS.get(repo_id)
    requested = revision or (str(known["revision"]) if known is not None else "main")
    config = resolve_pretrained_file(
        repo_id,
        "config.json",
        revision=requested,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    resolved = _resolved_revision(
        repo_id,
        config,
        requested=requested,
        cache_dir=cache_dir,
    )
    model_type = _read_model_type(config)
    if known is not None and model_type != known["model_type"]:
        raise ValueError("Pinned VibeVoice repository declared an unexpected model type.")
    index = _remote_optional(
        repo_id,
        "model.safetensors.index.json",
        revision=resolved,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if index is None:
        checkpoint = resolve_pretrained_file(
            repo_id,
            "model.safetensors",
            revision=resolved,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        shards: tuple[Path, ...] = ()
    else:
        names = _index_shard_names(index)
        shards = tuple(
            resolve_pretrained_file(
                repo_id,
                name,
                revision=resolved,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            ) for name in names)
        checkpoint = index
    model_paths = (config, checkpoint, *shards)
    root = config.parent.resolve()
    if any(path.parent.resolve() != root for path in model_paths):
        raise RuntimeError("VibeVoice model files did not resolve from one immutable snapshot.")
    if model_type == "vibevoice_asr":
        filenames = (
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
        resolved_files = {
            name:
            resolve_pretrained_file(
                repo_id,
                name,
                revision=resolved,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
            for name in filenames
        }
        if any(path.parent.resolve() != root for path in resolved_files.values()):
            raise RuntimeError("VibeVoice ASR processor files crossed snapshot roots.")
        processor = resolved_files["processor_config.json"]
        tokenizer = resolved_files["tokenizer.json"]
        tokenizer_config = resolved_files["tokenizer_config.json"]
        tokenizer_source = repo_id
        tokenizer_revision = resolved
        generation = resolve_pretrained_file(
            repo_id,
            "generation_config.json",
            revision=resolved,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        chat_template = resolve_pretrained_file(
            repo_id,
            "chat_template.jinja",
            revision=resolved,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        if (generation.parent.resolve() != root or chat_template.parent.resolve() != root):
            raise RuntimeError("VibeVoice ASR generation assets crossed snapshot roots.")
    else:
        processor = resolve_pretrained_file(
            repo_id,
            "preprocessor_config.json",
            revision=resolved,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        if processor.parent.resolve() != root:
            raise RuntimeError("VibeVoice processor crossed its model snapshot root.")
        (
            tokenizer_source,
            tokenizer_revision,
            tokenizer,
            tokenizer_config,
        ) = _resolve_tokenizer(
            root,
            model_type=model_type,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        generation = None
        chat_template = None
    return VibeVoiceArtifacts(
        source=repo_id,
        revision=resolved,
        root=root,
        model_type=model_type,
        config=config,
        checkpoint=checkpoint,
        shards=shards,
        processor_config=processor,
        tokenizer_source=tokenizer_source,
        tokenizer_revision=tokenizer_revision,
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        generation_config=generation,
        chat_template=chat_template,
    )


def resolve_vibevoice_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> VibeVoiceArtifacts:
    """Resolve VibeVoice without importing or executing repository code."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("VibeVoice source must be a non-empty path or Hub ID.")
    local = Path(source).expanduser()
    if local.exists():
        if revision is not None:
            raise ValueError("`revision` cannot be applied to a local artifact.")
        return _construct_local(
            local,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"VibeVoice model path was not found: {local}.")
    return _resolve_remote(
        str(source),
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )


__all__ = [
    "VibeVoiceArtifacts",
    "resolve_vibevoice_artifacts",
]
