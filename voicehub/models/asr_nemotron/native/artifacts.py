"""Coherent, immutable artifact resolution for native Nemotron 3.5 ASR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.asr_nemotron.native.metadata import NEMOTRON_ASR_CHECKPOINT_FILENAME, NEMOTRON_ASR_CHECKPOINTS
from voicehub.path_utils import is_explicit_local_path

_REQUIRED_FILES = (
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_OPTIONAL_FILES = ("generation_config.json", )
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{7,64}$")
_UNSAFE_WEIGHT_NAMES = frozenset({
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "model.bin",
    "model.pt",
    "model.pth",
    "model.ckpt",
    "model.nemo",
})


@dataclass(frozen=True, slots=True)
class NemotronASRArtifacts:
    """Files from one local directory or one immutable Hub snapshot."""

    source: str
    revision: str | None
    root: Path
    config: Path
    checkpoint: Path
    processor_config: Path
    tokenizer: Path
    tokenizer_config: Path
    generation_config: Path | None = None


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native Nemotron ASR requires {filename!r} in {root}.")
    return path


def _optional(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path if path.is_file() else None


def _reject_unsafe_weight_alternatives(root: Path) -> None:
    alternatives = sorted(
        path.name for path in root.iterdir() if path.is_file() and (
            path.name in _UNSAFE_WEIGHT_NAMES or
            (path.suffix == ".safetensors" and path.name != NEMOTRON_ASR_CHECKPOINT_FILENAME)))
    if alternatives:
        raise ValueError(
            "Native Nemotron ASR requires exactly one "
            f"{NEMOTRON_ASR_CHECKPOINT_FILENAME!r} checkpoint; found "
            f"unsupported weight artifact(s): {alternatives!r}.")


def _construct(
    root: Path,
    *,
    source: str,
    revision: str | None,
) -> NemotronASRArtifacts:
    _reject_unsafe_weight_alternatives(root)
    required = {filename: _required(root, filename) for filename in _REQUIRED_FILES}
    checkpoint = _required(
        root,
        NEMOTRON_ASR_CHECKPOINT_FILENAME,
    )
    return NemotronASRArtifacts(
        source=source,
        revision=revision,
        root=root,
        config=required["config.json"],
        checkpoint=checkpoint,
        processor_config=required["processor_config.json"],
        tokenizer=required["tokenizer.json"],
        tokenizer_config=required["tokenizer_config.json"],
        generation_config=_optional(root, "generation_config.json"),
    )


def _resolve_local(source: Path) -> NemotronASRArtifacts:
    if source.is_file():
        if source.name != NEMOTRON_ASR_CHECKPOINT_FILENAME:
            raise ValueError(
                "A direct native Nemotron checkpoint must be named "
                f"{NEMOTRON_ASR_CHECKPOINT_FILENAME!r}.")
        root = source.parent
    elif source.is_dir():
        root = source
    else:  # pragma: no cover - caller resolves existence.
        raise FileNotFoundError(f"Nemotron ASR path was not found: {source}.")
    return _construct(
        root.resolve(),
        source=str(source),
        revision=None,
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
    *paths: Path | None,
) -> None:
    root = root.resolve()
    mismatched = [path for path in paths if path is not None and path.parent.resolve() != root]
    if mismatched:
        raise RuntimeError(
            "Nemotron ASR artifacts did not resolve from one immutable "
            "snapshot: " + ", ".join(str(path) for path in mismatched) + ".")


def resolve_nemotron_asr_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> NemotronASRArtifacts:
    """Resolve the safe runtime graph without executing repository code."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Nemotron ASR source must be a non-empty path or Hub ID.")
    local = Path(source).expanduser()
    if local.exists():
        if revision is not None:
            raise ValueError("`revision` cannot be applied to a local Nemotron artifact.")
        return _resolve_local(local.resolve())
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Nemotron ASR model path was not found: {local}.")

    repo_id = str(source)
    known = NEMOTRON_ASR_CHECKPOINTS.get(repo_id)
    requested_revision = revision or (str(known["revision"]) if known is not None else "main")
    config = resolve_pretrained_file(
        repo_id,
        "config.json",
        revision=requested_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    cached_commit = get_cached_hugging_face_commit(
        repo_id,
        "config.json",
        revision=requested_revision,
        cache_dir=cache_dir,
    )
    resolved_revision = cached_commit or (
        requested_revision.lower() if _IMMUTABLE_REVISION.fullmatch(requested_revision) else None)
    if (resolved_revision is None or _IMMUTABLE_REVISION.fullmatch(resolved_revision) is None):
        raise RuntimeError(
            "VoiceHub could not prove an immutable Nemotron ASR revision "
            "after resolving `config.json`. Retry online or pass an "
            "explicit commit.")

    root = config.parent
    required = {
        filename:
        resolve_pretrained_file(
            repo_id,
            filename,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        for filename in _REQUIRED_FILES[1:]
    }
    checkpoint = resolve_pretrained_file(
        repo_id,
        NEMOTRON_ASR_CHECKPOINT_FILENAME,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    optional = {
        filename:
        _remote_optional(
            repo_id,
            filename,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        for filename in _OPTIONAL_FILES
    }
    _require_coherent_snapshot(
        root,
        *required.values(),
        checkpoint,
        *optional.values(),
    )
    return NemotronASRArtifacts(
        source=repo_id,
        revision=resolved_revision,
        root=root,
        config=config,
        checkpoint=checkpoint,
        processor_config=required["processor_config.json"],
        tokenizer=required["tokenizer.json"],
        tokenizer_config=required["tokenizer_config.json"],
        generation_config=optional["generation_config.json"],
    )


__all__ = [
    "NemotronASRArtifacts",
    "resolve_nemotron_asr_artifacts",
]
