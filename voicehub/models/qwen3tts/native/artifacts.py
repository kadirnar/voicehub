"""Coherent local/Hub artifact resolution for native Qwen3-TTS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.qwen3tts.native.metadata import QWEN3_TTS_CHECKPOINTS
from voicehub.path_utils import is_explicit_local_path

_ROOT_REQUIRED = (
    "config.json",
    "model.safetensors",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
)
_SPEECH_REQUIRED = (
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
)


@dataclass(frozen=True, slots=True)
class Qwen3TTSArtifacts:
    source: str
    revision: str | None
    config: Path
    checkpoint: Path
    vocab: Path
    merges: Path
    tokenizer_config: Path
    generation_config: Path | None
    speech_config: Path
    speech_checkpoint: Path
    speech_preprocessor_config: Path | None


def _required(root: Path, filename: str) -> Path:
    path = root.joinpath(*filename.split("/"))
    if not path.is_file():
        raise FileNotFoundError(f"Native Qwen3-TTS requires {filename!r} in {root}.")
    return path


def _optional(root: Path, filename: str) -> Path | None:
    path = root.joinpath(*filename.split("/"))
    return path if path.is_file() else None


def _resolve_local(source: Path) -> Qwen3TTSArtifacts:
    root = source.parent if source.is_file() else source
    if source.is_file() and source.name != "model.safetensors":
        raise ValueError("A direct Qwen3-TTS checkpoint must be model.safetensors.")
    paths = {filename: _required(root, filename) for filename in _ROOT_REQUIRED + _SPEECH_REQUIRED}
    return Qwen3TTSArtifacts(
        source=str(source),
        revision=None,
        config=paths["config.json"],
        checkpoint=source if source.is_file() else paths["model.safetensors"],
        vocab=paths["vocab.json"],
        merges=paths["merges.txt"],
        tokenizer_config=paths["tokenizer_config.json"],
        generation_config=_optional(root, "generation_config.json"),
        speech_config=paths["speech_tokenizer/config.json"],
        speech_checkpoint=paths["speech_tokenizer/model.safetensors"],
        speech_preprocessor_config=_optional(
            root,
            "speech_tokenizer/preprocessor_config.json",
        ),
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


def resolve_qwen3_tts_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Qwen3TTSArtifacts:
    """Resolve every architecture asset from one immutable snapshot."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Qwen3-TTS source must be a non-empty path or Hub ID.")
    local = Path(source).expanduser()
    if local.exists():
        return _resolve_local(local.resolve())
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Qwen3-TTS model path was not found: {local}.")

    repo_id = str(source)
    known = QWEN3_TTS_CHECKPOINTS.get(repo_id)
    requested = revision or (str(known["revision"]) if known is not None else "main")
    config = resolve_pretrained_file(
        repo_id,
        "config.json",
        revision=requested,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    pinned = get_cached_hugging_face_commit(
        repo_id,
        "config.json",
        revision=requested,
        cache_dir=cache_dir,
    )
    resolved_revision = pinned or requested
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
        for filename in _ROOT_REQUIRED[1:] + _SPEECH_REQUIRED
    }
    generation = _remote_optional(
        repo_id,
        "generation_config.json",
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    speech_preprocessor = _remote_optional(
        repo_id,
        "speech_tokenizer/preprocessor_config.json",
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    return Qwen3TTSArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=config,
        checkpoint=required["model.safetensors"],
        vocab=required["vocab.json"],
        merges=required["merges.txt"],
        tokenizer_config=required["tokenizer_config.json"],
        generation_config=generation,
        speech_config=required["speech_tokenizer/config.json"],
        speech_checkpoint=required["speech_tokenizer/model.safetensors"],
        speech_preprocessor_config=speech_preprocessor,
    )


__all__ = [
    "Qwen3TTSArtifacts",
    "resolve_qwen3_tts_artifacts",
]
