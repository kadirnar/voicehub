"""Coherent native artifact resolution for Kokoro."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.models.kokoro.native.configuration import KokoroArchitectureConfig
from voicehub.models.kokoro.native.metadata import KOKORO_CHECKPOINT_REVISION

_OFFICIAL_REPOSITORIES = {
    "hexgrad/Kokoro-82M": "kokoro-v1_0.pth",
    "hexgrad/Kokoro-82M-v1.1-zh": "kokoro-v1_1-zh.pth",
}
_NATIVE_CHECKPOINT_NAMES = (
    "model.safetensors",
    "kokoro-v1_0.safetensors",
    "pytorch_model.safetensors",
)


@dataclass(frozen=True)
class KokoroArtifacts:
    """One config/checkpoint pair resolved from the same source revision."""

    source: str | Path
    revision: str | None
    config_path: Path
    config: KokoroArchitectureConfig
    checkpoint: Path
    legacy_pytorch: bool
    official_legacy_checkpoint: bool


def _safe_filename(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("`checkpoint_filename` must be non-empty or None.")
    filename = value.strip()
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("`checkpoint_filename` must be one checkpoint-root filename.")
    if not filename.endswith((".safetensors", ".pth")):
        raise ValueError("Kokoro checkpoints must use .safetensors or the released .pth "
                         "format.")
    return filename


def resolve_kokoro_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> KokoroArtifacts:
    """Resolve Kokoro without importing Hugging Face client libraries."""
    if not isinstance(local_files_only, bool):
        raise TypeError("`local_files_only` must be a boolean.")
    checkpoint_filename = _safe_filename(checkpoint_filename)
    source_path = Path(source).expanduser()
    source_string = str(source)
    if source_path.is_file():
        raise NotADirectoryError(
            "A native Kokoro checkpoint requires its matching config and "
            "voice directory. Pass the containing artifact directory instead "
            f"of the file: {source_path}.")
    resolved_revision = revision
    if (resolved_revision is None and not source_path.exists() and source_string in _OFFICIAL_REPOSITORIES):
        resolved_revision = KOKORO_CHECKPOINT_REVISION
    hub_kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "revision": resolved_revision,
        "token": token,
        "local_files_only": local_files_only,
    }
    config_path = resolve_pretrained_file(
        source,
        "config.json",
        **hub_kwargs,
    )
    config = KokoroArchitectureConfig.from_dict(read_json_file(config_path))

    if source_path.is_dir():
        names = ((checkpoint_filename, ) if checkpoint_filename is not None else (
            *_NATIVE_CHECKPOINT_NAMES,
            *_OFFICIAL_REPOSITORIES.values(),
        ))
        candidates = [source_path / name for name in names if (source_path / name).is_file()]
        if len(candidates) != 1:
            found = ", ".join(path.name for path in candidates) or "none"
            raise FileNotFoundError(
                "Kokoro artifact directory must resolve exactly one "
                f"checkpoint; found: {found}.")
        checkpoint = candidates[0].resolve()
    else:
        if checkpoint_filename is None:
            try:
                checkpoint_filename = _OFFICIAL_REPOSITORIES[source_string]
            except KeyError as error:
                raise ValueError(
                    "Remote Kokoro repositories must declare "
                    "`checkpoint_filename`; only immutable official artifact "
                    "layouts are inferred.") from error
        checkpoint = resolve_pretrained_file(
            source,
            checkpoint_filename,
            **hub_kwargs,
        )

    legacy = checkpoint.suffix.lower() == ".pth"
    official_legacy = (
        source_string in _OFFICIAL_REPOSITORIES and
        checkpoint.name == _OFFICIAL_REPOSITORIES[source_string] and
        resolved_revision == KOKORO_CHECKPOINT_REVISION)
    return KokoroArtifacts(
        source=source,
        revision=resolved_revision,
        config_path=config_path,
        config=config,
        checkpoint=checkpoint,
        legacy_pytorch=legacy,
        official_legacy_checkpoint=official_legacy,
    )


__all__ = ["KokoroArtifacts", "resolve_kokoro_artifacts"]
