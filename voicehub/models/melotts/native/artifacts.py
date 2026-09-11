"""Deterministic artifact resolution for native and official MeloTTS
releases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.melotts.native.checkpoint import file_sha256
from voicehub.models.melotts.native.configuration import MeloTTSArchitectureConfig, load_melotts_config
from voicehub.models.melotts.native.metadata import MELOTTS_RELEASES


@dataclass(frozen=True, slots=True)
class MeloTTSArtifacts:
    config_path: Path
    checkpoint_path: Path
    config: MeloTTSArchitectureConfig
    legacy_checkpoint: bool
    release_alias: str | None = None
    repository: str | None = None
    revision: str | None = None
    expected_checkpoint_sha256: str | None = None


def _local_checkpoint(
    directory: Path,
    checkpoint_filename: str | None,
) -> Path:
    if checkpoint_filename is not None:
        candidate = directory / checkpoint_filename
        if not candidate.is_file():
            raise FileNotFoundError(f"MeloTTS checkpoint was not found: {candidate}.")
        return candidate.resolve()
    safe_checkpoint = directory / "model.safetensors"
    if safe_checkpoint.is_file():
        return safe_checkpoint.resolve()
    legacy_checkpoint = directory / "checkpoint.pth"
    if not legacy_checkpoint.is_file():
        raise FileNotFoundError(
            "MeloTTS artifact directory must contain `model.safetensors` "
            "or `checkpoint.pth`.")
    return legacy_checkpoint.resolve()


def resolve_melotts_artifacts(
    source: str | Path,
    *,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_filename: str | None = None,
    revision: str | None = None,
) -> MeloTTSArtifacts:
    """Resolve a local artifact, Hub repository, or pinned language alias."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("MeloTTS `source` must be a non-empty path or alias.")
    if checkpoint_filename is not None and (not isinstance(checkpoint_filename, str) or
                                            not checkpoint_filename.strip() or
                                            Path(checkpoint_filename).name != checkpoint_filename):
        raise ValueError("`checkpoint_filename` must be one plain file name.")

    source_text = str(source).strip()
    source_path = Path(source_text).expanduser()
    alias = (
        source_text.upper() if not source_path.exists() and source_text.upper() in MELOTTS_RELEASES else None)
    repository: str | None = None
    expected_checkpoint: str | None = None
    resolved_revision = revision
    if alias is not None:
        (
            repository,
            pinned_revision,
            expected_config,
            expected_checkpoint,
        ) = MELOTTS_RELEASES[alias]
        if revision is not None and revision != pinned_revision:
            raise ValueError(
                f"MeloTTS alias {alias!r} is pinned to {pinned_revision}; "
                "pass the repository ID directly to select another revision.")
        resolved_revision = pinned_revision
        if config_path is None:
            config_path = resolve_pretrained_file(
                repository,
                "config.json",
                revision=resolved_revision,
            )
        if checkpoint_path is None:
            checkpoint_path = resolve_pretrained_file(
                repository,
                checkpoint_filename or "checkpoint.pth",
                revision=resolved_revision,
            )
        actual_config = file_sha256(config_path)
        if actual_config != expected_config:
            raise ValueError("Pinned MeloTTS release configuration SHA-256 mismatch: "
                             f"{actual_config}.")
    else:
        path = source_path
        if path.exists():
            if path.is_dir():
                if config_path is None:
                    config_path = path / "config.json"
                if checkpoint_path is None:
                    checkpoint_path = _local_checkpoint(
                        path,
                        checkpoint_filename,
                    )
            elif path.is_file():
                if path.suffix.lower() == ".json":
                    config_path = config_path or path
                    if checkpoint_path is None:
                        checkpoint_path = _local_checkpoint(
                            path.parent,
                            checkpoint_filename,
                        )
                else:
                    checkpoint_path = checkpoint_path or path
                    config_path = config_path or path.parent / "config.json"
            else:  # pragma: no cover - unusual filesystem object
                raise ValueError(f"Unsupported MeloTTS artifact path: {path}.")
        else:
            repository = str(source)
            if config_path is None:
                config_path = resolve_pretrained_file(
                    repository,
                    "config.json",
                    revision=resolved_revision,
                )
            if checkpoint_path is None:
                filename = checkpoint_filename or "model.safetensors"
                checkpoint_path = resolve_pretrained_file(
                    repository,
                    filename,
                    revision=resolved_revision,
                )

    resolved_config = Path(config_path).expanduser().resolve()
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not resolved_config.is_file():
        raise FileNotFoundError(f"MeloTTS configuration was not found: {resolved_config}.")
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"MeloTTS checkpoint was not found: {resolved_checkpoint}.")
    suffix = resolved_checkpoint.suffix.lower()
    if suffix not in {".safetensors", ".pth", ".pt"}:
        raise ValueError("MeloTTS checkpoints must use .safetensors, .pth, or .pt.")
    return MeloTTSArtifacts(
        config_path=resolved_config,
        checkpoint_path=resolved_checkpoint,
        config=load_melotts_config(resolved_config),
        legacy_checkpoint=suffix != ".safetensors",
        release_alias=alias,
        repository=repository,
        revision=resolved_revision,
        expected_checkpoint_sha256=(expected_checkpoint if suffix != ".safetensors" else None),
    )


__all__ = [
    "MeloTTSArtifacts",
    "resolve_melotts_artifacts",
]
