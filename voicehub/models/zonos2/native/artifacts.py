"""Artifact resolution for pinned safe ZONOS2 inference and training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.zonos2.native.checkpoint import verify_file_integrity
from voicehub.models.zonos2.native.metadata import (
    ZONOS2_OFFICIAL_CHECKPOINT,
    ZONOS2_OFFICIAL_CHECKPOINT_REVISION,
    ZONOS2_PARAMS_SHA256,
    ZONOS2_SAFE_CONVERSION,
    ZONOS2_SAFE_CONVERSION_FILENAME,
    ZONOS2_SAFE_CONVERSION_REVISION,
    ZONOS2_SAFE_CONVERSION_SHA256,
    ZONOS2_SAFE_CONVERSION_SIZE,
    ZONOS2_SPEAKER_ENCODER,
    ZONOS2_SPEAKER_ENCODER_REVISION,
    ZONOS2_SPEAKER_ENCODER_SHA256,
    ZONOS2_SPEAKER_ENCODER_SIZE,
)


@dataclass(frozen=True, slots=True)
class Zonos2Artifacts:
    config: Path
    checkpoint: Path
    source: str
    revision: str | None
    safe_conversion: bool


@dataclass(frozen=True, slots=True)
class Zonos2SpeakerArtifacts:
    config: Path
    checkpoint: Path
    preprocessor_config: Path | None
    source: str
    revision: str | None


def _first_local_file(directory: Path, names: tuple[str, ...]) -> Path:
    found = [directory / name for name in names if (directory / name).is_file()]
    if not found:
        raise FileNotFoundError(f"Could not find any of {names!r} in {directory}.")
    if len(found) > 1:
        raise ValueError(
            f"ZONOS2 artifact directory is ambiguous; found "
            f"{[path.name for path in found]!r}.")
    return found[0]


def _resolve_first_remote(
    source: str,
    names: tuple[str, ...],
    **kwargs,
) -> Path:
    failures: list[str] = []
    for name in names:
        try:
            return resolve_pretrained_file(source, name, **kwargs)
        except FileNotFoundError as error:
            failures.append(str(error))
    raise FileNotFoundError(
        f"Could not resolve any ZONOS2 artifact {names!r} from {source!r}. " + " ".join(failures))


def resolve_zonos2_artifacts(
    pretrained_model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = False,
) -> Zonos2Artifacts:
    """Resolve a strict Safetensors checkpoint and architecture config.

    The official Zyphra repository currently contains only
    ``model.pth``. Supplying its repository ID therefore resolves the
    pinned independently converted BF16 Safetensors artifact while
    retaining Zyphra's pinned ``params.json``. Legacy pickle conversion
    is available only through the explicit converter in
    :mod:`voicehub.models.zonos2.native.checkpoint`.
    """
    source_path = Path(pretrained_model_name_or_path).expanduser()
    if source_path.is_dir():
        config = _first_local_file(
            source_path,
            ("config.json", "params.json"),
        )
        checkpoint = _first_local_file(
            source_path,
            ("model.safetensors", ZONOS2_SAFE_CONVERSION_FILENAME),
        )
        return Zonos2Artifacts(
            config=config.resolve(),
            checkpoint=checkpoint.resolve(),
            source=str(source_path),
            revision=None,
            safe_conversion=False,
        )
    if source_path.is_file():
        if source_path.suffix.lower() != ".safetensors":
            raise ValueError(
                "Native ZONOS2 loads Safetensors only. Convert a trusted "
                "legacy checkpoint explicitly before loading it.")
        config = _first_local_file(
            source_path.parent,
            ("config.json", "params.json"),
        )
        return Zonos2Artifacts(
            config=config.resolve(),
            checkpoint=source_path.resolve(),
            source=str(source_path),
            revision=None,
            safe_conversion=False,
        )

    source = str(pretrained_model_name_or_path)
    common = {
        "cache_dir": cache_dir,
        "token": token,
        "local_files_only": local_files_only,
    }
    safe_conversion = source in {
        ZONOS2_OFFICIAL_CHECKPOINT,
        ZONOS2_SAFE_CONVERSION,
    }
    if safe_conversion:
        checkpoint_revision = (
            ZONOS2_SAFE_CONVERSION_REVISION
            if revision is None or source == ZONOS2_OFFICIAL_CHECKPOINT else revision)
        checkpoint = resolve_pretrained_file(
            ZONOS2_SAFE_CONVERSION,
            ZONOS2_SAFE_CONVERSION_FILENAME,
            revision=checkpoint_revision,
            **common,
        )
        config = resolve_pretrained_file(
            ZONOS2_OFFICIAL_CHECKPOINT,
            "params.json",
            revision=ZONOS2_OFFICIAL_CHECKPOINT_REVISION,
            **common,
        )
        if verify_integrity and checkpoint_revision == ZONOS2_SAFE_CONVERSION_REVISION:
            verify_file_integrity(
                checkpoint,
                expected_size=ZONOS2_SAFE_CONVERSION_SIZE,
                expected_sha256=ZONOS2_SAFE_CONVERSION_SHA256,
            )
            verify_file_integrity(
                config,
                expected_sha256=ZONOS2_PARAMS_SHA256,
            )
        return Zonos2Artifacts(
            config=config,
            checkpoint=checkpoint,
            source=ZONOS2_SAFE_CONVERSION,
            revision=checkpoint_revision,
            safe_conversion=True,
        )

    resolved_revision = revision
    config = _resolve_first_remote(
        source,
        ("config.json", "params.json"),
        revision=resolved_revision,
        **common,
    )
    checkpoint = _resolve_first_remote(
        source,
        ("model.safetensors", ZONOS2_SAFE_CONVERSION_FILENAME),
        revision=resolved_revision,
        **common,
    )
    return Zonos2Artifacts(
        config=config,
        checkpoint=checkpoint,
        source=source,
        revision=resolved_revision,
        safe_conversion=False,
    )


def resolve_zonos2_speaker_artifacts(
    *,
    source: str = ZONOS2_SPEAKER_ENCODER,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = False,
) -> Zonos2SpeakerArtifacts:
    source_path = Path(source).expanduser()
    if source_path.is_dir():
        checkpoint = source_path / "model.safetensors"
        config = source_path / "config.json"
        preprocessor = source_path / "preprocessor_config.json"
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError(
                f"Speaker directory {source_path} requires config.json and "
                "model.safetensors.")
        return Zonos2SpeakerArtifacts(
            config=config.resolve(),
            checkpoint=checkpoint.resolve(),
            preprocessor_config=(preprocessor.resolve() if preprocessor.is_file() else None),
            source=str(source_path),
            revision=None,
        )
    resolved_revision = (
        ZONOS2_SPEAKER_ENCODER_REVISION
        if revision is None and source == ZONOS2_SPEAKER_ENCODER else revision)
    common = {
        "cache_dir": cache_dir,
        "revision": resolved_revision,
        "token": token,
        "local_files_only": local_files_only,
    }
    checkpoint = resolve_pretrained_file(source, "model.safetensors", **common)
    config = resolve_pretrained_file(source, "config.json", **common)
    try:
        preprocessor = resolve_pretrained_file(
            source,
            "preprocessor_config.json",
            **common,
        )
    except FileNotFoundError:
        preprocessor = None
    if (verify_integrity and source == ZONOS2_SPEAKER_ENCODER and
            resolved_revision == ZONOS2_SPEAKER_ENCODER_REVISION):
        verify_file_integrity(
            checkpoint,
            expected_size=ZONOS2_SPEAKER_ENCODER_SIZE,
            expected_sha256=ZONOS2_SPEAKER_ENCODER_SHA256,
        )
    return Zonos2SpeakerArtifacts(
        config=config,
        checkpoint=checkpoint,
        preprocessor_config=preprocessor,
        source=source,
        revision=resolved_revision,
    )


__all__ = [
    "Zonos2Artifacts",
    "Zonos2SpeakerArtifacts",
    "resolve_zonos2_artifacts",
    "resolve_zonos2_speaker_artifacts",
]
