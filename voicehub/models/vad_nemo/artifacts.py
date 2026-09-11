"""Artifact resolution for VoiceHub-native multilingual MarbleNet VAD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.vad_nemo.native.checkpoint import NATIVE_MARBLENET_VAD_FILENAME, convert_nemo_marblenet_checkpoint
from voicehub.models.vad_nemo.native.metadata import (
    MARBLENET_VAD_FILENAME,
    MARBLENET_VAD_REPOSITORY,
    MARBLENET_VAD_REVISION,
    MARBLENET_VAD_SHA256,
)
from voicehub.path_utils import is_explicit_local_path

_OFFICIAL_ALIASES = frozenset({
    "vad_multilingual_marblenet",
    "vad_multilingual_frame_marblenet",
    "nemo-marblenet-vad",
    MARBLENET_VAD_REPOSITORY,
})


@dataclass(frozen=True, slots=True)
class MarbleNetVADArtifacts:
    checkpoint: Path
    config: Path
    source: str
    revision: str | None
    converted_from_pickle: bool = False

    def __post_init__(self) -> None:
        for name in ("checkpoint", "config"):
            path = Path(getattr(self, name)).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"MarbleNet VAD {name} was not found: {path}.")
            object.__setattr__(self, name, path)


def _local_artifacts(
    source: Path,
    *,
    trust_pickle_checkpoint: bool,
) -> MarbleNetVADArtifacts:
    root = source if source.is_dir() else source.parent
    checkpoint = (
        source if source.is_file() and source.suffix.lower() == ".safetensors" else root /
        NATIVE_MARBLENET_VAD_FILENAME)
    config = root / "config.json"
    if checkpoint.is_file() and config.is_file():
        return MarbleNetVADArtifacts(
            checkpoint=checkpoint,
            config=config,
            source=str(source),
            revision=None,
        )

    legacy = None
    if source.is_file() and source.suffix.lower() in {".nemo", ".ckpt"}:
        legacy = source
    elif source.is_dir():
        candidates = sorted(
            path for path in source.iterdir() if path.is_file() and path.suffix.lower() in {".nemo", ".ckpt"})
        if len(candidates) > 1:
            raise ValueError(
                "A local MarbleNet directory may contain at most one "
                "top-level `.nemo` or `.ckpt` source.")
        legacy = candidates[0] if candidates else None
    if legacy is not None:
        if trust_pickle_checkpoint is not True:
            raise ValueError(
                "The local NeMo artifact contains a pickle-based checkpoint. "
                "Review it and pass `trust_pickle_checkpoint=True` for "
                "one-time conversion.")
        destination = root / ".voicehub-native" / "marblenet-vad"
        convert_nemo_marblenet_checkpoint(
            legacy,
            destination,
            trust_pickle_checkpoint=True,
            expected_sha256=(MARBLENET_VAD_SHA256 if legacy.name == MARBLENET_VAD_FILENAME else None),
        )
        return MarbleNetVADArtifacts(
            checkpoint=destination / NATIVE_MARBLENET_VAD_FILENAME,
            config=destination / "config.json",
            source=str(source),
            revision=None,
            converted_from_pickle=True,
        )
    raise FileNotFoundError(
        f"No complete native MarbleNet VAD artifact was found in {root}. "
        "Expected `model.safetensors` plus `config.json`, or one reviewed "
        "`.nemo`/`.ckpt` file for explicit conversion.")


def resolve_marblenet_vad_artifacts(
    source: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    token: str | bool | None,
    local_files_only: bool,
    trust_pickle_checkpoint: bool,
) -> MarbleNetVADArtifacts:
    """Resolve a safe native directory or convert the pinned release once."""
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _local_artifacts(
            source_path.resolve(),
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Local MarbleNet VAD model path was not found: {source_path}.")

    source_name = str(source)
    if source_name not in _OFFICIAL_ALIASES:
        checkpoint = resolve_pretrained_file(
            source_name,
            NATIVE_MARBLENET_VAD_FILENAME,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        config = resolve_pretrained_file(
            source_name,
            "config.json",
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        return MarbleNetVADArtifacts(
            checkpoint=checkpoint,
            config=config,
            source=source_name,
            revision=revision,
        )

    if revision is not None and revision != MARBLENET_VAD_REVISION:
        raise ValueError(
            "Official MarbleNet aliases are pinned to immutable revision "
            f"{MARBLENET_VAD_REVISION}; use a custom native repository for "
            "another revision.")
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "The official NeMo release publishes a pickle-based `.nemo` "
            "archive. Review its Apache-2.0 terms and pass "
            "`trust_pickle_checkpoint=True` once; subsequent loads use the "
            "snapshot-local Safetensors conversion.")
    source_checkpoint = resolve_pretrained_file(
        MARBLENET_VAD_REPOSITORY,
        MARBLENET_VAD_FILENAME,
        cache_dir=cache_dir,
        revision=MARBLENET_VAD_REVISION,
        token=token,
        local_files_only=local_files_only,
    )
    destination = source_checkpoint.parent / ".voicehub-native" / "marblenet-vad"
    checkpoint = destination / NATIVE_MARBLENET_VAD_FILENAME
    config = destination / "config.json"
    if not checkpoint.is_file() or not config.is_file():
        convert_nemo_marblenet_checkpoint(
            source_checkpoint,
            destination,
            trust_pickle_checkpoint=True,
            expected_sha256=MARBLENET_VAD_SHA256,
        )
    return MarbleNetVADArtifacts(
        checkpoint=checkpoint,
        config=config,
        source=source_name,
        revision=MARBLENET_VAD_REVISION,
        converted_from_pickle=True,
    )


__all__ = [
    "MarbleNetVADArtifacts",
    "resolve_marblenet_vad_artifacts",
]
