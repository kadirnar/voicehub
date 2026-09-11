"""Artifact resolution for VoiceHub-native PyanNet providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.models.vad_pyannote.native.checkpoint import (
    PYANNOTE_BROUHAHA_REVISION,
    PYANNOTE_SEGMENTATION_3_REVISION,
    PYANNOTE_SEGMENTATION_3_SHA256,
    PYANNOTE_SEGMENTATION_REVISION,
    PYANNOTE_SEGMENTATION_SHA256,
    config_for_variant,
    convert_pyannote_lightning_checkpoint,
)
from voicehub.path_utils import is_explicit_local_path


@dataclass(frozen=True, slots=True)
class PyanNetArtifacts:
    source: str
    revision: str | None
    config: Path
    checkpoint: Path
    converted_from_pickle: bool = False


_OFFICIAL_ARTIFACTS = {
    "segmentation": {
        "repo": "pyannote/segmentation",
        "revision": PYANNOTE_SEGMENTATION_REVISION,
        "sha256": PYANNOTE_SEGMENTATION_SHA256,
    },
    "powerset-segmentation": {
        "repo": "pyannote/segmentation-3.0",
        "revision": PYANNOTE_SEGMENTATION_3_REVISION,
        "sha256": PYANNOTE_SEGMENTATION_3_SHA256,
    },
    "brouhaha": {
        "repo": "pyannote/brouhaha",
        "revision": PYANNOTE_BROUHAHA_REVISION,
        # The gated Hub artifact could not be inspected without access. Its
        # digest is therefore intentionally not guessed. Strict namespace and
        # shape validation still runs during conversion.
        "sha256": None,
    },
}


def _local_artifacts(
    source: Path,
    *,
    variant: str,
    subfolder: str | None,
) -> PyanNetArtifacts:
    if source.is_file() and subfolder:
        raise ValueError("`subfolder` cannot be used when the model source is a file.")
    root = source if source.is_dir() else source.parent
    if subfolder:
        root = root / subfolder
    checkpoint = (source if source.is_file() else root / "model.safetensors")
    if checkpoint.suffix != ".safetensors" or not checkpoint.is_file():
        if source.is_file():
            raise ValueError(
                "Native PyanNet runtime accepts Safetensors only. Use "
                "`convert_pyannote_lightning_checkpoint` explicitly for "
                "a reviewed Lightning checkpoint.")
        raise FileNotFoundError(f"Native PyanNet requires model.safetensors in {root}.")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Native PyanNet requires config.json in {root}.")
    values = read_json_file(config_path)
    actual = config_for_variant(values.get("variant", variant)).variant
    if actual != variant:
        raise ValueError(
            f"PyanNet artifact variant {actual!r} does not match provider "
            f"variant {variant!r}.")
    return PyanNetArtifacts(
        source=str(source),
        revision=None,
        config=config_path,
        checkpoint=checkpoint,
    )


def resolve_pyannet_artifacts(
    source: str | Path,
    *,
    variant: str,
    cache_dir: str | Path | None,
    revision: str | None,
    subfolder: str | None,
    token: str | bool | None,
    local_files_only: bool,
    trust_pickle_checkpoint: bool,
) -> PyanNetArtifacts:
    """Resolve a safe native directory or explicitly convert an official
    file."""
    canonical_variant = config_for_variant(variant).variant
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _local_artifacts(
            source_path.resolve(),
            variant=canonical_variant,
            subfolder=subfolder,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"PyanNet model path was not found: {source_path}.")

    official = _OFFICIAL_ARTIFACTS[canonical_variant]
    configured_source = str(source)
    official_aliases = {
        "segmentation": {
            "pyannote/voice-activity-detection",
            "pyannote/segmentation",
        },
        "powerset-segmentation": {"pyannote/segmentation-3.0"},
        "brouhaha": {"pyannote/brouhaha"},
    }[canonical_variant]
    if configured_source not in official_aliases:
        config_path = resolve_pretrained_file(
            configured_source,
            "config.json",
            subfolder=subfolder or "",
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        checkpoint = resolve_pretrained_file(
            configured_source,
            "model.safetensors",
            subfolder=subfolder or "",
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        return PyanNetArtifacts(
            source=configured_source,
            revision=revision,
            config=config_path,
            checkpoint=checkpoint,
        )

    if trust_pickle_checkpoint is not True:
        raise ValueError(
            f"{official['repo']} publishes a Lightning pickle checkpoint, "
            "not Safetensors. Review and accept its model terms, then either "
            "run `convert_pyannote_lightning_checkpoint(...)` once or pass "
            "`trust_pickle_checkpoint=True` to this model constructor for "
            "the explicit restricted conversion boundary.")
    resolved_revision = revision or str(official["revision"])
    checkpoint = resolve_pretrained_file(
        str(official["repo"]),
        "pytorch_model.bin",
        subfolder=subfolder or "",
        cache_dir=cache_dir,
        revision=resolved_revision,
        token=token,
        local_files_only=local_files_only,
    )
    # Hub snapshots are already scoped by their immutable resolved revision.
    # Keeping conversion output next to the downloaded checkpoint prevents a
    # caller-provided revision from reusing an artifact converted from another
    # snapshot.
    destination = checkpoint.parent / ".voicehub-native" / canonical_variant
    safe_checkpoint = destination / "model.safetensors"
    config_path = destination / "config.json"
    if not safe_checkpoint.is_file() or not config_path.is_file():
        convert_pyannote_lightning_checkpoint(
            checkpoint,
            destination,
            variant=canonical_variant,
            trust_pickle_checkpoint=True,
            expected_sha256=official["sha256"],
        )
    return PyanNetArtifacts(
        source=str(official["repo"]),
        revision=resolved_revision,
        config=config_path,
        checkpoint=safe_checkpoint,
        converted_from_pickle=True,
    )


__all__ = ["PyanNetArtifacts", "resolve_pyannet_artifacts"]
