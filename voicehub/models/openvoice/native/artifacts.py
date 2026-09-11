"""Coherent local and Hub artifact resolution for OpenVoice V2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.models.openvoice.native.configuration import OpenVoiceConverterConfig
from voicehub.models.openvoice.native.metadata import (
    OPENVOICE_CHECKPOINT_REVISION,
    OPENVOICE_CONVERTER_CHECKPOINT,
    OPENVOICE_CONVERTER_CONFIG_SHA256,
    OPENVOICE_MODEL_ID,
)
from voicehub.path_utils import is_explicit_local_path


@dataclass(frozen=True, slots=True)
class OpenVoiceArtifacts:
    """One converter config and checkpoint from a coherent artifact root."""

    source: str
    revision: str | None
    config_path: Path
    config: OpenVoiceConverterConfig
    checkpoint_path: Path
    legacy_pytorch: bool
    expected_checkpoint_sha256: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_from_root(root: Path) -> Path:
    candidates = tuple(
        path for path in (
            root / "model.safetensors",
            root / "checkpoint.pth",
        ) if path.is_file())
    if not candidates:
        raise FileNotFoundError(
            "OpenVoice artifact root requires model.safetensors or "
            f"checkpoint.pth: {root}.")
    if len(candidates) != 1:
        raise ValueError("OpenVoice artifact root cannot contain both legacy and native "
                         "checkpoints.")
    return candidates[0]


def _local_root(source: Path) -> Path:
    root = source
    if (source / "converter").is_dir():
        root = source / "converter"
    if not root.is_dir():
        raise NotADirectoryError("OpenVoice requires a converter artifact directory.")
    return root.resolve()


def resolve_openvoice_artifacts(
    source: str | Path = OPENVOICE_MODEL_ID,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> OpenVoiceArtifacts:
    """Resolve the exact official release or a native local export."""
    if (not isinstance(source, (str, Path)) or not str(source).strip()):
        raise ValueError("OpenVoice source must be a non-empty path or Hub ID.")
    source_path = Path(source).expanduser()
    source_name = str(source)
    if source_path.exists():
        root = _local_root(source_path)
        config_path = root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"OpenVoice config.json was not found in {root}.")
        checkpoint_path = _checkpoint_from_root(root)
        return OpenVoiceArtifacts(
            source=str(source_path.resolve()),
            revision=None,
            config_path=config_path,
            config=OpenVoiceConverterConfig.from_dict(read_json_file(config_path)),
            checkpoint_path=checkpoint_path,
            legacy_pytorch=checkpoint_path.suffix == ".pth",
            expected_checkpoint_sha256=None,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"OpenVoice model path was not found: {source_path}.")
    if source_name != OPENVOICE_MODEL_ID:
        raise ValueError(
            "Native OpenVoice Hub loading is restricted to the audited "
            f"{OPENVOICE_MODEL_ID!r} repository. Export custom converters "
            "to a local native artifact first.")
    resolved_revision = revision or OPENVOICE_CHECKPOINT_REVISION
    if resolved_revision != OPENVOICE_CHECKPOINT_REVISION:
        raise ValueError(
            "The official OpenVoice V2 converter is audited only at "
            f"{OPENVOICE_CHECKPOINT_REVISION}; found {resolved_revision}.")
    common = {
        "revision": resolved_revision,
        "cache_dir": cache_dir,
        "token": token,
        "local_files_only": local_files_only,
    }
    config_path = resolve_pretrained_file(
        source_name,
        "config.json",
        subfolder="converter",
        **common,
    ).resolve()
    checkpoint_path = resolve_pretrained_file(
        source_name,
        "checkpoint.pth",
        subfolder="converter",
        **common,
    ).resolve()
    if config_path.parent != checkpoint_path.parent:
        raise RuntimeError("OpenVoice config and checkpoint did not resolve from one "
                           "immutable snapshot.")
    config_sha256 = _sha256(config_path)
    if config_sha256 != OPENVOICE_CONVERTER_CONFIG_SHA256:
        raise RuntimeError("Official OpenVoice config SHA-256 mismatch: "
                           f"{config_sha256}.")
    return OpenVoiceArtifacts(
        source=source_name,
        revision=resolved_revision,
        config_path=config_path,
        config=OpenVoiceConverterConfig.from_dict(read_json_file(config_path)),
        checkpoint_path=checkpoint_path,
        legacy_pytorch=True,
        expected_checkpoint_sha256=(OPENVOICE_CONVERTER_CHECKPOINT["sha256"]),
    )


__all__ = [
    "OpenVoiceArtifacts",
    "resolve_openvoice_artifacts",
]
