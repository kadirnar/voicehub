"""Strict checkpoint loading and safe export for Inflect v2."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.hub import read_json_file, resolve_pretrained_file, write_json_file
from voicehub.models.inflecttts.native.configuration import InflectV2Config
from voicehub.models.inflecttts.native.metadata import (
    INFLECT_LICENSE,
    INFLECT_MICRO_V2_CHECKPOINT_SHA256,
    INFLECT_MICRO_V2_CONFIG_SHA256,
    INFLECT_MICRO_V2_INVENTORY_FINGERPRINT,
    INFLECT_MICRO_V2_REPOSITORY,
    INFLECT_MICRO_V2_REVISION,
    INFLECT_NANO_V2_CHECKPOINT_SHA256,
    INFLECT_NANO_V2_CONFIG_SHA256,
    INFLECT_NANO_V2_INVENTORY_FINGERPRINT,
    INFLECT_NANO_V2_REPOSITORY,
    INFLECT_NANO_V2_REVISION,
    NATIVE_FORMAT,
    NATIVE_FORMAT_VERSION,
)

_OFFICIAL_RELEASES = {
    INFLECT_MICRO_V2_REPOSITORY: (
        INFLECT_MICRO_V2_REVISION,
        INFLECT_MICRO_V2_CHECKPOINT_SHA256,
        INFLECT_MICRO_V2_CONFIG_SHA256,
        INFLECT_MICRO_V2_INVENTORY_FINGERPRINT,
        9_356_513,
    ),
    INFLECT_NANO_V2_REPOSITORY: (
        INFLECT_NANO_V2_REVISION,
        INFLECT_NANO_V2_CHECKPOINT_SHA256,
        INFLECT_NANO_V2_CONFIG_SHA256,
        INFLECT_NANO_V2_INVENTORY_FINGERPRINT,
        3_966_721,
    ),
}


@dataclass(frozen=True, slots=True)
class InflectArtifacts:
    """One config and checkpoint resolved from a coherent artifact root."""

    source: str | Path
    revision: str | None
    config_path: Path
    config: InflectV2Config
    checkpoint_path: Path
    legacy_pytorch: bool
    expected_checkpoint_sha256: str | None
    expected_inventory_fingerprint: str | None
    discriminator_path: Path | None = None


@dataclass(frozen=True, slots=True)
class InflectCheckpointReport:
    """Validated checkpoint inventory details."""

    tensor_count: int
    parameter_count: int
    inventory_fingerprint: str
    missing_training_tensors: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_inventory_fingerprint(tensors: Mapping[str, Any]) -> str:
    """Hash names, PyTorch dtypes, and shapes without tensor values."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(name, str) or not name:
            raise ValueError("Checkpoint tensor names must be non-empty strings.")
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Checkpoint value {name!r} is not a tensor.")
        dtype = str(tensor.dtype).removeprefix("torch.")
        shape = ",".join(str(item) for item in tensor.shape)
        digest.update(f"{name}|{dtype}|{shape}\n".encode())
    return digest.hexdigest()


def _safe_checkpoint_filename(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("`checkpoint_filename` must be non-empty or None.")
    filename = value.strip()
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("`checkpoint_filename` must be one checkpoint-root filename.")
    if not filename.endswith((".safetensors", ".pth")):
        raise ValueError("Inflect checkpoints must use .safetensors or the released .pth "
                         "format.")
    return filename


def resolve_inflect_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> InflectArtifacts:
    """Resolve config and weights without a provider runtime or Hub SDK."""
    checkpoint_filename = _safe_checkpoint_filename(checkpoint_filename)
    source_path = Path(source).expanduser()
    source_name = str(source)
    if source_path.is_file():
        raise NotADirectoryError(
            "Inflect requires its matching config and checkpoint directory; "
            "pass the artifact root rather than a single file.")
    release = _OFFICIAL_RELEASES.get(source_name)
    resolved_revision = revision
    if release is not None:
        official_revision = release[0]
        if revision is None:
            resolved_revision = official_revision
        elif revision != official_revision:
            raise ValueError(
                f"{source_name} is audited only at immutable revision "
                f"{official_revision}; found {revision}.")
    hub_kwargs = {
        "cache_dir": cache_dir,
        "revision": resolved_revision,
        "token": token,
        "local_files_only": local_files_only,
    }
    config_path = resolve_pretrained_file(source, "config.json", **hub_kwargs)
    if release is not None:
        actual_config_sha256 = _file_sha256(config_path)
        if actual_config_sha256 != release[2]:
            raise ValueError("Official Inflect config SHA-256 mismatch: "
                             f"{actual_config_sha256}.")
    config = InflectV2Config.from_dict(read_json_file(config_path))

    if source_path.is_dir():
        if checkpoint_filename is not None:
            checkpoint = resolve_pretrained_file(
                source_path,
                checkpoint_filename,
            )
        else:
            safe_candidates = [
                source_path / name for name in ("model.safetensors", "pytorch_model.safetensors")
                if (source_path / name).is_file()
            ]
            if len(safe_candidates) > 1:
                raise ValueError(
                    "Inflect artifact root contains multiple Safetensors "
                    "checkpoints; set `checkpoint_filename` explicitly.")
            if safe_candidates:
                checkpoint = safe_candidates[0]
            else:
                checkpoint = resolve_pretrained_file(source_path, "model.pth")
    else:
        checkpoint = resolve_pretrained_file(
            source,
            checkpoint_filename or "model.pth",
            **hub_kwargs,
        )

    return InflectArtifacts(
        source=source,
        revision=resolved_revision,
        config_path=config_path,
        config=config,
        checkpoint_path=checkpoint.resolve(),
        legacy_pytorch=checkpoint.suffix.lower() == ".pth",
        expected_checkpoint_sha256=release[1] if release is not None else None,
        expected_inventory_fingerprint=release[3] if release is not None else None,
        discriminator_path=((source_path / "discriminator.safetensors").resolve() if source_path.is_dir() and
                            (source_path / "discriminator.safetensors").is_file() else None),
    )


def _read_legacy_checkpoint(
    artifacts: InflectArtifacts,
    *,
    trust_pickle_checkpoint: bool,
) -> dict[str, Tensor]:
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "The official Inflect release uses a PyTorch pickle container. "
            "Review its origin and pass `trust_pickle_checkpoint=True` for "
            "one restricted, weights-only load, then export Safetensors for "
            "steady-state use.")
    if artifacts.expected_checkpoint_sha256 is not None:
        actual_sha256 = _file_sha256(artifacts.checkpoint_path)
        if actual_sha256 != artifacts.expected_checkpoint_sha256:
            raise ValueError("Official Inflect checkpoint SHA-256 mismatch: "
                             f"{actual_sha256}.")
    try:
        payload = torch.load(
            artifacts.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:  # pragma: no cover - old unsupported PyTorch
        raise RuntimeError(
            "Inflect legacy conversion requires PyTorch with "
            "`torch.load(..., weights_only=True)` support.") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Inflect legacy checkpoint must contain a mapping.")
    if payload.get("format") not in {
            "inflect_vits_inference_checkpoint_v1",
            "inflect_v2_inference_checkpoint_v1",
    }:
        raise ValueError("Unexpected Inflect legacy checkpoint format.")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Inflect legacy checkpoint is missing `model` tensors.")
    if any(not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()):
        raise TypeError("Inflect legacy `model` must map names to tensors.")
    declared_parameters = payload.get("deployable_parameters")
    actual_parameters = sum(value.numel() for value in state.values())
    if declared_parameters != actual_parameters:
        raise ValueError(
            "Inflect legacy checkpoint parameter count does not match its "
            "declared metadata.")
    return dict(state)


def read_inflect_checkpoint(
    artifacts: InflectArtifacts,
    *,
    trust_pickle_checkpoint: bool = False,
) -> dict[str, Tensor]:
    """Read a native Safetensors file or explicitly trusted release pickle."""
    if artifacts.legacy_pytorch:
        return _read_legacy_checkpoint(
            artifacts,
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
    with SafeTensorReader(artifacts.checkpoint_path) as reader:
        return reader.state_dict()


def load_inflect_checkpoint(
    model: nn.Module,
    artifacts: InflectArtifacts,
    *,
    trust_pickle_checkpoint: bool = False,
    allow_fresh_training_components: bool = False,
) -> InflectCheckpointReport:
    """Validate every tensor before loading the exact native graph."""
    if not isinstance(model, nn.Module):
        raise TypeError("`model` must be a PyTorch module.")
    state = read_inflect_checkpoint(
        artifacts,
        trust_pickle_checkpoint=trust_pickle_checkpoint,
    )
    expected = model.state_dict()
    allowed_missing = (
        frozenset(name for name in expected
                  if name.startswith("enc_q.")) if allow_fresh_training_components else frozenset())
    missing = tuple(sorted(set(expected) - set(state)))
    unexpected = tuple(sorted(set(state) - set(expected)))
    invalid_missing = tuple(name for name in missing if name not in allowed_missing)
    shape_mismatch_items = []
    for name in sorted(set(state) & set(expected)):
        actual_shape = tuple(state[name].shape)
        expected_shape = tuple(expected[name].shape)
        if actual_shape != expected_shape:
            shape_mismatch_items.append((name, actual_shape, expected_shape))
    shape_mismatches = tuple(shape_mismatch_items)
    if invalid_missing or unexpected or shape_mismatches:
        details = []
        if invalid_missing:
            details.append("missing=" + ", ".join(invalid_missing[:12]))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected[:12]))
        if shape_mismatches:
            details.append(
                "shape mismatches=" +
                ", ".join(f"{name}:{actual!r}!={wanted!r}" for name, actual, wanted in shape_mismatches[:12]))
        raise ValueError("Inflect checkpoint inventory is incompatible: " + "; ".join(details))
    fingerprint = tensor_inventory_fingerprint(state)
    if (artifacts.expected_inventory_fingerprint is not None and
            fingerprint != artifacts.expected_inventory_fingerprint):
        raise ValueError("Official Inflect tensor inventory fingerprint mismatch: "
                         f"{fingerprint}.")
    incompatible = model.load_state_dict(state, strict=False)
    if (tuple(sorted(incompatible.missing_keys)) != missing or incompatible.unexpected_keys):
        raise RuntimeError("Inflect checkpoint changed during validated loading.")
    return InflectCheckpointReport(
        tensor_count=len(state),
        parameter_count=sum(value.numel() for value in state.values()),
        inventory_fingerprint=fingerprint,
        missing_training_tensors=missing,
    )


def load_inflect_discriminator(
    discriminator: nn.Module,
    path: str | Path,
) -> None:
    """Strictly restore the separately optimized discriminator state."""
    if not isinstance(discriminator, nn.Module):
        raise TypeError("`discriminator` must be a PyTorch module.")
    with SafeTensorReader(path) as reader:
        state = reader.state_dict()
    expected = discriminator.state_dict()
    missing = tuple(sorted(set(expected) - set(state)))
    unexpected = tuple(sorted(set(state) - set(expected)))
    mismatches = tuple(
        name for name in sorted(set(state) & set(expected))
        if tuple(state[name].shape) != tuple(expected[name].shape))
    if missing or unexpected or mismatches:
        raise ValueError(
            "Inflect discriminator checkpoint inventory is incompatible "
            f"(missing={missing[:8]!r}, unexpected={unexpected[:8]!r}, "
            f"shape_mismatches={mismatches[:8]!r}).")
    discriminator.load_state_dict(state, strict=True)


def export_inflect_checkpoint(
    model: nn.Module,
    config: InflectV2Config,
    directory: str | Path,
    *,
    discriminator: nn.Module | None = None,
) -> Path:
    """Export a safe artifact reloadable by fresh inference or training."""
    if not isinstance(model, nn.Module):
        raise TypeError("`model` must be a PyTorch module.")
    if not isinstance(config, InflectV2Config):
        raise TypeError("`config` must be an InflectV2Config.")
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_safetensors(
        state,
        destination / "model.safetensors",
        metadata={
            "format": NATIVE_FORMAT,
            "format_version": str(NATIVE_FORMAT_VERSION),
            "architecture": "inflect-v2",
            "license": INFLECT_LICENSE,
            "source_micro_revision": INFLECT_MICRO_V2_REVISION,
            "source_nano_revision": INFLECT_NANO_V2_REVISION,
            "tensor_inventory_fingerprint": tensor_inventory_fingerprint(state),
        },
    )
    write_json_file(destination / "config.json", config.to_dict())
    if discriminator is not None:
        discriminator_state = {
            name: value.detach().cpu().contiguous()
            for name, value in discriminator.state_dict().items()
        }
        save_safetensors(
            discriminator_state,
            destination / "discriminator.safetensors",
            metadata={
                "format": NATIVE_FORMAT,
                "format_version": str(NATIVE_FORMAT_VERSION),
                "component": "fresh-multi-period-discriminator",
                "license": INFLECT_LICENSE,
            },
        )
    return destination


def convert_inflect_legacy_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    trust_pickle_checkpoint: bool = False,
) -> Path:
    """Convert one reviewed release pickle into native Safetensors."""
    artifacts = resolve_inflect_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if not artifacts.legacy_pytorch:
        raise ValueError("Inflect conversion expects the released model.pth.")
    state = read_inflect_checkpoint(
        artifacts,
        trust_pickle_checkpoint=trust_pickle_checkpoint,
    )
    destination_path = Path(destination).expanduser()
    destination_path.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        state,
        destination_path / "model.safetensors",
        metadata={
            "format": NATIVE_FORMAT,
            "format_version": str(NATIVE_FORMAT_VERSION),
            "architecture": "inflect-v2",
            "license": INFLECT_LICENSE,
            "source_repository": str(source),
            "source_revision": artifacts.revision or "",
            "source_checkpoint_sha256": artifacts.expected_checkpoint_sha256 or "",
            "tensor_inventory_fingerprint": tensor_inventory_fingerprint(state),
        },
    )
    write_json_file(
        destination_path / "config.json",
        artifacts.config.to_dict(),
    )
    return destination_path


__all__ = [
    "INFLECT_LICENSE",
    "INFLECT_MICRO_V2_CHECKPOINT_SHA256",
    "INFLECT_MICRO_V2_CONFIG_SHA256",
    "INFLECT_MICRO_V2_INVENTORY_FINGERPRINT",
    "INFLECT_MICRO_V2_REPOSITORY",
    "INFLECT_MICRO_V2_REVISION",
    "INFLECT_NANO_V2_CHECKPOINT_SHA256",
    "INFLECT_NANO_V2_CONFIG_SHA256",
    "INFLECT_NANO_V2_INVENTORY_FINGERPRINT",
    "INFLECT_NANO_V2_REPOSITORY",
    "INFLECT_NANO_V2_REVISION",
    "InflectArtifacts",
    "InflectCheckpointReport",
    "convert_inflect_legacy_checkpoint",
    "export_inflect_checkpoint",
    "load_inflect_checkpoint",
    "load_inflect_discriminator",
    "read_inflect_checkpoint",
    "resolve_inflect_artifacts",
    "tensor_inventory_fingerprint",
]
