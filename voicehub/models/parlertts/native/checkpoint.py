"""Strict loading and safe export for native Parler-TTS checkpoints."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.models.parlertts.native.metadata import (
    PARLER_TTS_HEADER_FINGERPRINT,
    PARLER_TTS_PARAMETER_COUNT,
    PARLER_TTS_TENSOR_COUNT,
)

NATIVE_PARLER_TTS_FORMAT = "voicehub-parlertts-v1"


def tensor_inventory_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    """Hash sorted tensor names, dtypes, and shapes from a safe header."""
    if not isinstance(inventory, Mapping):
        raise TypeError("`inventory` must be a mapping.")
    rows = []
    for name, record in sorted(inventory.items()):
        if (not isinstance(name, str) or not name or not isinstance(record, tuple) or len(record) != 2):
            raise ValueError("Inventory entries must be name/(dtype, shape) pairs.")
        dtype, shape = record
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"Tensor {name!r} has an invalid dtype.")
        if (not isinstance(shape, tuple) or
                any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape)):
            raise ValueError(f"Tensor {name!r} has an invalid shape.")
        rows.append(f"{name}|{dtype}|{'x'.join(str(item) for item in shape)}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ParlerCheckpointReport:
    path: Path
    tensor_count: int
    parameter_count: int
    header_fingerprint: str


def inspect_parlertts_checkpoint(path: str | Path, ) -> ParlerCheckpointReport:
    """Inspect the namespace without materializing its 3.5 GB payload."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".safetensors":
        raise ValueError("Native Parler-TTS accepts Safetensors checkpoints.")
    with SafeTensorReader(source) as reader:
        inventory = {name: (reader.record(name).dtype, reader.tensor_shape(name)) for name in reader.keys()}
        parameter_count = sum(reader.record(name).number_of_elements for name in reader.keys())
    return ParlerCheckpointReport(
        path=source,
        tensor_count=len(inventory),
        parameter_count=parameter_count,
        header_fingerprint=tensor_inventory_fingerprint(inventory),
    )


def _expected_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    if not isinstance(model, nn.Module):
        raise TypeError("Parler-TTS checkpoint target must be an nn.Module.")
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict(keep_vars=True).items()}


def validate_parlertts_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    require_official_inventory: bool = False,
) -> ParlerCheckpointReport:
    """Reject missing, extra, or shape-incompatible checkpoint tensors."""
    expected = _expected_shapes(model)
    report = inspect_parlertts_checkpoint(path)
    with SafeTensorReader(report.path) as reader:
        missing = tuple(name for name in expected if name not in reader)
        unexpected = tuple(name for name in reader.keys() if name not in expected)
        mismatched = tuple((
            name,
            reader.tensor_shape(name),
            shape,
        ) for name, shape in expected.items() if name in reader and reader.tensor_shape(name) != shape)
    if missing or unexpected or mismatched:
        raise ValueError(
            "Parler-TTS checkpoint is incompatible: "
            f"missing={list(missing)!r}, unexpected={list(unexpected)!r}, "
            f"shape_mismatches={list(mismatched)!r}.")
    if require_official_inventory and (report.tensor_count != PARLER_TTS_TENSOR_COUNT or
                                       report.parameter_count != PARLER_TTS_PARAMETER_COUNT or
                                       report.header_fingerprint != PARLER_TTS_HEADER_FINGERPRINT):
        raise ValueError(
            "Parler-TTS checkpoint matches the configured graph but is not "
            "the audited Mini v1 artifact inventory.")
    return report


def load_parlertts_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    require_official_inventory: bool = False,
) -> ParlerCheckpointReport:
    """Load one tensor at a time after complete header validation."""
    report = validate_parlertts_checkpoint(
        model,
        path,
        require_official_inventory=require_official_inventory,
    )
    targets = model.state_dict(keep_vars=True)
    with torch.no_grad(), SafeTensorReader(report.path) as reader:
        for name, target in targets.items():
            target.copy_(reader.get_tensor(
                name,
                device=target.device,
                dtype=target.dtype,
            ))
    return report


def export_parlertts_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    """Export a portable, pickle-free checkpoint for fresh native reloads."""
    if not isinstance(model, nn.Module):
        raise TypeError("Parler-TTS export requires an nn.Module.")
    state = model.state_dict() if state_override is None else state_override
    expected = tuple(model.state_dict())
    if tuple(state) != expected:
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(
            "Parler-TTS export state is incomplete: "
            f"missing={missing!r}, unexpected={unexpected!r}.")
    tensors = {name: tensor.detach().to(device="cpu").contiguous() for name, tensor in state.items()}
    return save_safetensors(
        tensors,
        path,
        metadata={
            "format": NATIVE_PARLER_TTS_FORMAT
        },
    ).resolve()


__all__ = [
    "NATIVE_PARLER_TTS_FORMAT",
    "ParlerCheckpointReport",
    "export_parlertts_checkpoint",
    "inspect_parlertts_checkpoint",
    "load_parlertts_checkpoint",
    "tensor_inventory_fingerprint",
    "validate_parlertts_checkpoint",
]
