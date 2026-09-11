"""Strict Safetensors loading and portable export for native CSM."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.csm.native.metadata import (
    CSM_CHECKPOINT_HEADER_FINGERPRINT,
    CSM_CHECKPOINT_PARAMETER_COUNT,
    CSM_CHECKPOINT_TENSOR_COUNT,
    NATIVE_CSM_FORMAT,
)
from voicehub.models.csm.native.modeling import CSMModel

_FLOATING_DTYPES = frozenset({
    "BF16",
    "F16",
    "F32",
    "F64",
    "F8_E4M3",
    "F8_E5M2",
})


def tensor_inventory_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    """Hash a complete safe-header inventory deterministically."""
    rows = [
        f"{name}|{dtype}|{'x'.join(str(value) for value in shape)}"
        for name, (dtype, shape) in sorted(inventory.items())
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CSMCheckpointReport:
    path: Path
    tensor_count: int
    parameter_count: int
    header_fingerprint: str


def inspect_csm_checkpoint(path: str | Path) -> CSMCheckpointReport:
    """Inspect CSM metadata without materializing its six-gigabyte payload."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".safetensors":
        raise ValueError("Native CSM checkpoints must use Safetensors.")
    with SafeTensorReader(source) as reader:
        inventory = {
            name: (
                reader.record(name).dtype,
                reader.tensor_shape(name),
            )
            for name in reader.keys()
        }
        parameter_count = sum(reader.record(name).number_of_elements for name in reader.keys())
    return CSMCheckpointReport(
        path=source,
        tensor_count=len(inventory),
        parameter_count=parameter_count,
        header_fingerprint=tensor_inventory_fingerprint(inventory),
    )


def _validate_layout(
    model: nn.Module,
    reader: SafeTensorReader,
) -> tuple[str, ...]:
    targets = model.state_dict(keep_vars=True)
    expected_shapes = {name: tuple(value.shape) for name, value in targets.items()}
    expected_names = set(expected_shapes)
    actual_names = set(reader.keys())
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    mismatched = sorted((
        name,
        reader.tensor_shape(name),
        expected_shapes[name],
    ) for name in expected_names & actual_names if reader.tensor_shape(name) != expected_shapes[name])
    dtype_mismatches = sorted(
        (
            name,
            reader.record(name).dtype,
        ) for name in expected_names & actual_names
        if targets[name].is_floating_point() and reader.record(name).dtype not in _FLOATING_DTYPES)
    if missing or unexpected or mismatched or dtype_mismatches:
        raise CheckpointCompatibilityError(
            "CSM checkpoint does not match the native source graph: "
            f"missing={missing[:12]!r}, unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={mismatched[:12]!r}, "
            f"dtype_mismatches={dtype_mismatches[:12]!r}.")
    return tuple(sorted(expected_names))


def validate_csm_checkpoint(
    model: CSMModel,
    path: str | Path,
    *,
    require_official_inventory: bool = False,
) -> CSMCheckpointReport:
    """Validate all names, shapes, and dtypes before assigning one tensor."""
    report = inspect_csm_checkpoint(path)
    with SafeTensorReader(report.path) as reader:
        _validate_layout(model, reader)
    if require_official_inventory and (report.tensor_count != CSM_CHECKPOINT_TENSOR_COUNT or
                                       report.parameter_count != CSM_CHECKPOINT_PARAMETER_COUNT or
                                       report.header_fingerprint != CSM_CHECKPOINT_HEADER_FINGERPRINT):
        raise CheckpointCompatibilityError(
            "The checkpoint matches a CSM-shaped graph but not the audited "
            "sesame/csm-1b tensor inventory.")
    return report


def load_csm_checkpoint(
    model: CSMModel,
    path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    require_official_inventory: bool = False,
) -> CSMCheckpointReport:
    """Stream an already validated checkpoint into a native graph."""
    if not isinstance(model, CSMModel):
        raise TypeError("CSM checkpoint targets must be `CSMModel` instances.")
    report = validate_csm_checkpoint(
        model,
        path,
        require_official_inventory=require_official_inventory,
    )
    target_device = torch.device(device)
    with SafeTensorReader(report.path) as reader:
        names = _validate_layout(model, reader)
        with torch.no_grad():
            for name in names:
                value = reader.get_tensor(name)
                if dtype is not None and value.is_floating_point():
                    value = value.to(dtype=dtype)
                value = value.to(device=target_device)
                model.load_state_dict(
                    {name: value},
                    strict=False,
                    assign=True,
                )
    model.materialize_runtime_buffers(target_device)
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "CSM streaming load left meta tensors: " + ", ".join(remaining[:12]))
    return report


def export_csm_checkpoint(
    model: CSMModel,
    path: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    """Write a complete pickle-free CSM checkpoint for fresh reloads."""
    if not isinstance(model, CSMModel):
        raise TypeError("CSM export requires a native `CSMModel`.")
    state = (model.state_dict() if state_override is None else dict(state_override))
    expected = set(model.state_dict())
    actual = set(state)
    if actual != expected:
        raise ValueError(
            "CSM export state is incomplete: "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}.")
    tensors = {
        # The native writer transfers tensors one at a time.  Keeping this
        # mapping on its current device avoids retaining a second complete
        # 6.2 GB CPU state dict while exporting a CUDA training runtime.
        name: value.detach()
        for name, value in state.items()
    }
    return save_safetensors(
        tensors,
        path,
        metadata={
            "format": NATIVE_CSM_FORMAT,
            "architecture": "csm",
            "training_objective": "source-two-level-codebook-ce",
        },
    ).resolve()


__all__ = [
    "CSMCheckpointReport",
    "export_csm_checkpoint",
    "inspect_csm_checkpoint",
    "load_csm_checkpoint",
    "tensor_inventory_fingerprint",
    "validate_csm_checkpoint",
]
