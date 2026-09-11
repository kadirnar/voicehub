"""Strict Safetensors loading and portable export for native Zonos."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.hub import write_json_file
from voicehub.models.zonos.native.metadata import NATIVE_ZONOS_FORMAT
from voicehub.models.zonos.native.modeling import ZonosForCausalLM

_FLOATING_SAFE_DTYPES = frozenset({
    "F8_E4M3",
    "F8_E5M2",
    "F16",
    "BF16",
    "F32",
    "F64",
})


@dataclass(frozen=True, slots=True)
class ZonosCheckpointReport:
    path: Path
    tensor_count: int
    parameter_count: int
    header_fingerprint: str


def zonos_inventory_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    rows = [
        f"{name}|{dtype}|{'x'.join(str(item) for item in shape)}"
        for name, (dtype, shape) in sorted(inventory.items())
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def inspect_zonos_checkpoint(path: str | Path, ) -> ZonosCheckpointReport:
    source = Path(path).expanduser().resolve()
    with SafeTensorReader(source) as reader:
        inventory = {
            name: (
                reader.record(name).dtype,
                reader.tensor_shape(name),
            )
            for name in reader.keys()
        }
        parameter_count = sum(reader.record(name).number_of_elements for name in reader.keys())
    return ZonosCheckpointReport(
        path=source,
        tensor_count=len(inventory),
        parameter_count=parameter_count,
        header_fingerprint=zonos_inventory_fingerprint(inventory),
    )


def _validate_layout(
    model: nn.Module,
    reader: SafeTensorReader,
) -> tuple[str, ...]:
    expected_values = model.state_dict(keep_vars=True)
    expected_shapes = {name: tuple(value.shape) for name, value in expected_values.items()}
    expected_names = set(expected_shapes)
    actual_names = set(reader.keys())
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    shape_mismatches = sorted((
        name,
        reader.tensor_shape(name),
        expected_shapes[name],
    ) for name in expected_names & actual_names if reader.tensor_shape(name) != expected_shapes[name])
    dtype_mismatches = sorted((
        name,
        reader.record(name).dtype,
        str(expected_values[name].dtype),
    ) for name in expected_names & actual_names if (
        expected_values[name].is_floating_point() and reader.record(name).dtype not in _FLOATING_SAFE_DTYPES))
    if missing or unexpected or shape_mismatches or dtype_mismatches:
        raise CheckpointCompatibilityError(
            "Zonos checkpoint does not match the native dense Transformer: "
            f"missing={missing[:12]!r}, "
            f"unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={shape_mismatches[:12]!r}, "
            f"dtype_mismatches={dtype_mismatches[:12]!r}.")
    return tuple(sorted(expected_names))


def load_zonos_checkpoint(
    model: ZonosForCausalLM,
    path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> ZonosCheckpointReport:
    """Validate the complete header before assigning any model tensor."""
    if not isinstance(model, ZonosForCausalLM):
        raise TypeError("`model` must be a native ZonosForCausalLM.")
    report = inspect_zonos_checkpoint(path)
    with SafeTensorReader(report.path) as reader:
        names = _validate_layout(model, reader)
        with torch.no_grad():
            for name in names:
                value = reader.get_tensor(name)
                if dtype is not None and value.is_floating_point():
                    value = value.to(dtype=dtype)
                model.load_state_dict(
                    {name: value.to(device=device)},
                    strict=False,
                    assign=True,
                )
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "Zonos streaming load left meta tensors: " + ", ".join(remaining[:12]))
    return report


def export_zonos_checkpoint(
    model: ZonosForCausalLM,
    path: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    """Export a complete checkpoint reloadable by fresh native inference."""
    if not isinstance(model, ZonosForCausalLM):
        raise TypeError("`model` must be a native ZonosForCausalLM.")
    state = (model.state_dict() if state_override is None else dict(state_override))
    expected = set(model.state_dict())
    actual = set(state)
    if actual != expected:
        raise ValueError(
            "Zonos export state is incomplete: "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}.")
    if any(value.device.type == "meta" for value in state.values()):
        raise ValueError("Zonos cannot export unmaterialized meta tensors.")
    return save_safetensors(
        state,
        path,
        metadata={
            "format": NATIVE_ZONOS_FORMAT,
            "architecture": "zonos",
            "backbone": "dense-transformer",
            "training_objective": "delayed-codebook-causal-cross-entropy",
        },
    )


def save_zonos_pretrained(
    model: ZonosForCausalLM,
    directory: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    export_zonos_checkpoint(
        model,
        destination / "model.safetensors",
        state_override=state_override,
    )
    write_json_file(
        destination / "config.json",
        model.config.to_dict(),
    )
    return destination.resolve()


__all__ = [
    "ZonosCheckpointReport",
    "export_zonos_checkpoint",
    "inspect_zonos_checkpoint",
    "load_zonos_checkpoint",
    "save_zonos_pretrained",
    "zonos_inventory_fingerprint",
]
