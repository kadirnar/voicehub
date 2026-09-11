"""Strict Safetensors loading, export, and opt-in legacy conversion."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError, CheckpointIntegrityError
from voicehub.hub import write_json_file
from voicehub.models.zonos2.native.configuration import Zonos2ArchitectureConfig
from voicehub.models.zonos2.native.metadata import NATIVE_ZONOS2_FORMAT
from voicehub.models.zonos2.native.modeling import Zonos2ForCausalLM

_FLOATING_SAFE_DTYPES = {
    "F8_E4M3",
    "F8_E5M2",
    "F16",
    "BF16",
    "F32",
    "F64",
}


@dataclass(frozen=True, slots=True)
class Zonos2CheckpointReport:
    path: Path
    tensor_count: int
    parameter_count: int
    header_fingerprint: str


def _inventory_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    rows = [
        f"{name}|{dtype}|{'x'.join(str(item) for item in shape)}"
        for name, (dtype, shape) in sorted(inventory.items())
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def inspect_zonos2_checkpoint(path: str | Path) -> Zonos2CheckpointReport:
    source = Path(path).expanduser().resolve()
    with SafeTensorReader(source) as reader:
        inventory = {name: (reader.record(name).dtype, reader.tensor_shape(name)) for name in reader.keys()}
        parameter_count = sum(reader.record(name).number_of_elements for name in reader.keys())
    return Zonos2CheckpointReport(
        path=source,
        tensor_count=len(inventory),
        parameter_count=parameter_count,
        header_fingerprint=_inventory_fingerprint(inventory),
    )


def verify_file_integrity(
    path: str | Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    """Verify a pinned artifact when the caller requests full integrity."""
    source = Path(path).expanduser().resolve()
    if expected_size is not None and source.stat().st_size != expected_size:
        raise CheckpointIntegrityError(
            f"{source.name} has size {source.stat().st_size}; expected "
            f"{expected_size}.")
    if expected_sha256 is not None:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while block := stream.read(chunk_size):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise CheckpointIntegrityError(
                f"{source.name} has SHA-256 {actual}; expected "
                f"{expected_sha256}.")


def _expected_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(value.shape) for name, value in model.state_dict(keep_vars=True).items()}


def _validate_layout(
    model: nn.Module,
    reader: SafeTensorReader,
) -> tuple[str, ...]:
    expected_values = model.state_dict(keep_vars=True)
    expected = {name: tuple(value.shape) for name, value in expected_values.items()}
    actual_names = set(reader.keys())
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    mismatched = sorted((
        name,
        reader.tensor_shape(name),
        expected[name],
    ) for name in expected_names & actual_names if reader.tensor_shape(name) != expected[name])
    dtype_mismatches = []
    for name in expected_names & actual_names:
        source_dtype = reader.record(name).dtype
        if (expected_values[name].is_floating_point() and source_dtype not in _FLOATING_SAFE_DTYPES):
            dtype_mismatches.append((
                name,
                source_dtype,
                str(expected_values[name].dtype),
            ))
    dtype_mismatches.sort()
    if missing or unexpected or mismatched or dtype_mismatches:
        raise CheckpointCompatibilityError(
            "ZONOS2 checkpoint does not match the native architecture: "
            f"missing={missing[:12]!r}, unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={mismatched[:12]!r}, "
            f"dtype_mismatches={dtype_mismatches[:12]!r}.")
    return tuple(sorted(expected))


def load_zonos2_checkpoint(
    model: Zonos2ForCausalLM,
    path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> Zonos2CheckpointReport:
    """Validate the complete header before assigning any model tensor."""
    report = inspect_zonos2_checkpoint(path)
    with SafeTensorReader(report.path) as reader:
        names = _validate_layout(model, reader)
        with torch.no_grad():
            for name in names:
                value = reader.get_tensor(name)
                target_dtype = (dtype if dtype is not None and value.is_floating_point() else value.dtype)
                model.load_state_dict(
                    {name: value.to(
                        device=device,
                        dtype=target_dtype,
                    )},
                    strict=False,
                    assign=True,
                )
    model.materialize_runtime_buffers(device)
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "ZONOS2 streaming load left meta tensors: " + ", ".join(remaining[:12]))
    return report


def export_zonos2_checkpoint(
    model: Zonos2ForCausalLM,
    path: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    """Export a complete checkpoint reloadable by the native runtime."""
    state = model.state_dict() if state_override is None else dict(state_override)
    expected = set(model.state_dict())
    actual = set(state)
    if actual != expected:
        raise ValueError(
            "ZONOS2 export state is incomplete: "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}.")
    return save_safetensors(
        state,
        path,
        metadata={
            "format": NATIVE_ZONOS2_FORMAT,
            "architecture": "zonos2",
            "training_objective": "reconstructed-causal-codebook-ce",
        },
    )


def save_zonos2_pretrained(
    model: Zonos2ForCausalLM,
    directory: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    export_zonos2_checkpoint(
        model,
        destination / "model.safetensors",
        state_override=state_override,
    )
    write_json_file(
        destination / "config.json",
        model.config.to_dict(),
    )
    return destination.resolve()


def _normalize_legacy_state(state: Any, ) -> dict[str, Tensor]:
    if isinstance(state, Mapping) and "model" in state:
        state = state["model"]
    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError("Legacy ZONOS2 checkpoint must contain a tensor mapping.")
    normalized: dict[str, Tensor] = {}
    for raw_name, value in state.items():
        if not isinstance(raw_name, str) or not isinstance(value, Tensor):
            raise CheckpointCompatibilityError("Legacy ZONOS2 state must map string names to tensors only.")
        if (".router.ent_denom" in raw_name or ".router.normalized_entropy" in raw_name):
            continue
        name = raw_name
        if ".parametrizations." in name and ".original" in name:
            name = (name.replace(".parametrizations.", ".").replace(".original", ""))
        if name in normalized:
            raise CheckpointCompatibilityError(f"Legacy normalization produced duplicate tensor {name!r}.")
        normalized[name] = value
    return normalized


def convert_legacy_zonos2_checkpoint(
    legacy_path: str | Path,
    output_path: str | Path,
    *,
    config: Zonos2ArchitectureConfig,
    allow_unsafe_pickle: bool = False,
) -> Path:
    """Convert the official ``.pth`` representation to Safetensors.

    PyTorch pickle cannot provide the strict, streaming trust boundary
    used by VoiceHub. Conversion is therefore never implicit and
    requires the explicit ``allow_unsafe_pickle=True`` acknowledgement.
    ``weights_only=True`` is still enforced to reduce the accepted
    pickle surface.
    """
    if allow_unsafe_pickle is not True:
        raise PermissionError(
            "Legacy ZONOS2 conversion reads a PyTorch pickle. Pass "
            "`allow_unsafe_pickle=True` only for a trusted, integrity-verified "
            "checkpoint.")
    source = Path(legacy_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Legacy ZONOS2 checkpoint not found: {source}.")
    try:
        state = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:  # pragma: no cover - old PyTorch
        raise RuntimeError(
            "Legacy conversion requires a PyTorch release supporting "
            "`torch.load(..., weights_only=True)`.") from error
    normalized = _normalize_legacy_state(state)
    with torch.device("meta"):
        model = Zonos2ForCausalLM(config)
    expected = _expected_shapes(model)
    actual = set(normalized)
    if actual != set(expected):
        raise CheckpointCompatibilityError(
            "Legacy ZONOS2 checkpoint namespace is incompatible: "
            f"missing={sorted(set(expected) - actual)[:12]!r}, "
            f"unexpected={sorted(actual - set(expected))[:12]!r}.")
    mismatched = [(name, tuple(value.shape), expected[name]) for name, value in normalized.items()
                  if tuple(value.shape) != expected[name]]
    if mismatched:
        raise CheckpointCompatibilityError(
            "Legacy ZONOS2 checkpoint has shape mismatches: "
            f"{mismatched[:12]!r}.")
    return save_safetensors(
        normalized,
        output_path,
        metadata={
            "format": NATIVE_ZONOS2_FORMAT,
            "converted_from": "trusted-pytorch-pickle",
        },
    )


__all__ = [
    "Zonos2CheckpointReport",
    "convert_legacy_zonos2_checkpoint",
    "export_zonos2_checkpoint",
    "inspect_zonos2_checkpoint",
    "load_zonos2_checkpoint",
    "save_zonos2_pretrained",
    "verify_file_integrity",
]
