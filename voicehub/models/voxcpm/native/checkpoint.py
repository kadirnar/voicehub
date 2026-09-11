"""Strict VoxCPM2 Safetensors loading, conversion, and export."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.voxcpm.native.codec import VoxCPMAudioVAE
from voicehub.models.voxcpm.native.metadata import (
    NATIVE_VOXCPM2_CODEC_FORMAT,
    NATIVE_VOXCPM2_FORMAT,
    VOXCPM2_CHECKPOINT_HEADER_FINGERPRINT,
    VOXCPM2_CHECKPOINT_PARAMETER_COUNT,
    VOXCPM2_CHECKPOINT_TENSOR_COUNT,
    VOXCPM2_CODEC_HEADER_FINGERPRINT,
    VOXCPM2_CODEC_LEGACY_SHA256,
    VOXCPM2_CODEC_LEGACY_SIZE,
    VOXCPM2_CODEC_PARAMETER_COUNT,
    VOXCPM2_CODEC_TENSOR_COUNT,
)
from voicehub.models.voxcpm.native.modeling import VoxCPM2Model

_FLOATING_DTYPES = frozenset({
    "BF16",
    "F16",
    "F32",
    "F64",
    "F8_E4M3",
    "F8_E5M2",
})


def tensor_inventory_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    rows = [
        f"{name}\t{dtype}\t{','.join(str(value) for value in shape)}"
        for name, (dtype, shape) in sorted(inventory.items())
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class VoxCPMCheckpointReport:
    path: Path
    tensor_count: int
    parameter_count: int
    header_fingerprint: str


def inspect_voxcpm_checkpoint(path: str | Path) -> VoxCPMCheckpointReport:
    requested = Path(path).expanduser()
    if requested.suffix.lower() != ".safetensors":
        raise ValueError("Native VoxCPM checkpoints must use Safetensors.")
    source = requested.resolve()
    with SafeTensorReader(source) as reader:
        inventory = {
            name: (
                reader.record(name).dtype,
                reader.tensor_shape(name),
            )
            for name in reader.keys()
        }
        parameters = sum(reader.record(name).number_of_elements for name in reader.keys())
    return VoxCPMCheckpointReport(
        path=source,
        tensor_count=len(inventory),
        parameter_count=parameters,
        header_fingerprint=tensor_inventory_fingerprint(inventory),
    )


def _validate_layout(
    model: nn.Module,
    reader: SafeTensorReader,
) -> tuple[str, ...]:
    targets = model.state_dict(keep_vars=True)
    expected = {name: tuple(value.shape) for name, value in targets.items()}
    expected_names = set(expected)
    actual_names = set(reader.keys())
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    shapes = sorted((
        name,
        reader.tensor_shape(name),
        expected[name],
    ) for name in expected_names & actual_names if reader.tensor_shape(name) != expected[name])
    dtypes = sorted(
        (name, reader.record(name).dtype) for name in expected_names & actual_names
        if (targets[name].is_floating_point() and reader.record(name).dtype not in _FLOATING_DTYPES) or (
            not targets[name].is_floating_point() and
            reader.record(name).dtype not in {"I64", "I32", "I16", "I8", "U8", "BOOL"}))
    if missing or unexpected or shapes or dtypes:
        raise CheckpointCompatibilityError(
            "VoxCPM checkpoint does not match the native graph: "
            f"missing={missing[:12]!r}, unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={shapes[:12]!r}, "
            f"dtype_mismatches={dtypes[:12]!r}.")
    return tuple(sorted(expected_names))


def validate_voxcpm_checkpoint(
    model: VoxCPM2Model | VoxCPMAudioVAE,
    path: str | Path,
    *,
    require_official_inventory: bool = False,
) -> VoxCPMCheckpointReport:
    if not isinstance(model, (VoxCPM2Model, VoxCPMAudioVAE)):
        raise TypeError("VoxCPM checkpoint targets must be a native model or AudioVAE.")
    report = inspect_voxcpm_checkpoint(path)
    with SafeTensorReader(report.path) as reader:
        _validate_layout(model, reader)
    if require_official_inventory:
        if isinstance(model, VoxCPM2Model):
            expected = (
                VOXCPM2_CHECKPOINT_TENSOR_COUNT,
                VOXCPM2_CHECKPOINT_PARAMETER_COUNT,
                VOXCPM2_CHECKPOINT_HEADER_FINGERPRINT,
            )
        else:
            expected = (
                VOXCPM2_CODEC_TENSOR_COUNT,
                VOXCPM2_CODEC_PARAMETER_COUNT,
                VOXCPM2_CODEC_HEADER_FINGERPRINT,
            )
        actual = (
            report.tensor_count,
            report.parameter_count,
            report.header_fingerprint,
        )
        if actual != expected:
            raise CheckpointCompatibilityError(
                "The checkpoint matches a VoxCPM-shaped graph but not the "
                f"audited official inventory: actual={actual!r}, expected={expected!r}.")
    return report


def load_voxcpm_checkpoint(
    model: VoxCPM2Model | VoxCPMAudioVAE,
    path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    require_official_inventory: bool = False,
) -> VoxCPMCheckpointReport:
    report = validate_voxcpm_checkpoint(
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
                target = model.state_dict(keep_vars=True)[name]
                if dtype is not None and target.is_floating_point():
                    value = value.to(dtype=dtype)
                value = value.to(device=target_device)
                model.load_state_dict(
                    {name: value},
                    strict=False,
                    assign=True,
                )
    if isinstance(model, VoxCPM2Model):
        model.materialize_runtime_buffers(target_device)
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "VoxCPM streaming load left meta tensors: " + ", ".join(remaining[:12]))
    return report


def export_voxcpm_checkpoint(
    model: VoxCPM2Model | VoxCPMAudioVAE,
    path: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    if not isinstance(model, (VoxCPM2Model, VoxCPMAudioVAE)):
        raise TypeError("VoxCPM export requires a native model or AudioVAE.")
    model_state = model.state_dict()
    state = model_state if state_override is None else dict(state_override)
    expected = set(model_state)
    if state_override is not None:
        # Native LoRA modules deliberately preserve the original ``weight``
        # and ``bias`` names and add only adapter tensors. A merged export is
        # therefore the exact base namespace with those adapter-only tensors
        # removed. This keeps the resulting artifact directly loadable by the
        # unmodified VoxCPM2 graph.
        adapter_names = {name for name in expected if name.endswith(".lora_A") or name.endswith(".lora_B")}
        if adapter_names:
            expected -= adapter_names
    actual = set(state)
    if actual != expected:
        raise ValueError(
            "VoxCPM export state is incomplete: "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}.")
    metadata = {
        "format": (NATIVE_VOXCPM2_FORMAT if isinstance(model, VoxCPM2Model) else NATIVE_VOXCPM2_CODEC_FORMAT),
        "architecture": "voxcpm2",
    }
    if isinstance(model, VoxCPM2Model):
        metadata["training_objective"] = "source-cfm-plus-stop-ce"
    return save_safetensors(
        {
            name: value.detach()
            for name, value in state.items()
        },
        path,
        metadata=metadata,
    ).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def convert_legacy_voxcpm_codec(
    codec: VoxCPMAudioVAE,
    legacy_path: str | Path,
    destination: str | Path,
    *,
    trust_legacy_pickle: bool = False,
    verify_official_integrity: bool = True,
) -> Path:
    """Convert the upstream AudioVAE archive exactly once.

    ``torch.load`` is never called unless the caller explicitly opts
    into the legacy pickle boundary. PyTorch's restricted
    ``weights_only`` unpickler is mandatory, and the official artifact
    can be checked before parsing.
    """
    if not isinstance(codec, VoxCPMAudioVAE):
        raise TypeError("Legacy VoxCPM codec conversion requires VoxCPMAudioVAE.")
    if not trust_legacy_pickle:
        raise PermissionError(
            "The official VoxCPM2 AudioVAE is a legacy pickle archive. "
            "Set `trust_legacy_pickle=True` only after reviewing its provenance.")
    requested = Path(legacy_path).expanduser()
    if requested.suffix.lower() not in {".pth", ".pt", ".ckpt"}:
        raise ValueError("Legacy VoxCPM AudioVAE input must be a PyTorch archive.")
    source = requested.resolve()
    if verify_official_integrity:
        if source.stat().st_size != VOXCPM2_CODEC_LEGACY_SIZE:
            raise ValueError("Official VoxCPM AudioVAE size verification failed.")
        digest = _file_sha256(source)
        if digest != VOXCPM2_CODEC_LEGACY_SHA256:
            raise ValueError("Official VoxCPM AudioVAE SHA-256 verification failed.")
    try:
        document = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError("This PyTorch build cannot restrict legacy checkpoint unpickling.") from error
    if not isinstance(document, Mapping):
        raise CheckpointCompatibilityError("Legacy VoxCPM AudioVAE root must be a mapping.")
    state = document.get("state_dict", document)
    if not isinstance(state, Mapping) or any(not isinstance(name, str) or not isinstance(value, Tensor)
                                             for name, value in state.items()):
        raise CheckpointCompatibilityError("Legacy VoxCPM AudioVAE contains a non-tensor state dictionary.")
    result = codec.load_state_dict(dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise CheckpointCompatibilityError("Legacy VoxCPM AudioVAE state did not load strictly.")
    return export_voxcpm_checkpoint(codec, destination)


__all__ = [
    "VoxCPMCheckpointReport",
    "convert_legacy_voxcpm_codec",
    "export_voxcpm_checkpoint",
    "inspect_voxcpm_checkpoint",
    "load_voxcpm_checkpoint",
    "tensor_inventory_fingerprint",
    "validate_voxcpm_checkpoint",
]
