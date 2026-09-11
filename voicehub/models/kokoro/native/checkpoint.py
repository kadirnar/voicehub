"""Strict Kokoro checkpoint loading and legacy conversion."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from voicehub.checkpointing import SafeTensorReader
from voicehub.checkpointing.adapters import CheckpointCompatibilityReport, TensorShapeMismatch
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.kokoro.native.metadata import (
    KOKORO_CHECKPOINT_REVISION,
    KOKORO_LEGACY_HEADER_FINGERPRINT,
    KOKORO_LEGACY_PARAMETER_COUNT,
    KOKORO_LEGACY_TENSOR_COUNT,
    KOKORO_NATIVE_FORMAT,
    KOKORO_PYTORCH_SHA256,
)

_SAFETENSORS_DTYPES = {
    torch.bool: ("BOOL", 1),
    torch.uint8: ("U8", 1),
    torch.int8: ("I8", 1),
    torch.int16: ("I16", 2),
    torch.int32: ("I32", 4),
    torch.int64: ("I64", 8),
    torch.float16: ("F16", 2),
    torch.bfloat16: ("BF16", 2),
    torch.float32: ("F32", 4),
    torch.float64: ("F64", 8),
}


def _model_state(model: nn.Module) -> Mapping[str, torch.Tensor]:
    state = model.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise TypeError("Kokoro checkpoint target returned an invalid state_dict.")
    return state


def _instance_norm_defaults(model: nn.Module) -> tuple[str, ...]:
    names = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.InstanceNorm1d) or not module.affine:
            continue
        names.extend((
            f"{module_name}.weight",
            f"{module_name}.bias",
        ))
    return tuple(sorted(names))


def _legacy_to_native_name(component: str, name: str) -> str:
    if name.startswith("module."):
        name = name[7:]
    if name.endswith(".weight_g"):
        name = (name[:-len(".weight_g")] + ".parametrizations.weight.original0")
    elif name.endswith(".weight_v"):
        name = (name[:-len(".weight_v")] + ".parametrizations.weight.original1")
    return f"{component}.{name}"


def _native_to_legacy_name(name: str) -> tuple[str, str]:
    try:
        component, nested = name.split(".", 1)
    except ValueError as error:
        raise ValueError(f"Invalid native Kokoro tensor name {name!r}.") from error
    if nested.endswith(".parametrizations.weight.original0"):
        nested = (nested[:-len(".parametrizations.weight.original0")] + ".weight_g")
    elif nested.endswith(".parametrizations.weight.original1"):
        nested = (nested[:-len(".parametrizations.weight.original1")] + ".weight_v")
    return component, f"module.{nested}"


def _validate_tensor(
    value: Any,
    *,
    name: str,
    expected: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise CheckpointCompatibilityError(f"Kokoro checkpoint value {name!r} is not a tensor.")
    if tuple(value.shape) != tuple(expected.shape):
        raise CheckpointCompatibilityError(
            f"Kokoro checkpoint tensor {name!r} has shape "
            f"{tuple(value.shape)}, expected {tuple(expected.shape)}.")
    if value.layout != torch.strided:
        raise CheckpointCompatibilityError(f"Kokoro checkpoint tensor {name!r} must use strided layout.")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise CheckpointCompatibilityError(f"Kokoro checkpoint tensor {name!r} contains NaN or infinity.")
    return value.detach().cpu().contiguous()


def _legacy_inventory(payload: Any, ) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise CheckpointCompatibilityError("Legacy Kokoro checkpoint must be a component mapping.")
    expected_components = {
        "bert",
        "bert_encoder",
        "predictor",
        "decoder",
        "text_encoder",
    }
    supplied_components = set(payload)
    if supplied_components != expected_components:
        missing = sorted(expected_components - supplied_components)
        unexpected = sorted(supplied_components - expected_components)
        raise CheckpointCompatibilityError(
            "Legacy Kokoro components do not match the released graph "
            f"(missing={missing!r}, unexpected={unexpected!r}).")
    flattened: dict[str, torch.Tensor] = {}
    for component in sorted(expected_components):
        state = payload[component]
        if not isinstance(state, Mapping):
            raise CheckpointCompatibilityError(f"Legacy Kokoro component {component!r} is not a state dict.")
        for source_name, value in state.items():
            if not isinstance(source_name, str) or not source_name:
                raise CheckpointCompatibilityError("Legacy Kokoro tensor names must be non-empty strings.")
            target_name = _legacy_to_native_name(component, source_name)
            if target_name in flattened:
                raise CheckpointCompatibilityError(
                    f"Legacy Kokoro tensor mapping collides at {target_name!r}.")
            flattened[target_name] = value
    return flattened


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _save_safetensors_streaming(
    tensors: Mapping[str, torch.Tensor],
    path: str | Path,
    *,
    metadata: Mapping[str, str],
) -> Path:
    """Write standard Safetensors while retaining at most one byte payload."""
    if sys.byteorder != "little":  # pragma: no cover - uncommon platform
        raise RuntimeError("Kokoro Safetensors export requires little-endian.")
    names = tuple(sorted(tensors))
    if not names:
        raise ValueError("Kokoro Safetensors export cannot be empty.")
    normalized_metadata = dict(metadata)
    if any(not isinstance(key, str) or not isinstance(value, str)
           for key, value in normalized_metadata.items()):
        raise TypeError("Safetensors metadata must map strings to strings.")
    header: dict[str, Any] = {}
    offset = 0
    for name in names:
        value = tensors[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Kokoro state value {name!r} is not a tensor.")
        if value.layout != torch.strided or value.device.type == "meta":
            raise ValueError(f"Kokoro state value {name!r} is not serializable.")
        try:
            dtype_name, width = _SAFETENSORS_DTYPES[value.dtype]
        except KeyError as error:
            raise ValueError(f"Unsupported Kokoro tensor dtype {value.dtype}.") from error
        number_of_elements = value.numel()
        end = offset + number_of_elements * width
        header[name] = {
            "dtype": dtype_name,
            "shape": list(value.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    header["__metadata__"] = normalized_metadata
    encoded_header = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded_header += b" " * ((-len(encoded_header)) % 8)

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(struct.pack("<Q", len(encoded_header)))
            stream.write(encoded_header)
            # ``UntypedStorage._write_file`` writes through the descriptor,
            # bypassing Python's buffered stream. Flush the complete header
            # first so a small header still advances the descriptor offset
            # before the first tensor payload.
            stream.flush()
            for name in names:
                materialized = (
                    tensors[name].detach().to(device="cpu").contiguous().clone(
                        memory_format=torch.contiguous_format))
                # PyTorch's C++ storage writer avoids a second full Python
                # ``bytes`` allocation and keeps large checkpoint export
                # bounded to one tensor. ``save_size=False`` writes exactly
                # the raw little-endian tensor payload required by
                # Safetensors.
                materialized.untyped_storage()._write_file(
                    stream,
                    True,
                    False,
                    materialized.element_size(),
                )
                del materialized
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def import_legacy_kokoro_checkpoint(
    model: nn.Module,
    source_path: str | Path,
    *,
    output_path: str | Path | None = None,
    verify_official_hash: bool = False,
) -> Path:
    """Convert one released ``.pth`` into canonical Safetensors.

    PyTorch's restricted ``weights_only`` unpickler is mandatory. There
    is no unsafe compatibility fallback. All 548 released tensors and
    all shapes must match the constructed graph before an output file is
    written.
    """
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Kokoro checkpoint was not found: {source}.")
    if source.suffix.lower() not in {".pth", ".pt"}:
        raise ValueError("Legacy Kokoro import accepts only .pth or .pt files.")
    if verify_official_hash:
        observed = _sha256(source)
        if observed != KOKORO_PYTORCH_SHA256:
            raise CheckpointCompatibilityError(
                "Official Kokoro checkpoint SHA-256 mismatch: "
                f"expected {KOKORO_PYTORCH_SHA256}, found {observed}.")
    try:
        payload = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "Legacy Kokoro conversion requires a PyTorch release with "
            "`torch.load(..., weights_only=True)` support.") from error
    flattened = _legacy_inventory(payload)
    del payload

    model_state = _model_state(model)
    generated = set(_instance_norm_defaults(model))
    expected_sources = set(model_state) - generated
    supplied = set(flattened)
    missing = tuple(sorted(expected_sources - supplied))
    unexpected = tuple(sorted(supplied - expected_sources))
    mismatches = tuple(
        TensorShapeMismatch(
            name=name,
            checkpoint_shape=tuple(flattened[name].shape),
            model_shape=tuple(model_state[name].shape),
        ) for name in sorted(expected_sources & supplied)
        if tuple(flattened[name].shape) != tuple(model_state[name].shape))
    if (len(flattened) != KOKORO_LEGACY_TENSOR_COUNT or missing or unexpected or mismatches):
        raise CheckpointCompatibilityError(
            "Legacy Kokoro checkpoint inventory is incompatible: "
            f"tensors={len(flattened)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}, "
            f"shape_mismatches={len(mismatches)}.")

    canonical: dict[str, torch.Tensor] = {}
    for name, expected in model_state.items():
        if name in generated:
            suffix = name.rsplit(".", 1)[-1]
            if expected.device.type == "meta":
                factory = torch.ones if suffix == "weight" else torch.zeros
                canonical[name] = factory(
                    tuple(expected.shape),
                    dtype=expected.dtype,
                    device="cpu",
                )
            else:
                value = expected.detach().cpu()
                correct = (
                    bool(torch.equal(value, torch.ones_like(value))) if suffix == "weight" else bool(
                        torch.equal(value, torch.zeros_like(value))))
                if not correct:
                    raise CheckpointCompatibilityError(
                        "Kokoro's checkpoint-omitted InstanceNorm affine "
                        f"default is invalid for {name!r}.")
                canonical[name] = value.contiguous()
        else:
            canonical[name] = _validate_tensor(
                flattened[name],
                name=name,
                expected=expected,
            )
    destination = (
        source.with_suffix(".voicehub.safetensors")
        if output_path is None else Path(output_path).expanduser().resolve())
    _save_safetensors_streaming(
        canonical,
        destination,
        metadata={
            "format": KOKORO_NATIVE_FORMAT,
            "source_format": "torch-weights-only",
            "source_revision": KOKORO_CHECKPOINT_REVISION,
            "source_sha256": (KOKORO_PYTORCH_SHA256 if verify_official_hash else "unverified"),
            "training_scope": "preprocessed-decoder-reconstruction",
        },
    )
    return destination


def load_native_kokoro_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> CheckpointCompatibilityReport:
    """Strictly load every canonical Kokoro tensor from Safetensors."""
    path = Path(checkpoint_path).expanduser().resolve()
    model_state = _model_state(model)
    with SafeTensorReader(path) as reader:
        expected = set(model_state)
        available = set(reader.keys())
        missing = tuple(sorted(expected - available))
        unexpected = tuple(sorted(available - expected))
        mismatches = tuple(
            TensorShapeMismatch(
                name=name,
                checkpoint_shape=reader.tensor_shape(name),
                model_shape=tuple(model_state[name].shape),
            ) for name in sorted(expected & available)
            if reader.tensor_shape(name) != tuple(model_state[name].shape))
        mismatch_names = {item.name for item in mismatches}
        loaded = tuple(sorted(expected & available - mismatch_names))
        report = CheckpointCompatibilityReport(
            architecture="kokoro",
            adapter="voicehub-kokoro-safetensors@1",
            loaded=loaded,
            missing=missing,
            shape_mismatches=mismatches,
            unused_sources=unexpected,
        )
        report.require_compatible()
        with torch.no_grad():
            for name in loaded:
                value = reader.get_tensor(
                    name,
                    device=device,
                    dtype=(
                        dtype if dtype is not None and model_state[name].is_floating_point() else
                        model_state[name].dtype),
                )
                model.load_state_dict(
                    {name: value},
                    strict=False,
                    assign=True,
                )
    remaining = set(model.state_dict())
    if remaining != set(loaded):
        raise CheckpointCompatibilityError("Kokoro checkpoint assignment changed the model inventory.")
    return report


def save_native_kokoro_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Export the complete portable inference/fine-tuning state."""
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in _model_state(model).items()}
    resolved_metadata = {
        "format": KOKORO_NATIVE_FORMAT,
        "source_revision": KOKORO_CHECKPOINT_REVISION,
        "training_scope": "preprocessed-decoder-reconstruction",
    }
    resolved_metadata.update(dict(metadata or {}))
    return _save_safetensors_streaming(
        state,
        path,
        metadata=resolved_metadata,
    )


def _validate_voice_pack(
    value: Any,
    *,
    source: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise CheckpointCompatibilityError(f"Kokoro voice pack {source!r} must contain one tensor.")
    if (value.ndim != 3 or value.shape[0] < 1 or value.shape[1] != 1 or value.shape[2] != 256):
        raise CheckpointCompatibilityError(
            "Kokoro voice packs must have shape [phoneme_length, 1, 256]; "
            f"{source!r} has {tuple(value.shape)}.")
    if not value.is_floating_point():
        raise CheckpointCompatibilityError(f"Kokoro voice pack {source!r} must use a floating dtype.")
    if not bool(torch.isfinite(value).all()):
        raise CheckpointCompatibilityError(f"Kokoro voice pack {source!r} contains NaN or infinity.")
    return value.detach().cpu().contiguous()


def import_legacy_kokoro_voice(
    source_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Convert one released voice ``.pt`` with the restricted unpickler."""
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Kokoro voice pack was not found: {source}.")
    if source.suffix.lower() != ".pt":
        raise ValueError("Legacy Kokoro voice import accepts only .pt files.")
    try:
        payload = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "Legacy Kokoro voice conversion requires "
            "`torch.load(..., weights_only=True)` support.") from error
    voice = _validate_voice_pack(payload, source=str(source))
    destination = (
        source.with_suffix(".voicehub.safetensors")
        if output_path is None else Path(output_path).expanduser().resolve())
    return _save_safetensors_streaming(
        {"style": voice},
        destination,
        metadata={
            "format": "voicehub-kokoro-voice-v1",
            "source_format": "torch-weights-only",
            "source_revision": KOKORO_CHECKPOINT_REVISION,
        },
    )


def load_native_kokoro_voice(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load one strict native voice pack."""
    source = Path(path).expanduser().resolve()
    with SafeTensorReader(source) as reader:
        if reader.keys() != ("style", ):
            raise CheckpointCompatibilityError("Native Kokoro voice Safetensors must contain only `style`.")
        voice = reader.get_tensor("style", device=device, dtype=dtype)
    validated = _validate_voice_pack(voice, source=str(source))
    return validated.to(device=device, dtype=dtype)


def save_native_kokoro_voice(
    voice: torch.Tensor,
    path: str | Path,
) -> Path:
    """Export one validated portable voice pack."""
    value = _validate_voice_pack(voice, source="<runtime>")
    return _save_safetensors_streaming(
        {"style": value},
        path,
        metadata={"format": "voicehub-kokoro-voice-v1"},
    )


def legacy_kokoro_tensor_names(model: nn.Module) -> tuple[str, ...]:
    """Return the exact expected component-prefixed legacy namespace."""
    generated = set(_instance_norm_defaults(model))
    return tuple(
        sorted(
            f"{component}.{name}" for native_name in _model_state(model) if native_name not in generated
            for component, name in (_native_to_legacy_name(native_name), )))


__all__ = [
    "KOKORO_CHECKPOINT_REVISION",
    "KOKORO_LEGACY_HEADER_FINGERPRINT",
    "KOKORO_LEGACY_PARAMETER_COUNT",
    "KOKORO_LEGACY_TENSOR_COUNT",
    "KOKORO_NATIVE_FORMAT",
    "KOKORO_PYTORCH_SHA256",
    "import_legacy_kokoro_checkpoint",
    "import_legacy_kokoro_voice",
    "legacy_kokoro_tensor_names",
    "load_native_kokoro_checkpoint",
    "load_native_kokoro_voice",
    "save_native_kokoro_checkpoint",
    "save_native_kokoro_voice",
]
