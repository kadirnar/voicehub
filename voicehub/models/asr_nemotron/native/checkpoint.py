"""Strict Safetensors loading for native Nemotron 3.5 ASR."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import SafeTensorReader
from voicehub.checkpointing.adapters import CheckpointAdapter, CheckpointCompatibilityReport, TensorShapeMismatch
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan
from voicehub.models.asr_nemotron.native.configuration import NemotronASRArchitectureConfig
from voicehub.models.asr_nemotron.native.metadata import NEMOTRON_ASR_CHECKPOINTS
from voicehub.models.asr_nemotron.native.modeling import Nemotron3_5ASRForRNNT

NemotronConfigLike = (NemotronASRArchitectureConfig | Mapping[str, Any])

_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_FLOAT_CHECKPOINT_DTYPES = frozenset({
    "BF16",
    "F16",
    "F32",
    "F64",
})


def _coerce_config(config: NemotronConfigLike, ) -> NemotronASRArchitectureConfig:
    return NemotronASRArchitectureConfig.coerce(config)


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def native_nemotron_asr_tensor_shapes(config: NemotronConfigLike, ) -> dict[str, tuple[int, ...]]:
    """Return the exact persistent namespace without allocating storage."""
    resolved = _coerce_config(config)
    with torch.device("meta"):
        model = Nemotron3_5ASRForRNNT(
            resolved,
            initialize=False,
        )
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def native_nemotron_asr_tensor_names(config: NemotronConfigLike, ) -> tuple[str, ...]:
    return tuple(sorted(native_nemotron_asr_tensor_shapes(config)))


def nemotron_asr_header_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    """Hash canonical tensor name, Safetensors dtype, and shape rows."""
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("Nemotron ASR inventory must be a non-empty mapping.")
    rows: list[str] = []
    for name, value in sorted(inventory.items()):
        if (not isinstance(name, str) or not name or not isinstance(value, tuple) or len(value) != 2):
            raise ValueError("Invalid Nemotron tensor inventory record.")
        dtype, raw_shape = value
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("Tensor inventory dtype must be non-empty.")
        shape = tuple(raw_shape)
        if any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
               for dimension in shape):
            raise ValueError("Tensor inventory contains an invalid shape.")
        rows.append(f"{name}\t{dtype}\t" + ",".join(str(dimension) for dimension in shape))
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8"), ).hexdigest()


def _reader_inventory(reader: SafeTensorReader, ) -> dict[str, tuple[str, tuple[int, ...]]]:
    if not isinstance(reader, SafeTensorReader):
        raise TypeError("Nemotron ASR accepts a single SafeTensorReader only.")
    return {name: (reader.record(name).dtype, reader.record(name).shape) for name in reader.keys()}


def validate_published_nemotron_asr_inventory(
    reader: SafeTensorReader,
    *,
    source: str,
    revision: str | None,
) -> None:
    """Verify immutable public-checkpoint header facts before allocation."""
    expected = NEMOTRON_ASR_CHECKPOINTS.get(source)
    if expected is None or revision != expected["revision"]:
        return
    inventory = _reader_inventory(reader)
    parameters = sum(_product(shape) for _, shape in inventory.values())
    try:
        tensor_bytes = sum(_DTYPE_BYTES[dtype] * _product(shape) for dtype, shape in inventory.values())
    except KeyError as error:
        raise CheckpointCompatibilityError(
            "Published Nemotron checkpoint uses unsupported Safetensors "
            f"dtype {error.args[0]!r}.") from error
    fingerprint = nemotron_asr_header_fingerprint(inventory)
    with reader.path.open("rb") as stream:
        raw_header_size = stream.read(8)
    if len(raw_header_size) != 8:
        raise CheckpointCompatibilityError("Published Nemotron checkpoint has a truncated header prefix.")
    (header_size, ) = struct.unpack("<Q", raw_header_size)
    file_size = reader.path.stat().st_size
    errors: list[str] = []
    facts = (
        ("size", file_size, expected["size"]),
        ("header_size", header_size, expected["header_size"]),
        ("tensors", len(inventory), expected["tensors"]),
        ("parameters", parameters, expected["parameters"]),
        ("tensor_bytes", tensor_bytes, expected["tensor_bytes"]),
    )
    errors.extend(
        f"{name}={actual} (expected {wanted})" for name, actual, wanted in facts if actual != wanted)
    dtypes = {dtype for dtype, _ in inventory.values()}
    if dtypes != {expected["dtype"]}:
        errors.append(f"dtypes={sorted(dtypes)!r} "
                      f"(expected {[expected['dtype']]!r})")
    if fingerprint != expected["header_fingerprint"]:
        errors.append(f"header_fingerprint={fingerprint} "
                      f"(expected {expected['header_fingerprint']})")
    if errors:
        raise CheckpointCompatibilityError(
            "Published Nemotron ASR checkpoint verification failed: " + "; ".join(errors))


def materialize_nemotron_asr_nonpersistent_buffers(
    model: Nemotron3_5ASRForRNNT,
    *,
    device: str | torch.device,
) -> None:
    """Rebuild deterministic buffers omitted from Safetensors state."""
    if not isinstance(model, Nemotron3_5ASRForRNNT):
        raise TypeError("Nemotron buffer target has an incompatible graph.")
    config = model.config.encoder_config
    exponents = torch.arange(
        0,
        config.hidden_size,
        2,
        dtype=torch.float32,
        device=device,
    ) / config.hidden_size
    inv_freq = torch.pow(10000.0, -exponents)
    model.encoder.encode_positions.inv_freq = inv_freq


class NemotronASRCheckpointAdapter(CheckpointAdapter):
    """Identity-map an audited Nemotron RNN-T Safetensors namespace."""

    architecture_id = "nemotron-3.5-rnnt"
    adapter_id = "nemotron-3.5-rnnt-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        return (
            str(config.get("model_type", "")).lower() == "nemotron3_5_asr" and len(files) == 1 and
            files[0].name == "model.safetensors")

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        return TensorPlan(
            rules=tuple(CopyTensor(name, name) for name in native_nemotron_asr_tensor_names(config)), )

    def load_assign_streaming(
        self,
        model: Nemotron3_5ASRForRNNT,
        source: SafeTensorReader,
        config: NemotronConfigLike,
        *,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
        strict: bool = True,
    ) -> CheckpointCompatibilityReport:
        """Validate every header, then assign at most one tensor at a time."""
        type(self)._validate_identity()
        if not isinstance(model, Nemotron3_5ASRForRNNT):
            raise TypeError("Nemotron checkpoint target has an incompatible graph.")
        if dtype is not None and (not isinstance(dtype, torch.dtype) or not dtype.is_floating_point):
            raise TypeError("Nemotron execution `dtype` must be floating-point.")
        normalized = self._source(source)
        expected_shapes = native_nemotron_asr_tensor_shapes(config)
        expected = set(expected_shapes)
        available = set(normalized.keys())
        missing = tuple(sorted(expected - available))
        unused = tuple(sorted(available - expected))
        tensor_shape = getattr(normalized, "tensor_shape", None)
        mismatches: list[TensorShapeMismatch] = []
        for name in sorted(expected & available):
            checkpoint_shape = (
                tuple(tensor_shape(name)) if callable(tensor_shape) else tuple(
                    normalized.get_tensor(name).shape))
            if checkpoint_shape != expected_shapes[name]:
                mismatches.append(
                    TensorShapeMismatch(
                        name=name,
                        checkpoint_shape=checkpoint_shape,
                        model_shape=expected_shapes[name],
                    ))
        mismatch_names = {mismatch.name for mismatch in mismatches}
        loaded = tuple(sorted(expected & available - mismatch_names, ))
        report = CheckpointCompatibilityReport(
            architecture=self.architecture_id,
            adapter=self.qualified_id,
            loaded=loaded,
            missing=missing,
            shape_mismatches=tuple(mismatches),
            unused_sources=unused,
        )
        if strict:
            report.require_compatible()

        materialized: dict[str, torch.Tensor] = {}
        dtype_errors: list[str] = []
        if isinstance(normalized, SafeTensorReader):
            for name in report.loaded:
                checkpoint_dtype = normalized.record(name).dtype
                if checkpoint_dtype not in _FLOAT_CHECKPOINT_DTYPES:
                    dtype_errors.append(
                        f"{name!r} uses {checkpoint_dtype!r}, expected a "
                        "standard floating-point tensor")
        else:
            for name in report.loaded:
                value = normalized.get_tensor(name)
                if not isinstance(value, torch.Tensor):
                    dtype_errors.append(f"{name!r} is not a tensor")
                    continue
                materialized[name] = value
                if value.device.type == "meta":
                    dtype_errors.append(f"{name!r} is not materialized")
                elif value.layout != torch.strided:
                    dtype_errors.append(f"{name!r} is not strided")
                elif value.is_quantized:
                    dtype_errors.append(f"{name!r} is quantized")
                elif value.is_complex():
                    dtype_errors.append(f"{name!r} is complex")
                elif not value.is_floating_point():
                    dtype_errors.append(f"{name!r} uses non-floating dtype {value.dtype}")
        if dtype_errors:
            raise CheckpointCompatibilityError(
                "Nemotron checkpoint has incompatible tensor "
                "dtypes/layouts: " + "; ".join(dtype_errors[:8]))

        with torch.no_grad():
            for name in report.loaded:
                value = materialized.pop(
                    name,
                    None,
                )
                if value is None:
                    value = normalized.get_tensor(name)
                target_dtype = (dtype if dtype is not None and value.is_floating_point() else value.dtype)
                model.load_state_dict(
                    {
                        name: value.to(
                            device=device,
                            dtype=target_dtype,
                        ),
                    },
                    strict=False,
                    assign=True,
                )
                del value
        remaining = tuple(name for name, value in model.state_dict().items() if value.device.type == "meta")
        if remaining:
            raise CheckpointCompatibilityError(
                "Nemotron checkpoint assignment left meta tensors: " + ", ".join(remaining[:8]))
        materialize_nemotron_asr_nonpersistent_buffers(
            model,
            device=device,
        )
        return report


__all__ = [
    "NemotronASRCheckpointAdapter",
    "materialize_nemotron_asr_nonpersistent_buffers",
    "native_nemotron_asr_tensor_names",
    "native_nemotron_asr_tensor_shapes",
    "nemotron_asr_header_fingerprint",
    "validate_published_nemotron_asr_inventory",
]
