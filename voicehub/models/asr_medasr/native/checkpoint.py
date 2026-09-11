"""Strict Safetensors inventory validation and streaming MedASR loading."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import (
    CheckpointAdapter,
    CheckpointCompatibilityReport,
    SafeTensorReader,
    TensorShapeMismatch,
)
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan
from voicehub.models.asr_medasr.native.configuration import MedASRConfig
from voicehub.models.asr_medasr.native.metadata import MEDASR_CHECKPOINT, MEDASR_MODEL_ID, MEDASR_MODEL_REVISION
from voicehub.models.asr_medasr.native.modeling import MedASRForCTC

_FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})
_DTYPE_BYTES = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I64": 8,
}


def _coerce_config(config: MedASRConfig | Mapping[str, Any], ) -> MedASRConfig:
    return (config if isinstance(config, MedASRConfig) else MedASRConfig.from_dict(config))


def native_medasr_tensor_shapes(config: MedASRConfig | Mapping[str, Any], ) -> dict[str, tuple[int, ...]]:
    resolved = _coerce_config(config)
    model = MedASRForCTC(resolved, initialize=False)
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def native_medasr_tensor_dtypes(config: MedASRConfig | Mapping[str, Any], ) -> dict[str, str]:
    resolved = _coerce_config(config)
    model = MedASRForCTC(resolved, initialize=False)
    names = {}
    for name, tensor in model.state_dict().items():
        if tensor.dtype == torch.float32:
            names[name] = "F32"
        elif tensor.dtype == torch.int64:
            names[name] = "I64"
        else:  # pragma: no cover - graph invariant
            raise RuntimeError(f"Unexpected native MedASR dtype {tensor.dtype}.")
    return names


def medasr_header_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("MedASR tensor inventory must be a non-empty mapping.")
    rows = []
    for name, value in sorted(inventory.items()):
        if (not isinstance(name, str) or not name or not isinstance(value, tuple) or len(value) != 2):
            raise ValueError("MedASR tensor inventory contains an invalid record.")
        dtype, shape = value
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("MedASR tensor inventory dtypes must be strings.")
        dimensions = tuple(shape)
        if any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
               for dimension in dimensions):
            raise ValueError("MedASR tensor inventory contains an invalid shape.")
        rows.append(f"{name}\t{dtype}\t" + ",".join(str(value) for value in dimensions))
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8"), ).hexdigest()


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def medasr_reader_inventory(reader: SafeTensorReader, ) -> dict[str, tuple[str, tuple[int, ...]]]:
    if not isinstance(reader, SafeTensorReader):
        raise TypeError("MedASR checkpoints require a SafeTensorReader.")
    return {
        name: (
            reader.record(name).dtype,
            reader.record(name).shape,
        )
        for name in reader.keys()
    }


def validate_published_medasr_inventory(
    reader: SafeTensorReader,
    *,
    source: str,
    revision: str | None,
) -> None:
    """Fail closed when an official pinned checkpoint header differs."""
    if source != MEDASR_MODEL_ID or revision != MEDASR_MODEL_REVISION:
        return
    inventory = medasr_reader_inventory(reader)
    dtype_counts = Counter(dtype for dtype, _ in inventory.values())
    parameters = sum(_product(shape) for _, shape in inventory.values())
    tensor_data_bytes = sum(
        _DTYPE_BYTES.get(dtype, 0) * _product(shape) for dtype, shape in inventory.values())
    fingerprint = medasr_header_fingerprint(inventory)
    expected = MEDASR_CHECKPOINT
    errors = []
    facts = {
        "tensors": len(inventory),
        "parameters": parameters,
        "tensor_data_bytes": tensor_data_bytes,
        "header_fingerprint": fingerprint,
    }
    for name, actual in facts.items():
        if actual != expected[name]:
            errors.append(f"{name}={actual!r} (expected {expected[name]!r})")
    if dict(dtype_counts) != expected["dtype_counts"]:
        errors.append(f"dtype_counts={dict(dtype_counts)!r} "
                      f"(expected {expected['dtype_counts']!r})")
    if reader.path.stat().st_size != expected["file_bytes"]:
        errors.append(f"file_bytes={reader.path.stat().st_size} "
                      f"(expected {expected['file_bytes']})")
    if errors:
        raise CheckpointCompatibilityError(
            "Published MedASR checkpoint verification failed: " + "; ".join(errors))


class MedASRCheckpointAdapter(CheckpointAdapter):
    """Identity-map LASR tensors into the VoiceHub-owned native graph."""

    architecture_id = "lasr-ctc"
    adapter_id = "huggingface-lasr-ctc-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        return (
            str(config.get("model_type", "")).strip().lower() in {"asr_medasr", "lasr_ctc"} and
            any(path.suffix == ".safetensors" for path in files))

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        return TensorPlan(
            rules=tuple(CopyTensor(name, name) for name in sorted(native_medasr_tensor_shapes(config), )), )

    def load_assign_streaming(
        self,
        model: MedASRForCTC,
        source: Any,
        config: MedASRConfig | Mapping[str, Any],
        *,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
        strict: bool = True,
    ) -> CheckpointCompatibilityReport:
        """Validate all headers before assigning one tensor at a time."""
        type(self)._validate_identity()
        if not isinstance(model, MedASRForCTC):
            raise TypeError("MedASR checkpoint target has an incompatible graph.")
        if dtype is not None and (not isinstance(dtype, torch.dtype) or not dtype.is_floating_point):
            raise TypeError("MedASR execution `dtype` must be floating-point.")
        normalized = self._source(source)
        expected_shapes = native_medasr_tensor_shapes(config)
        expected_names = set(expected_shapes)
        available_names = set(normalized.keys())
        missing = tuple(sorted(expected_names - available_names))
        unused = tuple(sorted(available_names - expected_names))
        tensor_shape = getattr(normalized, "tensor_shape", None)
        mismatches = []
        for name in sorted(expected_names & available_names):
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
        loaded = tuple(sorted(expected_names & available_names - mismatch_names))
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

        model_state = model.state_dict()
        record = getattr(normalized, "record", None)
        tensor_dtype = getattr(normalized, "tensor_dtype", None)
        materialized: dict[str, torch.Tensor] = {}
        dtype_errors = []
        for name in report.loaded:
            target = model_state[name]
            if callable(record):
                checkpoint_dtype = record(name).dtype
                value = None
            elif callable(tensor_dtype):
                checkpoint_dtype = tensor_dtype(name)
                value = None
            else:
                value = normalized.get_tensor(name)
                materialized[name] = value
                if not isinstance(value, torch.Tensor):
                    dtype_errors.append(f"{name!r} is not a tensor")
                    continue
                if value.device.type == "meta":
                    dtype_errors.append(f"{name!r} is not materialized")
                    continue
                if value.layout != torch.strided:
                    dtype_errors.append(f"{name!r} is not strided")
                    continue
                if value.is_quantized:
                    dtype_errors.append(f"{name!r} is quantized")
                    continue
                if value.is_complex():
                    dtype_errors.append(f"{name!r} is complex")
                    continue
                checkpoint_dtype = (
                    "I64" if value.dtype == torch.int64 else
                    "F32" if value.dtype == torch.float32 else "BF16" if value.dtype == torch.bfloat16 else
                    "F16" if value.dtype == torch.float16 else str(value.dtype))
            if target.is_floating_point():
                if checkpoint_dtype not in _FLOAT_DTYPES:
                    dtype_errors.append(f"{name!r} uses {checkpoint_dtype!r}, expected floating")
            elif target.dtype == torch.int64:
                if checkpoint_dtype != "I64":
                    dtype_errors.append(f"{name!r} uses {checkpoint_dtype!r}, expected I64")
            else:  # pragma: no cover - graph invariant
                dtype_errors.append(f"{name!r} targets unsupported dtype {target.dtype}")
        if dtype_errors:
            raise CheckpointCompatibilityError(
                "MedASR checkpoint has incompatible tensor dtypes/layouts: " + "; ".join(dtype_errors[:8]))

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
                    {name: value.to(
                        device=device,
                        dtype=target_dtype,
                    )},
                    strict=False,
                    assign=True,
                )
                del value
        remaining = tuple(name for name, value in model.state_dict().items() if value.device.type == "meta")
        if remaining:
            raise CheckpointCompatibilityError(
                "MedASR checkpoint assignment left meta tensors: " + ", ".join(remaining[:5]))
        return report


__all__ = [
    "MedASRCheckpointAdapter",
    "medasr_header_fingerprint",
    "medasr_reader_inventory",
    "native_medasr_tensor_dtypes",
    "native_medasr_tensor_shapes",
    "validate_published_medasr_inventory",
]
