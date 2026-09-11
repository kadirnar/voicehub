"""Strict official/portable Safetensors loading for native Qwen3-ASR."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader
from voicehub.checkpointing.adapters import CheckpointAdapter, CheckpointCompatibilityReport, TensorShapeMismatch
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan
from voicehub.models.asr_qwen3.native.configuration import Qwen3ASRArchitectureConfig
from voicehub.models.asr_qwen3.native.metadata import QWEN3_ASR_CHECKPOINTS
from voicehub.models.asr_qwen3.native.modeling import (
    Qwen3ASRForConditionalGeneration,
    materialize_qwen3_asr_nonpersistent_buffers,
)

_ConfigValue = Qwen3ASRArchitectureConfig | Mapping[str, Any]
_TensorInventory = dict[str, tuple[str, tuple[int, ...]]]
_TensorShapes = dict[str, tuple[int, ...]]


def native_qwen3_asr_tensor_shapes(config: _ConfigValue) -> _TensorShapes:
    """Return the complete persistent namespace without allocating storage."""
    resolved = Qwen3ASRArchitectureConfig.coerce(config)
    with torch.device("meta"):
        model = Qwen3ASRForConditionalGeneration(
            resolved,
            initialize=False,
            tie_weights=False,
        )
    return {name: tuple(value.shape) for name, value in model.state_dict().items()}


def native_qwen3_asr_tensor_names(config: _ConfigValue) -> tuple[str, ...]:
    return tuple(sorted(native_qwen3_asr_tensor_shapes(config)))


def qwen3_asr_header_fingerprint(inventory: _TensorInventory) -> str:
    """Hash an exact name/dtype/shape inventory in a stable format."""
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("Qwen3-ASR inventory must be a non-empty mapping.")
    rows = []
    for name, value in sorted(inventory.items()):
        if (not isinstance(name, str) or not name or not isinstance(value, tuple) or len(value) != 2):
            raise ValueError("Invalid Qwen3-ASR tensor inventory record.")
        dtype, shape = value
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("Tensor inventory dtype must be non-empty.")
        dimensions = tuple(shape)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in dimensions):
            raise ValueError("Tensor inventory contains an invalid shape.")
        rows.append(f"{name}\t{dtype}\t{','.join(str(item) for item in dimensions)}")
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reader_inventory(reader: SafeTensorReader | ShardedSafeTensorReader, ) -> _TensorInventory:
    if isinstance(reader, SafeTensorReader):
        names = reader.keys()
        return {name: (reader.record(name).dtype, reader.record(name).shape) for name in names}
    inventory: dict[str, tuple[str, tuple[int, ...]]] = {}
    by_shard: dict[Path, list[str]] = {}
    names = reader.keys()
    for name in names:
        by_shard.setdefault(reader.index.shard_path(name), []).append(name)
    for shard, indexed_names in sorted(
            by_shard.items(),
            key=lambda item: item[0].name,
    ):
        with SafeTensorReader(shard) as shard_reader:
            actual_names = set(shard_reader.keys())
            expected_names = set(indexed_names)
            undeclared = actual_names - expected_names
            missing = expected_names - actual_names
            if undeclared or missing:
                raise CheckpointCompatibilityError(
                    f"Qwen3-ASR shard {shard.name!r} disagrees with its "
                    f"index: missing={sorted(missing)!r}, "
                    f"undeclared={sorted(undeclared)!r}.")
            for name in indexed_names:
                record = shard_reader.record(name)
                inventory[name] = (record.dtype, record.shape)
    return inventory


def validate_published_qwen3_asr_inventory(
    reader: SafeTensorReader | ShardedSafeTensorReader,
    *,
    source: str,
    revision: str | None,
) -> None:
    """Verify exact public checkpoint metadata when source is recognized."""
    expected = QWEN3_ASR_CHECKPOINTS.get(source)
    if expected is None or revision != expected["revision"]:
        return
    inventory = _reader_inventory(reader)
    fingerprint = qwen3_asr_header_fingerprint(inventory)
    parameters = sum(math_product(shape) for _, shape in inventory.values())
    errors = []
    if len(inventory) != expected["tensors"]:
        errors.append(f"tensors={len(inventory)} (expected {expected['tensors']})")
    if parameters != expected["parameters"]:
        errors.append(f"parameters={parameters} (expected {expected['parameters']})")
    if fingerprint != expected["header_fingerprint"]:
        errors.append(f"header_fingerprint={fingerprint} "
                      f"(expected {expected['header_fingerprint']})")
    if errors:
        raise CheckpointCompatibilityError(
            "Published Qwen3-ASR checkpoint inventory verification failed: " + "; ".join(errors))


def math_product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


class Qwen3ASRCheckpointAdapter(CheckpointAdapter):
    """Identity-map official Qwen3-ASR tensors into the native graph."""

    architecture_id = "qwen3-asr"
    adapter_id = "qwen3-asr-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        model_type = str(config.get("model_type", "")).lower().replace("-", "_")
        return model_type in {"qwen3_asr", "asr_qwen3"} and any(
            path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json") for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        return TensorPlan(
            rules=tuple(CopyTensor(name, name) for name in native_qwen3_asr_tensor_names(config)))

    def load_assign_streaming(
        self,
        model: Qwen3ASRForConditionalGeneration,
        source: Any,
        config: Qwen3ASRArchitectureConfig | Mapping[str, Any],
        *,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
        strict: bool = True,
    ) -> CheckpointCompatibilityReport:
        """Validate all headers, then assign one tensor at a time."""
        type(self)._validate_identity()
        if not isinstance(model, Qwen3ASRForConditionalGeneration):
            raise TypeError("Qwen3-ASR checkpoint target has an incompatible graph.")
        normalized = self._source(source)
        expected_shapes = native_qwen3_asr_tensor_shapes(config)
        expected = set(expected_shapes)
        available = set(normalized.keys())
        missing = tuple(sorted(expected - available))
        unused = tuple(sorted(available - expected))
        tensor_shape = getattr(normalized, "tensor_shape", None)
        mismatches = []
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
        mismatch_names = {value.name for value in mismatches}
        loaded = tuple(sorted(expected & available - mismatch_names))
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

        with torch.no_grad():
            for name in report.loaded:
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
                "Qwen3-ASR checkpoint assignment left meta tensors: " + ", ".join(remaining[:5]))
        model.tie_weights()
        materialize_qwen3_asr_nonpersistent_buffers(model, device=device)
        return report


__all__ = [
    "Qwen3ASRCheckpointAdapter",
    "native_qwen3_asr_tensor_names",
    "native_qwen3_asr_tensor_shapes",
    "qwen3_asr_header_fingerprint",
    "validate_published_qwen3_asr_inventory",
]
