"""Strict Safetensors loading for native Parakeet TDT."""

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
from voicehub.models.asr_parakeet_tdt.native.configuration import ParakeetTDTConfig
from voicehub.models.asr_parakeet_tdt.native.metadata import PARAKEET_TDT_CHECKPOINTS
from voicehub.models.asr_parakeet_tdt.native.modeling import ParakeetForTDT

TensorInventory = dict[str, tuple[str, tuple[int, ...]]]
ConfigValue = ParakeetTDTConfig | Mapping[str, Any]
_FLOAT_CHECKPOINT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})


def native_parakeet_tdt_tensor_shapes(config: ConfigValue) -> dict[str, tuple[int, ...]]:
    """Return the exact persistent namespace without allocating storage."""
    resolved = ParakeetTDTConfig.coerce(config)
    with torch.device("meta"):
        model = ParakeetForTDT(resolved, initialize=False)
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def native_parakeet_tdt_tensor_names(config: ConfigValue) -> tuple[str, ...]:
    return tuple(sorted(native_parakeet_tdt_tensor_shapes(config)))


def parakeet_tdt_header_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    """Hash sorted ``name|dtype|shape`` records from a Safetensors header."""
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("Parakeet TDT inventory must be a non-empty mapping.")
    rows = []
    for name, record in sorted(inventory.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("Parakeet TDT tensor names must be non-empty.")
        if not isinstance(record, tuple) or len(record) != 2:
            raise ValueError("Invalid Parakeet TDT tensor inventory record.")
        dtype, shape = record
        dimensions = tuple(shape)
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("Parakeet TDT tensor dtype must be non-empty.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in dimensions):
            raise ValueError("Parakeet TDT inventory contains an invalid shape.")
        rows.append(f"{name}|{dtype}|{'x'.join(str(value) for value in dimensions)}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _reader_inventory(reader: SafeTensorReader | ShardedSafeTensorReader, ) -> TensorInventory:
    if isinstance(reader, SafeTensorReader):
        return {name: (reader.record(name).dtype, reader.record(name).shape) for name in reader.keys()}
    inventory: TensorInventory = {}
    by_shard: dict[Path, list[str]] = {}
    for name in reader.keys():
        by_shard.setdefault(reader.index.shard_path(name), []).append(name)
    for shard, indexed_names in sorted(
            by_shard.items(),
            key=lambda item: item[0].name,
    ):
        with SafeTensorReader(shard) as shard_reader:
            actual = set(shard_reader.keys())
            expected = set(indexed_names)
            if actual != expected:
                raise CheckpointCompatibilityError(
                    f"Parakeet TDT shard {shard.name!r} disagrees with its "
                    f"index: missing={sorted(expected - actual)!r}, "
                    f"undeclared={sorted(actual - expected)!r}.")
            for name in indexed_names:
                record = shard_reader.record(name)
                inventory[name] = (record.dtype, record.shape)
    return inventory


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def validate_published_parakeet_tdt_inventory(
    reader: SafeTensorReader | ShardedSafeTensorReader,
    *,
    source: str,
    revision: str | None,
) -> None:
    """Verify the exact official header before any tensor is assigned."""
    expected = PARAKEET_TDT_CHECKPOINTS.get(source)
    if expected is None or revision != expected["revision"]:
        return
    inventory = _reader_inventory(reader)
    state_values = sum(_product(shape) for _, shape in inventory.values())
    fingerprint = parakeet_tdt_header_fingerprint(inventory)
    errors = []
    if len(inventory) != expected["tensors"]:
        errors.append(f"tensors={len(inventory)} (expected {expected['tensors']})")
    if state_values != expected["state_values"]:
        errors.append(f"state_values={state_values} "
                      f"(expected {expected['state_values']})")
    if fingerprint != expected["header_fingerprint"]:
        errors.append(f"header_fingerprint={fingerprint} "
                      f"(expected {expected['header_fingerprint']})")
    if errors:
        raise CheckpointCompatibilityError(
            "Published Parakeet TDT checkpoint inventory verification failed: " + "; ".join(errors))


class ParakeetTDTCheckpointAdapter(CheckpointAdapter):
    """Identity-map an official/native Parakeet TDT checkpoint."""

    architecture_id = "parakeet-tdt"
    adapter_id = "parakeet-tdt-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        model_type = str(config.get("model_type", "")).lower()
        architectures = config.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        is_tdt = model_type in {
            "parakeet_tdt", "asr_parakeet_tdt"
        } and (not architectures or any(str(value) == "ParakeetForTDT" for value in architectures))
        return is_tdt and any(
            path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json") for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        return TensorPlan(
            rules=tuple(CopyTensor(name, name) for name in native_parakeet_tdt_tensor_names(config)))

    def load_assign_streaming(
        self,
        model: ParakeetForTDT,
        source: Any,
        config: ParakeetTDTConfig | Mapping[str, Any],
        *,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
        strict: bool = True,
    ) -> CheckpointCompatibilityReport:
        """Validate complete headers, then assign one tensor at a time."""
        type(self)._validate_identity()
        if not isinstance(model, ParakeetForTDT):
            raise TypeError("Parakeet TDT checkpoint target has an incompatible graph.")
        normalized = self._source(source)
        expected_shapes = native_parakeet_tdt_tensor_shapes(config)
        expected_state = model.state_dict()
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

        if isinstance(normalized, (SafeTensorReader, ShardedSafeTensorReader)):
            source_inventory = _reader_inventory(normalized)
            dtype_errors = []
            for name in loaded:
                checkpoint_dtype = source_inventory[name][0]
                target = expected_state[name]
                if target.is_floating_point():
                    valid_dtype = checkpoint_dtype in _FLOAT_CHECKPOINT_DTYPES
                else:
                    valid_dtype = checkpoint_dtype == "I64"
                if not valid_dtype:
                    dtype_errors.append(f"{name}={checkpoint_dtype} for target {target.dtype}")
        else:
            dtype_errors = []
            for name in loaded:
                checkpoint_value = normalized.get_tensor(name)
                target = expected_state[name]
                valid_dtype = (
                    checkpoint_value.is_floating_point()
                    if target.is_floating_point() else checkpoint_value.dtype == target.dtype)
                if not valid_dtype:
                    dtype_errors.append(f"{name}={checkpoint_value.dtype} for target "
                                        f"{target.dtype}")
        if dtype_errors:
            raise CheckpointCompatibilityError(
                "Parakeet TDT checkpoint contains incompatible tensor dtypes: " + "; ".join(dtype_errors[:5]))

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
        remaining = tuple(name for name, value in model.state_dict().items() if value.device.type == "meta")
        if remaining:
            raise CheckpointCompatibilityError(
                "Parakeet TDT checkpoint assignment left meta tensors: " + ", ".join(remaining[:5]))
        return report


__all__ = [
    "ParakeetTDTCheckpointAdapter",
    "native_parakeet_tdt_tensor_names",
    "native_parakeet_tdt_tensor_shapes",
    "parakeet_tdt_header_fingerprint",
    "validate_published_parakeet_tdt_inventory",
]
