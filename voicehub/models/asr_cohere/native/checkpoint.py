"""Strict Safetensors loading for native Cohere Transcribe."""

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
from voicehub.models.asr_cohere.native.configuration import CohereAsrConfig
from voicehub.models.asr_cohere.native.metadata import COHERE_ASR_CHECKPOINTS
from voicehub.models.asr_cohere.native.modeling import CohereAsrForConditionalGeneration

TensorInventory = dict[str, tuple[str, tuple[int, ...]]]
ConfigValue = CohereAsrConfig | Mapping[str, Any]
_FLOAT_CHECKPOINT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})


def native_cohere_asr_tensor_shapes(config: ConfigValue, ) -> dict[str, tuple[int, ...]]:
    """Return the complete persistent namespace without allocating storage."""
    resolved = CohereAsrConfig.coerce(config)
    with torch.device("meta"):
        model = CohereAsrForConditionalGeneration(
            resolved,
            initialize=False,
        )
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def native_cohere_asr_tensor_names(config: ConfigValue, ) -> tuple[str, ...]:
    return tuple(sorted(native_cohere_asr_tensor_shapes(config)))


def cohere_asr_header_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    """Hash sorted ``name|dtype|shape`` records."""
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("Cohere ASR inventory must be a non-empty mapping.")
    rows = []
    for name, record in sorted(inventory.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("Cohere ASR tensor names must be non-empty.")
        if not isinstance(record, tuple) or len(record) != 2:
            raise ValueError("Invalid Cohere ASR tensor inventory record.")
        dtype, shape = record
        dimensions = tuple(shape)
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("Cohere ASR tensor dtype must be non-empty.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in dimensions):
            raise ValueError("Cohere ASR inventory contains an invalid shape.")
        rows.append(f"{name}|{dtype}|"
                    f"{'x'.join(str(value) for value in dimensions)}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _reader_inventory(reader: SafeTensorReader | ShardedSafeTensorReader, ) -> TensorInventory:
    if isinstance(reader, SafeTensorReader):
        return {name: (reader.record(name).dtype, reader.record(name).shape) for name in reader.keys()}
    inventory: TensorInventory = {}
    by_shard: dict[Path, list[str]] = {}
    for name in reader.keys():
        by_shard.setdefault(
            reader.index.shard_path(name),
            [],
        ).append(name)
    for shard, indexed_names in sorted(by_shard.items(), key=lambda item: item[0].name):
        with SafeTensorReader(shard) as shard_reader:
            actual = set(shard_reader.keys())
            expected = set(indexed_names)
            if actual != expected:
                raise CheckpointCompatibilityError(
                    f"Cohere ASR shard {shard.name!r} disagrees with its "
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


def validate_published_cohere_asr_inventory(
    reader: SafeTensorReader | ShardedSafeTensorReader,
    *,
    source: str,
    revision: str | None,
) -> None:
    """Verify the exact official header before assigning any tensor."""
    expected = COHERE_ASR_CHECKPOINTS.get(source)
    if expected is None or revision != expected["revision"]:
        return
    inventory = _reader_inventory(reader)
    state_values = sum(_product(shape) for _, shape in inventory.values())
    data_bytes = sum(
        _product(shape) * {
            "BF16": 2,
            "F16": 2,
            "F32": 4,
            "F64": 8,
            "I64": 8,
        }.get(dtype, 0) for dtype, shape in inventory.values())
    fingerprint = cohere_asr_header_fingerprint(inventory)
    errors = []
    checks = {
        "tensors": len(inventory),
        "state_values": state_values,
        "tensor_data_bytes": data_bytes,
        "header_fingerprint": fingerprint,
    }
    for name, actual in checks.items():
        if actual != expected[name]:
            errors.append(f"{name}={actual} (expected {expected[name]})")
    if errors:
        raise CheckpointCompatibilityError(
            "Published Cohere ASR checkpoint inventory verification failed: " + "; ".join(errors))


class CohereAsrCheckpointAdapter(CheckpointAdapter):
    """Identity-map the official or VoiceHub-native graph."""

    architecture_id = "cohere-asr"
    adapter_id = "cohere-asr-safetensors"
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
        compatible = (
            model_type == "cohere_asr" and (
                not architectures or
                any(str(value) == "CohereAsrForConditionalGeneration" for value in architectures)))
        return compatible and any(
            path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json") for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        return TensorPlan(
            rules=tuple(CopyTensor(name, name) for name in native_cohere_asr_tensor_names(config)))

    def load_assign_streaming(
        self,
        model: CohereAsrForConditionalGeneration,
        source: Any,
        config: ConfigValue,
        *,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
        strict: bool = True,
    ) -> CheckpointCompatibilityReport:
        """Validate names, shapes, and dtypes before mutating the graph."""
        type(self)._validate_identity()
        if not isinstance(model, CohereAsrForConditionalGeneration):
            raise TypeError("Cohere ASR checkpoint target has an incompatible graph.")
        normalized = self._source(source)
        expected_shapes = native_cohere_asr_tensor_shapes(config)
        target_state = model.state_dict()
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
        mismatched = {value.name for value in mismatches}
        loaded = tuple(sorted(expected & available - mismatched))
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

        dtype_errors = []
        if isinstance(normalized, (SafeTensorReader, ShardedSafeTensorReader)):
            inventory = _reader_inventory(normalized)
            for name in loaded:
                checkpoint_dtype = inventory[name][0]
                target = target_state[name]
                valid = (
                    checkpoint_dtype in _FLOAT_CHECKPOINT_DTYPES
                    if target.is_floating_point() else checkpoint_dtype == "I64")
                if not valid:
                    dtype_errors.append(f"{name}={checkpoint_dtype} for target "
                                        f"{target.dtype}")
        else:
            for name in loaded:
                value = normalized.get_tensor(name)
                target = target_state[name]
                valid = (
                    value.is_floating_point() if target.is_floating_point() else value.dtype == target.dtype)
                if not valid:
                    dtype_errors.append(f"{name}={value.dtype} for target {target.dtype}")
        if dtype_errors:
            raise CheckpointCompatibilityError(
                "Cohere ASR checkpoint contains incompatible tensor dtypes: " + "; ".join(dtype_errors[:5]))

        tied_names = (
            "transf_decoder._embedding.token_embedding.weight",
            "log_softmax.mlp.layer0.weight",
        )
        tied_values: dict[str, torch.Tensor] = {}
        if all(name in report.loaded for name in tied_names):
            tied_values = {name: normalized.get_tensor(name) for name in tied_names}
            if not torch.equal(tied_values[tied_names[0]], tied_values[tied_names[1]]):
                raise CheckpointCompatibilityError("Cohere ASR tied input/output embedding tensors disagree.")

        with torch.no_grad():
            for name in report.loaded:
                value = tied_values.get(name)
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
        model.tie_weights()
        remaining = tuple(name for name, value in model.state_dict().items() if value.device.type == "meta")
        if remaining:
            raise CheckpointCompatibilityError(
                "Cohere ASR checkpoint assignment left meta tensors: " + ", ".join(remaining[:5]))
        return report


__all__ = [
    "CohereAsrCheckpointAdapter",
    "cohere_asr_header_fingerprint",
    "native_cohere_asr_tensor_names",
    "native_cohere_asr_tensor_shapes",
    "validate_published_cohere_asr_inventory",
]
