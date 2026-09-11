"""Strict Safetensors loading for the native Granite Speech graph."""

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
from voicehub.models.asr_granite_speech.native.configuration import GraniteSpeechArchitectureConfig
from voicehub.models.asr_granite_speech.native.metadata import GRANITE_SPEECH_CHECKPOINTS
from voicehub.models.asr_granite_speech.native.modeling import (
    GraniteSpeechForConditionalGeneration,
    materialize_granite_speech_nonpersistent_buffers,
)

_ConfigValue = (GraniteSpeechArchitectureConfig | Mapping[str, Any])
_Reader = SafeTensorReader | ShardedSafeTensorReader
_TensorInventory = dict[str, tuple[str, tuple[int, ...]]]


def _coerce_config(config: _ConfigValue, ) -> GraniteSpeechArchitectureConfig:
    return (
        config if isinstance(config, GraniteSpeechArchitectureConfig) else
        GraniteSpeechArchitectureConfig.from_dict(config))


def native_granite_speech_tensor_shapes(config: _ConfigValue, ) -> dict[str, tuple[int, ...]]:
    """Return the persistent tensor namespace without allocating storage."""
    resolved = _coerce_config(config)
    with torch.device("meta"):
        model = GraniteSpeechForConditionalGeneration(
            resolved,
            initialize=False,
        )
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def native_granite_speech_tensor_names(config: _ConfigValue, ) -> tuple[str, ...]:
    return tuple(sorted(native_granite_speech_tensor_shapes(config), ))


def _dtype_bytes(dtype: str) -> int:
    normalized = dtype.upper()
    sizes = {
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
    try:
        return sizes[normalized]
    except KeyError as error:
        raise CheckpointCompatibilityError(
            f"Unsupported Granite Speech Safetensors dtype {dtype!r}.") from error


def _reader_inventory(reader: _Reader) -> _TensorInventory:
    if isinstance(reader, SafeTensorReader):
        return {
            name: (
                reader.record(name).dtype,
                reader.record(name).shape,
            )
            for name in reader.keys()
        }
    inventory: dict[str, tuple[str, tuple[int, ...]]] = {}
    by_shard: dict[Path, list[str]] = {}
    for name in reader.keys():
        by_shard.setdefault(
            reader.index.shard_path(name),
            [],
        ).append(name)
    for shard, names in sorted(
            by_shard.items(),
            key=lambda item: item[0].name,
    ):
        with SafeTensorReader(shard) as shard_reader:
            indexed = set(names)
            actual = set(shard_reader.keys())
            if indexed != actual:
                raise CheckpointCompatibilityError(
                    f"Granite Speech shard {shard.name!r} disagrees with "
                    "its index.")
            for name in names:
                record = shard_reader.record(name)
                inventory[name] = (
                    record.dtype,
                    record.shape,
                )
    return inventory


def validate_published_granite_speech_inventory(
    reader: SafeTensorReader | ShardedSafeTensorReader,
    *,
    source: str,
    revision: str | None,
) -> None:
    """Validate immutable public-checkpoint header facts before loading."""
    expected = GRANITE_SPEECH_CHECKPOINTS.get(source)
    if expected is None or revision != expected["revision"]:
        return
    inventory = _reader_inventory(reader)
    fingerprint = granite_speech_header_fingerprint(inventory)
    parameters = sum(math_product(shape) for _, shape in inventory.values())
    total_size = sum(_dtype_bytes(dtype) * math_product(shape) for dtype, shape in inventory.values())
    errors = []
    if len(inventory) != expected["tensors"]:
        errors.append(f"tensors={len(inventory)} "
                      f"(expected {expected['tensors']})")
    if parameters != expected["parameters"]:
        errors.append(f"parameters={parameters} "
                      f"(expected {expected['parameters']})")
    if total_size != expected["total_size"]:
        errors.append(f"total_size={total_size} "
                      f"(expected {expected['total_size']})")
    if fingerprint != expected["header_fingerprint"]:
        errors.append(f"header_fingerprint={fingerprint} "
                      f"(expected {expected['header_fingerprint']})")
    if errors:
        raise CheckpointCompatibilityError(
            "Published Granite Speech checkpoint verification failed: " + "; ".join(errors))


def granite_speech_header_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    """Hash an exact tensor name/dtype/shape inventory."""
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("Granite Speech inventory must be a non-empty mapping.")
    rows = []
    for name, value in sorted(inventory.items()):
        if (not isinstance(name, str) or not name or not isinstance(value, tuple) or len(value) != 2):
            raise ValueError("Invalid Granite Speech tensor inventory record.")
        dtype, shape = value
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("Tensor inventory dtype must be non-empty.")
        dimensions = tuple(shape)
        if any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
               for dimension in dimensions):
            raise ValueError("Tensor inventory contains an invalid shape.")
        rows.append(f"{name}\t{dtype}\t" + ",".join(str(dimension) for dimension in dimensions))
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8"), ).hexdigest()


def math_product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


class GraniteSpeechCheckpointAdapter(CheckpointAdapter):
    """Identity-map official Granite Speech tensors into VoiceHub."""

    architecture_id = "granite-speech"
    adapter_id = "granite-speech-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        model_type = str(config.get("model_type", "")).lower()
        return model_type == "granite_speech" and any(
            path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json") for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        return TensorPlan(
            rules=tuple(CopyTensor(name, name) for name in native_granite_speech_tensor_names(config)), )

    def load_assign_streaming(
        self,
        model: GraniteSpeechForConditionalGeneration,
        source: Any,
        config: _ConfigValue,
        *,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
        strict: bool = True,
    ) -> CheckpointCompatibilityReport:
        """Validate every header, then assign one tensor at a time."""
        type(self)._validate_identity()
        if not isinstance(
                model,
                GraniteSpeechForConditionalGeneration,
        ):
            raise TypeError("Granite Speech checkpoint target has an incompatible graph.")
        normalized = self._source(source)
        expected_shapes = native_granite_speech_tensor_shapes(config)
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
                "Granite Speech checkpoint assignment left meta tensors: " + ", ".join(remaining[:5]))
        model.tie_weights()
        materialize_granite_speech_nonpersistent_buffers(
            model,
            device=device,
        )
        return report


__all__ = [
    "GraniteSpeechCheckpointAdapter",
    "granite_speech_header_fingerprint",
    "native_granite_speech_tensor_names",
    "native_granite_speech_tensor_shapes",
    "validate_published_granite_speech_inventory",
]
