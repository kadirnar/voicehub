"""Strict, streaming Safetensors lifecycle for native VibeVoice graphs.

The published VibeVoice checkpoints already use the same parameter
namespace as the native VoiceHub modules.  This adapter therefore
performs an identity mapping, but still treats checkpoint headers as
untrusted input: every shard is reconciled with its index and every
name, shape, and dtype is validated before any model parameter is
assigned.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import (
    CheckpointAdapter,
    CheckpointCompatibilityError,
    CheckpointCompatibilityReport,
    CheckpointFormatError,
    CopyTensor,
    SafeTensorReader,
    ShardedSafeTensorReader,
    TensorPlan,
    TensorShapeMismatch,
    save_safetensors,
)
from voicehub.models.vibevoice.native.codec import VibeVoiceAcousticTokenizer
from voicehub.models.vibevoice.native.configuration import (
    VibeVoiceASRConfig,
    VibeVoiceTTSConfig,
    parse_vibevoice_config,
)
from voicehub.models.vibevoice.native.metadata import VIBEVOICE_CHECKPOINTS
from voicehub.models.vibevoice.native.modeling import (
    VibeVoiceASRForConditionalGeneration,
    VibeVoiceForConditionalGeneration,
    VibeVoiceRealtimeForConditionalGeneration,
)
from voicehub.neural.rotary import RotaryEmbedding

VibeVoiceConfig = VibeVoiceASRConfig | VibeVoiceTTSConfig
VibeVoiceModel = (
    VibeVoiceASRForConditionalGeneration
    | VibeVoiceForConditionalGeneration
    | VibeVoiceRealtimeForConditionalGeneration)
TensorInventory = dict[str, tuple[str, tuple[int, ...]]]
VibeVoiceReader = SafeTensorReader | ShardedSafeTensorReader

_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})
_INTEGER_DTYPES = {
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.uint8: "U8",
}
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


@dataclass(frozen=True, slots=True)
class VibeVoiceCheckpointInventory:
    """Validated header facts without materializing tensor payloads."""

    tensors: int
    parameters: int
    tensor_bytes: int
    dtypes: tuple[str, ...]
    header_fingerprint: str


def _coerce_config(value: VibeVoiceConfig | Mapping[str, Any]) -> VibeVoiceConfig:
    if isinstance(value, (VibeVoiceASRConfig, VibeVoiceTTSConfig)):
        return value
    return parse_vibevoice_config(value)


def build_vibevoice_model(
    config: VibeVoiceConfig | Mapping[str, Any],
    *,
    initialize: bool = True,
    device: Any = None,
    dtype: torch.dtype | None = None,
) -> VibeVoiceModel:
    """Construct the graph selected by a validated VibeVoice config."""
    resolved = _coerce_config(config)
    if isinstance(resolved, VibeVoiceASRConfig):
        return VibeVoiceASRForConditionalGeneration(
            resolved,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
    if resolved.is_streaming:
        return VibeVoiceRealtimeForConditionalGeneration(
            resolved,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
    return VibeVoiceForConditionalGeneration(
        resolved,
        initialize=initialize,
        device=device,
        dtype=dtype,
    )


def native_vibevoice_tensor_shapes(config: VibeVoiceConfig | Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return the full persistent namespace without allocating storage."""
    with torch.device("meta"):
        model = build_vibevoice_model(
            config,
            initialize=False,
        )
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def native_vibevoice_tensor_names(config: VibeVoiceConfig | Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(native_vibevoice_tensor_shapes(config)))


def vibevoice_header_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]]) -> str:
    """Hash the canonical published ``name<TAB>dtype<TAB>shape`` rows."""
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("VibeVoice inventory must be a non-empty mapping.")
    rows: list[str] = []
    for name, record in sorted(inventory.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("VibeVoice tensor names must be non-empty.")
        if not isinstance(record, tuple) or len(record) != 2:
            raise ValueError("Invalid VibeVoice tensor inventory record.")
        dtype, shape = record
        dimensions = tuple(shape)
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("VibeVoice tensor dtype must be non-empty.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in dimensions):
            raise ValueError("VibeVoice inventory contains an invalid shape.")
        rows.append(f"{name}\t{dtype}\t" + ",".join(str(value) for value in dimensions))
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reader_inventory(reader: VibeVoiceReader) -> TensorInventory:
    if isinstance(reader, SafeTensorReader):
        return {
            name: (reader.record(name).dtype, reader.record(name).shape)
            for name in reader.keys()  # noqa: SIM118 - reader is not a Mapping
        }

    indexed_by_shard: dict[Path, set[str]] = {}
    for name in reader.keys():  # noqa: SIM118 - reader is not a Mapping
        indexed_by_shard.setdefault(
            reader.index.shard_path(name).resolve(),
            set(),
        ).add(name)

    inventory: TensorInventory = {}
    for shard_path, indexed_names in sorted(
            indexed_by_shard.items(),
            key=lambda item: item[0].name,
    ):
        with SafeTensorReader(shard_path) as shard:
            actual_names = set(shard.keys())
            if actual_names != indexed_names:
                raise CheckpointCompatibilityError(
                    f"VibeVoice shard {shard_path.name!r} disagrees with its "
                    f"index: missing={sorted(indexed_names - actual_names)!r}, "
                    f"undeclared={sorted(actual_names - indexed_names)!r}.")
            for name in sorted(indexed_names):
                if name in inventory:
                    raise CheckpointCompatibilityError(
                        f"VibeVoice tensor {name!r} occurs in multiple shards.")
                record = shard.record(name)
                inventory[name] = (record.dtype, record.shape)
    if set(inventory) != set(reader.keys()):
        raise CheckpointCompatibilityError("VibeVoice sharded inventory is incomplete after reconciliation.")
    return inventory


def inspect_vibevoice_checkpoint(reader: VibeVoiceReader, ) -> VibeVoiceCheckpointInventory:
    """Inspect a coherent single or sharded checkpoint header."""
    if not isinstance(reader, (SafeTensorReader, ShardedSafeTensorReader)):
        raise TypeError("VibeVoice inspection requires a Safetensors reader.")
    inventory = _reader_inventory(reader)
    parameters = sum(prod(shape) for _, shape in inventory.values())
    try:
        tensor_bytes = sum(prod(shape) * _DTYPE_BYTES[dtype] for dtype, shape in inventory.values())
    except KeyError as error:
        raise CheckpointFormatError(f"VibeVoice checkpoint uses unknown dtype {error.args[0]!r}.") from error
    return VibeVoiceCheckpointInventory(
        tensors=len(inventory),
        parameters=parameters,
        tensor_bytes=tensor_bytes,
        dtypes=tuple(sorted({dtype
                             for dtype, _ in inventory.values()})),
        header_fingerprint=vibevoice_header_fingerprint(inventory),
    )


def _header_size(path: Path) -> int:
    try:
        with path.open("rb") as stream:
            encoded = stream.read(8)
    except OSError as error:
        raise CheckpointFormatError(f"Could not inspect VibeVoice shard {path}: {error}.") from error
    if len(encoded) != 8:
        raise CheckpointFormatError(f"VibeVoice shard {path.name!r} has no complete header length.")
    return int(struct.unpack("<Q", encoded)[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise CheckpointFormatError(f"Could not hash VibeVoice artifact {path}: {error}.") from error
    return digest.hexdigest()


def validate_published_vibevoice_inventory(
    reader: SafeTensorReader | ShardedSafeTensorReader,
    *,
    source: str,
    revision: str | None,
    verify_payload_hashes: bool = False,
) -> VibeVoiceCheckpointInventory:
    """Verify immutable official release facts before model assignment."""
    report = inspect_vibevoice_checkpoint(reader)
    expected = VIBEVOICE_CHECKPOINTS.get(source)
    if expected is None or revision != expected["revision"]:
        return report

    actual = {
        "tensors": report.tensors,
        "parameters": report.parameters,
        "tensor_bytes": report.tensor_bytes,
        "header_fingerprint": report.header_fingerprint,
    }
    errors = [
        f"{name}={value!r} (expected {expected[name]!r})" for name, value in actual.items()
        if value != expected[name]
    ]
    expected_dtype = str(expected["dtype"])
    if report.dtypes != (expected_dtype, ):
        errors.append(f"dtypes={report.dtypes!r} (expected {(expected_dtype,)!r})")

    if isinstance(reader, ShardedSafeTensorReader):
        index = reader.index.path
        if index.stat().st_size != expected["index_size"]:
            errors.append(f"index_size={index.stat().st_size} (expected {expected['index_size']})")
        if verify_payload_hashes and _sha256(index) != expected["index_sha256"]:
            errors.append("index SHA-256 differs from the published artifact")
        expected_shards = expected["shards"]
        actual_shards = {
            path.name
            for path in {
                reader.index.shard_path(name).resolve()
                for name in reader.keys()  # noqa: SIM118 - not a Mapping
            }
        }
        if actual_shards != set(expected_shards):
            errors.append(
                "shard files differ: "
                f"actual={sorted(actual_shards)!r}, "
                f"expected={sorted(expected_shards)!r}")
        for filename in sorted(actual_shards & set(expected_shards)):
            path = index.parent / filename
            facts = expected_shards[filename]
            if path.stat().st_size != facts["size"]:
                errors.append(f"{filename} size={path.stat().st_size} (expected {facts['size']})")
            if _header_size(path) != facts["header_size"]:
                errors.append(
                    f"{filename} header_size={_header_size(path)} "
                    f"(expected {facts['header_size']})")
            if verify_payload_hashes and _sha256(path) != facts["sha256"]:
                errors.append(f"{filename} SHA-256 differs")
    else:
        path = reader.path
        if path.stat().st_size != expected["size"]:
            errors.append(f"size={path.stat().st_size} (expected {expected['size']})")
        if _header_size(path) != expected["header_size"]:
            errors.append(f"header_size={_header_size(path)} (expected {expected['header_size']})")
        if verify_payload_hashes and _sha256(path) != expected["sha256"]:
            errors.append("model.safetensors SHA-256 differs")

    if errors:
        raise CheckpointCompatibilityError(
            "Published VibeVoice inventory verification failed: " + "; ".join(errors) + ".")
    return report


def _valid_dtype(checkpoint_dtype: str, target: torch.Tensor) -> bool:
    if target.is_floating_point():
        return checkpoint_dtype in _FLOAT_DTYPES
    if target.dtype == torch.bool:
        return checkpoint_dtype == "BOOL"
    return _INTEGER_DTYPES.get(target.dtype) == checkpoint_dtype


def _materialize_runtime_buffers(
    model: VibeVoiceModel,
    *,
    device: str | torch.device,
) -> None:
    """Rebuild non-persistent constants omitted from Safetensors."""
    resolved_device = torch.device(device)
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.inverse_frequency = 1.0 / (
                module.base**(
                    torch.arange(
                        0,
                        module.dimension,
                        2,
                        dtype=torch.float32,
                        device=resolved_device,
                    ) / module.dimension))
        elif isinstance(module, VibeVoiceAcousticTokenizer):
            module.fix_std = torch.tensor(
                module.config.fix_std,
                device=resolved_device,
                dtype=torch.float32,
            )


class VibeVoiceCheckpointAdapter(CheckpointAdapter):
    """Identity adapter for all three published native graphs."""

    architecture_id = "vibevoice"
    adapter_id = "vibevoice-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            parse_vibevoice_config(config)
        except (KeyError, TypeError, ValueError):
            return False
        return any(
            path.suffix == ".safetensors" or path.name == "model.safetensors.index.json" for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        return TensorPlan(
            rules=tuple(CopyTensor(name, name) for name in native_vibevoice_tensor_names(config)))

    def load_assign_streaming(
        self,
        model: VibeVoiceModel,
        reader: SafeTensorReader | ShardedSafeTensorReader,
        config: VibeVoiceConfig | Mapping[str, Any],
        *,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
        strict: bool = True,
    ) -> CheckpointCompatibilityReport:
        """Validate the complete header, then assign one tensor at a time."""
        type(self)._validate_identity()
        resolved = _coerce_config(config)
        expected_model = (
            VibeVoiceASRForConditionalGeneration if isinstance(resolved, VibeVoiceASRConfig) else (
                VibeVoiceRealtimeForConditionalGeneration
                if resolved.is_streaming else VibeVoiceForConditionalGeneration))
        if not isinstance(model, expected_model):
            raise TypeError("VibeVoice checkpoint target does not match the config graph.")
        if not isinstance(reader, (SafeTensorReader, ShardedSafeTensorReader)):
            raise TypeError("VibeVoice loading requires a strict Safetensors reader.")

        inventory = _reader_inventory(reader)
        target_state = model.state_dict()
        expected = set(target_state)
        available = set(inventory)
        missing = tuple(sorted(expected - available))
        unused = tuple(sorted(available - expected))
        mismatches = tuple(
            TensorShapeMismatch(
                name=name,
                checkpoint_shape=inventory[name][1],
                model_shape=tuple(target_state[name].shape),
            ) for name in sorted(expected & available)
            if inventory[name][1] != tuple(target_state[name].shape))
        mismatch_names = {item.name for item in mismatches}
        loaded = tuple(sorted((expected & available) - mismatch_names))
        report = CheckpointCompatibilityReport(
            architecture=self.architecture_id,
            adapter=self.qualified_id,
            loaded=loaded,
            missing=missing,
            shape_mismatches=mismatches,
            unused_sources=unused,
        )
        if strict:
            report.require_compatible()

        dtype_errors = [
            f"{name}={inventory[name][0]} for {target_state[name].dtype}" for name in loaded
            if not _valid_dtype(inventory[name][0], target_state[name])
        ]
        if dtype_errors:
            raise CheckpointCompatibilityError(
                "VibeVoice checkpoint contains incompatible dtypes: " + "; ".join(dtype_errors[:8]) + ".")

        with torch.no_grad():
            for name in loaded:
                value = reader.get_tensor(name)
                selected_dtype = (dtype if dtype is not None and value.is_floating_point() else value.dtype)
                model.load_state_dict(
                    {name: value.to(
                        device=device,
                        dtype=selected_dtype,
                    )},
                    strict=False,
                    assign=True,
                )
                del value
        _materialize_runtime_buffers(model, device=device)
        remaining = tuple(
            name for name, value in (
                *model.named_parameters(),
                *model.named_buffers(),
            ) if value.device.type == "meta")
        if remaining:
            raise CheckpointCompatibilityError(
                "VibeVoice checkpoint assignment left meta tensors: " + ", ".join(remaining[:8]) + ".")
        return report


def export_vibevoice_checkpoint(
    model: VibeVoiceModel,
    path: str | Path,
) -> Path:
    """Write a complete portable native checkpoint without pickle."""
    supported_models = (
        VibeVoiceASRForConditionalGeneration,
        VibeVoiceForConditionalGeneration,
        VibeVoiceRealtimeForConditionalGeneration,
    )
    if not isinstance(model, supported_models):
        raise TypeError("VibeVoice export requires a native VibeVoice model.")
    state = dict(model.state_dict())
    expected = set(native_vibevoice_tensor_names(model.config))
    if set(state) != expected:
        raise ValueError(
            "VibeVoice export state is incomplete: "
            f"missing={sorted(expected - set(state))[:8]!r}, "
            f"unexpected={sorted(set(state) - expected)[:8]!r}.")
    meta = [name for name, value in state.items() if value.device.type == "meta"]
    if meta:
        raise ValueError("VibeVoice export cannot serialize meta tensors: " + ", ".join(meta[:8]) + ".")
    return save_safetensors(
        state,
        path,
        metadata={
            "format": "pt",
            "voicehub_architecture": model.config.model_type,
        },
    ).resolve()


__all__ = [
    "VibeVoiceCheckpointAdapter",
    "VibeVoiceCheckpointInventory",
    "build_vibevoice_model",
    "export_vibevoice_checkpoint",
    "inspect_vibevoice_checkpoint",
    "native_vibevoice_tensor_names",
    "native_vibevoice_tensor_shapes",
    "validate_published_vibevoice_inventory",
    "vibevoice_header_fingerprint",
]
