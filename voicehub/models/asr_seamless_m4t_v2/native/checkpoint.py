"""Strict S2T subset loading for SeamlessM4T-v2 Safetensors."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import (
    CheckpointAdapter,
    CheckpointCompatibilityReport,
    CopyTensor,
    SafeTensorReader,
    ShardedSafeTensorReader,
    TensorPlan,
    TensorShapeMismatch,
)
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.asr_seamless_m4t_v2.native.configuration import SeamlessM4Tv2S2TConfig
from voicehub.models.asr_seamless_m4t_v2.native.metadata import SEAMLESS_M4T_V2_CHECKPOINTS
from voicehub.models.asr_seamless_m4t_v2.native.modeling import SeamlessM4Tv2ForSpeechToText

TensorReader = SafeTensorReader | ShardedSafeTensorReader
TensorInventory = dict[str, tuple[str, tuple[int, ...]]]
ConfigValue = SeamlessM4Tv2S2TConfig | Mapping[str, Any]

_ALIASES = frozenset({
    "lm_head.weight",
    "text_decoder.embed_tokens.weight",
})
_PERSISTED_PREFIXES = (
    "shared.",
    "speech_encoder.",
    "text_decoder.",
)
_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})
_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
}


def native_seamless_m4t_v2_tensor_shapes(config: ConfigValue, ) -> dict[str, tuple[int, ...]]:
    """Return the 1,429-tensor S2T namespace without allocating storage."""
    resolved = SeamlessM4Tv2S2TConfig.coerce(config)
    with torch.device("meta"):
        model = SeamlessM4Tv2ForSpeechToText(
            resolved,
            initialize=False,
        )
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items() if name not in _ALIASES}


def native_seamless_m4t_v2_tensor_names(config: ConfigValue, ) -> tuple[str, ...]:
    return tuple(sorted(native_seamless_m4t_v2_tensor_shapes(config)))


def seamless_m4t_v2_header_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    """Hash canonical name, Safetensors dtype, and comma-delimited shape
    rows."""
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("SeamlessM4T-v2 inventory must be a non-empty mapping.")
    rows = []
    for name, record in sorted(inventory.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("Tensor names must be non-empty strings.")
        if not isinstance(record, tuple) or len(record) != 2:
            raise ValueError("Invalid tensor inventory record.")
        dtype, shape_value = record
        shape = tuple(shape_value)
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("Tensor inventory dtypes must be non-empty.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape):
            raise ValueError("Tensor inventory contains an invalid shape.")
        rows.append(f"{name}\t{dtype}\t" + ",".join(str(value) for value in shape))
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8"), ).hexdigest()


def _reader_inventory(reader: TensorReader) -> TensorInventory:
    if isinstance(reader, SafeTensorReader):
        return {name: (reader.record(name).dtype, reader.record(name).shape) for name in reader.keys()}
    if not isinstance(reader, ShardedSafeTensorReader):
        raise TypeError("SeamlessM4T-v2 requires a Safetensors reader.")
    inventory: TensorInventory = {}
    by_shard: dict[Path, list[str]] = {}
    for name in reader.keys():
        by_shard.setdefault(
            reader.index.shard_path(name),
            [],
        ).append(name)
    for shard, indexed_names in sorted(
            by_shard.items(),
            key=lambda item: item[0].name,
    ):
        with SafeTensorReader(shard) as shard_reader:
            indexed = set(indexed_names)
            actual = set(shard_reader.keys())
            if indexed != actual:
                raise CheckpointCompatibilityError(
                    f"SeamlessM4T-v2 shard {shard.name!r} disagrees with "
                    f"its index: missing={sorted(indexed - actual)!r}, "
                    f"undeclared={sorted(actual - indexed)!r}.")
            for name in indexed_names:
                record = shard_reader.record(name)
                inventory[name] = (record.dtype, record.shape)
    return inventory


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def _inventory_facts(inventory: TensorInventory, ) -> tuple[int, int, int, str, set[str]]:
    parameters = sum(_product(shape) for _, shape in inventory.values())
    try:
        tensor_bytes = sum(_DTYPE_BYTES[dtype] * _product(shape) for dtype, shape in inventory.values())
    except KeyError as error:
        raise CheckpointCompatibilityError(
            "SeamlessM4T-v2 checkpoint uses unsupported Safetensors "
            f"dtype {error.args[0]!r}.") from error
    return (
        len(inventory),
        parameters,
        tensor_bytes,
        seamless_m4t_v2_header_fingerprint(inventory),
        {dtype
         for dtype, _ in inventory.values()},
    )


def validate_published_seamless_m4t_v2_inventory(
    reader: TensorReader,
    *,
    source: str,
    revision: str | None,
) -> bool:
    """Verify the immutable full checkpoint and its exact S2T projection.

    Returns ``True`` only when the reader is the audited full unified
    checkpoint. Portable S2T-only artifacts return ``False`` and are
    validated against the configured native graph by the adapter.
    """
    expected = SEAMLESS_M4T_V2_CHECKPOINTS.get(source)
    if expected is None or revision != expected["revision"]:
        return False
    inventory = _reader_inventory(reader)
    full_facts = _inventory_facts(inventory)
    subset = {
        name: record
        for name, record in inventory.items() if name.startswith(_PERSISTED_PREFIXES) and name not in _ALIASES
    }
    subset_facts = _inventory_facts(subset)
    checks = (
        ("full_tensor_count", full_facts[0], expected["full_tensor_count"]),
        (
            "full_parameter_count",
            full_facts[1],
            expected["full_parameter_count"],
        ),
        ("full_tensor_bytes", full_facts[2], expected["full_tensor_bytes"]),
        (
            "full_header_fingerprint",
            full_facts[3],
            expected["full_header_fingerprint"],
        ),
        ("s2t_tensor_count", subset_facts[0], expected["s2t_tensor_count"]),
        (
            "s2t_parameter_count",
            subset_facts[1],
            expected["s2t_parameter_count"],
        ),
        ("s2t_tensor_bytes", subset_facts[2], expected["s2t_tensor_bytes"]),
        (
            "s2t_header_fingerprint",
            subset_facts[3],
            expected["s2t_header_fingerprint"],
        ),
    )
    errors = [f"{name}={actual} (expected {wanted})" for name, actual, wanted in checks if actual != wanted]
    if full_facts[4] != {expected["dtype"]}:
        errors.append(f"dtypes={sorted(full_facts[4])!r} "
                      f"(expected {[expected['dtype']]!r})")
    if errors:
        raise CheckpointCompatibilityError(
            "Published SeamlessM4T-v2 checkpoint verification failed: " + "; ".join(errors))
    return True


class SeamlessM4Tv2S2TCheckpointAdapter(CheckpointAdapter):
    """Identity-map only the audited speech-to-text parameter subset."""

    architecture_id = "seamless-m4t-v2-s2t"
    adapter_id = "seamless-m4t-v2-s2t-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        model_type = str(config.get("model_type", "")).lower()
        return model_type == "seamless_m4t_v2" and any(
            path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json") for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        return TensorPlan(
            rules=tuple(CopyTensor(name, name) for name in native_seamless_m4t_v2_tensor_names(config)), )

    def load_assign_streaming(
        self,
        model: SeamlessM4Tv2ForSpeechToText,
        source: Any,
        config: ConfigValue,
        *,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
        strict: bool = True,
        allow_verified_full_checkpoint: bool = False,
    ) -> CheckpointCompatibilityReport:
        """Preflight all headers, then materialize one S2T tensor at a time."""
        type(self)._validate_identity()
        if not isinstance(model, SeamlessM4Tv2ForSpeechToText):
            raise TypeError("Checkpoint target has an incompatible S2T graph.")
        if dtype is not None and (not isinstance(dtype, torch.dtype) or not dtype.is_floating_point):
            raise TypeError("Execution `dtype` must be floating-point.")
        normalized = self._source(source)
        expected_shapes = native_seamless_m4t_v2_tensor_shapes(config)
        expected = set(expected_shapes)
        available = set(normalized.keys())
        missing = tuple(sorted(expected - available))
        unused = tuple(sorted(available - expected))
        shape_method = getattr(normalized, "tensor_shape", None)
        mismatches = []
        for name in sorted(expected & available):
            checkpoint_shape = (
                tuple(shape_method(name)) if callable(shape_method) else tuple(
                    normalized.get_tensor(name).shape))
            if checkpoint_shape != expected_shapes[name]:
                mismatches.append(
                    TensorShapeMismatch(
                        name=name,
                        checkpoint_shape=checkpoint_shape,
                        model_shape=expected_shapes[name],
                    ))
        mismatched_names = {value.name for value in mismatches}
        loaded = tuple(sorted(expected & available - mismatched_names))
        ignored = unused if allow_verified_full_checkpoint else ()
        report = CheckpointCompatibilityReport(
            architecture=self.architecture_id,
            adapter=self.qualified_id,
            loaded=loaded,
            missing=missing,
            shape_mismatches=tuple(mismatches),
            unused_sources=() if allow_verified_full_checkpoint else unused,
            ignored_sources=ignored,
        )
        if strict:
            report.require_compatible()

        target_state = model.state_dict()
        dtype_errors = []
        inventory = (
            _reader_inventory(normalized) if isinstance(
                normalized,
                (SafeTensorReader, ShardedSafeTensorReader),
            ) else None)
        for name in loaded:
            target = target_state[name]
            if inventory is not None:
                source_dtype = inventory[name][0]
                valid = source_dtype in _FLOAT_DTYPES
                description = source_dtype
            else:
                value = normalized.get_tensor(name)
                valid = (
                    isinstance(value, torch.Tensor) and value.layout is torch.strided and
                    value.is_floating_point() and not value.is_quantized and not value.is_complex() and
                    value.device.type != "meta")
                description = getattr(value, "dtype", type(value).__name__)
            if not target.is_floating_point() or not valid:
                dtype_errors.append(f"{name}={description} for target {target.dtype}")
        if dtype_errors:
            raise CheckpointCompatibilityError(
                "SeamlessM4T-v2 checkpoint contains incompatible tensors: " + "; ".join(dtype_errors[:5]))

        with torch.no_grad():
            for name in loaded:
                value = normalized.get_tensor(name)
                target_dtype = dtype if dtype is not None else value.dtype
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
                "SeamlessM4T-v2 assignment left meta tensors: " + ", ".join(remaining[:5]))
        return report


__all__ = [
    "SeamlessM4Tv2S2TCheckpointAdapter",
    "native_seamless_m4t_v2_tensor_names",
    "native_seamless_m4t_v2_tensor_shapes",
    "seamless_m4t_v2_header_fingerprint",
    "validate_published_seamless_m4t_v2_inventory",
]
