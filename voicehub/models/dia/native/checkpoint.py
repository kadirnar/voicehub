"""Strict checkpoint mapping for ``nari-labs/Dia-1.6B-0626``."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing.adapters import CheckpointAdapter, CheckpointCompatibilityReport, TensorShapeMismatch
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan
from voicehub.models.dia.native.configuration import DiaArchitectureConfig
from voicehub.models.dia.native.metadata import NARI_DIA_CHECKPOINT_REVISION, NARI_DIA_HEADER_FINGERPRINT

DiaConfigInput = DiaArchitectureConfig | Mapping[str, Any]


def native_dia_tensor_shapes(config: DiaConfigInput) -> dict[str, tuple[int, ...]]:
    """Return the exact persistent tensor namespace without allocating
    weights."""
    from voicehub.models.dia.native.modeling import DiaForConditionalGeneration

    resolved = DiaArchitectureConfig.coerce(config)
    with torch.device("meta"):
        model = DiaForConditionalGeneration(resolved)
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def native_dia_tensor_names(config: DiaConfigInput) -> tuple[str, ...]:
    return tuple(sorted(native_dia_tensor_shapes(config)))


def dia_header_fingerprint(
    tensor_shapes: Mapping[str, tuple[int, ...]],
    *,
    dtype: str = "F32",
) -> str:
    if not isinstance(tensor_shapes, Mapping):
        raise TypeError("`tensor_shapes` must be a mapping.")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("`dtype` must be a non-empty string.")
    rows = []
    for name, shape in sorted(tensor_shapes.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("Dia tensor names must be non-empty.")
        if (not isinstance(shape, tuple) or
                any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
                    for dimension in shape)):
            raise ValueError(f"Invalid Dia tensor shape for {name!r}.")
        rows.append(f"{name}|{dtype}|{'x'.join(str(item) for item in shape)}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


class HuggingFaceDiaCheckpointAdapter(CheckpointAdapter):
    """Identity-map the official converted Safetensors into native Dia."""

    architecture_id = "dia"
    adapter_id = "huggingface-dia-0626-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        architectures = config.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        return (
            str(config.get("model_type", "")).lower() == "dia" and
            (not architectures or "DiaForConditionalGeneration" in architectures) and any(
                path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json")
                for path in files))

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in native_dia_tensor_names(config)))

    def load_assign_streaming(
        self,
        model: Any,
        source: Any,
        config: Mapping[str, Any],
        *,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
        strict: bool = True,
    ) -> CheckpointCompatibilityReport:
        """Validate headers, then assign one tensor at a time into a meta
        graph."""
        type(self)._validate_identity()
        normalized = self._source(source)
        expected_shapes = native_dia_tensor_shapes(config)
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
        mismatch_names = {item.name for item in mismatches}
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
        if not callable(getattr(model, "load_state_dict", None)):
            raise TypeError("Dia checkpoint target must expose load_state_dict().")
        with torch.no_grad():
            for name in report.loaded:
                value = normalized.get_tensor(name)
                target_dtype = (dtype if dtype is not None and value.is_floating_point() else value.dtype)
                value = value.to(device=device, dtype=target_dtype)
                model.load_state_dict(
                    {name: value},
                    strict=False,
                    assign=True,
                )
                del value
        remaining_meta = tuple(
            name for name, value in model.state_dict().items() if value.device.type == "meta")
        if remaining_meta:
            raise CheckpointCompatibilityError(
                "Dia checkpoint assignment left meta tensors unresolved: " + ", ".join(remaining_meta[:5]))
        # Non-persistent RoPE and channel-offset buffers are deliberately
        # materialized on CPU while a meta parameter graph is assembled.
        model.to(device=device)
        return report


HFDiaCheckpointAdapter = HuggingFaceDiaCheckpointAdapter

__all__ = [
    "HFDiaCheckpointAdapter",
    "HuggingFaceDiaCheckpointAdapter",
    "NARI_DIA_CHECKPOINT_REVISION",
    "NARI_DIA_HEADER_FINGERPRINT",
    "dia_header_fingerprint",
    "native_dia_tensor_names",
    "native_dia_tensor_shapes",
]
