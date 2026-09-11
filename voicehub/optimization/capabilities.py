"""Capability declarations for optional inference and training passes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class OptimizationMode(str, Enum):
    """Execution phase in which an optimization plan will run."""

    INFERENCE = "inference"
    TRAINING = "training"

    @classmethod
    def coerce(cls, value: OptimizationMode | str) -> OptimizationMode:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("Optimization mode must be a string or OptimizationMode.")
        try:
            return cls(value.strip().lower())
        except ValueError as error:
            raise ValueError(
                f"Unknown optimization mode {value!r}; expected inference or "
                "training.") from error


def _tokens(
    values: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"`{name}` must be an iterable of strings, not a string.")
    result = tuple(value.strip().lower() for value in values if isinstance(value, str) and value.strip())
    if len(result) != len(values):
        raise ValueError(f"`{name}` must contain non-empty strings.")
    if not result and not allow_empty:
        raise ValueError(f"`{name}` must not be empty.")
    if len(result) != len(set(result)):
        raise ValueError(f"`{name}` cannot contain duplicates.")
    return result


def normalize_optimization_kind(value: str) -> str:
    """Normalize an architecture-level optimization capability token."""
    if not isinstance(value, str):
        raise TypeError("Optimization kinds must be strings.")
    normalized = value.strip().lower().replace("_", "-")
    if normalized.startswith("."):
        normalized = normalized[1:]
    if not normalized:
        raise ValueError("Optimization kinds must be non-empty strings.")
    return normalized


def normalize_optimization_dtype(value: str) -> str:
    """Normalize common PyTorch dtype aliases used by optimization APIs."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Optimization dtypes must be non-empty strings.")
    normalized = value.strip().lower()
    if normalized.startswith("torch."):
        normalized = normalized.removeprefix("torch.")
    aliases = {
        "bf16": "bfloat16",
        "double": "float64",
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "half": "float16",
    }
    return aliases.get(normalized, normalized)


@dataclass(frozen=True)
class OptimizationContext:
    """Runtime properties against which every pass is validated."""

    mode: OptimizationMode | str
    architecture: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    streaming: bool = False
    distributed: bool = False
    persist_result: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", OptimizationMode.coerce(self.mode))
        if self.architecture is not None:
            if not isinstance(self.architecture, str) or not self.architecture.strip():
                raise ValueError("`architecture` must be a non-empty string or None.")
            object.__setattr__(
                self,
                "architecture",
                self.architecture.strip().lower().replace("_", "-"),
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("`device` must be a non-empty string.")
        object.__setattr__(self, "device", self.device.strip().lower())
        object.__setattr__(
            self,
            "dtype",
            normalize_optimization_dtype(self.dtype),
        )
        for name in ("streaming", "distributed", "persist_result"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")


def bind_registered_architecture(
    context: OptimizationContext,
    model: Any,
) -> OptimizationContext:
    """Bind a wrapper's registered native architecture to one plan context.

    Plain PyTorch modules and extension models without a registered
    VoiceHub model type remain architecture-agnostic. Registered built-
    ins and plugins are canonicalized through the architecture registry
    so aliases cannot weaken compatibility checks.
    """
    if not isinstance(context, OptimizationContext):
        raise TypeError("`context` must be an OptimizationContext.")
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if not isinstance(model_type, str) or not model_type.strip():
        return context

    from voicehub.errors import UnknownModelError
    from voicehub.registry import get_model_spec
    from voicehub.runtime import get_architecture_spec

    try:
        model_spec = get_model_spec(model_type)
    except UnknownModelError:
        return context
    if model_spec.architecture is None:
        return context
    architecture = get_architecture_spec(model_spec.architecture)
    if context.architecture is not None:
        requested = get_architecture_spec(context.architecture)
        if requested.architecture_id != architecture.architecture_id:
            raise ValueError(
                "Optimization context architecture does not match the model "
                f"({requested.architecture_id!r} != "
                f"{architecture.architecture_id!r}).")
        if context.architecture == architecture.architecture_id:
            return context
    return replace(
        context,
        architecture=architecture.architecture_id,
    )


@dataclass(frozen=True)
class OptimizationCapabilities:
    """Properties an optimization pass guarantees after transformation."""

    modes: tuple[OptimizationMode | str, ...]
    devices: tuple[str, ...] = ("cpu", "cuda", "mps")
    dtypes: tuple[str, ...] = ("float32", "float16", "bfloat16")
    streaming_safe: bool = False
    distributed_safe: bool = False
    persistent: bool = False
    reversible: bool = False
    changes_parameter_names: bool = False
    changes_topology: bool = False
    portable_export: bool = False

    def __post_init__(self) -> None:
        modes = tuple(OptimizationMode.coerce(value) for value in self.modes)
        if not modes:
            raise ValueError("Optimization capabilities need at least one mode.")
        if len(modes) != len(set(modes)):
            raise ValueError("Optimization modes cannot contain duplicates.")
        object.__setattr__(self, "modes", modes)
        object.__setattr__(
            self,
            "devices",
            _tokens(self.devices, name="devices"),
        )
        dtypes = tuple(normalize_optimization_dtype(dtype) for dtype in _tokens(self.dtypes, name="dtypes"))
        if len(dtypes) != len(set(dtypes)):
            raise ValueError("`dtypes` cannot contain duplicates.")
        object.__setattr__(self, "dtypes", dtypes)
        for name in (
                "streaming_safe",
                "distributed_safe",
                "persistent",
                "reversible",
                "changes_parameter_names",
                "changes_topology",
                "portable_export",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.portable_export and not self.persistent:
            raise ValueError("Portable optimization export requires persistent pass state.")

    @property
    def alters_parameter_topology(self) -> bool:
        """Whether optimizer routing or canonical state can become stale."""
        return self.changes_parameter_names or self.changes_topology

    def incompatibilities(
        self,
        context: OptimizationContext,
    ) -> tuple[str, ...]:
        issues = []
        device_family = context.device.partition(":")[0]
        if context.mode not in self.modes:
            issues.append(f"mode {context.mode.value!r}")
        if context.device not in self.devices and device_family not in self.devices:
            issues.append(f"device {context.device!r}")
        if context.dtype not in self.dtypes:
            issues.append(f"dtype {context.dtype!r}")
        if context.streaming and not self.streaming_safe:
            issues.append("streaming execution")
        if context.distributed and not self.distributed_safe:
            issues.append("distributed execution")
        if context.persist_result and not self.persistent:
            issues.append("persistent checkpoint output")
        return tuple(issues)
