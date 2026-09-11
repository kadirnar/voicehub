"""Runtime compatibility and component bundles for native architectures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from voicehub.runtime.specifications import ArchitectureError, ArchitectureSpec, normalize_capability_token
from voicehub.tasks import SpeechTask


def _normalize_dtype(value: str) -> str:
    normalized = normalize_capability_token(value, name="dtype")
    return {
        "bf16": "bfloat16",
        "double": "float64",
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "half": "float16",
    }.get(normalized, normalized)


def _normalize_tokens(
    value: Iterable[str] | str,
    *,
    name: str,
) -> tuple[str, ...]:
    items = (value, ) if isinstance(value, str) else tuple(value)
    normalized = tuple(normalize_capability_token(item, name=name) for item in items)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate values.")
    return normalized


@dataclass(frozen=True)
class RuntimeRequirements:
    """Capabilities requested for one architecture runtime."""

    task: SpeechTask | str
    device: str = "cpu"
    dtype: str = "float32"
    checkpoint_format: str | None = None
    training: bool = False
    streaming: bool = False
    batched: bool = False
    distributed: bool = False
    export_format: str | None = None
    optimization_passes: tuple[str, ...] = ()
    required_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", SpeechTask.coerce(self.task))
        object.__setattr__(
            self,
            "device",
            normalize_capability_token(self.device, name="device"),
        )
        object.__setattr__(self, "dtype", _normalize_dtype(self.dtype))

        for field_name in ("checkpoint_format", "export_format"):
            value = getattr(self, field_name)
            if value is not None:
                value = normalize_capability_token(value, name=field_name)
            object.__setattr__(self, field_name, value)

        for field_name in ("optimization_passes", "required_features"):
            object.__setattr__(
                self,
                field_name,
                _normalize_tokens(getattr(self, field_name), name=field_name),
            )
        for field_name in ("training", "streaming", "batched", "distributed"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean.")
        if self.distributed and not self.training:
            raise ValueError("distributed execution requires training=True.")


@dataclass(frozen=True)
class CompatibilityIssue:
    """One actionable mismatch between a request and an architecture."""

    capability: str
    requested: Any
    supported: Any
    detail: str

    def __post_init__(self) -> None:
        capability = normalize_capability_token(
            self.capability,
            name="capability",
        )
        object.__setattr__(self, "capability", capability)
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("Compatibility issue detail must be non-empty.")
        object.__setattr__(self, "detail", self.detail.strip())


class ArchitectureCompatibilityError(ValueError, ArchitectureError):
    """Raised when a runtime request exceeds declared architecture support."""

    def __init__(
        self,
        spec: ArchitectureSpec,
        requirements: RuntimeRequirements,
        issues: Iterable[CompatibilityIssue],
    ) -> None:
        self.spec = spec
        self.requirements = requirements
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("ArchitectureCompatibilityError requires at least one issue.")
        details = "; ".join(issue.detail for issue in self.issues)
        super().__init__(
            f"Architecture {spec.qualified_id!r} is incompatible with the "
            f"requested runtime: {details}")


def inspect_compatibility(
    spec: ArchitectureSpec,
    requirements: RuntimeRequirements,
) -> tuple[CompatibilityIssue, ...]:
    """Return every known compatibility issue without raising."""
    if not isinstance(spec, ArchitectureSpec):
        raise TypeError("spec must be an ArchitectureSpec instance.")
    if not isinstance(requirements, RuntimeRequirements):
        raise TypeError("requirements must be a RuntimeRequirements instance.")

    capabilities = spec.capabilities
    issues: list[CompatibilityIssue] = []

    if not capabilities.supports_task(requirements.task):
        supported = tuple(task.value for task in capabilities.tasks)
        issues.append(
            CompatibilityIssue(
                "task",
                requirements.task.value,
                supported,
                f"task {requirements.task.value!r} is not supported "
                f"(supported: {', '.join(supported)})",
            ))
    if not capabilities.supports_device(requirements.device):
        issues.append(
            CompatibilityIssue(
                "device",
                requirements.device,
                capabilities.devices,
                f"device {requirements.device!r} is not supported "
                f"(supported: {', '.join(capabilities.devices)})",
            ))
    if not capabilities.supports_dtype(requirements.dtype):
        issues.append(
            CompatibilityIssue(
                "dtype",
                requirements.dtype,
                capabilities.dtypes,
                f"dtype {requirements.dtype!r} is not supported "
                f"(supported: {', '.join(capabilities.dtypes)})",
            ))
    if (requirements.checkpoint_format is not None and
            not capabilities.supports_checkpoint_format(requirements.checkpoint_format)):
        issues.append(
            CompatibilityIssue(
                "checkpoint-format",
                requirements.checkpoint_format,
                capabilities.checkpoint_formats,
                f"checkpoint format {requirements.checkpoint_format!r} is "
                f"not supported (supported: "
                f"{', '.join(capabilities.checkpoint_formats)})",
            ))
    for capability_name in (
            "training",
            "streaming",
            "batched_inference",
            "distributed_training",
    ):
        request_name = {
            "batched_inference": "batched",
            "distributed_training": "distributed",
        }.get(capability_name, capability_name)
        if getattr(requirements, request_name) and not getattr(
                capabilities,
                capability_name,
        ):
            display_name = capability_name.replace("_", " ")
            issues.append(
                CompatibilityIssue(
                    capability_name,
                    True,
                    False,
                    f"{display_name} is not supported",
                ))
    if (requirements.export_format is not None and
            not capabilities.supports_export_format(requirements.export_format)):
        issues.append(
            CompatibilityIssue(
                "export-format",
                requirements.export_format,
                capabilities.export_formats,
                f"export format {requirements.export_format!r} is not "
                f"supported",
            ))
    for optimization_pass in requirements.optimization_passes:
        if not capabilities.supports_optimization(optimization_pass):
            issues.append(
                CompatibilityIssue(
                    "optimization-pass",
                    optimization_pass,
                    capabilities.optimization_passes,
                    f"optimization pass {optimization_pass!r} is not supported",
                ))
    for feature in requirements.required_features:
        if not capabilities.has_feature(feature):
            issues.append(
                CompatibilityIssue(
                    "feature",
                    feature,
                    capabilities.features,
                    f"required feature {feature!r} is not supported",
                ))
    return tuple(issues)


def ensure_compatible(
    spec: ArchitectureSpec,
    requirements: RuntimeRequirements,
) -> None:
    """Raise a structured error if *spec* cannot satisfy *requirements*."""
    issues = inspect_compatibility(spec, requirements)
    if issues:
        raise ArchitectureCompatibilityError(spec, requirements, issues)


_MISSING = object()


@dataclass(frozen=True)
class RuntimeBundle:
    """Resolved objects used by inference, training, and optimisation.

    The bundle does not prescribe a tensor framework or model call
    signature. Architecture-specific builders create the objects, while
    generic VoiceHub orchestration addresses them through these stable
    semantic names.
    """

    spec: ArchitectureSpec
    model: Any
    config: Any = None
    processor: Any = None
    decoder: Any = None
    objective: Any = None
    checkpoint_adapter: Any = None
    components: Mapping[str, Any] = field(default_factory=dict)
    requirements: RuntimeRequirements | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ArchitectureSpec):
            raise TypeError("spec must be an ArchitectureSpec instance.")
        if self.model is None:
            raise ValueError("RuntimeBundle.model must be a resolved model object.")
        if self.requirements is not None:
            ensure_compatible(self.spec, self.requirements)
        if not isinstance(self.components, Mapping):
            raise TypeError("components must be a mapping.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        normalized_components: dict[str, Any] = {}
        reserved = {
            "checkpoint-adapter",
            "config",
            "decoder",
            "model",
            "objective",
            "processor",
            "spec",
        }
        for name, component in self.components.items():
            normalized = normalize_capability_token(
                name,
                name="component name",
            )
            if normalized in reserved:
                raise ValueError(f"Runtime component name {name!r} is reserved.")
            if normalized in normalized_components:
                raise ValueError(f"Runtime components contain duplicate name {normalized!r}.")
            normalized_components[normalized] = component
        object.__setattr__(
            self,
            "components",
            MappingProxyType(normalized_components),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @property
    def architecture_id(self) -> str:
        """Return the canonical architecture identifier."""
        return self.spec.architecture_id

    @property
    def training_ready(self) -> bool:
        """Whether the architecture declares a supported training path."""
        return self.spec.capabilities.training

    @property
    def streaming_ready(self) -> bool:
        """Whether the architecture declares a supported streaming path."""
        return self.spec.capabilities.streaming

    @property
    def resolved_components(self) -> Mapping[str, Any]:
        """Return every resolved component through one read-only mapping."""
        components = {
            "model": self.model,
        }
        for name in (
                "config",
                "processor",
                "decoder",
                "objective",
                "checkpoint_adapter",
        ):
            component = getattr(self, name)
            if component is not None:
                components[name.replace("_", "-")] = component
        components.update(self.components)
        return MappingProxyType(components)

    def get_component(self, name: str, default: Any = _MISSING) -> Any:
        """Return a resolved component by semantic name."""
        normalized = normalize_capability_token(name, name="component name")
        try:
            return self.resolved_components[normalized]
        except KeyError:
            if default is not _MISSING:
                return default
            available = ", ".join(self.resolved_components)
            raise KeyError(
                f"Runtime for {self.spec.qualified_id!r} has no component "
                f"{name!r}. Available components: {available}.") from None


# Descriptive alias for callers that treat requirements as an execution plan.
RuntimeRequest = RuntimeRequirements

__all__ = [
    "ArchitectureCompatibilityError",
    "CompatibilityIssue",
    "RuntimeBundle",
    "RuntimeRequest",
    "RuntimeRequirements",
    "ensure_compatible",
    "inspect_compatibility",
]
