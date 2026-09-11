"""Declarative contracts for VoiceHub-owned model architectures.

Architecture specifications contain metadata and import references only.
In particular, constructing or registering a specification must never
import a model implementation.  This keeps discovery inexpensive and
allows a process to inspect the complete architecture catalogue without
installing or initialising every execution backend.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeVar, cast

from voicehub.tasks import SpeechTask

_T = TypeVar("_T")


class ArchitectureError(RuntimeError):
    """Base exception for the native architecture layer."""


class ComponentResolutionError(ImportError, ArchitectureError):
    """Raised when a lazily referenced architecture component cannot
    resolve."""


def normalize_architecture_id(value: str) -> str:
    """Return the canonical form of a public architecture identifier."""
    if not isinstance(value, str):
        raise TypeError("architecture_id must be a string.")
    normalized = value.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("architecture_id must be a non-empty string.")
    return normalized


def normalize_capability_token(value: str, *, name: str) -> str:
    """Normalize one open-ended capability identifier."""
    if not isinstance(value, str):
        raise TypeError(f"{name} values must be strings.")
    normalized = value.strip().lower().replace("_", "-")
    if normalized.startswith("."):
        normalized = normalized[1:]
    if not normalized:
        raise ValueError(f"{name} values must be non-empty strings.")
    return normalized


def _normalize_tokens(
    values: Iterable[str] | str,
    *,
    name: str,
) -> tuple[str, ...]:
    items = (values, ) if isinstance(values, str) else tuple(values)
    normalized = tuple(normalize_capability_token(item, name=name) for item in items)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate values.")
    return normalized


def _normalize_dtype(value: str) -> str:
    normalized = normalize_capability_token(value, name="dtype")
    aliases = {
        "bf16": "bfloat16",
        "double": "float64",
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "half": "float16",
    }
    return aliases.get(normalized, normalized)


def _normalize_dtypes(values: Iterable[str] | str) -> tuple[str, ...]:
    items = (values, ) if isinstance(values, str) else tuple(values)
    normalized = tuple(_normalize_dtype(item) for item in items)
    if len(set(normalized)) != len(normalized):
        raise ValueError("dtypes must not contain duplicate values.")
    return normalized


def _freeze_metadata(value: Any) -> Any:
    """Recursively detach mutable containers used as specification metadata."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_metadata(item) for item in value)
    return value


@dataclass(frozen=True)
class LazyComponentReference:
    """An import target that remains unresolved until explicitly requested.

    ``attribute`` may be a dotted path within the imported module.
    Python's normal module cache handles repeat imports; VoiceHub
    deliberately does not cache the resolved object separately so tests,
    plugin reloaders, and development environments retain normal import
    semantics.
    """

    module: str
    attribute: str

    def __post_init__(self) -> None:
        for field_name in ("module", "attribute"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string.")
            value = value.strip()
            if not value:
                raise ValueError(f"{field_name} must be a non-empty string.")
            if ":" in value:
                raise ValueError(
                    f"{field_name} must not contain ':'. "
                    "Use LazyComponentReference.from_path() for import paths.")
            object.__setattr__(self, field_name, value)

        segments = self.attribute.split(".")
        if any(not segment or segment == "<locals>" for segment in segments):
            raise ValueError("attribute must be a resolvable dotted attribute path.")

    @property
    def path(self) -> str:
        """Return the canonical ``module:attribute`` import path."""
        return f"{self.module}:{self.attribute}"

    @classmethod
    def from_path(cls, path: str) -> LazyComponentReference:
        """Create a reference from a ``module:attribute`` path."""
        if not isinstance(path, str):
            raise TypeError("Component import paths must be strings.")
        module, separator, attribute = path.strip().partition(":")
        if not separator or not module or not attribute:
            raise ValueError("Component import paths must use the 'module:attribute' form.")
        return cls(module=module, attribute=attribute)

    @classmethod
    def coerce(
        cls,
        value: LazyComponentReference | str,
    ) -> LazyComponentReference:
        """Normalize a reference instance or canonical import-path string."""
        if isinstance(value, cls):
            return value
        return cls.from_path(value)

    def resolve(self, expected_type: type[_T] | None = None) -> _T | Any:
        """Import and return the referenced component.

        Args:
            expected_type: Optional runtime type constraint.  It is evaluated
                only after the target resolves and therefore does not weaken
                lazy discovery.
        """
        try:
            target: Any = importlib.import_module(self.module)
        except Exception as exc:
            raise ComponentResolutionError(
                f"Could not import module {self.module!r} while resolving "
                f"{self.path!r}: {exc}") from exc

        try:
            for segment in self.attribute.split("."):
                target = getattr(target, segment)
        except AttributeError as exc:
            raise ComponentResolutionError(
                f"Module {self.module!r} does not expose component "
                f"{self.attribute!r}.") from exc

        if expected_type is not None and not isinstance(target, expected_type):
            raise ComponentResolutionError(
                f"Component {self.path!r} resolved to {type(target).__name__}, "
                f"not {expected_type.__name__}.")
        return cast(_T, target)

    def instantiate(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve the target and call it with the supplied arguments."""
        component = self.resolve()
        if not callable(component):
            raise ComponentResolutionError(
                f"Component {self.path!r} resolved successfully but is not callable.")
        return component(*args, **kwargs)


# Short aliases make architecture declarations readable while retaining the
# explicit public name in generated documentation.
LazyComponent = LazyComponentReference
LazyComponentRef = LazyComponentReference


@dataclass(frozen=True)
class ArchitectureCapabilities:
    """Execution features guaranteed by one native architecture.

    Devices, dtypes, formats, verified automatic optimizations, and
    feature flags are open string sets rather than enums. Explicit
    optimization extensions validate runtime structure directly, so
    adding one does not require changing every architecture declaration.
    """

    tasks: tuple[SpeechTask | str, ...]
    devices: tuple[str, ...] = ("cpu", )
    dtypes: tuple[str, ...] = ("float32", )
    checkpoint_formats: tuple[str, ...] = ("safetensors", )
    training: bool = False
    streaming: bool = False
    batched_inference: bool = True
    distributed_training: bool = False
    export_formats: tuple[str, ...] = ()
    optimization_passes: tuple[str, ...] = ()
    features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tasks = ((self.tasks, ) if isinstance(self.tasks, (str, SpeechTask)) else tuple(self.tasks))
        if not tasks:
            raise ValueError("tasks must contain at least one speech task.")
        resolved_tasks = tuple(SpeechTask.coerce(task) for task in tasks)
        if len(set(resolved_tasks)) != len(resolved_tasks):
            raise ValueError("tasks must not contain duplicate values.")
        object.__setattr__(self, "tasks", resolved_tasks)

        for field_name in (
                "devices",
                "checkpoint_formats",
                "export_formats",
                "optimization_passes",
                "features",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_tokens(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(self, "dtypes", _normalize_dtypes(self.dtypes))

        if not self.devices:
            raise ValueError("devices must contain at least one execution device.")
        if not self.dtypes:
            raise ValueError("dtypes must contain at least one numeric dtype.")
        if not self.checkpoint_formats:
            raise ValueError("checkpoint_formats must contain at least one checkpoint format.")
        for field_name in (
                "training",
                "streaming",
                "batched_inference",
                "distributed_training",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean.")
        if self.distributed_training and not self.training:
            raise ValueError("distributed_training requires training support to be enabled.")

    def supports_task(self, task: SpeechTask | str) -> bool:
        """Return whether the architecture implements *task*."""
        return SpeechTask.coerce(task) in self.tasks

    def supports_device(self, device: str) -> bool:
        """Return whether *device* or its indexed variant is supported."""
        normalized = normalize_capability_token(device, name="device")
        family = normalized.partition(":")[0]
        return normalized in self.devices or family in self.devices

    def supports_dtype(self, dtype: str) -> bool:
        """Return whether *dtype* is part of the verified execution surface."""
        return _normalize_dtype(dtype) in self.dtypes

    def supports_checkpoint_format(self, checkpoint_format: str) -> bool:
        """Return whether the format can be loaded by the architecture."""
        normalized = normalize_capability_token(
            checkpoint_format,
            name="checkpoint_format",
        )
        return normalized in self.checkpoint_formats

    def supports_export_format(self, export_format: str) -> bool:
        """Return whether the architecture can export to *export_format*."""
        normalized = normalize_capability_token(
            export_format,
            name="export_format",
        )
        return normalized in self.export_formats

    def supports_optimization(self, optimization_pass: str) -> bool:
        """Return whether an optimization is statically verified here."""
        normalized = normalize_capability_token(
            optimization_pass,
            name="optimization_pass",
        )
        return normalized in self.optimization_passes

    def has_feature(self, feature: str) -> bool:
        """Return whether an open-ended architecture feature is available."""
        normalized = normalize_capability_token(feature, name="feature")
        return normalized in self.features


@dataclass(frozen=True)
class ArchitectureSpec:
    """Complete declaration for one VoiceHub-owned executable architecture."""

    architecture_id: str
    model_builder: LazyComponentReference | str
    capabilities: ArchitectureCapabilities
    version: str = "1"
    config: LazyComponentReference | str | None = None
    processor: LazyComponentReference | str | None = None
    decoder: LazyComponentReference | str | None = None
    objective: LazyComponentReference | str | None = None
    checkpoint_adapter: LazyComponentReference | str | None = None
    components: Mapping[
        str,
        LazyComponentReference | str,
    ] = field(default_factory=dict)
    upstream_revision: str | None = None
    license_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "architecture_id",
            normalize_architecture_id(self.architecture_id),
        )
        if not isinstance(self.version, str):
            raise TypeError("version must be a string.")
        version = self.version.strip()
        if not version:
            raise ValueError("version must be a non-empty string.")
        object.__setattr__(self, "version", version)

        if not isinstance(self.capabilities, ArchitectureCapabilities):
            raise TypeError("capabilities must be an ArchitectureCapabilities instance.")

        object.__setattr__(
            self,
            "model_builder",
            LazyComponentReference.coerce(self.model_builder),
        )
        for field_name in (
                "config",
                "processor",
                "decoder",
                "objective",
                "checkpoint_adapter",
        ):
            value = getattr(self, field_name)
            if value is not None:
                value = LazyComponentReference.coerce(value)
            object.__setattr__(self, field_name, value)

        if not isinstance(self.components, Mapping):
            raise TypeError("components must be a mapping of names to import references.")
        reserved = {
            "checkpoint-adapter",
            "config",
            "decoder",
            "model",
            "model-builder",
            "objective",
            "processor",
        }
        normalized_components: dict[str, LazyComponentReference] = {}
        for name, reference in self.components.items():
            normalized_name = normalize_capability_token(
                name,
                name="component name",
            )
            if normalized_name in reserved:
                raise ValueError(f"Component name {name!r} is reserved by ArchitectureSpec.")
            if normalized_name in normalized_components:
                raise ValueError(f"components contains duplicate name {normalized_name!r}.")
            normalized_components[normalized_name] = (LazyComponentReference.coerce(reference))
        object.__setattr__(
            self,
            "components",
            MappingProxyType(normalized_components),
        )

        for field_name in ("upstream_revision", "license_id"):
            value = getattr(self, field_name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{field_name} must be a non-empty string or None.")
                object.__setattr__(self, field_name, value.strip())

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def name(self) -> str:
        """Alias for the stable architecture identifier."""
        return self.architecture_id

    @property
    def qualified_id(self) -> str:
        """Return the version-qualified architecture identifier."""
        return f"{self.architecture_id}@{self.version}"

    @property
    def component_references(self) -> Mapping[str, LazyComponentReference]:
        """Return a read-only view of every declared import reference."""
        references: dict[str, LazyComponentReference] = {
            "model-builder": self.model_builder,
        }
        for name in (
                "config",
                "processor",
                "decoder",
                "objective",
                "checkpoint_adapter",
        ):
            reference = getattr(self, name)
            if reference is not None:
                references[name.replace("_", "-")] = reference
        references.update(self.components)
        return MappingProxyType(references)

    def get_component_reference(self, name: str) -> LazyComponentReference:
        """Return one declared reference without importing it."""
        normalized = normalize_capability_token(name, name="component name")
        aliases = {
            "builder": "model-builder",
            "model": "model-builder",
            "model-builder": "model-builder",
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return self.component_references[normalized]
        except KeyError:
            available = ", ".join(self.component_references)
            raise KeyError(
                f"Architecture {self.qualified_id!r} has no component "
                f"{name!r}. Available components: {available}.") from None

    def resolve_component(
        self,
        name: str,
        expected_type: type[_T] | None = None,
    ) -> _T | Any:
        """Resolve one component by semantic name."""
        return self.get_component_reference(name).resolve(expected_type)

    def supports_task(self, task: SpeechTask | str) -> bool:
        """Return whether this architecture supports *task*."""
        return self.capabilities.supports_task(task)


__all__ = [
    "ArchitectureCapabilities",
    "ArchitectureError",
    "ArchitectureSpec",
    "ComponentResolutionError",
    "LazyComponent",
    "LazyComponentRef",
    "LazyComponentReference",
    "normalize_architecture_id",
    "normalize_capability_token",
]
