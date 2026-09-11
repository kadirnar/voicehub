"""Transactional optimization passes over native PyTorch architectures."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable

from voicehub.optimization.capabilities import (
    OptimizationCapabilities,
    OptimizationContext,
    normalize_optimization_kind,
)
from voicehub.serialization_utils import reject_serialized_secrets

_JSON_SCALARS = (str, int, float, bool, type(None))


def canonical_json_tree(value: Any, *, path: str) -> Any:
    """Return a strict JSON tree without coercing keys or custom objects."""
    reject_serialized_secrets(value, owner=path)
    return _canonical_json_tree(value, path=path)


def _canonical_json_tree(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} contains a non-string mapping key.")
        output = {}
        for key in sorted(value):
            output[key] = _canonical_json_tree(
                value[key],
                path=f"{path}.{key}",
            )
        return output
    if isinstance(value, (tuple, list)):
        return [_canonical_json_tree(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number.")
    if isinstance(value, _JSON_SCALARS):
        return value
    raise TypeError(f"{path} contains {type(value).__name__}, which is not a strict "
                    "JSON value.")


def canonical_json_string(value: Any, *, path: str) -> str:
    """Encode a deterministic strict-JSON tree without implicit coercions."""
    normalized = canonical_json_tree(value, path=path)
    encoded = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    # The round trip is deliberate: manifests use only the exact subset
    # accepted by the corresponding checkpoint reader.
    if json.dumps(
            json.loads(encoded),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
    ) != encoded:
        raise TypeError(f"{path} is not stable across a JSON round trip.")
    return encoded


def _capability_manifest(capabilities: OptimizationCapabilities, ) -> dict[str, Any]:
    return {
        "modes": [mode.value for mode in capabilities.modes],
        "devices": list(capabilities.devices),
        "dtypes": list(capabilities.dtypes),
        "streaming_safe": capabilities.streaming_safe,
        "distributed_safe": capabilities.distributed_safe,
        "persistent": capabilities.persistent,
        "reversible": capabilities.reversible,
        "changes_parameter_names": capabilities.changes_parameter_names,
        "changes_topology": capabilities.changes_topology,
        "portable_export": capabilities.portable_export,
    }


class OptimizationError(RuntimeError):
    """Base error for pass discovery, validation, or application."""


class OptimizationCompatibilityError(ValueError, OptimizationError):
    """An optimization pass cannot satisfy the requested runtime."""


class OptimizationApplicationError(OptimizationError):
    """An optimization pass failed after a plan began."""

    def __init__(
            self,
            pass_id: str,
            cause: BaseException,
            *,
            rollback_errors: tuple[BaseException, ...] = (),
    ) -> None:
        self.pass_id = pass_id
        self.cause = cause
        self.rollback_errors = rollback_errors
        suffix = (f"; {len(rollback_errors)} rollback operation(s) also failed" if rollback_errors else "")
        super().__init__(f"Optimization pass {pass_id!r} failed: {cause}{suffix}.")


@dataclass(frozen=True)
class PassResult:
    """Output and optional reversible state from one optimization pass."""

    model: Any
    state: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model is None:
            raise ValueError("Optimization pass result must contain a model.")
        for name in ("state", "metadata"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"`{name}` must be a mapping.")
            object.__setattr__(self, name, MappingProxyType(dict(value)))


class OptimizationPass(ABC):
    """One versioned transformation of a VoiceHub-owned model graph."""

    pass_id: str
    pass_version: str
    optimization_kind: str | None = None
    requires_architecture_support: bool = False
    capabilities: OptimizationCapabilities

    @classmethod
    def validate_declaration(cls) -> None:
        for name in ("pass_id", "pass_version"):
            value = getattr(cls, name, None)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"Optimization pass {cls.__name__} must declare a "
                                f"non-empty `{name}`.")
        if not isinstance(
                getattr(cls, "capabilities", None),
                OptimizationCapabilities,
        ):
            raise TypeError(f"Optimization pass {cls.__name__} must declare "
                            "OptimizationCapabilities.")
        optimization_kind = getattr(cls, "optimization_kind", None)
        if optimization_kind is not None and (not isinstance(optimization_kind, str) or
                                              not optimization_kind.strip()):
            raise TypeError(
                f"Optimization pass {cls.__name__} must declare "
                "`optimization_kind` as a non-empty string or None.")
        if not isinstance(
                getattr(cls, "requires_architecture_support", None),
                bool,
        ):
            raise TypeError(
                f"Optimization pass {cls.__name__} must declare "
                "`requires_architecture_support` as a boolean.")
        capabilities = cls.capabilities
        if capabilities.reversible and cls.restore is OptimizationPass.restore:
            raise TypeError(f"Reversible optimization pass {cls.__name__} must override "
                            "restore().")
        if (capabilities.portable_export and
                cls.export_portable_state is OptimizationPass.export_portable_state):
            raise TypeError(
                f"Optimization pass {cls.__name__} declares portable export "
                "but does not override export_portable_state().")

    @property
    def compatibility_kind(self) -> str:
        """Return the architecture capability implemented by this pass.

        Pass IDs identify concrete implementations. Compatibility kinds
        are stable architecture-level categories such as ``compile`` or
        ``lora``; several independently versioned pass IDs may implement
        one kind.
        """
        type(self).validate_declaration()
        value = self.optimization_kind or self.pass_id
        return normalize_optimization_kind(value)

    @property
    def qualified_id(self) -> str:
        type(self).validate_declaration()
        return f"{self.pass_id}@{self.pass_version}"

    @abstractmethod
    def manifest_configuration(self) -> Mapping[str, Any]:
        """Return every option that can affect the transformed graph.

        The manager snapshots this mapping before any pass is applied.
        Implementations must therefore expose defaults as well as
        explicit caller options and must not derive values from the
        mutated model.
        """

    def runtime_manifest_status(
        self,
        result: PassResult,
    ) -> Mapping[str, Any] | None:
        """Optionally report evolving state after one application.

        Configuration and application metadata are immutable snapshots.
        Runtime state may evolve later—for example, a lazy compiler can
        select eager fallback—so passes may override this hook.
        Returning ``None`` omits the separate ``runtime_status``
        manifest field.
        """
        if not isinstance(result, PassResult):
            raise TypeError("`result` must be a PassResult.")
        return None

    def validate(self, model: Any, context: OptimizationContext) -> None:
        """Reject incompatible runtime properties without mutating
        ``model``."""
        type(self).validate_declaration()
        issues = self.capabilities.incompatibilities(context)
        if issues:
            raise OptimizationCompatibilityError(
                f"Optimization pass {self.qualified_id!r} does not support "
                f"{', '.join(issues)}.")

    def not_applicable_result(
        self,
        model: Any,
        *,
        reason: str,
    ) -> PassResult:
        """Return a reversible, explicitly reported model-preserving result."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Optimization fallback reasons must be non-empty strings.")
        return PassResult(
            model=model,
            state={
                "kind": "not-applicable",
                "model": model,
            },
            metadata={
                "outcome": "not-applicable",
                "reason": reason.strip(),
            },
        )

    @abstractmethod
    def apply(self, model: Any, context: OptimizationContext) -> PassResult:
        """Apply the pass after all plan members have validated."""

    def restore(
        self,
        model: Any,
        state: Mapping[str, Any],
        context: OptimizationContext,
    ) -> Any:
        """Undo a reversible pass.

        Reversible passes must override this method. The default refuses
        to imply reversibility from a capability declaration alone.
        """
        raise OptimizationError(f"Optimization pass {self.qualified_id!r} has no restore "
                                "implementation.")

    def route_optimizer_parameters(
        self,
        model: Any,
        *,
        optimizer_names: tuple[str, ...],
    ) -> Mapping[str, Iterable[tuple[str, Any]]]:
        """Return complete post-transform routes for a named recipe.

        A topology/name-changing training pass used with separate
        optimizers must override this method. The default fails closed
        rather than guessing from names changed by the pass.
        """
        raise OptimizationError(
            f"Optimization pass {self.qualified_id!r} has no complete "
            "optimizer-routing implementation.")

    def export_portable_state(
        self,
        model: Any,
        context: OptimizationContext,
    ) -> Mapping[str, Any]:
        """Export canonical state loadable by a fresh unoptimized runtime."""
        raise OptimizationError(
            f"Optimization pass {self.qualified_id!r} has no portable state "
            "export implementation.")


@dataclass(frozen=True)
class AppliedPass:
    """One successful step retained for reporting and restoration."""

    optimization_pass: OptimizationPass
    result: PassResult
    _manifest_json: str

    @property
    def pass_id(self) -> str:
        return self.optimization_pass.qualified_id

    def manifest_entry(self) -> dict[str, Any]:
        """Return immutable configuration plus current runtime outcome."""
        declaration = json.loads(self._manifest_json)
        runtime_status = self.optimization_pass.runtime_manifest_status(self.result, )
        if runtime_status is None:
            return declaration
        if not isinstance(runtime_status, Mapping):
            raise TypeError(f"Optimization pass {self.pass_id!r} runtime status "
                            "must be a mapping.")
        declaration["runtime_status"] = runtime_status
        return json.loads(
            canonical_json_string(
                declaration,
                path=f"optimization pass {self.pass_id!r} manifest",
            ))


@dataclass(frozen=True)
class OptimizationResult:
    """Final optimized model and ordered pass history."""

    model: Any
    context: OptimizationContext
    applied: tuple[AppliedPass, ...]

    def restore(self) -> Any:
        """Undo every pass in reverse order.

        The operation is available only when every applied pass
        explicitly guarantees reversibility.
        """
        irreversible = [
            item.pass_id for item in self.applied if not item.optimization_pass.capabilities.reversible
        ]
        if irreversible:
            raise OptimizationError(
                f"Optimization result contains irreversible passes: "
                f"{irreversible!r}.")
        model = self.model
        for item in reversed(self.applied):
            model = item.optimization_pass.restore(
                model,
                item.result.state,
                self.context,
            )
        return model

    def manifest_metadata(self) -> tuple[dict[str, Any], ...]:
        return tuple(item.manifest_entry() for item in self.applied)

    def manifest(self) -> dict[str, Any]:
        """Return a deterministic, checkpoint-safe optimization record."""
        manifest = {
            "format_version": 3,
            "context": {
                "mode": self.context.mode.value,
                "architecture": self.context.architecture,
                "device": self.context.device,
                "dtype": self.context.dtype,
                "streaming": self.context.streaming,
                "distributed": self.context.distributed,
                "persist_result": self.context.persist_result,
            },
            "passes": list(self.manifest_metadata()),
        }
        return json.loads(canonical_json_string(
            manifest,
            path="optimization manifest",
        ))

    def portable_state_dict(self, model: Any | None = None) -> Mapping[str, Any]:
        """Return state loadable by a fresh canonical model runtime.

        ``model`` may be a strategy-unwrapped execution handle. When
        omitted, the graph returned directly by the pass plan is used.
        """
        runtime = self.model if model is None else model
        topology_passes = tuple(
            item for item in self.applied if item.optimization_pass.capabilities.alters_parameter_topology)
        if not topology_passes:
            state_dict = getattr(runtime, "state_dict", None)
            if not callable(state_dict):
                raise TypeError("The optimized model does not expose state_dict().")
            result = state_dict()
        else:
            if len(topology_passes) != 1:
                raise OptimizationError(
                    "Portable export is ambiguous for more than one "
                    "topology/name-changing optimization pass.")
            applied = topology_passes[0]
            optimization_pass = applied.optimization_pass
            if not optimization_pass.capabilities.portable_export:
                raise OptimizationError(
                    f"Optimization pass {optimization_pass.qualified_id!r} "
                    "changes parameter names/topology and does not declare a "
                    "canonical portable export.")
            result = optimization_pass.export_portable_state(
                runtime,
                self.context,
            )
        if not isinstance(result, Mapping):
            raise TypeError("Portable optimization export must return a state mapping.")
        if any(not isinstance(name, str) or not name for name in result):
            raise TypeError("Portable optimization state keys must be non-empty strings.")
        return result


def snapshot_optimization_pass_declaration(optimization_pass: OptimizationPass, ) -> str:
    """Snapshot one pass declaration as deterministic strict JSON."""
    if not isinstance(optimization_pass, OptimizationPass):
        raise TypeError("`optimization_pass` must be an OptimizationPass.")
    optimization_pass.validate_declaration()
    configuration = optimization_pass.manifest_configuration()
    if not isinstance(configuration, Mapping):
        raise TypeError(
            f"Optimization pass {optimization_pass.qualified_id!r} "
            "manifest_configuration() must return a mapping.")
    return canonical_json_string(
        {
            "pass": optimization_pass.pass_id,
            "kind": optimization_pass.compatibility_kind,
            "version": optimization_pass.pass_version,
            "configuration": configuration,
            "capabilities": _capability_manifest(optimization_pass.capabilities),
        },
        path=(f"optimization pass {optimization_pass.qualified_id!r} "
              "declaration"),
    )


class OptimizationPassManager:
    """Validate an entire pass sequence, then apply it transactionally."""

    @staticmethod
    def resolve(
        passes: (str
                 | OptimizationPass
                 | Iterable[str | OptimizationPass]),
        *,
        registry: OptimizationPassRegistry | None = None,
    ) -> tuple[OptimizationPass, ...]:
        """Resolve a mixed named/instance plan at explicit application time.

        Names are resolved through the supplied registry, or the
        process-wide :data:`OPTIMIZATION_PASSES` registry. Pass
        factories remain untouched until this method is called, while
        pass instances are retained exactly as supplied.
        """
        if isinstance(passes, (str, OptimizationPass)):
            pass_specs = (passes, )
        else:
            try:
                pass_specs = tuple(passes)
            except TypeError as error:
                raise TypeError(
                    "`passes` must be a pass name, OptimizationPass, or an "
                    "iterable containing those values.") from error
        if registry is None:
            registry = OPTIMIZATION_PASSES
        if not isinstance(registry, OptimizationPassRegistry):
            raise TypeError("`registry` must be an OptimizationPassRegistry.")

        resolved = []
        for item in pass_specs:
            if isinstance(item, str):
                resolved.append(registry.create(item))
            elif isinstance(item, OptimizationPass):
                resolved.append(item)
            else:
                raise TypeError(
                    "Optimization plans may contain only pass names and "
                    "OptimizationPass instances.")
        return tuple(resolved)

    def apply_plan(
        self,
        model: Any,
        passes: (str
                 | OptimizationPass
                 | Iterable[str | OptimizationPass]),
        context: OptimizationContext,
        *,
        registry: OptimizationPassRegistry | None = None,
    ) -> OptimizationResult:
        """Resolve and transactionally apply a named/pass-object plan."""
        return self.apply(
            model,
            self.resolve(passes, registry=registry),
            context,
        )

    def apply(
        self,
        model: Any,
        passes: Iterable[OptimizationPass],
        context: OptimizationContext,
        *,
        declaration_snapshots: Iterable[str] | None = None,
    ) -> OptimizationResult:
        if not isinstance(context, OptimizationContext):
            raise TypeError("`context` must be an OptimizationContext.")
        architecture = None
        if context.architecture is not None:
            from voicehub.runtime import get_architecture_spec

            architecture = get_architecture_spec(context.architecture)
            if context.architecture != architecture.architecture_id:
                context = replace(
                    context,
                    architecture=architecture.architecture_id,
                )
        pass_sequence = tuple(passes)
        if any(not isinstance(item, OptimizationPass) for item in pass_sequence):
            raise TypeError("`passes` must contain OptimizationPass instances.")
        qualified_ids = tuple(item.qualified_id for item in pass_sequence)
        if len(qualified_ids) != len(set(qualified_ids)):
            raise ValueError("An optimization plan cannot repeat the same pass.")

        if declaration_snapshots is None:
            resolved_declaration_snapshots = tuple(
                snapshot_optimization_pass_declaration(item) for item in pass_sequence)
        else:
            resolved_declaration_snapshots = tuple(declaration_snapshots)
            if len(resolved_declaration_snapshots) != len(pass_sequence):
                raise ValueError(
                    "Optimization pass declaration snapshots must have "
                    "the same length as the pass sequence.")
            resolved_declaration_snapshots = tuple(
                self._validate_pass_declaration_snapshot(item, snapshot) for item, snapshot in zip(
                    pass_sequence,
                    resolved_declaration_snapshots,
                ))

        # Every compatibility check and configuration snapshot is deliberately
        # completed before the first model transformation.
        for item in pass_sequence:
            item.validate(model, context)
        if architecture is not None:
            self._validate_architecture_context(
                architecture,
                pass_sequence,
                context,
            )

        current = model
        applied: list[AppliedPass] = []
        for item, declaration_json in zip(
                pass_sequence,
                resolved_declaration_snapshots,
        ):
            try:
                result = item.apply(current, context)
                if not isinstance(result, PassResult):
                    raise TypeError(
                        f"{item.qualified_id} returned "
                        f"{type(result).__name__}, not PassResult.")
                current = result.model
                provisional = AppliedPass(
                    item,
                    result,
                    declaration_json,
                )
                applied.append(provisional)
                declaration = json.loads(declaration_json)
                declaration["metadata"] = result.metadata
                applied[-1] = replace(
                    provisional,
                    _manifest_json=canonical_json_string(
                        declaration,
                        path=(f"optimization pass {item.qualified_id!r} "
                              "manifest"),
                    ),
                )
                # Validate metadata before publishing the transformed graph.
                applied[-1].manifest_entry()
            except BaseException as error:
                rollback_errors = self._rollback(
                    current,
                    applied,
                    context,
                )
                raise OptimizationApplicationError(
                    item.qualified_id,
                    error,
                    rollback_errors=rollback_errors,
                ) from error
        return OptimizationResult(
            model=current,
            context=context,
            applied=tuple(applied),
        )

    @staticmethod
    def _snapshot_pass_declaration(optimization_pass: OptimizationPass, ) -> str:
        return snapshot_optimization_pass_declaration(optimization_pass, )

    @staticmethod
    def _validate_pass_declaration_snapshot(
        optimization_pass: OptimizationPass,
        snapshot: str,
    ) -> str:
        """Validate a resolver-owned snapshot without re-reading pass
        config."""
        if not isinstance(snapshot, str):
            raise TypeError("Optimization pass declaration snapshots must be strings.")
        try:
            declaration = json.loads(snapshot)
        except (TypeError, ValueError) as error:
            raise ValueError("Optimization pass declaration snapshot is not valid JSON.") from error
        canonical = canonical_json_string(
            declaration,
            path=(f"optimization pass {optimization_pass.qualified_id!r} "
                  "declaration snapshot"),
        )
        if canonical != snapshot:
            raise ValueError(
                "Optimization pass declaration snapshots must use canonical "
                "strict-JSON encoding.")
        expected = {
            "pass": optimization_pass.pass_id,
            "kind": optimization_pass.compatibility_kind,
            "version": optimization_pass.pass_version,
            "capabilities": _capability_manifest(optimization_pass.capabilities),
        }
        if not isinstance(declaration, dict):
            raise TypeError("Optimization pass declaration snapshot must contain an "
                            "object.")
        if set(declaration) != {
                "pass",
                "kind",
                "version",
                "configuration",
                "capabilities",
        }:
            raise ValueError("Optimization pass declaration snapshot has an invalid "
                             "schema.")
        for name, value in expected.items():
            if declaration[name] != value:
                raise ValueError(
                    "Optimization pass declaration snapshot no longer "
                    f"matches {name!r} for "
                    f"{optimization_pass.qualified_id!r}.")
        if not isinstance(declaration["configuration"], dict):
            raise TypeError("Optimization pass declaration configuration must contain "
                            "an object.")
        return snapshot

    @staticmethod
    def _validate_architecture_context(
        architecture: Any,
        passes: tuple[OptimizationPass, ...],
        context: OptimizationContext,
    ) -> None:
        capabilities = architecture.capabilities
        issues = []
        if not capabilities.supports_device(context.device):
            issues.append(f"device {context.device!r}")
        if not capabilities.supports_dtype(context.dtype):
            issues.append(f"dtype {context.dtype!r}")
        if context.mode.value == "training" and not capabilities.training:
            issues.append("training execution")
        if context.streaming and not capabilities.streaming:
            issues.append("streaming execution")
        if context.distributed:
            if context.mode.value == "inference":
                issues.append(
                    "distributed inference (the architecture schema verifies "
                    "distributed training only)")
            elif not capabilities.distributed_training:
                issues.append("distributed training")
        if issues:
            raise OptimizationCompatibilityError(
                f"Architecture {architecture.architecture_id!r} does not "
                f"support {', '.join(issues)}.")
        for item in passes:
            if (item.requires_architecture_support and
                    not capabilities.supports_optimization(item.compatibility_kind)):
                raise OptimizationCompatibilityError(
                    f"Architecture {architecture.architecture_id!r} does not "
                    "declare compatibility with optimization kind "
                    f"{item.compatibility_kind!r} required by pass "
                    f"{item.qualified_id!r}.")

    @staticmethod
    def _rollback(
        model: Any,
        applied: list[AppliedPass],
        context: OptimizationContext,
    ) -> tuple[BaseException, ...]:
        errors = []
        current = model
        for item in reversed(applied):
            if not item.optimization_pass.capabilities.reversible:
                errors.append(OptimizationError(f"Cannot rollback irreversible pass {item.pass_id!r}."))
                continue
            try:
                current = item.optimization_pass.restore(
                    current,
                    item.result.state,
                    context,
                )
            except BaseException as error:
                errors.append(error)
        return tuple(errors)


OptimizationPassFactory = Callable[[], OptimizationPass]


class OptimizationPassRegistry:
    """Lazy factory registry for native and optional optimization passes."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._factories: dict[str, OptimizationPassFactory] = {}

    @staticmethod
    def _name(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Optimization pass names must be non-empty strings.")
        return value.strip().lower()

    def register(
        self,
        name: str,
        factory: OptimizationPassFactory,
        *,
        exist_ok: bool = False,
    ) -> None:
        normalized = self._name(name)
        if not callable(factory):
            raise TypeError("Optimization pass factory must be callable.")
        with self._lock:
            if normalized in self._factories and not exist_ok:
                raise ValueError(f"Optimization pass {normalized!r} is already registered.")
            self._factories[normalized] = factory

    def create(self, name: str) -> OptimizationPass:
        normalized = self._name(name)
        with self._lock:
            factory = self._factories.get(normalized)
            available = tuple(sorted(self._factories))
        if factory is None:
            raise KeyError(f"Unknown optimization pass {normalized!r}; available: "
                           f"{available!r}.")
        result = factory()
        if not isinstance(result, OptimizationPass):
            raise TypeError(
                f"Optimization pass factory {normalized!r} returned "
                f"{type(result).__name__}.")
        return result

    def list(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._factories))

    def unregister(
        self,
        name: str,
        *,
        missing_ok: bool = False,
    ) -> OptimizationPassFactory | None:
        """Remove and return one factory."""
        if not isinstance(missing_ok, bool):
            raise TypeError("`missing_ok` must be a boolean.")
        normalized = self._name(name)
        with self._lock:
            try:
                return self._factories.pop(normalized)
            except KeyError:
                if missing_ok:
                    return None
                raise KeyError(f"No optimization pass {normalized!r} is registered.") from None


OPTIMIZATION_PASSES = OptimizationPassRegistry()


def register_optimization_pass(
    name: str,
    factory: OptimizationPassFactory | None = None,
    *,
    exist_ok: bool = False,
):
    """Register a lazy pass factory or decorate one.

    Explicitly registered passes are available to every speech model.
    The pass's ``validate()`` method remains responsible for checking
    whether a concrete runtime exposes the structure it needs.
    """
    if factory is None:

        def decorator(resolved_factory: OptimizationPassFactory):
            OPTIMIZATION_PASSES.register(
                name,
                resolved_factory,
                exist_ok=exist_ok,
            )
            return resolved_factory

        return decorator
    OPTIMIZATION_PASSES.register(name, factory, exist_ok=exist_ok)
    return factory


def unregister_optimization_pass(
    name: str,
    *,
    missing_ok: bool = False,
) -> OptimizationPassFactory | None:
    """Remove a process-wide optimization pass factory."""
    return OPTIMIZATION_PASSES.unregister(name, missing_ok=missing_ok)
