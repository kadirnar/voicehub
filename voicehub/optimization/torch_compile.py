"""Optional, checkpoint-safe ``torch.compile`` optimization support.

The pass compiles execution methods instead of passing the complete
module to :func:`torch.compile`.  PyTorch's returned ``OptimizedModule``
prefixes parameter names with ``_orig_mod``; retaining original objects
and replacing only ``forward`` keeps optimizer routes and state
dictionaries canonical. Training adapters remain unwrapped while each
unique resolved component is compiled in place.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import update_wrapper
from importlib import import_module
from threading import RLock
from types import MappingProxyType
from typing import Any

from voicehub.optimization.capabilities import OptimizationCapabilities, OptimizationContext, OptimizationMode
from voicehub.optimization.passes import (
    OptimizationCompatibilityError,
    OptimizationPass,
    PassResult,
    canonical_json_string,
)
from voicehub.optimization.protocols import OptimizationCompileTarget

_COMPILE_MODES = frozenset({
    "default",
    "max-autotune",
    "max-autotune-no-cudagraphs",
    "reduce-overhead",
})
INFERENCE_COMPILE_UNSAFE_FEATURE = "torch-compile-inference-unsafe"


def _freeze_json_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json_tree(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_tree(item) for item in value)
    return value


class TorchCompileRequirement(str, Enum):
    """Whether compile failures may use the original eager callable."""

    AUTO = "auto"
    REQUIRED = "required"

    @classmethod
    def coerce(
        cls,
        value: TorchCompileRequirement | str,
    ) -> TorchCompileRequirement:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("`requirement` must be 'auto' or 'required'.")
        try:
            return cls(value.strip().lower())
        except ValueError as error:
            raise ValueError(
                f"Unknown torch.compile requirement {value!r}; expected "
                "'auto' or 'required'.") from error


class TorchCompileError(RuntimeError):
    """Base error for explicit ``torch.compile`` failures."""


class TorchCompileUnavailableError(TorchCompileError):
    """The requested compiler or backend is unavailable."""


class TorchCompileRuntimeError(TorchCompileError):
    """A required compiled callable failed during lazy compilation."""


@dataclass(frozen=True, slots=True)
class TorchCompileConfig:
    """Serializable settings passed directly to :func:`torch.compile`."""

    backend: str = "inductor"
    mode: str | None = None
    fullgraph: bool = False
    dynamic: bool | None = None
    options: Mapping[str, Any] | None = None
    requirement: TorchCompileRequirement | str = TorchCompileRequirement.AUTO
    _options_json: str | None = field(
        init=False,
        repr=False,
        compare=True,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("`backend` must be a non-empty string.")
        object.__setattr__(self, "backend", self.backend.strip())
        if self.mode is not None:
            if not isinstance(self.mode, str) or not self.mode.strip():
                raise ValueError("`mode` must be a non-empty string or None.")
            normalized_mode = self.mode.strip().lower()
            if normalized_mode not in _COMPILE_MODES:
                expected = ", ".join(sorted(_COMPILE_MODES))
                raise ValueError(
                    f"Unsupported torch.compile mode {self.mode!r}; expected "
                    f"one of: {expected}.")
            object.__setattr__(self, "mode", normalized_mode)
        if not isinstance(self.fullgraph, bool):
            raise TypeError("`fullgraph` must be a boolean.")
        if self.dynamic is not None and not isinstance(self.dynamic, bool):
            raise TypeError("`dynamic` must be a boolean or None.")
        if self.mode is not None and self.options is not None:
            raise ValueError(
                "`mode` and `options` are mutually exclusive torch.compile "
                "configuration choices.")
        if self.options is not None:
            if not isinstance(self.options, Mapping):
                raise TypeError("`options` must be a mapping or None.")
            if any(not isinstance(key, str) or not key for key in self.options):
                raise ValueError("`options` keys must be non-empty strings.")
            try:
                options_json = canonical_json_string(
                    self.options,
                    path="torch.compile options",
                )
            except (TypeError, ValueError) as error:
                raise ValueError("`options` must contain only strict JSON values: "
                                 f"{error}") from error
            object.__setattr__(
                self,
                "options",
                _freeze_json_tree(json.loads(options_json)),
            )
            object.__setattr__(self, "_options_json", options_json)
        else:
            object.__setattr__(self, "_options_json", None)
        object.__setattr__(
            self,
            "requirement",
            TorchCompileRequirement.coerce(self.requirement),
        )

    def compile_kwargs(self) -> dict[str, Any]:
        """Return the exact public ``torch.compile`` keyword arguments."""
        values = {
            "backend": self.backend,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
        }
        if self.mode is not None:
            values["mode"] = self.mode
        if self.options is not None:
            values["options"] = json.loads(self._options_json)
        return values

    def manifest(self) -> dict[str, Any]:
        """Return a strict-JSON representation used by optimization plans."""
        return {
            "backend": self.backend,
            "mode": self.mode,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "options": (None if self._options_json is None else json.loads(self._options_json)),
            "requirement": self.requirement.value,
        }


@dataclass(frozen=True, slots=True)
class TorchCompileCapabilityReport:
    """Side-effect-free availability report for one named backend."""

    available: bool
    backend: str
    backend_available: bool
    torch_version: str | None
    available_backends: tuple[str, ...]
    reason: str | None = None


def torch_compile_architecture_incompatibility(
    config: TorchCompileConfig,
    architecture: Any | None,
    *,
    mode: OptimizationMode | str,
) -> str | None:
    """Return a declared architecture/config incompatibility, if any."""
    if not isinstance(config, TorchCompileConfig):
        raise TypeError("`config` must be a TorchCompileConfig.")
    resolved_mode = OptimizationMode.coerce(mode)
    if architecture is None or resolved_mode is not OptimizationMode.INFERENCE:
        return None
    capabilities = getattr(architecture, "capabilities", None)
    has_feature = getattr(capabilities, "has_feature", None)
    if not callable(has_feature) or not has_feature(INFERENCE_COMPILE_UNSAFE_FEATURE):
        return None
    architecture_id = getattr(
        architecture,
        "architecture_id",
        type(architecture).__name__,
    )
    return (
        f"Architecture {architecture_id!r} rejects torch.compile during "
        "inference because real-checkpoint validation changed generated "
        "tokens or audio under tested compiler policies. Training "
        "compilation remains available.")


def _available_backends(torch: Any) -> tuple[str, ...]:
    dynamo = getattr(torch, "_dynamo", None)
    list_backends = getattr(dynamo, "list_backends", None)
    if callable(list_backends):
        try:
            # Passing None includes PyTorch's debug backends, notably the
            # lightweight "eager" backend used for CPU validation.
            values = list_backends(None)
        except (TypeError, ValueError):
            values = list_backends()
        return tuple(sorted({value for value in values if isinstance(value, str) and value}))
    compiler = getattr(torch, "compiler", None)
    list_backends = getattr(compiler, "list_backends", None)
    if not callable(list_backends):
        return ()
    return tuple(sorted({value for value in list_backends() if isinstance(value, str) and value}))


def inspect_torch_compile(backend: str = "inductor", ) -> TorchCompileCapabilityReport:
    """Inspect compiler/backend availability without compiling a graph."""
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("`backend` must be a non-empty string.")
    backend = backend.strip()
    try:
        torch = import_module("torch")
    except (ImportError, ModuleNotFoundError) as error:
        return TorchCompileCapabilityReport(
            available=False,
            backend=backend,
            backend_available=False,
            torch_version=None,
            available_backends=(),
            reason=f"PyTorch is unavailable: {error}",
        )
    torch_version = str(getattr(torch, "__version__", "unknown"))
    if not callable(getattr(torch, "compile", None)):
        return TorchCompileCapabilityReport(
            available=False,
            backend=backend,
            backend_available=False,
            torch_version=torch_version,
            available_backends=(),
            reason="This PyTorch runtime does not expose torch.compile().",
        )
    try:
        backends = _available_backends(torch)
    # Backend registries are third-party extension points and may raise
    # arbitrary exceptions; capability inspection must remain fail-closed.
    except Exception as error:  # noqa: BLE001
        return TorchCompileCapabilityReport(
            available=False,
            backend=backend,
            backend_available=False,
            torch_version=torch_version,
            available_backends=(),
            reason=f"torch.compile backend discovery failed: {error}",
        )
    backend_available = backend in backends
    if not backend_available:
        return TorchCompileCapabilityReport(
            available=False,
            backend=backend,
            backend_available=False,
            torch_version=torch_version,
            available_backends=backends,
            reason=f"torch.compile backend {backend!r} is unavailable.",
        )
    return TorchCompileCapabilityReport(
        available=True,
        backend=backend,
        backend_available=True,
        torch_version=torch_version,
        available_backends=backends,
    )


def _compiler_exception_types(torch: Any) -> tuple[type[BaseException], ...]:
    dynamo = getattr(torch, "_dynamo", None)
    exceptions = getattr(dynamo, "exc", None)
    root = getattr(exceptions, "TorchDynamoException", None)
    if isinstance(root, type) and issubclass(root, BaseException):
        return (root, )
    return ()


def _is_compiler_error(
    error: BaseException,
    compiler_errors: tuple[type[BaseException], ...],
) -> bool:
    if compiler_errors and isinstance(error, compiler_errors):
        return True
    module = type(error).__module__
    return module.startswith(("torch._dynamo", "torch._inductor"))


class _CompiledCall:
    """Lazy compiled callable with a local, non-global fallback policy."""

    def __init__(
        self,
        eager: Callable[..., Any],
        compiled: Callable[..., Any],
        *,
        config: TorchCompileConfig,
        compiler_errors: tuple[type[BaseException], ...],
    ) -> None:
        self._eager = eager
        self._compiled = compiled
        self._config = config
        self._compiler_errors = compiler_errors
        self._lock = RLock()
        self._using_eager = False
        self._fallback_reason: str | None = None
        self._failure_reason: str | None = None
        update_wrapper(self, eager)

    @property
    def using_eager(self) -> bool:
        """Whether a compiler failure permanently selected eager execution."""
        return self._using_eager

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason

    @property
    def failure_reason(self) -> str | None:
        """Return the latest required compiler execution failure."""
        return self._failure_reason

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._using_eager:
            return self._eager(*args, **kwargs)
        try:
            return self._compiled(*args, **kwargs)
        except BaseException as error:
            if not _is_compiler_error(error, self._compiler_errors):
                raise
            if self._config.requirement is TorchCompileRequirement.REQUIRED:
                with self._lock:
                    self._failure_reason = (f"{type(error).__name__}: {error}")
                raise TorchCompileRuntimeError(
                    "Required torch.compile execution failed for backend "
                    f"{self._config.backend!r}: {error}") from error
            with self._lock:
                self._using_eager = True
                self._fallback_reason = f"{type(error).__name__}: {error}"
            return self._eager(*args, **kwargs)


class _StateDictSafeCompiledProxy:
    """Delegate model state while routing calls through a compiled callable."""

    def __init__(
        self,
        eager_model: Any,
        compiled_call: _CompiledCall,
    ) -> None:
        object.__setattr__(self, "_voicehub_eager_model", eager_model)
        object.__setattr__(self, "_voicehub_compiled_call", compiled_call)

    @property
    def compile_runtime(self) -> _CompiledCall:
        return object.__getattribute__(self, "_voicehub_compiled_call")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        compiled = object.__getattribute__(self, "_voicehub_compiled_call")
        return compiled(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        eager = object.__getattribute__(self, "_voicehub_eager_model")
        return getattr(eager, name)


def _state_dict_keys(model: Any) -> tuple[str, ...] | None:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        return None
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("Compiled runtime state_dict() must return a mapping.")
    if any(not isinstance(name, str) or not name for name in state):
        raise TypeError("Compiled runtime state keys must be non-empty strings.")
    return tuple(state)


@dataclass(frozen=True, slots=True)
class _CompileTarget:
    owner: Any | None
    attribute: str | None
    eager: Callable[..., Any]
    label: str


def _target_manifest_binding(
    target: _CompileTarget,
    model: Any,
) -> dict[str, str]:
    owner = model if target.owner is None else target.owner
    return {
        "label": target.label,
        "owner_type": (f"{type(owner).__module__}.{type(owner).__qualname__}"),
        "attribute": ("__call__" if target.attribute is None else target.attribute),
    }


def _is_unimplemented_callable(value: Any) -> bool:
    return (
        getattr(value, "__name__", None) == "_forward_unimplemented" or
        getattr(getattr(value, "__func__", None), "__name__", None) == "_forward_unimplemented")


def _module_forward_target(
    owner: Any,
    *,
    label: str,
) -> _CompileTarget | None:
    forward = getattr(owner, "forward", None)
    if (not callable(forward) or not callable(getattr(owner, "parameters", None)) or
            not callable(getattr(owner, "state_dict", None))):
        return None
    if _is_unimplemented_callable(forward):
        return None
    return _CompileTarget(
        owner=owner,
        attribute="forward",
        eager=forward,
        label=label,
    )


def _declared_execution_targets(
    model: Any,
    context: OptimizationContext,
) -> tuple[_CompileTarget, ...] | None:
    provider = getattr(model, "optimization_compile_targets", None)
    if not callable(provider):
        return None
    entries = provider(context.mode.value)
    if isinstance(entries, (str, bytes, Mapping)) or not isinstance(
            entries,
            Iterable,
    ):
        raise TypeError(
            "optimization_compile_targets() must return an iterable of "
            "compile-target declarations.")
    targets = []
    seen_identities = set()
    seen_labels = set()
    for entry in entries:
        if isinstance(entry, OptimizationCompileTarget):
            label = entry.label
            owner = entry.owner
            attribute = entry.attribute
        elif isinstance(entry, str):
            label = entry.strip()
            owner = model
            attribute = label
        elif (isinstance(entry, (tuple, list)) and len(entry) == 3):
            label, owner, attribute = entry
        else:
            raise TypeError(
                "Compile target entries must be method names or "
                "(label, owner, attribute) triples.")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Compile target labels must be non-empty strings.")
        label = label.strip()
        if not isinstance(attribute, str) or not attribute.strip():
            raise ValueError("Compile target attributes must be non-empty strings.")
        attribute = attribute.strip()
        eager = getattr(owner, attribute, None)
        if not callable(eager):
            raise TypeError(
                f"Compile target {label!r} does not resolve to callable "
                f"{type(owner).__name__}.{attribute}.")
        if _is_unimplemented_callable(eager):
            raise TypeError(
                f"Compile target {label!r} resolves to the inherited "
                f"unimplemented {type(owner).__name__}.{attribute}.")
        identity = (id(owner), attribute)
        if identity in seen_identities:
            raise ValueError(f"Compile target {label!r} duplicates an earlier target.")
        if label in seen_labels:
            raise ValueError(f"Compile target label {label!r} duplicates an earlier "
                             "target label.")
        seen_identities.add(identity)
        seen_labels.add(label)
        targets.append(_CompileTarget(
            owner=owner,
            attribute=attribute,
            eager=eager,
            label=label,
        ))
    return tuple(targets)


def _named_execution_target(
    model: Any,
    names: tuple[str, ...],
) -> _CompileTarget | None:
    for name in names:
        method = getattr(model, name, None)
        if callable(method):
            return _CompileTarget(
                owner=model,
                attribute=name,
                eager=method,
                label=name,
            )
    return None


def _adapter_component_targets(
    model: Any,
    context: OptimizationContext,
) -> tuple[_CompileTarget, ...]:
    """Return every unique adapter-owned module execution boundary."""
    candidates = []
    primary = getattr(model, "primary_model", None)
    if primary is not None:
        candidates.append(("primary_model", primary))
    components = getattr(model, "_components", ())
    if isinstance(components, (tuple, list)):
        for entry in components:
            if (isinstance(entry, (tuple, list)) and len(entry) == 2 and isinstance(entry[0], str)):
                candidates.append((entry[0], entry[1]))

    targets = []
    seen = set()
    seen_labels = set()
    for label, candidate in candidates:
        declared = _declared_execution_targets(candidate, context)
        if declared is not None:
            candidate_targets = declared
        else:
            direct = _module_forward_target(
                candidate,
                label=f"component:{label}",
            )
            candidate_targets = () if direct is None else (direct, )
        for target in candidate_targets:
            identity = (id(target.owner), target.attribute)
            if identity in seen:
                continue
            resolved_label = (target.label if declared is None else f"component:{label}.{target.label}")
            if resolved_label in seen_labels:
                raise ValueError(
                    f"Compile target label {resolved_label!r} duplicates "
                    "an earlier adapter component target.")
            seen.add(identity)
            seen_labels.add(resolved_label)
            targets.append(
                _CompileTarget(
                    owner=target.owner,
                    attribute=target.attribute,
                    eager=target.eager,
                    label=resolved_label,
                ))
    return tuple(targets)


def _execution_targets(
    model: Any,
    context: OptimizationContext,
) -> tuple[_CompileTarget, ...]:
    declared = _declared_execution_targets(model, context)
    if declared is not None:
        return declared
    direct = _module_forward_target(model, label="forward")
    if direct is not None:
        return (direct, )
    if context.mode is OptimizationMode.TRAINING:
        components = _adapter_component_targets(model, context)
        if components:
            return components
        compute_step = getattr(model, "compute_step", None)
        if callable(compute_step):
            return (
                _CompileTarget(
                    owner=model,
                    attribute="compute_step",
                    eager=compute_step,
                    label="compute_step",
                ), )
    inferred = _named_execution_target(
        model,
        (
            "infer",
            "synthesize",
            "generate",
            "synthesize_tokens",
            "run",
            "decode",
            "sample",
        ),
    )
    if inferred is not None:
        return (inferred, )
    if (callable(model) and not (callable(getattr(model, "parameters", None)) and
                                 callable(getattr(model, "state_dict", None)))):
        return (_CompileTarget(
            owner=None,
            attribute=None,
            eager=model,
            label="__call__",
        ), )
    return ()


class TorchCompilePass(OptimizationPass):
    """Compile a model execution boundary without changing model state keys."""

    pass_id = "torch.compile"
    pass_version = "1"
    optimization_kind = "compile"
    capabilities = OptimizationCapabilities(
        modes=(OptimizationMode.INFERENCE, OptimizationMode.TRAINING),
        devices=("cpu", "cuda"),
        dtypes=("float32", "float16", "bfloat16"),
        distributed_safe=True,
        persistent=True,
        reversible=True,
        changes_parameter_names=False,
        changes_topology=False,
    )

    def __init__(
        self,
        *,
        backend: str = "inductor",
        mode: str | None = None,
        fullgraph: bool = False,
        dynamic: bool | None = None,
        options: Mapping[str, Any] | None = None,
        requirement: TorchCompileRequirement | str = TorchCompileRequirement.AUTO,
        execution_targets: (tuple[OptimizationCompileTarget, ...] | None) = None,
    ) -> None:
        self.config = TorchCompileConfig(
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            options=options,
            requirement=requirement,
        )
        if execution_targets is not None:
            execution_targets = tuple(execution_targets)
            if not execution_targets:
                raise ValueError("`execution_targets` must not be empty.")
            if any(not isinstance(target, OptimizationCompileTarget) for target in execution_targets):
                raise TypeError("`execution_targets` must contain "
                                "OptimizationCompileTarget instances.")
            identities = tuple((id(target.owner), target.attribute) for target in execution_targets)
            if len(identities) != len(set(identities)):
                raise ValueError("`execution_targets` cannot repeat an owner method.")
            labels = tuple(target.label for target in execution_targets)
            if len(labels) != len(set(labels)):
                raise ValueError("`execution_targets` cannot repeat a target label.")
            unimplemented = tuple(
                target.label for target in execution_targets
                if _is_unimplemented_callable(getattr(target.owner, target.attribute), ))
            if unimplemented:
                raise TypeError(
                    "`execution_targets` cannot include inherited "
                    "unimplemented callables: "
                    f"{unimplemented!r}.")
        self.execution_targets = execution_targets

    def manifest_configuration(self) -> Mapping[str, Any]:
        values = self.config.manifest()
        values["execution_targets"] = (
            None if self.execution_targets is None else
            [{
                "label": target.label,
                "owner_type": (f"{type(target.owner).__module__}."
                               f"{type(target.owner).__qualname__}"),
                "attribute": target.attribute,
            } for target in self.execution_targets])
        return values

    def _execution_targets(
        self,
        model: Any,
        context: OptimizationContext,
    ) -> tuple[_CompileTarget, ...]:
        if self.execution_targets is None:
            return _execution_targets(model, context)
        return tuple(
            _CompileTarget(
                owner=target.owner,
                attribute=target.attribute,
                eager=getattr(target.owner, target.attribute),
                label=target.label,
            ) for target in self.execution_targets)

    def _architecture_incompatibility(
        self,
        context: OptimizationContext,
    ) -> str | None:
        architecture = None
        if context.architecture is not None:
            from voicehub.runtime import get_architecture_spec

            architecture = get_architecture_spec(context.architecture)
        return torch_compile_architecture_incompatibility(
            self.config,
            architecture,
            mode=context.mode,
        )

    def validate(self, model: Any, context: OptimizationContext) -> None:
        targets = self._execution_targets(model, context)
        if (not targets and self.config.requirement is TorchCompileRequirement.AUTO):
            return
        super().validate(model, context)
        architecture_issue = self._architecture_incompatibility(context)
        if (architecture_issue is not None and self.config.requirement is TorchCompileRequirement.REQUIRED):
            raise OptimizationCompatibilityError(architecture_issue)
        report = inspect_torch_compile(self.config.backend)
        issues = []
        if not targets:
            issues.append(f"{type(model).__name__} has no compilable execution callable")
        if context.persist_result and not callable(getattr(model, "state_dict", None)):
            issues.append(f"{type(model).__name__} has no checkpoint state_dict()")
        if not report.available:
            issues.append(report.reason or "torch.compile is unavailable")
        if issues and self.config.requirement is TorchCompileRequirement.REQUIRED:
            raise OptimizationCompatibilityError(
                "Required torch.compile optimization is unavailable: " + "; ".join(issues) + ".")

    def _fallback_or_raise(
        self,
        model: Any,
        reason: str,
        *,
        cause: BaseException | None = None,
    ) -> PassResult:
        if self.config.requirement is TorchCompileRequirement.REQUIRED:
            error = TorchCompileUnavailableError(
                "Required torch.compile optimization could not be prepared: "
                f"{reason}.")
            if cause is not None:
                raise error from cause
            raise error
        return PassResult(
            model=model,
            state={
                "kind": "eager-fallback",
                "model": model,
            },
            metadata={
                "outcome": "eager-fallback",
                "reason": reason,
            },
        )

    def apply(self, model: Any, context: OptimizationContext) -> PassResult:
        architecture_issue = self._architecture_incompatibility(context)
        if architecture_issue is not None:
            return self._fallback_or_raise(
                model,
                architecture_issue,
            )
        report = inspect_torch_compile(self.config.backend)
        if not report.available:
            return self._fallback_or_raise(
                model,
                report.reason or "torch.compile is unavailable",
            )
        targets = self._execution_targets(model, context)
        if not targets:
            return self._fallback_or_raise(
                model,
                f"{type(model).__name__} has no compilable execution callable",
            )
        target_bindings = tuple(_target_manifest_binding(target, model) for target in targets)
        if context.persist_result and not callable(getattr(model, "state_dict", None)):
            return self._fallback_or_raise(
                model,
                f"{type(model).__name__} has no checkpoint state_dict()",
            )
        torch = import_module("torch")
        try:
            before_keys = _state_dict_keys(model)
        # Custom state_dict implementations may raise arbitrary exceptions.
        except Exception as error:  # noqa: BLE001
            return self._fallback_or_raise(
                model,
                f"cannot inspect canonical state: {error}",
                cause=error,
            )
        if len(targets) == 1 and targets[0].attribute is None:
            target = targets[0]
            try:
                compiled = torch.compile(
                    target.eager,
                    **self.config.compile_kwargs(),
                )
            # torch.compile and custom backends have no narrow error contract.
            except Exception as error:  # noqa: BLE001
                return self._fallback_or_raise(
                    model,
                    f"{type(error).__name__}: {error}",
                    cause=error,
                )
            if not callable(compiled):
                return self._fallback_or_raise(
                    model,
                    "torch.compile() returned a non-callable result",
                )
            compiled_call = _CompiledCall(
                target.eager,
                compiled,
                config=self.config,
                compiler_errors=_compiler_exception_types(torch),
            )
            transformed = _StateDictSafeCompiledProxy(
                model,
                compiled_call,
            )
            state = {
                "kind": "proxy",
                "model": model,
                "compile_runtimes": ((target.label, compiled_call), ),
            }
            target_names = (target.label, )
        else:
            patches = []
            compile_runtimes = []
            try:
                for target in targets:
                    owner = target.owner
                    attribute = target.attribute
                    if owner is None or attribute is None:
                        raise TypeError("Cannot combine callable proxies with method "
                                        "compile targets.")
                    compiled = torch.compile(
                        target.eager,
                        **self.config.compile_kwargs(),
                    )
                    if not callable(compiled):
                        raise TypeError("torch.compile() returned a non-callable result")
                    compiled_call = _CompiledCall(
                        target.eager,
                        compiled,
                        config=self.config,
                        compiler_errors=_compiler_exception_types(torch),
                    )
                    values = getattr(owner, "__dict__", {})
                    had_instance_attribute = (isinstance(values, dict) and attribute in values)
                    original_instance_value = (values.get(attribute) if had_instance_attribute else None)
                    patch = {
                        "owner": owner,
                        "attribute": attribute,
                        "had_instance_attribute": had_instance_attribute,
                        "original_instance_value": original_instance_value,
                    }
                    setattr(owner, attribute, compiled_call)
                    patches.append(patch)
                    compile_runtimes.append((target.label, compiled_call))
            # Targets and compiler backends are extensible and may raise any
            # exception; restore every installed patch before falling back.
            except Exception as error:  # noqa: BLE001
                self._restore_method_patches(patches)
                return self._fallback_or_raise(
                    model,
                    f"{type(error).__name__}: {error}",
                    cause=error,
                )
            transformed = model
            state = {
                "kind": "patched-methods",
                "model": model,
                "patches": patches,
                "compile_runtimes": tuple(compile_runtimes),
            }
            target_names = tuple(target.label for target in targets)

        try:
            after_keys = _state_dict_keys(transformed)
        except BaseException:
            self.restore(transformed, state, context)
            raise
        if before_keys != after_keys:
            self.restore(transformed, state, context)
            return self._fallback_or_raise(
                model,
                "compiled execution changed canonical state-dict keys",
            )
        return PassResult(
            model=transformed,
            state=state,
            metadata={
                "outcome": "compiled",
                "execution_targets": target_names,
                "execution_target_bindings": target_bindings,
                "state_dict_safe": True,
                "torch_version": report.torch_version,
            },
        )

    def runtime_manifest_status(
        self,
        result: PassResult,
    ) -> Mapping[str, Any] | None:
        """Report a lazy compiler fallback after it actually occurs."""
        runtimes = result.state.get("compile_runtimes", ())
        fallbacks = []
        failures = []
        for label, runtime in runtimes:
            if runtime.failure_reason is not None:
                failures.append({
                    "execution_target": label,
                    "reason": runtime.failure_reason,
                })
            if runtime.using_eager:
                fallbacks.append({
                    "execution_target":
                    label,
                    "reason":
                    runtime.fallback_reason or "lazy compiler failure selected eager execution",
                })
        if failures:
            return {
                "outcome": "compile-error",
                "errors": failures,
            }
        if fallbacks:
            return {
                "outcome":
                ("eager-fallback" if len(fallbacks) == len(runtimes) else "partial-eager-fallback"),
                "fallbacks": fallbacks,
            }
        return None

    @staticmethod
    def _restore_method_patches(patches: Any) -> None:
        if not isinstance(patches, (tuple, list)):
            raise TorchCompileError("torch.compile restoration patches must be a sequence.")
        for patch in reversed(patches):
            if not isinstance(patch, Mapping):
                raise TorchCompileError("torch.compile restoration patch must be a mapping.")
            owner = patch.get("owner")
            attribute = patch.get("attribute")
            if owner is None or not isinstance(attribute, str) or not attribute:
                raise TorchCompileError("torch.compile restoration patch is incomplete.")
            if patch.get("had_instance_attribute") is True:
                setattr(
                    owner,
                    attribute,
                    patch.get("original_instance_value"),
                )
            else:
                try:
                    delattr(owner, attribute)
                except AttributeError as error:
                    raise TorchCompileError(f"Cannot restore compiled method {attribute!r}.") from error

    def restore(
        self,
        model: Any,
        state: Mapping[str, Any],
        context: OptimizationContext,
    ) -> Any:
        del context
        kind = state.get("kind")
        eager_model = state.get("model", model)
        if kind in {"eager-fallback", "proxy"}:
            return eager_model
        if kind != "patched-methods":
            raise TorchCompileError(f"Unknown torch.compile restoration state {kind!r}.")
        self._restore_method_patches(state.get("patches"))
        return eager_model


__all__ = [
    "INFERENCE_COMPILE_UNSAFE_FEATURE",
    "TorchCompileCapabilityReport",
    "TorchCompileConfig",
    "TorchCompileError",
    "TorchCompilePass",
    "TorchCompileRequirement",
    "TorchCompileRuntimeError",
    "TorchCompileUnavailableError",
    "inspect_torch_compile",
    "torch_compile_architecture_incompatibility",
]
