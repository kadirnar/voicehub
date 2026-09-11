"""Pluggable runtime strategies for speech inference.

An inference strategy owns optional runtime transformations such as
graph compilation, quantization, accelerator placement, or an external
serving engine. Model wrappers retain responsibility for loading
checkpoints and implementing their family-specific generation contract.

Strategies are registered as zero-argument factories so optional
optimization libraries remain lazy imports.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any, TypeVar

ModelT = TypeVar("ModelT")


class InferenceStrategy:
    """Lifecycle hooks for an inference optimization runtime.

    Implementations should keep :meth:`validate` side-effect free.
    VoiceHub can then reject an incompatible strategy before allocating
    model weights. :meth:`prepare` may return a wrapped or replaced
    runtime, while :meth:`restore_for_training` must return the model
    representation expected by the wrapper's training path.
    """

    name = "base"

    def validate(self, wrapper: Any) -> None:
        """Validate compatibility with a model wrapper before it is loaded."""

    def prepare(self, model: ModelT, *, wrapper: Any) -> ModelT:
        """Prepare and return a runtime for inference."""
        return model

    def restore_for_training(self, model: ModelT, *, wrapper: Any) -> ModelT:
        """Undo inference-only transformations before a training transition."""
        return model


class EagerInferenceStrategy(InferenceStrategy):
    """Default no-op strategy using each model's native eager runtime."""

    name = "eager"


class TorchCompileInferenceStrategy(InferenceStrategy):
    """Opt-in, reversible ``torch.compile`` inference preparation.

    The strategy patches executed ``forward`` methods in place instead of
    replacing the module with PyTorch's ``OptimizedModule`` wrapper. This
    keeps canonical state-dict keys intact and lets training transitions
    restore the original eager callables.

    Compilation itself remains lazy: the first real inference request pays
    PyTorch's graph compilation cost. Explicit use fails closed by default;
    pass ``requirement="auto"`` only when an eager fallback is acceptable.
    """

    name = "torch-compile"

    def __init__(
        self,
        *,
        backend: str = "inductor",
        mode: str | None = None,
        fullgraph: bool = False,
        dynamic: bool | None = True,
        options: dict[str, Any] | None = None,
        requirement: str = "required",
    ) -> None:
        from voicehub.optimization.torch_compile import TorchCompileConfig

        # Validate and freeze public options without importing or compiling a
        # model. TorchCompilePass receives the canonical values in prepare().
        self.config = TorchCompileConfig(
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            options=options,
            requirement=requirement,
        )
        self._active: dict[int, tuple[Any, Any, Any, Any]] = {}
        self._lock = RLock()

    def validate(self, wrapper: Any) -> None:
        from voicehub.optimization.passes import OptimizationCompatibilityError
        from voicehub.optimization.torch_compile import (
            TorchCompileRequirement,
            TorchCompileUnavailableError,
            inspect_torch_compile,
            torch_compile_architecture_incompatibility,
        )
        from voicehub.runtime import get_architecture_spec

        context = self._runtime_context(None, wrapper)
        architecture = (None if context.architecture is None else get_architecture_spec(context.architecture))
        architecture_issue = torch_compile_architecture_incompatibility(
            self.config,
            architecture,
            mode=context.mode,
        )
        if (architecture_issue is not None and self.config.requirement is TorchCompileRequirement.REQUIRED):
            raise OptimizationCompatibilityError(architecture_issue)

        report = inspect_torch_compile(self.config.backend)
        if (not report.available and self.config.requirement is TorchCompileRequirement.REQUIRED):
            raise TorchCompileUnavailableError(
                "Required torch.compile inference is unavailable: "
                f"{report.reason or 'unknown compiler error'}")

        requested_device = str(getattr(wrapper, "device", "cpu")).partition(":")[0]
        if requested_device == "mps":
            raise ValueError("TorchCompileInferenceStrategy supports CPU and CUDA runtimes, "
                             "not MPS.")

    @staticmethod
    def _runtime_context(model: Any, wrapper: Any):
        from voicehub.optimization.capabilities import (
            OptimizationContext,
            OptimizationMode,
            bind_registered_architecture,
        )

        parameter = None
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            parameter = next(iter(parameters()), None)
        if parameter is None:
            device = str(getattr(wrapper, "device", "cpu"))
            dtype = "float32"
        else:
            device = str(parameter.device)
            dtype = str(parameter.dtype).removeprefix("torch.")
        context = OptimizationContext(
            mode=OptimizationMode.INFERENCE,
            device=device,
            dtype=dtype,
            streaming=False,
            distributed=False,
            persist_result=False,
        )
        return bind_registered_architecture(context, wrapper)

    def prepare(self, model: ModelT, *, wrapper: Any) -> ModelT:
        from voicehub.optimization.torch_compile import TorchCompilePass

        key = id(wrapper)
        with self._lock:
            if key in self._active:
                raise RuntimeError(
                    "TorchCompileInferenceStrategy is already active for this "
                    "model wrapper.")
            optimization_pass = TorchCompilePass(
                **self.config.compile_kwargs(),
                requirement=self.config.requirement,
            )
            context = self._runtime_context(model, wrapper)
            optimization_pass.validate(model, context)
            result = optimization_pass.apply(model, context)
            self._active[key] = (
                wrapper,
                optimization_pass,
                result,
                context,
            )
            return result.model

    def restore_for_training(self, model: ModelT, *, wrapper: Any) -> ModelT:
        key = id(wrapper)
        with self._lock:
            active = self._active.pop(key, None)
            if active is None or active[0] is not wrapper:
                raise RuntimeError(
                    "TorchCompileInferenceStrategy has no active compiled "
                    "runtime for this model wrapper.")
            _, optimization_pass, result, context = active
            return optimization_pass.restore(
                model,
                result.state,
                context,
            )

    def runtime_metadata(self, wrapper: Any) -> dict[str, Any] | None:
        """Return current compile outcome metadata for diagnostics."""
        with self._lock:
            active = self._active.get(id(wrapper))
            if active is None or active[0] is not wrapper:
                return None
            _, optimization_pass, result, _ = active
            values = dict(result.metadata)
            runtime = optimization_pass.runtime_manifest_status(result)
            if runtime is not None:
                values["runtime_status"] = dict(runtime)
            return values


InferenceStrategyFactory = Callable[[], InferenceStrategy]

_BUILTIN_STRATEGY_NAME = EagerInferenceStrategy.name
_BUILTIN_INFERENCE_STRATEGIES: dict[str, InferenceStrategyFactory] = {
    _BUILTIN_STRATEGY_NAME: EagerInferenceStrategy,
    TorchCompileInferenceStrategy.name: TorchCompileInferenceStrategy,
}
_INFERENCE_STRATEGIES: dict[str, InferenceStrategyFactory] = dict(_BUILTIN_INFERENCE_STRATEGIES, )
_REGISTRY_LOCK = RLock()


def _normalize_strategy_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("Inference strategy names must be strings.")
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Inference strategy names cannot be empty.")
    return normalized


def register_inference_strategy(
    name: str,
    factory: InferenceStrategyFactory | type[InferenceStrategy],
    *,
    exist_ok: bool = False,
) -> None:
    """Register a zero-argument strategy factory.

    Factories are called only when :func:`get_inference_strategy`
    resolves the strategy, allowing implementations to import optional
    runtimes lazily. ``exist_ok=True`` intentionally replaces an
    existing custom registration. Built-in strategies cannot be
    replaced.
    """
    normalized = _normalize_strategy_name(name)
    if not callable(factory):
        raise TypeError("Inference strategy factories must be callable.")
    if isinstance(factory, type) and not issubclass(factory, InferenceStrategy):
        raise TypeError("Inference strategy classes must inherit InferenceStrategy.")

    with _REGISTRY_LOCK:
        if normalized in _BUILTIN_INFERENCE_STRATEGIES:
            if exist_ok and factory is _BUILTIN_INFERENCE_STRATEGIES[normalized]:
                return
            raise ValueError(f"The built-in {normalized!r} strategy cannot be replaced.")
        if normalized in _INFERENCE_STRATEGIES and not exist_ok:
            raise ValueError(f"An inference strategy named {normalized!r} is already registered.")
        _INFERENCE_STRATEGIES[normalized] = factory


def unregister_inference_strategy(name: str) -> None:
    """Remove a custom strategy registration.

    Built-in strategies cannot be removed.
    """
    normalized = _normalize_strategy_name(name)
    if normalized in _BUILTIN_INFERENCE_STRATEGIES:
        raise ValueError(f"The built-in {normalized!r} strategy cannot be unregistered.")

    with _REGISTRY_LOCK:
        try:
            del _INFERENCE_STRATEGIES[normalized]
        except KeyError as error:
            raise KeyError(f"No inference strategy named {normalized!r} is registered.") from error


def get_inference_strategy(strategy: str | InferenceStrategy | None = None, ) -> InferenceStrategy:
    """Resolve a strategy name or validate an existing strategy instance."""
    if strategy is None:
        strategy = _BUILTIN_STRATEGY_NAME
    if isinstance(strategy, InferenceStrategy):
        return strategy
    if not isinstance(strategy, str):
        raise TypeError("`inference_strategy` must be a name or InferenceStrategy instance.")

    normalized = _normalize_strategy_name(strategy)
    with _REGISTRY_LOCK:
        try:
            factory = _INFERENCE_STRATEGIES[normalized]
        except KeyError as error:
            available = ", ".join(sorted(_INFERENCE_STRATEGIES))
            raise KeyError(
                f"Unknown inference strategy {normalized!r}. Available strategies: {available}.") from error

    resolved = factory()
    if not isinstance(resolved, InferenceStrategy):
        raise TypeError(
            f"Inference strategy factory {normalized!r} returned "
            f"{type(resolved).__name__}, not an InferenceStrategy instance.")
    return resolved


def list_inference_strategies() -> tuple[str, ...]:
    """Return registered strategy names in deterministic order."""
    with _REGISTRY_LOCK:
        return tuple(sorted(_INFERENCE_STRATEGIES))
