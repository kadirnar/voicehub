"""Capability-driven optimization policy for every registered TTS model.

This module is the high-level counterpart to the transactional pass API
in ``voicehub.optimization.passes``.  It follows the same separation
used by Transformers: a serializable user configuration, architecture
capability declarations, an extensible implementation registry, and an
inspectable resolution step with safe native fallbacks.

The resolver never assumes that every TTS graph has interchangeable
attention or activation semantics.  FlashAttention-4 and custom kernels
are selected only for architectures that explicitly declare the
corresponding capability; ``torch.compile`` is selected independently
only when real-checkpoint inference validation explicitly permits
automatic selection, and always remains reversible.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from importlib import import_module
from typing import TYPE_CHECKING, Any

from voicehub.optimization.capabilities import OptimizationContext, OptimizationMode
from voicehub.optimization.passes import (
    OPTIMIZATION_PASSES,
    OptimizationPass,
    OptimizationPassRegistry,
    canonical_json_string,
    snapshot_optimization_pass_declaration,
)
from voicehub.optimization.torch_compile import (
    TorchCompileConfig,
    TorchCompilePass,
    TorchCompileRequirement,
    torch_compile_architecture_incompatibility,
)
from voicehub.tasks import SpeechTask

_INFERENCE_AUTO_COMPILE_VALIDATED_FEATURE = ("torch-compile-inference-auto-validated")

if TYPE_CHECKING:
    from voicehub.optimization.diffusion_cache import DiffusionCacheConfig, DiffusionCachePolicy
    from voicehub.optimization.diffusion_sampling import DiffusionSamplingConfig, DiffusionSamplingPolicy
    from voicehub.optimization.passes import OptimizationResult
    from voicehub.registry import ModelSpec
    from voicehub.runtime.specifications import ArchitectureSpec


class TTSOptimizationError(RuntimeError):
    """Base failure raised by the universal TTS optimization policy."""


class TTSOptimizationCompatibilityError(
        ValueError,
        TTSOptimizationError,
):
    """A requested implementation is not valid for a model or context."""


class TTSAttentionImplementation(str, Enum):
    """Attention policies understood by the built-in TTS resolver."""

    AUTO = "auto"
    NATIVE = "native"
    SDPA = "sdpa"
    FLASH_ATTENTION_4 = "flash_attention_4"

    @classmethod
    def coerce(
        cls,
        value: TTSAttentionImplementation | str,
    ) -> TTSAttentionImplementation:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("`attn_implementation` must be a string or "
                            "TTSAttentionImplementation.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "fa4": cls.FLASH_ATTENTION_4.value,
            "flash_attn_4": cls.FLASH_ATTENTION_4.value,
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown TTS attention implementation {value!r}; expected "
                f"one of: {choices}.") from error


class TTSKernelBackend(str, Enum):
    """Custom-kernel policy for architecture-owned fused operations."""

    AUTO = "auto"
    NATIVE = "native"
    TORCH = "torch"
    TRITON = "triton"
    CUDA_EXTENSION = "cuda_extension"

    @classmethod
    def coerce(
        cls,
        value: TTSKernelBackend | str,
    ) -> TTSKernelBackend:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("`kernel_backend` must be a string or TTSKernelBackend.")
        normalized = value.strip().lower().replace("-", "_")
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown TTS kernel backend {value!r}; expected one of: "
                f"{choices}.") from error


class TTSCompilePolicy(str, Enum):
    """Whether compilation may fall back, must work, or is disabled."""

    AUTO = "auto"
    REQUIRED = "required"
    DISABLED = "disabled"

    @classmethod
    def coerce(
        cls,
        value: TTSCompilePolicy | str | bool,
    ) -> TTSCompilePolicy:
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.REQUIRED if value else cls.DISABLED
        if not isinstance(value, str):
            raise TypeError("`compile` must be a boolean, string, or TTSCompilePolicy.")
        normalized = value.strip().lower()
        aliases = {
            "false": cls.DISABLED.value,
            "off": cls.DISABLED.value,
            "true": cls.REQUIRED.value,
            "on": cls.REQUIRED.value,
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown TTS compile policy {value!r}; expected one of: "
                f"{choices}.") from error


def _optimization_pass_names(values: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values, )
    else:
        try:
            values = tuple(values)
        except TypeError as error:
            raise TypeError(
                "`optimization_passes` must be an iterable of registered "
                "pass names.") from error
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("`optimization_passes` must contain non-empty strings.")
        normalized.append(value.strip().lower())
    if len(normalized) != len(set(normalized)):
        raise ValueError("`optimization_passes` cannot contain duplicates.")
    return tuple(normalized)


def _coerce_compile_config(
    value: TorchCompileConfig | Mapping[str, Any] | None,
    *,
    policy: TTSCompilePolicy,
) -> TorchCompileConfig:
    if value is None:
        config = TorchCompileConfig()
    elif isinstance(value, TorchCompileConfig):
        config = value
    elif isinstance(value, Mapping):
        config = TorchCompileConfig(**dict(value))
    else:
        raise TypeError("`compile_config` must be a TorchCompileConfig, mapping, or None.")

    requirement = (
        TorchCompileRequirement.REQUIRED
        if policy is TTSCompilePolicy.REQUIRED else TorchCompileRequirement.AUTO)
    if config.requirement is not requirement:
        config = replace(config, requirement=requirement)
    return config


@dataclass(frozen=True, slots=True)
class TTSOptimizationConfig:
    """Serializable, Transformers-style optimization settings for TTS.

    ``auto`` values may choose a verified implementation and otherwise
    retain the model's native PyTorch path.  Explicit FlashAttention-4,
    Triton, CUDA extension, and required compile requests fail during
    resolution when their static architecture/context requirements are
    not satisfied.
    """

    attn_implementation: (TTSAttentionImplementation | str) = TTSAttentionImplementation.AUTO
    kernel_backend: TTSKernelBackend | str = TTSKernelBackend.AUTO
    compile: TTSCompilePolicy | str | bool = TTSCompilePolicy.AUTO
    compile_config: TorchCompileConfig | Mapping[str, Any] | None = None
    diffusion_cache: DiffusionCachePolicy | str | bool = "disabled"
    diffusion_cache_config: DiffusionCacheConfig | Mapping[str, Any] | None = None
    diffusion_sampling: DiffusionSamplingPolicy | str | bool = "disabled"
    diffusion_sampling_config: (DiffusionSamplingConfig | Mapping[str, Any] | None) = None
    optimization_passes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        attention = TTSAttentionImplementation.coerce(self.attn_implementation)
        kernels = TTSKernelBackend.coerce(self.kernel_backend)
        compile_policy = TTSCompilePolicy.coerce(self.compile)
        compile_config = _coerce_compile_config(
            self.compile_config,
            policy=compile_policy,
        )
        diffusion_module = import_module("voicehub.optimization.diffusion_cache", )
        diffusion_cache = diffusion_module.DiffusionCachePolicy.coerce(self.diffusion_cache, )
        diffusion_cache_config = (
            diffusion_module.coerce_diffusion_cache_config(self.diffusion_cache_config, ))
        sampling_module = import_module("voicehub.optimization.diffusion_sampling")
        diffusion_sampling = sampling_module.DiffusionSamplingPolicy.coerce(self.diffusion_sampling, )
        diffusion_sampling_config = (
            sampling_module.coerce_diffusion_sampling_config(self.diffusion_sampling_config, ))
        extra_passes = _optimization_pass_names(self.optimization_passes)
        object.__setattr__(self, "attn_implementation", attention)
        object.__setattr__(self, "kernel_backend", kernels)
        object.__setattr__(self, "compile", compile_policy)
        object.__setattr__(self, "compile_config", compile_config)
        object.__setattr__(self, "diffusion_cache", diffusion_cache)
        object.__setattr__(
            self,
            "diffusion_cache_config",
            diffusion_cache_config,
        )
        object.__setattr__(
            self,
            "diffusion_sampling",
            diffusion_sampling,
        )
        object.__setattr__(
            self,
            "diffusion_sampling_config",
            diffusion_sampling_config,
        )
        object.__setattr__(
            self,
            "optimization_passes",
            extra_passes,
        )
        try:
            canonical_json_string(
                self.to_dict(),
                path="TTS optimization configuration",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "TTS optimization configuration must contain only strict "
                f"JSON values: {error}") from error

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        **overrides: Any,
    ) -> TTSOptimizationConfig:
        """Create a configuration from a JSON-compatible mapping."""
        if not isinstance(values, Mapping):
            raise TypeError("TTS optimization configuration must be a mapping.")
        config_values = dict(values)
        config_values.update(overrides)
        return cls(**config_values)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict-JSON-compatible configuration mapping."""
        return {
            "attn_implementation": self.attn_implementation.value,
            "kernel_backend": self.kernel_backend.value,
            "compile": self.compile.value,
            "compile_config": self.compile_config.manifest(),
            "diffusion_cache": self.diffusion_cache.value,
            "diffusion_cache_config": self.diffusion_cache_config.to_dict(),
            "diffusion_sampling": self.diffusion_sampling.value,
            "diffusion_sampling_config": self.diffusion_sampling_config.to_dict(),
            "optimization_passes": list(self.optimization_passes),
        }

    def to_json_string(self) -> str:
        """Serialize with deterministic ordering for reports/checkpoints."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    def resolve(
        self,
        target: str | Any,
        *,
        mode: OptimizationMode | str = OptimizationMode.INFERENCE,
        context: OptimizationContext | None = None,
        registry: OptimizationPassRegistry | None = None,
    ) -> TTSOptimizationPlan:
        """Resolve this policy for a registered model or architecture."""
        return resolve_tts_optimization(
            target,
            self,
            mode=mode,
            context=context,
            registry=registry,
        )


def coerce_tts_optimization_config(
        value: TTSOptimizationConfig | Mapping[str, Any] | None) -> TTSOptimizationConfig:
    """Normalize a public TTS optimization configuration value."""
    if value is None:
        return TTSOptimizationConfig()
    if isinstance(value, TTSOptimizationConfig):
        return value
    if isinstance(value, Mapping):
        return TTSOptimizationConfig.from_dict(value)
    raise TypeError("`optimization_config` must be a TTSOptimizationConfig, mapping, "
                    "or None.")


def tts_optimization_config_from_options(
    optimization_config: TTSOptimizationConfig | Mapping[str, Any] | None = None,
    *,
    attn_implementation: str | None = None,
    kernel_backend: str | None = None,
    torch_compile: bool | str | None = None,
    compile_config: TorchCompileConfig | Mapping[str, Any] | None = None,
    diffusion_cache: bool | str | None = None,
    diffusion_cache_config: DiffusionCacheConfig | Mapping[str, Any] | None = None,
    diffusion_sampling: bool | str | None = None,
    diffusion_sampling_config: (DiffusionSamplingConfig | Mapping[str, Any] | None) = None,
) -> TTSOptimizationConfig | None:
    """Merge ``from_pretrained``-style optimization keyword arguments.

    Supplying only a direct attention or kernel option does not enable
    any unrelated transformation.  Passing a full
    ``optimization_config`` retains its defaults and applies only the
    direct overrides supplied alongside it.
    """
    direct = any(
        value is not None for value in (
            attn_implementation,
            kernel_backend,
            torch_compile,
            compile_config,
            diffusion_cache,
            diffusion_cache_config,
            diffusion_sampling,
            diffusion_sampling_config,
        ))
    if optimization_config is None and not direct:
        return None
    if optimization_config is None:
        return TTSOptimizationConfig(
            attn_implementation=("native" if attn_implementation is None else attn_implementation),
            kernel_backend=("native" if kernel_backend is None else kernel_backend),
            compile=(
                "auto" if torch_compile is None and compile_config is not None else
                ("disabled" if torch_compile is None else torch_compile)),
            compile_config=compile_config,
            diffusion_cache=("disabled" if diffusion_cache is None else diffusion_cache),
            diffusion_cache_config=diffusion_cache_config,
            diffusion_sampling=("disabled" if diffusion_sampling is None else diffusion_sampling),
            diffusion_sampling_config=diffusion_sampling_config,
        )

    resolved = coerce_tts_optimization_config(optimization_config)
    if not direct:
        return resolved
    values = resolved.to_dict()
    if attn_implementation is not None:
        values["attn_implementation"] = attn_implementation
    if kernel_backend is not None:
        values["kernel_backend"] = kernel_backend
    if torch_compile is not None:
        values["compile"] = torch_compile
    if compile_config is not None:
        values["compile_config"] = compile_config
    if diffusion_cache is not None:
        values["diffusion_cache"] = diffusion_cache
    if diffusion_cache_config is not None:
        values["diffusion_cache_config"] = diffusion_cache_config
    if diffusion_sampling is not None:
        values["diffusion_sampling"] = diffusion_sampling
    if diffusion_sampling_config is not None:
        values["diffusion_sampling_config"] = diffusion_sampling_config
    return TTSOptimizationConfig(**values)


@dataclass(frozen=True, slots=True)
class TTSOptimizationDecision:
    """One requested policy choice and its statically resolved outcome."""

    feature: str
    requested: str
    selected: str
    implementation_pass: str | None
    reason: str

    def __post_init__(self) -> None:
        for name in ("feature", "requested", "selected", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`{name}` must be a non-empty string.")
        if self.implementation_pass is not None and (not isinstance(self.implementation_pass, str) or
                                                     not self.implementation_pass.strip()):
            raise ValueError("`implementation_pass` must be a non-empty string or None.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "requested": self.requested,
            "selected": self.selected,
            "implementation_pass": self.implementation_pass,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TTSOptimizationSupport:
    """Architecture-level implementations available to one TTS model."""

    model_type: str | None
    architecture: str | None
    attention_implementations: tuple[str, ...]
    kernel_backends: tuple[str, ...]
    compile: bool
    diffusion_cache: bool
    diffusion_sampling: bool
    diffusion_sampling_techniques: tuple[str, ...]
    optimization_kinds: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "architecture": self.architecture,
            "attention_implementations": list(self.attention_implementations),
            "kernel_backends": list(self.kernel_backends),
            "compile": self.compile,
            "diffusion_cache": self.diffusion_cache,
            "diffusion_sampling": self.diffusion_sampling,
            "diffusion_sampling_techniques": list(self.diffusion_sampling_techniques),
            "optimization_kinds": list(self.optimization_kinds),
        }


@dataclass(frozen=True, slots=True)
class TTSOptimizationPlan:
    """Resolved, ordered pass plan plus all fallback decisions."""

    config: TTSOptimizationConfig
    context: OptimizationContext
    support: TTSOptimizationSupport
    passes: tuple[OptimizationPass, ...]
    decisions: tuple[TTSOptimizationDecision, ...]
    _pass_declaration_snapshots: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _config_snapshot: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_config_snapshot",
            canonical_json_string(
                self.config.to_dict(),
                path="TTS optimization plan configuration",
            ),
        )
        snapshots = tuple(snapshot_optimization_pass_declaration(item) for item in self.passes)
        object.__setattr__(
            self,
            "_pass_declaration_snapshots",
            snapshots,
        )

    def __iter__(self):
        return iter(self.passes)

    def __len__(self) -> int:
        return len(self.passes)

    @property
    def model_type(self) -> str | None:
        return self.support.model_type

    @property
    def architecture(self) -> str | None:
        return self.support.architecture

    @property
    def pass_declaration_snapshots(self) -> tuple[str, ...]:
        """Return immutable declarations captured during resolution."""
        return self._pass_declaration_snapshots

    def manifest(self) -> dict[str, Any]:
        """Return a deterministic JSON record, including native fallbacks."""
        value = {
            "format_version":
            1,
            "target": {
                "model_type": self.model_type,
                "architecture": self.architecture,
            },
            "context": {
                "mode": self.context.mode.value,
                "architecture": self.context.architecture,
                "device": self.context.device,
                "dtype": self.context.dtype,
                "streaming": self.context.streaming,
                "distributed": self.context.distributed,
                "persist_result": self.context.persist_result,
            },
            "config":
            json.loads(self._config_snapshot),
            "support":
            self.support.to_dict(),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "passes": [{
                "pass": declaration["pass"],
                "kind": declaration["kind"],
                "version": declaration["version"],
                "configuration": declaration["configuration"],
            } for declaration in (json.loads(snapshot) for snapshot in self._pass_declaration_snapshots)],
        }
        return json.loads(canonical_json_string(
            value,
            path="TTS optimization plan manifest",
        ))


@dataclass(frozen=True, slots=True)
class TTSOptimizationResult:
    """Universal result for both transformed and native-fallback plans."""

    plan: TTSOptimizationPlan
    model: Any
    application: OptimizationResult | None = None

    @property
    def optimized(self) -> bool:
        """Whether at least one executable pass was applied."""
        return self.application is not None

    def restore(self) -> Any:
        """Restore transformed plans or return the untouched native model."""
        if self.application is None:
            return self.model
        return self.application.restore()

    def manifest(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "resolution": self.plan.manifest(),
            "application": (None if self.application is None else self.application.manifest()),
        }


def _resolve_target(target: str | Any, ) -> tuple[ModelSpec | None, ArchitectureSpec | None]:
    from voicehub.errors import UnknownModelError
    from voicehub.registry import get_model_spec
    from voicehub.runtime import UnknownArchitectureError, get_architecture_spec

    if isinstance(target, str):
        if not target.strip():
            raise ValueError("TTS optimization target cannot be empty.")
        try:
            model_spec = get_model_spec(target)
        except UnknownModelError as model_error:
            try:
                architecture = get_architecture_spec(target)
            except UnknownArchitectureError:
                raise model_error
            if not architecture.capabilities.supports_task(SpeechTask.TEXT_TO_SPEECH):
                raise TTSOptimizationCompatibilityError(
                    f"Architecture {architecture.architecture_id!r} is not "
                    "a text-to-speech architecture.")
            return None, architecture
    else:
        config = getattr(target, "config", None)
        model_type = getattr(config, "model_type", None)
        if not isinstance(model_type, str) or not model_type.strip():
            raise TypeError(
                "TTS optimization targets must be a registered model type, "
                "architecture ID, or model exposing config.model_type.")
        model_spec = get_model_spec(model_type)

    if model_spec.task is not SpeechTask.TEXT_TO_SPEECH:
        raise TTSOptimizationCompatibilityError(
            f"Model {model_spec.model_type!r} belongs to task "
            f"{model_spec.task.value!r}, not text-to-speech.")
    architecture = (
        None if model_spec.architecture is None else get_architecture_spec(model_spec.architecture))
    return model_spec, architecture


def _resolve_auto_device() -> str:
    try:
        torch = import_module("torch")
    except (ImportError, ModuleNotFoundError):
        return "cpu"
    cuda = getattr(torch, "cuda", None)
    if (cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available()):
        return "cuda"
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if (mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available()):
        return "mps"
    return "cpu"


def _resolve_context(
    target: str | Any,
    *,
    mode: OptimizationMode | str,
    context: OptimizationContext | None,
    architecture: ArchitectureSpec | None,
) -> OptimizationContext:
    normalized_mode = OptimizationMode.coerce(mode)
    if context is not None:
        if not isinstance(context, OptimizationContext):
            raise TypeError("`context` must be an OptimizationContext.")
        if context.mode is not normalized_mode:
            raise ValueError(
                "Optimization context mode does not match the requested mode "
                f"({context.mode.value!r} != {normalized_mode.value!r}).")
        resolved = context
    elif not isinstance(target, str) and callable(getattr(target, "_default_optimization_context", None)):
        resolved = target._default_optimization_context(normalized_mode)
    else:
        resolved = OptimizationContext(
            mode=normalized_mode,
            persist_result=normalized_mode is OptimizationMode.TRAINING,
        )

    if resolved.device == "auto":
        resolved = replace(resolved, device=_resolve_auto_device())
    if architecture is None:
        return resolved

    if resolved.architecture is not None:
        from voicehub.runtime import get_architecture_spec

        requested = get_architecture_spec(resolved.architecture)
        if requested.architecture_id != architecture.architecture_id:
            raise TTSOptimizationCompatibilityError(
                "Optimization context architecture does not match the "
                f"target ({requested.architecture_id!r} != "
                f"{architecture.architecture_id!r}).")
    if resolved.architecture != architecture.architecture_id:
        resolved = replace(
            resolved,
            architecture=architecture.architecture_id,
        )
    return resolved


def _validate_architecture_context(
    architecture: ArchitectureSpec | None,
    context: OptimizationContext,
) -> None:
    if architecture is None:
        return
    capabilities = architecture.capabilities
    issues = []
    if not capabilities.supports_device(context.device):
        issues.append(f"device {context.device!r}")
    if not capabilities.supports_dtype(context.dtype):
        issues.append(f"dtype {context.dtype!r}")
    if (context.mode is OptimizationMode.TRAINING and not capabilities.training):
        issues.append("training execution")
    if context.streaming and not capabilities.streaming:
        issues.append("streaming execution")
    if context.distributed and (context.mode is not OptimizationMode.TRAINING or
                                not capabilities.distributed_training):
        issues.append("distributed execution")
    if issues:
        raise TTSOptimizationCompatibilityError(
            f"Architecture {architecture.architecture_id!r} does not "
            f"support {', '.join(issues)}.")


def get_tts_optimization_support(target: str | Any, ) -> TTSOptimizationSupport:
    """Return the statically declared optimization surface for one target."""
    model_spec, architecture = _resolve_target(target)
    kinds = (() if architecture is None else architecture.capabilities.optimization_passes)
    attention = ["native"]
    if "sdpa" in kinds:
        attention.append("sdpa")
    if "attention-backend" in kinds:
        if "sdpa" not in attention:
            attention.append("sdpa")
        attention.append("flash_attention_4")

    kernels = ["native", "torch"]
    if "custom-kernels" in kinds:
        kernels.extend(("auto", "triton", "cuda_extension"))
    sampling_prefix = "diffusion-sampling-"
    sampling_techniques = (() if architecture is None else tuple(
        feature.removeprefix(sampling_prefix) for feature in architecture.capabilities.features
        if feature.startswith(sampling_prefix)))

    return TTSOptimizationSupport(
        model_type=(None if model_spec is None else model_spec.model_type),
        architecture=(None if architecture is None else architecture.architecture_id),
        attention_implementations=tuple(attention),
        kernel_backends=tuple(kernels),
        compile=architecture is None or "compile" in kinds,
        diffusion_cache="diffusion-cache" in kinds,
        diffusion_sampling="diffusion-sampling" in kinds,
        diffusion_sampling_techniques=sampling_techniques,
        optimization_kinds=tuple(kinds),
    )


def validate_tts_optimization_config(
    target: str | Any,
    config: TTSOptimizationConfig | Mapping[str, Any] | None = None,
    *,
    mode: OptimizationMode | str = OptimizationMode.INFERENCE,
) -> TTSOptimizationConfig:
    """Fail early for explicit choices the architecture cannot implement."""
    resolved = coerce_tts_optimization_config(config)
    _model_spec, architecture = _resolve_target(target)
    support = get_tts_optimization_support(target)
    attention = resolved.attn_implementation.value
    if (resolved.attn_implementation in {
            TTSAttentionImplementation.SDPA,
            TTSAttentionImplementation.FLASH_ATTENTION_4,
    } and attention not in support.attention_implementations):
        requirement = (
            "the attention-backend protocol" if resolved.attn_implementation
            is TTSAttentionImplementation.FLASH_ATTENTION_4 else "an SDPA-compatible attention path")
        raise TTSOptimizationCompatibilityError(
            f"Architecture {support.architecture!r} does not declare "
            f"{requirement} required by {attention!r}.")
    if (resolved.kernel_backend in {
            TTSKernelBackend.TRITON,
            TTSKernelBackend.CUDA_EXTENSION,
    } and resolved.kernel_backend.value not in support.kernel_backends):
        raise TTSOptimizationCompatibilityError(
            f"Architecture {support.architecture!r} does not declare "
            "custom-kernels compatibility required by "
            f"{resolved.kernel_backend.value!r}.")
    if (resolved.compile is TTSCompilePolicy.REQUIRED and not support.compile):
        raise TTSOptimizationCompatibilityError(
            f"Architecture {support.architecture!r} does not declare "
            "torch.compile compatibility.")
    if resolved.compile is TTSCompilePolicy.REQUIRED:
        compile_issue = torch_compile_architecture_incompatibility(
            resolved.compile_config,
            architecture,
            mode=mode,
        )
        if compile_issue is not None:
            raise TTSOptimizationCompatibilityError(compile_issue)
    diffusion_module = import_module("voicehub.optimization.diffusion_cache", )
    if (resolved.diffusion_cache is diffusion_module.DiffusionCachePolicy.REQUIRED and
            not support.diffusion_cache):
        raise TTSOptimizationCompatibilityError(
            f"Architecture {support.architecture!r} does not declare an "
            "architecture-owned diffusion-cache block surface.")
    sampling_module = import_module("voicehub.optimization.diffusion_sampling", )
    if (resolved.diffusion_sampling is sampling_module.DiffusionSamplingPolicy.REQUIRED and
            not support.diffusion_sampling):
        raise TTSOptimizationCompatibilityError(
            f"Architecture {support.architecture!r} does not declare an "
            "architecture-owned diffusion-sampling surface.")
    return resolved


def list_tts_optimization_support() -> tuple[TTSOptimizationSupport, ...]:
    """List optimization capabilities for every registered TTS model."""
    from voicehub.registry import list_model_specs

    return tuple(
        get_tts_optimization_support(spec.model_type)
        for spec in list_model_specs(task=SpeechTask.TEXT_TO_SPEECH))


def _pass_decision(
    feature: str,
    requested: str,
    selected: str,
    implementation_pass: OptimizationPass | None,
    reason: str,
) -> TTSOptimizationDecision:
    return TTSOptimizationDecision(
        feature=feature,
        requested=requested,
        selected=selected,
        implementation_pass=(None if implementation_pass is None else implementation_pass.qualified_id),
        reason=reason,
    )


def _resolve_kernels(
    config: TTSOptimizationConfig,
    support: TTSOptimizationSupport,
    context: OptimizationContext,
) -> tuple[OptimizationPass | None, TTSOptimizationDecision]:
    requested = config.kernel_backend
    custom = "custom-kernels" in support.optimization_kinds

    if requested is TTSKernelBackend.NATIVE:
        return None, _pass_decision(
            "kernels",
            requested.value,
            "native",
            None,
            "The architecture's existing kernels were retained.",
        )
    if not custom:
        if requested in {
                TTSKernelBackend.TRITON,
                TTSKernelBackend.CUDA_EXTENSION,
        }:
            raise TTSOptimizationCompatibilityError(
                f"Architecture {support.architecture!r} does not declare "
                "custom-kernels compatibility, so "
                f"{requested.value!r} cannot be requested.")
        return None, _pass_decision(
            "kernels",
            requested.value,
            "torch",
            None,
            "No architecture-specific fused kernel protocol is declared; "
            "the native PyTorch operations remain active.",
        )

    if (requested in {
            TTSKernelBackend.TRITON,
            TTSKernelBackend.CUDA_EXTENSION,
    } and context.device.partition(":")[0] != "cuda"):
        raise TTSOptimizationCompatibilityError(
            f"Kernel backend {requested.value!r} requires CUDA, but the "
            f"optimization context uses {context.device!r}.")

    module = import_module("voicehub.optimization.accelerators")
    selection = ("auto" if requested is TTSKernelBackend.AUTO else requested.value)
    optimization_pass = module.CustomKernelPass(backend=selection)
    selected = ("torch" if requested is TTSKernelBackend.AUTO else requested.value)
    reason = (
        "Automatic kernel selection conservatively retains the portable "
        "Torch implementation; accelerator kernels require an explicit "
        "backend or codec fidelity policy." if requested is TTSKernelBackend.AUTO else
        "The explicitly requested architecture-owned kernel backend "
        "will be validated before mutation.")
    return optimization_pass, _pass_decision(
        "kernels",
        requested.value,
        selected,
        optimization_pass,
        reason,
    )


def _resolve_attention(
    config: TTSOptimizationConfig,
    support: TTSOptimizationSupport,
    context: OptimizationContext,
) -> tuple[OptimizationPass | None, TTSOptimizationDecision]:
    requested = config.attn_implementation
    kinds = support.optimization_kinds
    selectable = "attention-backend" in kinds
    native_sdpa = "sdpa" in kinds

    if requested is TTSAttentionImplementation.NATIVE:
        return None, _pass_decision(
            "attention",
            requested.value,
            "native",
            None,
            "The architecture's native attention semantics were retained.",
        )
    if requested is TTSAttentionImplementation.AUTO:
        if selectable:
            return None, _pass_decision(
                "attention",
                requested.value,
                "native",
                None,
                "Automatic attention selection retains the architecture's "
                "native path. Request SDPA or FlashAttention-4 explicitly "
                "after checkpoint-level quality validation.",
            )
        selected = "sdpa" if native_sdpa else "native"
        reason = (
            "The architecture already owns a verified SDPA path."
            if native_sdpa else "No interchangeable attention-backend protocol is "
            "declared; native attention is the safe fallback.")
        return None, _pass_decision(
            "attention",
            requested.value,
            selected,
            None,
            reason,
        )
    if requested is TTSAttentionImplementation.SDPA:
        if not (native_sdpa or selectable):
            raise TTSOptimizationCompatibilityError(
                f"Architecture {support.architecture!r} does not declare "
                "an SDPA-compatible attention path.")
        if not selectable:
            return None, _pass_decision(
                "attention",
                requested.value,
                "sdpa",
                None,
                "The architecture's built-in SDPA path is already active.",
            )
        module = import_module("voicehub.optimization.accelerators")
        optimization_pass = module.FlashAttention4Pass(policy="disabled")
        return optimization_pass, _pass_decision(
            "attention",
            requested.value,
            "sdpa",
            optimization_pass,
            "FlashAttention-4 was disabled explicitly on every compatible "
            "selector target.",
        )

    if not selectable:
        raise TTSOptimizationCompatibilityError(
            f"Architecture {support.architecture!r} does not declare the "
            "attention-backend protocol required by FlashAttention-4.")
    device_family = context.device.partition(":")[0]
    if device_family != "cuda":
        raise TTSOptimizationCompatibilityError(
            "Explicit FlashAttention-4 requires a CUDA optimization context.")
    if context.dtype not in {"float16", "bfloat16"}:
        raise TTSOptimizationCompatibilityError("Explicit FlashAttention-4 requires float16 or bfloat16.")
    module = import_module("voicehub.optimization.accelerators")
    optimization_pass = module.FlashAttention4Pass(policy="required")
    return optimization_pass, _pass_decision(
        "attention",
        requested.value,
        requested.value,
        optimization_pass,
        "FlashAttention-4 was explicitly requested and will fail rather "
        "than silently changing attention semantics.",
    )


def _resolve_compile(
    config: TTSOptimizationConfig,
    support: TTSOptimizationSupport,
    context: OptimizationContext,
    architecture: ArchitectureSpec | None,
) -> tuple[OptimizationPass | None, TTSOptimizationDecision]:
    policy = config.compile
    if policy is TTSCompilePolicy.DISABLED:
        return None, _pass_decision(
            "compile",
            policy.value,
            "eager",
            None,
            "Compilation was disabled explicitly.",
        )
    if not support.compile:
        if policy is TTSCompilePolicy.REQUIRED:
            raise TTSOptimizationCompatibilityError(
                f"Architecture {support.architecture!r} does not declare "
                "torch.compile compatibility.")
        return None, _pass_decision(
            "compile",
            policy.value,
            "eager",
            None,
            "The architecture does not declare compile compatibility.",
        )

    compile_config = config.compile_config
    architecture_issue = torch_compile_architecture_incompatibility(
        compile_config,
        architecture,
        mode=context.mode,
    )
    if architecture_issue is not None:
        if policy is TTSCompilePolicy.REQUIRED:
            raise TTSOptimizationCompatibilityError(architecture_issue)
        return None, _pass_decision(
            "compile",
            policy.value,
            "eager",
            None,
            architecture_issue + " Automatic policy retained eager execution.",
        )
    if (policy is TTSCompilePolicy.AUTO and context.mode is OptimizationMode.INFERENCE):
        capabilities = getattr(architecture, "capabilities", None)
        has_feature = getattr(capabilities, "has_feature", None)
        if (not callable(has_feature) or not has_feature(_INFERENCE_AUTO_COMPILE_VALIDATED_FEATURE)):
            return None, _pass_decision(
                "compile",
                policy.value,
                "eager",
                None,
                "Automatic inference compilation requires retained "
                "real-checkpoint validation for this architecture. Use "
                "`compile='required'` only after benchmarking the target "
                "checkpoint and workload.",
            )
    optimization_pass = TorchCompilePass(
        backend=compile_config.backend,
        mode=compile_config.mode,
        fullgraph=compile_config.fullgraph,
        dynamic=compile_config.dynamic,
        options=compile_config.options,
        requirement=compile_config.requirement,
    )
    incompatibilities = optimization_pass.capabilities.incompatibilities(context)
    if incompatibilities:
        reason = ("torch.compile does not support " + ", ".join(incompatibilities))
        if policy is TTSCompilePolicy.REQUIRED:
            raise TTSOptimizationCompatibilityError(reason + ".")
        return None, _pass_decision(
            "compile",
            policy.value,
            "eager",
            None,
            reason + "; automatic policy retained eager execution.",
        )
    return optimization_pass, _pass_decision(
        "compile",
        policy.value,
        f"torch.compile:{compile_config.backend}",
        optimization_pass,
        (
            "Compilation is required and any compiler failure is an error." if policy
            is TTSCompilePolicy.REQUIRED else "Compiler/backend discovery and lazy execution failures "
            "may fall back locally to the original eager callable."),
    )


def _resolve_diffusion_cache(
    config: TTSOptimizationConfig,
    support: TTSOptimizationSupport,
    context: OptimizationContext,
) -> tuple[OptimizationPass | None, TTSOptimizationDecision]:
    module = import_module("voicehub.optimization.diffusion_cache")
    policy = config.diffusion_cache
    if policy is module.DiffusionCachePolicy.DISABLED:
        return None, _pass_decision(
            "diffusion_cache",
            policy.value,
            "disabled",
            None,
            "Approximate diffusion block caching is disabled by default.",
        )
    if context.mode is not OptimizationMode.INFERENCE:
        reason = (
            "Diffusion residual caching is inference-only because training "
            "timesteps and parameters change between calls.")
        if policy is module.DiffusionCachePolicy.REQUIRED:
            raise TTSOptimizationCompatibilityError(reason)
        return None, _pass_decision(
            "diffusion_cache",
            policy.value,
            "disabled",
            None,
            reason,
        )
    if not support.diffusion_cache:
        reason = (
            f"Architecture {support.architecture!r} does not declare an "
            "architecture-owned repeated-block cache surface.")
        if policy is module.DiffusionCachePolicy.REQUIRED:
            raise TTSOptimizationCompatibilityError(reason)
        return None, _pass_decision(
            "diffusion_cache",
            policy.value,
            "unsupported",
            None,
            reason,
        )
    optimization_pass = module.DiffusionCachePass(config.diffusion_cache_config, )
    incompatibilities = optimization_pass.capabilities.incompatibilities(context, )
    if incompatibilities:
        reason = ("Diffusion residual caching does not support " + ", ".join(incompatibilities))
        if policy is module.DiffusionCachePolicy.REQUIRED:
            raise TTSOptimizationCompatibilityError(reason + ".")
        return None, _pass_decision(
            "diffusion_cache",
            policy.value,
            "disabled",
            None,
            reason + "; automatic policy retained exact execution.",
        )
    return optimization_pass, _pass_decision(
        "diffusion_cache",
        policy.value,
        (
            "block-residual-cache:"
            f"{config.diffusion_cache_config.method.value}:"
            f"{config.diffusion_cache_config.predictor.value}"),
        optimization_pass,
        (
            "Approximate first/middle/last block caching was explicitly "
            "enabled. Samplers reset request state, CFG lanes remain separate, and "
            "training or gradient-enabled calls bypass the cache."),
    )


def _resolve_diffusion_sampling(
    config: TTSOptimizationConfig,
    support: TTSOptimizationSupport,
    context: OptimizationContext,
) -> tuple[OptimizationPass | None, TTSOptimizationDecision]:
    module = import_module("voicehub.optimization.diffusion_sampling")
    policy = config.diffusion_sampling
    if policy is module.DiffusionSamplingPolicy.DISABLED:
        return None, _pass_decision(
            "diffusion_sampling",
            policy.value,
            "disabled",
            None,
            "Approximate sampler-level diffusion acceleration is disabled "
            "by default.",
        )
    if context.mode is not OptimizationMode.INFERENCE:
        reason = ("Schedule rebuilding, guidance reduction, and prediction reuse "
                  "are inference-only.")
        if policy is module.DiffusionSamplingPolicy.REQUIRED:
            raise TTSOptimizationCompatibilityError(reason)
        return None, _pass_decision(
            "diffusion_sampling",
            policy.value,
            "disabled",
            None,
            reason,
        )
    if not support.diffusion_sampling:
        reason = (
            f"Architecture {support.architecture!r} does not declare an "
            "architecture-owned sampler integration.")
        if policy is module.DiffusionSamplingPolicy.REQUIRED:
            raise TTSOptimizationCompatibilityError(reason)
        return None, _pass_decision(
            "diffusion_sampling",
            policy.value,
            "unsupported",
            None,
            reason,
        )
    sampling_config = config.diffusion_sampling_config
    requested_techniques = []
    if (sampling_config.target_steps is not None and not {
            "schedule",
            "discrete-step-count",
    }.intersection(support.diffusion_sampling_techniques)):
        requested_techniques.append("schedule")
    if sampling_config.guidance.value != "native":
        requested_techniques.append("guidance")
    if sampling_config.prediction_cache.value != "disabled":
        requested_techniques.append("prediction-cache")
    if sampling_config.solver.value == "stork2":
        requested_techniques.append("stork2")
    missing = tuple(
        technique for technique in requested_techniques
        if technique not in support.diffusion_sampling_techniques)
    if missing:
        reason = (
            f"Architecture {support.architecture!r} does not declare "
            "diffusion sampling technique(s): " + ", ".join(missing) + ".")
        if policy is module.DiffusionSamplingPolicy.REQUIRED:
            raise TTSOptimizationCompatibilityError(reason)
        return None, _pass_decision(
            "diffusion_sampling",
            policy.value,
            "unsupported",
            None,
            reason + " Automatic policy retained native sampling.",
        )
    optimization_pass = module.DiffusionSamplingPass(config.diffusion_sampling_config, )
    incompatibilities = optimization_pass.capabilities.incompatibilities(context, )
    if incompatibilities:
        reason = ("Diffusion sampler acceleration does not support " + ", ".join(incompatibilities))
        if policy is module.DiffusionSamplingPolicy.REQUIRED:
            raise TTSOptimizationCompatibilityError(reason + ".")
        return None, _pass_decision(
            "diffusion_sampling",
            policy.value,
            "disabled",
            None,
            reason + "; automatic policy retained exact sampling.",
        )
    selected = (
        f"steps:{sampling_config.target_steps or 'native'}"
        f"+solver:{sampling_config.solver.value}"
        f"+guidance:{sampling_config.guidance.value}"
        f"+prediction:{sampling_config.prediction_cache.value}")
    return optimization_pass, _pass_decision(
        "diffusion_sampling",
        policy.value,
        selected,
        optimization_pass,
        (
            "Sampler-level acceleration was explicitly enabled. Native "
            "schedules are rebuilt before integration, CFG lanes remain "
            "isolated, and calibrated cache modes fail closed."),
    )


def _resolve_extra_passes(
    config: TTSOptimizationConfig,
    context: OptimizationContext,
    registry: OptimizationPassRegistry,
) -> tuple[
        tuple[OptimizationPass, ...],
        tuple[TTSOptimizationDecision, ...],
]:
    passes = []
    decisions = []
    for name in config.optimization_passes:
        optimization_pass = registry.create(name)
        incompatibilities = (optimization_pass.capabilities.incompatibilities(context))
        if incompatibilities:
            raise TTSOptimizationCompatibilityError(
                f"Registered optimization pass {name!r} does not support "
                f"{', '.join(incompatibilities)}.")
        passes.append(optimization_pass)
        decisions.append(
            _pass_decision(
                f"pass:{name}",
                name,
                optimization_pass.qualified_id,
                optimization_pass,
                "The explicitly registered pass is compatible with the "
                "runtime context; its validate() hook will inspect the "
                "loaded model before mutation.",
            ))
    return tuple(passes), tuple(decisions)


def resolve_tts_optimization(
    target: str | Any,
    config: TTSOptimizationConfig | Mapping[str, Any] | None = None,
    *,
    mode: OptimizationMode | str = OptimizationMode.INFERENCE,
    context: OptimizationContext | None = None,
    registry: OptimizationPassRegistry | None = None,
) -> TTSOptimizationPlan:
    """Resolve one safe, ordered policy for any registered TTS model.

    Resolution is side-effect free with respect to model weights and
    optional CUDA packages.  The returned plan may contain no executable
    passes; such a plan is still a successful native fallback with a
    complete decision manifest.
    """
    resolved_config = validate_tts_optimization_config(
        target,
        config,
        mode=mode,
    )
    _model_spec, architecture = _resolve_target(target)
    resolved_context = _resolve_context(
        target,
        mode=mode,
        context=context,
        architecture=architecture,
    )
    _validate_architecture_context(architecture, resolved_context)
    support = get_tts_optimization_support(target)
    pass_registry = OPTIMIZATION_PASSES if registry is None else registry
    if not isinstance(pass_registry, OptimizationPassRegistry):
        raise TypeError("`registry` must be an OptimizationPassRegistry.")

    kernel_pass, kernel_decision = _resolve_kernels(
        resolved_config,
        support,
        resolved_context,
    )
    attention_pass, attention_decision = _resolve_attention(
        resolved_config,
        support,
        resolved_context,
    )
    cache_pass, cache_decision = _resolve_diffusion_cache(
        resolved_config,
        support,
        resolved_context,
    )
    sampling_pass, sampling_decision = _resolve_diffusion_sampling(
        resolved_config,
        support,
        resolved_context,
    )
    extra_passes, extra_decisions = _resolve_extra_passes(
        resolved_config,
        resolved_context,
        pass_registry,
    )
    compile_pass, compile_decision = _resolve_compile(
        resolved_config,
        support,
        resolved_context,
        architecture,
    )

    passes = tuple(
        item for item in (
            kernel_pass,
            attention_pass,
            sampling_pass,
            cache_pass,
            *extra_passes,
            compile_pass,
        ) if item is not None)
    qualified_ids = tuple(item.qualified_id for item in passes)
    if len(qualified_ids) != len(set(qualified_ids)):
        raise TTSOptimizationCompatibilityError(
            "The resolved TTS optimization plan contains a duplicate pass. "
            "Remove the corresponding explicit `optimization_passes` entry.")

    return TTSOptimizationPlan(
        config=resolved_config,
        context=resolved_context,
        support=support,
        passes=passes,
        decisions=(
            kernel_decision,
            attention_decision,
            *extra_decisions,
            sampling_decision,
            cache_decision,
            compile_decision,
        ),
    )


def get_tts_optimization_config(
    target: str | Any,
    **overrides: Any,
) -> TTSOptimizationConfig:
    """Return a validated universal config for one registered TTS target.

    The target check catches task/model typos and architecture-declared
    option constraints immediately. Device and runtime compatibility
    remain deferred until :meth:`resolve` receives a concrete execution
    context.
    """
    return validate_tts_optimization_config(
        target,
        TTSOptimizationConfig(**overrides),
    )


__all__ = [
    "TTSAttentionImplementation",
    "TTSCompilePolicy",
    "TTSKernelBackend",
    "TTSOptimizationCompatibilityError",
    "TTSOptimizationConfig",
    "TTSOptimizationDecision",
    "TTSOptimizationError",
    "TTSOptimizationPlan",
    "TTSOptimizationResult",
    "TTSOptimizationSupport",
    "coerce_tts_optimization_config",
    "get_tts_optimization_config",
    "get_tts_optimization_support",
    "list_tts_optimization_support",
    "resolve_tts_optimization",
    "tts_optimization_config_from_options",
    "validate_tts_optimization_config",
]
