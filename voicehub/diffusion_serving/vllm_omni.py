"""Lazy vLLM-Omni feature probing and out-of-tree plugin registration."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, metadata, util
from types import ModuleType
from typing import Any

from voicehub.registry import normalize_model_type

_DISTRIBUTION_NAMES = ("vllm-omni", "vllm_omni")
_REGISTRY_MODULE = "vllm_omni.diffusion.registry"


@dataclass(frozen=True, slots=True)
class VLLMOmniFeatureStatus:
    """Installed-version and registry API status for vLLM-Omni."""

    installed: bool
    version: str | None
    register_diffusion_model: bool | None
    error: str | None = None

    @property
    def supports_out_of_tree_diffusion_plugins(self) -> bool:
        return self.register_diffusion_model is True


def _installed_version() -> str | None:
    for distribution_name in _DISTRIBUTION_NAMES:
        try:
            return metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _top_level_module_available() -> bool:
    try:
        return util.find_spec("vllm_omni") is not None
    except Exception:
        return False


def detect_vllm_omni_features(
    *,
    probe_registry: bool = True,
) -> VLLMOmniFeatureStatus:
    """Detect the optional engine and its public plugin registration API.

    Merely importing :mod:`voicehub.diffusion_serving` never imports
    vLLM-Omni.  With ``probe_registry=False`` this function also remains
    import-free and reports the feature as ``None`` (unknown).  The
    default explicit probe imports only ``vllm_omni.diffusion.registry``
    and checks that ``register_diffusion_model`` is callable.
    """
    version = _installed_version()
    installed = version is not None or _top_level_module_available()
    if not installed:
        return VLLMOmniFeatureStatus(
            installed=False,
            version=None,
            register_diffusion_model=False,
        )
    if not probe_registry:
        return VLLMOmniFeatureStatus(
            installed=True,
            version=version,
            register_diffusion_model=None,
        )
    try:
        registry = import_module(_REGISTRY_MODULE)
    except Exception as error:
        return VLLMOmniFeatureStatus(
            installed=True,
            version=version,
            register_diffusion_model=False,
            error=f"{type(error).__name__}: {error}",
        )
    supported = callable(getattr(registry, "register_diffusion_model", None))
    return VLLMOmniFeatureStatus(
        installed=True,
        version=version,
        register_diffusion_model=supported,
        error=(
            None if supported else
            (f"{_REGISTRY_MODULE} does not expose a callable "
             "register_diffusion_model.")),
    )


def _identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a non-empty Python identifier.")
    value = value.strip()
    if not value.isidentifier():
        raise ValueError(f"`{name}` must be a valid Python identifier.")
    return value


def _dotted_module(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("`module_name` must be a non-empty dotted module.")
    value = value.strip()
    if any(not part.isidentifier() for part in value.split(".")):
        raise ValueError("`module_name` must contain valid dot-separated Python "
                         "identifiers.")
    return value


@dataclass(frozen=True, slots=True)
class VLLMOmniDiffusionPlugin:
    """Experimental contract for a vLLM-Omni out-of-tree diffusion model.

    ``complete_tts_pipeline`` is deliberately explicit: registering only
    a denoiser is insufficient for VoiceHub TTS because the external
    pipeline must also own text/speaker conditioning and waveform or
    codec decoding.
    """

    model_type: str
    model_arch: str
    module_name: str
    class_name: str
    complete_tts_pipeline: bool = False
    pre_process_func_name: str | None = None
    post_process_func_name: str | None = None
    action_post_process_func_name: str | None = None
    ir_op_priority_func_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_type",
            normalize_model_type(self.model_type),
        )
        object.__setattr__(
            self,
            "model_arch",
            _identifier(self.model_arch, name="model_arch"),
        )
        object.__setattr__(
            self,
            "module_name",
            _dotted_module(self.module_name),
        )
        object.__setattr__(
            self,
            "class_name",
            _identifier(self.class_name, name="class_name"),
        )
        if not isinstance(self.complete_tts_pipeline, bool):
            raise TypeError("`complete_tts_pipeline` must be a boolean.")
        for field_name in (
                "pre_process_func_name",
                "post_process_func_name",
                "action_post_process_func_name",
                "ir_op_priority_func_name",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _identifier(value, name=field_name),
                )
        if self.complete_tts_pipeline and self.post_process_func_name is None:
            raise ValueError(
                "A complete TTS plugin must declare `post_process_func_name` "
                "for audio/codec output.")

    def registration_kwargs(self) -> dict[str, Any]:
        """Return the exact public vLLM-Omni registry call contract."""
        return {
            "model_arch": self.model_arch,
            "module_name": self.module_name,
            "class_name": self.class_name,
            "pre_process_func_name": self.pre_process_func_name,
            "post_process_func_name": self.post_process_func_name,
            "action_post_process_func_name": self.action_post_process_func_name,
            "ir_op_priority_func_name": self.ir_op_priority_func_name,
        }

    def register(self) -> None:
        """Lazily register this plugin in an installed vLLM-Omni runtime."""
        try:
            registry: ModuleType = import_module(_REGISTRY_MODULE)
        except Exception as error:
            raise RuntimeError(
                "vLLM-Omni is unavailable or failed to initialize; install a "
                "compatible engine before registering this plugin.") from error
        register = getattr(registry, "register_diffusion_model", None)
        if not callable(register):
            version = _installed_version() or "unknown"
            raise RuntimeError(
                "Installed vLLM-Omni "
                f"{version} does not expose "
                "vllm_omni.diffusion.registry.register_diffusion_model.")
        register(**self.registration_kwargs())


__all__ = [
    "VLLMOmniDiffusionPlugin",
    "VLLMOmniFeatureStatus",
    "detect_vllm_omni_features",
]
