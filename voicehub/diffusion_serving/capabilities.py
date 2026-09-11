"""Fail-closed capability records for diffusion serving engines.

The visual-diffusion projects in this module are not assumed to support
speech. A backend is considered a complete TTS backend only when the
registered model architecture declares a verified serving capability or
the caller supplies the experimental vLLM-Omni out-of-tree plugin
contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from voicehub.errors import VoiceHubError
from voicehub.registry import list_model_specs, normalize_model_type

if TYPE_CHECKING:
    from voicehub.diffusion_serving.vllm_omni import VLLMOmniDiffusionPlugin


class DiffusionServingCompatibilityError(ValueError, VoiceHubError):
    """Raised when an engine cannot preserve a TTS diffusion pipeline."""


class DiffusionServingBackend(str, Enum):
    """Serving runtimes kept distinct by both engine and modality."""

    NATIVE = "native"
    VLLM_OMNI = "vllm-omni"
    SGLANG_DIFFUSION = "sglang-diffusion"
    SGLANG_OMNI = "sglang-omni"

    @classmethod
    def coerce(
        cls,
        value: str | DiffusionServingBackend,
    ) -> DiffusionServingBackend:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("`backend` must be a string or DiffusionServingBackend.")
        normalized = value.strip().lower().replace("_", "-")
        aliases = {
            "native": cls.NATIVE,
            "voicehub": cls.NATIVE,
            "voicehub-native": cls.NATIVE,
            "vllm": cls.VLLM_OMNI,
            "vllm-omni": cls.VLLM_OMNI,
            "vllmomni": cls.VLLM_OMNI,
            "sglang-diffusion": cls.SGLANG_DIFFUSION,
            "sglangdiffusion": cls.SGLANG_DIFFUSION,
            "sglang-diffusion-runtime": cls.SGLANG_DIFFUSION,
            "sglang": cls.SGLANG_OMNI,
            "sglang-omni": cls.SGLANG_OMNI,
            "sglangomni": cls.SGLANG_OMNI,
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown diffusion serving backend {value!r}. "
                f"Choose one of: {choices}.") from error


@dataclass(frozen=True, slots=True)
class DiffusionServingCapability:
    """Static, dependency-free facts about one serving runtime."""

    backend: DiffusionServingBackend
    engine: str
    diffusion_modalities: tuple[str, ...]
    supports_tts: bool
    supports_tts_diffusion: bool
    verified_tts_models: tuple[str, ...] = ()
    supports_custom_plugins: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        modalities = tuple(dict.fromkeys(modality.strip().lower() for modality in self.diffusion_modalities))
        if any(not modality for modality in modalities):
            raise ValueError("`diffusion_modalities` must contain non-empty strings.")
        models = tuple(
            dict.fromkeys(normalize_model_type(model_type) for model_type in self.verified_tts_models))
        if self.supports_tts_diffusion and not self.supports_tts:
            raise ValueError("A TTS diffusion runtime must also declare `supports_tts`.")
        if models and not self.supports_tts_diffusion:
            raise ValueError("Verified TTS diffusion models require "
                             "`supports_tts_diffusion=True`.")
        object.__setattr__(self, "diffusion_modalities", modalities)
        object.__setattr__(self, "verified_tts_models", models)

    @property
    def supports_visual_diffusion(self) -> bool:
        """Whether this engine serves image or video diffusion pipelines."""
        return any(modality in {"image", "video"} for modality in self.diffusion_modalities)


_CAPABILITIES = (
    DiffusionServingCapability(
        backend=DiffusionServingBackend.NATIVE,
        engine="VoiceHub",
        diffusion_modalities=("audio", ),
        supports_tts=True,
        supports_tts_diffusion=True,
        notes=("VoiceHub owns the model-specific denoising/flow loop, "
               "conditioning, and audio decoding."),
    ),
    DiffusionServingCapability(
        backend=DiffusionServingBackend.VLLM_OMNI,
        engine="vLLM-Omni",
        diffusion_modalities=("audio", "image", "video"),
        supports_tts=True,
        supports_tts_diffusion=True,
        supports_custom_plugins=True,
        notes=(
            "Architectures with the verified vLLM-Omni serving capability "
            "are complete speech pipelines. Other VoiceHub TTS architectures "
            "require an explicit "
            "out-of-tree plugin and remain experimental."),
    ),
    DiffusionServingCapability(
        backend=DiffusionServingBackend.SGLANG_DIFFUSION,
        engine="SGLang Diffusion",
        diffusion_modalities=("image", "video"),
        supports_tts=False,
        supports_tts_diffusion=False,
        notes=(
            "SGLang Diffusion is a visual diffusion runtime; its image/video "
            "request and output contracts are not TTS audio contracts."),
    ),
    DiffusionServingCapability(
        backend=DiffusionServingBackend.SGLANG_OMNI,
        engine="SGLang-Omni",
        diffusion_modalities=(),
        supports_tts=True,
        supports_tts_diffusion=False,
        notes=(
            "SGLang-Omni can serve separately verified LLM-TTS pipelines, "
            "but it is not SGLang's visual diffusion runtime. Use "
            "voicehub.llm_serving for those models."),
    ),
)

_CAPABILITY_BY_BACKEND = {capability.backend: capability for capability in _CAPABILITIES}

_VERIFIED_MODEL_FEATURE_BY_BACKEND = {
    DiffusionServingBackend.NATIVE: "diffusion-serving-native",
    DiffusionServingBackend.VLLM_OMNI: "diffusion-serving-vllm-omni",
}


def _verified_tts_models(backend: DiffusionServingBackend, ) -> tuple[str, ...]:
    """Derive verified pairings from registered architecture capabilities."""
    required_feature = _VERIFIED_MODEL_FEATURE_BY_BACKEND.get(backend)
    if required_feature is None:
        return ()

    verified_models: list[str] = []
    for model_spec in list_model_specs(task="text-to-speech", native=True):
        architecture = model_spec.native_architecture
        if (architecture is not None and architecture.capabilities.has_feature(required_feature)):
            verified_models.append(model_spec.model_type)
    return tuple(sorted(verified_models))


def _with_registered_models(capability: DiffusionServingCapability, ) -> DiffusionServingCapability:
    """Return a current snapshot without freezing the extensible registry."""
    return replace(
        capability,
        verified_tts_models=_verified_tts_models(capability.backend),
    )


@dataclass(frozen=True, slots=True)
class DiffusionTTSServingPlan:
    """Resolved TTS diffusion support without constructing an engine client."""

    model_type: str
    capability: DiffusionServingCapability
    verified: bool
    experimental: bool = False
    plugin: VLLMOmniDiffusionPlugin | None = None

    @property
    def backend(self) -> DiffusionServingBackend:
        return self.capability.backend

    @property
    def uses_existing_llm_speech_bridge(self) -> bool:
        """Whether VoiceHub's existing Omni speech HTTP path is verified."""
        return (self.backend is DiffusionServingBackend.VLLM_OMNI and self.verified and not self.experimental)


def list_diffusion_serving_capabilities(
    *,
    supports_tts: bool | None = None,
    supports_visual_diffusion: bool | None = None,
) -> tuple[DiffusionServingCapability, ...]:
    """List serving facts without importing any optional engine."""
    if supports_tts is not None and not isinstance(supports_tts, bool):
        raise TypeError("`supports_tts` must be a boolean or None.")
    if (supports_visual_diffusion is not None and not isinstance(supports_visual_diffusion, bool)):
        raise TypeError("`supports_visual_diffusion` must be a boolean or None.")
    return tuple(
        _with_registered_models(capability) for capability in _CAPABILITIES
        if (supports_tts is None or capability.supports_tts is supports_tts) and (
            supports_visual_diffusion is None or
            capability.supports_visual_diffusion is supports_visual_diffusion))


def get_diffusion_serving_capability(backend: str | DiffusionServingBackend, ) -> DiffusionServingCapability:
    """Return one dependency-free backend capability record."""
    capability = _CAPABILITY_BY_BACKEND[DiffusionServingBackend.coerce(backend)]
    return _with_registered_models(capability)


def resolve_diffusion_tts_backend(
    model_type: str,
    backend: str | DiffusionServingBackend,
    *,
    plugin: VLLMOmniDiffusionPlugin | None = None,
) -> DiffusionTTSServingPlan:
    """Resolve complete TTS diffusion support, failing closed by default."""
    canonical_model = normalize_model_type(model_type)
    capability = get_diffusion_serving_capability(backend)

    if canonical_model in capability.verified_tts_models:
        if plugin is not None:
            raise ValueError("A verified TTS pairing does not accept an experimental "
                             "custom plugin.")
        return DiffusionTTSServingPlan(
            model_type=canonical_model,
            capability=capability,
            verified=True,
        )

    if capability.backend is DiffusionServingBackend.SGLANG_DIFFUSION:
        raise DiffusionServingCompatibilityError(
            "SGLang Diffusion reports supports_tts=False. Its visual "
            "image/video serving API cannot be used as a TTS audio backend.")

    if capability.backend is DiffusionServingBackend.SGLANG_OMNI:
        raise DiffusionServingCompatibilityError(
            "SGLang-Omni is a separate LLM-TTS runtime, not a diffusion-TTS "
            "runtime. Use voicehub.llm_serving and its verified model list.")

    if capability.backend is DiffusionServingBackend.VLLM_OMNI:
        if plugin is None:
            verified = ", ".join(capability.verified_tts_models)
            raise DiffusionServingCompatibilityError(
                f"vLLM-Omni has no verified complete TTS diffusion adapter "
                f"for {canonical_model!r}. Verified VoiceHub models: "
                f"{verified}. An explicit VLLMOmniDiffusionPlugin is required "
                "for experimental out-of-tree integration.")
        from voicehub.diffusion_serving.vllm_omni import VLLMOmniDiffusionPlugin

        if not isinstance(plugin, VLLMOmniDiffusionPlugin):
            raise TypeError("`plugin` must be a VLLMOmniDiffusionPlugin.")
        if plugin.model_type != canonical_model:
            raise DiffusionServingCompatibilityError(
                f"The plugin targets {plugin.model_type!r}, not "
                f"{canonical_model!r}.")
        if not plugin.complete_tts_pipeline:
            raise DiffusionServingCompatibilityError(
                "The experimental plugin does not declare a complete TTS "
                "pipeline. Registering a DiT alone does not provide text "
                "conditioning, audio post-processing, or codec/vocoder output.")
        return DiffusionTTSServingPlan(
            model_type=canonical_model,
            capability=capability,
            verified=False,
            experimental=True,
            plugin=plugin,
        )

    available = ", ".join(capability.verified_tts_models) or "none"
    raise DiffusionServingCompatibilityError(
        f"{capability.engine} has no diffusion-TTS support for "
        f"{canonical_model!r}. Supported models: {available}.")


__all__ = [
    "DiffusionServingBackend",
    "DiffusionServingCapability",
    "DiffusionServingCompatibilityError",
    "DiffusionTTSServingPlan",
    "get_diffusion_serving_capability",
    "list_diffusion_serving_capabilities",
    "resolve_diffusion_tts_backend",
]
