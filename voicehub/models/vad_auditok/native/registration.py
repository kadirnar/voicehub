"""Lazy declaration for the native short-term-energy detector."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

AUDITOK_REFERENCE_REVISION = "833ae725aef73a489366cc5940b831e16223059f"
DEFAULT_ENERGY_VAD_ALIASES = ("auditok-energy", "short-term-energy")


def create_energy_vad_architecture_spec() -> ArchitectureSpec:
    """Create the immutable energy-detector architecture declaration."""
    return ArchitectureSpec(
        architecture_id="energy-vad",
        version="1",
        model_builder=("voicehub.models.vad_auditok.native.modeling:"
                       "EnergyVoiceActivityDetector"),
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.VOICE_ACTIVITY_DETECTION, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", ),
            checkpoint_formats=("none", ),
            training=False,
            streaming=False,
            batched_inference=False,
            features=(
                "algorithmic",
                "fixed-threshold",
                "otsu-threshold",
                "percentile-threshold",
            ),
        ),
        upstream_revision=AUDITOK_REFERENCE_REVISION,
        license_id="MIT",
        metadata={
            "family": "short-term-energy",
            "implementation": "voicehub-native",
            "reference_source": ("https://github.com/amsehili/auditok/tree/"
                                 f"{AUDITOK_REFERENCE_REVISION}"),
        },
    )


def register_energy_vad_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_ENERGY_VAD_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register and return the native energy-detector specification."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_energy_vad_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "AUDITOK_REFERENCE_REVISION",
    "DEFAULT_ENERGY_VAD_ALIASES",
    "create_energy_vad_architecture_spec",
    "register_energy_vad_architecture",
]
