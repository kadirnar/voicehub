"""Declarative architecture bundle for native generic VAD dispatch.

The dispatcher is a closed bundle of independently registered VoiceHub
architectures. Configuration selects a reviewed family; serialized
metadata cannot introduce arbitrary implementation imports.
"""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_VAD_DISPATCH_ALIASES = (
    "native-vad",
    "generic-native-vad",
)


def create_vad_dispatch_architecture_spec() -> ArchitectureSpec:
    """Create the lazy declaration for the verified native VAD bundle."""
    return ArchitectureSpec(
        architecture_id="native-vad-dispatch",
        version="1",
        model_builder=("voicehub.models.vad_sherpa_onnx.modeling:"
                       "SherpaONNXVADForVoiceActivityDetection"),
        config=("voicehub.models.vad_sherpa_onnx.configuration:"
                "SherpaONNXVADConfig"),
        components={
            "silero-vad": ("voicehub.models.vad_silero.native.modeling:"
                           "SileroVADModel"),
            "ten-vad": "voicehub.models.vad_ten.native.modeling:TENVADModel",
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.VOICE_ACTIVITY_DETECTION, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", ),
            checkpoint_formats=(
                "safetensors",
                "torchscript-state-dict",
                "explicit-onnx-weight-conversion",
            ),
            training=True,
            streaming=True,
            batched_inference=True,
            distributed_training=False,
            features=(
                "closed-dispatch",
                "silero-vad",
                "ten-vad",
                "strict-config-resolution",
                "sherpa-compatible-segmentation",
                "frame-probabilities",
            ),
        ),
        license_id="Apache-2.0",
        metadata={
            "implementation": "voicehub-native",
            "tensor_backend": "pytorch",
            "dispatch_policy": "closed-verified-families",
            "families": (
                "silero-vad",
                "ten-vad",
            ),
            "family_licenses": {
                "silero-vad": "MIT",
                "ten-vad": "LicenseRef-TEN-VAD-Open-Source-License",
            },
        },
    )


def register_vad_dispatch_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_VAD_DISPATCH_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register the native VAD dispatcher without resolving model graphs."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_vad_dispatch_architecture_spec()
    target.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )
    return spec


__all__ = [
    "DEFAULT_VAD_DISPATCH_ALIASES",
    "create_vad_dispatch_architecture_spec",
    "register_vad_dispatch_architecture",
]
