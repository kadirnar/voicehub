"""Lazy declaration for the native WebRTC fixed-point detector."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.vad_webrtc.native.metadata import PY_WEBRTCVAD_SOURCE_REVISION
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_WEBRTC_VAD_ALIASES = ("native-webrtc-vad", "webrtc-gmm")


def create_webrtc_vad_architecture_spec() -> ArchitectureSpec:
    """Create the immutable native WebRTC VAD architecture declaration."""
    return ArchitectureSpec(
        architecture_id="webrtc-vad",
        version="1",
        model_builder=("voicehub.models.vad_webrtc.native.detector:NativeWebRTCVAD"),
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.VOICE_ACTIVITY_DETECTION, ),
            devices=("cpu", ),
            dtypes=("int16", ),
            checkpoint_formats=("none", ),
            training=False,
            streaming=True,
            batched_inference=False,
            features=(
                "fixed-point",
                "six-band-filterbank",
                "adaptive-gmm",
                "hangover-smoothing",
                "8khz",
                "16khz",
                "32khz",
                "48khz",
            ),
        ),
        upstream_revision=PY_WEBRTCVAD_SOURCE_REVISION,
        license_id="BSD-3-Clause",
        metadata={
            "family":
            "webrtc-vad",
            "implementation":
            "voicehub-native",
            "training_boundary": (
                "WebRTC VAD is an adaptive fixed-point algorithm with no "
                "trainable autograd parameters; fine-tuning is not applicable."),
        },
    )


def register_webrtc_vad_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_WEBRTC_VAD_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register and return the native WebRTC VAD specification."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_webrtc_vad_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_WEBRTC_VAD_ALIASES",
    "create_webrtc_vad_architecture_spec",
    "register_webrtc_vad_architecture",
]
