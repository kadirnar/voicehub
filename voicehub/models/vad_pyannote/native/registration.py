"""Lazy architecture declaration for VoiceHub's native PyanNet family."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

from .metadata import BROUHAHA_SOURCE_REVISION, PYANNOTE_AUDIO_3_SOURCE_REVISION

DEFAULT_PYANNET_ALIASES = (
    "native-pyannet",
    "pyannote-segmentation",
    "pyannote-powerset-segmentation",
    "pyannote-brouhaha",
)


def create_pyannet_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="pyannet",
        version="1",
        model_builder="voicehub.models.vad_pyannote.native.modeling:PyanNet",
        config=("voicehub.models.vad_pyannote.native.configuration:PyanNetConfig"),
        decoder=("voicehub.models.vad_pyannote.native.inference:"
                 "PyanNetFrameInference"),
        objective=("voicehub.models.vad_pyannote.native.objective:pyannet_loss"),
        checkpoint_adapter=(
            "voicehub.models.vad_pyannote.native.checkpoint:"
            "PyanNetSafeTensorsCheckpointAdapter"),
        components={
            "powerset": ("voicehub.models.vad_pyannote.native.powerset:Powerset"),
            "lightning-converter":
            ("voicehub.models.vad_pyannote.native.checkpoint:"
             "convert_pyannote_lightning_checkpoint"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.VOICE_ACTIVITY_DETECTION, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", ),
            checkpoint_formats=(
                "safetensors",
                "trusted-lightning-conversion",
            ),
            training=True,
            streaming=False,
            batched_inference=True,
            features=(
                "sincnet",
                "powerset",
                "hamming-overlap-add",
                "hysteresis-segmentation",
                "snr-regression",
                "c50-regression",
                "frame-fine-tuning",
            ),
        ),
        upstream_revision=PYANNOTE_AUDIO_3_SOURCE_REVISION,
        license_id="MIT",
        metadata={
            "family":
            "pyannet",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "pyannote_source_revision":
            PYANNOTE_AUDIO_3_SOURCE_REVISION,
            "brouhaha_source_revision":
            BROUHAHA_SOURCE_REVISION,
            "checkpoint_boundary": (
                "Official artifacts are Lightning pickle files. VoiceHub "
                "requires explicit one-time restricted conversion and uses "
                "Safetensors for runtime and fine-tuning thereafter."),
            "brouhaha_checkpoint_license":
            "OpenRAIL",
        },
    )


def register_pyannet_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_PYANNET_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_pyannet_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_PYANNET_ALIASES",
    "create_pyannet_architecture_spec",
    "register_pyannet_architecture",
]
