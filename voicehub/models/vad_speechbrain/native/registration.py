"""Lazy architecture declaration for native SpeechBrain CRDNN VAD."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

from .metadata import SPEECHBRAIN_TRAINING_SOURCE_REVISION, SPEECHBRAIN_VAD_CHECKPOINT_LICENSE, SPEECHBRAIN_VAD_REVISION

DEFAULT_SPEECHBRAIN_VAD_ALIASES = (
    "speechbrain-vad",
    "speechbrain-crdnn",
    "vad-crdnn-libriparty",
)


def create_speechbrain_vad_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="speechbrain-crdnn-vad",
        version="1",
        model_builder=("voicehub.models.vad_speechbrain.native.modeling:"
                       "SpeechBrainCRDNNVADModel"),
        config=("voicehub.models.vad_speechbrain.native.configuration:"
                "SpeechBrainCRDNNVADConfig"),
        decoder=("voicehub.models.vad_speechbrain.native.inference:"
                 "SpeechBrainVADInference"),
        objective=(
            "voicehub.models.vad_speechbrain.native.objective:"
            "speechbrain_vad_binary_cross_entropy"),
        checkpoint_adapter=(
            "voicehub.models.vad_speechbrain.native.checkpoint:"
            "SpeechBrainVADSafeTensorsCheckpointAdapter"),
        components={
            "frontend": ("voicehub.models.vad_speechbrain.native.frontend:"
                         "SpeechBrainVADFrontend"),
            "pickle-converter":
            ("voicehub.models.vad_speechbrain.native.checkpoint:"
             "convert_speechbrain_vad_checkpoint"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.VOICE_ACTIVITY_DETECTION, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", ),
            checkpoint_formats=(
                "safetensors",
                "trusted-pickle-conversion",
            ),
            training=True,
            streaming=False,
            batched_inference=True,
            features=(
                "legacy-speechbrain-fbank",
                "sentence-cmvn",
                "crdnn",
                "bidirectional-gru",
                "source-compatible-chunking",
                "hysteresis-segmentation",
                "raw-audio-fine-tuning",
                "masked-binary-cross-entropy",
            ),
        ),
        upstream_revision=SPEECHBRAIN_TRAINING_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family":
            "speechbrain-crdnn-vad",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "published_artifact_revision":
            SPEECHBRAIN_VAD_REVISION,
            "checkpoint_license":
            SPEECHBRAIN_VAD_CHECKPOINT_LICENSE,
            "training_boundary": (
                "The pinned SpeechBrain repository recipe is GRU-only and "
                "does not instantiate the published CRDNN checkpoint. "
                "VoiceHub preserves the published CRDNN graph and integrates "
                "the author recipe's frame BCE/label alignment, but does not "
                "claim exact reproduction of an unpublished CRDNN recipe."),
        },
    )


def register_speechbrain_vad_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_SPEECHBRAIN_VAD_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_speechbrain_vad_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_SPEECHBRAIN_VAD_ALIASES",
    "create_speechbrain_vad_architecture_spec",
    "register_speechbrain_vad_architecture",
]
