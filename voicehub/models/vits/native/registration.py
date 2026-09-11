"""Lazy declaration for VoiceHub's native VITS and MMS-TTS family."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.vits.native.checkpoint import (
    FACEBOOK_MMS_TTS_ENG_REVISION,
    ORIGINAL_VITS_REVISION,
    TRANSFORMERS_VITS_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_VITS_ALIASES = (
    "native-vits",
    "mms-tts",
    "hf-vits",
)


def create_vits_architecture_spec() -> ArchitectureSpec:
    """Create the immutable native VITS declaration."""
    return ArchitectureSpec(
        architecture_id="vits",
        version="1",
        model_builder="voicehub.models.vits.native.modeling:VitsModel",
        config="voicehub.models.vits.native.configuration:VitsConfig",
        processor="voicehub.models.vits.native.frontend:VitsTokenizer",
        objective="voicehub.models.vits.native.losses:VitsGeneratorLoss",
        checkpoint_adapter=("voicehub.models.vits.native.checkpoint:"
                            "HuggingFaceVitsCheckpointAdapter"),
        components={
            "alignment-search": ("voicehub.models.vits.native.alignment:maximum_path"),
            "discriminator": ("voicehub.models.vits.native.losses:"
                              "VitsMultiPeriodDiscriminator"),
            "native-checkpoint-adapter":
            ("voicehub.models.vits.native.checkpoint:"
             "NativeVitsCheckpointAdapter"),
            "training-support": ("voicehub.models.vits.native.losses:"
                                 "VITS_TRAINING_SUPPORT"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            optimization_passes=("compile", "custom-kernels"),
            features=(
                "vits-family",
                "vits-wavenet-gate",
                "character-frontend",
                "declarative-language-providers",
                "monotonic-alignment-search",
                "posterior-encoder",
                "stochastic-duration-prediction",
                "torch-compile-inference-unsafe",
                "normalizing-flow",
                "hifigan-decoder",
                "multi-period-discriminator",
                "checkpoint-conversion",
                "full-adversarial-fine-tuning",
                "native-acoustic-frontend",
                "fine-tuning-requires-explicit-acoustic-config",
            ),
        ),
        upstream_revision=ORIGINAL_VITS_REVISION,
        license_id="MIT",
        metadata={
            "family":
            "vits",
            "vits_architecture_kind":
            "classic",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "full_finetuning_ready":
            True,
            "training_boundary": (
                "The generator, random-initialized source discriminator, "
                "acoustic frontend, and two optimizer phases are native. "
                "Checkpoint-specific FFT/mel settings remain an explicit "
                "recipe input because MMS-TTS metadata omits them."),
            "original_source": ("https://github.com/jaywalnut310/vits/tree/"
                                f"{ORIGINAL_VITS_REVISION}"),
            "transformers_reference_revision":
            TRANSFORMERS_VITS_REVISION,
            "transformers_source": (
                "https://github.com/huggingface/transformers/tree/"
                f"{TRANSFORMERS_VITS_REVISION}/src/transformers/models/vits"),
            "reference_checkpoint":
            "facebook/mms-tts-eng",
            "reference_checkpoint_revision":
            FACEBOOK_MMS_TTS_ENG_REVISION,
            "reference_checkpoint_weight_license":
            "CC-BY-NC-4.0",
            "reference_checkpoint_source":
            ("https://huggingface.co/facebook/mms-tts-eng/tree/"
             f"{FACEBOOK_MMS_TTS_ENG_REVISION}"),
        },
    )


def register_vits_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_VITS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register the lazy VITS declaration and return it."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_vits_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_VITS_ALIASES",
    "create_vits_architecture_spec",
    "register_vits_architecture",
]
