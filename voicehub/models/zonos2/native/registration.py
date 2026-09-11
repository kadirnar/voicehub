"""Lazy declaration for the VoiceHub-native ZONOS2 architecture."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.zonos2.native.metadata import (
    ZONOS2_OFFICIAL_CHECKPOINT,
    ZONOS2_OFFICIAL_CHECKPOINT_LICENSE,
    ZONOS2_OFFICIAL_CHECKPOINT_REVISION,
    ZONOS2_PARAMETER_COUNT,
    ZONOS2_SAFE_CONVERSION,
    ZONOS2_SAFE_CONVERSION_REVISION,
    ZONOS2_SOURCE,
    ZONOS2_SOURCE_LICENSE,
    ZONOS2_SOURCE_REVISION,
    ZONOS2_TENSOR_COUNT,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_ZONOS2_ALIASES = (
    "zonos-2",
    "native-zonos2",
    "zonos2-sonic-moe",
)


def create_zonos2_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="zonos2",
        version="1",
        model_builder=("voicehub.models.zonos2.native.modeling:Zonos2ForCausalLM"),
        config=("voicehub.models.zonos2.native.configuration:"
                "Zonos2ArchitectureConfig"),
        processor=("voicehub.models.zonos2.native.prompting:build_zonos2_prompt"),
        decoder=("voicehub.components.audio.codecs.dac.model.dac:DAC"),
        objective=("voicehub.models.zonos2.native.objective:"
                   "zonos2_causal_cross_entropy"),
        checkpoint_adapter=("voicehub.models.zonos2.native.checkpoint:"
                            "load_zonos2_checkpoint"),
        components={
            "artifact-resolver": ("voicehub.models.zonos2.native.artifacts:"
                                  "resolve_zonos2_artifacts"),
            "checkpoint-exporter": ("voicehub.models.zonos2.native.checkpoint:"
                                    "export_zonos2_checkpoint"),
            "speaker-encoder": ("voicehub.models.zonos2.native.speaker:"
                                "load_zonos2_speaker_encoder"),
            "runtime": ("voicehub.models.zonos2.native.runtime:"
                        "NativeZonos2Runtime"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=False,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=("compile", "sdpa"),
            features=(
                "llm-tts-codec",
                "raw-utf8-text",
                "nine-delayed-dac-codebooks",
                "sonic-mixture-of-experts",
                "speaker-cloning",
                "speaking-rate-conditioning",
                "quality-conditioning",
                "full-model-gradients",
                "strict-safetensors-reload",
            ),
        ),
        upstream_revision=ZONOS2_SOURCE_REVISION,
        license_id=ZONOS2_SOURCE_LICENSE,
        metadata={
            "external_llm_backend_blocker": ("Zonos2 requires a custom multi-stream engine model."),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            ZONOS2_SOURCE,
            "reference_checkpoint":
            ZONOS2_OFFICIAL_CHECKPOINT,
            "reference_checkpoint_revision": (ZONOS2_OFFICIAL_CHECKPOINT_REVISION),
            "checkpoint_license":
            ZONOS2_OFFICIAL_CHECKPOINT_LICENSE,
            "safe_conversion":
            ZONOS2_SAFE_CONVERSION,
            "safe_conversion_revision":
            ZONOS2_SAFE_CONVERSION_REVISION,
            "reference_tensor_count":
            ZONOS2_TENSOR_COUNT,
            "reference_parameter_count":
            ZONOS2_PARAMETER_COUNT,
            "official_safetensors_published":
            False,
            "training_boundary": (
                "The differentiable graph and next-row codebook CE are "
                "verified. Zyphra has not published its original optimizer, "
                "dataset pipeline, or training loop; the VoiceHub objective "
                "is explicitly reconstructed."),
            "full_model_gradient_ready":
            True,
            "author_verified_training_recipe":
            False,
        },
    )


def register_zonos2_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_ZONOS2_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_zonos2_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_ZONOS2_ALIASES",
    "create_zonos2_architecture_spec",
    "register_zonos2_architecture",
]
