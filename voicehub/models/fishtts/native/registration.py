"""Lazy architecture declaration for VoiceHub-native Fish Speech S2."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.fishtts.native.metadata import (
    FISH_S2_CHECKPOINT,
    FISH_S2_CHECKPOINT_LICENSE,
    FISH_S2_CHECKPOINT_REVISION,
    FISH_S2_PARAMETER_COUNT,
    FISH_S2_TENSOR_COUNT,
    FISH_SPEECH_SOURCE,
    FISH_SPEECH_SOURCE_LICENSE,
    FISH_SPEECH_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_FISH_S2_ALIASES = (
    "fishtts",
    "fish-speech",
    "s2-pro",
)


def create_fish_s2_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="fish-s2",
        version="1",
        model_builder=("voicehub.models.fishtts.native.modeling:"
                       "FishS2ForConditionalGeneration"),
        config=("voicehub.models.fishtts.native.configuration:FishS2Config"),
        processor=("voicehub.models.fishtts.native.prompting:build_fish_prompt"),
        decoder=("voicehub.models.fishtts.native.codec:FishModifiedDAC"),
        objective=("voicehub.models.fishtts.training:"
                   "FishSpeechTrainingAdapter.compute_source_losses"),
        checkpoint_adapter=("voicehub.models.fishtts.native.checkpoint:"
                            "load_fish_semantic_checkpoint"),
        components={
            "artifact-resolver":
            ("voicehub.models.fishtts.native.artifacts:"
             "resolve_fish_semantic_artifacts"),
            "checkpoint-exporter":
            ("voicehub.models.fishtts.native.checkpoint:"
             "export_fish_semantic_checkpoint"),
            "codec-checkpoint": ("voicehub.models.fishtts.native.checkpoint:"
                                 "load_fish_codec_checkpoint"),
            "legacy-codec-converter":
            ("voicehub.models.fishtts.native.checkpoint:"
             "convert_legacy_fish_codec"),
            "runtime": ("voicehub.models.fishtts.native.runtime:FishS2Runtime"),
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
                "dual-autoregressive-generation",
                "ten-codebook-modified-dac",
                "multilingual-inline-control",
                "speaker-cloning",
                "semantic-full-model-gradients",
                "strict-safetensors-reload",
                "frozen-codec-training-boundary",
            ),
        ),
        upstream_revision=FISH_SPEECH_SOURCE_REVISION,
        license_id=FISH_SPEECH_SOURCE_LICENSE,
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            FISH_SPEECH_SOURCE,
            "reference_checkpoint":
            FISH_S2_CHECKPOINT,
            "reference_checkpoint_revision":
            FISH_S2_CHECKPOINT_REVISION,
            "checkpoint_license":
            FISH_S2_CHECKPOINT_LICENSE,
            "commercial_use":
            False,
            "reference_tensor_count":
            FISH_S2_TENSOR_COUNT,
            "reference_parameter_count":
            FISH_S2_PARAMETER_COUNT,
            "official_semantic_safetensors_published":
            True,
            "official_codec_safetensors_published":
            False,
            "training_boundary": (
                "The source-faithful slow-token and fast residual-codebook "
                "objectives are trainable. ModifiedDAC remains the released "
                "offline tokenizer and is frozen by the integrated recipe."),
            "full_model_gradient_ready":
            True,
            "author_verified_training_recipe":
            True,
        },
    )


def register_fish_s2_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_FISH_S2_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_fish_s2_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_FISH_S2_ALIASES",
    "create_fish_s2_architecture_spec",
    "register_fish_s2_architecture",
]
