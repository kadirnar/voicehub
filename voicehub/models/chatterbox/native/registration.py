"""Lazy declaration for VoiceHub-native Chatterbox."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.chatterbox.native.metadata import (
    CHATTERBOX_CHECKPOINT,
    CHATTERBOX_CHECKPOINT_LICENSE,
    CHATTERBOX_CHECKPOINT_REVISION,
    CHATTERBOX_COMMUNITY_TRAINING_LICENSE,
    CHATTERBOX_COMMUNITY_TRAINING_REVISION,
    CHATTERBOX_COMMUNITY_TRAINING_SOURCE,
    CHATTERBOX_COMPONENT_INVENTORIES,
    CHATTERBOX_PARAMETER_COUNT,
    CHATTERBOX_SOURCE,
    CHATTERBOX_SOURCE_LICENSE,
    CHATTERBOX_SOURCE_REVISION,
    CHATTERBOX_TENSOR_COUNT,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_CHATTERBOX_ALIASES = (
    "native-chatterbox",
    "chatterbox-tts",
    "resemble-chatterbox",
)


def create_chatterbox_architecture_spec() -> ArchitectureSpec:
    """Describe the verified English Chatterbox runtime without importing
    it."""
    return ArchitectureSpec(
        architecture_id="chatterbox",
        version="1",
        model_builder=("voicehub.models.chatterbox.modeling:"
                       "ChatterboxForTextToSpeech"),
        config=("voicehub.models.chatterbox.modeling:"
                "ChatterboxConfig"),
        processor=("voicehub.models.chatterbox.models.tokenizers:"
                   "EnTokenizer"),
        decoder="voicehub.models.chatterbox.models.s3gen:S3Gen",
        objective=("voicehub.models.chatterbox.training:"
                   "ChatterboxTrainingAdapter"),
        checkpoint_adapter=("voicehub.models.chatterbox.checkpoint:"
                            "load_module_safetensors"),
        components={
            "t3": "voicehub.models.chatterbox.models.t3:T3",
            "s3tokenizer": ("voicehub.models.chatterbox.models.s3tokenizer:"
                            "S3Tokenizer"),
            "voice-encoder": ("voicehub.models.chatterbox.models.voice_encoder:"
                              "VoiceEncoder"),
            "watermark": ("voicehub.models.chatterbox.watermark:"
                          "NativePerthWatermarker"),
            "checkpoint-exporter": ("voicehub.models.chatterbox.checkpoint:"
                                    "export_chatterbox_runtime"),
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
            optimization_passes=(
                "compile",
                "sdpa",
                "diffusion-cache",
                "diffusion-sampling",
            ),
            features=(
                "llm-tts-codec",
                "diffusion-family",
                "diffusion-serving-native",
                "diffusion-kind-conditional-flow-matching",
                "diffusion-operation-denoiser",
                "diffusion-operation-classifier-free-guidance",
                "diffusion-operation-euler-solver",
                "diffusion-sampling-schedule",
                "diffusion-sampling-prediction-cache",
                "zero-shot-voice-cloning",
                "native-bpe-tokenizer",
                "native-s3tokenizer",
                "native-perth-watermark",
                "t3-causal-token-objective",
                "s3gen-flow-matching-objective",
                "raw-audio-fine-tuning",
                "lora-fine-tuning",
                "strict-safetensors-reload",
            ),
        ),
        upstream_revision=CHATTERBOX_SOURCE_REVISION,
        license_id=CHATTERBOX_SOURCE_LICENSE,
        metadata={
            "external_llm_backend_blocker": ("Chatterbox requires prompt embeddings and synchronized CFG."),
            "diffusion_architecture_kind":
            "conditional-flow-matching",
            "diffusion_operations": (
                "denoiser",
                "classifier-free-guidance",
                "euler-solver",
            ),
            "diffusion_sampling_capabilities": (
                "schedule",
                "prediction-cache",
            ),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            CHATTERBOX_SOURCE,
            "reference_checkpoint":
            CHATTERBOX_CHECKPOINT,
            "reference_checkpoint_revision": (CHATTERBOX_CHECKPOINT_REVISION),
            "checkpoint_license":
            CHATTERBOX_CHECKPOINT_LICENSE,
            "reference_tensor_count":
            CHATTERBOX_TENSOR_COUNT,
            "reference_parameter_count":
            CHATTERBOX_PARAMETER_COUNT,
            "component_inventories":
            CHATTERBOX_COMPONENT_INVENTORIES,
            "community_training_source": (CHATTERBOX_COMMUNITY_TRAINING_SOURCE),
            "community_training_revision": (CHATTERBOX_COMMUNITY_TRAINING_REVISION),
            "community_training_license": (CHATTERBOX_COMMUNITY_TRAINING_LICENSE),
            "training_boundary": (
                "T3 token cross-entropy and S3Gen conditional flow matching "
                "are separate optimizer jobs. The released objectives and "
                "community data contract are integrated, with VoiceHub "
                "corrections for causal shifting, prompt masking, padded "
                "attention keys, and frozen frontend evaluation. Resemble AI "
                "did not publish a complete end-to-end optimizer recipe."),
            "author_end_to_end_recipe_published":
            False,
            "full_finetuning_ready":
            True,
        },
    )


def register_chatterbox_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_CHATTERBOX_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register native Chatterbox in a target architecture registry."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_chatterbox_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_CHATTERBOX_ALIASES",
    "create_chatterbox_architecture_spec",
    "register_chatterbox_architecture",
]
