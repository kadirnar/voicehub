"""Lazy declaration for VoiceHub's native Irodori-TTS architecture."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.irodoritts.native.metadata import (
    IRODORI_CHECKPOINTS,
    IRODORI_CODEC_ID,
    IRODORI_CODEC_REVISION,
    IRODORI_SOURCE_LICENSE,
    IRODORI_SOURCE_REVISION,
    IRODORI_TOKENIZER_ID,
    IRODORI_TOKENIZER_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_IRODORI_ALIASES = (
    "irodori",
    "irodori-tts",
    "native-irodoritts",
)


def create_irodori_architecture_spec() -> ArchitectureSpec:
    """Describe the audited Irodori graph without importing PyTorch."""
    default_checkpoint = IRODORI_CHECKPOINTS["v3"]
    return ArchitectureSpec(
        architecture_id="irodoritts-rf-dit",
        version="3",
        model_builder=("voicehub.models.irodoritts.native.modeling:"
                       "TextToLatentRFDiT"),
        config=("voicehub.models.irodoritts.native.configuration:"
                "IrodoriModelConfig"),
        processor=("voicehub.models.irodoritts.native.training:"
                   "IrodoriBatchProcessor"),
        decoder=("voicehub.models.irodoritts.native.codec:"
                 "IrodoriDACVAECodec"),
        objective=("voicehub.models.irodoritts.native.training:"
                   "irodori_training_step"),
        checkpoint_adapter=("voicehub.models.irodoritts.native.checkpoint:"
                            "IrodoriCheckpointAdapter"),
        components={
            "checkpoint-exporter":
            ("voicehub.models.irodoritts.native.checkpoint:"
             "save_irodori_safetensors"),
            "duration-predictor": ("voicehub.models.irodoritts.native.modeling:"
                                   "DurationPredictor"),
            "runtime": ("voicehub.models.irodoritts.native.runtime:"
                        "InferenceRuntime"),
            "text-tokenizer": ("voicehub.models.irodoritts.native.tokenization:"
                               "IrodoriTokenizer"),
            "trainer-adapter": ("voicehub.models.irodoritts.training:"
                                "NativeIrodoriTrainingAdapter"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=(
                "compile",
                "sdpa",
                "custom-kernels",
                "diffusion-cache",
                "diffusion-sampling",
            ),
            features=(
                "diffusion-family",
                "diffusion-serving-native",
                "diffusion-kind-rectified-flow",
                "diffusion-operation-denoiser",
                "diffusion-operation-classifier-free-guidance",
                "diffusion-operation-euler-solver",
                "diffusion-sampling-schedule",
                "diffusion-sampling-guidance",
                "diffusion-sampling-prediction-cache",
                "48-khz-waveform",
                "caption-conditioning",
                "classifier-free-guidance",
                "duration-prediction",
                "fused-diffusion-modulation-kernels",
                "full-model-finetuning",
                "native-dacvae",
                "raw-audio-finetuning",
                "rectified-flow-matching",
                "reference-voice-conditioning",
                "strict-safetensors-reload",
                "unigram-byte-fallback-tokenizer",
                "voice-design",
            ),
        ),
        upstream_revision=IRODORI_SOURCE_REVISION,
        license_id=IRODORI_SOURCE_LICENSE,
        metadata={
            "diffusion_architecture_kind":
            "rectified-flow",
            "diffusion_operations": (
                "denoiser",
                "classifier-free-guidance",
                "euler-solver",
            ),
            "diffusion_sampling_capabilities": (
                "schedule",
                "guidance",
                "prediction-cache",
            ),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            "https://github.com/Aratako/Irodori-TTS",
            "source_revision":
            IRODORI_SOURCE_REVISION,
            "reference_checkpoint":
            default_checkpoint["model_id"],
            "reference_checkpoint_revision":
            default_checkpoint["revision"],
            "reference_checkpoint_sha256":
            default_checkpoint["lfs_sha256"],
            "reference_tensor_count":
            default_checkpoint["tensors"],
            "reference_parameter_count":
            default_checkpoint["parameters"],
            "reference_safetensors_header_fingerprint": (default_checkpoint["header_fingerprint"]),
            "codec_checkpoint":
            IRODORI_CODEC_ID,
            "codec_checkpoint_revision":
            IRODORI_CODEC_REVISION,
            "text_tokenizer":
            IRODORI_TOKENIZER_ID,
            "text_tokenizer_revision":
            IRODORI_TOKENIZER_REVISION,
            "full_finetuning_ready":
            True,
            "training_boundary": (
                "All RF-DiT and optional duration-predictor parameters are "
                "trainable with the released rectified-flow objective. The "
                "Semantic-DACVAE remains frozen and supplies raw-audio "
                "targets; pre-encoded continuous latents are also accepted."),
            "inference_boundary": (
                "The native runtime covers v2, v3, and VoiceDesign "
                "safetensors checkpoints. SilentCipher watermark parity is "
                "not claimed."),
        },
    )


def register_irodori_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_IRODORI_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register the native Irodori declaration."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_irodori_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_IRODORI_ALIASES",
    "create_irodori_architecture_spec",
    "register_irodori_architecture",
]
