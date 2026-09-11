"""Lazy architecture declaration for VoiceHub-native CosyVoice."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.cosyvoice.native.metadata import (
    COSYVOICE3_MODEL_ID,
    COSYVOICE3_MODEL_REVISION,
    COSYVOICE_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_COSYVOICE_ALIASES = (
    "cosyvoice",
    "cosyvoice3",
    "native-cosyvoice",
)


def create_cosyvoice_architecture_spec() -> ArchitectureSpec:
    """Describe the native graph without importing PyTorch."""
    return ArchitectureSpec(
        architecture_id="cosyvoice-native",
        version="3",
        model_builder=("voicehub.models.cosyvoice.native.modeling:"
                       "CosyVoiceNativeModel"),
        config=("voicehub.models.cosyvoice.native.configuration:"
                "CosyVoiceArchitectureConfig"),
        processor=("voicehub.models.cosyvoice.native.tokenization:"
                   "CosyVoiceTextTokenizer"),
        decoder=("voicehub.models.cosyvoice.native.vocoder:"
                 "CosyVoiceHiFTGenerator"),
        objective=("voicehub.models.cosyvoice.native.modeling:"
                   "CosyVoiceNativeModel.forward"),
        checkpoint_adapter=("voicehub.models.cosyvoice.native.checkpoint:"
                            "load_cosyvoice_checkpoint"),
        components={
            "artifact-resolver":
            ("voicehub.models.cosyvoice.native.artifacts:"
             "resolve_cosyvoice_artifacts"),
            "flow-matcher": ("voicehub.models.cosyvoice.native.flow:"
                             "CosyVoiceFlowMatchingModel"),
            "language-model": ("voicehub.models.cosyvoice.native.language_model:"
                               "CosyVoiceLanguageModel"),
            "legacy-converter":
            ("voicehub.models.cosyvoice.native.checkpoint:"
             "convert_audited_cosyvoice_legacy_checkpoint"),
            "speech-tokenizer":
            ("voicehub.models.cosyvoice.native.speech_tokenizer:"
             "CosyVoiceSpeechTokenizer"),
            "runtime": ("voicehub.models.cosyvoice.native.runtime:"
                        "load_cosyvoice_runtime"),
            "trainer-adapter": ("voicehub.models.cosyvoice.training_cosyvoice:"
                                "CosyVoiceTrainingAdapter"),
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
                "custom-kernels",
                "diffusion-cache",
                "diffusion-sampling",
            ),
            features=(
                "llm-tts-codec",
                "diffusion-family",
                "diffusion-serving-native",
                "diffusion-serving-vllm-omni",
                "diffusion-kind-conditional-flow-matching",
                "diffusion-operation-denoiser",
                "diffusion-operation-classifier-free-guidance",
                "diffusion-operation-euler-solver",
                "diffusion-sampling-schedule",
                "diffusion-sampling-guidance",
                "diffusion-sampling-prediction-cache",
                "diffusion-sampling-stork2",
                "audited-legacy-conversion",
                "causal-hift",
                "fused-diffusion-modulation-kernels",
                "family-extensible-component-boundaries",
                "flow-matching-finetuning",
                "full-llm-finetuning",
                "hifigan-adversarial-finetuning",
                "multilingual",
                "native-qwen2",
                "native-s3tokenizer-v3-encoder",
                "native-s3tokenizer-v3-fsq",
                "raw-audio-speech-token-extraction",
                "strict-safetensors",
                "voice-cloning-with-precomputed-prompt",
            ),
        ),
        upstream_revision=COSYVOICE_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "diffusion_architecture_kind":
            "conditional-flow-matching",
            "diffusion_operations": (
                "denoiser",
                "classifier-free-guidance",
                "euler-solver",
            ),
            "diffusion_sampling_capabilities": (
                "schedule",
                "guidance",
                "prediction-cache",
                "stork2",
            ),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "reference_checkpoint":
            COSYVOICE3_MODEL_ID,
            "reference_checkpoint_revision":
            COSYVOICE3_MODEL_REVISION,
            "checkpoint_license":
            "Apache-2.0",
            "executable_checkpoint_compatibility":
            "cosyvoice3-only",
            "training_boundary": (
                "LM, conditional flow matcher, and HiFT generator/"
                "discriminator are trainable with their author objectives. "
                "Text and speech-token frontends remain frozen; the native "
                "S3Tokenizer is optional because precomputed speech tokens "
                "remain a supported dataset boundary."),
            "speech_tokenizer_boundary": (
                "The published ONNX graph is never executed at runtime. Its "
                "fully audited encoder and FSQ initializers require explicit "
                "one-time conversion to strict Safetensors."),
            "parity_boundary": (
                "Official tensor inventories are exact. Numerical waveform "
                "parity is not claimed without checkpoint-level evidence."),
        },
    )


def register_cosyvoice_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_COSYVOICE_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_cosyvoice_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_COSYVOICE_ALIASES",
    "create_cosyvoice_architecture_spec",
    "register_cosyvoice_architecture",
]
