"""Lazy architecture declarations for native Microsoft VibeVoice models."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.vibevoice.native.metadata import (
    MICROSOFT_VIBEVOICE_LICENSE,
    MICROSOFT_VIBEVOICE_SOURCE,
    MICROSOFT_VIBEVOICE_SOURCE_REVISION,
    TRANSFORMERS_LICENSE,
    TRANSFORMERS_SOURCE,
    TRANSFORMERS_VIBEVOICE_ASR_AUDITED_REVISION,
    VIBEVOICE_ASR_REPOSITORY,
    VIBEVOICE_CHECKPOINTS,
    VIBEVOICE_REALTIME_REPOSITORY,
    VIBEVOICE_TTS_REPOSITORY,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_VIBEVOICE_ASR_ALIASES = (
    "native-vibevoice-asr",
    "microsoft-vibevoice-asr",
)
DEFAULT_VIBEVOICE_TTS_ALIASES = (
    "native-vibevoice-tts",
    "microsoft-vibevoice",
    "vibevoice",
)


def create_vibevoice_asr_architecture_spec() -> ArchitectureSpec:
    """Describe the audited ASR graph without importing PyTorch."""
    checkpoint = VIBEVOICE_CHECKPOINTS[VIBEVOICE_ASR_REPOSITORY]
    return ArchitectureSpec(
        architecture_id="vibevoice-asr",
        version="1",
        model_builder=("voicehub.models.vibevoice.native.modeling:"
                       "VibeVoiceASRForConditionalGeneration"),
        config=("voicehub.models.vibevoice.native.configuration:"
                "VibeVoiceASRConfig"),
        processor=("voicehub.models.vibevoice.native.processing:"
                   "VibeVoiceASRProcessor"),
        objective="torch.nn.functional:cross_entropy",
        checkpoint_adapter=("voicehub.models.vibevoice.native.checkpoint:"
                            "VibeVoiceCheckpointAdapter"),
        components={
            "artifact-resolver":
            ("voicehub.models.vibevoice.native.artifacts:"
             "resolve_vibevoice_artifacts"),
            "runtime": ("voicehub.models.vibevoice.native.runtime:"
                        "VibeVoiceRuntime"),
            "tokenizer": ("voicehub.models.vibevoice.native.tokenization:"
                          "VibeVoiceTokenizer"),
            "acoustic-encoder":
            ("voicehub.models.vibevoice.native.asr_codec:"
             "VibeVoiceASRTokenizerEncoder"),
            "trainer-adapter":
            ("voicehub.models.asr_vibevoice.training_asr_vibevoice:"
             "NativeVibeVoiceASRTrainingAdapter"),
            "checkpoint-exporter":
            ("voicehub.models.vibevoice.native.checkpoint:"
             "export_vibevoice_checkpoint"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
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
                "causal-multimodal-language-model",
                "continuous-acoustic-encoder",
                "continuous-semantic-encoder",
                "speaker-attributed-segments",
                "segment-timestamps",
                "prompt-context",
                "greedy-generation",
                "strict-sharded-safetensors",
                "portable-native-export",
                "no-external-model-runtime",
            ),
        ),
        upstream_revision=TRANSFORMERS_VIBEVOICE_ASR_AUDITED_REVISION,
        license_id=TRANSFORMERS_LICENSE,
        metadata={
            "implementation": "voicehub-native",
            "tensor_backend": "pytorch",
            "architecture_source": TRANSFORMERS_SOURCE,
            "architecture_source_revision": (TRANSFORMERS_VIBEVOICE_ASR_AUDITED_REVISION),
            "family_source": MICROSOFT_VIBEVOICE_SOURCE,
            "family_source_revision": MICROSOFT_VIBEVOICE_SOURCE_REVISION,
            "reference_checkpoint": VIBEVOICE_ASR_REPOSITORY,
            "reference_checkpoint_revision": checkpoint["revision"],
            "reference_checkpoint_license": checkpoint["license"],
            "reference_tensor_count": checkpoint["tensors"],
            "reference_parameter_count": checkpoint["parameters"],
            "reference_header_fingerprint": checkpoint["header_fingerprint"],
            "generation_scope": "greedy-autoregressive",
            "streaming_scope": "not-exposed",
        },
    )


def create_vibevoice_tts_architecture_spec() -> ArchitectureSpec:
    """Describe the audited non-streaming and realtime TTS graphs lazily."""
    training_checkpoint = VIBEVOICE_CHECKPOINTS[VIBEVOICE_TTS_REPOSITORY]
    realtime_checkpoint = VIBEVOICE_CHECKPOINTS[VIBEVOICE_REALTIME_REPOSITORY]
    return ArchitectureSpec(
        architecture_id="vibevoice-tts",
        version="1",
        model_builder=("voicehub.models.vibevoice.native.checkpoint:"
                       "build_vibevoice_model"),
        config=("voicehub.models.vibevoice.native.configuration:"
                "parse_vibevoice_config"),
        processor=("voicehub.models.vibevoice.native.processing:"
                   "VibeVoiceTTSProcessor"),
        decoder=("voicehub.models.vibevoice.native.codec:"
                 "VibeVoiceAcousticTokenizer"),
        objective=("voicehub.models.vibevoice.native.modeling:"
                   "VibeVoiceForConditionalGeneration"),
        checkpoint_adapter=("voicehub.models.vibevoice.native.checkpoint:"
                            "VibeVoiceCheckpointAdapter"),
        components={
            "artifact-resolver":
            ("voicehub.models.vibevoice.native.artifacts:"
             "resolve_vibevoice_artifacts"),
            "runtime": ("voicehub.models.vibevoice.native.runtime:"
                        "VibeVoiceRuntime"),
            "tokenizer": ("voicehub.models.vibevoice.native.tokenization:"
                          "VibeVoiceTokenizer"),
            "diffusion-head": ("voicehub.models.vibevoice.native.diffusion:"
                               "VibeVoiceDiffusionHead"),
            "diffusion-solver": ("voicehub.models.vibevoice.native.diffusion:"
                                 "VibeVoiceDPMSolver"),
            "trainer-adapter": ("voicehub.models.vibevoice.training:"
                                "VibeVoiceTrainingAdapter"),
            "checkpoint-exporter":
            ("voicehub.models.vibevoice.native.checkpoint:"
             "export_vibevoice_checkpoint"),
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
                "custom-kernels",
                "diffusion-cache",
                "diffusion-sampling",
            ),
            features=(
                "llm-tts-codec",
                "diffusion-family",
                "diffusion-kind-denoising-diffusion",
                "diffusion-operation-denoiser",
                "diffusion-operation-classifier-free-guidance",
                "diffusion-operation-dpm-solver-plus-plus",
                "diffusion-sampling-schedule",
                "diffusion-sampling-guidance",
                "diffusion-sampling-prediction-cache",
                "causal-language-model",
                "continuous-speech-codecs",
                "diffusion-acoustic-head",
                "dpm-solver",
                "fused-diffusion-modulation-kernels",
                "preprocessed-latent-finetuning",
                "frozen-codecs",
                "strict-sharded-safetensors",
                "portable-native-export",
                "verified-low-level-realtime-stages",
                "high-level-realtime-generation-fails-closed",
                "realtime-finetuning-fails-closed",
                "no-external-model-runtime",
            ),
        ),
        upstream_revision=MICROSOFT_VIBEVOICE_SOURCE_REVISION,
        license_id=MICROSOFT_VIBEVOICE_LICENSE,
        metadata={
            "external_llm_backend_blocker": ("VibeVoice couples language-model hidden states to diffusion."),
            "diffusion_architecture_kind": "denoising-diffusion",
            "diffusion_operations": (
                "denoiser",
                "classifier-free-guidance",
                "dpm-solver-plus-plus",
            ),
            "diffusion_sampling_capabilities": (
                "schedule",
                "guidance",
                "prediction-cache",
            ),
            "implementation": "voicehub-native",
            "tensor_backend": "pytorch",
            "source": MICROSOFT_VIBEVOICE_SOURCE,
            "source_revision": MICROSOFT_VIBEVOICE_SOURCE_REVISION,
            "training_checkpoint": VIBEVOICE_TTS_REPOSITORY,
            "training_checkpoint_revision": training_checkpoint["revision"],
            "training_checkpoint_tensor_count": training_checkpoint["tensors"],
            "training_checkpoint_parameter_count": (training_checkpoint["parameters"]),
            "training_checkpoint_header_fingerprint": (training_checkpoint["header_fingerprint"]),
            "realtime_checkpoint": VIBEVOICE_REALTIME_REPOSITORY,
            "realtime_checkpoint_revision": realtime_checkpoint["revision"],
            "realtime_checkpoint_tensor_count": realtime_checkpoint["tensors"],
            "realtime_checkpoint_parameter_count": (realtime_checkpoint["parameters"]),
            "realtime_checkpoint_header_fingerprint": (realtime_checkpoint["header_fingerprint"]),
            "reference_checkpoint_license": MICROSOFT_VIBEVOICE_LICENSE,
            "training_scope": "non-streaming-1.5b-preprocessed-latents",
            "high_level_inference_scope": "fails-closed-pending-parity",
        },
    )


def _register(
    spec: ArchitectureSpec,
    *,
    registry: ArchitectureRegistry | None,
    aliases: Iterable[str],
    exist_ok: bool,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


def register_vibevoice_asr_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_VIBEVOICE_ASR_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    return _register(
        create_vibevoice_asr_architecture_spec(),
        registry=registry,
        aliases=aliases,
        exist_ok=exist_ok,
    )


def register_vibevoice_tts_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_VIBEVOICE_TTS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    return _register(
        create_vibevoice_tts_architecture_spec(),
        registry=registry,
        aliases=aliases,
        exist_ok=exist_ok,
    )


__all__ = [
    "DEFAULT_VIBEVOICE_ASR_ALIASES",
    "DEFAULT_VIBEVOICE_TTS_ALIASES",
    "create_vibevoice_asr_architecture_spec",
    "create_vibevoice_tts_architecture_spec",
    "register_vibevoice_asr_architecture",
    "register_vibevoice_tts_architecture",
]
