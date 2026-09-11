"""Lazy architecture declaration for VoiceHub-native VoxCPM2."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.voxcpm.native.metadata import (
    VOXCPM2_CHECKPOINT_HEADER_FINGERPRINT,
    VOXCPM2_CHECKPOINT_PARAMETER_COUNT,
    VOXCPM2_CHECKPOINT_REPOSITORY,
    VOXCPM2_CHECKPOINT_REVISION,
    VOXCPM2_CHECKPOINT_SHA256,
    VOXCPM2_CHECKPOINT_TENSOR_COUNT,
    VOXCPM2_CODEC_HEADER_FINGERPRINT,
    VOXCPM2_CODEC_LEGACY_SHA256,
    VOXCPM2_CODEC_PARAMETER_COUNT,
    VOXCPM2_CODEC_TENSOR_COUNT,
    VOXCPM2_LICENSE,
    VOXCPM2_SOURCE_REPOSITORY,
    VOXCPM2_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_VOXCPM2_ALIASES = (
    "native-voxcpm",
    "native-voxcpm2",
    "vox-cpm-2",
)


def create_voxcpm2_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="voxcpm2",
        version="2",
        model_builder=("voicehub.models.voxcpm.native.modeling:VoxCPM2Model"),
        config=("voicehub.models.voxcpm.native.configuration:"
                "VoxCPM2ArchitectureConfig"),
        processor=("voicehub.models.voxcpm.native.processing:VoxCPM2Processor"),
        decoder=("voicehub.models.voxcpm.native.codec:VoxCPMAudioVAE"),
        objective=("voicehub.models.voxcpm.native.modeling:VoxCPM2Model.forward"),
        checkpoint_adapter=("voicehub.models.voxcpm.native.checkpoint:"
                            "load_voxcpm_checkpoint"),
        components={
            "artifact-resolver": ("voicehub.models.voxcpm.native.artifacts:"
                                  "resolve_voxcpm2_artifacts"),
            "checkpoint-exporter": ("voicehub.models.voxcpm.native.checkpoint:"
                                    "export_voxcpm_checkpoint"),
            "legacy-codec-converter":
            ("voicehub.models.voxcpm.native.checkpoint:"
             "convert_legacy_voxcpm_codec"),
            "lora": ("voicehub.models.voxcpm.native.lora:"
                     "inject_voxcpm_lora"),
            "runtime": ("voicehub.models.voxcpm.native.runtime:"
                        "load_voxcpm2_runtime"),
            "text-tokenizer": ("voicehub.models.voxcpm.native.processing:"
                               "VoxCPM2Tokenizer"),
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
                "diffusion-serving-vllm-omni",
                "diffusion-kind-conditional-flow-matching",
                "diffusion-operation-denoiser",
                "diffusion-operation-classifier-free-guidance",
                "diffusion-operation-euler-solver",
                "diffusion-sampling-schedule",
                "diffusion-sampling-guidance",
                "diffusion-sampling-prediction-cache",
                "30-language-multilingual",
                "audio-continuation",
                "cjk-multichar-token-split",
                "conditional-flow-matching",
                "controllable-voice-cloning",
                "full-sft",
                "grouped-query-attention",
                "native-audiovae-v2",
                "native-sentencepiece-bpe-tokenizer",
                "published-lora-topology",
                "reference-audio-isolation",
                "strict-safetensors-reload",
                "tokenizer-free-audio",
                "voice-design",
            ),
        ),
        upstream_revision=VOXCPM2_SOURCE_REVISION,
        license_id=VOXCPM2_LICENSE,
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
            ),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            VOXCPM2_SOURCE_REPOSITORY,
            "source_revision":
            VOXCPM2_SOURCE_REVISION,
            "reference_checkpoint":
            VOXCPM2_CHECKPOINT_REPOSITORY,
            "reference_checkpoint_revision":
            VOXCPM2_CHECKPOINT_REVISION,
            "reference_checkpoint_sha256":
            VOXCPM2_CHECKPOINT_SHA256,
            "reference_tensor_count":
            VOXCPM2_CHECKPOINT_TENSOR_COUNT,
            "reference_parameter_count":
            VOXCPM2_CHECKPOINT_PARAMETER_COUNT,
            "reference_safetensors_header_fingerprint": (VOXCPM2_CHECKPOINT_HEADER_FINGERPRINT),
            "codec_source_format":
            "pytorch-pickle",
            "codec_legacy_sha256":
            VOXCPM2_CODEC_LEGACY_SHA256,
            "codec_tensor_count":
            VOXCPM2_CODEC_TENSOR_COUNT,
            "codec_parameter_count":
            VOXCPM2_CODEC_PARAMETER_COUNT,
            "codec_header_fingerprint":
            VOXCPM2_CODEC_HEADER_FINGERPRINT,
            "checkpoint_license":
            VOXCPM2_LICENSE,
            "full_finetuning_ready":
            True,
            "lora_finetuning_ready":
            True,
            "training_boundary": (
                "The trainable 577-tensor graph uses the published flow-"
                "matching and stop-token cross-entropy objectives with equal "
                "default weights. AudioVAE V2 is frozen, exactly as in the "
                "official fine-tuner. Raw audio is encoded at 16 kHz; "
                "pre-encoded latent patches avoid loading the codec."),
            "codec_trust_boundary": (
                "Upstream publishes AudioVAE V2 only as a digest-pinned "
                "legacy pickle. VoiceHub requires explicit one-time trust, "
                "uses torch.load(weights_only=True), validates all 312 "
                "tensors, and exports Safetensors for steady-state use."),
        },
    )


def register_voxcpm2_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_VOXCPM2_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_voxcpm2_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_VOXCPM2_ALIASES",
    "create_voxcpm2_architecture_spec",
    "register_voxcpm2_architecture",
]
