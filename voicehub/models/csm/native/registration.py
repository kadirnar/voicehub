"""Lazy architecture declaration for VoiceHub-native Sesame CSM."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.csm.native.metadata import (
    CSM_CHECKPOINT_HEADER_FINGERPRINT,
    CSM_CHECKPOINT_PARAMETER_COUNT,
    CSM_CHECKPOINT_REPOSITORY,
    CSM_CHECKPOINT_REVISION,
    CSM_CHECKPOINT_SHA256,
    CSM_CHECKPOINT_TENSOR_COUNT,
    CSM_SOURCE_REPOSITORY,
    CSM_SOURCE_REVISION,
    CSM_TOKENIZER_SHA256,
    CSM_TORCHTUNE_REVISION,
    MIMI_CHECKPOINT_HEADER_FINGERPRINT,
    MIMI_CHECKPOINT_PARAMETER_COUNT,
    MIMI_CHECKPOINT_REPOSITORY,
    MIMI_CHECKPOINT_REVISION,
    MIMI_CHECKPOINT_SHA256,
    MIMI_CHECKPOINT_TENSOR_COUNT,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_CSM_ALIASES = (
    "native-csm",
    "sesame-csm",
    "csm-1b",
)


def create_csm_architecture_spec() -> ArchitectureSpec:
    """Describe CSM without importing PyTorch or constructing its graph."""
    return ArchitectureSpec(
        architecture_id="csm",
        version="1",
        model_builder="voicehub.models.csm.native.modeling:CSMModel",
        config=("voicehub.models.csm.native.configuration:"
                "CSMArchitectureConfig"),
        processor="voicehub.models.csm.native.processing:CSMProcessor",
        decoder="voicehub.models.csm.native.mimi:load_mimi",
        objective=("voicehub.models.csm.training:"
                   "CSMTrainingBackend.forward_loss"),
        checkpoint_adapter=("voicehub.models.csm.native.checkpoint:"
                            "load_csm_checkpoint"),
        components={
            "artifact-resolver": ("voicehub.models.csm.native.artifacts:"
                                  "resolve_csm_artifacts"),
            "checkpoint-exporter": ("voicehub.models.csm.native.checkpoint:"
                                    "export_csm_checkpoint"),
            "mimi-codec": "voicehub.models.csm.native.mimi:load_mimi",
            "runtime": ("voicehub.models.csm.native.runtime:"
                        "load_csm_runtime"),
            "text-tokenizer": ("voicehub.models.csm.native.processing:"
                               "CSMTextTokenizer"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda"),
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
                "conversational-multi-speaker",
                "grouped-query-attention",
                "hierarchical-32-codebook",
                "kv-cache",
                "llama3-scaled-rope",
                "native-byte-bpe-tokenizer",
                "native-mimi-codec",
                "preencoded-code-fine-tuning",
                "raw-audio-fine-tuning",
                "strict-safetensors-reload",
                "two-level-codebook-cross-entropy",
                "watermark-postprocessor-boundary",
            ),
        ),
        upstream_revision=CSM_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "external_llm_backend_blocker": ("CSM requires a hidden-state-conditioned depth decoder."),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            CSM_SOURCE_REPOSITORY,
            "source_revision":
            CSM_SOURCE_REVISION,
            "torchtune_math_revision":
            CSM_TORCHTUNE_REVISION,
            "reference_checkpoint":
            CSM_CHECKPOINT_REPOSITORY,
            "reference_checkpoint_revision":
            CSM_CHECKPOINT_REVISION,
            "reference_checkpoint_sha256":
            CSM_CHECKPOINT_SHA256,
            "reference_tensor_count":
            CSM_CHECKPOINT_TENSOR_COUNT,
            "reference_parameter_count":
            CSM_CHECKPOINT_PARAMETER_COUNT,
            "reference_safetensors_header_fingerprint": (CSM_CHECKPOINT_HEADER_FINGERPRINT),
            "tokenizer_sha256":
            CSM_TOKENIZER_SHA256,
            "codec_checkpoint":
            MIMI_CHECKPOINT_REPOSITORY,
            "codec_checkpoint_revision":
            MIMI_CHECKPOINT_REVISION,
            "codec_checkpoint_sha256":
            MIMI_CHECKPOINT_SHA256,
            "codec_tensor_count":
            MIMI_CHECKPOINT_TENSOR_COUNT,
            "codec_parameter_count":
            MIMI_CHECKPOINT_PARAMETER_COUNT,
            "codec_safetensors_header_fingerprint": (MIMI_CHECKPOINT_HEADER_FINGERPRINT),
            "checkpoint_license":
            "Apache-2.0",
            "codec_checkpoint_license":
            "CC-BY-4.0",
            "checkpoint_access":
            "gated",
            "full_finetuning_ready":
            True,
            "training_boundary": (
                "The trainable CSM graph optimizes the published "
                "next-frame codebook-zero and within-frame depth-decoder "
                "cross-entropies. Raw PCM is encoded by a separately pinned, "
                "frozen native Mimi graph; pre-encoded 32-codebook targets "
                "do not require loading Mimi."),
            "watermark_boundary": (
                "SilentCipher is a separately trained postprocessor and is "
                "never claimed unless an injected postprocessor explicitly "
                "declares watermarks_audio=true."),
        },
    )


def register_csm_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_CSM_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register native CSM in a target architecture registry."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_csm_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_CSM_ALIASES",
    "create_csm_architecture_spec",
    "register_csm_architecture",
]
