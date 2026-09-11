"""Lazy architecture declaration for VoiceHub-native Cohere Transcribe."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_cohere.native.metadata import COHERE_ASR_CHECKPOINTS, COHERE_TRANSFORMERS_REVISION
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_COHERE_ASR_ALIASES = (
    "cohere-transcribe",
    "native-cohere-asr",
)


def create_cohere_asr_architecture_spec() -> ArchitectureSpec:
    """Describe the audited Cohere graph without importing PyTorch."""
    checkpoint = COHERE_ASR_CHECKPOINTS["CohereLabs/cohere-transcribe-03-2026"]
    return ArchitectureSpec(
        architecture_id="cohere-asr",
        version="1",
        model_builder=("voicehub.models.asr_cohere.native.modeling:"
                       "CohereAsrForConditionalGeneration"),
        config="voicehub.models.asr_cohere.native.configuration:CohereAsrConfig",
        processor="voicehub.models.asr_cohere.native.processing:CohereAsrProcessor",
        decoder=("voicehub.models.asr_cohere.native.modeling:CohereGenerateOutput"),
        objective="voicehub.objectives.sequence:sequence_cross_entropy",
        checkpoint_adapter=("voicehub.models.asr_cohere.native.checkpoint:"
                            "CohereAsrCheckpointAdapter"),
        components={
            "encoder": ("voicehub.models.asr_cohere.native.modeling:ConformerEncoder"),
            "frontend": ("voicehub.models.asr_cohere.native.processing:"
                         "CohereAsrFeatureExtractor"),
            "runtime": ("voicehub.models.asr_cohere.native.runtime:CohereAsrRuntime"),
            "tokenizer": ("voicehub.models.asr_cohere.native.tokenization:"
                          "CohereAsrTokenizer"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            features=(
                "fastconformer",
                "gradient-checkpointing",
                "language-conditioned",
                "long-form-quiet-boundary-segmentation",
                "portable-export",
                "prompt-conditioned-decoder",
            ),
        ),
        upstream_revision=COHERE_TRANSFORMERS_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family": "cohere-asr",
            "implementation": "voicehub-native",
            "tensor_backend": "pytorch",
            "reference_checkpoint": "CohereLabs/cohere-transcribe-03-2026",
            "reference_checkpoint_revision": checkpoint["revision"],
            "reference_checkpoint_header_fingerprint": (checkpoint["header_fingerprint"]),
            "access": "gated",
            "decoding": "greedy-only",
            "languages": 14,
        },
    )


def register_cohere_asr_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_COHERE_ASR_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register Cohere Transcribe in an architecture registry."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be ArchitectureRegistry or None.")
    spec = create_cohere_asr_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_COHERE_ASR_ALIASES",
    "create_cohere_asr_architecture_spec",
    "register_cohere_asr_architecture",
]
