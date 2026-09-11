"""Lazy architecture declaration for native Nemotron 3.5 ASR."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_nemotron.native.metadata import NEMOTRON_ASR_CHECKPOINTS, TRANSFORMERS_SOURCE_REVISION
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_NEMOTRON_ASR_ALIASES = (
    "native-nemotron-asr",
    "nemotron-3.5-asr",
    "nvidia-nemotron-asr",
)


def create_nemotron_asr_architecture_spec() -> ArchitectureSpec:
    """Create the immutable native Nemotron architecture declaration."""
    return ArchitectureSpec(
        architecture_id="nemotron-3.5-rnnt",
        version="1",
        model_builder=("voicehub.models.asr_nemotron.native.modeling:"
                       "Nemotron3_5ASRForRNNT"),
        config=("voicehub.models.asr_nemotron.native.configuration:"
                "NemotronASRArchitectureConfig"),
        processor=("voicehub.models.asr_nemotron.native.processing:"
                   "NemotronASRProcessor"),
        decoder=("voicehub.models.asr_nemotron.native.modeling:"
                 "NemotronGenerateOutput"),
        objective=("voicehub.models.asr_nemotron.native.loss:rnnt_loss"),
        checkpoint_adapter=("voicehub.models.asr_nemotron.native.checkpoint:"
                            "NemotronASRCheckpointAdapter"),
        components={
            "encoder": ("voicehub.models.asr_nemotron.native.modeling:"
                        "NemotronFastConformerEncoder"),
            "frontend": ("voicehub.models.asr_nemotron.native.frontend:"
                         "NemotronLogMelFrontend"),
            "runtime": ("voicehub.models.asr_nemotron.native.runtime:"
                        "NemotronASRRuntime"),
            "tokenizer": ("voicehub.models.asr_nemotron.native.tokenization:"
                          "NemotronASRTokenizer"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=True,
            batched_inference=True,
            distributed_training=True,
            features=(
                "automatic-language-identification",
                "cache-aware-fastconformer",
                "gradient-checkpointing",
                "greedy-rnnt",
                "language-prompt-conditioning",
                "multilingual",
                "native-rnnt-loss",
                "portable-export",
                "token-timestamps",
            ),
        ),
        upstream_revision=TRANSFORMERS_SOURCE_REVISION,
        license_id="OpenMDW-1.1",
        metadata={
            "family":
            "nemotron-3.5-rnnt",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "reference_checkpoints":
            NEMOTRON_ASR_CHECKPOINTS,
            "decoding":
            "greedy-only",
            "training_scope": (
                "Full-model differentiable RNN-T fine-tuning is supported. "
                "No accuracy change over the pinned checkpoint is claimed."),
        },
    )


def register_nemotron_asr_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_NEMOTRON_ASR_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = (ARCHITECTURE_REGISTRY if registry is None else registry)
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_nemotron_asr_architecture_spec()
    target.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )
    return spec


__all__ = [
    "DEFAULT_NEMOTRON_ASR_ALIASES",
    "create_nemotron_asr_architecture_spec",
    "register_nemotron_asr_architecture",
]
