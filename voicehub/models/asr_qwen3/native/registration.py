"""Lazy native-architecture declaration for Qwen3-ASR."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_qwen3.native.metadata import QWEN3_ASR_CHECKPOINTS, QWEN3_ASR_SOURCE_REVISION
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_QWEN3_ASR_ALIASES = ("native-qwen3-asr", )


def create_qwen3_asr_architecture_spec() -> ArchitectureSpec:
    """Create the immutable, lazily resolved Qwen3-ASR declaration."""
    return ArchitectureSpec(
        architecture_id="qwen3-asr",
        version="1",
        model_builder=("voicehub.models.asr_qwen3.native.modeling:"
                       "Qwen3ASRForConditionalGeneration"),
        config=("voicehub.models.asr_qwen3.native.configuration:"
                "Qwen3ASRArchitectureConfig"),
        processor=("voicehub.models.asr_qwen3.native.processing:"
                   "Qwen3ASRProcessor"),
        decoder="voicehub.generation.engine:AutoregressiveGenerator",
        objective="voicehub.objectives.sequence:sequence_cross_entropy",
        checkpoint_adapter=("voicehub.models.asr_qwen3.native.checkpoint:"
                            "Qwen3ASRCheckpointAdapter"),
        components={
            "audio-encoder": ("voicehub.models.asr_qwen3.native.modeling:"
                              "Qwen3ASRAudioEncoder"),
            "tokenizer": ("voicehub.models.asr_qwen3.native.tokenization:"
                          "Qwen3ASRTokenizer"),
            "runtime": ("voicehub.models.asr_qwen3.native.runtime:"
                        "Qwen3ASRRuntime"),
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
            optimization_passes=("compile", "lora"),
            features=(
                "audio-language-model",
                "completion-only-labels",
                "hotwords",
                "language-identification",
                "long-form",
                "multilingual",
                "qwen3",
                "kv-cache",
                "checkpoint-conversion",
            ),
        ),
        upstream_revision=QWEN3_ASR_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family":
            "qwen3-asr",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source": ("https://github.com/QwenLM/Qwen3-ASR/tree/"
                       f"{QWEN3_ASR_SOURCE_REVISION}"),
            "reference_checkpoints":
            QWEN3_ASR_CHECKPOINTS,
            "streaming_scope": (
                "The official convenience API recomputes the full accumulated "
                "audio prefix; the native graph therefore advertises only "
                "buffered offline sessions."),
            "timestamps":
            ("Qwen3-ASR does not emit timestamps. Forced alignment is a "
             "separate architecture."),
        },
    )


def register_qwen3_asr_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_QWEN3_ASR_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_qwen3_asr_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_QWEN3_ASR_ALIASES",
    "create_qwen3_asr_architecture_spec",
    "register_qwen3_asr_architecture",
]
