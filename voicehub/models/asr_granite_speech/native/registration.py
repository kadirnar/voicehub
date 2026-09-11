"""Lazy native-architecture declaration for Granite Speech."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_granite_speech.native.metadata import (
    GRANITE_SPEECH_CHECKPOINTS,
    GRANITE_SPEECH_RELEASE_SOURCE_REVISION,
    GRANITE_SPEECH_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_GRANITE_SPEECH_ALIASES = (
    "native-granite-speech",
    "granite-asr",
)


def create_granite_speech_architecture_spec() -> ArchitectureSpec:
    """Create the immutable native Granite Speech declaration."""
    return ArchitectureSpec(
        architecture_id="granite-speech",
        version="1",
        model_builder=(
            "voicehub.models.asr_granite_speech.native.modeling:"
            "GraniteSpeechForConditionalGeneration"),
        config=("voicehub.models.asr_granite_speech.native.configuration:"
                "GraniteSpeechArchitectureConfig"),
        processor=("voicehub.models.asr_granite_speech.native.processing:"
                   "GraniteSpeechProcessor"),
        decoder="voicehub.generation.engine:AutoregressiveGenerator",
        objective=("voicehub.objectives.sequence:sequence_cross_entropy"),
        checkpoint_adapter=(
            "voicehub.models.asr_granite_speech.native.checkpoint:"
            "GraniteSpeechCheckpointAdapter"),
        components={
            "audio-encoder":
            ("voicehub.models.asr_granite_speech.native.modeling:"
             "GraniteSpeechCTCEncoder"),
            "audio-projector":
            ("voicehub.models.asr_granite_speech.native.modeling:"
             "GraniteSpeechEncoderProjector"),
            "language-model": ("voicehub.models.causal_lm.native.modeling:"
                               "GraniteForCausalLM"),
            "runtime": ("voicehub.models.asr_granite_speech.native.runtime:"
                        "GraniteSpeechRuntime"),
            "tokenizer": ("voicehub.models.asr_granite_speech.native.tokenization:"
                          "GraniteSpeechTokenizer"),
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
                "conformer",
                "hotwords",
                "kv-cache",
                "multilingual",
                "qformer",
                "safe-export",
            ),
        ),
        upstream_revision=GRANITE_SPEECH_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family":
            "granite-speech",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source": (
                "https://github.com/huggingface/transformers/tree/"
                f"{GRANITE_SPEECH_SOURCE_REVISION}/src/transformers/"
                "models/granite_speech"),
            "reference_checkpoints":
            GRANITE_SPEECH_CHECKPOINTS,
            "checkpoint_release_source_revision": (GRANITE_SPEECH_RELEASE_SOURCE_REVISION),
            "training_scope": (
                "End-to-end completion-only causal fine-tuning and "
                "VoiceHub-native LoRA are supported. No accuracy change over "
                "the pinned upstream checkpoint is claimed."),
        },
    )


def register_granite_speech_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_GRANITE_SPEECH_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = (ARCHITECTURE_REGISTRY if registry is None else registry)
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_granite_speech_architecture_spec()
    target.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )
    return spec


__all__ = [
    "DEFAULT_GRANITE_SPEECH_ALIASES",
    "create_granite_speech_architecture_spec",
    "register_granite_speech_architecture",
]
