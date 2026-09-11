"""Lazy architecture discovery for VoiceHub's native Whisper family.

Creating or registering this specification imports no model, tokenizer,
checkpoint, or generation implementation.  Component modules resolve
only when a runtime explicitly requests them.
"""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

OPENAI_WHISPER_REVISION = "04f449b8a437f1bbd3dba5c9f826aca972e7709a"
TRANSFORMERS_WHISPER_REVISION = "ebea912f0bb6f9e28ad2df04acd9b4df035933a9"
DEFAULT_WHISPER_ALIASES = (
    "native-whisper",
    "openai-whisper",
    "hf-whisper",
)


def create_whisper_architecture_spec() -> ArchitectureSpec:
    """Create the immutable, entirely lazy native Whisper declaration."""
    return ArchitectureSpec(
        architecture_id="whisper",
        version="1",
        model_builder=("voicehub.models.asr_whisper_native.native.modeling:WhisperModel"),
        config=("voicehub.models.asr_whisper_native.native.configuration:WhisperConfig"),
        processor=("voicehub.models.asr_whisper_native.native.tokenization:WhisperTokenizer"),
        decoder=("voicehub.models.asr_whisper_native.native.decoding:"
                 "WhisperGenerationAdapter"),
        objective=("voicehub.objectives.sequence:Seq2SeqCrossEntropyLoss"),
        checkpoint_adapter=(
            "voicehub.models.asr_whisper_native.native.checkpoint:"
            "HuggingFaceWhisperCheckpointAdapter"),
        components={
            "openai-checkpoint-adapter":
            ("voicehub.models.asr_whisper_native.native.checkpoint:"
             "OpenAIWhisperCheckpointAdapter"),
            "native-checkpoint-adapter":
            ("voicehub.models.asr_whisper_native.native.checkpoint:"
             "NativeWhisperCheckpointAdapter"),
            "token-metadata": ("voicehub.models.asr_whisper_native.native.decoding:WhisperTokenSet"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", "pt", "pytorch"),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            features=(
                "encoder-decoder",
                "kv-cache",
                "language-detection",
                "timestamps",
                "checkpoint-conversion",
            ),
        ),
        upstream_revision=OPENAI_WHISPER_REVISION,
        license_id="MIT",
        metadata={
            "family":
            "whisper",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "openai_source": ("https://github.com/openai/whisper/tree/"
                              f"{OPENAI_WHISPER_REVISION}"),
            "transformers_reference_revision":
            TRANSFORMERS_WHISPER_REVISION,
            "transformers_source": (
                "https://github.com/huggingface/transformers/tree/"
                f"{TRANSFORMERS_WHISPER_REVISION}/src/transformers/models/whisper"),
        },
    )


def register_whisper_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_WHISPER_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register the lazy Whisper declaration and return it."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_whisper_architecture_spec()
    target.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )
    return spec


__all__ = [
    "DEFAULT_WHISPER_ALIASES",
    "OPENAI_WHISPER_REVISION",
    "TRANSFORMERS_WHISPER_REVISION",
    "create_whisper_architecture_spec",
    "register_whisper_architecture",
]
