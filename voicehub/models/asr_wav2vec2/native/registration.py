"""Lazy architecture declaration for VoiceHub's native Wav2Vec2 CTC."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_wav2vec2.native.checkpoint import FACEBOOK_WAV2VEC2_BASE_960H_REVISION
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

TRANSFORMERS_WAV2VEC2_REVISION = ("ebea912f0bb6f9e28ad2df04acd9b4df035933a9")
DEFAULT_WAV2VEC2_ALIASES = (
    "native-wav2vec2",
    "wav2vec2-ctc",
    "hf-wav2vec2",
)


def create_wav2vec2_architecture_spec() -> ArchitectureSpec:
    """Create the immutable native Wav2Vec2 CTC declaration."""
    return ArchitectureSpec(
        architecture_id="wav2vec2",
        version="1",
        model_builder=("voicehub.models.asr_wav2vec2.native.modeling:Wav2Vec2ForCTC"),
        config=("voicehub.models.asr_wav2vec2.native.configuration:Wav2Vec2Config"),
        objective="voicehub.objectives.ctc:CTCLoss",
        checkpoint_adapter=(
            "voicehub.models.asr_wav2vec2.native.checkpoint:"
            "HuggingFaceWav2Vec2CheckpointAdapter"),
        capabilities=ArchitectureCapabilities(
            tasks=(
                SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
                SpeechTask.VOICE_ACTIVITY_DETECTION,
            ),
            devices=("cpu", "cuda"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            features=(
                "bidirectional-encoder",
                "convolutional-frontend",
                "ctc",
                "audio-classification",
                "frame-classification",
                "layerdrop",
                "spec-augment",
                "checkpoint-conversion",
                "no-kv-cache",
            ),
        ),
        upstream_revision=TRANSFORMERS_WAV2VEC2_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family":
            "wav2vec2",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "transformers_source": (
                "https://github.com/huggingface/transformers/tree/"
                f"{TRANSFORMERS_WAV2VEC2_REVISION}/src/transformers/models/"
                "wav2vec2"),
            "reference_checkpoint":
            "facebook/wav2vec2-base-960h",
            "reference_checkpoint_revision": (FACEBOOK_WAV2VEC2_BASE_960H_REVISION),
            "reference_checkpoint_source": (
                "https://huggingface.co/facebook/wav2vec2-base-960h/tree/"
                f"{FACEBOOK_WAV2VEC2_BASE_960H_REVISION}"),
        },
    )


def register_wav2vec2_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_WAV2VEC2_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register the lazy Wav2Vec2 declaration and return it."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_wav2vec2_architecture_spec()
    target.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )
    return spec


__all__ = [
    "DEFAULT_WAV2VEC2_ALIASES",
    "TRANSFORMERS_WAV2VEC2_REVISION",
    "create_wav2vec2_architecture_spec",
    "register_wav2vec2_architecture",
]
