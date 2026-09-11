"""Declarative architecture bundle for native generic ASR dispatch.

The dispatcher is deliberately a closed bundle of independently
registered VoiceHub architectures. Adding a future family requires an
explicit component reference and compatibility tests; checkpoint
metadata can never cause arbitrary repository code to execute.
"""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_ASR_DISPATCH_ALIASES = (
    "native-asr",
    "generic-native-asr",
)


def create_asr_dispatch_architecture_spec() -> ArchitectureSpec:
    """Create the lazy declaration for the verified native ASR bundle."""
    return ArchitectureSpec(
        architecture_id="native-asr-dispatch",
        version="1",
        model_builder=("voicehub.models.asr_transformers.modeling:"
                       "TransformersASRForSpeechRecognition"),
        config=("voicehub.models.asr_transformers."
                "configuration:TransformersASRConfig"),
        components={
            "whisper": "voicehub.models.asr_whisper_native.native.modeling:WhisperModel",
            "wav2vec2": "voicehub.models.asr_wav2vec2.native.modeling:Wav2Vec2ForCTC",
            "hubert": "voicehub.models.asr_hubert.native.modeling:HubertForCTC",
            "wavlm": "voicehub.models.asr_wavlm.native.modeling:WavLMForCTC",
            "moonshine":
            ("voicehub.models.asr_moonshine.native.modeling:"
             "MoonshineForConditionalGeneration"),
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
                "closed-dispatch",
                "ctc",
                "speech-seq2seq",
                "strict-config-resolution",
                "portable-export",
            ),
        ),
        license_id="Apache-2.0",
        metadata={
            "implementation": "voicehub-native",
            "tensor_backend": "pytorch",
            "dispatch_policy": "closed-verified-families",
            "families": (
                "whisper",
                "wav2vec2",
                "hubert",
                "wavlm",
                "moonshine",
            ),
        },
    )


def register_asr_dispatch_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_ASR_DISPATCH_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register the native ASR dispatcher without resolving its graphs."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_asr_dispatch_architecture_spec()
    target.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )
    return spec


__all__ = [
    "DEFAULT_ASR_DISPATCH_ALIASES",
    "create_asr_dispatch_architecture_spec",
    "register_asr_dispatch_architecture",
]
