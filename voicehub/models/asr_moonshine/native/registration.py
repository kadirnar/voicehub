"""Lazy architecture declaration for VoiceHub's native Moonshine ASR."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_moonshine.native.checkpoint import USEFULSENSORS_MOONSHINE_TINY_REVISION
from voicehub.models.asr_moonshine.native.configuration import (
    MOONSHINE_MAIN_LIBRARY_REVISION,
    TRANSFORMERS_MOONSHINE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_MOONSHINE_ALIASES = (
    "native-moonshine",
    "moonshine-asr",
    "hf-moonshine",
)


def create_moonshine_architecture_spec() -> ArchitectureSpec:
    """Create the immutable and entirely lazy Moonshine declaration."""
    return ArchitectureSpec(
        architecture_id="moonshine",
        version="1",
        model_builder=("voicehub.models.asr_moonshine.native.modeling:"
                       "MoonshineForConditionalGeneration"),
        config=("voicehub.models.asr_moonshine.native.configuration:MoonshineConfig"),
        processor=("voicehub.models.asr_moonshine.native.processing:MoonshineProcessor"),
        objective="voicehub.objectives.sequence:Seq2SeqCrossEntropyLoss",
        checkpoint_adapter=(
            "voicehub.models.asr_moonshine.native.checkpoint:"
            "HuggingFaceMoonshineCheckpointAdapter"),
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            optimization_passes=("compile", ),
            features=(
                "raw-waveform-convolutional-frontend",
                "rotary-encoder-decoder",
                "teacher-forced-cross-entropy",
                "sentencepiece-bpe",
                "byte-fallback",
                "greedy-decoding",
                "checkpoint-conversion",
                "no-remote-code",
            ),
        ),
        upstream_revision=MOONSHINE_MAIN_LIBRARY_REVISION,
        license_id="MIT",
        metadata={
            "family":
            "moonshine",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "main_library_source":
            ("https://github.com/moonshine-ai/moonshine/tree/"
             f"{MOONSHINE_MAIN_LIBRARY_REVISION}"),
            "transformers_reference_revision":
            TRANSFORMERS_MOONSHINE_REVISION,
            "transformers_source": (
                "https://github.com/huggingface/transformers/tree/"
                f"{TRANSFORMERS_MOONSHINE_REVISION}/src/transformers/models/"
                "moonshine"),
            "reference_checkpoint":
            "UsefulSensors/moonshine-tiny",
            "reference_checkpoint_revision": (USEFULSENSORS_MOONSHINE_TINY_REVISION),
        },
    )


def register_moonshine_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_MOONSHINE_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register the lazy Moonshine declaration and return it."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_moonshine_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_MOONSHINE_ALIASES",
    "create_moonshine_architecture_spec",
    "register_moonshine_architecture",
]
