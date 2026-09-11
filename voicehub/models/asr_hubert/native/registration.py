"""Lazy architecture declaration for VoiceHub-native HuBERT CTC."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_hubert.native.checkpoint import (
    FACEBOOK_HUBERT_LARGE_LS960_FT_REVISION,
    FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_REVISION,
    TRANSFORMERS_HUBERT_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_HUBERT_ALIASES = ("native-hubert", "hubert-ctc")


def create_hubert_architecture_spec() -> ArchitectureSpec:
    """Create the immutable HuBERT CTC declaration."""
    return ArchitectureSpec(
        architecture_id="hubert",
        version="1",
        model_builder="voicehub.models.asr_hubert.native.modeling:HubertForCTC",
        config="voicehub.models.asr_hubert.native.configuration:HubertConfig",
        processor=("voicehub.models.asr_wav2vec2.processing:"
                   "Wav2Vec2Processor"),
        objective="voicehub.objectives.ctc:ctc_loss",
        checkpoint_adapter=(
            "voicehub.models.asr_hubert.native.checkpoint:"
            "HuggingFaceHubertCheckpointAdapter"),
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
                "ctc",
                "stable-layer-norm",
                "spec-augment",
                "word-timestamps",
                "checkpoint-conversion",
            ),
        ),
        upstream_revision=TRANSFORMERS_HUBERT_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family":
            "hubert",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "reference_checkpoint":
            "facebook/hubert-large-ls960-ft",
            "reference_checkpoint_revision":
            FACEBOOK_HUBERT_LARGE_LS960_FT_REVISION,
            "reference_safetensors_revision":
            FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_REVISION,
            "reference_source": ("https://huggingface.co/facebook/hubert-large-ls960-ft"),
            "transformers_source": (
                "https://github.com/huggingface/transformers/tree/"
                f"{TRANSFORMERS_HUBERT_REVISION}/src/transformers/models/"
                "hubert"),
        },
    )


def register_hubert_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_HUBERT_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register the lazy HuBERT declaration and return it."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_hubert_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_HUBERT_ALIASES",
    "create_hubert_architecture_spec",
    "register_hubert_architecture",
]
