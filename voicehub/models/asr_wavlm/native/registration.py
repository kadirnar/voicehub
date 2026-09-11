"""Lazy architecture declaration for VoiceHub-native WavLM CTC."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_wavlm.native.checkpoint import (
    MICROSOFT_WAVLM_SOURCE_REVISION,
    TRANSFORMERS_WAVLM_REVISION,
    WAVLM_BASE_PLUS_CTC_REVISION,
    WAVLM_BASE_PLUS_CTC_SAFETENSORS_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_WAVLM_ALIASES = (
    "native-wavlm",
    "wavlm-ctc",
    "hf-wavlm",
)


def create_wavlm_architecture_spec() -> ArchitectureSpec:
    """Create the immutable WavLM CTC declaration."""
    return ArchitectureSpec(
        architecture_id="wavlm",
        version="1",
        model_builder="voicehub.models.asr_wavlm.native.modeling:WavLMForCTC",
        config="voicehub.models.asr_wavlm.native.configuration:WavLMConfig",
        processor=("voicehub.models.asr_wavlm.processing:WavLMProcessor"),
        objective="voicehub.objectives.ctc:ctc_loss",
        checkpoint_adapter=(
            "voicehub.models.asr_wavlm.native.checkpoint:"
            "HuggingFaceWavLMCheckpointAdapter"),
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
                "gated-relative-position-bias",
                "stable-layer-norm",
                "spec-augment",
                "word-timestamps",
                "checkpoint-conversion",
                "no-kv-cache",
            ),
        ),
        upstream_revision=TRANSFORMERS_WAVLM_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family":
            "wavlm",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "reference_checkpoint":
            "patrickvonplaten/wavlm-libri-clean-100h-base-plus",
            "reference_checkpoint_revision":
            WAVLM_BASE_PLUS_CTC_REVISION,
            "reference_safetensors_revision":
            WAVLM_BASE_PLUS_CTC_SAFETENSORS_REVISION,
            "reference_source":
            ("https://huggingface.co/patrickvonplaten/"
             "wavlm-libri-clean-100h-base-plus"),
            "microsoft_source":
            ("https://github.com/microsoft/unilm/tree/"
             f"{MICROSOFT_WAVLM_SOURCE_REVISION}/wavlm"),
            "transformers_source": (
                "https://github.com/huggingface/transformers/tree/"
                f"{TRANSFORMERS_WAVLM_REVISION}/src/transformers/models/wavlm"),
        },
    )


def register_wavlm_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_WAVLM_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register the lazy WavLM declaration and return it."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_wavlm_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_WAVLM_ALIASES",
    "create_wavlm_architecture_spec",
    "register_wavlm_architecture",
]
