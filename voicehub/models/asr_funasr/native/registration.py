"""Lazy architecture declaration for native SenseVoiceSmall."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_funasr.native.metadata import (
    FUNASR_SOURCE_REVISION,
    SENSEVOICE_CHECKPOINT_SHA256,
    SENSEVOICE_MODEL_LICENSE,
    SENSEVOICE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_SENSEVOICE_ALIASES = (
    "sensevoice",
    "funasr-sensevoice-small",
    "native-sensevoice-small",
)


def create_sensevoice_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="sensevoice-small",
        version="1",
        model_builder=("voicehub.models.asr_funasr.native.modeling:"
                       "SenseVoiceSmallForCTC"),
        config=("voicehub.models.asr_funasr.native.configuration:"
                "SenseVoiceSmallConfig"),
        processor=("voicehub.models.asr_funasr.native.frontend:"
                   "SenseVoiceFrontend"),
        decoder=("voicehub.models.asr_funasr.native.decoding:"
                 "ctc_greedy_tokens"),
        objective=("voicehub.models.asr_funasr.native.modeling:"
                   "SenseVoiceSmallForCTC"),
        checkpoint_adapter=(
            "voicehub.models.asr_funasr.native.checkpoint:"
            "SenseVoiceSafeTensorsCheckpointAdapter"),
        components={
            "tokenizer": ("voicehub.models.asr_funasr.native.tokenization:"
                          "SenseVoiceTokenizer"),
            "pickle-converter":
            ("voicehub.models.asr_funasr.native.checkpoint:"
             "convert_sensevoice_small_checkpoint"),
            "training-adapter":
            ("voicehub.models.asr_funasr.native.training:"
             "NativeSenseVoiceTrainingAdapter"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=(
                "safetensors",
                "trust-gated-verified-pytorch-conversion",
            ),
            export_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            optimization_passes=("compile", "sdpa"),
            features=(
                "sanm",
                "ctc",
                "sentencepiece-unigram",
                "multilingual-asr",
                "language-identification",
                "emotion-recognition",
                "audio-event-detection",
                "forced-ctc-word-timestamps",
                "raw-audio-fine-tuning",
                "portable-native-export",
            ),
        ),
        upstream_revision=FUNASR_SOURCE_REVISION,
        license_id="MIT",
        metadata={
            "checkpoint_license":
            SENSEVOICE_MODEL_LICENSE,
            "family":
            "sensevoice-small",
            "implementation":
            "voicehub-native",
            "published_artifact_revision":
            SENSEVOICE_REVISION,
            "reference_checkpoint_sha256":
            SENSEVOICE_CHECKPOINT_SHA256,
            "verified_scope": (
                "Only the published SenseVoiceSmall SANM-CTC graph is "
                "checkpoint-verified. Paraformer, Fun-ASR-Nano, and other "
                "FunASR registry models require separate architecture "
                "contracts."),
        },
    )


def register_sensevoice_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_SENSEVOICE_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_sensevoice_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_SENSEVOICE_ALIASES",
    "create_sensevoice_architecture_spec",
    "register_sensevoice_architecture",
]
