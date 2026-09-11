"""Lazy architecture declaration for native WeNet GigaSpeech U2++."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_wenet.native.metadata import (
    GIGASPEECH_ARCHIVE_SHA256,
    GIGASPEECH_CHECKPOINT_LICENSE,
    GIGASPEECH_CHECKPOINT_PROVIDER,
    GIGASPEECH_CHECKPOINT_STATUS,
    GIGASPEECH_DOCUMENTATION_NOTE,
    GIGASPEECH_DOCUMENTATION_PATH,
    GIGASPEECH_MODEL_VERSION,
    WENET_CHECKPOINT_LISTING_URL,
    WENET_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_WENET_U2PP_ALIASES = (
    "wenet-asr",
    "wenet-u2pp",
    "gigaspeech-u2pp",
)


def create_wenet_u2pp_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="wenet-gigaspeech-u2pp",
        version="1",
        model_builder=("voicehub.models.asr_wenet.native.modeling:"
                       "WeNetU2PPForASR"),
        config=("voicehub.models.asr_wenet.native.configuration:"
                "WeNetU2PPConfig"),
        processor=("voicehub.models.asr_wenet.native.tokenization:"
                   "WeNetGigaSpeechTokenizer"),
        objective=("voicehub.models.asr_wenet.native.modeling:"
                   "wenet_u2pp_hybrid_loss"),
        checkpoint_adapter=(
            "voicehub.models.asr_wenet.native.checkpoint:"
            "WeNetU2PPSafeTensorsCheckpointAdapter"),
        components={
            "attention_decoder": ("voicehub.models.asr_wenet.native.decoding:"
                                  "attention_rescore"),
            "wenet-converter":
            ("voicehub.models.asr_wenet.native.checkpoint:"
             "convert_wenet_gigaspeech_checkpoint"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=(
                "safetensors",
                "trust-gated-verified-pytorch-conversion",
            ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            features=(
                "conformer",
                "u2pp",
                "ctc-prefix-beam-search",
                "bidirectional-attention-rescoring",
                "sentencepiece-unigram",
                "raw-audio-fine-tuning",
                "token-timestamps",
            ),
        ),
        upstream_revision=WENET_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family":
            "wenet-u2pp",
            "implementation":
            "voicehub-native",
            "reference_checkpoint":
            "gigaspeech-u2pp-conformer",
            "reference_checkpoint_version":
            GIGASPEECH_MODEL_VERSION,
            "reference_checkpoint_sha256":
            GIGASPEECH_ARCHIVE_SHA256,
            "reference_checkpoint_url":
            WENET_CHECKPOINT_LISTING_URL,
            "reference_checkpoint_status":
            GIGASPEECH_CHECKPOINT_STATUS,
            "checkpoint_provider":
            GIGASPEECH_CHECKPOINT_PROVIDER,
            "documentation_checkpoint_path":
            GIGASPEECH_DOCUMENTATION_PATH,
            "documentation_checkpoint_note":
            GIGASPEECH_DOCUMENTATION_NOTE,
            "checkpoint_license":
            GIGASPEECH_CHECKPOINT_LICENSE,
            "verified_scope": (
                "Only the 20210728 English GigaSpeech U2++ Conformer graph "
                "is checkpoint-verified. Other WeNet graph variants require "
                "their own native architecture contracts."),
        },
    )


def register_wenet_u2pp_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_WENET_U2PP_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_wenet_u2pp_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_WENET_U2PP_ALIASES",
    "create_wenet_u2pp_architecture_spec",
    "register_wenet_u2pp_architecture",
]
