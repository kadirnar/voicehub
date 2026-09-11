"""Lazy architecture declaration for native ESPnet Transformer ASR."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_espnet.native.metadata import (
    ESPNET_CHECKPOINT_LICENSE,
    ESPNET_REVISION,
    ESPNET_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_ESPNET_ALIASES = (
    "espnet-asr",
    "native-espnet-asr",
    "librispeech-transformer-e18",
)


def create_espnet_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="espnet-librispeech-transformer-e18",
        version="1",
        model_builder=("voicehub.models.asr_espnet.native.modeling:"
                       "ESPnetLibriSpeechTransformerForASR"),
        config=("voicehub.models.asr_espnet.native.configuration:"
                "ESPnetLibriSpeechTransformerConfig"),
        decoder=("voicehub.models.asr_espnet.native.decoding:"
                 "ESPnetJointBeamSearch"),
        objective=("voicehub.models.asr_espnet.native.modeling:"
                   "espnet_label_smoothed_loss"),
        checkpoint_adapter=(
            "voicehub.models.asr_espnet.native.checkpoint:"
            "ESPnetASRSafeTensorsCheckpointAdapter"),
        components={
            "frontend": ("voicehub.models.asr_espnet.native.frontend:"
                         "ESPnetDefaultFrontend"),
            "global-mvn": ("voicehub.models.asr_espnet.native.frontend:"
                           "ESPnetGlobalMVN"),
            "language-model":
            ("voicehub.models.asr_espnet.native.modeling:"
             "ESPnetSequentialRNNLanguageModel"),
            "pickle-converter":
            ("voicehub.models.asr_espnet.native.checkpoint:"
             "convert_espnet_librispeech_checkpoints"),
            "specaugment": ("voicehub.models.asr_espnet.native.frontend:"
                            "ESPnetSpecAugment"),
            "tokenizer": ("voicehub.models.asr_espnet.native.tokenization:"
                          "ESPnetLibriSpeechTokenizer"),
            "training-adapter":
            ("voicehub.models.asr_espnet.native.training:"
             "NativeESPnetASRTrainingAdapter"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", ),
            checkpoint_formats=(
                "safetensors",
                "trusted-pickle-conversion",
            ),
            export_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            optimization_passes=("compile", "sdpa"),
            features=(
                "checkpoint-stored-slaney-log-mel",
                "global-cmvn",
                "conv2d6",
                "transformer-encoder",
                "transformer-decoder",
                "ctc-prefix-beam-search",
                "lstm-language-model",
                "sentencepiece-unigram-remapping",
                "combined-ctc-seq2seq-fine-tuning",
                "published-specaugment",
                "portable-native-export",
            ),
        ),
        upstream_revision=ESPNET_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "checkpoint_license":
            ESPNET_CHECKPOINT_LICENSE,
            "family":
            "espnet-transformer-asr",
            "implementation":
            "voicehub-native",
            "language":
            "en",
            "published_artifact_revision":
            ESPNET_REVISION,
            "tensor_backend":
            "pytorch",
            "training_boundary": (
                "The graph, raw-waveform frontend, global MVN, SpecAugment, "
                "hybrid 0.3 CTC/0.7 attention loss, Adam optimizer, and "
                "25,000-step WarmupLR match the pinned recipe. Corpus "
                "preparation and speed perturbation remain explicit dataset "
                "operations."),
            "verified_scope": (
                "Only the LibriSpeech Transformer e18 release is "
                "checkpoint-compatible; other ESPnet graph families are "
                "rejected."),
        },
    )


def register_espnet_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_ESPNET_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_espnet_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_ESPNET_ALIASES",
    "create_espnet_architecture_spec",
    "register_espnet_architecture",
]
