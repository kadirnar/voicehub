"""Lazy architecture declaration for native SpeechBrain CRDNN ASR."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_speechbrain.native.metadata import (
    SPEECHBRAIN_ASR_CHECKPOINT_LICENSE,
    SPEECHBRAIN_ASR_REVISION,
    SPEECHBRAIN_ASR_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_SPEECHBRAIN_ASR_ALIASES = (
    "speechbrain-asr",
    "asr-crdnn-rnnlm-librispeech",
)


def create_speechbrain_asr_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="speechbrain-crdnn-asr",
        version="1",
        model_builder=("voicehub.models.asr_speechbrain.native.modeling:"
                       "SpeechBrainCRDNNForASR"),
        config=("voicehub.models.asr_speechbrain.native.configuration:"
                "SpeechBrainCRDNNASRConfig"),
        decoder=("voicehub.models.asr_speechbrain.native.decoding:"
                 "SpeechBrainRNNLMBeamSearch"),
        objective=("voicehub.models.asr_speechbrain.native.modeling:"
                   "speechbrain_sequence_loss"),
        checkpoint_adapter=(
            "voicehub.models.asr_speechbrain.native.checkpoint:"
            "SpeechBrainASRSafeTensorsCheckpointAdapter"),
        components={
            "frontend": ("voicehub.models.asr_speechbrain.native.frontend:"
                         "SpeechBrainASRFrontend"),
            "tokenizer": ("voicehub.tokenization:"
                          "SentencePieceUnigramTokenizer"),
            "pickle-converter":
            ("voicehub.models.asr_speechbrain.native.checkpoint:"
             "convert_speechbrain_asr_checkpoints"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", ),
            checkpoint_formats=(
                "safetensors",
                "trusted-pickle-conversion",
            ),
            training=True,
            streaming=False,
            batched_inference=True,
            features=(
                "legacy-speechbrain-fbank",
                "released-global-cmvn",
                "crdnn",
                "bidirectional-lstm",
                "location-aware-gru-decoder",
                "rnnlm-shallow-fusion",
                "sentencepiece-unigram",
                "combined-ctc-seq2seq-fine-tuning",
                "label-smoothed-sequence-nll",
                "portable-native-export",
            ),
        ),
        upstream_revision=SPEECHBRAIN_ASR_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family":
            "speechbrain-crdnn-asr",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "published_artifact_revision":
            SPEECHBRAIN_ASR_REVISION,
            "checkpoint_license":
            SPEECHBRAIN_ASR_CHECKPOINT_LICENSE,
            "language":
            "en",
            "training_boundary": (
                "The graph, objectives, five-epoch CTC schedule, Adadelta "
                "hyperparameters, validation beam, corpus WER, and NewBob "
                "scheduler match the pinned author recipe. OpenRIR "
                "corruption and speed perturbation remain explicit dataset "
                "transforms so training never downloads data from inside a "
                "model forward pass."),
        },
    )


def register_speechbrain_asr_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_SPEECHBRAIN_ASR_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_speechbrain_asr_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_SPEECHBRAIN_ASR_ALIASES",
    "create_speechbrain_asr_architecture_spec",
    "register_speechbrain_asr_architecture",
]
