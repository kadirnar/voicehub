"""Lazy architecture declaration for native Higgs Audio v2."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.higgstts.native.metadata import (
    HIGGS_AUDIO_V2_CHECKPOINT_HEADER_FINGERPRINT,
    HIGGS_AUDIO_V2_CHECKPOINT_LICENSE,
    HIGGS_AUDIO_V2_CHECKPOINT_PARAMETER_COUNT,
    HIGGS_AUDIO_V2_CHECKPOINT_SHA256,
    HIGGS_AUDIO_V2_CHECKPOINT_TENSOR_COUNT,
    HIGGS_AUDIO_V2_CODE_LICENSE,
    HIGGS_AUDIO_V2_CODEC_HEADER_FINGERPRINT,
    HIGGS_AUDIO_V2_CODEC_PARAMETER_COUNT,
    HIGGS_AUDIO_V2_CODEC_TENSOR_COUNT,
    HIGGS_AUDIO_V2_REPOSITORY,
    HIGGS_AUDIO_V2_REVISION,
    HIGGS_AUDIO_V2_SOURCE_REPOSITORY,
    HIGGS_AUDIO_V2_SOURCE_REVISION,
    HIGGS_AUDIO_V2_TOKENIZER_REPOSITORY,
    HIGGS_AUDIO_V2_TOKENIZER_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_HIGGS_AUDIO_V2_ALIASES = (
    "native-higgs-audio-v2",
    "native-higgstts",
)


def create_higgs_audio_v2_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="higgs_audio_v2",
        version="2",
        model_builder=("voicehub.models.higgstts.native.modeling:"
                       "HiggsAudioV2ForConditionalGeneration"),
        config=("voicehub.models.higgstts.native.configuration:"
                "HiggsAudioV2Config"),
        processor=("voicehub.models.higgstts.native.processing:"
                   "HiggsAudioV2Processor"),
        decoder=("voicehub.models.higgstts.native.tokenizer:"
                 "HiggsAudioV2TokenizerModel"),
        objective=(
            "voicehub.models.higgstts.native.modeling:"
            "HiggsAudioV2ForConditionalGeneration.forward"),
        checkpoint_adapter=("voicehub.models.higgstts.native.checkpoint:"
                            "load_higgs_checkpoint"),
        components={
            "artifact-resolver":
            ("voicehub.models.higgstts.native.artifacts:"
             "resolve_higgs_audio_v2_artifacts"),
            "audio-tokenizer": ("voicehub.models.higgstts.native.tokenizer:"
                                "HiggsAudioV2TokenizerModel"),
            "checkpoint-exporter": ("voicehub.models.higgstts.native.checkpoint:"
                                    "export_higgs_checkpoint"),
            "generator": ("voicehub.models.higgstts.native.generation:"
                          "HiggsAudioV2Generator"),
            "runtime": ("voicehub.models.higgstts.native.runtime:"
                        "load_higgs_audio_v2_runtime"),
            "text-tokenizer": ("voicehub.models.higgstts.native.processing:"
                               "HiggsAudioV2TextTokenizer"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=False,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=("compile", "sdpa"),
            features=(
                "llm-tts-codec",
                "24-khz-waveform",
                "25-hz-audio-tokens",
                "custom-license-checkpoint",
                "delayed-multicodebook-generation",
                "expressive-speech",
                "full-sft",
                "llama3-byte-bpe",
                "multilingual",
                "native-dac-hubert-tokenizer",
                "repetition-aware-sampling",
                "strict-safetensors-reload",
                "voice-cloning",
            ),
        ),
        upstream_revision=HIGGS_AUDIO_V2_SOURCE_REVISION,
        license_id=HIGGS_AUDIO_V2_CODE_LICENSE,
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            HIGGS_AUDIO_V2_SOURCE_REPOSITORY,
            "source_revision":
            HIGGS_AUDIO_V2_SOURCE_REVISION,
            "reference_checkpoint":
            HIGGS_AUDIO_V2_REPOSITORY,
            "reference_checkpoint_revision":
            HIGGS_AUDIO_V2_REVISION,
            "reference_checkpoint_sha256": (HIGGS_AUDIO_V2_CHECKPOINT_SHA256),
            "reference_tensor_count": (HIGGS_AUDIO_V2_CHECKPOINT_TENSOR_COUNT),
            "reference_parameter_count": (HIGGS_AUDIO_V2_CHECKPOINT_PARAMETER_COUNT),
            "reference_safetensors_header_fingerprint": (HIGGS_AUDIO_V2_CHECKPOINT_HEADER_FINGERPRINT),
            "audio_tokenizer_checkpoint": (HIGGS_AUDIO_V2_TOKENIZER_REPOSITORY),
            "audio_tokenizer_revision": (HIGGS_AUDIO_V2_TOKENIZER_REVISION),
            "audio_tokenizer_tensor_count": (HIGGS_AUDIO_V2_CODEC_TENSOR_COUNT),
            "audio_tokenizer_parameter_count": (HIGGS_AUDIO_V2_CODEC_PARAMETER_COUNT),
            "audio_tokenizer_header_fingerprint": (HIGGS_AUDIO_V2_CODEC_HEADER_FINGERPRINT),
            "checkpoint_license":
            HIGGS_AUDIO_V2_CHECKPOINT_LICENSE,
            "full_finetuning_ready":
            True,
            "training_boundary": (
                "The trainable dual-FFN decoder uses the source-authored "
                "sum of eight delayed-codebook causal cross-entropies and "
                "optional text causal cross-entropy. The 201M-parameter "
                "HuBERT/DAC tokenizer is frozen. Boson does not publish a "
                "complete optimizer or schedule recipe; VoiceHub's trainer "
                "owns that orchestration without claiming author parity."),
        },
    )


def register_higgs_audio_v2_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_HIGGS_AUDIO_V2_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_higgs_audio_v2_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_HIGGS_AUDIO_V2_ALIASES",
    "create_higgs_audio_v2_architecture_spec",
    "register_higgs_audio_v2_architecture",
]
