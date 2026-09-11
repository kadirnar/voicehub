"""Lazy declaration for the VoiceHub-native SpeechT5 architecture."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.speecht5.native.metadata import (
    NATIVE_SPEECHT5_FORMAT,
    SPEECHT5_HIFIGAN_CONFIG_SHA256,
    SPEECHT5_HIFIGAN_REFERENCE_INVENTORY,
    SPEECHT5_PROCESSOR_INTEGRITY,
    SPEECHT5_REFERENCE_INVENTORY,
    SPEECHT5_SOURCE_FILES,
    SPEECHT5_SOURCE_LICENSE,
    SPEECHT5_SOURCE_REPOSITORY,
    SPEECHT5_SOURCE_REVISION,
    SPEECHT5_SOURCE_TAG,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_SPEECHT5_ALIASES = (
    "native-speecht5",
    "speecht5-tts",
    "microsoft-speecht5-tts",
)


def create_speecht5_architecture_spec() -> ArchitectureSpec:
    """Describe the audited SpeechT5 graph without importing PyTorch."""
    return ArchitectureSpec(
        architecture_id="speecht5",
        version="1",
        model_builder=("voicehub.models.speecht5.native_modeling:"
                       "SpeechT5ForTextToSpeechModel"),
        config=("voicehub.models.speecht5.native_configuration:"
                "NativeSpeechT5Config"),
        processor=("voicehub.models.speecht5.processing:"
                   "SpeechT5Processor"),
        decoder=("voicehub.models.speecht5.native_modeling:"
                 "SpeechT5HifiGan"),
        objective=("voicehub.models.speecht5.training:"
                   "NativeSpeechT5TrainingAdapter"),
        checkpoint_adapter=("voicehub.models.speecht5.checkpoint:"
                            "SpeechT5CheckpointAdapter"),
        components={
            "artifact-resolver": ("voicehub.models.speecht5.artifacts:"
                                  "resolve_speecht5_artifacts"),
            "checkpoint-exporter": ("voicehub.models.speecht5.checkpoint:"
                                    "save_speecht5_checkpoint"),
            "checkpoint-loader": ("voicehub.models.speecht5.checkpoint:"
                                  "load_speecht5_checkpoint"),
            "feature-extractor": ("voicehub.models.speecht5.processing:"
                                  "SpeechT5FeatureExtractor"),
            "inference-runtime": ("voicehub.models.speecht5.modeling:"
                                  "SpeechT5ForTextToSpeech"),
            "public-config": ("voicehub.models.speecht5.modeling:"
                              "SpeechT5Config"),
            "text-tokenizer": ("voicehub.models.speecht5.processing:"
                               "SpeechT5Tokenizer"),
            "vocoder-artifact-resolver":
            ("voicehub.models.speecht5.artifacts:"
             "resolve_speecht5_hifigan_artifacts"),
            "vocoder-checkpoint-adapter":
            ("voicehub.models.speecht5.checkpoint:"
             "SpeechT5HifiGanCheckpointAdapter"),
            "vocoder-config":
            ("voicehub.models.speecht5.native_configuration:"
             "NativeSpeechT5HifiGanConfig"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", "bin", "pytorch"),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=("compile", ),
            features=(
                "encoder-decoder-text-to-spectrogram",
                "sentencepiece-character-tokenizer",
                "native-log-mel-frontend",
                "speaker-xvector-conditioning",
                "native-hifigan-vocoder",
                "guided-multihead-attention-loss",
                "raw-audio-fine-tuning",
                "full-acoustic-model-fine-tuning",
                "frozen-vocoder",
                "batched-generation",
                "strict-checkpoint-validation",
                "restricted-pytorch-import",
                "safetensors-export",
                "no-external-runtime",
            ),
        ),
        upstream_revision=SPEECHT5_SOURCE_REVISION,
        license_id=SPEECHT5_SOURCE_LICENSE,
        metadata={
            "family":
            "speecht5",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            SPEECHT5_SOURCE_REPOSITORY,
            "source_revision":
            SPEECHT5_SOURCE_REVISION,
            "source_tag":
            SPEECHT5_SOURCE_TAG,
            "source_files":
            SPEECHT5_SOURCE_FILES,
            "reference_checkpoint":
            SPEECHT5_REFERENCE_INVENTORY,
            "vocoder_reference_checkpoint":
            SPEECHT5_HIFIGAN_REFERENCE_INVENTORY,
            "processor_asset_sha256":
            SPEECHT5_PROCESSOR_INTEGRITY,
            "vocoder_config_sha256":
            SPEECHT5_HIFIGAN_CONFIG_SHA256,
            "native_checkpoint_format":
            NATIVE_SPEECHT5_FORMAT,
            "official_safetensors_published":
            False,
            "checkpoint_import_boundary":
            "restricted-pytorch-weights-only-plus-complete-inventory",
            "training_scope":
            "complete-text-to-spectrogram",
            "training_objective": (
                "pre-postnet-and-postnet-l1",
                "weighted-stop-token-bce",
                "guided-multihead-cross-attention",
            ),
            "always_frozen_components": ("vocoder", ),
            "raw_audio_finetuning_ready":
            True,
            "full_finetuning_ready":
            True,
            "speaker_embedding_dimension":
            512,
            "mel_bins":
            80,
            "sampling_rate":
            16_000,
            "official_checkpoint_language":
            "en",
        },
    )


def register_speecht5_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_SPEECHT5_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register SpeechT5 without importing its graph or PyTorch."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_speecht5_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_SPEECHT5_ALIASES",
    "create_speecht5_architecture_spec",
    "register_speecht5_architecture",
]
