"""Lazy architecture declaration for VoiceHub-native NeuTTS."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.neutts.native.metadata import (
    NEUCODEC_REFERENCE,
    NEUCODEC_SOURCE_REVISION,
    NEUTTS_SOURCE_REVISION,
    NEUTTS_TRAINING_SOURCE,
    NEUTTS_VARIANTS,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_NEUTTS_ALIASES = (
    "native-neutts",
    "neuphonic-neutts",
    "neutts-air",
    "neutts-nano",
    "neutts-2e",
)


def create_neutts_architecture_spec() -> ArchitectureSpec:
    """Describe the native LM, tokenizer, NeuCodec, and Air objective."""
    return ArchitectureSpec(
        architecture_id="neutts",
        version="1",
        model_builder=("voicehub.models.neutts.native.modeling:NeuTTSBackbone"),
        config=("voicehub.models.neutts.native.configuration:"
                "NeuTTSBackboneConfig"),
        processor=("voicehub.models.neutts.native.tokenization:NeuTTSTokenizer"),
        decoder="voicehub.models.neutts.native.neucodec:NeuCodecModel",
        objective=("voicehub.models.neutts.training:NeuTTSTrainingAdapter"),
        checkpoint_adapter=("voicehub.models.neutts.native.checkpoint:"
                            "NeuTTSCheckpointAdapter"),
        components={
            "artifact-resolver": ("voicehub.models.neutts.native.artifacts:"
                                  "resolve_neutts_artifacts"),
            "audio-codec": ("voicehub.models.neutts.native.neucodec:NeuCodecModel"),
            "codec-artifact-resolver":
            ("voicehub.models.neutts.native.artifacts:"
             "resolve_neucodec_artifacts"),
            "codec-checkpoint-adapter":
            ("voicehub.models.neutts.native.checkpoint:"
             "NeuCodecCheckpointAdapter"),
            "runtime": ("voicehub.models.neutts.native.modeling:NeuTTSRuntime"),
            "sft-dataset": ("voicehub.models.neutts.training:NeuTTSSFTDataset"),
            "wrapper": ("voicehub.models.neutts.modeling:"
                        "NeuTTSForTextToSpeech"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda"),
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
                "completion-only-codec-language-modeling",
                "emotion-control",
                "frozen-neucodec",
                "llama-linear-rope",
                "multilingual",
                "native-byte-bpe-tokenizer",
                "native-neucodec",
                "phoneme-injection",
                "preencoded-code-fine-tuning",
                "qwen2-qwen3-llama-backbones",
                "raw-audio-fine-tuning",
                "strict-safetensors-reload",
                "torch-compile-inference-unsafe",
                "voice-cloning",
            ),
        ),
        upstream_revision=NEUTTS_SOURCE_REVISION,
        license_id="NeuTTS-Open-License-1.0",
        metadata={
            "external_llm_backend_blocker": (
                "NeuTTS requires checkpoint-gated RoPE behavior and "
                "minimum-token EOS masking that are not represented by the "
                "generic server contract."),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            "neuphonic/neutts",
            "source_revision":
            NEUTTS_SOURCE_REVISION,
            "reference_checkpoints": {
                model_id: {
                    "revision": values["revision"],
                    "license": values["license"],
                    "tensor_count": values["tensor_count"],
                }
                for model_id, values in NEUTTS_VARIANTS.items()
            },
            "codec_checkpoint":
            NEUCODEC_REFERENCE["model_id"],
            "codec_checkpoint_revision":
            NEUCODEC_REFERENCE["revision"],
            "codec_checkpoint_sha256":
            NEUCODEC_REFERENCE["sha256"],
            "codec_checkpoint_size":
            NEUCODEC_REFERENCE["size"],
            "codec_tensor_count":
            NEUCODEC_REFERENCE["tensor_count"],
            "codec_source_revision":
            NEUCODEC_SOURCE_REVISION,
            "training_source":
            NEUTTS_TRAINING_SOURCE["repository"],
            "training_source_revision":
            NEUTTS_TRAINING_SOURCE["revision"],
            "training_recipe":
            NEUTTS_TRAINING_SOURCE["recipe"],
            "verified_training_family":
            NEUTTS_TRAINING_SOURCE["model_family"],
            "full_finetuning_ready":
            True,
            "training_boundary": (
                "The pinned completion-only objective is verified for "
                "NeuTTS-Air. Its language model is trainable while native "
                "NeuCodec remains frozen; raw audio or precomputed codes are "
                "accepted. Nano and 2E objectives fail closed until an "
                "author-equivalent recipe is verified."),
            "phonemizer_boundary": (
                "Phoneme variants require precomputed phonemes or an "
                "explicitly injected phonemizer; no eSpeak process or "
                "grapheme approximation runs implicitly."),
        },
    )


def register_neutts_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_NEUTTS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register NeuTTS without importing its PyTorch graph."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_neutts_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_NEUTTS_ALIASES",
    "create_neutts_architecture_spec",
    "register_neutts_architecture",
]
