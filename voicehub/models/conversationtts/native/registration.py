"""Lazy native architecture declaration for ConversationTTS."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.conversationtts.native.metadata import (
    CONVERSATIONTTS_CHECKPOINT_FILENAME,
    CONVERSATIONTTS_CHECKPOINT_REPOSITORY,
    CONVERSATIONTTS_CHECKPOINT_REVISION,
    CONVERSATIONTTS_CHECKPOINT_SHA256,
    CONVERSATIONTTS_CHECKPOINT_SIZE,
    CONVERSATIONTTS_LICENSE,
    CONVERSATIONTTS_MIMI_FILENAME,
    CONVERSATIONTTS_MIMI_REPOSITORY,
    CONVERSATIONTTS_MIMI_REVISION,
    CONVERSATIONTTS_SOURCE_REPOSITORY,
    CONVERSATIONTTS_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_CONVERSATIONTTS_ALIASES = (
    "native-conversationtts",
    "conversation-tts",
    "speechfoundation",
)


def create_conversationtts_architecture_spec() -> ArchitectureSpec:
    """Describe the audited graph without importing PyTorch."""
    return ArchitectureSpec(
        architecture_id="conversationtts",
        version="1",
        model_builder=("voicehub.models.conversationtts.native.modeling:"
                       "ConversationTTSModel"),
        config=("voicehub.models.conversationtts.native.modeling:"
                "ConversationTTSArchitectureConfig"),
        processor=("voicehub.models.conversationtts.native.processing:"
                   "build_conversationtts_sequence"),
        decoder=(
            "voicehub.models.conversationtts.source.conversationtts.tools."
            "tokenizer.MimiCodec.mimi_tokenizer:MimiTokenizer"),
        objective=("voicehub.models.conversationtts.training:"
                   "ConversationTTSTrainingAdapter"),
        checkpoint_adapter=(
            "voicehub.models.conversationtts.native.checkpoint:"
            "load_conversationtts_checkpoint"),
        components={
            "checkpoint-exporter":
            ("voicehub.models.conversationtts.native.checkpoint:"
             "export_conversationtts_checkpoint"),
            "decoder-core": ("voicehub.models.conversationtts.native.decoder:"
                             "ConversationDecoder"),
            "text-tokenizer": (
                "voicehub.models.conversationtts.source.conversationtts."
                "tools.tokenizer.Text2ID.text_tokenizer:TextTokenizer"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=(
                "pytorch-weights-only",
                "safetensors",
            ),
            training=True,
            streaming=False,
            batched_inference=False,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=(
                "compile",
                "sdpa",
                "attention-backend",
                "custom-kernels",
            ),
            features=(
                "llm-tts-codec",
                "autoregressive-32-codebook-audio",
                "flash-attention-4-optional",
                "frozen-native-mimi",
                "fused-swiglu-kernels",
                "multilingual",
                "native-byte-bpe",
                "raw-audio-fine-tuning",
                "safetensors-export",
                "speaker-context",
                "strict-checkpoint-validation",
                "voice-cloning",
                "no-external-model-runtime",
            ),
        ),
        upstream_revision=CONVERSATIONTTS_SOURCE_REVISION,
        license_id=CONVERSATIONTTS_LICENSE,
        metadata={
            "external_llm_backend_blocker": (
                "ConversationTTS requires a global transformer plus a "
                "hidden-state conditioned depth decoder."),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            CONVERSATIONTTS_SOURCE_REPOSITORY,
            "source_revision":
            CONVERSATIONTTS_SOURCE_REVISION,
            "reference_checkpoint":
            CONVERSATIONTTS_CHECKPOINT_REPOSITORY,
            "reference_checkpoint_revision": (CONVERSATIONTTS_CHECKPOINT_REVISION),
            "reference_checkpoint_filename": (CONVERSATIONTTS_CHECKPOINT_FILENAME),
            "reference_checkpoint_sha256": (CONVERSATIONTTS_CHECKPOINT_SHA256),
            "reference_checkpoint_size": (CONVERSATIONTTS_CHECKPOINT_SIZE),
            "reference_checkpoint_boundary": (
                "The published checkpoint is a PyTorch archive. VoiceHub "
                "loads it only through weights_only=True and exports "
                "steady-state artifacts as Safetensors."),
            "codec_checkpoint":
            CONVERSATIONTTS_MIMI_REPOSITORY,
            "codec_checkpoint_revision":
            CONVERSATIONTTS_MIMI_REVISION,
            "codec_checkpoint_filename":
            CONVERSATIONTTS_MIMI_FILENAME,
            "training_objective": ("published-two-level-masked-codebook-cross-entropy"),
            "training_inputs": ("raw-text-audio-or-source-framed-token-streams"),
            "full_finetuning_ready":
            True,
            "inference_reloadable_export":
            True,
            "commercial_use":
            False,
        },
    )


def register_conversationtts_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_CONVERSATIONTTS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_conversationtts_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_CONVERSATIONTTS_ALIASES",
    "create_conversationtts_architecture_spec",
    "register_conversationtts_architecture",
]
