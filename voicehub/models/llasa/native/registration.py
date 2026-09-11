"""Lazy architecture declaration for VoiceHub-native LLaSA."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_LLASSA_ALIASES = (
    "native-llasa",
    "llasa-tts",
    "llasa-1b-multilingual",
)

LLASSA_CHECKPOINT_REPOSITORY = "HKUSTAudio/Llasa-1B-Multilingual"
LLASSA_CHECKPOINT_REVISION = "7f094cb62b0a9779b334c60d039a61c5a6e04456"
LLASSA_CHECKPOINT_TENSOR_COUNT = 147
LLASSA_TRAINING_REPOSITORY = "zhenye234/LLaSA_training"
LLASSA_TRAINING_REVISION = "479acd5277220f78a72093f63755c0892838d0c5"
XCODEC2_CHECKPOINT_REPOSITORY = "HKUSTAudio/xcodec2-hf"
XCODEC2_CHECKPOINT_REVISION = "64bd034d12d441299cdd535b15c33efd6ccdf252"
XCODEC2_CHECKPOINT_SHA256 = "611a63e4dff70c19bd4718d701bb7bc522acf6293a109ab62f5db2f7ff395114"
XCODEC2_CHECKPOINT_SIZE = 2_517_231_448
XCODEC2_CHECKPOINT_TENSOR_COUNT = 811
XCODEC2_REFERENCE_REVISION = "7f5d5d1aaca3cc3d236c80ec8cb34d06f08a5fb8"


def create_llasa_architecture_spec() -> ArchitectureSpec:
    """Describe the complete native LLaSA LM/tokenizer/codec runtime."""
    return ArchitectureSpec(
        architecture_id="llasa",
        version="1",
        model_builder=("voicehub.models.causal_lm.native.modeling:"
                       "LlamaForCausalLM"),
        config=("voicehub.models.causal_lm.native.configuration:"
                "LlamaConfig"),
        processor="voicehub.models.llasa.tokenization_llasa:LlasaTokenizer",
        decoder="voicehub.models.llasa.xcodec2:XCodec2Model",
        objective="voicehub.models.llasa.training:LlasaTrainingAdapter",
        checkpoint_adapter=(
            "voicehub.models.causal_lm.native.checkpoint:"
            "HuggingFaceCausalLMCheckpointAdapter"),
        components={
            "artifact-resolver": "voicehub.models.llasa.artifacts:resolve_llasa_artifacts",
            "audio-codec": "voicehub.models.llasa.xcodec2:XCodec2Model",
            "codec-artifact-resolver": "voicehub.models.llasa.artifacts:resolve_xcodec2_artifacts",
            "codec-checkpoint-adapter": "voicehub.models.llasa.checkpoint:XCodec2CheckpointAdapter",
            "runtime": "voicehub.models.llasa.modeling:LlasaForTextToSpeech",
            "sft-dataset": "voicehub.models.llasa.training:LlasaSFTDataset",
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
                "frozen-xcodec2",
                "llama3-scaled-rope",
                "multilingual",
                "native-byte-bpe-tokenizer",
                "native-xcodec2",
                "preencoded-code-fine-tuning",
                "raw-audio-fine-tuning",
                "strict-safetensors-reload",
                "voice-cloning",
            ),
        ),
        upstream_revision=LLASSA_CHECKPOINT_REVISION,
        license_id="CC-BY-NC-4.0",
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "reference_checkpoint":
            LLASSA_CHECKPOINT_REPOSITORY,
            "reference_checkpoint_revision":
            LLASSA_CHECKPOINT_REVISION,
            "reference_tensor_count":
            LLASSA_CHECKPOINT_TENSOR_COUNT,
            "reference_vocabulary_size":
            193_800,
            "codec_checkpoint":
            XCODEC2_CHECKPOINT_REPOSITORY,
            "codec_checkpoint_revision":
            XCODEC2_CHECKPOINT_REVISION,
            "codec_checkpoint_sha256":
            XCODEC2_CHECKPOINT_SHA256,
            "codec_checkpoint_size":
            XCODEC2_CHECKPOINT_SIZE,
            "codec_tensor_count":
            XCODEC2_CHECKPOINT_TENSOR_COUNT,
            "codec_architecture_reference_revision":
            XCODEC2_REFERENCE_REVISION,
            "training_source":
            LLASSA_TRAINING_REPOSITORY,
            "training_source_revision":
            LLASSA_TRAINING_REVISION,
            "checkpoint_license":
            "CC-BY-NC-4.0",
            "full_finetuning_ready":
            True,
            "training_boundary": (
                "The LLaSA language model is trainable with completion-only "
                "cross-entropy. The separately pinned native XCodec2 graph "
                "is frozen and may be bypassed with precomputed codes."),
        },
    )


def register_llasa_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_LLASSA_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register LLaSA without importing its model or codec graphs."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_llasa_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_LLASSA_ALIASES",
    "LLASSA_CHECKPOINT_REPOSITORY",
    "LLASSA_CHECKPOINT_REVISION",
    "LLASSA_CHECKPOINT_TENSOR_COUNT",
    "LLASSA_TRAINING_REPOSITORY",
    "LLASSA_TRAINING_REVISION",
    "XCODEC2_CHECKPOINT_REPOSITORY",
    "XCODEC2_CHECKPOINT_REVISION",
    "XCODEC2_CHECKPOINT_SHA256",
    "XCODEC2_CHECKPOINT_SIZE",
    "XCODEC2_CHECKPOINT_TENSOR_COUNT",
    "XCODEC2_REFERENCE_REVISION",
    "create_llasa_architecture_spec",
    "register_llasa_architecture",
]
