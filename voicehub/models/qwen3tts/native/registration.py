"""Lazy native-architecture declaration for Qwen3-TTS."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.qwen3tts.native.metadata import QWEN3_TTS_CHECKPOINTS, QWEN3_TTS_SOURCE
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_QWEN3_TTS_ALIASES = (
    "native-qwen3-tts",
    "qwen3-tts-12hz",
)


def create_qwen3_tts_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="qwen3-tts",
        version="1",
        model_builder=("voicehub.models.qwen3tts.native.modeling:"
                       "Qwen3TTSForConditionalGeneration"),
        config=("voicehub.models.qwen3tts.native.configuration:"
                "Qwen3TTSArchitectureConfig"),
        processor=("voicehub.models.qwen3tts.native.runtime:Qwen3TTSProcessor"),
        decoder=("voicehub.models.qwen3tts.native.codec:Qwen3TTSSpeechDecoder"),
        objective="voicehub.objectives.sequence:sequence_cross_entropy",
        checkpoint_adapter=("voicehub.models.qwen3tts.native.checkpoint:"
                            "load_qwen3_tts_model_checkpoint"),
        components={
            "artifact-resolver": ("voicehub.models.qwen3tts.native.artifacts:"
                                  "resolve_qwen3_tts_artifacts"),
            "speaker-encoder": ("voicehub.models.qwen3tts.native.modeling:"
                                "Qwen3TTSSpeakerEncoder"),
            "speech-encoder": ("voicehub.models.qwen3tts.native.encoder:"
                               "Qwen3TTSSpeechEncoder"),
            "text-tokenizer": ("voicehub.models.qwen3tts.native.tokenization:"
                               "Qwen3TTSTextTokenizer"),
            "lora": ("voicehub.models.qwen3tts.lora:"
                     "inject_qwen3_tts_lora"),
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
            optimization_passes=(
                "compile",
                "lora",
                "sdpa",
                "attention-backend",
                "custom-kernels",
            ),
            features=(
                "llm-tts-codec",
                "autoregressive-codebooks",
                "custom-voice",
                "delayed-codebook-sft",
                "flash-attention-4-optional",
                "fused-codec-snake-beta-kernels",
                "fused-swiglu-kernels",
                "multilingual",
                "native-icl-reference-audio",
                "native-speech-tokenizer-encoder",
                "native-lora-fine-tuning",
                "speaker-encoder",
                "voice-clone-xvector",
                "voice-design",
            ),
        ),
        upstream_revision=str(QWEN3_TTS_SOURCE["revision"]),
        license_id="Apache-2.0",
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            QWEN3_TTS_SOURCE,
            "official_training_documentation": (
                f"{QWEN3_TTS_SOURCE['repository']}/blob/"
                f"{QWEN3_TTS_SOURCE['revision']}/finetuning/README.md"),
            "official_training_source": (
                f"{QWEN3_TTS_SOURCE['repository']}/blob/"
                f"{QWEN3_TTS_SOURCE['revision']}/finetuning/sft_12hz.py"),
            "reference_checkpoints":
            QWEN3_TTS_CHECKPOINTS,
            "training_boundary": (
                "Exact official 12 Hz Base single-speaker SFT objective with "
                "pre-extracted 16-codebook targets and frozen speaker encoder. "
                "Full-model SFT remains the default; opt-in VoiceHub-native "
                "LoRA adapts only the talker and residual code-predictor "
                "attention/MLP projections."),
            "lora_finetuning_ready":
            True,
            "lora_implementation":
            "voicehub-native-no-peft",
            "lora_export": (
                "adapter-only Safetensors plus clone-merged inference "
                "checkpoint without mutating live training weights"),
            "upstream_lora_recipe_published":
            False,
            "icl_reference_encoder": (
                "Checkpoint-exact native Mimi-derived encoder, causal "
                "Transformer, downsampler, and semantic/acoustic residual "
                "quantizers support raw-reference ICL codes; legacy local "
                "decoder-only exports remain loadable without an encoder."),
            "reference_audio_boundary": (
                "Native paths and URLs accept PCM WAVE; other containers "
                "must be supplied as predecoded tensors."),
        },
    )


def register_qwen3_tts_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_QWEN3_TTS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_qwen3_tts_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_QWEN3_TTS_ALIASES",
    "create_qwen3_tts_architecture_spec",
    "register_qwen3_tts_architecture",
]
