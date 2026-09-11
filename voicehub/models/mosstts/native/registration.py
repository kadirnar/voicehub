"""Lazy architecture declaration for VoiceHub-native MOSS-TTS."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.mosstts.native.metadata import (
    MOSS_TTS_CHECKPOINTS,
    OPENMOSS_LICENSE,
    OPENMOSS_TTS_SOURCE,
    OPENMOSS_TTS_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_MOSSTTS_ALIASES = ("mosstts", )


def create_mosstts_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="moss-tts",
        version="1",
        model_builder=("voicehub.models.mosstts.native.modeling:"
                       "build_mosstts_model"),
        config=("voicehub.models.mosstts.native.configuration:MossTTSConfig"),
        processor=("voicehub.models.mosstts.native.processing:MossTTSProcessor"),
        decoder=("voicehub.models.mosstts.native.codec:"
                 "NativeMossAudioCodec"),
        objective=("voicehub.models.mosstts.native.training:"
                   "NativeMossTTSTrainingAdapter"),
        checkpoint_adapter=("voicehub.models.mosstts.native.checkpoint:"
                            "load_mosstts_checkpoint"),
        components={
            "artifact-resolver": ("voicehub.models.mosstts.native.artifacts:"
                                  "resolve_mosstts_artifacts"),
            "checkpoint-exporter": ("voicehub.models.mosstts.native.checkpoint:"
                                    "export_mosstts_checkpoint"),
            "codec-loader": ("voicehub.models.mosstts.native.codec_checkpoint:"
                             "load_moss_audio_tokenizer"),
            "codec-model-v1":
            ("voicehub.models.mosstts.native.codec_modeling_v1:"
             "MossAudioTokenizerV1Model"),
            "codec-model-v2": ("voicehub.models.mosstts.native.codec_modeling:"
                               "MossAudioTokenizerV2Model"),
            "runtime": ("voicehub.models.mosstts.native.runtime:"
                        "MossTTSRuntime"),
            "tokenizer": ("voicehub.models.mosstts.native.tokenization:"
                          "MossTextTokenizer"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=("compile", "sdpa"),
            features=(
                "llm-tts-codec",
                "delay-pattern-generation",
                "local-depth-transformer",
                "local-v1.5-binary-control-head",
                "realtime-training-graph",
                "qwen-byte-bpe",
                "raw-audio-fine-tuning",
                "preencoded-rvq-fine-tuning",
                "strict-safetensors-reload",
                "native-codec-v1-v2",
                "frozen-codec-training-boundary",
            ),
        ),
        upstream_revision=OPENMOSS_TTS_SOURCE_REVISION,
        license_id=OPENMOSS_LICENSE,
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            OPENMOSS_TTS_SOURCE,
            "reference_checkpoints": {
                repository: dict(facts)
                for repository, facts in MOSS_TTS_CHECKPOINTS.items()
            },
            "commercial_use":
            True,
            "full_model_gradient_ready":
            True,
            "training_boundary": (
                "All four semantic graphs train from raw waveforms or "
                "pre-encoded RVQ matrices. The independently versioned native "
                "codec is frozen during semantic-model fine-tuning."),
            "realtime_generation": (
                "Buffered high-level generation follows the published "
                "prefill/depth schedule. Incremental queue/transport streaming "
                "is a separate future runtime surface."),
        },
    )


def register_mosstts_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_MOSSTTS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_mosstts_architecture_spec()
    target.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )
    return spec


__all__ = [
    "DEFAULT_MOSSTTS_ALIASES",
    "create_mosstts_architecture_spec",
    "register_mosstts_architecture",
]
