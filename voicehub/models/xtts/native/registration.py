"""Lazy declaration for VoiceHub's native XTTS v2 architecture."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.xtts.native.metadata import (
    XTTS2_CHECKPOINT_LICENSE,
    XTTS2_CHECKPOINT_REPOSITORY,
    XTTS2_CHECKPOINT_REVISION,
    XTTS2_CONFIG_SHA256,
    XTTS2_DVAE_SHA256,
    XTTS2_DVAE_SIZE_BYTES,
    XTTS2_DVAE_STORED_ELEMENT_COUNT,
    XTTS2_DVAE_TENSOR_COUNT,
    XTTS2_MEL_STATS_SHA256,
    XTTS2_MEL_STATS_SIZE_BYTES,
    XTTS2_NATIVE_PARAMETER_COUNT,
    XTTS2_NATIVE_TENSOR_COUNT,
    XTTS2_SOURCE_LICENSE,
    XTTS2_SOURCE_REPOSITORY,
    XTTS2_SOURCE_REVISION,
    XTTS2_VOCAB_SHA256,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_XTTS2_ALIASES = (
    "native-xtts",
    "xtts",
    "xtts-v2",
)


def create_xtts2_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="xtts2",
        version="2",
        model_builder="voicehub.models.xtts.native.modeling:XTTS2Model",
        config="voicehub.models.xtts.native.configuration:XTTS2Config",
        processor="voicehub.models.xtts.native.tokenizer:XTTS2Tokenizer",
        decoder="voicehub.models.xtts.native.decoder:HifiDecoder",
        objective="voicehub.models.xtts.native.gpt:XTTS2GPT.forward",
        checkpoint_adapter=("voicehub.models.xtts.native.checkpoint:"
                            "load_xtts2_checkpoint"),
        components={
            "acoustic-code-autoencoder": ("voicehub.models.xtts.native.dvae:XTTS2DVAE"),
            "acoustic-code-encoder": ("voicehub.models.xtts.native.dvae:"
                                      "XTTS2TrainingAudioEncoder"),
            "checkpoint-exporter": ("voicehub.models.xtts.native.checkpoint:"
                                    "save_xtts2_checkpoint"),
            "conditioning-encoder": ("voicehub.models.xtts.native.conditioning:"
                                     "ConditioningEncoder"),
            "legacy-converter":
            ("voicehub.models.xtts.native.checkpoint:"
             "convert_trusted_legacy_xtts2_checkpoint"),
            "legacy-dvae-converter":
            ("voicehub.models.xtts.native.dvae_checkpoint:"
             "convert_trusted_legacy_xtts2_dvae_checkpoint"),
            "legacy-mel-stats-converter":
            ("voicehub.models.xtts.native.dvae_checkpoint:"
             "convert_trusted_legacy_xtts2_mel_stats"),
            "speaker-encoder": ("voicehub.models.xtts.native.decoder:"
                                "ResNetSpeakerEncoder"),
            "trainer-adapter": ("voicehub.models.xtts.training_xtts:"
                                "XTTSTrainingAdapter"),
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
                "17-languages",
                "24-khz-waveform",
                "autoregressive-acoustic-tokens",
                "gpt-finetuning",
                "hifigan-decoder",
                "native-dvae-acoustic-encoder",
                "perceiver-conditioning",
                "precomputed-audio-code-training",
                "raw-waveform-code-preparation",
                "reference-voice-cloning",
                "resnet-speaker-encoder",
                "separate-dvae-artifact",
                "strict-safetensors-runtime",
            ),
        ),
        upstream_revision=XTTS2_SOURCE_REVISION,
        license_id=XTTS2_SOURCE_LICENSE,
        metadata={
            "external_llm_backend_blocker":
            ("XTTS requires conditioning embeddings and generated hidden "
             "states."),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            XTTS2_SOURCE_REPOSITORY,
            "source_revision":
            XTTS2_SOURCE_REVISION,
            "reference_checkpoint":
            XTTS2_CHECKPOINT_REPOSITORY,
            "reference_checkpoint_revision":
            XTTS2_CHECKPOINT_REVISION,
            "reference_config_sha256":
            XTTS2_CONFIG_SHA256,
            "reference_vocab_sha256":
            XTTS2_VOCAB_SHA256,
            "reference_native_tensor_count":
            XTTS2_NATIVE_TENSOR_COUNT,
            "reference_native_parameter_count":
            XTTS2_NATIVE_PARAMETER_COUNT,
            "reference_dvae_sha256":
            XTTS2_DVAE_SHA256,
            "reference_dvae_size_bytes":
            XTTS2_DVAE_SIZE_BYTES,
            "reference_dvae_tensor_count":
            XTTS2_DVAE_TENSOR_COUNT,
            "reference_dvae_stored_element_count":
            XTTS2_DVAE_STORED_ELEMENT_COUNT,
            "reference_mel_stats_sha256":
            XTTS2_MEL_STATS_SHA256,
            "reference_mel_stats_size_bytes":
            XTTS2_MEL_STATS_SIZE_BYTES,
            "checkpoint_license":
            XTTS2_CHECKPOINT_LICENSE,
            "steady_state_checkpoint_format":
            "safetensors",
            "training_boundary": (
                "The complete autoregressive GPT is trainable with the "
                "source text and acoustic-token cross-entropies. Audio codes "
                "may be precomputed or produced by the separately loaded, "
                "frozen native DVAE; the speaker encoder and HiFi-GAN "
                "decoder remain frozen."),
            "dvae_artifact_boundary": (
                "Coqui distributes dvae.pth and mel_stats.pth separately "
                "from model.pth at the pinned checkpoint revision. VoiceHub "
                "requires explicit restricted conversion and then loads "
                "only strict Safetensors artifacts."),
            "legacy_boundary": (
                "Coqui publishes model.pth. Native runtime loading never "
                "deserializes it; an explicit weights-only one-time "
                "conversion produces the required model.safetensors."),
        },
    )


def register_xtts2_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_XTTS2_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_xtts2_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_XTTS2_ALIASES",
    "create_xtts2_architecture_spec",
    "register_xtts2_architecture",
]
