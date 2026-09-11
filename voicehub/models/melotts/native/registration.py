"""Lazy declaration for VoiceHub's native MeloTTS architecture."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.melotts.native.metadata import (
    MELOTTS_SOURCE_LICENSE,
    MELOTTS_SOURCE_REPOSITORY,
    MELOTTS_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_MELOTTS_ALIASES = (
    "melo",
    "melo-tts",
    "native-melotts",
)


def create_melotts_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="melotts",
        version="1",
        model_builder=("voicehub.models.melotts.native.modeling:"
                       "build_melotts_model"),
        config=("voicehub.models.melotts.native.configuration:"
                "MeloTTSArchitectureConfig"),
        processor=("voicehub.models.melotts.native.frontend:"
                   "NativeMeloTTSFrontend"),
        decoder=("voicehub.models.melotts.source.melo.models:Generator"),
        objective=("voicehub.models.melotts.native.training:"
                   "MeloTTSTrainingModel"),
        checkpoint_adapter=("voicehub.models.melotts.native.checkpoint:"
                            "load_melotts_checkpoint"),
        components={
            "runtime": ("voicehub.models.melotts.native.runtime:"
                        "MeloTTSRuntime"),
            "artifact-resolver": ("voicehub.models.melotts.native.artifacts:"
                                  "resolve_melotts_artifacts"),
            "legacy-importer":
            ("voicehub.models.melotts.native.checkpoint:"
             "convert_legacy_melotts_checkpoint"),
            "exporter": ("voicehub.models.melotts.native.checkpoint:"
                         "export_melotts_checkpoint"),
            "waveform-discriminator":
            ("voicehub.models.melotts.source.melo.models:"
             "MultiPeriodDiscriminator"),
            "duration-discriminator": ("voicehub.models.melotts.source.melo.models:"
                                       "DurationDiscriminator"),
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
            optimization_passes=("compile", "custom-kernels"),
            features=(
                "vits-family",
                "vits-wavenet-gate",
                "multilingual-vits2",
                "multi-speaker",
                "explicit-phone-tone-language-input",
                "explicit-bert-feature-input",
                "preprocessed-finetuning",
                "native-hifigan",
                "training-only-mpd",
                "training-only-duration-discriminator",
                "legacy-checkpoint-conversion",
            ),
        ),
        upstream_revision=MELOTTS_SOURCE_REVISION,
        license_id=MELOTTS_SOURCE_LICENSE,
        metadata={
            "vits_architecture_kind":
            "vits2",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            MELOTTS_SOURCE_REPOSITORY,
            "source_revision":
            MELOTTS_SOURCE_REVISION,
            "checkpoint_family": ("MeloTTS EN/EN_V2/EN_NEWEST/FR/JP/ES/ZH/KR"),
            "training_boundary": (
                "Caller supplies checkpoint-compatible phone, tone, "
                "language, 1024-channel BERT, 768-channel Japanese-BERT, "
                "linear spectrogram, speaker, and waveform features."),
            "raw_text_frontend":
            "unsupported-without-exact-upstream-features",
            "full_finetuning_ready":
            True,
        },
    )


def register_melotts_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_MELOTTS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_melotts_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_MELOTTS_ALIASES",
    "create_melotts_architecture_spec",
    "register_melotts_architecture",
]
