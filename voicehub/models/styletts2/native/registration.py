"""Lazy declaration for VoiceHub's native StyleTTS 2 architecture."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.styletts2.native.metadata import (
    STYLETTS2_SOURCE_LICENSE,
    STYLETTS2_SOURCE_REPOSITORY,
    STYLETTS2_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_STYLETTS2_ALIASES = (
    "native-styletts2",
    "style-tts2",
    "styletts2-libritts",
)


def create_styletts2_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="styletts2",
        version="1",
        model_builder=("voicehub.models.styletts2.native.modeling:"
                       "build_styletts2_model"),
        config=("voicehub.models.styletts2.native.configuration:"
                "StyleTTS2ArchitectureConfig"),
        processor=("voicehub.models.styletts2.native.frontend:"
                   "NativeStyleTTS2Frontend"),
        decoder=("voicehub.models.styletts2.source.styletts2.Modules.hifigan:"
                 "Decoder"),
        objective=("voicehub.models.styletts2.native.training:"
                   "StyleTTS2TrainingModel"),
        checkpoint_adapter=("voicehub.models.styletts2.native.checkpoint:"
                            "load_styletts2_checkpoint"),
        components={
            "runtime": ("voicehub.models.styletts2.native.runtime:"
                        "StyleTTS2Runtime"),
            "legacy-importer":
            ("voicehub.models.styletts2.native.checkpoint:"
             "convert_legacy_styletts2_checkpoint"),
            "exporter": ("voicehub.models.styletts2.native.checkpoint:"
                         "export_styletts2_checkpoint"),
            "mel-frontend": ("voicehub.models.styletts2.native.frontend:"
                             "StyleTTS2MelSpectrogram"),
            "istftnet-decoder": ("voicehub.models.styletts2.source.styletts2.Modules."
                                 "istftnet:Decoder"),
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
                "diffusion-cache",
                "diffusion-sampling",
            ),
            features=(
                "diffusion-family",
                "diffusion-serving-native",
                "diffusion-kind-style-diffusion",
                "diffusion-operation-denoiser",
                "diffusion-operation-classifier-free-guidance",
                "diffusion-operation-adpm2-solver",
                "diffusion-sampling-schedule",
                "multispeaker-style-diffusion",
                "voice-cloning",
                "multispeaker-reference-style-required",
                "native-plbert",
                "native-hifigan",
                "native-istftnet",
                "preprocessed-teacher-forced-finetuning",
                "training-only-mpd-msd",
                "explicit-phoneme-input",
                "legacy-checkpoint-conversion",
            ),
        ),
        upstream_revision=STYLETTS2_SOURCE_REVISION,
        license_id=STYLETTS2_SOURCE_LICENSE,
        metadata={
            "diffusion_architecture_kind":
            "style-diffusion",
            "diffusion_operations": (
                "denoiser",
                "classifier-free-guidance",
                "adpm2-solver",
            ),
            "diffusion_sampling_capabilities": ("schedule", ),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            STYLETTS2_SOURCE_REPOSITORY,
            "source_revision":
            STYLETTS2_SOURCE_REVISION,
            "checkpoint_family": ("StyleTTS2 single-speaker iSTFTNet and "
                                  "LibriTTS/finetune HiFi-GAN"),
            "training_boundary": (
                "Caller supplies phoneme IDs, monotonic alignments, "
                "normalized mels, reference mels, F0/noise targets, and "
                "waveforms. Raw G2P, alignment, F0 extraction, and WavLM "
                "objectives are intentionally not inferred."),
            "full_finetuning_ready":
            True,
            "raw_text_frontend":
            "explicit-phonemes-only",
        },
    )


def register_styletts2_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_STYLETTS2_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_styletts2_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_STYLETTS2_ALIASES",
    "create_styletts2_architecture_spec",
    "register_styletts2_architecture",
]
