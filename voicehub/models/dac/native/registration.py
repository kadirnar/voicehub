"""Lazy declaration for VoiceHub's native Descript DAC architecture."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.dac.native.metadata import (
    DESCRIPT_DAC_44KHZ_HEADER_FINGERPRINT,
    DESCRIPT_DAC_44KHZ_REVISION,
    TRANSFORMERS_DAC_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_DAC_ALIASES = (
    "descript-dac",
    "native-dac",
)


def create_dac_architecture_spec() -> ArchitectureSpec:
    """Create the immutable native DAC declaration."""
    return ArchitectureSpec(
        architecture_id="dac",
        version="1",
        model_builder="voicehub.models.dac.native.modeling:DacModel",
        config="voicehub.models.dac.native.configuration:DacConfig",
        objective=("voicehub.components.audio.codecs.dac.nn.loss:"
                   "MultiScaleSTFTLoss"),
        checkpoint_adapter=("voicehub.models.dac.native.checkpoint:"
                            "HuggingFaceDacCheckpointAdapter"),
        components={
            "discriminator": ("voicehub.components.audio.codecs.dac.model.discriminator:"
                              "Discriminator"),
            "gan-objective": ("voicehub.components.audio.codecs.dac.nn.loss:GANLoss"),
            "mel-objective": ("voicehub.components.audio.codecs.dac.nn.loss:"
                              "MelSpectrogramLoss"),
            "quantizer": ("voicehub.components.audio.codecs.dac.nn.quantize:"
                          "ResidualVectorQuantize"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, SpeechTask.AUDIO_CODEC),
            devices=("cpu", "cuda"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            optimization_passes=("compile", ),
            features=(
                "audio-codec",
                "residual-vector-quantization",
                "multi-period-discriminator",
                "multi-resolution-discriminator",
                "native-dsp",
                "checkpoint-conversion",
            ),
        ),
        upstream_revision=DESCRIPT_DAC_44KHZ_REVISION,
        license_id="MIT",
        metadata={
            "family":
            "descript-dac",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "reference_checkpoint":
            "descript/dac_44khz",
            "reference_checkpoint_revision":
            DESCRIPT_DAC_44KHZ_REVISION,
            "reference_safetensors_header_fingerprint": (DESCRIPT_DAC_44KHZ_HEADER_FINGERPRINT),
            "reference_checkpoint_source":
            ("https://huggingface.co/descript/dac_44khz/tree/"
             f"{DESCRIPT_DAC_44KHZ_REVISION}"),
            "transformers_reference_revision":
            TRANSFORMERS_DAC_REVISION,
            "transformers_source": (
                "https://github.com/huggingface/transformers/tree/"
                f"{TRANSFORMERS_DAC_REVISION}/src/transformers/models/dac"),
            "original_source": ("https://github.com/descriptinc/descript-audio-codec"),
        },
    )


def register_dac_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_DAC_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_dac_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_DAC_ALIASES",
    "create_dac_architecture_spec",
    "register_dac_architecture",
]
