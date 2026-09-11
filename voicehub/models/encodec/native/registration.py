"""Lazy declaration for VoiceHub's native Meta Encodec architecture."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.components.audio.codecs.encodec.metadata import (
    ENCODEC_24KHZ_RELEASE,
    ENCODEC_48KHZ_RELEASE,
    ENCODEC_SOURCE_LICENSE,
    ENCODEC_SOURCE_REPOSITORY,
    ENCODEC_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_ENCODEC_ALIASES = (
    "meta-encodec",
    "native-encodec",
    "encodec-24khz",
    "encodec-48khz",
)


def create_encodec_architecture_spec() -> ArchitectureSpec:
    """Create the immutable declaration for both official Encodec graphs."""
    releases = (ENCODEC_24KHZ_RELEASE, ENCODEC_48KHZ_RELEASE)
    return ArchitectureSpec(
        architecture_id="encodec",
        version="1",
        model_builder=("voicehub.components.audio.codecs.encodec.model:"
                       "EncodecModel.from_config"),
        config=("voicehub.components.audio.codecs.encodec.configuration:"
                "EncodecConfig"),
        checkpoint_adapter=(
            "voicehub.components.audio.codecs.encodec.checkpoint:"
            "load_encodec_safetensors"),
        decoder=("voicehub.components.audio.codecs.encodec.layers:"
                 "SEANetDecoder"),
        components={
            "artifact-resolver":
            ("voicehub.components.audio.codecs.encodec.artifacts:"
             "resolve_encodec_checkpoint"),
            "checkpoint-exporter":
            ("voicehub.components.audio.codecs.encodec.checkpoint:"
             "save_encodec_safetensors"),
            "encoder": ("voicehub.components.audio.codecs.encodec.layers:"
                        "SEANetEncoder"),
            "quantizer": ("voicehub.components.audio.codecs.encodec.quantization:"
                          "ResidualVectorQuantizer"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, SpeechTask.AUDIO_CODEC),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=("compile", ),
            features=(
                "audio-codec",
                "seanet",
                "residual-vector-quantization",
                "bandwidth-conditioned",
                "segmented-overlap-add",
                "straight-through-fine-tuning",
                "checkpoint-conversion",
            ),
        ),
        upstream_revision=ENCODEC_SOURCE_REVISION,
        license_id=ENCODEC_SOURCE_LICENSE,
        metadata={
            "family":
            "meta-encodec",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            ENCODEC_SOURCE_REPOSITORY,
            "native_checkpoint_format":
            "voicehub-encodec-v1",
            "reference_releases":
            tuple({
                "model_name": release.model_name,
                "sample_rate": release.sample_rate,
                "channels": release.channels,
                "tensor_count": release.tensor_count,
                "state_values": release.state_values,
                "inventory_fingerprint": release.inventory_fingerprint,
            } for release in releases),
            "legacy_checkpoint_policy": (
                "Official .th archives require explicit trust and exact "
                "digest, size, namespace, shape, and inventory validation."),
        },
    )


def register_encodec_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_ENCODEC_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register the native Encodec declaration and compatibility aliases."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_encodec_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_ENCODEC_ALIASES",
    "create_encodec_architecture_spec",
    "register_encodec_architecture",
]
