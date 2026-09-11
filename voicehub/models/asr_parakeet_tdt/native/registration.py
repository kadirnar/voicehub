"""Lazy architecture declaration for native Parakeet TDT."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_parakeet_tdt.native.metadata import PARAKEET_TDT_CHECKPOINTS, PARAKEET_TRANSFORMERS_REVISION
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_PARAKEET_TDT_ALIASES = (
    "native-parakeet-tdt",
    "nvidia-parakeet-tdt",
)


def create_parakeet_tdt_architecture_spec() -> ArchitectureSpec:
    checkpoint = PARAKEET_TDT_CHECKPOINTS["nvidia/parakeet-tdt-0.6b-v3"]
    return ArchitectureSpec(
        architecture_id="parakeet-tdt",
        version="1",
        model_builder=("voicehub.models.asr_parakeet_tdt.native.modeling:ParakeetForTDT"),
        config=("voicehub.models.asr_parakeet_tdt.native.configuration:"
                "ParakeetTDTConfig"),
        objective="voicehub.models.asr_parakeet_tdt.native.loss:tdt_loss",
        checkpoint_adapter=(
            "voicehub.models.asr_parakeet_tdt.native.checkpoint:"
            "ParakeetTDTCheckpointAdapter"),
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            features=(
                "fastconformer",
                "relative-position-attention",
                "token-duration-transducer",
                "greedy-duration-decoding",
                "word-timestamps",
                "gradient-checkpointing",
                "portable-export",
            ),
        ),
        upstream_revision=PARAKEET_TRANSFORMERS_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family": "parakeet-tdt",
            "implementation": "voicehub-native",
            "tensor_backend": "pytorch",
            "reference_checkpoint": "nvidia/parakeet-tdt-0.6b-v3",
            "reference_checkpoint_revision": checkpoint["revision"],
            "reference_checkpoint_license": checkpoint["license"],
            "reference_checkpoint_header_fingerprint": (checkpoint["header_fingerprint"]),
            "decoding": "greedy-only",
            "language_mode": "automatic",
        },
    )


def register_parakeet_tdt_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_PARAKEET_TDT_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be ArchitectureRegistry or None.")
    spec = create_parakeet_tdt_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_PARAKEET_TDT_ALIASES",
    "create_parakeet_tdt_architecture_spec",
    "register_parakeet_tdt_architecture",
]
