"""Lazy architecture declaration for native SeamlessM4T-v2 S2T."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_seamless_m4t_v2.native.metadata import (
    SEAMLESS_M4T_V2_CHECKPOINTS,
    TRANSFORMERS_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_SEAMLESS_M4T_V2_ALIASES = (
    "native-seamless-m4t-v2",
    "facebook-seamless-m4t-v2",
)


def create_seamless_m4t_v2_architecture_spec() -> ArchitectureSpec:
    checkpoint = SEAMLESS_M4T_V2_CHECKPOINTS["facebook/seamless-m4t-v2-large"]
    return ArchitectureSpec(
        architecture_id="seamless-m4t-v2-s2t",
        version="1",
        model_builder=("voicehub.models.asr_seamless_m4t_v2.native.modeling:"
                       "SeamlessM4Tv2ForSpeechToText"),
        config=("voicehub.models.asr_seamless_m4t_v2.native.configuration:"
                "SeamlessM4Tv2S2TConfig"),
        objective="torch.nn.functional:cross_entropy",
        checkpoint_adapter=(
            "voicehub.models.asr_seamless_m4t_v2.native.checkpoint:"
            "SeamlessM4Tv2S2TCheckpointAdapter"),
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
                "conformer",
                "relative-key-attention",
                "language-conditioned-seq2seq",
                "sentencepiece-bpe",
                "greedy-generation",
                "gradient-checkpointing",
                "portable-s2t-export",
            ),
        ),
        upstream_revision=TRANSFORMERS_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "decoding": "greedy-only",
            "family": "seamless-m4t-v2-s2t",
            "implementation": "voicehub-native",
            "reference_checkpoint": "facebook/seamless-m4t-v2-large",
            "reference_checkpoint_revision": checkpoint["revision"],
            "reference_checkpoint_license": checkpoint["license"],
            "reference_checkpoint_header_fingerprint": (checkpoint["full_header_fingerprint"]),
            "s2t_header_fingerprint": (checkpoint["s2t_header_fingerprint"]),
            "tensor_backend": "pytorch",
        },
    )


def register_seamless_m4t_v2_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_SEAMLESS_M4T_V2_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be ArchitectureRegistry or None.")
    spec = create_seamless_m4t_v2_architecture_spec()
    target.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )
    return spec


__all__ = [
    "DEFAULT_SEAMLESS_M4T_V2_ALIASES",
    "create_seamless_m4t_v2_architecture_spec",
    "register_seamless_m4t_v2_architecture",
]
