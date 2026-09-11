"""Lazy architecture declaration for native NeMo QuartzNet CTC."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.asr_nemo.native.metadata import NEMO_SOURCE_REVISION, QUARTZNET_SHA256, QUARTZNET_VERSION
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_NEMO_CTC_ALIASES = (
    "nemo-asr",
    "nemo-ctc",
    "native-quartznet",
    "quartznet15x5",
)


def create_nemo_ctc_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="nemo-quartznet-ctc",
        version="1",
        model_builder=("voicehub.models.asr_nemo.native.modeling:"
                       "NeMoQuartzNetForCTC"),
        config=("voicehub.models.asr_nemo.native.configuration:"
                "NeMoQuartzNetCTCConfig"),
        processor=("voicehub.models.asr_nemo.native.tokenization:"
                   "NeMoCharacterTokenizer"),
        objective="voicehub.objectives.ctc:CTCLoss",
        checkpoint_adapter=(
            "voicehub.models.asr_nemo.native.checkpoint:"
            "NeMoCTCSafeTensorsCheckpointAdapter"),
        components={
            "frontend": ("voicehub.models.asr_nemo.native.frontend:"
                         "NeMoFilterbankFeatures"),
            "nemo-converter":
            ("voicehub.models.asr_nemo.native.checkpoint:"
             "convert_nemo_quartznet_checkpoint"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=(
                "safetensors",
                "verified-nemo-conversion",
            ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            features=(
                "ctc",
                "character-tokenization",
                "log-mel",
                "depthwise-separable-convolution",
                "raw-audio-fine-tuning",
                "word-timestamps",
            ),
        ),
        upstream_revision=NEMO_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family":
            "quartznet",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "reference_checkpoint":
            "nvidia/nemo/stt_en_quartznet15x5",
            "reference_checkpoint_version":
            QUARTZNET_VERSION,
            "reference_checkpoint_sha256":
            QUARTZNET_SHA256,
            "checkpoint_terms":
            "NVIDIA NGC Terms of Use",
            "verified_scope": (
                "QuartzNet/Jasper character-CTC only. Parakeet TDT, RNN-T, "
                "Canary, Citrinet, Conformer, and FastConformer are distinct "
                "graphs and are rejected by this runtime."),
        },
    )


def register_nemo_ctc_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_NEMO_CTC_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_nemo_ctc_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_NEMO_CTC_ALIASES",
    "create_nemo_ctc_architecture_spec",
    "register_nemo_ctc_architecture",
]
