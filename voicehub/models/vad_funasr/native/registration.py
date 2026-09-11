"""Lazy native architecture declaration for FunASR FSMN VAD."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.vad_funasr.native.metadata import FUNASR_SOURCE_REVISION
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_FSMN_VAD_ALIASES = (
    "native-fsmn-vad",
    "funasr-fsmn",
    "funasr-vad",
)


def create_fsmn_vad_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="fsmn-vad",
        version="1",
        model_builder=("voicehub.models.vad_funasr.native.modeling:FSMNVADModel"),
        config=("voicehub.models.vad_funasr.native.configuration:FSMNVADConfig"),
        decoder=("voicehub.models.vad_funasr.native.inference:FSMNVADDecoder"),
        objective=("voicehub.models.vad_funasr.native.objective:fsmn_vad_loss"),
        checkpoint_adapter=(
            "voicehub.models.vad_funasr.native.checkpoint:"
            "FSMNVADSafeTensorsCheckpointAdapter"),
        components={
            "frontend":
            "voicehub.models.vad_funasr.native.frontend:FSMNVADFrontend",
            "pickle-converter":
            ("voicehub.models.vad_funasr.native.checkpoint:"
             "convert_funasr_fsmn_checkpoint"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.VOICE_ACTIVITY_DETECTION, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", ),
            checkpoint_formats=(
                "safetensors",
                "trusted-pickle-conversion",
            ),
            training=True,
            streaming=True,
            batched_inference=True,
            features=(
                "kaldi-fbank",
                "lfr",
                "cmvn",
                "fsmn",
                "frame-scores",
                "pdf-cross-entropy",
                "binary-vad-fine-tuning",
                "endpoint-state-machine",
            ),
        ),
        upstream_revision=FUNASR_SOURCE_REVISION,
        license_id="MIT",
        metadata={
            "family":
            "fsmn-vad",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "checkpoint_license":
            "Apache-2.0",
            "training_boundary": (
                "The public inference artifact does not publish its original "
                "data recipe or loss implementation. VoiceHub supports exact "
                "248-PDF cross-entropy and grouped speech/silence NLL."),
        },
    )


def register_fsmn_vad_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_FSMN_VAD_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_fsmn_vad_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_FSMN_VAD_ALIASES",
    "create_fsmn_vad_architecture_spec",
    "register_fsmn_vad_architecture",
]
