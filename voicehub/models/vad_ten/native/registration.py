"""Lazy architecture declaration for VoiceHub-native TEN VAD."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.vad_ten.native.metadata import (
    KALDI_NATIVE_FBANK_VERSION,
    SHERPA_ONNX_REVISION,
    TEN_VAD_ONNX_SHA256,
    TEN_VAD_REVISION,
    TEN_VAD_SOURCE_LICENSE,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_TEN_VAD_ALIASES = (
    "tenvad",
    "ten-vad-native",
    "sherpa-onnx-vad",
)


def create_ten_vad_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="ten-vad",
        version="1",
        model_builder="voicehub.models.vad_ten.native.modeling:TENVADModel",
        config="voicehub.models.vad_ten.native.configuration:TENVADConfig",
        objective=("voicehub.models.vad_ten.native.objective:"
                   "ten_vad_binary_cross_entropy"),
        checkpoint_adapter=(
            "voicehub.models.vad_ten.native.checkpoint:"
            "TENVADSafeTensorsCheckpointAdapter"),
        components={
            "frontend": "voicehub.models.vad_ten.native.frontend:TENVADFrontend",
            "onnx-converter":
            ("voicehub.models.vad_ten.native.checkpoint:"
             "convert_ten_vad_onnx_checkpoint"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.VOICE_ACTIVITY_DETECTION, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", ),
            checkpoint_formats=(
                "safetensors",
                "explicit-onnx-weight-conversion",
            ),
            training=True,
            streaming=True,
            batched_inference=True,
            features=(
                "sherpa-compatible-log-mel-frontend",
                "three-frame-context",
                "separable-convolutions",
                "two-layer-streaming-lstm",
                "explicit-four-state-streaming",
                "raw-audio-fine-tuning",
                "masked-binary-cross-entropy",
                "safe-onnx-weight-conversion",
            ),
        ),
        upstream_revision=TEN_VAD_REVISION,
        license_id=TEN_VAD_SOURCE_LICENSE,
        metadata={
            "family":
            "ten-vad",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source_onnx_sha256":
            TEN_VAD_ONNX_SHA256,
            "sherpa_compatibility_revision":
            SHERPA_ONNX_REVISION,
            "kaldi_native_fbank_reference":
            KALDI_NATIVE_FBANK_VERSION,
            "training_boundary": (
                "TEN does not publish the training recipe for this graph. "
                "VoiceHub fine-tunes the exact released graph with a "
                "reconstructed, masked window-level BCE objective and does "
                "not claim reproduction of the unpublished source recipe."),
            "license_notice": (
                "TEN VAD is subject to the upstream non-standard Open Source "
                "License and its additional deployment restrictions. Review "
                "THIRD_PARTY_LICENSE before conversion or deployment."),
        },
    )


def register_ten_vad_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_TEN_VAD_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_ten_vad_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_TEN_VAD_ALIASES",
    "create_ten_vad_architecture_spec",
    "register_ten_vad_architecture",
]
