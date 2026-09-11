"""Lazy declaration for VoiceHub-native Supertonic 3."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.supertonic.native.metadata import (
    SUPERTONIC_CHECKPOINT_LICENSE,
    SUPERTONIC_CHECKPOINT_REPOSITORY,
    SUPERTONIC_CHECKPOINT_REVISION,
    SUPERTONIC_SOURCE_LICENSE,
    SUPERTONIC_SOURCE_REPOSITORY,
    SUPERTONIC_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_SUPERTONIC_ALIASES = (
    "native-supertonic",
    "supertonic-3",
)


def create_supertonic_architecture_spec() -> ArchitectureSpec:
    """Create the immutable reviewed-graph architecture declaration."""
    return ArchitectureSpec(
        architecture_id="supertonic",
        version="1.7.3",
        model_builder=("voicehub.models.supertonic.native.runtime:"
                       "NativeSupertonicRuntime"),
        config=("voicehub.models.supertonic.native.configuration:"
                "SupertonicArchitectureConfig"),
        processor=("voicehub.models.supertonic.native.frontend:"
                   "SupertonicUnicodeProcessor"),
        decoder=("voicehub.models.supertonic.native.runtime:"
                 "NativeSupertonicRuntime.synthesize"),
        objective=("voicehub.models.supertonic.native.runtime:"
                   "NativeSupertonicRuntime.fine_tuning_loss"),
        checkpoint_adapter=("voicehub.models.supertonic.native.checkpoint:"
                            "load_supertonic_native_weights"),
        components={
            "artifact-resolver":
            ("voicehub.models.supertonic.native.artifacts:"
             "resolve_supertonic_artifacts"),
            "onnx-graph-runtime": ("voicehub.neural.onnx:NativeONNXGraph"),
            "style-loader": ("voicehub.models.supertonic.native.frontend:"
                             "SupertonicStyle"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("onnx", "safetensors"),
            training=True,
            streaming=False,
            batched_inference=True,
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
                "diffusion-kind-flow-matching",
                "diffusion-operation-denoiser",
                "diffusion-operation-iterative-estimator",
                "diffusion-sampling-discrete-step-count",
                "multilingual-unicode-frontend",
                "reviewed-onnx-import",
                "pytorch-native-execution",
                "flow-matching",
                "duration-prediction",
                "native-vocoder",
                "precomputed-latent-fine-tuning",
                "no-external-runtime",
            ),
        ),
        upstream_revision=SUPERTONIC_SOURCE_REVISION,
        license_id=SUPERTONIC_SOURCE_LICENSE,
        metadata={
            "diffusion_architecture_kind": "flow-matching",
            "diffusion_operations": (
                "denoiser",
                "iterative-estimator",
            ),
            "diffusion_sampling_capabilities": ("discrete-step-count", ),
            "family": "flow-matching-tts",
            "implementation": "voicehub-native",
            "tensor_backend": "pytorch",
            "source": SUPERTONIC_SOURCE_REPOSITORY,
            "checkpoint": SUPERTONIC_CHECKPOINT_REPOSITORY,
            "checkpoint_revision": SUPERTONIC_CHECKPOINT_REVISION,
            "checkpoint_license": SUPERTONIC_CHECKPOINT_LICENSE,
            "training_scope": "published-inference-graph-with-precomputed-latents",
            "full_raw_audio_finetuning_ready": False,
            "author_training_recipe_published": False,
            "parity_reference": "onnxruntime-cpu",
            "measured_max_absolute_error": {
                "duration_predictor": 5.9604645e-08,
                "text_encoder": 2.682209e-05,
                "vector_estimator": 3.8594e-06,
                "vocoder": 2.413988e-06,
            },
        },
    )


def register_supertonic_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_SUPERTONIC_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register Supertonic without loading graph tensors or PyTorch."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_supertonic_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_SUPERTONIC_ALIASES",
    "create_supertonic_architecture_spec",
    "register_supertonic_architecture",
]
