"""Lazy native architecture declaration for Echo-TTS."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

ECHO_SOURCE_REVISION = "2ed95fce62d33bf7b56f835fd9ec0f0b6fb9155e"
DEFAULT_ECHO_ALIASES = (
    "echo-flow",
    "echo-tts",
)


def create_echo_architecture_spec() -> ArchitectureSpec:
    """Create the immutable, lazy Echo architecture specification."""
    return ArchitectureSpec(
        architecture_id="echo-dit",
        version="1",
        model_builder="voicehub.models.echo.model:EchoDiT",
        processor="voicehub.models.echo.sampling:tokenizer_encode",
        decoder=("voicehub.models.echo.sampling:"
                 "sample_euler_cfg_independent_guidances"),
        objective=("voicehub.models.echo.training:EchoTrainingAdapter"),
        checkpoint_adapter=("voicehub.models.echo.sampling:load_model_from_hf"),
        components={
            "blockwise-decoder":
            ("voicehub.models.echo.sampling_blockwise:"
             "sample_blockwise_euler_cfg_independent_guidances"),
            "fish-s1-dac":
            "voicehub.models.echo.autoencoder:DAC",
            "pca-loader":
            "voicehub.models.echo.sampling:load_pca_state_from_hf",
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=False,
            distributed_training=True,
            optimization_passes=(
                "compile",
                "custom-kernels",
                "diffusion-cache",
                "diffusion-sampling",
            ),
            features=(
                "diffusion-family",
                "diffusion-serving-native",
                "diffusion-kind-rectified-flow",
                "diffusion-operation-denoiser",
                "diffusion-operation-classifier-free-guidance",
                "diffusion-operation-euler-solver",
                "diffusion-sampling-schedule",
                "diffusion-sampling-guidance",
                "diffusion-sampling-prediction-cache",
                "rectified-flow",
                "voice-cloning",
                "byte-tokenizer",
                "classifier-free-guidance",
                "fused-diffusion-modulation-kernels",
                "native-audio-codec",
                "blockwise-generation",
            ),
        ),
        upstream_revision=ECHO_SOURCE_REVISION,
        license_id="MIT",
        metadata={
            "diffusion_architecture_kind": "rectified-flow",
            "diffusion_operations": (
                "denoiser",
                "classifier-free-guidance",
                "euler-solver",
            ),
            "diffusion_sampling_capabilities": (
                "schedule",
                "guidance",
                "prediction-cache",
            ),
            "implementation": "voicehub-native",
            "tensor_backend": "pytorch",
            "source": ("https://github.com/jordandare/echo-tts/tree/"
                       f"{ECHO_SOURCE_REVISION}"),
            "checkpoint_license": "CC-BY-NC-SA-4.0",
            "training_boundary": "precomputed-fish-codec-latents",
        },
    )


def register_echo_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_ECHO_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register Echo without importing its multi-billion-parameter graph."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_echo_architecture_spec()
    target.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )
    return spec


__all__ = [
    "DEFAULT_ECHO_ALIASES",
    "ECHO_SOURCE_REVISION",
    "create_echo_architecture_spec",
    "register_echo_architecture",
]
