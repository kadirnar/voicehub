"""Lazy architecture declaration for VoiceHub's native Silero VAD."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

OFFICIAL_SILERO_VAD_VERSION = "6.2.1"
OFFICIAL_SILERO_VAD_REVISION = ("7e30209a3e901f9842f81b225f3e93d8199902b1")
DEFAULT_SILERO_VAD_ALIASES = (
    "native-silero-vad",
    "silero",
    "silero-v6",
)


def create_silero_vad_architecture_spec() -> ArchitectureSpec:
    """Create the immutable, entirely lazy Silero VAD declaration."""
    return ArchitectureSpec(
        architecture_id="silero-vad",
        version="1",
        model_builder=("voicehub.models.vad_silero.native.modeling:SileroVADModel"),
        config=("voicehub.models.vad_silero.native.configuration:SileroVADConfig"),
        decoder=("voicehub.models.vad_silero.native.segmentation:"
                 "SileroVADSegmenter"),
        objective=("voicehub.models.vad_silero.native.objective:"
                   "SileroVADBinaryCrossEntropyLoss"),
        checkpoint_adapter=(
            "voicehub.models.vad_silero.native.checkpoint:"
            "OfficialSileroVADSafeTensorsCheckpointAdapter"),
        components={
            "stream": ("voicehub.models.vad_silero.native.modeling:"
                       "SileroVADStream"),
            "torchscript-checkpoint-adapter":
            ("voicehub.models.vad_silero.native.checkpoint:"
             "OfficialSileroVADTorchScriptCheckpointAdapter"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.VOICE_ACTIVITY_DETECTION, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", ),
            checkpoint_formats=(
                "safetensors",
                "torchscript-state-dict",
            ),
            training=True,
            streaming=True,
            batched_inference=True,
            distributed_training=False,
            features=(
                "8khz",
                "16khz",
                "frame-probabilities",
                "explicit-stream-state",
                "hysteresis-segmentation",
                "max-duration-segmentation",
                "decoder-fine-tuning",
                "checkpoint-conversion",
            ),
        ),
        upstream_revision=OFFICIAL_SILERO_VAD_REVISION,
        license_id="MIT",
        metadata={
            "family":
            "silero-vad",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "upstream_version":
            OFFICIAL_SILERO_VAD_VERSION,
            "upstream_source":
            ("https://github.com/snakers4/silero-vad/tree/"
             f"{OFFICIAL_SILERO_VAD_REVISION}"),
            "reference_safetensors":
            "src/silero_vad/data/silero_vad_16k.safetensors",
            "reference_safetensors_header_fingerprint":
            "1abaf3b9cfbf3990230392263d17d18ccd63e63471a965289da55335f09a7af8",
            "safetensors_sample_rates": (16_000, ),
            "torchscript_state_dict_sample_rates": (8_000, 16_000),
            "graph_source": (
                "https://github.com/snakers4/silero-vad/blob/"
                f"{OFFICIAL_SILERO_VAD_REVISION}/src/silero_vad/"
                "tinygrad_model.py"),
            "segmentation_source": (
                "https://github.com/snakers4/silero-vad/blob/"
                f"{OFFICIAL_SILERO_VAD_REVISION}/src/silero_vad/utils_vad.py"),
            "training_source":
            ("https://github.com/snakers4/silero-vad/tree/"
             f"{OFFICIAL_SILERO_VAD_REVISION}/tuning"),
            "training_scope": (
                "The released supervised recipe tunes the LSTM and final "
                "convolution with frame-level binary cross entropy. The "
                "native graph is differentiable through the convolutional "
                "encoder; its Fourier basis remains fixed."),
        },
    )


def register_silero_vad_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_SILERO_VAD_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register the lazy Silero VAD declaration and return it."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_silero_vad_architecture_spec()
    target.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )
    return spec


__all__ = [
    "DEFAULT_SILERO_VAD_ALIASES",
    "OFFICIAL_SILERO_VAD_REVISION",
    "OFFICIAL_SILERO_VAD_VERSION",
    "create_silero_vad_architecture_spec",
    "register_silero_vad_architecture",
]
