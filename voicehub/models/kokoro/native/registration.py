"""Lazy declaration for VoiceHub's native Kokoro architecture."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.kokoro.native.metadata import KOKORO_CHECKPOINT_REVISION, KOKORO_SOURCE_REVISION
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_KOKORO_ALIASES = (
    "native-kokoro",
    "kokoro-82m",
    "styletts2-kokoro",
)


def create_kokoro_architecture_spec() -> ArchitectureSpec:
    """Create the immutable native Kokoro declaration."""
    return ArchitectureSpec(
        architecture_id="kokoro",
        version="1",
        model_builder="voicehub.models.kokoro.model:KModel",
        config=("voicehub.models.kokoro.native.configuration:"
                "KokoroArchitectureConfig"),
        processor=("voicehub.models.kokoro.pipeline:GraphemeFallbackFrontend"),
        decoder="voicehub.models.kokoro.istftnet:Decoder",
        objective=("voicehub.models.kokoro.training:"
                   "KokoroPreprocessedTrainingModel"),
        checkpoint_adapter=("voicehub.models.kokoro.native.checkpoint:"
                            "load_native_kokoro_checkpoint"),
        components={
            "albert": ("voicehub.models.kokoro.native.albert:"
                       "KokoroAlbertModel"),
            "phoneme-frontend": ("voicehub.models.kokoro.pipeline:PhonemeFrontend"),
            "legacy-importer":
            ("voicehub.models.kokoro.native.checkpoint:"
             "import_legacy_kokoro_checkpoint"),
            "voice-pack-loader": ("voicehub.models.kokoro.native.checkpoint:"
                                  "load_native_kokoro_voice"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=False,
            distributed_training=True,
            optimization_passes=("compile", ),
            features=(
                "phoneme-input",
                "plbert",
                "prosody-predictor",
                "istftnet",
                "voice-mixing",
                "checkpoint-conversion",
                "preprocessed-decoder-fine-tuning",
                "raw-text-g2p-is-not-source-equivalent",
                "full-author-recipe-unavailable",
            ),
        ),
        upstream_revision=KOKORO_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source": ("https://github.com/hexgrad/kokoro/tree/"
                       f"{KOKORO_SOURCE_REVISION}"),
            "reference_checkpoint":
            "hexgrad/Kokoro-82M",
            "reference_checkpoint_revision":
            KOKORO_CHECKPOINT_REVISION,
            "checkpoint_license":
            "Apache-2.0",
            "training_boundary": (
                "Released PL-BERT/prosody/text/iSTFTNet graph with "
                "precomputed phonemes, style, alignment/duration, F0, "
                "energy, and waveform targets."),
            "full_finetuning_ready":
            False,
        },
    )


def register_kokoro_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_KOKORO_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register Kokoro without importing its graph."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_kokoro_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_KOKORO_ALIASES",
    "KOKORO_SOURCE_REVISION",
    "create_kokoro_architecture_spec",
    "register_kokoro_architecture",
]
