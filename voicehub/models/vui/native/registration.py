"""Lazy declaration for the native Vui 100M and Fluac graph."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_VUI_ALIASES = (
    "native-vui",
    "vui-100m",
    "vui-fluac",
)

_SOURCE_REVISION = "8656f9f175161cb19a7f6c1ff4374c9b56739b4a"
_CHECKPOINT_REVISION = "8dc2bd9993a8118b6e2b71f3d9d92d1deb80e5f7"


def create_vui_architecture_spec() -> ArchitectureSpec:
    """Create Vui's immutable native-runtime declaration."""
    return ArchitectureSpec(
        architecture_id="vui",
        version="1",
        model_builder="voicehub.models.vui.model:Vui",
        config="voicehub.models.vui.config:Config",
        processor="voicehub.models.vui.tok:CustomByT5Tokenizer",
        decoder="voicehub.models.vui.fluac:Fluac",
        objective=("voicehub.models.vui.training:VuiTrainingAdapter."
                   "execute_training_phase"),
        checkpoint_adapter=("voicehub.models.vui.model:Vui.from_pretrained"),
        components={
            "artifact-resolver": ("voicehub.models.vui.artifacts:resolve_vui_artifacts"),
            "delayed-codebook-pattern": ("voicehub.models.vui.patterns:DelayedPatternProvider"),
            "voice-activity-detector": ("voicehub.models.vui.vad:detect_voice_activity"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("pytorch", "safetensors"),
            training=True,
            streaming=False,
            batched_inference=False,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=("compile", ),
            features=(
                "llm-tts-codec",
                "byte-tokenizer",
                "delayed-codebooks",
                "fluac-codec",
                "voice-prompting",
                "native-vad",
                "standalone-safetensors-export",
                "no-external-runtime",
                "torch-compile-inference-unsafe",
            ),
        ),
        upstream_revision=_SOURCE_REVISION,
        license_id="MIT",
        metadata={
            "external_llm_backend_blocker":
            ("Vui samples multiple codebooks through architecture-specific "
             "heads."),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            "https://github.com/fluxions-ai/vui",
            "checkpoint":
            "fluxions/vui",
            "checkpoint_revision":
            _CHECKPOINT_REVISION,
            "family_boundary": (
                "Pinned Fluac-based Vui 100M family. The later Vui Nano "
                "release is a distinct architecture and is not treated as "
                "checkpoint-compatible."),
            "training_boundary": (
                "Architecture-consistent delayed-codebook cross-entropy; "
                "the author dataset and optimization recipe are unpublished."),
            "compile_boundary": (
                "Inference torch.compile is rejected because real-checkpoint "
                "tests changed generated sequences and audio with both "
                "dynamic=True and compiler-default specialization. Training "
                "compilation remains available for decoder.forward."),
        },
    )


def register_vui_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_VUI_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register Vui without importing PyTorch or allocating the graph."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_vui_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_VUI_ALIASES",
    "create_vui_architecture_spec",
    "register_vui_architecture",
]
