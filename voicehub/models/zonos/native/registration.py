"""Lazy declaration for the VoiceHub-native Zonos v0.1 Transformer."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.zonos.native.metadata import (
    NATIVE_ZONOS_FORMAT,
    ZONOS_DAC_REPOSITORY,
    ZONOS_DAC_REVISION,
    ZONOS_HYBRID_REPOSITORY,
    ZONOS_SOURCE,
    ZONOS_SOURCE_LICENSE,
    ZONOS_SOURCE_REVISION,
    ZONOS_TRANSFORMER_HEADER_FINGERPRINT,
    ZONOS_TRANSFORMER_LICENSE,
    ZONOS_TRANSFORMER_PARAMETER_COUNT,
    ZONOS_TRANSFORMER_REPOSITORY,
    ZONOS_TRANSFORMER_REVISION,
    ZONOS_TRANSFORMER_TENSOR_COUNT,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_ZONOS_ALIASES = (
    "native-zonos",
    "zonos-v0.1",
    "zonos-v0.1-transformer",
)


def create_zonos_architecture_spec() -> ArchitectureSpec:
    """Describe the audited dense graph without importing PyTorch."""
    return ArchitectureSpec(
        architecture_id="zonos",
        version="1",
        model_builder=("voicehub.models.zonos.native.modeling:ZonosForCausalLM"),
        config=("voicehub.models.zonos.native.configuration:"
                "ZonosArchitectureConfig"),
        processor=("voicehub.models.zonos.native.frontend:tokenize_phonemes"),
        decoder="voicehub.models.zonos.native.codec:ZonosDACCodec",
        objective=("voicehub.models.zonos.training:ZonosTrainingAdapter"),
        checkpoint_adapter=("voicehub.models.zonos.native.checkpoint:"
                            "load_zonos_checkpoint"),
        components={
            "artifact-resolver": ("voicehub.models.zonos.native.artifacts:"
                                  "resolve_zonos_artifacts"),
            "checkpoint-exporter": ("voicehub.models.zonos.native.checkpoint:"
                                    "export_zonos_checkpoint"),
            "conditioning": ("voicehub.models.zonos.native.frontend:"
                             "make_condition_dict"),
            "runtime": ("voicehub.models.zonos.native.runtime:"
                        "NativeZonosRuntime"),
            "sampler": ("voicehub.models.zonos.native.sampling:"
                        "generate_zonos_codes"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=False,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=("compile", "sdpa"),
            features=(
                "llm-tts-codec",
                "dense-transformer",
                "delayed-codebook-language-model",
                "frozen-native-descript-dac",
                "multilingual-phoneme-conditioning",
                "precomputed-phoneme-boundary",
                "precomputed-speaker-embedding",
                "raw-audio-fine-tuning",
                "safetensors-export",
                "strict-checkpoint-validation",
                "voice-cloning",
                "no-external-model-runtime",
            ),
        ),
        upstream_revision=ZONOS_SOURCE_REVISION,
        license_id=ZONOS_SOURCE_LICENSE,
        metadata={
            "external_llm_backend_blocker": ("Zonos requires multi-codebook CFG generation."),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            ZONOS_SOURCE,
            "source_revision":
            ZONOS_SOURCE_REVISION,
            "reference_checkpoint":
            ZONOS_TRANSFORMER_REPOSITORY,
            "reference_checkpoint_revision":
            ZONOS_TRANSFORMER_REVISION,
            "reference_checkpoint_license":
            ZONOS_TRANSFORMER_LICENSE,
            "reference_tensor_count":
            ZONOS_TRANSFORMER_TENSOR_COUNT,
            "reference_parameter_count":
            ZONOS_TRANSFORMER_PARAMETER_COUNT,
            "reference_header_fingerprint":
            ZONOS_TRANSFORMER_HEADER_FINGERPRINT,
            "native_checkpoint_format":
            NATIVE_ZONOS_FORMAT,
            "codec_checkpoint":
            ZONOS_DAC_REPOSITORY,
            "codec_checkpoint_revision":
            ZONOS_DAC_REVISION,
            "hybrid_checkpoint":
            ZONOS_HYBRID_REPOSITORY,
            "hybrid_support":
            False,
            "hybrid_boundary": (
                "The hybrid checkpoint contains a distinct Mamba-2 graph. "
                "VoiceHub rejects it rather than claiming Transformer "
                "checkpoint compatibility."),
            "text_frontend_boundary": (
                "The checkpoint consumes eSpeak-compatible phonemes. "
                "VoiceHub accepts precomputed phonemes or an injected "
                "frontend and does not depend on an external G2P runtime."),
            "training_objective":
            "reconstructed-delayed-codebook-causal-cross-entropy",
            "training_objective_author_verified":
            False,
            "full_finetuning_ready":
            True,
            "inference_reloadable_export":
            True,
        },
    )


def register_zonos_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_ZONOS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_zonos_architecture_spec()
    target.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )
    return spec


__all__ = [
    "DEFAULT_ZONOS_ALIASES",
    "create_zonos_architecture_spec",
    "register_zonos_architecture",
]
