"""Lazy declaration for VoiceHub-native GPT-SoVITS classic S2."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.gptsovits.native.metadata import (
    GPT_SOVITS_LICENSE,
    GPT_SOVITS_REPOSITORY,
    GPT_SOVITS_REVISION,
    GPT_SOVITS_SOURCE_REVISION,
    GPT_SOVITS_VARIANTS,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_GPT_SOVITS_ALIASES = (
    "native-gptsovits",
    "gpt-sovits",
    "gpt-sovits-v1",
    "gpt-sovits-v2",
    "gpt-sovits-v2-pro",
    "gpt-sovits-v2-pro-plus",
)


def create_gptsovits_architecture_spec() -> ArchitectureSpec:
    """Describe the audited staged graph without allocating it."""
    return ArchitectureSpec(
        architecture_id="gptsovits",
        version="2",
        model_builder=("voicehub.models.gptsovits.native.training:"
                       "build_staged_training_model"),
        config=("voicehub.models.gptsovits.native.configuration:"
                "GPTSoVITSS2Config"),
        processor=("voicehub.models.gptsovits.native.frontend:"
                   "validate_prepared_inference"),
        decoder=("voicehub.models.gptsovits.native.modeling:"
                 "GPTSoVITSSynthesizer"),
        objective=("voicehub.models.gptsovits.native.training:"
                   "GPTSoVITSStagedTrainingModel"),
        checkpoint_adapter=("voicehub.models.gptsovits.native.checkpoint:"
                            "load_gptsovits_checkpoints"),
        components={
            "artifact-resolver":
            ("voicehub.models.gptsovits.native.checkpoint:"
             "resolve_gptsovits_artifacts"),
            "checkpoint-converter":
            ("voicehub.models.gptsovits.native.checkpoint:"
             "convert_gptsovits_legacy_checkpoints"),
            "checkpoint-exporter":
            ("voicehub.models.gptsovits.native.checkpoint:"
             "export_gptsovits_checkpoint"),
            "inference-runtime": ("voicehub.models.gptsovits.native.runtime:"
                                  "GPTSoVITSRuntime"),
            "s1-semantic-model": ("voicehub.models.gptsovits.native.semantic:"
                                  "GPTSoVITSSemanticModel"),
            "s2-discriminator": ("voicehub.models.gptsovits.native.modeling:"
                                 "build_s2_discriminator"),
            "s2-generator": ("voicehub.models.gptsovits.native.modeling:"
                             "build_s2_generator"),
            "trainer-adapter": ("voicehub.models.gptsovits.training:"
                                "GPTSoVITSTrainingAdapter"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16"),
            checkpoint_formats=("safetensors", "pytorch"),
            training=True,
            streaming=False,
            batched_inference=False,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=("compile", "sdpa", "custom-kernels"),
            features=(
                "llm-tts-codec",
                "vits-family",
                "vits-wavenet-gate",
                "autoregressive-semantic-s1",
                "vits-gan-acoustic-s2",
                "frozen-residual-vector-quantizer",
                "voice-cloning",
                "prepared-multilingual-phonemes",
                "prepared-bert-features",
                "prepared-cnhubert-features",
                "source-faithful-staged-fine-tuning",
                "separate-s1-s2-generator-s2-discriminator-phases",
                "strict-checkpoint-inventory",
                "digest-pinned-pytorch-import",
                "safetensors-export",
                "fresh-inference-reload",
                "no-external-runtime",
            ),
        ),
        upstream_revision=GPT_SOVITS_SOURCE_REVISION,
        license_id=GPT_SOVITS_LICENSE,
        metadata={
            "external_llm_backend_blocker":
            ("GPT-SoVITS uses phoneme/BERT embeddings and a custom "
             "semantic head."),
            "vits_architecture_kind":
            "hybrid-acoustic",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source_repository":
            "RVC-Boss/GPT-SoVITS",
            "source_revision":
            GPT_SOVITS_SOURCE_REVISION,
            "checkpoint_repository":
            GPT_SOVITS_REPOSITORY,
            "checkpoint_revision":
            GPT_SOVITS_REVISION,
            "supported_variants": (
                "v1",
                "v2",
                "v2Pro",
                "v2ProPlus",
            ),
            "unsupported_variants": (
                "v3",
                "v4",
                "LoRA",
            ),
            "reference_checkpoints": {
                variant: {
                    component: {
                        "sha256": checkpoint.sha256,
                        "tensor_count": checkpoint.tensor_count,
                        "parameter_count": checkpoint.parameter_count,
                        "inventory_fingerprint": checkpoint.inventory_fingerprint,
                    }
                    for component, checkpoint in (
                        ("s1", release.s1),
                        ("s2_generator", release.s2_generator),
                        ("s2_discriminator", release.s2_discriminator),
                    )
                }
                for variant, release in GPT_SOVITS_VARIANTS.items()
            },
            "checkpoint_import_boundary":
            ("immutable-revision-plus-sha256-plus-weights-only-plus-"
             "explicit-trust-plus-exact-inventory"),
            "official_safetensors_published":
            False,
            "native_checkpoint_format":
            "voicehub-native-gpt-sovits",
            "raw_text_frontend_available":
            False,
            "frontend_boundary": (
                "Exact variant-specific phoneme IDs, Chinese-RoBERTa "
                "features, CNHubert features, and reference spectrograms are "
                "caller-supplied. Pro variants additionally require the "
                "released 20,480-dimensional speaker-verification embedding."),
            "training_scope":
            ("S1 sum cross-entropy plus source-faithful S2 "
             "VITS/LSGAN/mel/KL/codebook objectives"),
            "full_finetuning_ready":
            True,
            "inference_reloadable_export":
            True,
        },
    )


def register_gptsovits_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_GPT_SOVITS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_gptsovits_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_GPT_SOVITS_ALIASES",
    "create_gptsovits_architecture_spec",
    "register_gptsovits_architecture",
]
