"""Lazy architecture declaration for native Inflect v2."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.inflecttts.native.metadata import (
    INFLECT_LICENSE,
    INFLECT_MICRO_V2_CHECKPOINT_SHA256,
    INFLECT_MICRO_V2_CONFIG_SHA256,
    INFLECT_MICRO_V2_INVENTORY_FINGERPRINT,
    INFLECT_MICRO_V2_REPOSITORY,
    INFLECT_MICRO_V2_REVISION,
    INFLECT_NANO_V2_CHECKPOINT_SHA256,
    INFLECT_NANO_V2_CONFIG_SHA256,
    INFLECT_NANO_V2_INVENTORY_FINGERPRINT,
    INFLECT_NANO_V2_REPOSITORY,
    INFLECT_NANO_V2_REVISION,
    INFLECT_SOURCE_REPOSITORY,
    INFLECT_SOURCE_REVISION,
)
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

INFLECT_GITHUB_REVISION = INFLECT_SOURCE_REVISION
DEFAULT_INFLECT_ALIASES = (
    "native-inflecttts",
    "inflect-v2",
    "inflect-micro-v2",
    "inflect-nano-v2",
)


def create_inflect_architecture_spec() -> ArchitectureSpec:
    """Describe the audited graph without importing PyTorch."""
    return ArchitectureSpec(
        architecture_id="inflecttts",
        version="2",
        model_builder=("voicehub.models.inflecttts.native.modeling:"
                       "build_inflect_model"),
        config=("voicehub.models.inflecttts.native.configuration:"
                "InflectV2Config"),
        processor=("voicehub.models.inflecttts.native.frontend:phonemes_to_ids"),
        decoder=("voicehub.models.inflecttts.native.modeling:Generator"),
        objective=("voicehub.models.inflecttts.native.training:"
                   "InflectV2TrainingModel"),
        checkpoint_adapter=("voicehub.models.inflecttts.native.checkpoint:"
                            "load_inflect_checkpoint"),
        components={
            "artifact-resolver":
            ("voicehub.models.inflecttts.native.checkpoint:"
             "resolve_inflect_artifacts"),
            "checkpoint-converter":
            ("voicehub.models.inflecttts.native.checkpoint:"
             "convert_inflect_legacy_checkpoint"),
            "checkpoint-exporter":
            ("voicehub.models.inflecttts.native.checkpoint:"
             "export_inflect_checkpoint"),
            "inference-runtime": ("voicehub.models.inflecttts.native.runtime:"
                                  "InflectV2Runtime"),
            "multi-period-discriminator":
            ("voicehub.models.inflecttts.native.modeling:"
             "MultiPeriodDiscriminator"),
            "trainer-adapter": ("voicehub.models.inflecttts.training:"
                                "InflectTTSTrainingAdapter"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", "pytorch"),
            training=True,
            streaming=False,
            batched_inference=False,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=("compile", "custom-kernels"),
            features=(
                "vits-family",
                "vits-wavenet-gate",
                "fixed-single-voice",
                "deterministic-duration-predictor",
                "monotonic-alignment-search",
                "native-vits-generator",
                "native-vits-posterior",
                "native-multi-period-discriminator",
                "full-vits-warm-start-objective",
                "preprocessed-linear-spectrogram-training",
                "explicit-espeak-compatible-phonemes",
                "strict-checkpoint-validation",
                "trusted-pickle-conversion",
                "safetensors-export",
                "fresh-inference-reload",
                "no-external-runtime",
            ),
        ),
        upstream_revision=INFLECT_MICRO_V2_REVISION,
        license_id=INFLECT_LICENSE,
        metadata={
            "vits_architecture_kind": "classic",
            "implementation": "voicehub-native",
            "tensor_backend": "pytorch",
            "source_repository": INFLECT_SOURCE_REPOSITORY,
            "source_revision": INFLECT_GITHUB_REVISION,
            "released_runtime_revision": INFLECT_MICRO_V2_REVISION,
            "reference_checkpoints": {
                "micro": {
                    "repository": INFLECT_MICRO_V2_REPOSITORY,
                    "revision": INFLECT_MICRO_V2_REVISION,
                    "checkpoint_sha256": INFLECT_MICRO_V2_CHECKPOINT_SHA256,
                    "config_sha256": INFLECT_MICRO_V2_CONFIG_SHA256,
                    "tensor_count": 410,
                    "parameter_count": 9_356_513,
                    "inventory_fingerprint": INFLECT_MICRO_V2_INVENTORY_FINGERPRINT,
                },
                "nano": {
                    "repository": INFLECT_NANO_V2_REPOSITORY,
                    "revision": INFLECT_NANO_V2_REVISION,
                    "checkpoint_sha256": INFLECT_NANO_V2_CHECKPOINT_SHA256,
                    "config_sha256": INFLECT_NANO_V2_CONFIG_SHA256,
                    "tensor_count": 410,
                    "parameter_count": 3_966_721,
                    "inventory_fingerprint": INFLECT_NANO_V2_INVENTORY_FINGERPRINT,
                },
            },
            "official_safetensors_published": False,
            "native_safetensors_namespace": "exact released generator state-dict names",
            "checkpoint_import_boundary": "explicit-trust-plus-weights-only-plus-sha256-plus-inventory",
            "raw_text_frontend_available": False,
            "frontend_boundary": "preprocessed en-us phonemes or exact token IDs",
            "training_scope": "full VITS objective warm-started from deployable generator",
            "fresh_training_components": (
                "posterior-encoder",
                "multi-period-discriminator",
            ),
            "fresh_discriminator_parameter_count": 46_747_132,
            "author_recipe_recovered": False,
            "full_finetuning_ready": True,
        },
    )


def register_inflect_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_INFLECT_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_inflect_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_INFLECT_ALIASES",
    "INFLECT_GITHUB_REVISION",
    "create_inflect_architecture_spec",
    "register_inflect_architecture",
]
