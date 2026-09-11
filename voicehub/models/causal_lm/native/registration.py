"""Lazy declaration for VoiceHub's dense decoder-only LM family."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.models.causal_lm.native.checkpoint import REFERENCE_CAUSAL_LM_CHECKPOINTS
from voicehub.models.causal_lm.native.configuration import TRANSFORMERS_CAUSAL_LM_REVISION
from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_CAUSAL_LM_ALIASES = (
    "native-causal-lm",
    "granite",
    "llama",
    "qwen2",
    "qwen3",
)


def create_causal_lm_architecture_spec() -> ArchitectureSpec:
    """Create the immutable native Granite/Llama/Qwen declaration."""
    source_root = (
        "https://github.com/huggingface/transformers/tree/"
        f"{TRANSFORMERS_CAUSAL_LM_REVISION}/src/transformers/models")
    return ArchitectureSpec(
        architecture_id="causal-lm",
        version="1",
        model_builder=("voicehub.models.causal_lm.native.modeling:"
                       "CausalLMForCausalLM"),
        config=("voicehub.models.causal_lm.native.configuration:CausalLMConfig"),
        decoder="voicehub.generation.engine:AutoregressiveGenerator",
        objective="voicehub.objectives.sequence:sequence_cross_entropy",
        checkpoint_adapter=(
            "voicehub.models.causal_lm.native.checkpoint:"
            "HuggingFaceCausalLMCheckpointAdapter"),
        components={
            "granite-config": ("voicehub.models.causal_lm.native.configuration:GraniteConfig"),
            "granite-model": ("voicehub.models.causal_lm.native.modeling:GraniteForCausalLM"),
            "llama-config": ("voicehub.models.causal_lm.native.configuration:LlamaConfig"),
            "llama-model": ("voicehub.models.causal_lm.native.modeling:LlamaForCausalLM"),
            "qwen2-config": ("voicehub.models.causal_lm.native.configuration:Qwen2Config"),
            "qwen2-model": ("voicehub.models.causal_lm.native.modeling:Qwen2ForCausalLM"),
            "qwen3-config": ("voicehub.models.causal_lm.native.configuration:Qwen3Config"),
            "qwen3-model": ("voicehub.models.causal_lm.native.modeling:Qwen3ForCausalLM"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=True,
            batched_inference=True,
            distributed_training=True,
            optimization_passes=("compile", ),
            features=(
                "llm-tts-codec",
                "decoder-only",
                "causal-language-modeling",
                "grouped-query-attention",
                "kv-cache",
                "granite",
                "llama",
                "qwen2",
                "qwen3",
                "rmsnorm",
                "rope",
                "llama3-rope",
                "swiglu",
                "checkpoint-conversion",
            ),
        ),
        upstream_revision=TRANSFORMERS_CAUSAL_LM_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family": ("granite", "llama", "qwen2", "qwen3"),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "transformers_sources": {
                "granite": f"{source_root}/granite",
                "llama": f"{source_root}/llama",
                "qwen2": f"{source_root}/qwen2",
                "qwen3": f"{source_root}/qwen3",
            },
            "reference_checkpoints":
            REFERENCE_CAUSAL_LM_CHECKPOINTS,
            "scope":
            "dense-decoder-backbone",
            "unsupported_semantics": (
                "mixture-of-experts",
                "rope-scaling-other-than-llama3",
                "sliding-window-attention",
            ),
        },
    )


def register_causal_lm_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_CAUSAL_LM_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register the lazy native decoder declaration and return it."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_causal_lm_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_CAUSAL_LM_ALIASES",
    "TRANSFORMERS_CAUSAL_LM_REVISION",
    "create_causal_lm_architecture_spec",
    "register_causal_lm_architecture",
]
