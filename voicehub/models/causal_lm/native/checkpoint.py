"""Strict Safetensors mapping for native Granite, Llama, Qwen2, and Qwen3.

Official Hugging Face tensor namespaces were reviewed at Transformers revision
``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``.  The native graph intentionally
uses the same stable parameter names, so conversion is a reviewable rename-only
plan and supports constant-memory streaming loads.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voicehub.checkpointing.adapters import CheckpointAdapter
from voicehub.checkpointing.safetensors import SafeTensorReader, ShardedSafeTensorReader
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan
from voicehub.models.causal_lm.native.configuration import CausalLMConfig

TensorMapping = tuple[tuple[str, str], ...]
TensorShapes = dict[str, tuple[int, ...]]
CausalLMTensorSource = SafeTensorReader | ShardedSafeTensorReader
REFERENCE_CAUSAL_LM_CHECKPOINTS = MappingProxyType({
    "llama":
    MappingProxyType({
        "model_id": "optimum-internal-testing/tiny-random-llama",
        "revision": "ffd1b09a4fa19a9086ed4e36d57eda60fcb2c2c0",
        "tensor_count": 21,
    }),
    "qwen2":
    MappingProxyType({
        "model_id": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        "revision": "4b10ebee6e13a2669155516652960f50984399fd",
        "tensor_count": 27,
    }),
    "qwen3":
    MappingProxyType({
        "model_id": "trl-internal-testing/tiny-Qwen3ForCausalLM",
        "revision": "52b2e48b0004586eff92c403efa5ce5547c43a45",
        "tensor_count": 25,
    }),
})


def native_causal_lm_tensor_shapes(config: CausalLMConfig | Mapping[str, Any], ) -> TensorShapes:
    """Return the complete native parameter namespace and expected shapes."""
    resolved = CausalLMConfig.coerce(config)
    hidden_size = resolved.hidden_size
    head_dim = resolved.head_dim
    query_size = resolved.num_attention_heads * head_dim
    key_value_size = resolved.num_key_value_heads * head_dim
    shapes: TensorShapes = {
        "model.embed_tokens.weight": (
            resolved.vocab_size,
            hidden_size,
        ),
    }
    for index in range(resolved.num_hidden_layers):
        prefix = f"model.layers.{index}"
        attention_prefix = f"{prefix}.self_attn"
        shapes[f"{attention_prefix}.q_proj.weight"] = (
            query_size,
            hidden_size,
        )
        shapes[f"{attention_prefix}.k_proj.weight"] = (
            key_value_size,
            hidden_size,
        )
        shapes[f"{attention_prefix}.v_proj.weight"] = (
            key_value_size,
            hidden_size,
        )
        shapes[f"{attention_prefix}.o_proj.weight"] = (
            hidden_size,
            query_size,
        )
        if resolved.qkv_bias:
            shapes[f"{attention_prefix}.q_proj.bias"] = (query_size, )
            shapes[f"{attention_prefix}.k_proj.bias"] = (key_value_size, )
            shapes[f"{attention_prefix}.v_proj.bias"] = (key_value_size, )
        if resolved.attention_output_bias:
            shapes[f"{attention_prefix}.o_proj.bias"] = (hidden_size, )
        if resolved.uses_qk_norm:
            shapes[f"{attention_prefix}.q_norm.weight"] = (head_dim, )
            shapes[f"{attention_prefix}.k_norm.weight"] = (head_dim, )

        mlp_prefix = f"{prefix}.mlp"
        shapes[f"{mlp_prefix}.gate_proj.weight"] = (
            resolved.intermediate_size,
            hidden_size,
        )
        shapes[f"{mlp_prefix}.up_proj.weight"] = (
            resolved.intermediate_size,
            hidden_size,
        )
        shapes[f"{mlp_prefix}.down_proj.weight"] = (
            hidden_size,
            resolved.intermediate_size,
        )
        if resolved.model_type in {"granite", "llama"} and resolved.mlp_bias:
            shapes[f"{mlp_prefix}.gate_proj.bias"] = (resolved.intermediate_size, )
            shapes[f"{mlp_prefix}.up_proj.bias"] = (resolved.intermediate_size, )
            shapes[f"{mlp_prefix}.down_proj.bias"] = (hidden_size, )
        shapes[f"{prefix}.input_layernorm.weight"] = (hidden_size, )
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (hidden_size, )
    shapes["model.norm.weight"] = (hidden_size, )
    shapes["lm_head.weight"] = (resolved.vocab_size, hidden_size)
    return shapes


def native_causal_lm_tensor_names(config: CausalLMConfig | Mapping[str, Any], ) -> tuple[str, ...]:
    """Return every persistent model tensor in canonical sorted order."""
    return tuple(sorted(native_causal_lm_tensor_shapes(config)))


def huggingface_causal_lm_tensor_mapping(
    config: CausalLMConfig | Mapping[str, Any],
    *,
    source_prefix: str = "",
) -> TensorMapping:
    """Map an official dense HF checkpoint namespace into VoiceHub."""
    if not isinstance(source_prefix, str):
        raise TypeError("`source_prefix` must be a string.")
    resolved = CausalLMConfig.coerce(config)
    mapping = []
    for target in native_causal_lm_tensor_names(resolved):
        source = target
        if resolved.tie_word_embeddings and target == "lm_head.weight":
            source = "model.embed_tokens.weight"
        mapping.append((f"{source_prefix}{source}", target))
    return tuple(mapping)


def huggingface_causal_lm_tensor_shapes(
    config: CausalLMConfig | Mapping[str, Any],
    *,
    source_prefix: str = "",
) -> TensorShapes:
    """Return the strict official source namespace and expected shapes."""
    if not isinstance(source_prefix, str):
        raise TypeError("`source_prefix` must be a string.")
    resolved = CausalLMConfig.coerce(config)
    native_shapes = native_causal_lm_tensor_shapes(resolved)
    result: TensorShapes = {}
    for source, target in huggingface_causal_lm_tensor_mapping(
            resolved,
            source_prefix=source_prefix,
    ):
        result[source] = native_shapes[target]
    return result


class HuggingFaceCausalLMCheckpointAdapter(CheckpointAdapter):
    """Load official dense Granite/Llama/Qwen Safetensors."""

    architecture_id = "causal-lm"
    adapter_id = "huggingface-llama-qwen-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        model_type = str(config.get("model_type", "")).lower()
        architectures = config.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        known_architectures = {
            "GraniteForCausalLM",
            "LlamaForCausalLM",
            "Qwen2ForCausalLM",
            "Qwen3ForCausalLM",
        }
        declares_family = (
            model_type in {"granite", "llama", "qwen2", "qwen3"} or
            any(str(name) in known_architectures for name in architectures))
        has_safetensors = any(
            path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json") for path in files)
        return declares_family and has_safetensors

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        source_prefix = config.get("_checkpoint_prefix", "")
        if not isinstance(source_prefix, str):
            raise TypeError("`_checkpoint_prefix` must be a string.")
        return TensorPlan(
            rules=tuple(
                CopyTensor(source, target) for source, target in huggingface_causal_lm_tensor_mapping(
                    config,
                    source_prefix=source_prefix,
                )),
            ignored_source_patterns=(
                f"{source_prefix}lm_head.weight",
                f"{source_prefix}*rotary_emb.inv_freq",
                f"{source_prefix}*rotary_emb.original_inv_freq",
            ),
        )


def open_causal_lm_tensor_source(path: str | Path, ) -> AbstractContextManager[CausalLMTensorSource]:
    """Open an unambiguous single-file or sharded local checkpoint."""
    logical = Path(path).expanduser()
    resolved = logical.resolve()
    if logical.is_file():
        # Standard Hugging Face snapshots expose model filenames as
        # symlinks to extensionless content-addressed blobs. Classify the
        # logical artifact name before resolving that symlink.
        if logical.name.endswith(".safetensors.index.json"):
            return ShardedSafeTensorReader(logical)
        if logical.suffix == ".safetensors":
            return SafeTensorReader(resolved)
        raise ValueError("Causal-LM checkpoint files must be Safetensors or an index JSON.")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Causal-LM checkpoint path was not found: {resolved}")

    index = resolved / "model.safetensors.index.json"
    single = resolved / "model.safetensors"
    if index.is_file() and single.is_file():
        raise ValueError(
            "Checkpoint directory contains both single-file and sharded "
            "model artifacts; choose one explicitly.")
    if index.is_file():
        return ShardedSafeTensorReader(index)
    if single.is_file():
        return SafeTensorReader(single)
    candidates = tuple(sorted(resolved.glob("*.safetensors")))
    if len(candidates) == 1:
        return SafeTensorReader(candidates[0])
    if not candidates:
        raise FileNotFoundError(f"No Safetensors checkpoint was found in {resolved}.")
    raise ValueError(
        "Checkpoint directory contains multiple Safetensors files without "
        "model.safetensors.index.json.")


HFCausalLMCheckpointAdapter = HuggingFaceCausalLMCheckpointAdapter

__all__ = [
    "HFCausalLMCheckpointAdapter",
    "HuggingFaceCausalLMCheckpointAdapter",
    "REFERENCE_CAUSAL_LM_CHECKPOINTS",
    "TensorMapping",
    "TensorShapes",
    "huggingface_causal_lm_tensor_mapping",
    "huggingface_causal_lm_tensor_shapes",
    "native_causal_lm_tensor_names",
    "native_causal_lm_tensor_shapes",
    "open_causal_lm_tensor_source",
]
