"""VoiceHub-owned dense causal-LM architectures with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.causal_lm.native."
_EXPORTS = {
    "CausalLMConfig": _PACKAGE + "configuration",
    "CausalLMDecoderLayer": _PACKAGE + "modeling",
    "CausalLMForCausalLM": _PACKAGE + "modeling",
    "CausalLMModel": _PACKAGE + "modeling",
    "CausalLMModelOutput": _PACKAGE + "modeling",
    "CausalLMOutput": _PACKAGE + "modeling",
    "CausalSelfAttention": _PACKAGE + "modeling",
    "DEFAULT_CAUSAL_LM_ALIASES": _PACKAGE + "registration",
    "GatedMLP": _PACKAGE + "modeling",
    "GraniteConfig": _PACKAGE + "configuration",
    "GraniteForCausalLM": _PACKAGE + "modeling",
    "GraniteModel": _PACKAGE + "modeling",
    "HFCausalLMCheckpointAdapter": _PACKAGE + "checkpoint",
    "HuggingFaceCausalLMCheckpointAdapter": _PACKAGE + "checkpoint",
    "LlamaConfig": _PACKAGE + "configuration",
    "LlamaForCausalLM": _PACKAGE + "modeling",
    "LlamaModel": _PACKAGE + "modeling",
    "Qwen2Config": _PACKAGE + "configuration",
    "Qwen2ForCausalLM": _PACKAGE + "modeling",
    "Qwen2Model": _PACKAGE + "modeling",
    "Qwen3Config": _PACKAGE + "configuration",
    "Qwen3ForCausalLM": _PACKAGE + "modeling",
    "Qwen3Model": _PACKAGE + "modeling",
    "REFERENCE_CAUSAL_LM_CHECKPOINTS": _PACKAGE + "checkpoint",
    "SUPPORTED_CAUSAL_LM_FAMILIES": _PACKAGE + "configuration",
    "TRANSFORMERS_CAUSAL_LM_REVISION": _PACKAGE + "configuration",
    "create_causal_lm_architecture_spec": _PACKAGE + "registration",
    "huggingface_causal_lm_tensor_mapping": _PACKAGE + "checkpoint",
    "huggingface_causal_lm_tensor_shapes": _PACKAGE + "checkpoint",
    "native_causal_lm_tensor_names": _PACKAGE + "checkpoint",
    "native_causal_lm_tensor_shapes": _PACKAGE + "checkpoint",
    "open_causal_lm_tensor_source": _PACKAGE + "checkpoint",
    "register_causal_lm_architecture": _PACKAGE + "registration",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public component only when it is requested."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
