"""Strict Safetensors adapters for native NeuTTS and NeuCodec."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing import CheckpointAdapter, CopyTensor, TensorPlan
from voicehub.models.causal_lm.native.checkpoint import HuggingFaceCausalLMCheckpointAdapter


class NeuTTSCheckpointAdapter(HuggingFaceCausalLMCheckpointAdapter):
    """Identity-compatible Llama/Qwen loader with a NeuTTS manifest ID."""

    architecture_id = "neutts-backbone"
    adapter_id = "official-neutts-safetensors"
    adapter_version = "1"


class NeuCodecCheckpointAdapter(CheckpointAdapter):
    """Load the official self-contained codec conversion without renaming."""

    architecture_id = "neucodec"
    adapter_id = "official-neucodec-safetensors"
    adapter_version = "1"

    def __init__(self, tensor_names: tuple[str, ...]) -> None:
        if (not isinstance(tensor_names, tuple) or not tensor_names or
                len(set(tensor_names)) != len(tensor_names) or
                any(not isinstance(name, str) or not name for name in tensor_names)):
            raise ValueError("`tensor_names` must be a non-empty unique tuple.")
        self.tensor_names = tuple(sorted(tensor_names))

    @classmethod
    def for_model(cls, model: Any) -> NeuCodecCheckpointAdapter:
        state_dict = getattr(model, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("NeuCodec checkpoint target must expose state_dict().")
        return cls(tuple(state_dict()))

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        return (
            str(config.get("model_type", "")).lower() == "neucodec" and
            any(path.suffix == ".safetensors" for path in files))

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        del config
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in self.tensor_names))


__all__ = [
    "NeuCodecCheckpointAdapter",
    "NeuTTSCheckpointAdapter",
]
