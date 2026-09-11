"""Strict immutable-checkpoint mapping for native WavLM CTC.

The public ASR checkpoint originally contains a legacy PyTorch pickle at
revision ``02c289c4471cd1ba4b0ff3e7c304afe395c5026a``.  VoiceHub pins
Safetensors conversion commit
``561f43a6081f379876b6633a38526aabe140ba3b``, whose sole parent is that
original revision.  Its 250-tensor inventory and shape fingerprint are
frozen below; VoiceHub never deserializes the pickle.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing.adapters import CheckpointAdapter
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan
from voicehub.models.asr_wav2vec2.native.checkpoint import native_wav2vec2_tensor_shapes as _wav2vec2_shapes
from voicehub.models.asr_wav2vec2.native.checkpoint import safetensors_header_fingerprint
from voicehub.models.asr_wavlm.native.configuration import WavLMConfig

MICROSOFT_WAVLM_SOURCE_REVISION = "833df7e7832e5064a281131ee64a481afa8e5b95"
TRANSFORMERS_WAVLM_REVISION = "ebea912f0bb6f9e28ad2df04acd9b4df035933a9"
WAVLM_BASE_PLUS_CTC_REVISION = "02c289c4471cd1ba4b0ff3e7c304afe395c5026a"
WAVLM_BASE_PLUS_CTC_SAFETENSORS_REVISION = ("561f43a6081f379876b6633a38526aabe140ba3b")
WAVLM_BASE_PLUS_CTC_SAFETENSORS_SHA256 = ("cc6b213ad14d4589568ad844841f6d3b3c58d12f2326cce14804530c02ff2dd1")
WAVLM_BASE_PLUS_CTC_HEADER_FINGERPRINT = ("81881f33a6f12eabd0d11f85bdd8fec5e1c46a9284bf16d02987a4d5b234ab9b")

TensorMapping = tuple[tuple[str, str], ...]
TensorShapes = dict[str, tuple[int, ...]]


def native_wavlm_tensor_shapes(config: WavLMConfig | Mapping[str, Any], ) -> TensorShapes:
    """Return the complete native WavLM CTC namespace and tensor shapes."""
    resolved = WavLMConfig.coerce(config)
    shared_shapes = _wav2vec2_shapes(resolved)
    shapes = {("wavlm." + name.removeprefix("wav2vec2.") if name.startswith("wav2vec2.") else name): shape
              for name, shape in shared_shapes.items()}
    head_size = resolved.hidden_size // resolved.num_attention_heads
    for index in range(resolved.num_hidden_layers):
        attention_prefix = f"wavlm.encoder.layers.{index}.attention"
        shapes[f"{attention_prefix}.gru_rel_pos_const"] = (
            1,
            resolved.num_attention_heads,
            1,
            1,
        )
        shapes[f"{attention_prefix}.gru_rel_pos_linear.weight"] = (
            8,
            head_size,
        )
        shapes[f"{attention_prefix}.gru_rel_pos_linear.bias"] = (8, )
    shapes["wavlm.encoder.layers.0.attention.rel_attn_embed.weight"] = (
        resolved.num_buckets,
        resolved.num_attention_heads,
    )
    if (resolved.mask_time_prob > 0.0 or resolved.mask_feature_prob > 0.0):
        shapes["wavlm.masked_spec_embed"] = (resolved.hidden_size, )
    return shapes


def native_wavlm_tensor_names(config: WavLMConfig | Mapping[str, Any], ) -> tuple[str, ...]:
    """Return every persistent WavLM tensor in canonical sorted order."""
    return tuple(sorted(native_wavlm_tensor_shapes(config)))


def huggingface_wavlm_tensor_mapping(
    config: WavLMConfig | Mapping[str, Any],
    *,
    source_prefix: str = "",
) -> TensorMapping:
    """Map an official Hugging Face WavLM CTC namespace to VoiceHub."""
    if not isinstance(source_prefix, str):
        raise TypeError("`source_prefix` must be a string.")
    return tuple((f"{source_prefix}{name}", name) for name in native_wavlm_tensor_names(config))


def huggingface_wavlm_tensor_shapes(
    config: WavLMConfig | Mapping[str, Any],
    *,
    source_prefix: str = "",
) -> TensorShapes:
    """Return the complete expected source tensor inventory."""
    if not isinstance(source_prefix, str):
        raise TypeError("`source_prefix` must be a string.")
    return {f"{source_prefix}{name}": shape for name, shape in native_wavlm_tensor_shapes(config).items()}


class HuggingFaceWavLMCheckpointAdapter(CheckpointAdapter):
    """Load WavLM CTC Safetensors into the VoiceHub-owned graph."""

    architecture_id = "wavlm"
    adapter_id = "huggingface-wavlm-ctc-safetensors"
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
        declares_wavlm = (
            model_type in {"wavlm", "asr_wavlm"} or any(str(name) == "WavLMForCTC" for name in architectures))
        has_safetensors = any(
            path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json") for path in files)
        return declares_wavlm and has_safetensors

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        source_prefix = config.get("_checkpoint_prefix", "")
        if not isinstance(source_prefix, str):
            raise TypeError("`_checkpoint_prefix` must be a string.")
        return TensorPlan(
            rules=tuple(
                CopyTensor(source, target) for source, target in huggingface_wavlm_tensor_mapping(
                    config,
                    source_prefix=source_prefix,
                )),
            ignored_source_patterns=("*position_ids", ),
        )


HFWavLMCheckpointAdapter = HuggingFaceWavLMCheckpointAdapter

__all__ = [
    "HFWavLMCheckpointAdapter",
    "HuggingFaceWavLMCheckpointAdapter",
    "MICROSOFT_WAVLM_SOURCE_REVISION",
    "TRANSFORMERS_WAVLM_REVISION",
    "TensorMapping",
    "TensorShapes",
    "WAVLM_BASE_PLUS_CTC_HEADER_FINGERPRINT",
    "WAVLM_BASE_PLUS_CTC_REVISION",
    "WAVLM_BASE_PLUS_CTC_SAFETENSORS_REVISION",
    "WAVLM_BASE_PLUS_CTC_SAFETENSORS_SHA256",
    "huggingface_wavlm_tensor_mapping",
    "huggingface_wavlm_tensor_shapes",
    "native_wavlm_tensor_names",
    "native_wavlm_tensor_shapes",
    "safetensors_header_fingerprint",
]
