"""Strict official-checkpoint mapping for native HuBERT CTC.

The tensor inventory is pinned to the 424-tensor Safetensors conversion
of ``facebook/hubert-large-ls960-ft`` at immutable revision
``ba42e7f7a888fd65f7af7849c452e3e7d5216aad``. That conversion has parent
revision ``ece5fabbf034c1073acae96d5401b25be96709d8`` and is byte-for-
byte tensor-equivalent to the official PyTorch checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing.adapters import CheckpointAdapter
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan
from voicehub.models.asr_hubert.native.configuration import HubertConfig
from voicehub.models.asr_wav2vec2.native.checkpoint import native_wav2vec2_tensor_shapes as _wav2vec2_shapes
from voicehub.models.asr_wav2vec2.native.checkpoint import safetensors_header_fingerprint

TRANSFORMERS_HUBERT_REVISION = "ebea912f0bb6f9e28ad2df04acd9b4df035933a9"
FACEBOOK_HUBERT_LARGE_LS960_FT_REVISION = ("ece5fabbf034c1073acae96d5401b25be96709d8")
FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_REVISION = ("ba42e7f7a888fd65f7af7849c452e3e7d5216aad")
FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_SHA256 = (
    "1fefcd85b08c83451afd1df872bc11b92333dc4b5393506def29a20baa69c4ed")
FACEBOOK_HUBERT_LARGE_LS960_FT_HEADER_FINGERPRINT = (
    "c41b63b3e040e0ff791721caa449733917ca7f0f627516e86f6a5ed42e382b19")

TensorMapping = tuple[tuple[str, str], ...]
TensorShapes = dict[str, tuple[int, ...]]


def native_hubert_tensor_shapes(config: HubertConfig | Mapping[str, Any], ) -> TensorShapes:
    """Return the complete native HuBERT CTC namespace and shapes."""
    resolved = HubertConfig.coerce(config)
    wav2vec2_shapes = _wav2vec2_shapes(resolved)
    shapes = {("hubert." + name.removeprefix("wav2vec2.") if name.startswith("wav2vec2.") else name): shape
              for name, shape in wav2vec2_shapes.items()}
    if not resolved.feat_proj_layer_norm:
        shapes.pop("hubert.feature_projection.layer_norm.weight")
        shapes.pop("hubert.feature_projection.layer_norm.bias")
    if (resolved.mask_time_prob > 0.0 or resolved.mask_feature_prob > 0.0):
        shapes["hubert.masked_spec_embed"] = (resolved.hidden_size, )
    return shapes


def native_hubert_tensor_names(config: HubertConfig | Mapping[str, Any], ) -> tuple[str, ...]:
    """Return every persistent HuBERT tensor in canonical sorted order."""
    return tuple(sorted(native_hubert_tensor_shapes(config)))


def huggingface_hubert_tensor_mapping(
    config: HubertConfig | Mapping[str, Any],
    *,
    source_prefix: str = "",
) -> TensorMapping:
    """Map the official Hugging Face HuBERT namespace to VoiceHub."""
    if not isinstance(source_prefix, str):
        raise TypeError("`source_prefix` must be a string.")
    return tuple((f"{source_prefix}{name}", name) for name in native_hubert_tensor_names(config))


def huggingface_hubert_tensor_shapes(
    config: HubertConfig | Mapping[str, Any],
    *,
    source_prefix: str = "",
) -> TensorShapes:
    """Return expected official source tensor shapes."""
    if not isinstance(source_prefix, str):
        raise TypeError("`source_prefix` must be a string.")
    return {f"{source_prefix}{name}": shape for name, shape in native_hubert_tensor_shapes(config).items()}


class HuggingFaceHubertCheckpointAdapter(CheckpointAdapter):
    """Load official HuBERT CTC Safetensors into the native graph."""

    architecture_id = "hubert"
    adapter_id = "huggingface-hubert-ctc-safetensors"
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
        declares_hubert = (
            model_type in {"hubert", "asr_hubert"} or
            any(str(name) == "HubertForCTC" for name in architectures))
        has_safetensors = any(
            path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json") for path in files)
        return declares_hubert and has_safetensors

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        source_prefix = config.get("_checkpoint_prefix", "")
        if not isinstance(source_prefix, str):
            raise TypeError("`_checkpoint_prefix` must be a string.")
        return TensorPlan(
            rules=tuple(
                CopyTensor(source, target) for source, target in huggingface_hubert_tensor_mapping(
                    config,
                    source_prefix=source_prefix,
                )),
            ignored_source_patterns=("*position_ids", ),
        )


HFHubertCheckpointAdapter = HuggingFaceHubertCheckpointAdapter

__all__ = [
    "FACEBOOK_HUBERT_LARGE_LS960_FT_HEADER_FINGERPRINT",
    "FACEBOOK_HUBERT_LARGE_LS960_FT_REVISION",
    "FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_REVISION",
    "FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_SHA256",
    "HFHubertCheckpointAdapter",
    "HuggingFaceHubertCheckpointAdapter",
    "TRANSFORMERS_HUBERT_REVISION",
    "TensorMapping",
    "TensorShapes",
    "huggingface_hubert_tensor_mapping",
    "huggingface_hubert_tensor_shapes",
    "native_hubert_tensor_names",
    "native_hubert_tensor_shapes",
    "safetensors_header_fingerprint",
]
