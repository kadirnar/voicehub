"""Strict Hugging Face checkpoint mapping for native Wav2Vec2 CTC.

The namespace and shapes were checked against the Safetensors header of
``facebook/wav2vec2-base-960h`` at immutable revision
``22aad52d435eb6dbaf354bdad9b0da84ce7d6156``.  The implementation uses only
VoiceHub checkpoint rules and never imports Transformers or Safetensors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing.adapters import CheckpointAdapter
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan
from voicehub.models.asr_wav2vec2.native.configuration import Wav2Vec2Config

FACEBOOK_WAV2VEC2_BASE_960H_REVISION = ("22aad52d435eb6dbaf354bdad9b0da84ce7d6156")
FACEBOOK_WAV2VEC2_BASE_960H_HEADER_FINGERPRINT = (
    "c56641e58f851eaa253cade293069281563c63aaa38aa03e6fe6b724aa193710")

TensorMapping = tuple[tuple[str, str], ...]
TensorShapes = dict[str, tuple[int, ...]]
_ConfigInput = Wav2Vec2Config | Mapping[str, Any]
_BaseTensorShapes = tuple[Wav2Vec2Config, TensorShapes]


def native_wav2vec2_tensor_shapes(config: Wav2Vec2Config | Mapping[str, Any], ) -> TensorShapes:
    """Return the complete native CTC state-dict namespace and shapes."""
    resolved = Wav2Vec2Config.coerce(config)
    hidden_size = resolved.hidden_size
    shapes: TensorShapes = {}

    input_channels = 1
    for index, (output_channels, kernel_size) in enumerate(zip(resolved.conv_dim, resolved.conv_kernel)):
        prefix = f"wav2vec2.feature_extractor.conv_layers.{index}"
        shapes[f"{prefix}.conv.weight"] = (
            output_channels,
            input_channels,
            kernel_size,
        )
        if resolved.conv_bias:
            shapes[f"{prefix}.conv.bias"] = (output_channels, )
        has_normalization = (resolved.feat_extract_norm == "layer" or index == 0)
        if has_normalization:
            shapes[f"{prefix}.layer_norm.weight"] = (output_channels, )
            shapes[f"{prefix}.layer_norm.bias"] = (output_channels, )
        input_channels = output_channels

    projection_prefix = "wav2vec2.feature_projection"
    shapes[f"{projection_prefix}.layer_norm.weight"] = (resolved.conv_dim[-1], )
    shapes[f"{projection_prefix}.layer_norm.bias"] = (resolved.conv_dim[-1], )
    shapes[f"{projection_prefix}.projection.weight"] = (
        hidden_size,
        resolved.conv_dim[-1],
    )
    shapes[f"{projection_prefix}.projection.bias"] = (hidden_size, )

    position_prefix = "wav2vec2.encoder.pos_conv_embed.conv"
    shapes[f"{position_prefix}.weight_g"] = (
        1,
        1,
        resolved.num_conv_pos_embeddings,
    )
    shapes[f"{position_prefix}.weight_v"] = (
        hidden_size,
        hidden_size // resolved.num_conv_pos_embedding_groups,
        resolved.num_conv_pos_embeddings,
    )
    shapes[f"{position_prefix}.bias"] = (hidden_size, )
    shapes["wav2vec2.encoder.layer_norm.weight"] = (hidden_size, )
    shapes["wav2vec2.encoder.layer_norm.bias"] = (hidden_size, )

    for index in range(resolved.num_hidden_layers):
        prefix = f"wav2vec2.encoder.layers.{index}"
        for projection in ("k_proj", "v_proj", "q_proj", "out_proj"):
            shapes[f"{prefix}.attention.{projection}.weight"] = (
                hidden_size,
                hidden_size,
            )
            shapes[f"{prefix}.attention.{projection}.bias"] = (hidden_size, )
        shapes[f"{prefix}.layer_norm.weight"] = (hidden_size, )
        shapes[f"{prefix}.layer_norm.bias"] = (hidden_size, )
        shapes[f"{prefix}.feed_forward.intermediate_dense.weight"] = (
            resolved.intermediate_size,
            hidden_size,
        )
        shapes[f"{prefix}.feed_forward.intermediate_dense.bias"] = (resolved.intermediate_size, )
        shapes[f"{prefix}.feed_forward.output_dense.weight"] = (
            hidden_size,
            resolved.intermediate_size,
        )
        shapes[f"{prefix}.feed_forward.output_dense.bias"] = (hidden_size, )
        shapes[f"{prefix}.final_layer_norm.weight"] = (hidden_size, )
        shapes[f"{prefix}.final_layer_norm.bias"] = (hidden_size, )

    shapes["lm_head.weight"] = (resolved.vocab_size, hidden_size)
    shapes["lm_head.bias"] = (resolved.vocab_size, )
    return shapes


def native_wav2vec2_tensor_names(config: Wav2Vec2Config | Mapping[str, Any], ) -> tuple[str, ...]:
    """Return every persistent tensor in canonical sorted order."""
    return tuple(sorted(native_wav2vec2_tensor_shapes(config)))


def _native_wav2vec2_base_tensor_shapes(config: _ConfigInput) -> _BaseTensorShapes:
    resolved = Wav2Vec2Config.coerce(config)
    shapes = native_wav2vec2_tensor_shapes(resolved)
    del shapes["lm_head.weight"]
    del shapes["lm_head.bias"]
    return resolved, shapes


def native_wav2vec2_sequence_classification_tensor_shapes(config: _ConfigInput) -> TensorShapes:
    """Return the official clip-classification checkpoint namespace."""
    resolved, shapes = _native_wav2vec2_base_tensor_shapes(config)
    if resolved.use_weighted_layer_sum:
        shapes["layer_weights"] = (resolved.num_hidden_layers + 1, )
    shapes["projector.weight"] = (
        resolved.classifier_proj_size,
        resolved.hidden_size,
    )
    shapes["projector.bias"] = (resolved.classifier_proj_size, )
    shapes["classifier.weight"] = (
        resolved.num_labels,
        resolved.classifier_proj_size,
    )
    shapes["classifier.bias"] = (resolved.num_labels, )
    return shapes


def native_wav2vec2_frame_classification_tensor_shapes(config: _ConfigInput) -> TensorShapes:
    """Return the official frame-classification checkpoint namespace."""
    resolved, shapes = _native_wav2vec2_base_tensor_shapes(config)
    if resolved.use_weighted_layer_sum:
        shapes["layer_weights"] = (resolved.num_hidden_layers + 1, )
    shapes["classifier.weight"] = (
        resolved.num_labels,
        resolved.hidden_size,
    )
    shapes["classifier.bias"] = (resolved.num_labels, )
    return shapes


def huggingface_wav2vec2_tensor_mapping(
    config: Wav2Vec2Config | Mapping[str, Any],
    *,
    source_prefix: str = "",
) -> TensorMapping:
    """Map an official Hugging Face Wav2Vec2 CTC namespace to VoiceHub."""
    if not isinstance(source_prefix, str):
        raise TypeError("`source_prefix` must be a string.")
    return tuple((f"{source_prefix}{target}", target) for target in native_wav2vec2_tensor_names(config))


def huggingface_wav2vec2_tensor_shapes(
    config: Wav2Vec2Config | Mapping[str, Any],
    *,
    source_prefix: str = "",
) -> TensorShapes:
    """Return expected official source tensor shapes."""
    if not isinstance(source_prefix, str):
        raise TypeError("`source_prefix` must be a string.")
    return {f"{source_prefix}{name}": shape for name, shape in native_wav2vec2_tensor_shapes(config).items()}


def safetensors_header_fingerprint(
    tensor_shapes: Mapping[str, tuple[int, ...]],
    *,
    dtype: str = "F32",
) -> str:
    """Fingerprint sorted names, dtype, and shapes from a header inventory."""
    if not isinstance(tensor_shapes, Mapping):
        raise TypeError("`tensor_shapes` must be a mapping.")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("`dtype` must be a non-empty string.")
    rows = []
    for name, shape in sorted(tensor_shapes.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("Tensor names must be non-empty strings.")
        if (not isinstance(shape, tuple) or
                any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
                    for dimension in shape)):
            raise ValueError(f"Tensor {name!r} must have a non-negative integer shape.")
        encoded_shape = "x".join(str(dimension) for dimension in shape)
        rows.append(f"{name}|{dtype}|{encoded_shape}")
    canonical = "\n".join(rows).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class HuggingFaceWav2Vec2CheckpointAdapter(CheckpointAdapter):
    """Load official Wav2Vec2 CTC Safetensors into the native graph."""

    architecture_id = "wav2vec2"
    adapter_id = "huggingface-wav2vec2-ctc-safetensors"
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
        declares_wav2vec2 = (
            model_type == "wav2vec2" or any(str(name) == "Wav2Vec2ForCTC" for name in architectures))
        has_safetensors = any(
            path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json") for path in files)
        return declares_wav2vec2 and has_safetensors

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        source_prefix = config.get("_checkpoint_prefix", "")
        if not isinstance(source_prefix, str):
            raise TypeError("`_checkpoint_prefix` must be a string.")
        return TensorPlan(
            rules=tuple(
                CopyTensor(source, target) for source, target in huggingface_wav2vec2_tensor_mapping(
                    config,
                    source_prefix=source_prefix,
                )),
            ignored_source_patterns=(
                "*position_ids",
                "*wav2vec2.masked_spec_embed",
            ),
        )


class HuggingFaceWav2Vec2ClassificationCheckpointAdapter(CheckpointAdapter):
    """Load official Wav2Vec2 clip or frame classifiers."""

    architecture_id = "wav2vec2"
    adapter_id = "huggingface-wav2vec2-classification-safetensors"
    adapter_version = "1"

    _SEQUENCE_ARCHITECTURES = frozenset({
        "Wav2Vec2ForAudioClassification",
        "Wav2Vec2ForSequenceClassification",
    })
    _FRAME_ARCHITECTURES = frozenset({
        "Wav2Vec2ForAudioFrameClassification",
    })

    @classmethod
    def architecture_family(cls, config: Mapping[str, Any]) -> str:
        configured = str(config.get("_classification_family", "")).strip().lower()
        if configured in {"audio-classification", "sequence-classification"}:
            return "sequence-classification"
        if configured == "frame-classification":
            return "frame-classification"
        architectures = config.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        names = frozenset(str(name) for name in architectures)
        if names.intersection(cls._SEQUENCE_ARCHITECTURES):
            return "sequence-classification"
        if names.intersection(cls._FRAME_ARCHITECTURES):
            return "frame-classification"
        raise ValueError(
            "Wav2Vec2 classification checkpoints must declare a sequence- "
            "or frame-classification architecture.")

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        if str(config.get("model_type", "")).lower() != "wav2vec2":
            return False
        try:
            self.architecture_family(config)
        except ValueError:
            return False
        return any(
            path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json") for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        family = self.architecture_family(config)
        if family == "frame-classification":
            shapes = native_wav2vec2_frame_classification_tensor_shapes(config)
        else:
            shapes = native_wav2vec2_sequence_classification_tensor_shapes(config)
        source_prefix = config.get("_checkpoint_prefix", "")
        if not isinstance(source_prefix, str):
            raise TypeError("`_checkpoint_prefix` must be a string.")
        return TensorPlan(
            rules=tuple(CopyTensor(f"{source_prefix}{name}", name) for name in sorted(shapes)),
            ignored_source_patterns=(
                "*position_ids",
                "*wav2vec2.masked_spec_embed",
            ),
        )


HFWav2Vec2CheckpointAdapter = HuggingFaceWav2Vec2CheckpointAdapter

__all__ = [
    "FACEBOOK_WAV2VEC2_BASE_960H_HEADER_FINGERPRINT",
    "FACEBOOK_WAV2VEC2_BASE_960H_REVISION",
    "HFWav2Vec2CheckpointAdapter",
    "HuggingFaceWav2Vec2ClassificationCheckpointAdapter",
    "HuggingFaceWav2Vec2CheckpointAdapter",
    "TensorMapping",
    "TensorShapes",
    "huggingface_wav2vec2_tensor_mapping",
    "huggingface_wav2vec2_tensor_shapes",
    "native_wav2vec2_tensor_names",
    "native_wav2vec2_tensor_shapes",
    "native_wav2vec2_sequence_classification_tensor_shapes",
    "native_wav2vec2_frame_classification_tensor_shapes",
    "safetensors_header_fingerprint",
]
