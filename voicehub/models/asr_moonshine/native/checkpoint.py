"""Strict checkpoint mapping for official native Moonshine Safetensors."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing.adapters import CheckpointAdapter
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan
from voicehub.models.asr_moonshine.native.configuration import MoonshineConfig
from voicehub.models.asr_wav2vec2.native.checkpoint import safetensors_header_fingerprint

USEFULSENSORS_MOONSHINE_TINY_REVISION = ("390624ed33d594443aa4aa221f5b9f283b545b5a")
USEFULSENSORS_MOONSHINE_BASE_REVISION = ("7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf")
USEFULSENSORS_MOONSHINE_TINY_HEADER_FINGERPRINT = (
    "37c91ded4815d54a88d26dfe0d586f198d0ea8106a31c79cbbe1b52a7c6f9041")
USEFULSENSORS_MOONSHINE_BASE_HEADER_FINGERPRINT = (
    "b6a2fb3e03f6e8cf4227e526bc48eedc9b9d9668d195fb518e5ff19b4c26e6b1")
USEFULSENSORS_MOONSHINE_TINY_FILE_BYTES = 108_389_192
USEFULSENSORS_MOONSHINE_BASE_FILE_BYTES = 246_079_928

TensorShapes = dict[str, tuple[int, ...]]
TensorMapping = tuple[tuple[str, str], ...]


def native_moonshine_tensor_shapes(config: MoonshineConfig | Mapping[str, Any], ) -> TensorShapes:
    """Return every persistent tensor and shape in the native graph."""
    resolved = MoonshineConfig.coerce(config)
    hidden = resolved.hidden_size
    intermediate = resolved.intermediate_size
    shapes: TensorShapes = {
        "model.encoder.conv1.weight": (hidden, 1, 127),
        "model.encoder.conv2.weight": (2 * hidden, hidden, 7),
        "model.encoder.conv2.bias": (2 * hidden, ),
        "model.encoder.conv3.weight": (hidden, 2 * hidden, 3),
        "model.encoder.conv3.bias": (hidden, ),
        "model.encoder.groupnorm.weight": (hidden, ),
        "model.encoder.groupnorm.bias": (hidden, ),
        "model.encoder.layer_norm.weight": (hidden, ),
        "model.decoder.embed_tokens.weight": (
            resolved.vocab_size,
            hidden,
        ),
        "model.decoder.norm.weight": (hidden, ),
    }
    encoder_attention = (
        resolved.encoder_num_attention_heads,
        resolved.encoder_num_key_value_heads,
    )
    decoder_attention = (
        resolved.decoder_num_attention_heads,
        resolved.decoder_num_key_value_heads,
    )
    for index in range(resolved.encoder_num_hidden_layers):
        prefix = f"model.encoder.layers.{index}"
        attention_prefix = f"{prefix}.self_attn"
        _add_attention_shapes(
            shapes,
            prefix=attention_prefix,
            config=resolved,
            attention_heads=encoder_attention[0],
            key_value_heads=encoder_attention[1],
        )
        shapes.update({
            f"{prefix}.input_layernorm.weight": (hidden, ),
            f"{prefix}.post_attention_layernorm.weight": (hidden, ),
            f"{prefix}.mlp.fc1.weight": (intermediate, hidden),
            f"{prefix}.mlp.fc1.bias": (intermediate, ),
            f"{prefix}.mlp.fc2.weight": (hidden, intermediate),
            f"{prefix}.mlp.fc2.bias": (hidden, ),
        })
    for index in range(resolved.decoder_num_hidden_layers):
        prefix = f"model.decoder.layers.{index}"
        for attention_name in ("self_attn", "encoder_attn"):
            _add_attention_shapes(
                shapes,
                prefix=f"{prefix}.{attention_name}",
                config=resolved,
                attention_heads=decoder_attention[0],
                key_value_heads=decoder_attention[1],
            )
        shapes.update({
            f"{prefix}.input_layernorm.weight": (hidden, ),
            f"{prefix}.post_attention_layernorm.weight": (hidden, ),
            f"{prefix}.final_layernorm.weight": (hidden, ),
            f"{prefix}.mlp.fc1.weight": (2 * intermediate, hidden),
            f"{prefix}.mlp.fc1.bias": (2 * intermediate, ),
            f"{prefix}.mlp.fc2.weight": (hidden, intermediate),
            f"{prefix}.mlp.fc2.bias": (hidden, ),
        })
    return shapes


def _add_attention_shapes(
    shapes: TensorShapes,
    *,
    prefix: str,
    config: MoonshineConfig,
    attention_heads: int,
    key_value_heads: int,
) -> None:
    hidden = config.hidden_size
    head_dim = hidden // attention_heads
    projection_dimensions = {
        "q_proj": attention_heads * head_dim,
        "k_proj": key_value_heads * head_dim,
        "v_proj": key_value_heads * head_dim,
    }
    for name, output_dimension in projection_dimensions.items():
        shapes[f"{prefix}.{name}.weight"] = (output_dimension, hidden)
        if config.attention_bias:
            shapes[f"{prefix}.{name}.bias"] = (output_dimension, )
    shapes[f"{prefix}.o_proj.weight"] = (hidden, attention_heads * head_dim)


def native_moonshine_tensor_names(config: MoonshineConfig | Mapping[str, Any], ) -> tuple[str, ...]:
    return tuple(sorted(native_moonshine_tensor_shapes(config)))


def huggingface_moonshine_tensor_mapping(
    config: MoonshineConfig | Mapping[str, Any],
    *,
    source_prefix: str = "",
) -> TensorMapping:
    if not isinstance(source_prefix, str):
        raise TypeError("`source_prefix` must be a string.")
    return tuple((f"{source_prefix}{name}", name) for name in native_moonshine_tensor_names(config))


def huggingface_moonshine_tensor_shapes(
    config: MoonshineConfig | Mapping[str, Any],
    *,
    source_prefix: str = "",
) -> TensorShapes:
    return {
        source: native_moonshine_tensor_shapes(config)[target]
        for source, target in huggingface_moonshine_tensor_mapping(
            config,
            source_prefix=source_prefix,
        )
    }


class HuggingFaceMoonshineCheckpointAdapter(CheckpointAdapter):
    """Load published Moonshine Safetensors without external runtimes."""

    architecture_id = "moonshine"
    adapter_id = "huggingface-moonshine-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        model_type = str(config.get("model_type", "")).strip().lower()
        architectures = config.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        declares_moonshine = (
            model_type in {"asr_moonshine", "moonshine"} or
            any(str(name) == "MoonshineForConditionalGeneration" for name in architectures))
        safe = any(
            path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json") for path in files)
        return declares_moonshine and safe

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        source_prefix = config.get("_checkpoint_prefix", "")
        if not isinstance(source_prefix, str):
            raise TypeError("`_checkpoint_prefix` must be a string.")
        return TensorPlan(
            rules=tuple(
                CopyTensor(source, target) for source, target in huggingface_moonshine_tensor_mapping(
                    config,
                    source_prefix=source_prefix,
                )),
            ignored_source_patterns=("*position_ids", ),
        )


HFMoonshineCheckpointAdapter = HuggingFaceMoonshineCheckpointAdapter

__all__ = [
    "HFMoonshineCheckpointAdapter",
    "HuggingFaceMoonshineCheckpointAdapter",
    "TensorMapping",
    "TensorShapes",
    "USEFULSENSORS_MOONSHINE_BASE_FILE_BYTES",
    "USEFULSENSORS_MOONSHINE_BASE_HEADER_FINGERPRINT",
    "USEFULSENSORS_MOONSHINE_BASE_REVISION",
    "USEFULSENSORS_MOONSHINE_TINY_FILE_BYTES",
    "USEFULSENSORS_MOONSHINE_TINY_HEADER_FINGERPRINT",
    "USEFULSENSORS_MOONSHINE_TINY_REVISION",
    "huggingface_moonshine_tensor_mapping",
    "huggingface_moonshine_tensor_shapes",
    "native_moonshine_tensor_names",
    "native_moonshine_tensor_shapes",
    "safetensors_header_fingerprint",
]
