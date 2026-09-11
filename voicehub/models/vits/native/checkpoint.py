"""Strict Safetensors mapping for native VITS and MMS-TTS.

The canonical namespace is verified against the 762-tensor header of
``facebook/mms-tts-eng`` at immutable revision
``c71de0fe7204c83f1c10820a7d696d0b450048ba``.  Mapping is declarative
and imports neither Transformers nor Safetensors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing.adapters import CheckpointAdapter
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan
from voicehub.models.vits.native.configuration import VitsConfig

ORIGINAL_VITS_REVISION = "2e561ba58618d021b5b8323d3765880f7e0ecfdb"
TRANSFORMERS_VITS_REVISION = "ebea912f0bb6f9e28ad2df04acd9b4df035933a9"
FACEBOOK_MMS_TTS_ENG_REVISION = "c71de0fe7204c83f1c10820a7d696d0b450048ba"
FACEBOOK_MMS_TTS_ENG_HEADER_FINGERPRINT = ("3a4b98a214895e2e4dd27ee96f85c7b646d8e3d26942c3157840778e6d0ef424")

TensorMapping = tuple[tuple[str, str], ...]
TensorShapes = dict[str, tuple[int, ...]]


def _conv1d(
    shapes: TensorShapes,
    prefix: str,
    output_channels: int,
    input_channels: int,
    kernel_size: int,
    *,
    groups: int = 1,
    bias: bool = True,
) -> None:
    shapes[f"{prefix}.weight"] = (
        output_channels,
        input_channels // groups,
        kernel_size,
    )
    if bias:
        shapes[f"{prefix}.bias"] = (output_channels, )


def _weight_normalized_conv1d(
    shapes: TensorShapes,
    prefix: str,
    output_channels: int,
    input_channels: int,
    kernel_size: int,
) -> None:
    shapes[f"{prefix}.weight_g"] = (output_channels, 1, 1)
    shapes[f"{prefix}.weight_v"] = (
        output_channels,
        input_channels,
        kernel_size,
    )
    shapes[f"{prefix}.bias"] = (output_channels, )


def _linear(
    shapes: TensorShapes,
    prefix: str,
    output_features: int,
    input_features: int,
    *,
    bias: bool = True,
) -> None:
    shapes[f"{prefix}.weight"] = (output_features, input_features)
    if bias:
        shapes[f"{prefix}.bias"] = (output_features, )


def _layer_norm(
    shapes: TensorShapes,
    prefix: str,
    channels: int,
) -> None:
    shapes[f"{prefix}.weight"] = (channels, )
    shapes[f"{prefix}.bias"] = (channels, )


def _wavenet_shapes(
    shapes: TensorShapes,
    prefix: str,
    config: VitsConfig,
    *,
    layers: int,
) -> None:
    hidden = config.hidden_size
    if config.speaker_embedding_size:
        _weight_normalized_conv1d(
            shapes,
            f"{prefix}.cond_layer",
            2 * hidden * layers,
            config.speaker_embedding_size,
            1,
        )
    for index in range(layers):
        _weight_normalized_conv1d(
            shapes,
            f"{prefix}.in_layers.{index}",
            2 * hidden,
            hidden,
            config.wavenet_kernel_size,
        )
        output_channels = 2 * hidden if index < layers - 1 else hidden
        _weight_normalized_conv1d(
            shapes,
            f"{prefix}.res_skip_layers.{index}",
            output_channels,
            hidden,
            1,
        )


def _depth_separable_shapes(
    shapes: TensorShapes,
    prefix: str,
    config: VitsConfig,
) -> None:
    hidden = config.hidden_size
    kernel = config.duration_predictor_kernel_size
    for index in range(config.depth_separable_num_layers):
        _conv1d(
            shapes,
            f"{prefix}.convs_dilated.{index}",
            hidden,
            hidden,
            kernel,
            groups=hidden,
        )
        _conv1d(
            shapes,
            f"{prefix}.convs_pointwise.{index}",
            hidden,
            hidden,
            1,
        )
        _layer_norm(shapes, f"{prefix}.norms_1.{index}", hidden)
        _layer_norm(shapes, f"{prefix}.norms_2.{index}", hidden)


def _duration_flow_shapes(
    shapes: TensorShapes,
    prefix: str,
    config: VitsConfig,
) -> None:
    half_channels = config.depth_separable_channels // 2
    _conv1d(
        shapes,
        f"{prefix}.conv_pre",
        config.hidden_size,
        half_channels,
        1,
    )
    _depth_separable_shapes(shapes, f"{prefix}.conv_dds", config)
    _conv1d(
        shapes,
        f"{prefix}.conv_proj",
        half_channels * (config.duration_predictor_flow_bins * 3 - 1),
        config.hidden_size,
        1,
    )


def _stochastic_duration_shapes(
    shapes: TensorShapes,
    prefix: str,
    config: VitsConfig,
) -> None:
    hidden = config.hidden_size
    _conv1d(shapes, f"{prefix}.conv_pre", hidden, hidden, 1)
    _conv1d(shapes, f"{prefix}.conv_proj", hidden, hidden, 1)
    _depth_separable_shapes(shapes, f"{prefix}.conv_dds", config)
    if config.speaker_embedding_size:
        _conv1d(
            shapes,
            f"{prefix}.cond",
            hidden,
            config.speaker_embedding_size,
            1,
        )
    channels = config.depth_separable_channels
    shapes[f"{prefix}.flows.0.translate"] = (channels, 1)
    shapes[f"{prefix}.flows.0.log_scale"] = (channels, 1)
    for index in range(1, config.duration_predictor_num_flows + 1):
        _duration_flow_shapes(
            shapes,
            f"{prefix}.flows.{index}",
            config,
        )

    _conv1d(shapes, f"{prefix}.post_conv_pre", hidden, 1, 1)
    _conv1d(shapes, f"{prefix}.post_conv_proj", hidden, hidden, 1)
    _depth_separable_shapes(shapes, f"{prefix}.post_conv_dds", config)
    shapes[f"{prefix}.post_flows.0.translate"] = (channels, 1)
    shapes[f"{prefix}.post_flows.0.log_scale"] = (channels, 1)
    for index in range(1, config.duration_predictor_num_flows + 1):
        _duration_flow_shapes(
            shapes,
            f"{prefix}.post_flows.{index}",
            config,
        )


def _deterministic_duration_shapes(
    shapes: TensorShapes,
    prefix: str,
    config: VitsConfig,
) -> None:
    filters = config.duration_predictor_filter_channels
    _conv1d(
        shapes,
        f"{prefix}.conv_1",
        filters,
        config.hidden_size,
        config.duration_predictor_kernel_size,
    )
    _layer_norm(shapes, f"{prefix}.norm_1", filters)
    _conv1d(
        shapes,
        f"{prefix}.conv_2",
        filters,
        filters,
        config.duration_predictor_kernel_size,
    )
    _layer_norm(shapes, f"{prefix}.norm_2", filters)
    _conv1d(shapes, f"{prefix}.proj", 1, filters, 1)
    if config.speaker_embedding_size:
        _conv1d(
            shapes,
            f"{prefix}.cond",
            config.hidden_size,
            config.speaker_embedding_size,
            1,
        )


def native_vits_tensor_shapes(config: VitsConfig | Mapping[str, Any], ) -> TensorShapes:
    """Return every native generator tensor and its checkpoint shape."""
    resolved = VitsConfig.coerce(config)
    shapes: TensorShapes = {
        "text_encoder.embed_tokens.weight": (
            resolved.vocab_size,
            resolved.hidden_size,
        ),
    }
    hidden = resolved.hidden_size
    for index in range(resolved.num_hidden_layers):
        prefix = f"text_encoder.encoder.layers.{index}"
        if resolved.window_size:
            relative_length = resolved.window_size * 2 + 1
            relative_shape = (
                1,
                relative_length,
                hidden // resolved.num_attention_heads,
            )
            shapes[f"{prefix}.attention.emb_rel_k"] = relative_shape
            shapes[f"{prefix}.attention.emb_rel_v"] = relative_shape
        for projection in ("k_proj", "v_proj", "q_proj", "out_proj"):
            _linear(
                shapes,
                f"{prefix}.attention.{projection}",
                hidden,
                hidden,
                bias=resolved.use_bias,
            )
        _layer_norm(shapes, f"{prefix}.layer_norm", hidden)
        _conv1d(
            shapes,
            f"{prefix}.feed_forward.conv_1",
            resolved.ffn_dim,
            hidden,
            resolved.ffn_kernel_size,
        )
        _conv1d(
            shapes,
            f"{prefix}.feed_forward.conv_2",
            hidden,
            resolved.ffn_dim,
            resolved.ffn_kernel_size,
        )
        _layer_norm(shapes, f"{prefix}.final_layer_norm", hidden)
    _conv1d(
        shapes,
        "text_encoder.project",
        resolved.flow_size * 2,
        hidden,
        1,
    )

    half_flow = resolved.flow_size // 2
    for index in range(resolved.prior_encoder_num_flows):
        prefix = f"flow.flows.{index}"
        _conv1d(shapes, f"{prefix}.conv_pre", hidden, half_flow, 1)
        _wavenet_shapes(
            shapes,
            f"{prefix}.wavenet",
            resolved,
            layers=resolved.prior_encoder_num_wavenet_layers,
        )
        _conv1d(shapes, f"{prefix}.conv_post", half_flow, hidden, 1)

    _conv1d(
        shapes,
        "decoder.conv_pre",
        resolved.upsample_initial_channel,
        resolved.flow_size,
        7,
    )
    decoder_channels = resolved.upsample_initial_channel
    residual_index = 0
    for stage, kernel_size in enumerate(resolved.upsample_kernel_sizes):
        output_channels = resolved.upsample_initial_channel // (2**(stage + 1))
        shapes[f"decoder.upsampler.{stage}.weight"] = (
            decoder_channels,
            output_channels,
            kernel_size,
        )
        shapes[f"decoder.upsampler.{stage}.bias"] = (output_channels, )
        for residual_kernel, dilations in zip(
                resolved.resblock_kernel_sizes,
                resolved.resblock_dilation_sizes,
        ):
            for layer, _ in enumerate(dilations):
                _conv1d(
                    shapes,
                    f"decoder.resblocks.{residual_index}.convs1.{layer}",
                    output_channels,
                    output_channels,
                    residual_kernel,
                )
                _conv1d(
                    shapes,
                    f"decoder.resblocks.{residual_index}.convs2.{layer}",
                    output_channels,
                    output_channels,
                    residual_kernel,
                )
            residual_index += 1
        decoder_channels = output_channels
    _conv1d(
        shapes,
        "decoder.conv_post",
        1,
        decoder_channels,
        7,
        bias=False,
    )
    if resolved.speaker_embedding_size:
        _conv1d(
            shapes,
            "decoder.cond",
            resolved.upsample_initial_channel,
            resolved.speaker_embedding_size,
            1,
        )

    if resolved.use_stochastic_duration_prediction:
        _stochastic_duration_shapes(
            shapes,
            "duration_predictor",
            resolved,
        )
    else:
        _deterministic_duration_shapes(
            shapes,
            "duration_predictor",
            resolved,
        )
    if resolved.is_multispeaker:
        shapes["embed_speaker.weight"] = (
            resolved.num_speakers,
            resolved.speaker_embedding_size,
        )

    _conv1d(
        shapes,
        "posterior_encoder.conv_pre",
        hidden,
        resolved.spectrogram_bins,
        1,
    )
    _wavenet_shapes(
        shapes,
        "posterior_encoder.wavenet",
        resolved,
        layers=resolved.posterior_encoder_num_wavenet_layers,
    )
    _conv1d(
        shapes,
        "posterior_encoder.conv_proj",
        resolved.flow_size * 2,
        hidden,
        1,
    )
    return shapes


def native_vits_tensor_names(config: VitsConfig | Mapping[str, Any], ) -> tuple[str, ...]:
    """Return all canonical generator tensor names in sorted order."""
    return tuple(sorted(native_vits_tensor_shapes(config)))


def huggingface_vits_tensor_mapping(
    config: VitsConfig | Mapping[str, Any],
    *,
    source_prefix: str = "",
) -> TensorMapping:
    """Map the official Hugging Face VITS namespace to VoiceHub."""
    if not isinstance(source_prefix, str):
        raise TypeError("`source_prefix` must be a string.")
    return tuple((f"{source_prefix}{name}", name) for name in native_vits_tensor_names(config))


def huggingface_vits_tensor_shapes(
    config: VitsConfig | Mapping[str, Any],
    *,
    source_prefix: str = "",
) -> TensorShapes:
    """Return the complete expected upstream VITS tensor inventory."""
    if not isinstance(source_prefix, str):
        raise TypeError("`source_prefix` must be a string.")
    return {f"{source_prefix}{name}": shape for name, shape in native_vits_tensor_shapes(config).items()}


def safetensors_header_fingerprint(
    tensor_shapes: Mapping[str, tuple[int, ...]],
    *,
    dtype: str = "F32",
) -> str:
    """Hash canonical ``name|dtype|shape`` rows from a tensor header."""
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
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


class HuggingFaceVitsCheckpointAdapter(CheckpointAdapter):
    """Load official VITS/MMS-TTS Safetensors into the native graph."""

    architecture_id = "vits"
    adapter_id = "huggingface-vits-safetensors"
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
        declares_vits = (model_type == "vits" or any("vits" in str(name).lower() for name in architectures))
        has_safetensors = any(
            path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json") for path in files)
        return declares_vits and has_safetensors

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        source_prefix = config.get("_checkpoint_prefix", "")
        if not isinstance(source_prefix, str):
            raise TypeError("`_checkpoint_prefix` must be a string.")
        return TensorPlan(
            rules=tuple(
                CopyTensor(source, target) for source, target in huggingface_vits_tensor_mapping(
                    config,
                    source_prefix=source_prefix,
                )), )


class NativeVitsCheckpointAdapter(CheckpointAdapter):
    """Load VoiceHub's canonical native VITS Safetensors namespace."""

    architecture_id = "vits"
    adapter_id = "voicehub-vits-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        return (
            config.get("voicehub_checkpoint_format") == "native-vits-v1" and any(
                path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json")
                for path in files))

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in native_vits_tensor_names(config)), )


HFVitsCheckpointAdapter = HuggingFaceVitsCheckpointAdapter

__all__ = [
    "FACEBOOK_MMS_TTS_ENG_HEADER_FINGERPRINT",
    "FACEBOOK_MMS_TTS_ENG_REVISION",
    "HFVitsCheckpointAdapter",
    "HuggingFaceVitsCheckpointAdapter",
    "NativeVitsCheckpointAdapter",
    "ORIGINAL_VITS_REVISION",
    "TRANSFORMERS_VITS_REVISION",
    "TensorMapping",
    "TensorShapes",
    "huggingface_vits_tensor_mapping",
    "huggingface_vits_tensor_shapes",
    "native_vits_tensor_names",
    "native_vits_tensor_shapes",
    "safetensors_header_fingerprint",
]
