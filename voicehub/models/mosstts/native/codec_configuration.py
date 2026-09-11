# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native MOSS Audio Tokenizer model configuration.

The upstream checkpoints publish Transformers configuration files, but
the codec architecture itself does not need Transformers.  This module
preserves the checkpoint schema in a small, dependency-free
configuration object.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def _legacy_transformer(
    input_dimension: int,
    output_dimension: int,
    *,
    model_dimension: int,
    heads: int,
    layers: int,
    feedforward_dimension: int,
) -> dict[str, Any]:
    return {
        "module_type": "Transformer",
        "input_dimension": input_dimension,
        "output_dimension": output_dimension,
        "d_model": model_dimension,
        "num_heads": heads,
        "num_layers": layers,
        "dim_feedforward": feedforward_dimension,
        "causal": True,
        "norm": "layer_norm",
        "positional_embedding": "rope",
        "max_period": 10_000,
        "gating": "none",
        "layer_scale": 0.01,
        "conv_layout": True,
    }


def _v1_encoder_defaults() -> list[dict[str, Any]]:
    return [
        {
            "module_type": "PatchedPretransform",
            "patch_size": 240,
        },
        _legacy_transformer(
            240,
            384,
            model_dimension=768,
            heads=12,
            layers=12,
            feedforward_dimension=3_072,
        ),
        {
            "module_type": "PatchedPretransform",
            "patch_size": 2,
        },
        _legacy_transformer(
            768,
            384,
            model_dimension=768,
            heads=12,
            layers=12,
            feedforward_dimension=3_072,
        ),
        {
            "module_type": "PatchedPretransform",
            "patch_size": 2,
        },
        _legacy_transformer(
            768,
            640,
            model_dimension=768,
            heads=12,
            layers=12,
            feedforward_dimension=3_072,
        ),
        {
            "module_type": "PatchedPretransform",
            "patch_size": 2,
        },
        _legacy_transformer(
            1_280,
            768,
            model_dimension=1_280,
            heads=20,
            layers=32,
            feedforward_dimension=5_120,
        ),
    ]


def _v1_decoder_defaults() -> list[dict[str, Any]]:
    return [
        _legacy_transformer(
            768,
            1_280,
            model_dimension=1_280,
            heads=20,
            layers=32,
            feedforward_dimension=5_120,
        ),
        {
            "module_type": "PatchedPretransform",
            "patch_size": 2,
        },
        _legacy_transformer(
            640,
            768,
            model_dimension=768,
            heads=12,
            layers=12,
            feedforward_dimension=3_072,
        ),
        {
            "module_type": "PatchedPretransform",
            "patch_size": 2,
        },
        _legacy_transformer(
            384,
            768,
            model_dimension=768,
            heads=12,
            layers=12,
            feedforward_dimension=3_072,
        ),
        {
            "module_type": "PatchedPretransform",
            "patch_size": 2,
        },
        _legacy_transformer(
            384,
            240,
            model_dimension=768,
            heads=12,
            layers=12,
            feedforward_dimension=3_072,
        ),
        {
            "module_type": "PatchedPretransform",
            "patch_size": 240,
        },
    ]


class MossAudioTokenizerConfig:
    r"""
    This is the configuration class to store the configuration of a [`MossAudioTokenizerModel`]. It is used to instantiate a
    MossAudioTokenizer model according to the specified arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [OpenMOSS-Team/MOSS-Audio-Tokenizer-v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        sampling_rate (`int`, *optional*, defaults to 48000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        downsample_rate (`int`, *optional*, defaults to 3840):
            Total downsampling rate from waveform to tokens.
        causal_transformer_context_duration (`float`, *optional*, defaults to 10.0):
            Legacy global fallback context duration in seconds for causal transformer. If an individual transformer
            entry in `encoder_kwargs` or `decoder_kwargs` provides `context_duration`, that per-module value takes
            precedence.
        encoder_kwargs (`list[dict]`, *optional*):
            List of encoder module configurations. Each dict specifies a module type and its parameters.
        decoder_kwargs (`list[dict]`, *optional*):
            List of decoder module configurations in execution order.
        number_channels (`int`, *optional*, defaults to 2):
            Number of audio channels exposed by the public waveform interface.
        enable_channel_interleave (`bool`, *optional*, defaults to `True`):
            Whether to flatten multi-channel waveforms into a single internal stream before codec inference.
        attention_implementation (`str`, *optional*, defaults to `"sdpa"`):
            Checkpoint attention preference. VoiceHub executes the graph with
            native PyTorch SDPA; published `"flash_attention_2"` values are
            normalized without changing learned parameters.
        compute_dtype (`str`, *optional*, defaults to `"fp32"`):
            Inference compute dtype for non-quantizer modules. Supported values are `"fp32"`, `"bf16"`.
        codec_weight_dtype (`str`, *optional*, defaults to `"fp32"`):
            Parameter dtype for encoder and decoder modules. The quantizer remains fp32 because it explicitly disables
            autocast and performs numerically sensitive codebook operations in fp32.
        quantizer_type (`str`, *optional*, defaults to `"rlfq"`):
            Quantizer type. Options include `"rvq"`, `"spec_rvq"`, `"rlfq"`, `"random_prefix_rlfq"`.
        quantizer_kwargs (`dict`, *optional*):
            Configuration for the quantizer including `input_dim`, `rvq_dim`, `output_dim`, `num_quantizers`,
            `codebook_size`, and `codebook_dim`.

    Example:

    ```python
    >>> from voicehub.models.mosstts.native import MossAudioTokenizerConfig
    >>> from voicehub.models.mosstts.native import MossAudioTokenizerV2Model

    >>> # Initializing a MossAudioTokenizer style configuration
    >>> configuration = MossAudioTokenizerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MossAudioTokenizerV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "moss-audio-tokenizer"

    # Backward-compatible alias used by some checkpoints.
    attribute_map = {"sample_rate": "sampling_rate"}

    sampling_rate: int
    downsample_rate: int
    causal_transformer_context_duration: float
    encoder_kwargs: list[dict[str, Any]]
    decoder_kwargs: list[dict[str, Any]]
    number_channels: int
    enable_channel_interleave: bool
    attention_implementation: str
    compute_dtype: str
    codec_weight_dtype: str
    quantizer_type: str
    quantizer_kwargs: dict[str, Any]

    def __init__(
        self,
        version: str | None = None,
        sampling_rate: int = 48000,
        downsample_rate: int = 3840,
        causal_transformer_context_duration: float = 10.0,
        encoder_kwargs: list[dict[str, Any]] | None = None,
        decoder_kwargs: list[dict[str, Any]] | None = None,
        number_channels: int | None = None,
        enable_channel_interleave: bool = True,
        attention_implementation: str = "sdpa",
        compute_dtype: str = "fp32",
        codec_weight_dtype: str = "fp32",
        quantizer_type: str = "rlfq",
        quantizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        # Some checkpoints might include an incorrect/legacy `model_type` (e.g. "speech_tokenizer").
        # We drop it to avoid overriding the class-level `model_type`.
        kwargs.pop("model_type", None)
        if "channels_numbers" in kwargs:
            number_channels = kwargs.pop("channels_numbers")
        if "enable_channel_interleave" in kwargs:
            enable_channel_interleave = kwargs.pop("enable_channel_interleave")
        if "attention_backend" in kwargs and attention_implementation == "sdpa":
            attention_implementation = kwargs.pop("attention_backend")
        if "codec_compute_dtype" in kwargs and compute_dtype == "fp32":
            compute_dtype = kwargs.pop("codec_compute_dtype")
        if "codec_load_dtype" in kwargs and codec_weight_dtype == "fp32":
            codec_weight_dtype = kwargs.pop("codec_load_dtype")
        reversed_decoder_kwargs = kwargs.pop("reversed_decoder_kwargs", None)

        # `version` is accepted for compatibility but not used in modeling.
        self.version = version
        self.sampling_rate = sampling_rate
        self.downsample_rate = downsample_rate
        self.causal_transformer_context_duration = causal_transformer_context_duration
        self.number_channels = (
            int(number_channels) if number_channels is not None else
            (1 if sampling_rate == 24_000 and downsample_rate == 1_920 else 2))
        self.enable_channel_interleave = enable_channel_interleave
        # Published v2 configs request FlashAttention 2.  The operation has
        # the same learned parameters as PyTorch SDPA, so native VoiceHub
        # normalizes that optional-backend preference at the configuration
        # boundary instead of importing an external kernel package.
        self.attention_implementation = (
            "sdpa" if attention_implementation == "flash_attention_2" else attention_implementation)
        self.compute_dtype = compute_dtype
        self.codec_weight_dtype = codec_weight_dtype
        is_v1 = (sampling_rate == 24_000 and downsample_rate == 1_920 and self.number_channels == 1)

        # Default encoder configuration
        if encoder_kwargs is None:
            encoder_kwargs = _v1_encoder_defaults() if is_v1 else [
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 240,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 240,
                    "output_dimension": 384,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 1.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 768,
                    "output_dimension": 384,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 2.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 768,
                    "output_dimension": 384,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 4.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 768,
                    "output_dimension": 384,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 8.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 768,
                    "output_dimension": 640,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 10.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 1280,
                    "output_dimension": 768,
                    "d_model": 1280,
                    "num_heads": 20,
                    "num_layers": 32,
                    "dim_feedforward": 5120,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 10.0,
                },
            ]
        else:
            encoder_kwargs = [dict(module_kwargs) for module_kwargs in encoder_kwargs]
        uses_module_context = any(
            "context_duration" in module_kwargs for module_kwargs in encoder_kwargs
            if module_kwargs.get("module_type") == "Transformer")
        for module_kwargs in encoder_kwargs:
            if (uses_module_context and module_kwargs.get("module_type") == "Transformer"):
                module_kwargs.setdefault("context_duration", causal_transformer_context_duration)
        self.encoder_kwargs = encoder_kwargs

        # Default decoder configuration (execution order)
        if decoder_kwargs is None and reversed_decoder_kwargs is not None:
            reversed_decoder_kwargs = [dict(module_kwargs) for module_kwargs in reversed_decoder_kwargs]
            decoder_kwargs = []
            for module_kwargs in reversed_decoder_kwargs[::-1]:
                if module_kwargs.get("module_type") != "Transformer":
                    decoder_kwargs.append(module_kwargs)
                    continue
                module_kwargs = dict(module_kwargs)
                module_kwargs["input_dimension"], module_kwargs["output_dimension"] = (
                    module_kwargs["output_dimension"],
                    module_kwargs["input_dimension"],
                )
                decoder_kwargs.append(module_kwargs)

        if decoder_kwargs is None:
            decoder_kwargs = _v1_decoder_defaults() if is_v1 else [
                {
                    "module_type": "Transformer",
                    "input_dimension": 768,
                    "output_dimension": 1280,
                    "d_model": 1280,
                    "num_heads": 20,
                    "num_layers": 32,
                    "dim_feedforward": 5120,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 10.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 640,
                    "output_dimension": 768,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 10.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 384,
                    "output_dimension": 768,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 8.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 384,
                    "output_dimension": 768,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 4.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 384,
                    "output_dimension": 768,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 2.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 384,
                    "output_dimension": 240,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 1.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 240,
                },
            ]
        else:
            decoder_kwargs = [dict(module_kwargs) for module_kwargs in decoder_kwargs]
        for module_kwargs in decoder_kwargs:
            if (uses_module_context and module_kwargs.get("module_type") == "Transformer"):
                module_kwargs.setdefault("context_duration", causal_transformer_context_duration)
        self.decoder_kwargs = decoder_kwargs

        # Default quantizer configuration
        if quantizer_kwargs is None:
            quantizer_kwargs = {
                "input_dim": 768,
                "rvq_dim": 512,
                "output_dim": 768,
                "num_quantizers": 32,
                "codebook_size": 1024,
                "codebook_dim": 8,
                "quantizer_type": "rlfq",
            }

        # Handle quantizer_type from kwargs or config
        kw_qtype = quantizer_kwargs.get("quantizer_type", None)
        if kw_qtype is not None:
            self.quantizer_type = kw_qtype
        else:
            self.quantizer_type = quantizer_type
            quantizer_kwargs["quantizer_type"] = quantizer_type

        self.quantizer_kwargs = dict(quantizer_kwargs)
        self.return_dict = bool(kwargs.pop("return_dict", True))
        self.extra_config = copy.deepcopy(kwargs)
        self._validate()

    def _validate(self) -> None:
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be greater than zero")
        if self.downsample_rate <= 0:
            raise ValueError("downsample_rate must be greater than zero")
        if self.number_channels <= 0:
            raise ValueError("number_channels must be greater than zero")
        if self.attention_implementation != "sdpa":
            raise ValueError(
                "The native MOSS codec supports attention_implementation='sdpa' "
                "only; flash-attn is intentionally not a runtime dependency.")
        if self.compute_dtype not in {"fp32", "float32", "bf16", "bfloat16"}:
            raise ValueError("compute_dtype must be one of: fp32, float32, bf16, bfloat16")
        if self.codec_weight_dtype not in {
                "fp32",
                "float32",
                "bf16",
                "bfloat16",
        }:
            raise ValueError("codec_weight_dtype must be one of: fp32, float32, bf16, bfloat16")
        if self.num_quantizers <= 0:
            raise ValueError("quantizer_kwargs.num_quantizers must be greater than zero")
        if self.codebook_size <= 1:
            raise ValueError("quantizer_kwargs.codebook_size must be greater than one")

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        **overrides: Any,
    ) -> MossAudioTokenizerConfig:
        """Build a configuration from an official checkpoint mapping."""
        if not isinstance(values, Mapping):
            raise TypeError("Codec configuration must be a mapping")
        payload = copy.deepcopy(dict(values))
        payload.update(copy.deepcopy(overrides))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """Return the lossless JSON-compatible checkpoint configuration."""
        values = {
            "model_type": self.model_type,
            "version": self.version,
            "sampling_rate": self.sampling_rate,
            "downsample_rate": self.downsample_rate,
            "causal_transformer_context_duration": (self.causal_transformer_context_duration),
            "encoder_kwargs": self.encoder_kwargs,
            "decoder_kwargs": self.decoder_kwargs,
            "number_channels": self.number_channels,
            "enable_channel_interleave": self.enable_channel_interleave,
            "attention_implementation": self.attention_implementation,
            "compute_dtype": self.compute_dtype,
            "codec_weight_dtype": self.codec_weight_dtype,
            "code_dim": int(self.quantizer_kwargs["input_dim"]),
            "quantizer_type": self.quantizer_type,
            "quantizer_kwargs": self.quantizer_kwargs,
            "return_dict": self.return_dict,
        }
        values.update(self.extra_config)
        return copy.deepcopy(values)

    @property
    def sample_rate(self) -> int:
        """Backward-compatible alias used by v1 checkpoints."""
        return self.sampling_rate

    @property
    def num_quantizers(self) -> int:
        """Return the number of quantizers from quantizer_kwargs."""
        return self.quantizer_kwargs.get("num_quantizers", 32)

    @property
    def codebook_size(self) -> int:
        """Return the codebook size from quantizer_kwargs."""
        return self.quantizer_kwargs.get("codebook_size", 4096)

    @property
    def frame_rate(self) -> float:
        """Return the frame rate (tokens per second)."""
        return self.sampling_rate / self.downsample_rate


__all__ = ["MossAudioTokenizerConfig"]
