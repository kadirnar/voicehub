"""Strict checkpoint conversion for VoiceHub's native Whisper graph.

Mappings cover OpenAI Whisper checkpoints at revision
``04f449b8a437f1bbd3dba5c9f826aca972e7709a`` and Hugging Face Transformers
Whisper Safetensors at revision
``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``.  Conversion is expressed only
through VoiceHub checkpoint rules; neither upstream package is imported.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing.adapters import CheckpointAdapter
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan
from voicehub.models.asr_whisper_native.native.configuration import WhisperConfig

TensorMapping = tuple[tuple[str, str], ...]


def _attention_names(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}.query.weight",
        f"{prefix}.query.bias",
        f"{prefix}.key.weight",
        f"{prefix}.value.weight",
        f"{prefix}.value.bias",
        f"{prefix}.out.weight",
        f"{prefix}.out.bias",
    )


def native_whisper_tensor_names(config: WhisperConfig | Mapping[str, Any], ) -> tuple[str, ...]:
    """Return every persistent tensor in the native model namespace."""
    resolved = WhisperConfig.coerce(config)
    names = [
        "encoder.conv1.weight",
        "encoder.conv1.bias",
        "encoder.conv2.weight",
        "encoder.conv2.bias",
        "encoder.positional_embedding",
    ]
    for index in range(resolved.encoder_layers):
        prefix = f"encoder.blocks.{index}"
        names.extend(_attention_names(f"{prefix}.attn"))
        names.extend((
            f"{prefix}.attn_ln.weight",
            f"{prefix}.attn_ln.bias",
            f"{prefix}.mlp.0.weight",
            f"{prefix}.mlp.0.bias",
            f"{prefix}.mlp.2.weight",
            f"{prefix}.mlp.2.bias",
            f"{prefix}.mlp_ln.weight",
            f"{prefix}.mlp_ln.bias",
        ))
    names.extend(("encoder.ln_post.weight", "encoder.ln_post.bias"))

    names.extend((
        "decoder.token_embedding.weight",
        "decoder.positional_embedding",
    ))
    for index in range(resolved.decoder_layers):
        prefix = f"decoder.blocks.{index}"
        names.extend(_attention_names(f"{prefix}.attn"))
        names.extend((
            f"{prefix}.attn_ln.weight",
            f"{prefix}.attn_ln.bias",
        ))
        names.extend(_attention_names(f"{prefix}.cross_attn"))
        names.extend((
            f"{prefix}.cross_attn_ln.weight",
            f"{prefix}.cross_attn_ln.bias",
            f"{prefix}.mlp.0.weight",
            f"{prefix}.mlp.0.bias",
            f"{prefix}.mlp.2.weight",
            f"{prefix}.mlp.2.bias",
            f"{prefix}.mlp_ln.weight",
            f"{prefix}.mlp_ln.bias",
        ))
    names.extend(("decoder.ln.weight", "decoder.ln.bias"))
    return tuple(names)


def openai_whisper_tensor_mapping(config: WhisperConfig | Mapping[str, Any], ) -> TensorMapping:
    """Map OpenAI's ``model_state_dict`` namespace to VoiceHub."""
    return tuple((name, name) for name in native_whisper_tensor_names(config))


def _hf_attention_mapping(
    source_prefix: str,
    target_prefix: str,
) -> list[tuple[str, str]]:
    projection_names = (
        ("q_proj.weight", "query.weight"),
        ("q_proj.bias", "query.bias"),
        ("k_proj.weight", "key.weight"),
        ("v_proj.weight", "value.weight"),
        ("v_proj.bias", "value.bias"),
        ("out_proj.weight", "out.weight"),
        ("out_proj.bias", "out.bias"),
    )
    return [(f"{source_prefix}.{source}", f"{target_prefix}.{target}") for source, target in projection_names]


def huggingface_whisper_tensor_mapping(
    config: WhisperConfig | Mapping[str, Any],
    *,
    source_prefix: str = "model.",
) -> TensorMapping:
    """Map official Hugging Face Whisper Safetensors to VoiceHub.

    ``source_prefix`` is configurable for bare ``WhisperModel``
    artifacts; conditional-generation checkpoints use the default
    ``"model."`` prefix.
    """
    resolved = WhisperConfig.coerce(config)
    if not isinstance(source_prefix, str):
        raise TypeError("`source_prefix` must be a string.")
    mapping: list[tuple[str, str]] = [
        (
            f"{source_prefix}encoder.conv1.weight",
            "encoder.conv1.weight",
        ),
        (
            f"{source_prefix}encoder.conv1.bias",
            "encoder.conv1.bias",
        ),
        (
            f"{source_prefix}encoder.conv2.weight",
            "encoder.conv2.weight",
        ),
        (
            f"{source_prefix}encoder.conv2.bias",
            "encoder.conv2.bias",
        ),
        (
            f"{source_prefix}encoder.embed_positions.weight",
            "encoder.positional_embedding",
        ),
    ]

    for index in range(resolved.encoder_layers):
        source = f"{source_prefix}encoder.layers.{index}"
        target = f"encoder.blocks.{index}"
        mapping.extend(_hf_attention_mapping(
            f"{source}.self_attn",
            f"{target}.attn",
        ))
        mapping.extend((
            (
                f"{source}.self_attn_layer_norm.weight",
                f"{target}.attn_ln.weight",
            ),
            (
                f"{source}.self_attn_layer_norm.bias",
                f"{target}.attn_ln.bias",
            ),
            (f"{source}.fc1.weight", f"{target}.mlp.0.weight"),
            (f"{source}.fc1.bias", f"{target}.mlp.0.bias"),
            (f"{source}.fc2.weight", f"{target}.mlp.2.weight"),
            (f"{source}.fc2.bias", f"{target}.mlp.2.bias"),
            (
                f"{source}.final_layer_norm.weight",
                f"{target}.mlp_ln.weight",
            ),
            (
                f"{source}.final_layer_norm.bias",
                f"{target}.mlp_ln.bias",
            ),
        ))
    mapping.extend((
        (
            f"{source_prefix}encoder.layer_norm.weight",
            "encoder.ln_post.weight",
        ),
        (
            f"{source_prefix}encoder.layer_norm.bias",
            "encoder.ln_post.bias",
        ),
        (
            f"{source_prefix}decoder.embed_tokens.weight",
            "decoder.token_embedding.weight",
        ),
        (
            f"{source_prefix}decoder.embed_positions.weight",
            "decoder.positional_embedding",
        ),
    ))

    for index in range(resolved.decoder_layers):
        source = f"{source_prefix}decoder.layers.{index}"
        target = f"decoder.blocks.{index}"
        mapping.extend(_hf_attention_mapping(
            f"{source}.self_attn",
            f"{target}.attn",
        ))
        mapping.extend((
            (
                f"{source}.self_attn_layer_norm.weight",
                f"{target}.attn_ln.weight",
            ),
            (
                f"{source}.self_attn_layer_norm.bias",
                f"{target}.attn_ln.bias",
            ),
        ))
        mapping.extend(_hf_attention_mapping(
            f"{source}.encoder_attn",
            f"{target}.cross_attn",
        ))
        mapping.extend((
            (
                f"{source}.encoder_attn_layer_norm.weight",
                f"{target}.cross_attn_ln.weight",
            ),
            (
                f"{source}.encoder_attn_layer_norm.bias",
                f"{target}.cross_attn_ln.bias",
            ),
            (f"{source}.fc1.weight", f"{target}.mlp.0.weight"),
            (f"{source}.fc1.bias", f"{target}.mlp.0.bias"),
            (f"{source}.fc2.weight", f"{target}.mlp.2.weight"),
            (f"{source}.fc2.bias", f"{target}.mlp.2.bias"),
            (
                f"{source}.final_layer_norm.weight",
                f"{target}.mlp_ln.weight",
            ),
            (
                f"{source}.final_layer_norm.bias",
                f"{target}.mlp_ln.bias",
            ),
        ))
    mapping.extend((
        (
            f"{source_prefix}decoder.layer_norm.weight",
            "decoder.ln.weight",
        ),
        (
            f"{source_prefix}decoder.layer_norm.bias",
            "decoder.ln.bias",
        ),
    ))
    return tuple(mapping)


class OpenAIWhisperCheckpointAdapter(CheckpointAdapter):
    """Load OpenAI ``.pt`` model-state tensors into the native graph."""

    architecture_id = "whisper"
    adapter_id = "openai-whisper"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        has_dimensions = all(
            name in config for name in (
                "n_mels",
                "n_audio_ctx",
                "n_audio_state",
                "n_audio_head",
                "n_audio_layer",
                "n_vocab",
                "n_text_ctx",
                "n_text_state",
                "n_text_head",
                "n_text_layer",
            ))
        return has_dimensions and any(path.suffix == ".pt" for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        return TensorPlan(
            rules=tuple(
                CopyTensor(source, target) for source, target in openai_whisper_tensor_mapping(config)), )


class HuggingFaceWhisperCheckpointAdapter(CheckpointAdapter):
    """Load official Hugging Face Whisper Safetensors without Transformers."""

    architecture_id = "whisper"
    adapter_id = "huggingface-whisper-safetensors"
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
        declares_whisper = (
            model_type == "whisper" or any("whisper" in str(name).lower() for name in architectures))
        has_safetensors = any(
            path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json") for path in files)
        return declares_whisper and has_safetensors

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        source_prefix = config.get("_checkpoint_prefix")
        if source_prefix is None:
            architectures = config.get('architectures', ())
            if isinstance(architectures, str):
                architectures = (architectures, )
            bare_model = any(str(name) == "WhisperModel" for name in architectures)
            source_prefix = "" if bare_model else "model."
        source_prefix = str(source_prefix)
        return TensorPlan(
            rules=tuple(
                CopyTensor(source, target) for source, target in huggingface_whisper_tensor_mapping(
                    config,
                    source_prefix=source_prefix,
                )),
            ignored_source_patterns=(
                "proj_out.weight",
                "*position_ids",
            ),
        )


class NativeWhisperCheckpointAdapter(CheckpointAdapter):
    """Load VoiceHub's canonical Whisper Safetensors namespace."""

    architecture_id = "whisper"
    adapter_id = "voicehub-whisper-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        return (
            config.get("voicehub_checkpoint_format") == "native-whisper-v1" and any(
                path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json")
                for path in files))

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in native_whisper_tensor_names(config)))


HFWhisperCheckpointAdapter = HuggingFaceWhisperCheckpointAdapter

__all__ = [
    "HFWhisperCheckpointAdapter",
    "HuggingFaceWhisperCheckpointAdapter",
    "NativeWhisperCheckpointAdapter",
    "OpenAIWhisperCheckpointAdapter",
    "TensorMapping",
    "huggingface_whisper_tensor_mapping",
    "native_whisper_tensor_names",
    "openai_whisper_tensor_mapping",
]
