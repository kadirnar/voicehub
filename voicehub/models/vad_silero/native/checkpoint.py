"""Strict adapters for official Silero VAD v6.2.1 checkpoints.

The standalone 16 kHz Safetensors header and the two TorchScript state-dict
namespaces were inspected at immutable upstream revision
``7e30209a3e901f9842f81b225f3e93d8199902b1``.  Loading uses VoiceHub's
checkpoint subsystem and PyTorch tensors only; the scripted upstream graph is
never required for native execution.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing.adapters import CheckpointAdapter
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan
from voicehub.models.vad_silero.native.configuration import SileroVADConfig

OFFICIAL_SILERO_VAD_VERSION = "6.2.1"
OFFICIAL_SILERO_VAD_REVISION = ("7e30209a3e901f9842f81b225f3e93d8199902b1")
OFFICIAL_SILERO_VAD_16K_HEADER_FINGERPRINT = (
    "1abaf3b9cfbf3990230392263d17d18ccd63e63471a965289da55335f09a7af8")

TensorShapes = dict[str, tuple[int, ...]]
TensorMapping = tuple[tuple[str, str], ...]
ConfigLike = SileroVADConfig | Mapping[str, Any] | None


def native_silero_vad_tensor_shapes(config: ConfigLike = None, ) -> TensorShapes:
    """Return the complete native state namespace and exact shapes."""
    resolved = SileroVADConfig.coerce(config or {})
    recurrent_size = resolved.recurrent_size
    gate_size = recurrent_size * 4
    return {
        "stft_conv.weight": (resolved.spectrum_bins * 2, 1, resolved.filter_length),
        "conv1.weight": (128, resolved.spectrum_bins, 3),
        "conv1.bias": (128, ),
        "conv2.weight": (64, 128, 3),
        "conv2.bias": (64, ),
        "conv3.weight": (64, 64, 3),
        "conv3.bias": (64, ),
        "conv4.weight": (recurrent_size, 64, 3),
        "conv4.bias": (recurrent_size, ),
        "lstm_cell.weight_ih": (gate_size, recurrent_size),
        "lstm_cell.weight_hh": (gate_size, recurrent_size),
        "lstm_cell.bias_ih": (gate_size, ),
        "lstm_cell.bias_hh": (gate_size, ),
        "final_conv.weight": (1, recurrent_size, 1),
        "final_conv.bias": (1, ),
    }


def native_silero_vad_tensor_names(config: ConfigLike = None, ) -> tuple[str, ...]:
    """Return every native persistent tensor in canonical order."""
    return tuple(sorted(native_silero_vad_tensor_shapes(config)))


def official_safetensors_tensor_mapping(config: ConfigLike = None, ) -> TensorMapping:
    """Map the released standalone 16 kHz Safetensors namespace."""
    resolved = SileroVADConfig.coerce(config or {})
    if resolved.sampling_rate != 16_000:
        raise ValueError(
            "Silero v6.2.1 does not release an 8 kHz Safetensors checkpoint; "
            "import the 8 kHz TorchScript state dict instead.")
    return tuple((name, name) for name in native_silero_vad_tensor_names(resolved))


def official_torchscript_tensor_mapping(config: ConfigLike = None, ) -> TensorMapping:
    """Map one branch of the official merged TorchScript state dict."""
    resolved = SileroVADConfig.coerce(config or {})
    prefix = "_model." if resolved.sampling_rate == 16_000 else "_model_8k."
    source_suffixes = {
        "stft_conv.weight": "stft.forward_basis_buffer",
        "conv1.weight": "encoder.0.reparam_conv.weight",
        "conv1.bias": "encoder.0.reparam_conv.bias",
        "conv2.weight": "encoder.1.reparam_conv.weight",
        "conv2.bias": "encoder.1.reparam_conv.bias",
        "conv3.weight": "encoder.2.reparam_conv.weight",
        "conv3.bias": "encoder.2.reparam_conv.bias",
        "conv4.weight": "encoder.3.reparam_conv.weight",
        "conv4.bias": "encoder.3.reparam_conv.bias",
        "lstm_cell.weight_ih": "decoder.rnn.weight_ih",
        "lstm_cell.weight_hh": "decoder.rnn.weight_hh",
        "lstm_cell.bias_ih": "decoder.rnn.bias_ih",
        "lstm_cell.bias_hh": "decoder.rnn.bias_hh",
        "final_conv.weight": "decoder.decoder.2.weight",
        "final_conv.bias": "decoder.decoder.2.bias",
    }
    return tuple(
        sorted((prefix + source_suffixes[target], target)
               for target in native_silero_vad_tensor_names(resolved)))


def tensor_inventory_fingerprint(
    tensor_shapes: Mapping[str, tuple[int, ...]],
    *,
    dtype: str = "F32",
) -> str:
    """Hash sorted tensor names, dtypes, and shapes."""
    if not isinstance(tensor_shapes, Mapping):
        raise TypeError("`tensor_shapes` must be a mapping.")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("`dtype` must be a non-empty string.")
    rows: list[str] = []
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


class OfficialSileroVADSafeTensorsCheckpointAdapter(CheckpointAdapter):
    """Load the official standalone 16 kHz Safetensors checkpoint."""

    architecture_id = "silero-vad"
    adapter_id = "official-silero-vad-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            resolved = SileroVADConfig.coerce(config)
        except (TypeError, ValueError):
            return False
        return (
            resolved.sampling_rate == 16_000 and
            any(path.name == "silero_vad_16k.safetensors" for path in files))

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        return TensorPlan(
            rules=tuple(
                CopyTensor(source, target)
                for source, target in official_safetensors_tensor_mapping(config)), )


class OfficialSileroVADTorchScriptCheckpointAdapter(CheckpointAdapter):
    """Import weights from an official ``silero_vad.jit`` state dict."""

    architecture_id = "silero-vad"
    adapter_id = "official-silero-vad-torchscript-state-dict"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            SileroVADConfig.coerce(config)
        except (TypeError, ValueError):
            return False
        return any(path.suffix in (".jit", ".pt") for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        resolved = SileroVADConfig.coerce(config)
        ignored_branch = ("_model_8k.*" if resolved.sampling_rate == 16_000 else "_model.*")
        return TensorPlan(
            rules=tuple(
                CopyTensor(source, target)
                for source, target in official_torchscript_tensor_mapping(resolved)),
            ignored_source_patterns=(ignored_branch, ),
        )


OfficialSileroVADCheckpointAdapter = (OfficialSileroVADSafeTensorsCheckpointAdapter)

__all__ = [
    "OFFICIAL_SILERO_VAD_16K_HEADER_FINGERPRINT",
    "OFFICIAL_SILERO_VAD_REVISION",
    "OFFICIAL_SILERO_VAD_VERSION",
    "OfficialSileroVADCheckpointAdapter",
    "OfficialSileroVADSafeTensorsCheckpointAdapter",
    "OfficialSileroVADTorchScriptCheckpointAdapter",
    "TensorMapping",
    "TensorShapes",
    "native_silero_vad_tensor_names",
    "native_silero_vad_tensor_shapes",
    "official_safetensors_tensor_mapping",
    "official_torchscript_tensor_mapping",
    "tensor_inventory_fingerprint",
]
