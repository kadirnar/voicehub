"""Strict published-checkpoint conversion for VoiceHub's native DAC graph."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.checkpointing.adapters import CheckpointAdapter, CheckpointCompatibilityReport
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan, TensorResolver, TensorRule
from voicehub.models.dac.native.configuration import DacConfig
from voicehub.models.dac.native.metadata import (
    DESCRIPT_DAC_44KHZ_HEADER_FINGERPRINT,
    DESCRIPT_DAC_44KHZ_REVISION,
    TRANSFORMERS_DAC_REVISION,
)

_ENCODER_BLOCK = re.compile(r"^encoder\.block\.(?P<block>\d+)\.(?P<component>.+)$")
_DECODER_BLOCK = re.compile(r"^decoder\.block\.(?P<block>\d+)\.(?P<component>.+)$")
_RESIDUAL_COMPONENT = re.compile(
    r"^res_unit(?P<unit>[123])\."
    r"(?P<layer>snake1|snake2|conv1|conv2)"
    r"(?:\.(?P<parameter>alpha|weight|bias))?$")


def _residual_name(prefix: str, component: str) -> str:
    match = _RESIDUAL_COMPONENT.fullmatch(component)
    if match is None:
        raise KeyError(f"Unknown DAC residual tensor component {component!r}.")
    unit_index = int(match.group("unit")) - 1
    layer = match.group("layer")
    parameter = match.group("parameter")
    layer_index = {
        "snake1": 0,
        "conv1": 1,
        "snake2": 2,
        "conv2": 3,
    }[layer]
    if layer.startswith("snake"):
        if parameter != "alpha":
            raise KeyError(f"Invalid DAC Snake parameter {parameter!r}.")
        return f"{prefix}.{unit_index}.block.{layer_index}.alpha"
    if parameter not in {"weight", "bias"}:
        raise KeyError(f"Invalid DAC convolution parameter {parameter!r}.")
    return f"{prefix}.{unit_index}.block.{layer_index}.{parameter}"


def native_dac_tensor_name(
    source_name: str,
    config: DacConfig | Mapping[str, Any],
) -> str:
    """Translate one published Transformers DAC tensor name."""
    resolved = DacConfig.coerce(config)
    final_block = len(resolved.downsampling_ratios) + 1
    direct_prefixes = {
        "encoder.conv1.": "encoder.block.0.",
        "encoder.snake1.": f"encoder.block.{final_block}.",
        "encoder.conv2.": f"encoder.block.{final_block + 1}.",
        "decoder.conv1.": "decoder.model.0.",
        "decoder.snake1.": f"decoder.model.{final_block}.",
        "decoder.conv2.": f"decoder.model.{final_block + 1}.",
    }
    for source_prefix, target_prefix in direct_prefixes.items():
        if source_name.startswith(source_prefix):
            return target_prefix + source_name.removeprefix(source_prefix)

    match = _ENCODER_BLOCK.fullmatch(source_name)
    if match is not None:
        block = int(match.group("block")) + 1
        if block > len(resolved.downsampling_ratios):
            raise KeyError(f"Unknown DAC encoder block in {source_name!r}.")
        component = match.group("component")
        prefix = f"encoder.block.{block}.block"
        if component == "snake1.alpha":
            return f"{prefix}.3.alpha"
        if component in {"conv1.weight", "conv1.bias"}:
            return f"{prefix}.4.{component.rsplit('.', 1)[1]}"
        return _residual_name(prefix, component)

    match = _DECODER_BLOCK.fullmatch(source_name)
    if match is not None:
        block = int(match.group("block")) + 1
        if block > len(resolved.downsampling_ratios):
            raise KeyError(f"Unknown DAC decoder block in {source_name!r}.")
        component = match.group("component")
        prefix = f"decoder.model.{block}.block"
        if component == "snake1.alpha":
            return f"{prefix}.0.alpha"
        if component in {"conv_t1.weight", "conv_t1.bias"}:
            return f"{prefix}.1.{component.rsplit('.', 1)[1]}"
        residual = _RESIDUAL_COMPONENT.fullmatch(component)
        if residual is None:
            raise KeyError(f"Unknown DAC decoder tensor {source_name!r}.")
        unit_index = int(residual.group("unit")) + 1
        layer = residual.group("layer")
        parameter = residual.group("parameter")
        layer_index = {
            "snake1": 0,
            "conv1": 1,
            "snake2": 2,
            "conv2": 3,
        }[layer]
        if layer.startswith("snake"):
            if parameter != "alpha":
                raise KeyError(f"Invalid DAC Snake parameter {parameter!r}.")
            return f"{prefix}.{unit_index}.block.{layer_index}.alpha"
        if parameter not in {"weight", "bias"}:
            raise KeyError(f"Invalid DAC convolution parameter {parameter!r}.")
        return f"{prefix}.{unit_index}.block.{layer_index}.{parameter}"

    if source_name.startswith("quantizer.quantizers."):
        return source_name
    raise KeyError(f"Unknown published DAC tensor name {source_name!r}.")


def huggingface_dac_tensor_names(config: DacConfig | Mapping[str, Any], ) -> tuple[str, ...]:
    """Return the complete official DAC Safetensors namespace."""
    resolved = DacConfig.coerce(config)
    names: list[str] = []
    for prefix in ("encoder.conv1", "encoder.conv2", "decoder.conv1", "decoder.conv2"):
        names.extend((f"{prefix}.weight", f"{prefix}.bias"))
    names.extend(("encoder.snake1.alpha", "decoder.snake1.alpha"))
    for block in range(len(resolved.downsampling_ratios)):
        encoder_prefix = f"encoder.block.{block}"
        decoder_prefix = f"decoder.block.{block}"
        names.extend((
            f"{encoder_prefix}.snake1.alpha",
            f"{encoder_prefix}.conv1.weight",
            f"{encoder_prefix}.conv1.bias",
            f"{decoder_prefix}.snake1.alpha",
            f"{decoder_prefix}.conv_t1.weight",
            f"{decoder_prefix}.conv_t1.bias",
        ))
        for unit in range(1, 4):
            for prefix in (encoder_prefix, decoder_prefix):
                names.extend((
                    f"{prefix}.res_unit{unit}.snake1.alpha",
                    f"{prefix}.res_unit{unit}.conv1.weight",
                    f"{prefix}.res_unit{unit}.conv1.bias",
                    f"{prefix}.res_unit{unit}.snake2.alpha",
                    f"{prefix}.res_unit{unit}.conv2.weight",
                    f"{prefix}.res_unit{unit}.conv2.bias",
                ))
    for index in range(resolved.n_codebooks):
        prefix = f"quantizer.quantizers.{index}"
        names.extend((
            f"{prefix}.in_proj.weight",
            f"{prefix}.in_proj.bias",
            f"{prefix}.out_proj.weight",
            f"{prefix}.out_proj.bias",
            f"{prefix}.codebook.weight",
        ))
    return tuple(sorted(names))


def huggingface_dac_tensor_shapes(config: DacConfig | Mapping[str, Any], ) -> dict[str, tuple[int, ...]]:
    """Return source shapes implied by one validated DAC configuration."""
    from voicehub.models.dac.native.modeling import DacModel

    resolved = DacConfig.coerce(config)
    with torch.device("meta"):
        state = DacModel(resolved).state_dict()
    shapes = {}
    for source_name in huggingface_dac_tensor_names(resolved):
        target_name = native_dac_tensor_name(source_name, resolved)
        if (source_name.endswith(".weight") and ".codebook." not in source_name):
            target_name = (target_name.removesuffix(".weight") + ".weight_v")
        shapes[source_name] = tuple(state[target_name].shape)
    return shapes


def dac_tensor_inventory_fingerprint(
    tensor_shapes: Mapping[str, tuple[int, ...]],
    *,
    dtype: str = "F32",
) -> str:
    """Hash sorted source names, dtypes, and shapes from a safe header."""
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
        rows.append(f"{name}|{dtype}|{'x'.join(str(value) for value in shape)}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WeightNormalizedTensor(TensorRule):
    """Expand one effective convolution weight into weight-norm parameters."""

    source: str
    weight_target: str
    scale_target: str

    @property
    def source_names(self) -> tuple[str, ...]:
        return (self.source, )

    @property
    def target_names(self) -> tuple[str, ...]:
        return self.weight_target, self.scale_target

    def apply(self, resolver: TensorResolver) -> Mapping[str, Tensor]:
        tensor = resolver.get(self.source)
        if not isinstance(tensor, Tensor) or tensor.ndim < 2:
            raise TypeError(f"DAC convolution tensor {self.source!r} must have rank >= 2.")
        dimensions = tuple(range(1, tensor.ndim))
        scale = torch.linalg.vector_norm(
            tensor.float(),
            dim=dimensions,
            keepdim=True,
        ).to(dtype=tensor.dtype)
        return {
            self.weight_target: tensor,
            self.scale_target: scale,
        }


class HuggingFaceDacCheckpointAdapter(CheckpointAdapter):
    """Convert official Transformers DAC Safetensors into the native graph."""

    architecture_id = "dac"
    adapter_id = "huggingface-dac-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        return (
            str(config.get("model_type", "")).lower() == "dac" and
            any(path.suffix == ".safetensors" for path in files))

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        resolved = DacConfig.from_dict(config)
        rules: list[TensorRule] = []
        for source_name in huggingface_dac_tensor_names(resolved):
            target_name = native_dac_tensor_name(source_name, resolved)
            if (source_name.endswith(".weight") and ".codebook." not in source_name):
                prefix = target_name.removesuffix(".weight")
                rules.append(
                    WeightNormalizedTensor(
                        source=source_name,
                        weight_target=f"{prefix}.weight_v",
                        scale_target=f"{prefix}.weight_g",
                    ))
            else:
                rules.append(CopyTensor(source_name, target_name))
        return TensorPlan(rules=tuple(rules))

    def load_assign(
        self,
        model: Any,
        source,
        config: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> CheckpointCompatibilityReport:
        """Validate and assign tensors, including into a meta-device graph."""
        report, converted = self.inspect(model, source, config)
        if strict:
            report.require_compatible()
        mismatch_names = {item.name for item in report.shape_mismatches}
        loadable = {
            name: tensor
            for name, tensor in converted.items() if name in report.loaded and name not in mismatch_names
        }
        model.load_state_dict(loadable, strict=False, assign=True)
        return report


HFDacCheckpointAdapter = HuggingFaceDacCheckpointAdapter

__all__ = [
    "DESCRIPT_DAC_44KHZ_HEADER_FINGERPRINT",
    "DESCRIPT_DAC_44KHZ_REVISION",
    "HFDacCheckpointAdapter",
    "HuggingFaceDacCheckpointAdapter",
    "TRANSFORMERS_DAC_REVISION",
    "WeightNormalizedTensor",
    "dac_tensor_inventory_fingerprint",
    "huggingface_dac_tensor_names",
    "huggingface_dac_tensor_shapes",
    "native_dac_tensor_name",
]
