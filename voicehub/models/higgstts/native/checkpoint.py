"""Strict, streaming Safetensors I/O for native Higgs Audio v2."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.higgstts.native.metadata import (
    HIGGS_AUDIO_V2_CHECKPOINT_HEADER_FINGERPRINT,
    HIGGS_AUDIO_V2_CHECKPOINT_PARAMETER_COUNT,
    HIGGS_AUDIO_V2_CHECKPOINT_TENSOR_COUNT,
    HIGGS_AUDIO_V2_CODEC_HEADER_FINGERPRINT,
    HIGGS_AUDIO_V2_CODEC_PARAMETER_COUNT,
    HIGGS_AUDIO_V2_CODEC_TENSOR_COUNT,
    NATIVE_HIGGS_AUDIO_V2_CODEC_FORMAT,
    NATIVE_HIGGS_AUDIO_V2_FORMAT,
)
from voicehub.models.higgstts.native.modeling import HiggsAudioV2ForConditionalGeneration
from voicehub.models.higgstts.native.tokenizer import HiggsAudioV2TokenizerModel

_FLOATING_DTYPES = frozenset({
    "BF16",
    "F16",
    "F32",
    "F64",
    "F8_E4M3",
    "F8_E5M2",
})
_INTEGER_DTYPES = frozenset({
    "BOOL",
    "I8",
    "I16",
    "I32",
    "I64",
    "U8",
    "U16",
    "U32",
    "U64",
})
_CODEC_SOURCE_TO_NATIVE = {
    ("semantic_model.encoder.pos_conv_embed.conv."
     "parametrizations.weight.original0"):
    "semantic_model.encoder.pos_conv_embed.conv.weight_g",
    ("semantic_model.encoder.pos_conv_embed.conv."
     "parametrizations.weight.original1"):
    "semantic_model.encoder.pos_conv_embed.conv.weight_v",
}
_CODEC_NATIVE_TO_SOURCE = {target: source for source, target in _CODEC_SOURCE_TO_NATIVE.items()}


def tensor_inventory_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    rows = [
        f"{name}\t{dtype}\t{','.join(str(value) for value in shape)}"
        for name, (dtype, shape) in sorted(inventory.items())
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HiggsCheckpointReport:
    path: Path
    tensor_count: int
    parameter_count: int
    header_fingerprint: str


def inspect_higgs_checkpoint(path: str | Path, ) -> HiggsCheckpointReport:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".safetensors":
        raise ValueError("Native Higgs checkpoints must use Safetensors.")
    with SafeTensorReader(source) as reader:
        inventory = {
            name: (
                reader.record(name).dtype,
                reader.tensor_shape(name),
            )
            for name in reader.keys()
        }
        parameter_count = sum(reader.record(name).number_of_elements for name in reader.keys())
    return HiggsCheckpointReport(
        path=source,
        tensor_count=len(inventory),
        parameter_count=parameter_count,
        header_fingerprint=tensor_inventory_fingerprint(inventory),
    )


def _source_to_native_name(
    model: nn.Module,
    source_name: str,
) -> str:
    if isinstance(model, HiggsAudioV2TokenizerModel):
        return _CODEC_SOURCE_TO_NATIVE.get(source_name, source_name)
    return source_name


def _validate_layout(
    model: nn.Module,
    reader: SafeTensorReader,
) -> tuple[tuple[str, str], ...]:
    targets = model.state_dict(keep_vars=True)
    source_to_target = {
        source_name: _source_to_native_name(model, source_name)
        for source_name in reader.keys()
    }
    mapped_targets = set(source_to_target.values())
    expected_targets = set(targets)
    missing = sorted(expected_targets - mapped_targets)
    unexpected = sorted(mapped_targets - expected_targets)
    duplicate_targets = sorted(
        target for target, count in Counter(source_to_target.values()).items() if count > 1)
    shapes = []
    dtypes = []
    for source_name, target_name in source_to_target.items():
        if target_name not in targets:
            continue
        source_shape = reader.tensor_shape(source_name)
        target = targets[target_name]
        if source_shape != tuple(target.shape):
            shapes.append((source_name, source_shape, tuple(target.shape)))
        source_dtype = reader.record(source_name).dtype
        allowed = (_FLOATING_DTYPES if target.is_floating_point() else _INTEGER_DTYPES)
        if source_dtype not in allowed:
            dtypes.append((source_name, source_dtype))
    if missing or unexpected or duplicate_targets or shapes or dtypes:
        raise CheckpointCompatibilityError(
            "Higgs checkpoint does not match the native graph: "
            f"missing={missing[:12]!r}, "
            f"unexpected={unexpected[:12]!r}, "
            f"duplicate_targets={duplicate_targets[:12]!r}, "
            f"shape_mismatches={shapes[:12]!r}, "
            f"dtype_mismatches={dtypes[:12]!r}.")
    return tuple(sorted(source_to_target.items()))


def validate_higgs_checkpoint(
    model: (HiggsAudioV2ForConditionalGeneration
            | HiggsAudioV2TokenizerModel),
    path: str | Path,
    *,
    require_official_inventory: bool = False,
) -> HiggsCheckpointReport:
    if not isinstance(model, (
            HiggsAudioV2ForConditionalGeneration,
            HiggsAudioV2TokenizerModel,
    )):
        raise TypeError("Higgs checkpoint target must be a native model or tokenizer.")
    report = inspect_higgs_checkpoint(path)
    with SafeTensorReader(report.path) as reader:
        _validate_layout(model, reader)
    if require_official_inventory:
        if isinstance(model, HiggsAudioV2TokenizerModel):
            expected = (
                HIGGS_AUDIO_V2_CODEC_TENSOR_COUNT,
                HIGGS_AUDIO_V2_CODEC_PARAMETER_COUNT,
                HIGGS_AUDIO_V2_CODEC_HEADER_FINGERPRINT,
            )
        else:
            expected = (
                HIGGS_AUDIO_V2_CHECKPOINT_TENSOR_COUNT,
                HIGGS_AUDIO_V2_CHECKPOINT_PARAMETER_COUNT,
                HIGGS_AUDIO_V2_CHECKPOINT_HEADER_FINGERPRINT,
            )
        actual = (
            report.tensor_count,
            report.parameter_count,
            report.header_fingerprint,
        )
        if actual != expected:
            raise CheckpointCompatibilityError(
                "Checkpoint matches a Higgs-shaped graph but not the "
                "audited official tensor inventory: "
                f"actual={actual!r}, expected={expected!r}.")
    return report


def load_higgs_checkpoint(
    model: (HiggsAudioV2ForConditionalGeneration
            | HiggsAudioV2TokenizerModel),
    path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    require_official_inventory: bool = False,
) -> HiggsCheckpointReport:
    report = validate_higgs_checkpoint(
        model,
        path,
        require_official_inventory=require_official_inventory,
    )
    target_device = torch.device(device)
    with SafeTensorReader(report.path) as reader:
        assignments = _validate_layout(model, reader)
        target_state = model.state_dict(keep_vars=True)
        with torch.no_grad():
            for source_name, target_name in assignments:
                value = reader.get_tensor(source_name)
                target = target_state[target_name]
                if dtype is not None and target.is_floating_point():
                    value = value.to(dtype=dtype)
                value = value.to(device=target_device)
                model.load_state_dict(
                    {target_name: value},
                    strict=False,
                    assign=True,
                )
    if isinstance(model, HiggsAudioV2ForConditionalGeneration):
        model.model.materialize_runtime_buffers(target_device)
    else:
        model.freeze_semantic_model()
    meta_tensors = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if meta_tensors:
        raise CheckpointCompatibilityError(
            "Higgs streaming load left meta tensors: " + ", ".join(meta_tensors[:12]))
    return report


def export_higgs_checkpoint(
    model: (HiggsAudioV2ForConditionalGeneration
            | HiggsAudioV2TokenizerModel),
    path: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    if not isinstance(model, (
            HiggsAudioV2ForConditionalGeneration,
            HiggsAudioV2TokenizerModel,
    )):
        raise TypeError("Higgs export requires a native model or tokenizer.")
    model_state = model.state_dict()
    state = (model_state if state_override is None else dict(state_override))
    expected = set(model_state)
    actual = set(state)
    if actual != expected:
        raise ValueError(
            "Higgs export state is incomplete: "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}.")
    codec = isinstance(model, HiggsAudioV2TokenizerModel)
    exported = {(_CODEC_NATIVE_TO_SOURCE.get(name, name) if codec else name): value.detach()
                for name, value in state.items()}
    metadata = {
        "architecture": "higgs_audio_v2",
        "format": (NATIVE_HIGGS_AUDIO_V2_CODEC_FORMAT if codec else NATIVE_HIGGS_AUDIO_V2_FORMAT),
    }
    if not codec:
        metadata["training_objective"] = ("delayed-codebook-causal-ce-plus-text-causal-ce")
    return save_safetensors(
        exported,
        path,
        metadata=metadata,
    ).resolve()


__all__ = [
    "HiggsCheckpointReport",
    "export_higgs_checkpoint",
    "inspect_higgs_checkpoint",
    "load_higgs_checkpoint",
    "tensor_inventory_fingerprint",
    "validate_higgs_checkpoint",
]
