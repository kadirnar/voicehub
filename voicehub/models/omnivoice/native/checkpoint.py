"""Strict, streaming Safetensors I/O for OmniVoice and Higgs Audio V2."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.omnivoice.native.codec import HiggsAudioV2Tokenizer
from voicehub.models.omnivoice.native.metadata import (
    HIGGS_AUDIO_V2_HEADER_FINGERPRINT,
    HIGGS_AUDIO_V2_PARAMETER_COUNT,
    HIGGS_AUDIO_V2_TENSOR_COUNT,
    NATIVE_HIGGS_AUDIO_V2_FORMAT,
    NATIVE_OMNIVOICE_FORMAT,
    OMNIVOICE_MODEL_HEADER_FINGERPRINT,
    OMNIVOICE_MODEL_PARAMETER_COUNT,
    OMNIVOICE_MODEL_TENSOR_COUNT,
)
from voicehub.models.omnivoice.native.modeling import OmniVoiceModel

_FLOATING_DTYPES = frozenset({"BF16", "F16", "F32", "F64", "F8_E4M3", "F8_E5M2"})
_INTEGER_DTYPES = frozenset({"BOOL", "I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64"})


def tensor_inventory_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    """Hash stable name/dtype/shape rows using the audited convention."""
    rows = [
        f"{name}\t{dtype}\t{','.join(str(value) for value in shape)}"
        for name, (dtype, shape) in sorted(inventory.items())
    ]
    payload = ("\n".join(rows) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class OmniVoiceCheckpointReport:
    path: Path
    tensor_count: int
    parameter_count: int
    header_fingerprint: str


def inspect_omnivoice_checkpoint(path: str | Path, ) -> OmniVoiceCheckpointReport:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".safetensors":
        raise ValueError("Native OmniVoice checkpoints must use Safetensors.")
    with SafeTensorReader(source) as reader:
        names = tuple(reader.keys())
        inventory = {name: (reader.record(name).dtype, reader.tensor_shape(name)) for name in names}
        count = sum(reader.record(name).number_of_elements for name in names)
    return OmniVoiceCheckpointReport(
        path=source,
        tensor_count=len(inventory),
        parameter_count=count,
        header_fingerprint=tensor_inventory_fingerprint(inventory),
    )


def _validate_layout(model: nn.Module, reader: SafeTensorReader) -> tuple[str, ...]:
    state = model.state_dict(keep_vars=True)
    expected_names = set(state)
    actual_names = set(reader.keys())
    common = expected_names & actual_names
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    shape_mismatches = sorted((name, reader.tensor_shape(name), tuple(state[name].shape)) for name in common
                              if reader.tensor_shape(name) != tuple(state[name].shape))
    dtype_mismatches = sorted(
        (name, reader.record(name).dtype) for name in common
        if (state[name].is_floating_point() and reader.record(name).dtype not in _FLOATING_DTYPES) or
        (not state[name].is_floating_point() and reader.record(name).dtype not in _INTEGER_DTYPES))
    if missing or unexpected or shape_mismatches or dtype_mismatches:
        raise CheckpointCompatibilityError(
            "Checkpoint does not match the native OmniVoice graph: "
            f"missing={missing[:12]!r}, unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={shape_mismatches[:12]!r}, "
            f"dtype_mismatches={dtype_mismatches[:12]!r}.")
    return tuple(sorted(expected_names))


def _official_inventory(model: OmniVoiceModel | HiggsAudioV2Tokenizer, ) -> tuple[int, int, str]:
    if isinstance(model, OmniVoiceModel):
        return (
            OMNIVOICE_MODEL_TENSOR_COUNT,
            OMNIVOICE_MODEL_PARAMETER_COUNT,
            OMNIVOICE_MODEL_HEADER_FINGERPRINT,
        )
    return (
        HIGGS_AUDIO_V2_TENSOR_COUNT,
        HIGGS_AUDIO_V2_PARAMETER_COUNT,
        HIGGS_AUDIO_V2_HEADER_FINGERPRINT,
    )


def validate_omnivoice_checkpoint(
    model: OmniVoiceModel | HiggsAudioV2Tokenizer,
    path: str | Path,
    *,
    require_official_inventory: bool = False,
) -> OmniVoiceCheckpointReport:
    if not isinstance(model, (OmniVoiceModel, HiggsAudioV2Tokenizer)):
        raise TypeError("Checkpoint target must be OmniVoiceModel or HiggsAudioV2Tokenizer.")
    report = inspect_omnivoice_checkpoint(path)
    with SafeTensorReader(report.path) as reader:
        _validate_layout(model, reader)
    if require_official_inventory:
        actual = (
            report.tensor_count,
            report.parameter_count,
            report.header_fingerprint,
        )
        expected = _official_inventory(model)
        if actual != expected:
            raise CheckpointCompatibilityError(
                "Checkpoint matches the graph but not the audited official "
                f"inventory: actual={actual!r}, expected={expected!r}.")
    return report


def load_omnivoice_checkpoint(
    model: OmniVoiceModel | HiggsAudioV2Tokenizer,
    path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    require_official_inventory: bool = False,
) -> OmniVoiceCheckpointReport:
    report = validate_omnivoice_checkpoint(
        model,
        path,
        require_official_inventory=require_official_inventory,
    )
    target_device = torch.device(device)
    with SafeTensorReader(report.path) as reader, torch.no_grad():
        names = _validate_layout(model, reader)
        for name in names:
            value = reader.get_tensor(name)
            target = model.state_dict(keep_vars=True)[name]
            if dtype is not None and target.is_floating_point():
                value = value.to(dtype=dtype)
            model.load_state_dict(
                {name: value.to(device=target_device)},
                strict=False,
                assign=True,
            )
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError("Streaming load left meta tensors: " + ", ".join(remaining[:12]))
    return report


def export_omnivoice_checkpoint(
    model: OmniVoiceModel | HiggsAudioV2Tokenizer,
    path: str | Path,
) -> Path:
    if not isinstance(model, (OmniVoiceModel, HiggsAudioV2Tokenizer)):
        raise TypeError("Checkpoint export requires OmniVoiceModel or "
                        "HiggsAudioV2Tokenizer.")
    state: dict[str, Tensor] = {name: value.detach() for name, value in model.state_dict().items()}
    if any(value.device.type == "meta" for value in state.values()):
        raise ValueError("Cannot export an unmaterialized OmniVoice graph.")
    is_model = isinstance(model, OmniVoiceModel)
    return save_safetensors(
        state,
        path,
        metadata={
            "architecture": ("omnivoice" if is_model else "higgs-audio-v2-tokenizer"),
            "format": (NATIVE_OMNIVOICE_FORMAT if is_model else NATIVE_HIGGS_AUDIO_V2_FORMAT),
            "training_objective": ("weighted-codebook-masked-cross-entropy" if is_model else "frozen-codec"),
        },
    ).resolve()


__all__ = [
    "OmniVoiceCheckpointReport",
    "export_omnivoice_checkpoint",
    "inspect_omnivoice_checkpoint",
    "load_omnivoice_checkpoint",
    "tensor_inventory_fingerprint",
    "validate_omnivoice_checkpoint",
]
