"""Strict Safetensors checkpoint boundary for native XTTS v2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError

NATIVE_XTTS2_FORMAT = "voicehub-native-xtts2-v1"
_IGNORED_TRAINER_PREFIXES = (
    "dvae.",
    "torch_mel_spectrogram_dvae.",
    "torch_mel_spectrogram_style_encoder.",
)


@dataclass(frozen=True, slots=True)
class XTTS2CheckpointInventory:
    path: Path
    tensor_count: int
    parameter_count: int
    header_fingerprint: str


def inspect_xtts2_checkpoint(path: str | Path) -> XTTS2CheckpointInventory:
    source = Path(path).expanduser().resolve()
    if source.suffix != ".safetensors":
        raise ValueError("Native XTTS v2 checkpoints must use `.safetensors`.")
    with SafeTensorReader(source) as reader:
        rows = []
        count = 0
        parameters = 0
        for name in sorted(reader.keys()):
            record = reader.record(name)
            shape = reader.tensor_shape(name)
            count += 1
            parameters += record.number_of_elements
            rows.append(f"{name}|{record.dtype}|{'x'.join(map(str, shape))}")
    return XTTS2CheckpointInventory(
        path=source,
        tensor_count=count,
        parameter_count=parameters,
        header_fingerprint=hashlib.sha256("\n".join(rows).encode("utf-8"), ).hexdigest(),
    )


def load_xtts2_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> XTTS2CheckpointInventory:
    inventory = inspect_xtts2_checkpoint(path)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    with SafeTensorReader(inventory.path) as reader:
        names = set(reader.keys())
        missing = sorted(set(expected) - names)
        unexpected = sorted(names - set(expected))
        mismatched = sorted((name, reader.tensor_shape(name), expected[name])
                            for name in names & set(expected) if reader.tensor_shape(name) != expected[name])
        if missing or unexpected or mismatched:
            raise CheckpointCompatibilityError(
                "XTTS v2 checkpoint namespace is incompatible: "
                f"missing={missing[:8]!r}, unexpected={unexpected[:8]!r}, "
                f"shape_mismatches={mismatched[:8]!r}.", )
        with torch.no_grad():
            for name in sorted(expected):
                value = reader.get_tensor(name)
                if not value.is_floating_point():
                    raise CheckpointCompatibilityError(f"XTTS v2 tensor {name!r} is not floating-point.")
                target_dtype = dtype if dtype is not None and value.is_floating_point() else value.dtype
                model.load_state_dict(
                    {name: value.to(device=device, dtype=target_dtype)},
                    strict=False,
                    assign=True,
                )
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "XTTS v2 checkpoint assignment left meta tensors: " + ", ".join(remaining[:8]))
    return inventory


def convert_trusted_legacy_xtts2_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    trust_legacy_pickle: bool = False,
) -> Path:
    """One-time conversion boundary for Coqui's published ``model.pth``.

    This function necessarily invokes PyTorch's restricted legacy
    unpickler. It is never called by steady-state loading and requires
    an explicit trust decision. The resulting Safetensors file is the
    only accepted runtime checkpoint.
    """
    if trust_legacy_pickle is not True:
        raise PermissionError(
            "XTTS v2's official model.pth is a legacy pickle container; "
            "set `trust_legacy_pickle=True` only for a reviewed one-time conversion.", )
    legacy_path = Path(source).expanduser().resolve()
    try:
        payload = torch.load(
            legacy_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError("This PyTorch version cannot restrict legacy XTTS deserialization.", ) from error
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("Legacy XTTS v2 checkpoint has no tensor `model` mapping.")
    normalized = {}
    for name, value in state.items():
        if name.startswith("xtts."):
            name = name.removeprefix("xtts.")
        if name.startswith(_IGNORED_TRAINER_PREFIXES):
            continue
        # PyTorch's weight-norm loader performs this migration implicitly for
        # pickle state dicts. Safetensors has no executable load hooks, so make
        # the one-to-one namespace upgrade explicit during conversion.
        if name.endswith(".weight_g"):
            name = name.removesuffix(".weight_g") + ".parametrizations.weight.original0"
        elif name.endswith(".weight_v"):
            name = name.removesuffix(".weight_v") + ".parametrizations.weight.original1"
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Legacy XTTS v2 state item {name!r} is not a tensor.")
        if name in normalized:
            raise ValueError(f"Legacy XTTS v2 conversion produced duplicate tensor {name!r}.")
        if (not value.is_floating_point() or value.is_quantized or value.is_complex()):
            raise TypeError(f"Legacy XTTS v2 tensor {name!r} is not a portable "
                            "floating-point weight.")
        if not torch.isfinite(value.detach()).all().item():
            raise ValueError(f"Legacy XTTS v2 tensor {name!r} contains non-finite values.")
        normalized[name] = value.detach().cpu().contiguous()
    return save_safetensors(
        normalized,
        destination,
        metadata={
            "format": NATIVE_XTTS2_FORMAT,
            "conversion_boundary": "trusted-legacy-coqui-model.pth",
        },
    ).resolve()


def save_xtts2_checkpoint(model: nn.Module, path: str | Path) -> Path:
    state = {}
    for name, value in model.state_dict().items():
        if value.device.type == "meta":
            raise CheckpointCompatibilityError(f"XTTS v2 tensor {name!r} is not materialized.")
        if not value.is_floating_point() or value.is_quantized or value.is_complex():
            raise CheckpointCompatibilityError(f"XTTS v2 tensor {name!r} is not a portable floating tensor.")
        if not torch.isfinite(value.detach()).all().item():
            raise CheckpointCompatibilityError(f"XTTS v2 tensor {name!r} contains non-finite values.")
        state[name] = value.detach().cpu().contiguous()
    return save_safetensors(
        state,
        path,
        metadata={
            "format": NATIVE_XTTS2_FORMAT
        },
    ).resolve()


__all__ = [
    "NATIVE_XTTS2_FORMAT",
    "XTTS2CheckpointInventory",
    "convert_trusted_legacy_xtts2_checkpoint",
    "inspect_xtts2_checkpoint",
    "load_xtts2_checkpoint",
    "save_xtts2_checkpoint",
]
