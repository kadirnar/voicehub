"""Strict MeloTTS legacy conversion and Safetensors steady state."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.hub import write_json_file
from voicehub.models.melotts.native.configuration import MeloTTSArchitectureConfig
from voicehub.models.melotts.native.metadata import MELOTTS_NATIVE_FORMAT
from voicehub.models.melotts.native.modeling import DEPLOYABLE_MELOTTS_COMPONENTS


@dataclass(frozen=True, slots=True)
class MeloTTSCheckpointReport:
    path: Path
    tensor_count: int
    parameter_count: int
    inventory_fingerprint: str
    legacy: bool


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    rows = (
        f"{name}|{dtype}|{','.join(str(item) for item in shape)}"
        for name, (dtype, shape) in sorted(inventory.items()))
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _expected_state(model: nn.Module) -> Mapping[str, Tensor]:
    state = model.state_dict(keep_vars=True)
    if not state:
        raise ValueError("MeloTTS returned an empty generator state dict.")
    prefixes = {name.split(".", 1)[0] for name in state}
    expected = set(DEPLOYABLE_MELOTTS_COMPONENTS)
    if prefixes != expected:
        raise ValueError(
            "MeloTTS generator component inventory is incomplete: "
            f"missing={sorted(expected - prefixes)!r}, "
            f"unexpected={sorted(prefixes - expected)!r}.")
    return state


def inspect_melotts_checkpoint(path: str | Path, ) -> MeloTTSCheckpointReport:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".safetensors":
        raise ValueError("MeloTTS checkpoint inspection accepts Safetensors only.")
    with SafeTensorReader(source) as reader:
        inventory = {name: (reader.record(name).dtype, reader.tensor_shape(name)) for name in reader.keys()}
        parameter_count = sum(reader.record(name).number_of_elements for name in reader.keys())
    return MeloTTSCheckpointReport(
        path=source,
        tensor_count=len(inventory),
        parameter_count=parameter_count,
        inventory_fingerprint=_fingerprint(inventory),
        legacy=False,
    )


def _validated_safe_names(
    model: nn.Module,
    reader: SafeTensorReader,
) -> tuple[str, ...]:
    expected = _expected_state(model)
    expected_names = set(expected)
    actual_names = set(reader.keys())
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    mismatches = sorted((
        name,
        reader.tensor_shape(name),
        tuple(expected[name].shape),
    ) for name in expected_names & actual_names if reader.tensor_shape(name) != tuple(expected[name].shape))
    if missing or unexpected or mismatches:
        raise CheckpointCompatibilityError(
            "MeloTTS Safetensors namespace mismatch: "
            f"missing={missing[:12]!r}, "
            f"unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={mismatches[:12]!r}.")
    return tuple(sorted(expected_names))


def load_melotts_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> MeloTTSCheckpointReport:
    """Validate the complete checkpoint before assigning one tensor."""
    report = inspect_melotts_checkpoint(path)
    targets = _expected_state(model)
    with SafeTensorReader(report.path) as reader:
        names = _validated_safe_names(model, reader)
        state: dict[str, Tensor] = {}
        for name in names:
            target = targets[name]
            value = reader.get_tensor(name, device=device)
            value = value.to(
                dtype=(dtype if target.is_floating_point() and dtype is not None else target.dtype))
            state[name] = value
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CheckpointCompatibilityError("MeloTTS Safetensors assignment failed strict validation.")
    return report


def _legacy_state(payload: Any) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise CheckpointCompatibilityError("Legacy MeloTTS checkpoint must contain a mapping.")
    state = payload.get("model", payload)
    if not isinstance(state, Mapping) or not state:
        raise CheckpointCompatibilityError("Legacy MeloTTS checkpoint is missing its `model` state.")
    if any(not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()):
        raise CheckpointCompatibilityError("Legacy MeloTTS model state must map names to tensors.")
    return state


def read_legacy_melotts_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    trust_pickle_checkpoint: bool,
    expected_sha256: str | None = None,
) -> OrderedDict[str, Tensor]:
    """Read one reviewed pickle container with PyTorch's restricted loader."""
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "Official MeloTTS checkpoints use a PyTorch pickle container. "
            "Review the source and pass `trust_pickle_checkpoint=True` once, "
            "then export Safetensors for steady-state use.")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MeloTTS checkpoint was not found: {source}.")
    if source.suffix.lower() not in {".pth", ".pt"}:
        raise ValueError("Legacy MeloTTS import accepts .pth or .pt files only.")
    if expected_sha256 is not None:
        actual_sha256 = file_sha256(source)
        if actual_sha256 != expected_sha256:
            raise CheckpointCompatibilityError(
                "Pinned MeloTTS checkpoint SHA-256 mismatch: "
                f"{actual_sha256}.")
    try:
        payload = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:  # pragma: no cover - unsupported old PyTorch
        raise RuntimeError("MeloTTS conversion requires safe weights-only loading.") from error
    raw = _legacy_state(payload)
    normalized: dict[str, Tensor] = {}
    for raw_name, value in raw.items():
        name = (raw_name[7:] if raw_name.startswith("module.") else raw_name)
        if name in normalized:
            raise CheckpointCompatibilityError(f"Legacy MeloTTS tensor mapping collides at {name!r}.")
        normalized[name] = value

    expected = _expected_state(model)
    expected_names = set(expected)
    actual_names = set(normalized)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    mismatches = []
    for name in expected_names & actual_names:
        actual_shape = tuple(normalized[name].shape)
        expected_shape = tuple(expected[name].shape)
        if actual_shape != expected_shape:
            mismatches.append((name, actual_shape, expected_shape))
    mismatches.sort()
    if missing or unexpected or mismatches:
        raise CheckpointCompatibilityError(
            "Legacy MeloTTS checkpoint is incompatible: "
            f"missing={missing[:12]!r}, "
            f"unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={mismatches[:12]!r}.")
    return OrderedDict((
        name,
        normalized[name].detach().cpu().contiguous(),
    ) for name in sorted(normalized))


def convert_legacy_melotts_checkpoint(
    model: nn.Module,
    source: str | Path,
    destination: str | Path,
    *,
    trust_pickle_checkpoint: bool,
    expected_sha256: str | None = None,
) -> Path:
    state = read_legacy_melotts_checkpoint(
        model,
        source,
        trust_pickle_checkpoint=trust_pickle_checkpoint,
        expected_sha256=expected_sha256,
    )
    return save_safetensors(
        state,
        destination,
        metadata={
            "format": MELOTTS_NATIVE_FORMAT,
            "architecture": "melotts",
            "converted_from": Path(source).name,
        },
    ).resolve()


def export_melotts_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    expected = _expected_state(model)
    state = expected if state_override is None else dict(state_override)
    if set(state) != set(expected):
        raise ValueError(
            "MeloTTS export state is incomplete: "
            f"missing={sorted(set(expected) - set(state))!r}, "
            f"unexpected={sorted(set(state) - set(expected))!r}.")
    for name, value in state.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"MeloTTS state value {name!r} is not a tensor.")
        if tuple(value.shape) != tuple(expected[name].shape):
            raise ValueError(f"MeloTTS state value {name!r} has the wrong shape.")
    return save_safetensors(
        {
            name: value.detach().cpu().contiguous()
            for name, value in state.items()
        },
        path,
        metadata={
            "format": MELOTTS_NATIVE_FORMAT,
            "architecture": "melotts",
            "training_boundary": "precomputed-linguistic-features",
        },
    ).resolve()


def save_melotts_pretrained(
    model: nn.Module,
    config: MeloTTSArchitectureConfig,
    directory: str | Path,
) -> Path:
    if not isinstance(config, MeloTTSArchitectureConfig):
        raise TypeError("`config` must be a MeloTTSArchitectureConfig.")
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    export_melotts_checkpoint(
        model,
        destination / "model.safetensors",
    )
    write_json_file(destination / "config.json", config.to_dict())
    return destination.resolve()


__all__ = [
    "MeloTTSCheckpointReport",
    "convert_legacy_melotts_checkpoint",
    "export_melotts_checkpoint",
    "file_sha256",
    "inspect_melotts_checkpoint",
    "load_melotts_checkpoint",
    "read_legacy_melotts_checkpoint",
    "save_melotts_pretrained",
]
